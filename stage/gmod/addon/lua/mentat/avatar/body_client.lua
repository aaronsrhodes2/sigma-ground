-- avatar/body_client.lua -- the client half of "the avatar wears the REAL Skippy": reassemble and verify what the server sends
-- ("mentat_avatar_body", see body_net.lua for the wire), decode the six SVM2 parts, build the meshes, and draw/animate the figure.
-- Nothing is trusted: sizes, order, CRCs, counts. ANY failure (a missing, corrupt or late part, a build error) leaves the whole
-- procedural can in place (never a half-built mix): avatar_client.lua draws the real body only when Body.ready is true.
if not CLIENT then return {} end

local SVM1 = include("mentat/holodeck/svm1.lua")
local B = include("mentat/avatar/body.lua")

local BC = {}
BC.CAP_BYTES = 256 * 1024
BC.MAX_RAW = 1024 * 1024
BC.PARTIAL_TIMEOUT = 30
BC.WANT_EVERY = 8
BC.ITEMS = #B.PARTS + 1            -- the manifest and the parts

BC.items = {}                      -- index -> { crc =, data = }
BC.partial = {}                    -- index -> { count =, total =, crc =, got =, n =, bytes =, t = }
BC.parts = {}                      -- name -> { plain = IMesh | nil, glow = IMesh | nil, tris = n }
BC.ready, BC.state, BC.why = false, "none", nil
BC.stats = { chunks = 0, rejected = 0 }
BC.asked = 0
BC.anim = { head = 0, hat_p = 0, hat_r = 0, dir = nil }

local function now() return SysTime() end

local function wipe_meshes()
  for _, p in pairs(BC.parts) do
    for _, k in ipairs({ "plain", "glow" }) do
      local m = p[k]
      if m and m.Destroy then pcall(m.Destroy, m) end
    end
  end
  BC.parts = {}
end

local function failed(why)
  wipe_meshes()
  BC.ready, BC.state, BC.why = false, "failed", tostring(why)
  return false, BC.why
end

local function build_mesh(rows)
  if #rows == 0 then return nil end
  local m = Mesh()
  mesh.Begin(m, MATERIAL_TRIANGLES, #rows / 3)
  for i = 1, #rows do
    local v = rows[i]
    mesh.Position(Vector(v[1], v[2], v[3]))
    mesh.Normal(Vector(v[4], v[5], v[6]))
    mesh.Color(v[7], v[8], v[9], 255)
    mesh.TexCoord(0, 0, 0)
    mesh.AdvanceVertex()
  end
  mesh.End()
  return m
end

--- Decode everything held and build the meshes. All or nothing. Returns true, or false and why.
function BC.build()
  if not (Mesh and mesh and mesh.Begin) then return failed("no mesh library") end
  local m = util.JSONToTable(BC.items[1].data)
  local piv, why = B.parse_manifest(m)
  if not piv then return failed("manifest: " .. tostring(why)) end
  local built, tris = {}, 0
  local ok, err = pcall(function()
    for i, name in ipairs(B.PARTS) do
      local blob = BC.items[i + 1].data
      local size = SVM1.lzma_size(blob)
      if not size or size < 40 or size > BC.MAX_RAW then error(name .. ": bad LZMA header") end
      local okd, raw = pcall(util.Decompress, blob)
      if not okd or type(raw) ~= "string" or #raw ~= size then error(name .. ": util.Decompress failed") end
      local dec, w2 = SVM1.decode(raw, true)
      if not dec or not dec.svm2 then error(name .. ": " .. tostring(w2 or "no normals")) end
      local plain, glow = {}, {}
      B.emit(dec, function(g, x, y, z, nx, ny, nz, r, gg, b)
        local rows = g and glow or plain
        rows[#rows + 1] = { x, y, z, nx, ny, nz, r, gg, b }
      end)
      built[name] = { plain = build_mesh(plain), glow = build_mesh(glow), tris = dec.nt }
      tris = tris + dec.nt
    end
  end)
  if not ok then
    for _, p in pairs(built) do for _, k in ipairs({ "plain", "glow" }) do if p[k] and p[k].Destroy then pcall(p[k].Destroy, p[k]) end end end
    return failed(err)
  end
  wipe_meshes()
  BC.parts = built
  local k = B.HU_PER_M
  local function hu(v) return { x = v.x * k, y = v.y * k, z = v.z * k } end
  BC.pv = { head = hu(piv.head), hat = hu(piv.hat), hat_top = hu(piv.hat_top), wrist = hu(piv.wrist), tip = hu(piv.tip) }
  BC.rest = B.unit({ x = piv.tip.x - piv.wrist.x, y = piv.tip.y - piv.wrist.y, z = piv.tip.z - piv.wrist.z })
  BC.height_m, BC.height_hu, BC.tris = piv.height, piv.height * k, tris
  BC.anim.dir = { x = BC.rest.x, y = BC.rest.y, z = BC.rest.z }
  BC.ready, BC.state, BC.why = true, "ready", nil
  return true
end

--- All seven items held and none still arriving: build (once per new set).
local function maybe_build()
  for i = 1, BC.ITEMS do if not BC.items[i] then return end end
  for index, p in pairs(BC.partial) do                            -- a half-received item that went quiet is dropped
    if now() - p.t > BC.PARTIAL_TIMEOUT then BC.partial[index] = nil end
  end
  if next(BC.partial) ~= nil then return end
  if BC.built_set == BC.set_id then return end
  BC.built_set = BC.set_id
  BC.build()
end

local function set_id()
  local t = {}
  for i = 1, BC.ITEMS do t[i] = BC.items[i] and BC.items[i].crc or 0 end
  return table.concat(t, ":")
end

--- A chunk arrived.
function BC.receive()
  local index = net.ReadUInt(8)
  local i = net.ReadUInt(8)
  local count = net.ReadUInt(8)
  local total = net.ReadUInt(20)
  local crc = net.ReadUInt(32)
  local n = net.ReadUInt(16)
  local function bad() BC.stats.rejected = BC.stats.rejected + 1 end
  if index < 1 or index > BC.ITEMS or count < 1 or count > math.ceil(BC.CAP_BYTES / (16 * 1024)) or i < 1 or i > count then return bad() end
  if total < 1 or total > BC.CAP_BYTES or n < 1 or n > 16 * 1024 then return bad() end
  local data = net.ReadData(n)
  if type(data) ~= "string" or #data ~= n then return bad() end
  if BC.items[index] and BC.items[index].crc == crc then return end        -- already have exactly this one
  local p = BC.partial[index]
  if p and (p.count ~= count or p.total ~= total or p.crc ~= crc or now() - p.t > BC.PARTIAL_TIMEOUT) then p = nil end
  if not p then p = { count = count, total = total, crc = crc, got = {}, n = 0, bytes = 0, t = now() }; BC.partial[index] = p end
  BC.stats.chunks = BC.stats.chunks + 1
  if p.got[i] then return end
  p.got[i] = data
  p.n = p.n + 1
  p.bytes = p.bytes + #data
  if p.bytes > BC.CAP_BYTES then BC.partial[index] = nil; return bad() end
  if p.n < p.count then BC.state = BC.ready and BC.state or "loading"; return end
  BC.partial[index] = nil
  local blob = table.concat(p.got)
  if #blob ~= total or tonumber(util.CRC(blob)) ~= crc then BC.items[index] = nil; return bad() end
  BC.items[index] = { crc = crc, data = blob }
  BC.set_id = set_id()
  maybe_build()
end
net.Receive("mentat_avatar_body", BC.receive)

--- Ask the server for everything, at most once every few seconds.
function BC.want()
  local t = now()
  if t - BC.asked < BC.WANT_EVERY then return false end
  BC.asked = t
  net.Start("mentat_avatar_body_want")
  net.WriteUInt(0, 8)
  net.SendToServer()
  return true
end

--- Called now and then: if the server has the body and this client does not, ask.
function BC.poll()
  if BC.ready then return end
  if GetGlobalBool("mentat_avatar_body", false) and BC.state ~= "failed" then BC.want() end
end
timer.Create("Mentat.AvatarBodyAsk", 3, 0, BC.poll)
hook.Add("InitPostEntity", "Mentat.AvatarBodyHello", function() BC.poll() end)

concommand.Add("mentat_avatar_body_info", function()
  local n = 0
  for _ in pairs(BC.items) do n = n + 1 end
  print(string.format("[avatar] body: %s%s; items %d/%d; %s triangles; chunks %d, rejected %d", BC.state, BC.why and (" (" .. BC.why .. ")") or "", n, BC.ITEMS,
    tostring(BC.tris or 0), BC.stats.chunks, BC.stats.rejected))
end, nil, "What this client holds of Skippy's real body")

-- ------------------------------------------------------------------------------------------------ drawing
local function mat4(rows)
  local m = Matrix()
  for r = 1, 4 do for c = 1, 4 do m:SetField(r, c, rows[(r - 1) * 4 + c]) end end
  return m
end

local function draw_part(name, glow)
  local p = BC.parts[name]
  if not p then return end
  if p.plain then p.plain:Draw() end
  if p.glow then
    render.SetColorModulation(glow, glow, glow)                  -- the tongue brightens with his voice
    p.glow:Draw()
    render.SetColorModulation(1, 1, 1)
  end
end

--- Where the hand should aim, as a direction in his own frame (unit), or the kit's rest direction.
local function hand_goal(bot, yaw, s, zoff, aim)
  if not aim then return BC.rest end
  local w = BC.pv.wrist
  local rad = math.rad(yaw)
  local c, sn = math.cos(rad), math.sin(rad)
  local lx, ly, lz = w.x * s, w.y * s, (w.z + zoff) * s                -- the wrist, in the model frame scaled
  local pos = bot:GetPos()
  local wx, wy, wz = pos.x + lx * c - ly * sn, pos.y + lx * sn + ly * c, pos.z + lz
  local dx, dy, dz = aim.x - wx, aim.y - wy, aim.z - wz
  local ux, uy = dx * c + dy * sn, -dx * sn + dy * c               -- into his frame (rotate by -yaw)
  local l = math.sqrt(ux * ux + uy * uy + dz * dz)
  if l < 1e-3 then return BC.rest end
  return { x = ux / l, y = uy / l, z = dz / l }
end

--- Draw the whole figure. ctx = { level (0..1 voice), lx, ly (the one he talks to, in his frame, or nil), aim (world Vector or nil),
--- dt, height_m, scale, hide = function(name) -> bool, hat_tilt = function(lx, ly, strength, max) }.
function BC.draw(bot, ctx)
  local dt = ctx.dt
  local yaw = bot:EyeAngles().y
  local s = (ctx.height_m / BC.height_m) * ctx.scale
  local zoff = -0.5 * BC.height_hu                                      -- his middle hovers where the can's middle did
  local A = BC.anim
  local level = ctx.level

  -- the head turns toward the one he talks to (small angles); the hat leans with his voice; the hand swings to what he is told to aim at
  local hy_goal = ctx.lx and B.head_yaw(ctx.lx, ctx.ly, 0.6) or 0
  A.head = A.head + (hy_goal - A.head) * (1 - math.exp(-6 * dt))
  local tp, tr = 0, 0
  if ctx.lx then tp, tr = ctx.hat_tilt(ctx.lx, ctx.ly, 0.35 + 0.65 * level, B.MAX_NOD) end
  A.hat_p = A.hat_p + (tp - A.hat_p) * (1 - math.exp(-6 * dt))
  A.hat_r = A.hat_r + (tr - A.hat_r) * (1 - math.exp(-6 * dt))
  local goal = hand_goal(bot, yaw, s, zoff, ctx.aim)
  local k = 1 - math.exp(-7 * dt)
  A.dir = B.unit({ x = A.dir.x + (goal.x - A.dir.x) * k, y = A.dir.y + (goal.y - A.dir.y) * k, z = A.dir.z + (goal.z - A.dir.z) * k })
  local rot, used = B.point_rotation(BC.rest, A.dir, B.MAX_SWING)
  A.swing = used

  local base = Matrix()
  base:Translate(bot:GetPos())
  base:Rotate(Angle(0, yaw, 0))
  base:Scale(Vector(s, s, s))
  base:Translate(Vector(0, 0, zoff))
  local glow = 0.75 + 0.25 * level
  local function shown(name) return not ctx.hide(name) end

  cam.PushModelMatrix(base)
  if shown("can") then draw_part("can", glow) end
  if shown("jacket") then draw_part("jacket", glow) end
  if shown("hand_b") then draw_part("hand_b", glow) end
  -- the pointing hand about the wrist
  cam.PushModelMatrix(mat4(B.about(rot, BC.pv.wrist)), true)
  if shown("hand_a") then draw_part("hand_a", glow) end
  cam.PopModelMatrix()
  -- the head about the neck; the hat sits on the head, so it turns with it and then tips about its own pivot
  local hp, tp3 = BC.pv.head, BC.pv.hat
  cam.PushModelMatrix(mat4(B.about(B.rodrigues({ x = 0, y = 0, z = 1 }, A.head), hp)), true)
  if shown("head") then draw_part("head", glow) end
  local hm = Matrix()
  hm:Translate(Vector(tp3.x, tp3.y, tp3.z))
  hm:Rotate(Angle(A.hat_p, 0, A.hat_r))
  hm:Translate(Vector(-tp3.x, -tp3.y, -tp3.z))
  cam.PushModelMatrix(hm, true)
  if shown("hat") then draw_part("hat", glow) end
  cam.PopModelMatrix()
  cam.PopModelMatrix()
  cam.PopModelMatrix()
end

BC.Body = B
return BC
