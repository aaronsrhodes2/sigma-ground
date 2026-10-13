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
BC.parts = {}                      -- name -> { plain = { IMesh, ... }, glow = { IMesh, ... }, tris = n } (a mesh holds at most 30000 vertices)
BC.MESH_VERTS = 30000
BC.ready, BC.state, BC.why = false, "none", nil
BC.stats = { chunks = 0, rejected = 0 }
BC.asked = 0
BC.anim = { head = 0, hat_p = 0, hat_r = 0, dir = nil }

local function now() return SysTime() end

local function wipe_meshes()
  for _, p in pairs(BC.parts) do
    for _, k in ipairs({ "plain", "glow" }) do
      for _, m in ipairs(p[k] or {}) do
        if m.Destroy then pcall(m.Destroy, m) end
      end
    end
  end
  BC.parts = {}
end

local function failed(why)
  wipe_meshes()
  BC.ready, BC.state, BC.why = false, "failed", tostring(why)
  return false, BC.why
end

-- ------------------------------------------------------------------------------------------------ lighting
-- The parts are drawn with a VertexLit material: the engine shades them from the vertex normals with the lights set in mentat/lit.lua (the
-- map's own light and sun at his position, a key toward the viewer, a rim light from behind him, a fill), so silver looks like metal and the
-- coat has folds. The same kit lights the found objects (holodeck/meshclient.lua), so everything stands in one light. Without the lighting
-- calls (an old client, the offline tests) the shading is baked into the vertex colours instead (B.shade) and the flat colour material is used.
local Lit = include("mentat/lit.lua")
local LIT = Lit.LIT
BC.LIT = LIT
local material_of
--- The lit material of one colour (made once): a VertexLit white texture tinted by $color, with a gloss that suits the colour.
local function lit_material(r, g, b)
  return Lit.material(r, g, b, { kind = material_of(r, g, b) })
end
--- Which material a colour wears: silver and white are the brushed metal / glossy paint, the gold trim a little glossy, navy and black plain paint.
function material_of(r, g, b)
  if r > 170 and g > 170 and b > 170 then return "metal" end
  if r > 200 and g > 180 and b < 80 then return "gold" end
  return "paint"
end

--- Set the engine's lights for drawing him. Returns true when they were set (call BC.unlight afterwards).
function BC.setup_light(bot, level)
  return Lit.setup(bot:GetPos() + Vector(0, 0, 10), "avatar")
end

function BC.unlight() Lit.unlight() end
BC.AMBIENT_FLOOR = Lit.AMBIENT_FLOOR
BC.BOB_M = 0.012                     -- metres of bob

local function build_mesh(rows)
  local out = {}
  local per = BC.MESH_VERTS - BC.MESH_VERTS % 3
  for first = 1, #rows, per do
    local last = math.min(first + per - 1, #rows)
    local m = Mesh()
    mesh.Begin(m, MATERIAL_TRIANGLES, (last - first + 1) / 3)
    for i = first, last do
      local v = rows[i]
      mesh.Position(Vector(v[1], v[2], v[3]))
      mesh.Normal(Vector(v[4], v[5], v[6]))
      mesh.Color(v[7], v[8], v[9], 255)
      mesh.TexCoord(0, 0, 0)
      mesh.AdvanceVertex()
    end
    mesh.End()
    out[#out + 1] = m
  end
  return out
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
      local plain, glow, groups, order = {}, {}, {}, {}
      B.emit(dec, function(g, x, y, z, nx, ny, nz, r, gg, b)
        if g then
          glow[#glow + 1] = { x, y, z, nx, ny, nz, r, gg, b }
        elseif LIT then
          -- a lit mesh ignores vertex colours: one mesh per colour, drawn with the colour as the material's modulation
          local k = r * 65536 + gg * 256 + b
          local gr = groups[k]
          if not gr then gr = { r = r, g = gg, b = b, rows = {} }; groups[k] = gr; order[#order + 1] = gr end
          gr.rows[#gr.rows + 1] = { x, y, z, nx, ny, nz, 255, 255, 255 }
        else
          plain[#plain + 1] = { x, y, z, nx, ny, nz, r, gg, b }
        end
      end, LIT)
      local meshes, cols = {}, {}
      if LIT then
        for _, gr in ipairs(order) do
          for _, m in ipairs(build_mesh(gr.rows)) do meshes[#meshes + 1] = m; cols[#cols + 1] = { gr.r, gr.g, gr.b } end
        end
      else
        meshes = build_mesh(plain)
      end
      built[name] = { plain = meshes, cols = cols, glow = build_mesh(glow), tris = dec.nt }
      tris = tris + dec.nt
    end
  end)
  if not ok then
    for _, p in pairs(built) do
      for _, k in ipairs({ "plain", "glow" }) do for _, m in ipairs(p[k] or {}) do if m.Destroy then pcall(m.Destroy, m) end end end
    end
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

-- ------------------------------------------------------------------------------------------------ ground contact: a soft blob shadow
--- A soft dark patch on the ground under him (a trace down; it fades as he floats higher), so he is not a sticker on the scene.
function BC.draw_shadow(bot, s)
  local r = 0.2 * B.HU_PER_M * s
  Lit.draw_shadow(bot:GetPos(), r, r, 0, 1)
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
  if LIT then
    for i, m in ipairs(p.plain) do
      local c = p.cols[i]
      render.SetMaterial(lit_material(c[1], c[2], c[3]).mat)
      m:Draw()
    end
  else
    for _, m in ipairs(p.plain) do m:Draw() end
  end
  if #p.glow > 0 then
    if LIT then render.SetColorMaterial() end                      -- the tongue is its own light: flat, saturated
    render.SetColorModulation(glow, glow, glow)                  -- the tongue brightens with his voice
    for _, m in ipairs(p.glow) do m:Draw() end
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
--- dt, height_m, scale, hide = function(name) -> bool, hat_tilt = function(lx, ly, strength, max),
--- pose = the director's body language (avatar/anim.lua: tilt, spin, bounce, hat tip, head pitch/yaw/roll, free hand) or nil, tongue = glow 0..1 or nil }.
function BC.draw(bot, ctx)
  local dt = ctx.dt
  local yaw = bot:EyeAngles().y
  local s = (ctx.height_m / BC.height_m) * ctx.scale
  local bob = BC.BOB_M * math.sin(CurTime() * 1.9) * B.HU_PER_M * s      -- he floats: a slow bob (metres), drawn, not simulated
  local zoff = -0.5 * BC.height_hu + bob                                -- his middle hovers where the can's middle did
  local A = BC.anim
  local level = ctx.level
  local P = ctx.pose                                                   -- the director's body language, or nil (then he is drawn exactly as before)

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
  if P then                                                              -- the director's body language: a bounce, then a tilt and turn about his middle
    local mid = 0.5 * BC.height_hu
    base:Translate(Vector(0, 0, P.bounce * B.HU_PER_M + mid))
    base:Rotate(Angle(P.tilt_p, P.spin, P.tilt_r))
    base:Translate(Vector(0, 0, -mid))
  end
  local glow = ctx.tongue or (0.75 + 0.25 * level)
  local function shown(name) return not ctx.hide(name) end

  if LIT then
    render.OverrideDepthEnable(true, false)                            -- the shadow tests depth but never writes it
    BC.draw_shadow(bot, s)
    render.OverrideDepthEnable(true, true)
  end
  local lit = BC.setup_light(bot, level)
  cam.PushModelMatrix(base)
  if shown("can") then draw_part("can", glow) end
  if shown("jacket") then draw_part("jacket", glow) end
  if shown("hand_b") then draw_part("hand_b", glow) end
  -- the pointing hand about the wrist (the director's free-hand swing, for emphasis and conducting, is turned about the same wrist first)
  if P and (P.hand > 0.01) then
    local w = BC.pv.wrist
    local wm = Matrix()
    wm:Translate(Vector(w.x, w.y, w.z))
    wm:Rotate(Angle(0, 45 * P.hand_side * P.hand, 60 * P.hand))
    wm:Translate(Vector(-w.x, -w.y, -w.z))
    cam.PushModelMatrix(wm, true)
  end
  cam.PushModelMatrix(mat4(B.about(rot, BC.pv.wrist)), true)
  if shown("hand_a") then draw_part("hand_a", glow) end
  cam.PopModelMatrix()
  if P and (P.hand > 0.01) then cam.PopModelMatrix() end
  -- the head about the neck; the hat sits on the head, so it turns with it and then tips about its own pivot
  local hp, tp3 = BC.pv.head, BC.pv.hat
  if P then
    local hmx = Matrix()                                                -- the head: nods with the voice (the jaw), turns, tips
    hmx:Translate(Vector(hp.x, hp.y, hp.z))
    hmx:Rotate(Angle(P.head_pitch, A.head + P.head_yaw, P.head_roll))
    hmx:Translate(Vector(-hp.x, -hp.y, -hp.z))
    cam.PushModelMatrix(hmx, true)
  else
    cam.PushModelMatrix(mat4(B.about(B.rodrigues({ x = 0, y = 0, z = 1 }, A.head), hp)), true)
  end
  if shown("head") then draw_part("head", glow) end
  local hm = Matrix()
  if P and P.hat_tip > 0.001 then hm:Translate(Vector(0, 0, P.hat_tip * 0.025 * B.HU_PER_M)) end      -- a hat tip lifts the hat first
  hm:Translate(Vector(tp3.x, tp3.y, tp3.z))
  hm:Rotate(Angle(A.hat_p + (P and P.hat_p or 0), 0, A.hat_r + (P and P.hat_r or 0)))
  hm:Translate(Vector(-tp3.x, -tp3.y, -tp3.z))
  cam.PushModelMatrix(hm, true)
  if shown("hat") then draw_part("hat", glow) end
  cam.PopModelMatrix()
  cam.PopModelMatrix()
  cam.PopModelMatrix()
  if lit then BC.unlight() end
end

BC.Body = B
return BC
