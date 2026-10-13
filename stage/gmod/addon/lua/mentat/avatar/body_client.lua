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
-- The parts are drawn with a VertexLit material: the engine shades them from the vertex normals with the lights set below (the map's
-- own light and sun at his position, a rim light from behind him, a fill), so silver looks like metal and the coat has folds. Without
-- the lighting calls (an old client, the offline tests) the shading is baked into the vertex colours instead (B.shade) and the flat colour
-- material is used.
local LIT = CLIENT and CreateMaterial ~= nil and render and render.SuppressEngineLighting ~= nil and render.SetModelLighting ~= nil
  and render.SetLocalModelLights ~= nil and render.SetMaterial ~= nil
BC.LIT = LIT
local mats = {}            -- "r_g_b" -> { mat = IMaterial, kind = "metal" | "gold" | "paint" }
local KINDS = {
  metal = { ["$phong"] = 1, ["$phongexponent"] = 45, ["$phongboost"] = 2.6, ["$phongfresnelranges"] = "[0.4 0.8 1.6]" },
  gold = { ["$phong"] = 1, ["$phongexponent"] = 30, ["$phongboost"] = 1.4, ["$phongfresnelranges"] = "[0.3 0.7 1.3]" },
  paint = { ["$phong"] = 1, ["$phongexponent"] = 18, ["$phongboost"] = 0.7, ["$phongfresnelranges"] = "[0.3 0.6 1]" },
}
local material_of
--- The lit material of one colour (made once): a VertexLit white texture tinted by $color, with a gloss that suits the colour.
local function lit_material(r, g, b)
  local key = r .. "_" .. g .. "_" .. b
  local m = mats[key]
  if m then return m end
  local kind = material_of(r, g, b)
  local tint = string.format("[%.3f %.3f %.3f]", r / 255, g / 255, b / 255)
  local p = { ["$basetexture"] = "color/white", ["$model"] = 1, ["$color"] = tint }
  for k, v in pairs(KINDS[kind]) do p[k] = v end
  m = { mat = CreateMaterial("mentat_skippy_" .. key .. "_2", "VertexLitGeneric", p), kind = kind }
  mats[key] = m
  return m
end
--- Which material a colour wears: silver and white are the brushed metal / glossy paint, the gold trim a little glossy, navy and black plain paint.
function material_of(r, g, b)
  if r > 170 and g > 170 and b > 170 then return "metal" end
  if r > 200 and g > 180 and b < 80 then return "gold" end
  return "paint"
end

-- tuning knobs (client convars): multipliers on the ambient, the key, the rim, and the specular boost
local cv_amb = CreateClientConVar("mentat_avatar_amb", "0.15", true, false, "his ambient light, times the map's")
local cv_key = CreateClientConVar("mentat_avatar_key", "1.0", true, false, "his key (sun) light")
local cv_rim = CreateClientConVar("mentat_avatar_rim", "0.45", true, false, "his rim light")
local cv_spec = CreateClientConVar("mentat_avatar_spec", "0.8", true, false, "his specular highlight boost")
local base_boost = { metal = 2.6, paint = 0.7, gold = 1.4 }

BC.light = { cache_t = -1, amb = {}, key_dir = Vector and Vector(0, 0, -1) or nil }
local function lum(v) return 0.3 * v.x + 0.59 * v.y + 0.11 * v.z end

--- Set the engine's lights for drawing him. Returns true when they were set (call BC.unlight afterwards).
function BC.setup_light(bot, level)
  if not LIT then return false end
  local L = BC.light
  local t = SysTime()
  local pos = bot:GetPos() + Vector(0, 0, 10)
  if t - L.cache_t > 0.15 then                          -- the map's light at his position, six directions, a few times a second
    L.cache_t = t
    local dirs = { [BOX_FRONT or 0] = Vector(1, 0, 0), [BOX_BACK or 1] = Vector(-1, 0, 0), [BOX_RIGHT or 2] = Vector(0, -1, 0),
      [BOX_LEFT or 3] = Vector(0, 1, 0), [BOX_TOP or 4] = Vector(0, 0, 1), [BOX_BOTTOM or 5] = Vector(0, 0, -1) }
    local top = 0
    for box, d in pairs(dirs) do
      local c = render.ComputeLighting and render.ComputeLighting(pos, d) or Vector(0.4, 0.4, 0.4)
      if d.z > 0.5 then top = lum(c) end
      L.amb[box] = c
    end
    L.top = top
    local sun = util.GetSunInfo and util.GetSunInfo() or nil
    if sun and sun.direction and sun.direction.z > 0.1 then L.sun = sun.direction else L.sun = Vector(0.45, 0.35, 0.82):GetNormalized() end
  end
  render.SuppressEngineLighting(true)
  local floor = BC.AMBIENT_FLOOR
  local ak = cv_amb:GetFloat()
  for box, c in pairs(L.amb) do
    -- never black indoors; a little lift so the shadow side keeps its colour
    render.SetModelLighting(box, math.max(c.x * ak, floor * ak), math.max(c.y * ak, floor * ak), math.max(c.z * ak, floor * ak * 1.05))
  end
  for _, m in pairs(mats) do m.mat:SetFloat("$phongboost", base_boost[m.kind] * cv_spec:GetFloat()) end
  local eye = EyePos and EyePos() or (pos + Vector(100, 0, 0))
  local toeye = eye - pos
  toeye.z = 0
  if toeye:LengthSqr() < 1 then toeye = Vector(1, 0, 0) end
  toeye:Normalize()
  local day = math.Clamp(L.top * 1.6, 0.35, 1.0)        -- how sunny it is at his position
  local keyc, rimc = cv_key:GetFloat() * day, cv_rim:GetFloat() * (0.5 + 0.5 * day)
  -- the key leans on the sun's height and sits toward the viewer's upper left (a back-lit figure would show you only his shadow side)
  local left = Vector(-toeye.y, toeye.x, 0)
  local keyto = L.sun * 0.45 + (toeye * 0.7 + left * 0.5) * 0.55 + Vector(0, 0, 0.35)
  keyto:Normalize()
  -- (the light's `dir` is the way the light travels)
  render.SetLocalModelLights({
    { type = MATERIAL_LIGHT_DIRECTIONAL, color = Vector(keyc, keyc * 0.96, keyc * 0.88), dir = -keyto },                        -- key: sun-ish, warm
    { type = MATERIAL_LIGHT_DIRECTIONAL, color = Vector(rimc * 0.55, rimc * 0.7, rimc), dir = toeye + Vector(0, 0, -0.3) },    -- rim: from behind him toward you
    { type = MATERIAL_LIGHT_DIRECTIONAL, color = Vector(0.3, 0.32, 0.38) * cv_key:GetFloat(), dir = -toeye + Vector(0.3, -0.4, -0.5) },  -- soft fill from your side
  })
  return true
end

function BC.unlight()
  if not LIT then return end
  render.SetLocalModelLights()
  render.SuppressEngineLighting(false)
end
BC.AMBIENT_FLOOR = 0.2
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
local shadow
local function shadow_mesh()
  if shadow or not LIT then return shadow end
  local mat = CreateMaterial("mentat_skippy_blob_1", "UnlitGeneric", { ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 1,
    ["$translucent"] = 1, ["$nocull"] = 1 })
  local m = Mesh()
  local rings = { { 0, 120 }, { 0.45, 85 }, { 0.75, 38 }, { 1, 0 } }
  local n = 24
  mesh.Begin(m, MATERIAL_TRIANGLES, n * (1 + 2 * (#rings - 2)))
  local function vert(r, a, alpha)
    mesh.Position(Vector(r * math.cos(a), r * math.sin(a), 0))
    mesh.Normal(Vector(0, 0, 1))
    mesh.Color(0, 0, 0, alpha)
    mesh.TexCoord(0, 0, 0)
    mesh.AdvanceVertex()
  end
  for i = 0, n - 1 do
    local a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
    vert(0, 0, rings[1][2]); vert(rings[2][1], a0, rings[2][2]); vert(rings[2][1], a1, rings[2][2])
    for k = 2, #rings - 1 do
      local r0, c0, r1, c1 = rings[k][1], rings[k][2], rings[k + 1][1], rings[k + 1][2]
      vert(r0, a0, c0); vert(r1, a0, c1); vert(r1, a1, c1)
      vert(r0, a0, c0); vert(r1, a1, c1); vert(r0, a1, c0)
    end
  end
  mesh.End()
  shadow = { mesh = m, mat = mat }
  return shadow
end

--- A soft dark patch on the ground under him (a trace down; it fades as he floats higher), so he is not a sticker on the scene.
function BC.draw_shadow(bot, s)
  local sh = shadow_mesh()
  if not sh then return end
  local pos = bot:GetPos()
  local tr = util.TraceLine({ start = pos, endpos = pos - Vector(0, 0, 900), mask = MASK_SOLID_BRUSHONLY })
  if not tr.Hit then return end
  local h = pos.z - tr.HitPos.z
  local fade = math.Clamp(1 - h / 700, 0, 1)
  if fade <= 0.02 then return end
  local ang = tr.HitNormal:Angle()
  ang.p = ang.p + 90
  local m = Matrix()
  m:Translate(tr.HitPos + tr.HitNormal * 0.6)
  m:Rotate(ang)
  local r = 0.2 * B.HU_PER_M * s * (1 + 0.15 * (1 - fade))
  m:Scale(Vector(r, r, r))
  render.SetMaterial(sh.mat)
  render.SetBlend(fade)
  cam.PushModelMatrix(m)
  sh.mesh:Draw()
  cam.PopModelMatrix()
  render.SetBlend(1)
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
--- dt, height_m, scale, hide = function(name) -> bool, hat_tilt = function(lx, ly, strength, max) }.
function BC.draw(bot, ctx)
  local dt = ctx.dt
  local yaw = bot:EyeAngles().y
  local s = (ctx.height_m / BC.height_m) * ctx.scale
  local bob = BC.BOB_M * math.sin(CurTime() * 1.9) * B.HU_PER_M * s      -- he floats: a slow bob (metres), drawn, not simulated
  local zoff = -0.5 * BC.height_hu + bob                                -- his middle hovers where the can's middle did
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
  if lit then BC.unlight() end
end

BC.Body = B
return BC
