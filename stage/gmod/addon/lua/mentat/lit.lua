-- lit.lua -- the one lighting and material kit every drawn thing in the game shares (Skippy's body, found objects), so a kettle and
-- the avatar stand in the same light. Client only. Extracted from avatar/body_client.lua (2026-10-12, the found-object look pass).
--
--   Lit.LIT                   true when the engine has the lighting calls (an old client and the offline tests do not: callers then bake
--                             a shade into the vertex colours and use a flat material, see Lit.shade)
--   Lit.material(r, g, b, o)  a VertexLitGeneric material, made once per look: o = { kind = "metal" | "gold" | "paint" } (Skippy's three), or
--                             { spec = 0..1, metal = 0..1, alpha = 0..255 } (a found object's material cue, see stage/skippyverse/surface.py)
--   Lit.setup(pos, key)       set the engine's lights for drawing something at `pos` (the map's own light and sun there, a key toward the
--                             viewer's upper left, a rim from behind, a fill); `key` names the thing so each keeps its own cache. true when set.
--   Lit.unlight()             put the engine back
--   Lit.draw_shadow(...)      a soft contact blob on the ground under a thing
-- Convars (shared by the avatar and the objects): mentat_avatar_amb / key / rim / spec.
local Lit = {}

local LIT = CLIENT and CreateMaterial ~= nil and render and render.SuppressEngineLighting ~= nil and render.SetModelLighting ~= nil
  and render.SetLocalModelLights ~= nil and render.SetMaterial ~= nil
Lit.LIT = LIT and true or false

local cv_amb, cv_key, cv_rim, cv_spec
if CreateClientConVar then
  cv_amb = CreateClientConVar("mentat_avatar_amb", "0.15", true, false, "the ambient light on drawn things (Skippy and found objects), times the map's")
  cv_key = CreateClientConVar("mentat_avatar_key", "1.0", true, false, "the key (sun) light on drawn things")
  cv_rim = CreateClientConVar("mentat_avatar_rim", "0.45", true, false, "the rim light on drawn things")
  cv_spec = CreateClientConVar("mentat_avatar_spec", "0.8", true, false, "the specular highlight boost on drawn things")
end
local function cvf(cv, d) return cv and cv:GetFloat() or d end

-- Skippy's three kinds of surface
Lit.KINDS = {
  metal = { ["$phong"] = 1, ["$phongexponent"] = 45, ["$phongboost"] = 2.6, ["$phongfresnelranges"] = "[0.4 0.8 1.6]" },
  gold = { ["$phong"] = 1, ["$phongexponent"] = 30, ["$phongboost"] = 1.4, ["$phongfresnelranges"] = "[0.3 0.7 1.3]" },
  paint = { ["$phong"] = 1, ["$phongexponent"] = 18, ["$phongboost"] = 0.7, ["$phongfresnelranges"] = "[0.3 0.6 1]" },
}
Lit.BASE_BOOST = { metal = 2.6, paint = 0.7, gold = 1.4 }

local mats = {}            -- name -> { mat = IMaterial, boost = the material's own base boost }

--- The phong parameters of a found object's cue: spec 0..1 (highlight strength), metal 0..1.
function Lit.cue_params(spec, metal)
  spec, metal = math.max(0, math.min(1, spec or 0)), math.max(0, math.min(1, metal or 0))
  local boost = (0.25 + 2.3 * spec) * (1 + 0.9 * metal)
  return { ["$phong"] = 1, ["$phongexponent"] = math.floor(10 + 80 * spec), ["$phongboost"] = boost,
    ["$phongfresnelranges"] = metal > 0.5 and "[0.4 0.8 1.6]" or "[0.3 0.7 1.3]" }, boost
end

--- The lit material of one look (made once). o.kind: Skippy's kinds; else o.spec / o.metal / o.alpha (0..255, below 250 = see-through).
--- `fade` true makes the variant that can be drawn at a blend below 1 (the materialise fade): translucent, so the engine honours SetBlend.
function Lit.material(r, g, b, o, fade)
  o = o or {}
  local alpha = o.alpha or 255
  local see = alpha < 250
  local key
  local params, boost
  local tr, tg, tb = r / 255, g / 255, b / 255
  if o.kind then
    key = string.format("k_%s_%d_%d_%d", o.kind, r, g, b)
    params, boost = {}, Lit.BASE_BOOST[o.kind]
    for k, v in pairs(Lit.KINDS[o.kind]) do params[k] = v end
  else
    local q = function(x) return math.floor((x or 0) * 16 + 0.5) end
    key = string.format("c_%d_%d_%d_s%d_m%d_a%d", r, g, b, q(o.spec), q(o.metal), math.floor(alpha / 16))
    params, boost = Lit.cue_params(o.spec, o.metal)
    local dark = 1 - 0.45 * math.max(0, math.min(1, o.metal or 0))        -- a metal has little diffuse: the highlight carries it
    tr, tg, tb = tr * dark, tg * dark, tb * dark
    if (o.metal or 0) > 0.5 then params["$phongtint"] = string.format("[%.3f %.3f %.3f]", 0.5 + 0.5 * r / 255, 0.5 + 0.5 * g / 255, 0.5 + 0.5 * b / 255) end
  end
  if fade or see then key = key .. "_t" end
  local m = mats[key]
  if m then return m end
  local p = { ["$basetexture"] = "color/white", ["$model"] = 1, ["$color"] = string.format("[%.3f %.3f %.3f]", tr, tg, tb) }
  for k, v in pairs(params) do p[k] = v end
  p["$phongboost"] = boost * cvf(cv_spec, 0.8)
  if fade or see then p["$translucent"] = 1; p["$alpha"] = see and (alpha / 255) or 1 end
  m = { mat = CreateMaterial("mentat_lit_" .. key .. "_3", "VertexLitGeneric", p), boost = boost, see = see }
  mats[key] = m
  return m
end

--- A flat self-lit material of one colour (a found object's glowing parts, Skippy's tongue).
local glows = {}
function Lit.glow_material(r, g, b)
  local key = r .. "_" .. g .. "_" .. b
  if not glows[key] then
    glows[key] = CreateMaterial("mentat_glow_" .. key .. "_1", "UnlitGeneric", { ["$basetexture"] = "color/white", ["$color"] = string.format("[%.3f %.3f %.3f]", r / 255, g / 255, b / 255) })
  end
  return glows[key]
end

--- The brightness 0..1 of a surface with this (unit) normal in the baked fallback: the same ambient + key + fill as avatar/body.lua's B.shade.
function Lit.shade(nx, ny, nz)
  local k = math.max(0, (nx * 0.50 + ny * 0.30 + nz * 0.81) / 0.998)
  local f = math.max(0, (nx * -0.62 + ny * -0.50 + nz * 0.30) / 0.84)
  return math.min(1, 0.50 + 0.45 * k + 0.2 * f)
end

Lit.AMBIENT_FLOOR = 0.2
local caches = {}
local function lum(v) return 0.3 * v.x + 0.59 * v.y + 0.11 * v.z end
local BOX_DIRS
local function dirs()
  if not BOX_DIRS then
    BOX_DIRS = { [BOX_FRONT or 0] = Vector(1, 0, 0), [BOX_BACK or 1] = Vector(-1, 0, 0), [BOX_RIGHT or 2] = Vector(0, -1, 0),
      [BOX_LEFT or 3] = Vector(0, 1, 0), [BOX_TOP or 4] = Vector(0, 0, 1), [BOX_BOTTOM or 5] = Vector(0, 0, -1) }
  end
  return BOX_DIRS
end

--- Set the engine's lights for drawing a thing at `pos` (a Vector, the thing's middle). Returns true when they were set (then call Lit.unlight).
function Lit.setup(pos, key)
  if not LIT then return false end
  key = key or 0
  local L = caches[key]
  if not L then L = { cache_t = -1, amb = {} }; caches[key] = L end
  local t = SysTime()
  if t - L.cache_t > 0.15 then                          -- the map's light at the thing, six directions, a few times a second
    L.cache_t = t
    local top = 0
    for box, d in pairs(dirs()) do
      local c = render.ComputeLighting and render.ComputeLighting(pos, d) or Vector(0.4, 0.4, 0.4)
      if d.z > 0.5 then top = lum(c) end
      L.amb[box] = c
    end
    L.top = top
    local sun = util.GetSunInfo and util.GetSunInfo() or nil
    if sun and sun.direction and sun.direction.z > 0.1 then L.sun = sun.direction else L.sun = Vector(0.45, 0.35, 0.82):GetNormalized() end
  end
  render.SuppressEngineLighting(true)
  local floor = Lit.AMBIENT_FLOOR
  local ak = cvf(cv_amb, 0.15)
  for box, c in pairs(L.amb) do
    -- never black indoors; a little lift so the shadow side keeps its colour
    render.SetModelLighting(box, math.max(c.x * ak, floor * ak), math.max(c.y * ak, floor * ak), math.max(c.z * ak, floor * ak * 1.05))
  end
  local spec = cvf(cv_spec, 0.8)
  if L.spec_set ~= spec then
    L.spec_set = spec
    for _, m in pairs(mats) do m.mat:SetFloat("$phongboost", m.boost * spec) end
  end
  local eye = EyePos and EyePos() or (pos + Vector(100, 0, 0))
  local toeye = eye - pos
  toeye.z = 0
  if toeye:LengthSqr() < 1 then toeye = Vector(1, 0, 0) end
  toeye:Normalize()
  local day = math.Clamp(L.top * 1.6, 0.35, 1.0)        -- how sunny it is there
  local kk = cvf(cv_key, 1.0)
  local keyc, rimc = kk * day, cvf(cv_rim, 0.45) * (0.5 + 0.5 * day)
  -- the key leans on the sun's height and sits toward the viewer's upper left (a back-lit figure would show you only its shadow side)
  local left = Vector(-toeye.y, toeye.x, 0)
  local keyto = L.sun * 0.45 + (toeye * 0.7 + left * 0.5) * 0.55 + Vector(0, 0, 0.35)
  keyto:Normalize()
  -- (the light's `dir` is the way the light travels)
  render.SetLocalModelLights({
    { type = MATERIAL_LIGHT_DIRECTIONAL, color = Vector(keyc, keyc * 0.96, keyc * 0.88), dir = -keyto },                        -- key: sun-ish, warm
    { type = MATERIAL_LIGHT_DIRECTIONAL, color = Vector(rimc * 0.55, rimc * 0.7, rimc), dir = toeye + Vector(0, 0, -0.3) },    -- rim: from behind it toward you
    { type = MATERIAL_LIGHT_DIRECTIONAL, color = Vector(0.3, 0.32, 0.38) * kk, dir = -toeye + Vector(0.3, -0.4, -0.5) },        -- soft fill from your side
  })
  return true
end

--- A material made after the last Lit.setup has the base boost until the next refresh; make the next setup re-apply the gain.
function Lit.refresh_gain()
  for _, L in pairs(caches) do L.spec_set = nil end
end

function Lit.unlight()
  if not LIT then return end
  render.SetLocalModelLights()
  render.SuppressEngineLighting(false)
end

-- ------------------------------------------------------------------------------------------------ ground contact: a soft blob shadow
local shadow
local function shadow_mesh()
  if shadow or not LIT then return shadow end
  local mat = CreateMaterial("mentat_skippy_blob_1", "UnlitGeneric", { ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 1,
    ["$translucent"] = 1, ["$nocull"] = 1 })
  local m = Mesh()
  local rings = { { 0, 205 }, { 0.5, 170 }, { 0.8, 85 }, { 1, 0 } }          -- (look pass: the first profile, 120/85/38, vanished under the thing it sat beneath)
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

--- A soft dark patch on the ground under `pos` (a trace down; it fades as the thing floats higher). `rx`, `ry`: the blob's radii in HU along
--- `yaw` degrees and across it; `strength` 0..1 scales the darkness. Returns true when something was drawn.
function Lit.draw_shadow(pos, rx, ry, yaw, strength)
  local sh = shadow_mesh()
  if not sh then return false end
  local tr = util.TraceLine({ start = pos, endpos = pos - Vector(0, 0, 900), mask = MASK_SOLID_BRUSHONLY })
  if not tr.Hit then return false end
  local h = pos.z - tr.HitPos.z
  local fade = math.Clamp(1 - h / 700, 0, 1) * (strength or 1)
  if fade <= 0.02 then return false end
  local ang = tr.HitNormal:Angle()
  ang.p = ang.p + 90
  local m = Matrix()
  m:Translate(tr.HitPos + tr.HitNormal * 0.6)
  m:Rotate(ang)
  if yaw and yaw ~= 0 then m:Rotate(Angle(0, yaw, 0)) end
  local grow = 1 + 0.15 * (1 - math.Clamp(1 - h / 700, 0, 1))
  m:Scale(Vector(rx * grow, ry * grow, rx * grow))
  render.SetMaterial(sh.mat)
  render.SetBlend(fade)
  cam.PushModelMatrix(m)
  sh.mesh:Draw()
  cam.PopModelMatrix()
  render.SetBlend(1)
  return true
end

return Lit
