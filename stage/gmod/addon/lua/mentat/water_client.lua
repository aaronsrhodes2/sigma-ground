-- water_client.lua -- what a client does for water: receives the server's splashes, tints the view under the surface, and swims the same way
-- the server does (prediction).  The look of the water itself is mentat/water/render.lua, drawn by the holodeck_water entity.
if not CLIENT then return end
Mentat = Mentat or {}
MentatWaterClient = MentatWaterClient or {}
local Wc = MentatWaterClient
Mentat.WaterClient = Wc

local C = include("mentat/water/constants.lua")
local Vol = include("mentat/water/volumes.lua")
local R = include("mentat/water/render.lua")
local Splash = include("mentat/water/splash.lua")
local FxState = include("mentat/water/fx.lua")
local Move = include("mentat/water/swim_move.lua")
local WC = include("mentat/world/constants.lua")
Wc.render, Wc.volumes = R, Vol
local HU_M = C.HU_M
Move.cd = WC.CD.box.value

local function g_now() return GetGlobalFloat("mentat_world_g_m_s2", C.data.g_earth.value) end

-- ------------------------------------------------------------------------------------------------ splashes from the server
net.Receive("mentat_water_splash", function()
  local idx = net.ReadUInt(16)
  local kind = net.ReadUInt(3)
  local x, y, z = net.ReadFloat(), net.ReadFloat(), net.ReadFloat()
  local mass, speed = net.ReadFloat(), net.ReadFloat()
  local ent = Entity(idx)
  if not (IsValid(ent) and ent.GetBasin) then return end
  local rec = Vol.record(ent)
  if not rec then return end
  local g = g_now()
  if g <= 0 then return end
  local plan = Splash.plan(mass, speed, rec.L.density_kg_m3, g)
  if not plan then return end
  if kind == 3 then plan.n = 0 end                                                  -- a body bobbing: a ring, no spray
  local st = R.state_of(ent)
  local depth = math.max(0.02, rec.b.depth_m * rec.fill)
  FxState.add_splash(st.fx, x * HU_M, y * HU_M, plan, CurTime(), g, depth, rec.L.surface_tension_n_m / rec.L.density_kg_m3)
end)

-- ------------------------------------------------------------------------------------------------ under the surface
--- The water body the camera is in and how deep, or nil.
function Wc.camera_water()
  local eye = EyePos()
  for _, rec in ipairs(Vol.list()) do
    if rec.fill > 0 and R.eye_inside(rec, eye) then return rec, (rec.surf - eye.z) * HU_M end
  end
  return nil
end

hook.Add("HUDPaintBackground", "MentatWater.Tint", function()
  local rec, depth = Wc.camera_water()
  if not rec then return end
  local r, g, b, a = R.underwater_look(depth)
  surface.SetDrawColor(r, g, b, a)
  surface.DrawRect(0, 0, ScrW(), ScrH())
end)

hook.Add("SetupWorldFog", "MentatWater.Fog", function()
  local rec, depth = Wc.camera_water()
  if not rec then return end
  local r, g, b, _, dist = R.underwater_look(depth)
  render.FogMode(MATERIAL_FOG_LINEAR)
  render.FogStart(0)
  render.FogEnd(dist)
  render.FogMaxDensity(0.97)
  render.FogColor(r, g, b)
  return true
end)

hook.Add("SetupSkyboxFog", "MentatWater.SkyFog", function(scale)
  local rec, depth = Wc.camera_water()
  if not rec then return end
  local r, g, b, _, dist = R.underwater_look(depth)
  render.FogMode(MATERIAL_FOG_LINEAR)
  render.FogStart(0)
  render.FogEnd(dist * scale)
  render.FogMaxDensity(0.97)
  render.FogColor(r, g, b)
  return true
end)

-- ------------------------------------------------------------------------------------------------ swimming (predicted with the server's numbers)
Move.install()

-- ------------------------------------------------------------------------------------------------ the cost
concommand.Add("mentat_water_perf", function()
  local s = R.stats
  print(string.format("[water] draw %.3f ms mean (EMA), worst %.3f ms, %d triangles last frame, %d frames", s.draw_ms, s.worst_ms, s.tris, s.frames))
end, nil, "The water's drawing cost (CPU side, milliseconds per frame)")
