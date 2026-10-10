-- probe_caps.lua -- what can this engine do for the ENGINE experiments? Records raw answers, grades nothing.
--
-- Trigger: data/mentat/probe_caps.txt (any content). Writes data/mentat/probe_caps.json.
-- Questions: can we define a surface material with chosen friction/elasticity at runtime; does this map
-- have water; do the constraint and physics-object functions the experiments need exist; what are the
-- engine's velocity limits; what friction/elasticity/density does Source itself assign common materials.
if not SERVER then return end

local TRIG = "mentat/probe_caps.txt"
if not file.Exists(TRIG, "DATA") then return end
file.Delete(TRIG)

local function has(tbl, name) return type(tbl) == "table" and tbl[name] ~= nil end

local function run()
  local out = { map = game.GetMap(), tick = engine.TickInterval() }

  -- 1) water in the map
  out.water = {}
  for _, e in ipairs(ents.GetAll()) do
    local c = e:GetClass()
    if c:find("water") or c:find("buoy") then
      out.water[#out.water + 1] = { class = c, pos = { e:GetPos().x, e:GetPos().y, e:GetPos().z },
                                    mins = { e:OBBMins().x, e:OBBMins().y, e:OBBMins().z },
                                    maxs = { e:OBBMaxs().x, e:OBBMaxs().y, e:OBBMaxs().z } }
    end
  end

  -- 2) function availability
  out.fn = {}
  for _, n in ipairs({ "Axis", "Elastic", "Rope", "Weld", "Ballsocket", "Hydraulic", "Slider", "Winch", "NoCollide" }) do
    out.fn["constraint." .. n] = has(constraint, n)
  end
  local ph = ents.Create("prop_physics")
  ph:SetModel("models/props_junk/PopCan01a.mdl")
  ph:Spawn()
  local po = ph:GetPhysicsObject()
  for _, n in ipairs({ "SetInertia", "GetInertia", "EnableGravity", "SetBuoyancyRatio", "GetMassCenter", "IsDragEnabled",
                       "GetDamping", "SetDamping", "SetMaterial", "GetMaterial", "ApplyForceCenter", "AddAngleVelocity" }) do
    out.fn["PhysObj:" .. n] = IsValid(po) and po[n] ~= nil
  end
  out.fn["physenv.GetPerformanceSettings"] = physenv.GetPerformanceSettings ~= nil
  out.fn["util.GetSurfaceData"] = util.GetSurfaceData ~= nil
  out.fn["physenv.AddSurfaceData"] = physenv.AddSurfaceData ~= nil
  if IsValid(ph) then ph:Remove() end

  -- 3) engine limits
  if physenv.GetPerformanceSettings then out.performance = physenv.GetPerformanceSettings() end
  out.gravity = { physenv.GetGravity().x, physenv.GetGravity().y, physenv.GetGravity().z }
  out.air_density = physenv.GetAirDensity()

  -- 4) what Source itself assigns common materials
  out.surfaces = {}
  for _, nm in ipairs({ "default", "metal", "pottery", "porcelain", "glass", "wood", "concrete", "rubber", "plastic",
                        "ice", "gmod_ice", "gmod_bouncy", "metal_bouncy", "rock", "brick", "dirt", "water" }) do
    local i = util.GetSurfaceIndex(nm)
    local d = (i >= 0 and util.GetSurfaceData) and util.GetSurfaceData(i) or nil
    out.surfaces[nm] = { index = i, data = d }
  end

  -- 5) a runtime-defined surface: do the numbers we ask for come back?
  local kv = [["mentat_probe" { "base" "default" "density" "2700" "friction" "0.30" "elasticity" "0.60" "thickness" "0" "dampening" "0" }]]
  local okadd, errAdd = pcall(physenv.AddSurfaceData, kv)
  local idx = util.GetSurfaceIndex("mentat_probe")
  out.custom_surface = { add_ok = okadd, add_err = (not okadd) and tostring(errAdd) or nil, index = idx,
                         data = (idx >= 0 and util.GetSurfaceData) and util.GetSurfaceData(idx) or nil }

  file.CreateDir("mentat")
  file.Write("mentat/probe_caps.json", util.TableToJSON(out, true))
  print("[mentat-probe-caps] wrote data/mentat/probe_caps.json")
end

hook.Add("PlayerInitialSpawn", "MentatProbeCaps", function(ply)
  timer.Simple(5, function() if IsValid(ply) then run() end end)
end)
print("[mentat-probe-caps] armed")
