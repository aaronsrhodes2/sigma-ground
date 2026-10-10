-- mentat_stage.lua -- loader for the Mentat GMod stage (sigma-ground experiments as
-- watchable, operable scenarios). Runs on BOTH realms.
AddCSLuaFile()

Mentat = Mentat or {}

local function mod(path)
  if SERVER then AddCSLuaFile(path) end
  return include(path)
end

-- Pure-Lua core (also unit-tested offline under LuaJIT; see sigma_ground/radiance/tests).
Mentat.units  = mod("mentat/units.lua")
Mentat.interp = mod("mentat/interp.lua")
Mentat.shapes = mod("mentat/shapes_core.lua")
mod("mentat/common.lua")

if SERVER then
  -- Scenarios are generated files (python -m sigma_ground.radiance.export_gmod <bundle>).
  -- Clients need them too: the body entity builds its render mesh from the same data.
  for _, f in ipairs((file.Find("mentat/scenarios/*.lua", "LUA"))) do
    AddCSLuaFile("mentat/scenarios/" .. f)
  end
  AddCSLuaFile("mentat/client.lua")
  include("mentat/server.lua")
  include("mentat/autotest.lua")
  include("mentat/experiments.lua")
else
  include("mentat/client.lua")
end
