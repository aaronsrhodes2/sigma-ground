-- mentat_water.lua -- loads the construct's water (mentat/water.lua) after the stage and the holodeck are up.
-- Autorun files load in alphabetical order; "mentat_stage.lua" (which loads the holodeck) runs before this one.
AddCSLuaFile()
if SERVER then
  for _, f in ipairs({ "data", "constants", "basin", "hydro", "waves", "splash", "volumes", "swim", "swim_move", "fx", "render" }) do
    AddCSLuaFile("mentat/water/" .. f .. ".lua")
  end
  AddCSLuaFile("mentat/world/constants_data.lua")
  AddCSLuaFile("mentat/world/constants.lua")
  AddCSLuaFile("mentat/water_client.lua")
  include("mentat/water.lua")
else
  include("mentat/water_client.lua")
end
