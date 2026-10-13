-- mentat_world.lua -- loads the construct's physics (mentat/world.lua) after the stage and the holodeck are up.
-- Autorun files load in alphabetical order, so "mentat_stage.lua" (which loads the holodeck) has run before this one.
AddCSLuaFile()
if SERVER then
  AddCSLuaFile("mentat/world/constants_data.lua")
  AddCSLuaFile("mentat/world/constants.lua")
  AddCSLuaFile("mentat/world/aero.lua")
  AddCSLuaFile("mentat/world_client.lua")
  include("mentat/world.lua")
else
  include("mentat/world_client.lua")
end
