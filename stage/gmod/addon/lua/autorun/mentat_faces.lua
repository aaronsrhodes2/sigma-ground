-- mentat_faces.lua -- a player's picture as their head (stage/gmod/holodeck/README.md, section Faces). Its own loader, so the
-- stage loader is untouched. Server: faces_server.lua (records, commands, the bridge, sending). Client: faces_client.lua (drawing).
AddCSLuaFile()

if SERVER then
  AddCSLuaFile("mentat/faces_head.lua")
  AddCSLuaFile("mentat/faces_client.lua")
  include("mentat/faces_server.lua")
else
  include("mentat/faces_client.lua")
end
