-- holodeck_found -- server part: a box of physics at runtime, of the card's size and mass. No model is set (none is needed, and a
-- server without the game's content has none to give).
AddCSLuaFile("shared.lua")
AddCSLuaFile("cl_init.lua")
AddCSLuaFile("mentat/holodeck/svm1.lua")             -- the client's decoder and mesh builder (the server sends them with this entity)
AddCSLuaFile("mentat/holodeck/meshclient.lua")
AddCSLuaFile("mentat/fx/transporter_model.lua")      -- the transporter effect: stages and the pure coherence function
AddCSLuaFile("mentat/fx/transporter.lua")
include("shared.lua")

util.AddNetworkString("holodeck_found_visual")

function ENT:Initialize()
  local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
  if mx.x - mn.x < 0.5 or mx.y - mn.y < 0.5 or mx.z - mn.z < 0.5 then
    ErrorNoHalt("[holodeck_found] no box was set before Spawn\n")
    self:Remove()
    return
  end
  if not self:PhysicsInitBox(mn, mx) then
    ErrorNoHalt("[holodeck_found] the physics box could not be built\n")
    self:Remove()
    return
  end
  self:SetSolid(SOLID_VPHYSICS)
  self:SetMoveType(MOVETYPE_VPHYSICS)
  self:DrawShadow(false)
  local po = self:GetPhysicsObject()
  if not IsValid(po) then
    ErrorNoHalt("[holodeck_found] the box has no physics object\n")
    self:Remove()
    return
  end
  local mass = self:GetFoundMass()
  if mass > 0 then po:SetMass(math.Clamp(mass, 0.1, 50000)) end       -- the card's mass; 0 keeps the engine's own
  po:Wake()
end

--- A visual changed: tell every client (they redraw). The physics stays the card's box for every kind.
function ENT:OnVisualChanged(kind)
  net.Start("holodeck_found_visual")
  net.WriteUInt(self:EntIndex(), 16)
  net.WriteString(kind)
  net.WriteUInt(self:GetMeshId(), 8)
  net.Broadcast()
end
