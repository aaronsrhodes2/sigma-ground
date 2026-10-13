-- holodeck_water -- server part: the basin's physics (a floor and walls of convex pieces, static) and its resizing.  The water's effects
-- on bodies and players are mentat/water.lua's.
AddCSLuaFile("shared.lua")
AddCSLuaFile("cl_init.lua")
AddCSLuaFile("mentat/fx/transporter_model.lua")
AddCSLuaFile("mentat/fx/transporter.lua")
include("shared.lua")

local function build_physics(self)
  local b = self:GetBasin()
  if not b then return false end
  local Basin = include("mentat/water/basin.lua")
  local parts = {}
  for i, pts in ipairs(Basin.parts(b)) do
    local vs = {}
    for j, p in ipairs(pts) do vs[j] = Vector(p[1], p[2], p[3]) end
    parts[i] = vs
  end
  if self.PhysicsDestroy then self:PhysicsDestroy() end
  if not self:PhysicsInitMultiConvex(parts) then return false end
  self:SetSolid(SOLID_VPHYSICS)
  self:SetMoveType(MOVETYPE_NONE)
  if self.EnableCustomCollisions then self:EnableCustomCollisions(true) end
  local po = self:GetPhysicsObject()
  if not IsValid(po) then return false end
  po:EnableMotion(false)
  self:DrawShadow(false)
  return true
end

function ENT:Initialize()
  if not self:GetBasin() then
    ErrorNoHalt("[holodeck_water] no basin was set before Spawn\n")
    self:Remove()
    return
  end
  if not build_physics(self) then
    ErrorNoHalt("[holodeck_water] the basin's physics could not be built\n")
    self:Remove()
    return
  end
  self:SetFxStage(0)
end

--- Change the basin (deeper, bigger, ...): rebuild the physics, bump the sequence so clients redraw.  Returns the new basin.
function ENT:Resize(shape, l, w, d)
  self:SetBasin(shape or self:GetShape(), l, w, d)
  build_physics(self)
  return self:GetBasin()
end

function ENT:SetLevel(fill) self:SetFill(math.max(0, math.min(1, fill))); self:SetSeq(self:GetSeq() + 1) end
function ENT:SetLiquidKey(key) self:SetLiquid(key); self:SetSeq(self:GetSeq() + 1) end
