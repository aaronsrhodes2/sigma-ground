-- holodeck_water -- shared part.  A WATER BODY of the holodeck: a basin standing on the floor (floor slab and walls, real collision) holding
-- a liquid of the sigma-ground library's properties.  Everything about the water itself (buoyancy, drag, swimming) is done by
-- mentat/water.lua; this entity is the basin, its numbers and its look (cl_init.lua).  No model, no material and no sound of the game is
-- used: the server builds the physics from convex pieces, every client draws it with the mesh API.
--
-- Local frame: x along the length, y along the width, z up from the floor.  Numbers in Hammer units (1 HU = 0.01905 m).
--   BoxMins/BoxMaxs   the outer box (the planner's footprint, like holodeck_found)
--   Shape             "rect" | "round"
--   Liquid            the library's key: water | seawater | ethanol | glycerol | mercury
--   LengthM/WidthM/DepthM   the INSIDE of the basin, in metres (the player's numbers)
--   Fill              0..1 how full (drain = 0)
--   Seq               bumps whenever the basin or its contents change (clients rebuild)
local TM = include("mentat/fx/transporter_model.lua")
local Basin = include("mentat/water/basin.lua")

ENT.Type = "anim"
ENT.Base = "base_anim"
ENT.PrintName = "Holodeck water"
ENT.Author = "sigma-ground"
ENT.Spawnable = false
ENT.AdminOnly = true
ENT.RenderGroup = RENDERGROUP_TRANSLUCENT
ENT.HolodeckStatic = true             -- the holodeck glue leaves a basin where it is (it never turns it into a loose physics prop)
ENT.IsHolodeckWater = true

function ENT:SetupDataTables()
  self:NetworkVar("Vector", 0, "BoxMins")
  self:NetworkVar("Vector", 1, "BoxMaxs")
  self:NetworkVar("String", 0, "FoundName")
  self:NetworkVar("String", 1, "Shape")
  self:NetworkVar("String", 2, "Liquid")
  self:NetworkVar("Float", 0, "LengthM")
  self:NetworkVar("Float", 1, "WidthM")
  self:NetworkVar("Float", 2, "DepthM")
  self:NetworkVar("Float", 3, "Fill")
  self:NetworkVar("Int", 0, "Seq")
  self:NetworkVar("Float", 4, "BornAt")
  self:NetworkVar("Int", 1, "FxStage")
  self:NetworkVar("Float", 5, "FxProgress")
  self:NetworkVar("Float", 6, "FxBeat")
end

--- Set everything in one call (before Spawn).  def = { mins, maxs, name, water = { shape, l_m, w_m, depth_m, liquid, fill } }.
--- The box is rebuilt from the water's own numbers (the basin is the contract), so the registry's box and the water cannot disagree.
function ENT:SetFound(def)
  def = def or {}
  local w = def.water
  if type(w) ~= "table" then return false, "no water" end
  local l, wd, d = tonumber(w.l_m), tonumber(w.w_m), tonumber(w.depth_m)
  if not (l and wd and d) or l ~= l or wd ~= wd or d ~= d or l <= 0 or wd <= 0 or d <= 0 then return false, "bad size" end
  self:SetBasin(w.shape, l, wd, d)
  self:SetLiquid(tostring(w.liquid or "water"))
  self:SetFill(math.max(0, math.min(1, tonumber(w.fill) or 1)))
  self:SetFoundName(tostring(def.name or "water"):sub(1, 60))
  self:SetBornAt(CurTime and CurTime() or 0)
  return true
end

function ENT:SetBasin(shape, l, w, d)
  local b = Basin.new(shape, l, w, d)
  self.basin = b
  self:SetShape(b.shape)
  self:SetLengthM(l); self:SetWidthM(w); self:SetDepthM(d)
  self:SetBoxMins(Vector(b.mins[1], b.mins[2], b.mins[3]))
  self:SetBoxMaxs(Vector(b.maxs[1], b.maxs[2], b.maxs[3]))
  self:SetSeq(self:GetSeq() + 1)
  return b
end

--- The basin table (rebuilt from the networked numbers; the same on both realms).
function ENT:GetBasin()
  local l, w, d = self:GetLengthM(), self:GetWidthM(), self:GetDepthM()
  local b = self.basin
  if not b or b.l_m ~= l or b.w_m ~= w or b.depth_m ~= d or b.shape ~= self:GetShape() then
    if l > 0 and w > 0 and d > 0 then b = Basin.new(self:GetShape() == "round" and "round" or "rect", l, w, d); self.basin = b else return nil end
  end
  return b
end

--- Water surface height in the entity's local z (HU), and in the world.
function ENT:SurfaceLocalZ()
  local b = self:GetBasin()
  return b and Basin.surface_z(b, self:GetFill()) or 0
end
function ENT:SurfaceWorldZ() return self:GetPos().z + self:SurfaceLocalZ() end

function ENT:FxStageName() return TM.STAGES[self:GetFxStage()] end
