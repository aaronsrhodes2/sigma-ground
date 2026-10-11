-- holodeck_found -- shared part. A PLACEHOLDER BOX for an object the holodeck found in the SkippyVerse inventory (a card from
-- stage/skippyverse: name, source, licence, size, mass) that has no game model. It needs no model file and no game content, so it
-- works on a server without Half-Life 2 or Counter-Strike: Source.
--
-- The server builds the physics (a box of the card's size, the card's mass); every client draws the box and a floating label
-- "<name> - <source>, <licence>, size <basis>" (cl_init.lua). The numbers are the card's, in Hammer units (1 HU = 0.01905 m).
--
-- THE REAL SHAPE: ENT:SetVisual(kind, data). Kind "box" is the placeholder. Kind "mesh" ({ id = n }) says the card's real shape is model
-- number n of mentat/holodeck/meshnet.lua: the server relays its triangles to the clients once per model, and each client draws them
-- at the SAME pose (same origin, same yaw; the box of the card stays the contract, so the planner's footprint, the registry, undo and
-- the limits never notice the swap). ONLY THE DRAWING changes: the physics stays the card's box. A client that does not (yet) have
-- the mesh, or fails to build it, keeps drawing the box. Server: SetVisual validates, stores the kind and number in NetworkVars (they
-- reach every client, including those who join later) and also sends "holodeck_found_visual". Client: Draw reads the NetworkVars.
ENT.Type = "anim"
ENT.Base = "base_anim"
ENT.PrintName = "Holodeck found object"
ENT.Author = "sigma-ground"
ENT.Spawnable = false
ENT.AdminOnly = true
ENT.RenderGroup = RENDERGROUP_TRANSLUCENT     -- the box is translucent

ENT.VISUALS = { box = true, mesh = true }     -- the kinds SetVisual accepts
ENT.VISUAL_KINDS = { "box", "mesh" }          -- their order, as the networked integer VisualKind

local MAX_TEXT = 80

function ENT:SetupDataTables()
  self:NetworkVar("Vector", 0, "BoxMins")      -- the card's footprint box, entity-local, Hammer units
  self:NetworkVar("Vector", 1, "BoxMaxs")
  self:NetworkVar("String", 0, "FoundName")    -- the card's name ("fire hydrant")
  self:NetworkVar("String", 1, "FoundSource")  -- where it came from ("Poly Haven")
  self:NetworkVar("String", 2, "FoundLicence") -- "CC0", "CC-BY-4.0", ...
  self:NetworkVar("String", 3, "FoundBasis")   -- how the size is known: measured | sourced | estimated | guessed
  self:NetworkVar("Float", 0, "FoundMass")     -- kg applied to the body (0 = the engine's own)
  self:NetworkVar("Int", 0, "VisualKind")      -- index (from 0) into VISUAL_KINDS
  self:NetworkVar("Int", 1, "MeshId")          -- the model number of the mesh to draw (1..64), 0 = none
end

local function text(s)
  s = tostring(s or ""):gsub("[%c\127]", " "):gsub("%s+", " ")
  return string.sub(s, 1, MAX_TEXT)
end

--- Set everything the card says in one call (before Spawn). def = { mins = {x,y,z}, maxs = {x,y,z}, name, source, licence, basis,
--- mass }. Returns true, or false and why when the box is not a box.
function ENT:SetFound(def)
  def = def or {}
  local mn, mx = def.mins, def.maxs
  if type(mn) ~= "table" or type(mx) ~= "table" then return false, "no box" end
  for i = 1, 3 do
    local a, b = tonumber(mn[i]), tonumber(mx[i])
    if not a or not b or a ~= a or b ~= b or b - a < 0.5 or b - a > 20000 then return false, "bad box" end
  end
  self:SetBoxMins(Vector(mn[1], mn[2], mn[3]))
  self:SetBoxMaxs(Vector(mx[1], mx[2], mx[3]))
  self:SetFoundName(text(def.name))
  self:SetFoundSource(text(def.source))
  self:SetFoundLicence(text(def.licence))
  self:SetFoundBasis(text(def.basis))
  self:SetFoundMass(math.max(0, tonumber(def.mass) or 0))
  self:SetVisualKind(0)
  return true
end

--- The label every client floats above the box: "<name> - <source>, <licence>, size <basis>".
function ENT:LabelText()
  local basis = self:GetFoundBasis()
  return self:GetFoundName() .. " - " .. self:GetFoundSource() .. ", " .. self:GetFoundLicence() .. ", size " .. (basis ~= "" and basis or "unknown")
end

--- Change what the object looks like (never what it is made of). Returns true, or false and why.
--- "box": the placeholder. "mesh": data = { id = <model number 1..64> }; the mesh itself travels by net message (meshnet.lua).
function ENT:SetVisual(kind, data)
  if not self.VISUALS[kind] then return false, "unsupported visual kind: " .. tostring(kind) end
  if kind == "mesh" then
    local id = type(data) == "table" and tonumber(data.id)
    if not id or id ~= id or id < 1 or id > 64 or id % 1 ~= 0 then return false, "a mesh visual needs an id from 1 to 64" end
    self:SetMeshId(id)
  else
    self:SetMeshId(0)
  end
  for i, k in ipairs(self.VISUAL_KINDS) do
    if k == kind then self:SetVisualKind(i - 1) end
  end
  if SERVER and self.OnVisualChanged then self:OnVisualChanged(kind) end
  return true
end

--- Which visual is current ("box").
function ENT:GetVisual()
  return self.VISUAL_KINDS[self:GetVisualKind() + 1] or "box"
end
