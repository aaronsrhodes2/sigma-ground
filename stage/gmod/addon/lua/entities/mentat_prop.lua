-- mentat_prop -- one standalone body of an ENGINE experiment (a sphere, box, cylinder or ring with a
-- chosen mass and surface material). Unlike mentat_body it is not tied to a replayed scenario: the
-- experiment framework (mentat/experiments.lua) spawns it from a spec.
--
-- SERVER: the physics object (true sphere / box, or convex hulls for cylinders and rings), with the
--         spec's mass, surface material, gravity switch and damping. Air drag is always off.
-- CLIENT: draws an IMesh of the same shape, lit, tinted by the spec colour.
AddCSLuaFile()

ENT.Type = "anim"
ENT.Base = "base_anim"
ENT.PrintName = "Mentat Prop"
ENT.Author = "sigma-ground"
ENT.Spawnable = false
ENT.AdminOnly = true
ENT.RenderGroup = RENDERGROUP_OPAQUE

function ENT:SetupDataTables()
  self:NetworkVar("String", 0, "Shape")        -- sphere | box | cylinder | ring
  self:NetworkVar("Vector", 0, "Dims")         -- see Mentat.shapes.build_prop for the meaning per shape
  self:NetworkVar("Vector", 1, "Tint")
  self:NetworkVar("Int", 0, "Seg")
end

--- The shapes_core dimension table for a shape and its networked Dims vector (metres).
local function dims_table(shape, v, seg)
  if shape == "sphere" then return { radius = v.x, seg = seg > 0 and seg or nil } end
  if shape == "box" then return { size = { v.x, v.y, v.z } } end
  if shape == "cylinder" then return { radius = v.x, height = v.y, seg = seg > 0 and seg or nil } end
  if shape == "ring" then return { outer_radius = v.x, inner_radius = v.y, height = v.z, seg = seg > 0 and seg or nil } end
end

if SERVER then
  local MASS_MIN, MASS_MAX = 0.01, 50000

  local function fail(self, why)
    ErrorNoHalt("[mentat] prop not built: ", why, "\n")
    self.MentatReady, self.MentatError = false, why
    self:Remove()
  end

  function ENT:Initialize()
    self.MentatReady = false
    local def = self.MentatDef
    if not def then return fail(self, "no definition") end
    local shape, scale = self:GetShape(), Mentat.HU
    local dims = dims_table(shape, self:GetDims(), self:GetSeg())
    if not dims then return fail(self, "bad shape " .. tostring(shape)) end
    local g = Mentat.shapes.build_prop(shape, dims, scale)

    self:SetModel("models/props_junk/PopCan01a.mdl")      -- placeholder: bounds/PVS only
    local surf = def.surface or "default"
    local ok
    if shape == "sphere" then
      ok = self:PhysicsInitSphere(dims.radius * scale, surf)             -- a true sphere: rolls without facets
    elseif shape == "box" then
      local h = Vector(g.maxs[1], g.maxs[2], g.maxs[3])
      ok = self:PhysicsInitBox(-h, h, surf)
    else
      local hulls = {}
      for _, hull in ipairs(g.hulls) do
        local pts = {}
        for i, p in ipairs(hull) do pts[i] = Vector(p[1], p[2], p[3]) end
        hulls[#hulls + 1] = pts
      end
      ok = self:PhysicsInitMultiConvex(hulls, surf, Vector(0, 0, 0))
    end
    if not ok then return fail(self, "physics init failed for " .. shape) end
    self:SetSolid(SOLID_VPHYSICS)
    self:SetMoveType(MOVETYPE_VPHYSICS)
    self:EnableCustomCollisions(true)
    self:DrawShadow(false)

    local phys = self:GetPhysicsObject()
    if not IsValid(phys) then return fail(self, "no physics object") end
    local mass = math.Clamp(def.mass_kg or (g.volume_m3 * 1000), MASS_MIN, MASS_MAX)
    phys:SetMass(mass)
    phys:EnableDrag(false)                                  -- Source's air drag is not physics (measured)
    local _, ang = phys:GetDamping()
    local lin = def.damping_linear or 0                      -- the engine's own 0.1/s is removed unless asked for
    phys:SetDamping(lin, def.damping_angular or ang)
    phys:EnableGravity(def.gravity ~= false)
    -- A free body is NOT frozen: the experiment keeps it awake, weightless and spinning until the release (see
    -- hold() in experiments.lua; a frozen-then-woken body picked up engine damping at random). Static bodies freeze.
    if def.static then phys:EnableMotion(false) else phys:EnableGravity(false) end
    phys:Wake()
    self.MentatMass = phys:GetMass()
    self.MentatReady = true
  end
end

if CLIENT then
  local material
  local function get_material()
    if not material then
      material = CreateMaterial("mentat_prop_v1", "VertexLitGeneric", {
        ["$basetexture"] = "color/white", ["$model"] = 1, ["$nocull"] = 1 })
    end
    return material
  end

  local function destroy(self)
    if self.MentatMesh then self.MentatMesh:Destroy(); self.MentatMesh = nil end
  end

  function ENT:Initialize() self.MentatNext = 0 end

  function ENT:Think()
    if not self.MentatGeo and CurTime() >= (self.MentatNext or 0) then
      local shape = self:GetShape()
      if shape ~= "" then
        local dims = dims_table(shape, self:GetDims(), self:GetSeg())
        if dims then
          local g = Mentat.shapes.build_prop(shape, dims, Mentat.HU)
          self.MentatGeo = g
          self:SetRenderBounds(Vector(g.mins[1], g.mins[2], g.mins[3]), Vector(g.maxs[1], g.maxs[2], g.maxs[3]))
        end
      end
      self.MentatNext = CurTime() + 1
    end
    self:NextThink(CurTime() + 0.2)
    return true
  end

  function ENT:Draw()
    local g = self.MentatGeo
    if not g then return end
    if not self.MentatMesh then
      local m = Mesh()
      mesh.Begin(m, MATERIAL_TRIANGLES, #g.verts / 3)
      for _, v in ipairs(g.verts) do
        mesh.Position(Vector(v[1], v[2], v[3]))
        mesh.Normal(Vector(v[4], v[5], v[6]))
        mesh.TexCoord(0, 0, 0)
        mesh.AdvanceVertex()
      end
      mesh.End()
      self.MentatMesh = m
    end
    local mat = Matrix()
    mat:SetTranslation(self:GetPos())
    mat:SetAngles(self:GetAngles())
    cam.PushModelMatrix(mat)
    render.SetMaterial(get_material())
    local c = self:GetTint()
    render.SetColorModulation(c.x, c.y, c.z)
    self.MentatMesh:Draw()
    cam.PopModelMatrix()
    render.SetColorModulation(1, 1, 1)
  end

  function ENT:OnRemove() destroy(self) end
end
