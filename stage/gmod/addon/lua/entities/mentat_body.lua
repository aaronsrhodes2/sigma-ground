-- mentat_body -- one sigma-ground body on the stage.
--
-- SERVER: a VPhysics object built from the scenario's convex hulls (so it collides and
--         can be handed to Havok on release). While replaying it is frozen and moved by
--         Mentat.Apply() each tick.
-- CLIENT: draws an IMesh built from the SAME shape data as the physics hull.
--
-- The entity ORIGIN is the body's pivot (= centre of mass), matching how the web
-- viewer places a body-bound leaf at  pos + R*(local - pivot).
AddCSLuaFile()

ENT.Type = "anim"
ENT.Base = "base_anim"
ENT.PrintName = "Mentat Body"
ENT.Author = "sigma-ground"
ENT.Spawnable = false
ENT.AdminOnly = true
ENT.RenderGroup = RENDERGROUP_OPAQUE

function ENT:SetupDataTables()
  self:NetworkVar("String", 0, "Scenario")
  self:NetworkVar("Int", 0, "BodyIndex")
  self:NetworkVar("Bool", 0, "Released")
end

if SERVER then
  function ENT:Initialize()
    local name, idx = self:GetScenario(), self:GetBodyIndex()
    local geo = Mentat.GetGeometry(name, idx)
    local scn = Mentat.GetScenario(name)
    if not geo or not scn then
      ErrorNoHalt("[mentat] body without geometry: ", tostring(name), " #", tostring(idx), "\n")
      self:Remove()
      return
    end
    local body = scn.bodies[idx]

    self:SetModel("models/props_junk/PopCan01a.mdl")   -- placeholder: needed for bounds/PVS only

    local hulls = {}
    for _, sh in ipairs(geo.shapes) do
      local pts = {}
      for i, p in ipairs(sh.hull) do pts[i] = Vector(p[1], p[2], p[3]) end
      hulls[#hulls + 1] = pts
    end
    -- Centre of mass at the entity origin == the bundle's pivot (physics.com_m).
    if not self:PhysicsInitMultiConvex(hulls, body.surfaceprop or "default", Vector(0, 0, 0)) then
      ErrorNoHalt("[mentat] PhysicsInitMultiConvex failed for ", tostring(name), " #", tostring(idx), "\n")
      self:Remove()
      return
    end
    self:SetSolid(SOLID_VPHYSICS)
    self:SetMoveType(MOVETYPE_VPHYSICS)
    self:EnableCustomCollisions(true)
    self:DrawShadow(false)

    local phys = self:GetPhysicsObject()
    if IsValid(phys) then
      local mass = body.mass_kg or (geo.volume_m3 * 1000)   -- fallback: water density, flagged by the exporter
      phys:SetMass(math.Clamp(mass, 0.1, 50000))
      phys:EnableMotion(false)                              -- kinematic until released
      phys:Wake()
    end
  end

  --- Hand this body to Havok at its current pose. velocity/angular velocity are in the
  --- Source world (HU/s, deg/s).
  function ENT:MentatRelease(vel, angvel_deg)
    self.MentatReleased = true
    self:SetReleased(true)
    local phys = self:GetPhysicsObject()
    if not IsValid(phys) then return false end
    phys:EnableMotion(true)
    phys:Wake()
    phys:SetVelocity(vel)
    if angvel_deg and angvel_deg:LengthSqr() > 1e-9 then
      phys:SetAngleVelocityInstantaneous(angvel_deg)        -- frame/units UNVERIFIED in Phase 0; see README
    end
    return true
  end
end

if CLIENT then
  local material

  local function get_material()
    if not material then
      material = CreateMaterial("mentat_body_v1", "VertexLitGeneric", {
        ["$basetexture"] = "color/white",
        ["$model"] = 1,
        ["$nocull"] = 1,        -- winding is verified offline; nocull makes the first look forgiving
      })
    end
    return material
  end

  local function build_meshes(geo)
    local list = {}
    for _, sh in ipairs(geo.shapes) do
      local m = Mesh()
      local verts = sh.verts
      mesh.Begin(m, MATERIAL_TRIANGLES, #verts / 3)
      for _, v in ipairs(verts) do
        mesh.Position(Vector(v[1], v[2], v[3]))
        mesh.Normal(Vector(v[4], v[5], v[6]))
        mesh.TexCoord(0, 0, 0)
        mesh.AdvanceVertex()
      end
      mesh.End()
      list[#list + 1] = { mesh = m, r = sh.color[1], g = sh.color[2], b = sh.color[3] }
    end
    return list
  end

  -- NetworkVars are not populated yet inside ENT:Initialize on the client, so geometry
  -- is resolved lazily from Draw.
  function ENT:Initialize() end

  function ENT:ResolveGeometry()
    if self.MentatGeo ~= nil then return self.MentatGeo or nil end
    local name = self:GetScenario()
    if name == "" then return nil end                      -- not networked yet: retry next frame
    local geo = Mentat.GetGeometry(name, self:GetBodyIndex())
    self.MentatGeo = geo or false
    if geo then
      self:SetRenderBounds(Vector(geo.mins[1], geo.mins[2], geo.mins[3]),
                           Vector(geo.maxs[1], geo.maxs[2], geo.maxs[3]))
    end
    return geo
  end

  function ENT:Draw()
    local geo = self:ResolveGeometry()
    if not geo then return end
    self.MentatMeshes = self.MentatMeshes or build_meshes(geo)

    local m = Matrix()
    m:SetTranslation(self:GetPos())
    m:SetAngles(self:GetAngles())
    cam.PushModelMatrix(m)
    render.SetMaterial(get_material())
    for _, e in ipairs(self.MentatMeshes) do
      render.SetColorModulation(e.r, e.g, e.b)
      e.mesh:Draw()
    end
    cam.PopModelMatrix()
    render.SetColorModulation(1, 1, 1)
  end

  function ENT:OnRemove()
    if self.MentatMeshes then
      for _, e in ipairs(self.MentatMeshes) do e.mesh:Destroy() end
      self.MentatMeshes = nil
    end
  end
end
