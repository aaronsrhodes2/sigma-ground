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

local MASS_MIN, MASS_MAX = 0.1, 50000     -- PhysObj:SetMass clamps to this range

function ENT:SetupDataTables()
  self:NetworkVar("String", 0, "Scenario")
  self:NetworkVar("Int", 0, "BodyIndex")
  self:NetworkVar("Bool", 0, "Released")
end

if SERVER then
  -- Initialize runs inside Spawn(). A body that cannot be built removes itself, but an entity
  -- removed there still passes IsValid() until the next tick (verified in game by the review),
  -- so success is announced with MentatReady and callers must test that, not IsValid.
  local function fail(self, why)
    ErrorNoHalt("[mentat] body not built: ", why, "\n")
    self.MentatReady = false
    self.MentatError = why
    self:Remove()
  end

  function ENT:Initialize()
    self.MentatReady = false
    local name, idx = self:GetScenario(), self:GetBodyIndex()
    local geo = Mentat.GetGeometry(name, idx)
    local scn = Mentat.GetScenario(name)
    if not geo or not scn then return fail(self, "no geometry for " .. tostring(name) .. " #" .. tostring(idx)) end
    local body = scn.bodies[idx]

    self:SetModel("models/props_junk/PopCan01a.mdl")   -- placeholder: needed for bounds/PVS only

    local hulls = {}
    for _, sh in ipairs(geo.shapes) do
      local pts = {}
      for i, p in ipairs(sh.hull) do pts[i] = Vector(p[1], p[2], p[3]) end
      hulls[#hulls + 1] = pts
    end

    -- A surface name GMod does not know silently degrades to the default material (or fails).
    local sp = body.surfaceprop or "default"
    if util.GetSurfaceIndex(sp) == -1 then
      ErrorNoHalt("[mentat] unknown surface property '", sp, "' for ", tostring(name), ": using 'default'\n")
      sp = "default"
    end

    -- Centre of mass at the entity origin == the bundle's pivot (physics.com_m).
    if not self:PhysicsInitMultiConvex(hulls, sp, Vector(0, 0, 0)) then
      return fail(self, "PhysicsInitMultiConvex failed for " .. tostring(name) .. " #" .. tostring(idx))
    end
    self:SetSolid(SOLID_VPHYSICS)
    self:SetMoveType(MOVETYPE_VPHYSICS)
    self:EnableCustomCollisions(true)
    self:DrawShadow(false)

    local phys = self:GetPhysicsObject()
    if not IsValid(phys) then return fail(self, "no physics object after PhysicsInitMultiConvex") end

    local want = body.mass_kg or (geo.volume_m3 * 1000)   -- fallback: water density, flagged by the exporter
    local mass = math.Clamp(want, MASS_MIN, MASS_MAX)
    if mass ~= want then
      ErrorNoHalt(string.format("[mentat] %s #%d: mass %.6g kg is outside GMod's %g..%g kg and was clamped to %g\n",
        tostring(name), idx, want, MASS_MIN, MASS_MAX, mass))
    end
    phys:SetMass(mass)
    self.MentatMass, self.MentatMassWanted = mass, want

    -- Source slows every physics object in two ways, both far stronger than the real drag in the
    -- bundles: air drag (~9 % over a 3 m fall) and a LINEAR DAMPING of exactly 0.1 per second (measured).
    -- A released body falls in VACUUM so its flight is exactly checkable, so both go. ANGULAR damping
    -- is deliberately kept at the engine default: with it zeroed a landed cylinder tipped over and
    -- rolled forever (observed), because that damping is what lets a rolling body come to rest.
    phys:EnableDrag(false)
    local okd, _lin, ang = pcall(phys.GetDamping, phys)
    if okd and ang then phys:SetDamping(0, ang) end

    phys:EnableMotion(false)                              -- kinematic until released
    phys:Wake()
    self.MentatReady = true
  end

  --- Hand this body to Havok at its current pose. vel is the Source-world velocity (HU/s);
  --- angvel_local_deg is degrees/s about the body's LOCAL forward/left/up (what the setter takes).
  function ENT:MentatRelease(vel, angvel_local_deg)
    self.MentatReleased = true
    self:SetReleased(true)
    local phys = self:GetPhysicsObject()
    if not IsValid(phys) then return false end
    phys:EnableMotion(true)
    phys:Wake()
    phys:SetVelocity(vel)
    -- Always written, zero included, so a spin left from an earlier release cannot survive.
    phys:SetAngleVelocityInstantaneous(angvel_local_deg or Vector(0, 0, 0))
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
        ["$nocull"] = 1,        -- the triangles are wound CCW-outward; nocull makes front/back irrelevant
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

  local function destroy_meshes(self)
    if self.MentatMeshes then
      for _, e in ipairs(self.MentatMeshes) do e.mesh:Destroy() end
      self.MentatMeshes = nil
    end
  end

  function ENT:Initialize()
    self.MentatGeo, self.MentatEpoch, self.MentatNextTry, self.MentatWarned = nil, nil, 0, false
  end

  --- Resolve geometry and render bounds. Called from Think (not Draw): the entity must have correct
  --- bounds BEFORE it is ever drawn or the engine culls it. A failure is retried, never cached forever.
  function ENT:ResolveGeometry()
    local epoch = GetGlobal2Int("mentat_epoch", 0)
    if self.MentatEpoch ~= epoch then                     -- scenarios were regenerated: drop stale copies
      self.MentatEpoch = epoch
      Mentat.ReloadScenarios()
      destroy_meshes(self)
      self.MentatGeo = nil
    end
    if self.MentatGeo then return self.MentatGeo end
    if CurTime() < self.MentatNextTry then return nil end
    local name = self:GetScenario()
    if name == "" then return nil end                      -- not networked yet: try again next frame
    local geo = Mentat.GetGeometry(name, self:GetBodyIndex())
    if not geo then
      self.MentatNextTry = CurTime() + 2
      if not self.MentatWarned then
        self.MentatWarned = true
        local _, why = Mentat.GetScenario(name)
        ErrorNoHalt("[mentat] client cannot build body '", name, "' #", self:GetBodyIndex(), ": ", tostring(why or "no geometry"),
          " (a scenario over ~64 KB compressed never reaches clients)\n")
      end
      return nil
    end
    self.MentatGeo = geo
    self:SetRenderBounds(Vector(geo.mins[1], geo.mins[2], geo.mins[3]), Vector(geo.maxs[1], geo.maxs[2], geo.maxs[3]))
    return geo
  end

  function ENT:Think()
    self:ResolveGeometry()
    self:NextThink(CurTime() + 0.25)
    return true
  end

  function ENT:Draw()
    local geo = self.MentatGeo
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
    destroy_meshes(self)
  end
end
