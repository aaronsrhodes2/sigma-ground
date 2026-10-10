-- measure_models.lua -- load each model in the engine and record what a layout engine needs to know:
-- its oriented bounding box, physics mass and material, skins, and which horizontal side its tall geometry is on
-- (a backrest, a headboard), derived from the physics mesh. Nothing here is guessed from the file name.
-- args: {models = {"models/props_c17/furniturecouch001a.mdl", ...}}
local DIRS = { "+x", "-x", "+y", "-y" }

local function facing_stats(ent, mins, maxs)
  -- all physics-mesh vertices of all physics objects, in the entity's own (unrotated) frame
  local pts = {}
  for i = 0, ent:GetPhysicsObjectCount() - 1 do
    local po = ent:GetPhysicsObjectNum(i)
    local mesh = IsValid(po) and po:GetMesh()
    if mesh then
      for _, v in ipairs(mesh) do pts[#pts + 1] = ent:WorldToLocal(po:LocalToWorld(v.pos)) end
    end
  end
  if #pts == 0 then return nil end
  local h = maxs.z - mins.z
  local cx, cy = (mins.x + maxs.x) / 2, (mins.y + maxs.y) / 2
  local hx, hy = (maxs.x - mins.x) / 2, (maxs.y - mins.y) / 2
  local count = { ["+x"] = 0, ["-x"] = 0, ["+y"] = 0, ["-y"] = 0 }
  local high_total = 0
  for _, p in ipairs(pts) do
    if h > 0 and (p.z - mins.z) / h > 0.7 then                       -- the tall part of the body
      high_total = high_total + 1
      if hx > 0 and (p.x - cx) / hx > 0.5 then count["+x"] = count["+x"] + 1 end
      if hx > 0 and (p.x - cx) / hx < -0.5 then count["-x"] = count["-x"] + 1 end
      if hy > 0 and (p.y - cy) / hy > 0.5 then count["+y"] = count["+y"] + 1 end
      if hy > 0 and (p.y - cy) / hy < -0.5 then count["-y"] = count["-y"] + 1 end
    end
  end
  local bx = (count["+x"] - count["-x"]) / math.max(1, count["+x"] + count["-x"])
  local by = (count["+y"] - count["-y"]) / math.max(1, count["+y"] + count["-y"])
  local back, conf
  if math.abs(bx) >= math.abs(by) then back, conf = (bx >= 0 and "+x" or "-x"), math.abs(bx)
  else back, conf = (by >= 0 and "+y" or "-y"), math.abs(by) end
  return { vertices = #pts, high_vertices = high_total, bias_x = bx, bias_y = by, back = back, confidence = conf }
end

-- The flat top of a thing, from the physics mesh: the highest level at which a large upward-facing area sits
-- (a table top, a desk top, a dresser top). Returns nil for things with no usable top (a lamp, a barrel).
local function top_stats(ent, mins, maxs)
  local h = maxs.z - mins.z
  local foot = (maxs.x - mins.x) * (maxs.y - mins.y)
  local bins = {}
  for i = 0, ent:GetPhysicsObjectCount() - 1 do
    local po = ent:GetPhysicsObjectNum(i)
    local mesh = IsValid(po) and po:GetMesh()
    if mesh then
      for t = 1, #mesh - 2, 3 do
        local a = ent:WorldToLocal(po:LocalToWorld(mesh[t].pos))
        local b = ent:WorldToLocal(po:LocalToWorld(mesh[t + 1].pos))
        local c = ent:WorldToLocal(po:LocalToWorld(mesh[t + 2].pos))
        local n = (b - a):Cross(c - a)
        local len = n:Length()
        if len > 1e-6 and math.abs(n.z) / len > 0.95 then
          local key = math.floor((a.z + b.z + c.z) / 3 * 2 + 0.5) / 2
          local bin = bins[key]
          if not bin then bin = { area = 0, x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9 }; bins[key] = bin end
          bin.area = bin.area + len / 2
          for _, p in ipairs({ a, b, c }) do
            bin.x0, bin.x1 = math.min(bin.x0, p.x), math.max(bin.x1, p.x)
            bin.y0, bin.y1 = math.min(bin.y0, p.y), math.max(bin.y1, p.y)
          end
        end
      end
    end
  end
  local best_key
  for key, bin in pairs(bins) do
    if bin.area >= 0.12 * foot and key >= mins.z + 0.15 * h and (not best_key or key > best_key) then best_key = key end
  end
  if not best_key then return nil end
  local bin = bins[best_key]
  return { z = best_key, area = bin.area, x0 = bin.x0, x1 = bin.x1, y0 = bin.y0, y1 = bin.y1, footprint = foot }
end

-- What the engine itself knows about the collision body: volume, area, convex pieces, centre of mass, inertia, damping,
-- and the surface-property row (friction, elasticity, density) its material name selects.
local function phys_stats(ent, material)
  local o = { volume = 0, surface_area = 0, convexes = 0, triangles = 0 }
  for i = 0, ent:GetPhysicsObjectCount() - 1 do
    local po = ent:GetPhysicsObjectNum(i)
    if IsValid(po) then
      o.volume = o.volume + po:GetVolume()
      o.surface_area = o.surface_area + po:GetSurfaceArea()
      local conv = po:GetMeshConvexes()
      o.convexes = o.convexes + (conv and #conv or 0)
      local mesh = po:GetMesh()
      o.triangles = o.triangles + (mesh and math.floor(#mesh / 3) or 0)
      if i == 0 then
        local mc, inr = po:GetMassCenter(), po:GetInertia()
        o.mass_center = { mc.x, mc.y, mc.z }
        o.inertia = { inr.x, inr.y, inr.z }
        local ld, ad = po:GetDamping()
        o.damping = { ld, ad }
      end
    end
  end
  if material then
    local sd = util.GetSurfaceData(util.GetSurfaceIndex(material))
    if sd then
      o.surface = { friction = sd.friction, elasticity = sd.elasticity, density = sd.density, thickness = sd.thickness,
                    dampening = sd.dampening, maxspeedfactor = sd.maxSpeedFactor }
    end
  end
  return o
end

return function(args, done)
  local out = {}
  for _, mdl in ipairs(args.models or {}) do
    local r = { model = mdl }
    if not util.IsValidModel(mdl) then
      r.missing = true
    else
      local e = ents.Create("prop_physics")
      e:SetModel(mdl)
      e:SetPos(Vector(0, 0, 4000))
      e:SetAngles(Angle(0, 0, 0))
      e:Spawn()
      if not IsValid(e) then
        r.no_physics = true                                                  -- prop_physics removes itself without a collision model
      else
        local mins, maxs = e:GetModelBounds()
        local rmins, rmaxs = e:GetModelRenderBounds()
        r.mins = { mins.x, mins.y, mins.z }
        r.maxs = { maxs.x, maxs.y, maxs.z }
        r.render_mins = { rmins.x, rmins.y, rmins.z }
        r.render_maxs = { rmaxs.x, rmaxs.y, rmaxs.z }
        r.skins = e:SkinCount()
        r.bodygroups = e:GetNumBodyGroups()
        r.phys_objects = e:GetPhysicsObjectCount()
        local mass, mat = 0, nil
        for i = 0, e:GetPhysicsObjectCount() - 1 do
          local po = e:GetPhysicsObjectNum(i)
          if IsValid(po) then mass = mass + po:GetMass(); mat = mat or po:GetMaterial() end
        end
        r.mass = mass
        r.material = mat
        r.facing = facing_stats(e, mins, maxs)
        r.top = top_stats(e, mins, maxs)
        r.phys = phys_stats(e, mat)
        e:Remove()
      end
    end
    out[#out + 1] = r
  end
  done({ count = #out, models = out })
end
