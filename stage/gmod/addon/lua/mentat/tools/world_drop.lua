-- world_drop.lua -- measure the construct's physics in the real engine: real drops (fall time, impact speed) and whether layouts still settle.
-- Headless (dedicated server): python stage/gmod/holodeck/world_drop.py   (grades the result against the sigma-ground library).
--
-- args = { height_m = 3, presets = {"earth","moon"}, modes = {"real","vacuum","engine"}, objects = {{name=, model=, mass_kg=}, ...},
--          layouts = {"a pile of crates", ...}, settle = {8, 20, 32} }
--   modes: real   = the construct (gravity + the real aerodynamic drag of world.lua, Source's own drag off)
--          vacuum = gravity only (our drag off, Source's drag off): the engine's integration alone
--          engine = Source's own air drag on, ours off (what the game does by default)
if not SERVER then return end

local function vec(v) return { v.x, v.y, v.z } end

return function(args, done)
  local W, H = MentatWorld, Holodeck
  if not W then return done(nil, "the construct (mentat/world.lua) is not loaded on this server") end
  local C = W.constants
  local dt = engine.TickInterval()
  local height_m = args.height_m or 3
  local objects = args.objects or {
    { name = "crate", model = "models/props_junk/wood_crate001a.mdl", mass_kg = 12 },
    { name = "ball", model = "models/hunter/misc/sphere025x025.mdl", mass_kg = 0.45 },
    { name = "plate", model = "models/hunter/plates/plate025x025.mdl", mass_kg = 0.005 },
  }
  local presets = args.presets or { "earth", "moon" }
  local modes = args.modes or { "real", "vacuum", "engine" }
  local result = { dt = dt, height_m = height_m, drops = {}, layouts = {}, engine = {} }

  local function floor_z(x, y)
    local z0 = 0
    for _, e in ipairs(ents.FindByClass("info_player_start")) do z0 = e:GetPos().z; break end
    local tr = util.TraceLine({ start = Vector(x, y, z0 + 200), endpos = Vector(x, y, z0 - 20000), mask = MASK_SOLID_BRUSHONLY })
    return tr.Hit and tr.HitPos.z or z0
  end

  local function configure(preset, mode)
    GetConVar("mentat_world_enabled"):SetBool(true)
    GetConVar("mentat_world_preset"):SetString(preset)
    GetConVar("mentat_world_aero"):SetBool(mode == "real")
    GetConVar("mentat_world_engine_drag"):SetBool(mode == "engine")
    W.applied = {}
    W.apply(false)
  end

  local jobs = {}
  for _, p in ipairs(presets) do for _, m in ipairs(modes) do for _, o in ipairs(objects) do jobs[#jobs + 1] = { preset = p, mode = m, obj = o } end end end

  -- ----------------------------------------------------------------------- drops
  local function run_drop(job, cb)
    configure(job.preset, job.mode)
    timer.Simple(0.4, function()                                  -- sv_gravity is a console command: it lands on the next frame
      local x, y = 0, 0
      local fz = floor_z(x, y)
      local ent = ents.Create("prop_physics")
      ent:SetModel(job.obj.model)
      ent:SetPos(Vector(x, y, fz + 400))
      ent:Spawn()
      ent:Activate()
      local po = ent:GetPhysicsObject()
      if not IsValid(po) then ent:Remove(); return cb({ error = "no physics for " .. job.obj.model }) end
      po:SetMass(job.obj.mass_kg)
      local mn, mx = ent:OBBMins(), ent:OBBMaxs()
      local h_hu = height_m / C.HU_M
      ent:SetPos(Vector(x, y, fz - mn.z + h_hu))                  -- the bottom of its box height_m above the floor
      ent:SetAngles(Angle(0, 0, 0))
      po:EnableMotion(false)
      W.track(ent)
      local rec_row = {
        preset = job.preset, mode = job.mode, name = job.obj.name, model = job.obj.model, mass_kg = po:GetMass(),
        obb_hu = { vec(mn), vec(mx) }, drop_hu = h_hu, floor_z = fz,
        tracked = W.tracked[ent] and { class = W.tracked[ent].class, cd = W.tracked[ent].cd, dims_m = W.tracked[ent].dims } or nil,
        engine_drag_enabled = po.IsDragEnabled and po:IsDragEnabled() or nil,
        gravity_hu = -physenv.GetGravity().z, sv_gravity = GetConVar("sv_gravity"):GetFloat(),
      }
      timer.Simple(1.0, function()
        if not IsValid(ent) then return cb({ error = "entity vanished" }) end
        po = ent:GetPhysicsObject()
        local z_start = ent:LocalToWorld(ent:OBBCenter()).z
        local samples, n = {}, 0
        po:EnableMotion(true)
        po:Wake()
        local hook_name = "WorldDrop" .. tostring(ent:EntIndex())
        hook.Add("Tick", hook_name, function()
          if not IsValid(ent) then hook.Remove("Tick", hook_name); return cb({ error = "entity vanished mid-fall" }) end
          n = n + 1
          local p = ent:GetPhysicsObject()
          local z = ent:LocalToWorld(ent:OBBCenter()).z
          local v = p:GetVelocity()
          samples[#samples + 1] = { n, z, v.z }
          local at_rest = #samples > 30 and math.abs(samples[#samples][2] - samples[#samples - 20][2]) < 0.05 and z < z_start - 0.5 * h_hu
          if n >= 330 or at_rest then
            hook.Remove("Tick", hook_name)
            rec_row.z_start = z_start
            rec_row.samples = samples
            rec_row.z_rest = z
            ent:Remove()
            cb(rec_row)
          end
        end)
      end)
    end)
  end

  -- ----------------------------------------------------------------------- layouts
  local function snapshot()
    local rows, asleep, moving = {}, 0, 0
    for _, id in ipairs(H.order) do
      local e = H.registry[id]
      if e and IsValid(e.ent) then
        local po = e.ent:GetPhysicsObject()
        local a = IsValid(po) and po:IsAsleep() or false
        if a then asleep = asleep + 1 end
        local sp = IsValid(po) and po:GetVelocity():Length() or -1
        if sp > 1 then moving = moving + 1 end
        rows[#rows + 1] = { id = id, kind = e.kind, pos = vec(e.ent:GetPos()), ang = { e.ent:GetAngles().p, e.ent:GetAngles().y, e.ent:GetAngles().r }, asleep = a, speed = sp }
      end
    end
    return { count = #rows, asleep = asleep, moving = moving, rows = rows }
  end

  local function run_layout(job, cb)
    if job.control then GetConVar("mentat_world_enabled"):SetBool(false); W.applied = {}; W.apply(false)
    else configure(job.preset, "real") end
    H.Handle(nil, "clear everything", { pos = { x = 0, y = 0, z = floor_z(0, 0) } })
    timer.Simple(2, function()
      local fz = floor_z(0, 0)
      local waited = 0
      H.Handle(nil, job.text, { pos = { x = 0, y = 0, z = fz }, yaw = 0, seed = 1 })
      local out = { text = job.text, preset = job.preset, control = job.control, floor_z = fz, snaps = {} }
      local settle = args.settle or { 8, 20, 32 }
      local i = 0
      local function nextsnap()
        i = i + 1
        if i > #settle then
          H.Handle(nil, "clear everything", { pos = { x = 0, y = 0, z = fz } })
          return cb(out)
        end
        timer.Simple(settle[i] - (settle[i - 1] or 0), function()
          local s = snapshot(); s.t = settle[i]
          out.snaps[#out.snaps + 1] = s
          nextsnap()
        end)
      end
      timer.Simple(0, nextsnap)
    end)
  end
  local ljobs = {}
  for _, p in ipairs(presets) do for _, t in ipairs(args.layouts or {}) do ljobs[#ljobs + 1] = { preset = p, text = t } end end
  for _, t in ipairs(args.layouts or {}) do ljobs[#ljobs + 1] = { preset = "earth", text = t, control = true } end

  -- ----------------------------------------------------------------------- go
  result.engine = {
    gravity_in_force_hu = -physenv.GetGravity().z, sv_gravity = GetConVar("sv_gravity"):GetFloat(), sv_gravity_default = GetConVar("sv_gravity"):GetDefault(),
    air_density = physenv.GetAirDensity(),
    tick = dt, map = game.GetMap(), hu_m = C.HU_M,
  }
  local ji, li = 0, 0
  local function next_layout()
    li = li + 1
    local job = ljobs[li]
    if not job then
      configure("earth", "real")
      return done(result)
    end
    run_layout(job, function(r) result.layouts[#result.layouts + 1] = r; next_layout() end)
  end
  local function next_drop()
    ji = ji + 1
    local job = jobs[ji]
    if not job then return next_layout() end
    run_drop(job, function(r) result.drops[#result.drops + 1] = r; timer.Simple(0.2, next_drop) end)
  end
  next_drop()
end
