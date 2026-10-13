-- water.lua -- water for the holodeck (server side).  REAL-WORLD PHYSICS ONLY.
--
-- What this does, and ONLY this:
--   * water bodies as ledger objects: "add a pool / pond / lake / sea" makes a `holodeck_water` basin through the holodeck's own pipeline
--     (placement, owner, limits, undo, remove, move all work as for any thing); the liquid is one the sigma-ground library has data for;
--   * Archimedes buoyancy F = rho_liquid g V_submerged and hydrodynamic drag F = 1/2 rho Cd A v^2 (+ Stokes' viscous term) on every awake
--     rigid body in the water, each physics tick, with g from the construct's world constants (world.lua) -- the Moon's water is lighter;
--   * players swim (water/swim.lua + swim_move.lua); splashes and ripples are sent to the clients, scaled by the entry's real energy.
-- NOT modelled: thermodynamics (no evaporation, no freezing), a particle fluid (Phase C), added mass, surface tension on bodies, currents,
-- wind waves.  Nothing of the game's content is used.  Numbers: mentat/water/data.lua (generated from the library); words: water/intent.lua.
--
--   mentat_water_enabled   1   0 = water bodies stay (as basins) but act on nothing
--   mentat_water_max_tick  48  objects processed per physics tick across all water (a cap, not physics)
if not SERVER then return end

Mentat = Mentat or {}
MentatWater = MentatWater or {}
local W = MentatWater
Mentat.Water = W

local C = include("mentat/water/constants.lua")
local Basin = include("mentat/water/basin.lua")
local Hydro = include("mentat/water/hydro.lua")
local Splash = include("mentat/water/splash.lua")
local Intent = include("mentat/water/intent.lua")
local Vol = include("mentat/water/volumes.lua")
local Swim = include("mentat/water/swim.lua")
local Move = include("mentat/water/swim_move.lua")
local Aero = include("mentat/world/aero.lua")
local WC = include("mentat/world/constants.lua")
W.constants, W.basin, W.hydro, W.splash, W.intent, W.volumes, W.swim = C, Basin, Hydro, Splash, Intent, Vol, Swim

local HU_M = WC.HU_M
local pi, sqrt, abs, min, max, floor = math.pi, math.sqrt, math.abs, math.min, math.max, math.floor

local cv_enabled = CreateConVar("mentat_water_enabled", "1", FCVAR_ARCHIVE, "1 = water bodies act on bodies and swimmers; 0 = they are only basins")
local cv_per_tick = CreateConVar("mentat_water_max_tick", tostring(C.LIMITS.max_bodies_per_tick.value), FCVAR_ARCHIVE, "objects processed per physics tick across all water (a cap, not physics)")

util.AddNetworkString("mentat_water_splash")

W.log = W.log or {}
W.recs = W.recs or setmetatable({}, { __mode = "k" })      -- ent -> { dims, class, cd, shape, wet, calm, ... }
W.counters = W.counters or { processed = 0, ticks = 0, us = 0, splashes = 0, woken = 0, slept = 0 }
W.ang_frame = W.ang_frame or "local"                        -- how AddAngleVelocity and GetAngleVelocity read their vector: MEASURED in the real engine (tools/water_live.lua): the body's own axes
W.cursor = W.cursor or 0
Move.cd = WC.CD.box.value

local function note(text)
  W.log[#W.log + 1] = text
  if #W.log > 100 then table.remove(W.log, 1) end
  print("[water] " .. text)
end

local function gravity() return Move.gravity() end
W.gravity = gravity

-- ------------------------------------------------------------------------------------------------ which objects
local function is_free_object(ent)
  if not IsValid(ent) then return false end
  if ent.IsHolodeckWater then return false end
  if ent.IsPlayer and ent:IsPlayer() then return false end
  if ent.IsNPC and ent:IsNPC() then return false end
  if ent.IsNextBot and ent:IsNextBot() then return false end
  if ent.IsVehicle and ent:IsVehicle() then return false end
  local cls = ent:GetClass()
  if cls == "prop_physics" or cls == "prop_physics_multiplayer" or cls == "holodeck_found" then return true end
  return ent.HolodeckId ~= nil
end
W.is_free_object = is_free_object

local function entry_of(ent)
  local H = Holodeck
  if ent.HolodeckId and H and H.registry then return H.registry[ent.HolodeckId] end
  return nil
end

--- What the object is called in words.
local function name_of(ent)
  local e = entry_of(ent)
  if e and e.kind then return e.kind end
  local m = ent.GetModel and ent:GetModel() or ""
  return (m:match("([^/]+)%.mdl$") or ent:GetClass())
end
W.name_of = name_of

--- The object's real size, shape class, drag coefficient and displaced-volume model, measured once from its bounding box.
local function describe(ent)
  local mn, mx = ent:OBBMins(), ent:OBBMaxs()
  local dims = { (mx.x - mn.x) * HU_M, (mx.y - mn.y) * HU_M, (mx.z - mn.z) * HU_M }
  local class = Aero.shape_class(name_of(ent), dims)
  local measured
  local e = entry_of(ent)
  if e and e.variant and e.variant.found and tonumber(e.variant.found.volume_m3) then measured = tonumber(e.variant.found.volume_m3) end
  if ent.WaterVolumeM3 and tonumber(ent.WaterVolumeM3) then measured = tonumber(ent.WaterVolumeM3) end
  local shape = Hydro.shape(class, dims, measured)
  return { dims = dims, class = class, cd = WC.CD[class].value, shape = shape,
           mass_basis = (e and e.variant and e.variant.mass_status) or "engine", wet = false, calm = 0, next_check = 0, bob_t = 0 }
end

local function rec_of(ent)
  local r = W.recs[ent]
  if not r then r = describe(ent); W.recs[ent] = r end
  return r
end
W.rec_of = rec_of

local function axes_of(ent)
  local a = ent:GetAngles()
  local f, r, u = a:Forward(), a:Right(), a:Up()
  return { f.x, f.y, f.z }, { -r.x, -r.y, -r.z }, { u.x, u.y, u.z }          -- local +y is the engine's -Right
end

--- The pose of an object in metres for Hydro.
local function pose_of(ent)
  local ax, ay, az = axes_of(ent)
  local c = ent:LocalToWorld(ent:OBBCenter())
  return { centre = { c.x * HU_M, c.y * HU_M, c.z * HU_M }, ax = ax, ay = ay, az = az }, c
end

-- ------------------------------------------------------------------------------------------------ angular velocity (the engine's frame)
local function get_w(ent, po)
  local v = po:GetAngleVelocity()
  if W.ang_frame == "local" then
    local ax, ay, az = axes_of(ent)
    return Vector(ax[1] * v.x + ay[1] * v.y + az[1] * v.z, ax[2] * v.x + ay[2] * v.y + az[2] * v.z, ax[3] * v.x + ay[3] * v.y + az[3] * v.z)
  end
  return v
end
local function add_w(ent, po, dw)
  if W.ang_frame == "local" then
    local ax, ay, az = axes_of(ent)
    dw = Vector(ax[1] * dw.x + ax[2] * dw.y + ax[3] * dw.z, ay[1] * dw.x + ay[2] * dw.y + ay[3] * dw.z, az[1] * dw.x + az[2] * dw.y + az[3] * dw.z)
  end
  po:AddAngleVelocity(dw)
end

-- ------------------------------------------------------------------------------------------------ messages to the clients
W.last_net = W.last_net or 0
W.net_budget = W.net_budget or { t = 0, n = 0 }
--- Tell every client about a splash or a ripple.  kind: 1 enter, 2 exit, 3 bob.  Rate-limited (a cap, not physics).
local function send_splash(body, kind, x, y, mass, speed)
  local b = W.net_budget
  local now = CurTime()
  if now - b.t > 1 then b.t, b.n = now, 0 end
  if b.n >= 24 then return false end
  b.n = b.n + 1
  net.Start("mentat_water_splash")
  net.WriteUInt(body.ent:EntIndex(), 16)
  net.WriteUInt(kind, 3)
  net.WriteFloat(x); net.WriteFloat(y); net.WriteFloat(body.surf)
  net.WriteFloat(mass); net.WriteFloat(speed)
  net.Broadcast()
  W.counters.splashes = W.counters.splashes + 1
  return true
end
W.send_splash = send_splash

-- ------------------------------------------------------------------------------------------------ the physics of one object
--- Net vertical force check for a sleeping body: does the water now lift it?  Returns the ratio buoyancy / weight.
local function lift_ratio(ent, rec, body, po)
  local pose = pose_of(ent)
  local vsub = Hydro.submerged(rec.shape, pose, body.surf * HU_M)
  local mass = po:GetMass()
  if mass <= 0 then return 0 end
  return Hydro.buoyancy(body.L.density_kg_m3, gravity(), vsub) / (mass * gravity()), vsub
end

--- One physics step of one awake object in a body of water.  Returns a table of what was done (for the tests), or nil when it was left alone.
function W.step_object(ent, rec, body, dt)
  local po = ent:GetPhysicsObject()
  if not IsValid(po) then return nil end
  if po.IsMotionEnabled and not po:IsMotionEnabled() then return nil end
  local mass = po:GetMass()
  if not (mass and mass > 0) then return nil end
  local g = gravity()
  local rho, eta = body.L.density_kg_m3, body.L.viscosity_pa_s
  local pose, centre = pose_of(ent)
  local vsub, cen, frac = Hydro.submerged(rec.shape, pose, body.surf * HU_M)
  local v0 = po:GetVelocity()                                               -- HU/s
  local vx, vy, vz = v0.x * HU_M, v0.y * HU_M, v0.z * HU_M                  -- m/s
  if vsub <= 0 then
    if rec.wet then
      rec.wet = false
      if vz > 0.2 then W.send_splash(body, 2, centre.x, centre.y, mass, vz) end
    end
    return nil
  end
  -- entering: the speed into the surface and the real energy of the body
  if not rec.wet then
    rec.wet = true
    rec.calm = 0
    if vz < -0.05 then W.send_splash(body, 1, centre.x, centre.y, mass, -vz) end
  end
  -- buoyancy: a velocity step.  The engine applies gravity itself right after this hook, so the drag below is taken on the velocity the
  -- body will have AFTER that step (v + (a_buoy - g) dt): then a body whose buoyancy equals its weight has exactly zero net acceleration
  -- and the drag cannot leak a spurious force into the equilibrium.
  local fb = Hydro.buoyancy(rho, g, vsub)
  local g_eng = g
  if physenv and physenv.GetGravity then
    local ok, gv = pcall(physenv.GetGravity)
    if ok and gv then g_eng = -gv.z * HU_M end
  end
  local gdt = g_eng * dt
  local dvz = fb / mass * dt
  local v1x, v1y, v1z = vx, vy, vz + dvz - gdt
  -- drag on the immersed part of the body's projected area, quadratic + Stokes
  local sp = sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
  local v2x, v2y, v2z = v1x, v1y, v1z
  local area_eff = 0
  if sp > 1e-6 then
    local nx, ny, nz = v1x / sp, v1y / sp, v1z / sp
    local ex, ey, ez = pose.ax, pose.ay, pose.az
    local nl = { nx * ex[1] + ny * ex[2] + nz * ex[3], nx * ey[1] + ny * ey[2] + nz * ey[3], nx * ez[1] + ny * ez[2] + nz * ez[3] }
    area_eff = Aero.area(rec.class, rec.dims, nl) * frac
    local k2, k1 = Hydro.drag_k(rho, rec.cd, area_eff, mass, eta)
    local sp2 = Hydro.speed_after(sp, k2, k1, dt)
    local s = sp2 / sp
    v2x, v2y, v2z = v1x * s, v1y * s, v1z * s
  end
  -- a body piercing the surface radiates waves as it moves up and down: heave damping (estimated, see data.lua "radiation")
  if frac > 0 and frac < 1 then
    local d = 0.01
    local vp = Hydro.submerged(rec.shape, pose, body.surf * HU_M + d)
    local vm = Hydro.submerged(rec.shape, pose, body.surf * HU_M - d)
    local aw = (vp - vm) / (2 * d)                                         -- dV/dz = the waterplane area
    if aw > 1e-6 then
      local c = Hydro.heave_radiation_c(rho, g, aw, vsub / aw, mass)
      v2z = v2z / (1 + c * dt)
    end
  end
  v2z = v2z + gdt                                                          -- back to the velocity the engine is handed
  local dv = Vector((v2x - vx) / HU_M, (v2y - vy) / HU_M, (v2z - vz) / HU_M)
  if dv:LengthSqr() > 1e-12 then po:AddVelocity(dv) end
  -- the torque of the buoyant force about the centre of mass (the centroid of the displaced volume is the centre of buoyancy)
  local c = pose.centre
  local rx, ry, rz = cen[1] - c[1], cen[2] - c[2], cen[3] - c[3]
  local tx, ty, tz = ry * fb - rz * 0, rz * 0 - rx * fb, 0                  -- r x (0, 0, fb)
  local dims = rec.shape.kind == "box" and rec.shape.dims or rec.dims
  local I1, I2, I3 = Hydro.box_inertia(mass, dims)
  local ax, ay, az = pose.ax, pose.ay, pose.az
  local t1, t2, t3 = tx * ax[1] + ty * ax[2] + tz * ax[3], tx * ay[1] + ty * ay[2] + tz * ay[3], tx * az[1] + ty * az[2] + tz * az[3]
  local d1, d2, d3 = t1 / I1 * dt, t2 / I2 * dt, t3 / I3 * dt                -- rad/s along the body's axes
  local wx = d1 * ax[1] + d2 * ay[1] + d3 * az[1]
  local wy = d1 * ax[2] + d2 * ay[2] + d3 * az[2]
  local wz = d1 * ax[3] + d2 * ay[3] + d3 * az[3]
  -- angular drag: tau = 1/2 rho Cd A (L/4)^3 w^2, omega' = omega / (1 + c |omega| dt)
  local w = get_w(ent, po)
  local wl = w:Length() * pi / 180                                           -- rad/s
  local L = (rec.dims[1] + rec.dims[2] + rec.dims[3]) / 3
  local cdrag = 0
  if wl > 1e-6 then
    local A = Aero.area(rec.class, rec.dims, { 0.577, 0.577, 0.577 }) * frac
    cdrag = Hydro.ang_drag_c(rho, rec.cd, A, L, (I1 + I2 + I3) / 3)
    local scale = 1 / (1 + cdrag * wl * dt)
    local dd = w * (scale - 1)
    wx, wy, wz = wx + dd.x * pi / 180, wy + dd.y * pi / 180, wz + dd.z * pi / 180
  end
  local DEG = 180 / pi
  if abs(wx) + abs(wy) + abs(wz) > 1e-9 then add_w(ent, po, Vector(wx * DEG, wy * DEG, wz * DEG)) end
  -- bobbing: a body moving through the surface disturbs it
  if frac > 0 and frac < 1 and abs(v2z) > 0.08 then
    local now = CurTime()
    if now - (rec.bob_t or 0) > 0.6 then
      rec.bob_t = now
      W.send_splash(body, 3, centre.x, centre.y, mass, abs(v2z))
    end
  end
  -- settled: floating at its draft and not moving, or resting on the bottom -> sleep (the engine sleeps what rests; a floater needs this)
  local speed2 = sqrt(vx * vx + vy * vy + vz * vz)               -- the speed the body had when this tick began (the engine adds its own gravity step after the hook)
  local weight = mass * g
  if weight > 0 and speed2 < 0.03 and abs(fb - weight) / weight < 0.015 and frac < 1 then
    rec.calm = (rec.calm or 0) + dt
    if rec.calm > 1.2 and not (po.IsAsleep and po:IsAsleep()) then
      po:SetVelocity(Vector(0, 0, 0))
      po:Sleep()
      W.counters.slept = W.counters.slept + 1
      rec.calm = 0
    end
  else
    rec.calm = 0
  end
  return { buoyancy_n = fb, v_sub = vsub, frac = frac, area = area_eff, dv = dv, drag_c = cdrag }
end

-- ------------------------------------------------------------------------------------------------ the air's drag stays out of the water
--- world.lua applies the AIR's drag to every free object each tick; a body under water feels the water's drag instead (and never both).
function W.hook_world()
  local WW = MentatWorld
  if not (WW and WW.drag_step) or WW.water_wrapped then return false end
  local inner = WW.drag_step
  WW.drag_step = function(ent, rec, dt)
    local r = W.recs[ent]
    if r and r.wet and cv_enabled:GetBool() then return nil end
    return inner(ent, rec, dt)
  end
  WW.water_wrapped = true
  return true
end

-- ------------------------------------------------------------------------------------------------ the tick
local function scan_objects()
  local out = {}
  for _, cls in ipairs({ "prop_physics", "prop_physics_multiplayer", "holodeck_found" }) do
    for _, e in ipairs(ents.FindByClass(cls)) do if is_free_object(e) then out[#out + 1] = e end end
  end
  local H = Holodeck
  if H and H.registry then
    local seen = {}
    for _, e in ipairs(out) do seen[e] = true end
    for _, entry in pairs(H.registry) do
      local e = entry.ent
      if e and not seen[e] and is_free_object(e) then seen[e] = true; out[#out + 1] = e end
    end
  end
  return out
end

W.objs, W.objs_t = W.objs or {}, W.objs_t or -1
local function objects()
  local now = CurTime()
  if now - W.objs_t > 0.5 then W.objs, W.objs_t = scan_objects(), now end
  return W.objs
end
W.objects = objects

--- Which body's water the object overlaps (bounding sphere against the footprint and the water's height range), or nil.
local function body_for(bodies, pos, radius_hu)
  for _, b in ipairs(bodies) do
    if b.surf > b.floor and pos.z + radius_hu >= b.floor and pos.z - radius_hu <= b.surf and Vol.inside_xy(b, pos.x, pos.y, radius_hu) then return b end
  end
  return nil
end

function W.on_tick()
  if not cv_enabled:GetBool() then return end
  if not MentatWorld or not MentatWorld.water_wrapped then W.hook_world() end
  local bodies = Vol.list()
  W.bodies_cached = bodies
  if #bodies == 0 then return end
  local t0 = SysTime()
  local dt = engine.TickInterval()
  local list = objects()
  local n = #list
  if n == 0 then return end
  local budget = max(1, cv_per_tick:GetInt())
  local now = CurTime()
  local done = 0
  local start = W.cursor
  for k = 1, min(n, budget) do
    local i = (start + k - 1) % n + 1
    local ent = list[i]
    done = done + 1
    if IsValid(ent) then
      local po = ent:GetPhysicsObject()
      if IsValid(po) then
        local rec = rec_of(ent)
        local radius = 0.5 * sqrt(rec.dims[1] ^ 2 + rec.dims[2] ^ 2 + rec.dims[3] ^ 2) / HU_M
        local pos = ent:LocalToWorld(ent:OBBCenter())
        local body = body_for(bodies, pos, radius)
        if not body then
          if rec.wet then rec.wet = false end
        elseif po.IsAsleep and po:IsAsleep() then
          if now >= rec.next_check then                                     -- a sleeping body is watched, not pushed: does the water lift it now?
            rec.next_check = now + 0.3
            local ratio = lift_ratio(ent, rec, body, po)
            if ratio and ratio > 1.03 then po:Wake(); W.counters.woken = W.counters.woken + 1 end
          end
        else
          W.step_object(ent, rec, body, dt)
        end
      end
    end
  end
  W.cursor = (start + done) % n
  W.counters.processed = W.counters.processed + done
  W.counters.ticks = W.counters.ticks + 1
  W.counters.us = W.counters.us + (SysTime() - t0) * 1e6
end

-- ------------------------------------------------------------------------------------------------ the ledger: water bodies as holodeck objects
local function Hd() return Holodeck end

--- All water entries of the ledger: { entry, ent, water (the variant's numbers), volume }.
function W.entries()
  local H = Hd()
  local out = {}
  if not (H and H.registry) then return out end
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e and e.variant and e.variant.water then out[#out + 1] = { entry = e, ent = e.ent, water = e.variant.water, volume = C.volume(e.variant.water.shape, e.variant.water.l_m, e.variant.water.w_m, e.variant.water.depth_m) } end
  end
  return out
end

local LETTERS = "abcdefghijklmnopqrstuvwxyz"
W.serial = W.serial or 0
local function new_noun(base)
  W.serial = W.serial + 1
  local n = W.serial
  local a, b = (n - 1) % 26 + 1, floor((n - 1) / 26) % 26 + 1
  return base .. "x" .. LETTERS:sub(b, b) .. LETTERS:sub(a, a)
end

--- A kind record for the holodeck's library: one variant, a `holodeck_water` of this basin.
function W.make_kind(base_noun, shape, l, w, depth, liquid_key)
  local b = Basin.new(shape, l, w, depth)
  local id = new_noun(base_noun)
  local fp = Basin.footprint(b)
  local mins = { b.mins[1], b.mins[2], b.mins[3] }
  local maxs = { b.maxs[1], b.maxs[2], b.maxs[3] }
  -- the drop zone for "put a ball in the pool": a rectangle inside the water, half a metre above the rim
  local inset = shape == "round" and 0.55 or 0.8
  local top = { x0 = -b.hx * inset, x1 = b.hx * inset, y0 = -b.hy * inset, y1 = b.hy * inset, z = b.top_z + 0.5 / HU_M }
  local label = base_noun
  local kind = {
    kind = id, names = { id }, approx = {}, category = "found", seat = false, surface = true, wall = "free", found = true,
    variants = { {
      model = "holodeck/water/" .. id, front = 0, front_source = "unknown", symmetric = true, top = top,
      w = fp.w, d = fp.d, h = fp.h, mins = mins, maxs = maxs, cx = 0, cy = 0,
      mass = 0, mass_status = "estimated", material = "default", skins = 1, entity = "holodeck_water",
      found = { card = "water", name = label, source = "water", licence = "", basis = "constructed" },
      water = { shape = b.shape, l_m = l, w_m = w, depth_m = depth, liquid = liquid_key, fill = 1, base = base_noun },
    } },
  }
  return kind, id
end

local function say(ply, text, speak)
  local H = Hd()
  if H and H.Say then H.Say(ply, text, nil, { speak = speak ~= false })
  elseif IsValid(ply) and ply.ChatPrint then ply:ChatPrint("[Skippy] " .. text) end
end
local function tell1(ply, text)
  local H = Hd()
  if H and H.Tell then H.Tell(ply, text)
  elseif IsValid(ply) and ply.ChatPrint then ply:ChatPrint("[Skippy] " .. text) end
end
--- The holodeck's lines are at most 240 characters: longer text goes out in pieces, cut at spaces.
local function tell(ply, text)
  text = tostring(text)
  while #text > 230 do
    local cut = text:sub(1, 230):match("^(.*)%s") or text:sub(1, 230)
    tell1(ply, cut)
    text = text:sub(#cut + 1):gsub("^%s+", "")
  end
  if text ~= "" then tell1(ply, text) end
end
local function who(ply) return (IsValid(ply) and ply.IsPlayer and ply:IsPlayer()) and (ply:Nick() .. " " .. tostring(ply:SteamID64())) or "console" end
local function key_of(ply) return (IsValid(ply) and ply.IsPlayer and ply:IsPlayer()) and ply:SteamID64() or "console" end

local function manages(ply)
  if not (IsValid(ply) and ply.IsPlayer and ply:IsPlayer()) then return true end
  if game.SinglePlayer() then return true end
  if ply.IsListenServerHost and ply:IsListenServerHost() then return true end
  return (ply.IsAdmin and ply:IsAdmin()) and true or false
end

--- The water body a sentence means: by noun ("the pool") among the entries, else the one nearest the player, else the player's latest.
function W.target(ply, noun)
  local list = W.entries()
  if #list == 0 then return nil end
  local best, bd
  local pos = (IsValid(ply) and ply.GetPos) and ply:GetPos() or nil
  for _, it in ipairs(list) do
    local ok = true
    if noun and it.water.base then
      local base = it.water.base
      if noun == "ocean" then noun = "sea" end
      ok = (base == noun)
    end
    if ok and it.entry then
      local d = 0
      if pos and it.ent and IsValid(it.ent) then d = it.ent:GetPos():DistToSqr(pos) end
      if not bd or d < bd then best, bd = it, d end
    end
  end
  if not best and noun then return W.target(ply, nil) end
  return best
end

local function can_touch(ply, it)
  if manages(ply) then return true end
  return it.entry.owner == key_of(ply)
end

-- ------------------------------------------------------------------------------------------------ reading back
local function fmt_dims(w)
  if w.shape == "round" then return string.format("%s m across, %s m deep", C.g3(w.l_m), C.g3(w.depth_m)) end
  return string.format("%s m by %s m, %s m deep", C.g3(w.l_m), C.g3(w.w_m), C.g3(w.depth_m))
end

local function liquid_line(key)
  local L = C.LIQUIDS[key]
  return string.format("%s: %s kg/m3, viscosity %s Pa s, at %.0f K, from the sigma-ground library.", L.label, C.g3(L.density_kg_m3), C.g3(L.viscosity_pa_s), L.T_K)
end
local function liquid_source(key)
  local L = C.LIQUIDS[key]
  return "Source: " .. L.source .. " [" .. L.status .. "]"
end

--- The honest caveat about what buoyancy rests on.
local CAVEAT = "A thing floats or sinks by its mass over its volume: the mass is the card's (often only guessed) and the volume is its bounding shape's, so this is only as good as those. Phase B improves them."

function W.describe_body(it)
  local w = it.water
  local g = gravity()
  return string.format("A %s of %s, %s. %s Gravity here is %s m/s2, so a floating thing sinks in to its density over the water's.",
    w.base or "pool", w.liquid, fmt_dims(w), liquid_line(w.liquid), C.g3(g))
end

--- What the objects near / in the water do.  Returns lines (strings) and a spoken summary.
function W.floats_report(ply, item)
  local bodies = Vol.list()
  local lines, floating, sinking = {}, 0, 0
  if #bodies == 0 then return { "There is no water here." }, "There is no water here." end
  local pos = (IsValid(ply) and ply.GetPos) and ply:GetPos() or Vector(0, 0, 0)
  for _, ent in ipairs(objects()) do
    if IsValid(ent) then
      local po = ent:GetPhysicsObject()
      if IsValid(po) and po:GetMass() > 0 then
        local rec = rec_of(ent)
        local radius = 0.5 * sqrt(rec.dims[1] ^ 2 + rec.dims[2] ^ 2 + rec.dims[3] ^ 2) / HU_M
        local c = ent:LocalToWorld(ent:OBBCenter())
        local body = body_for(bodies, c, radius) or nil
        local near = false
        for _, b in ipairs(bodies) do if Vol.inside_xy(b, c.x, c.y, 2000) then near = true; break end end
        local name = name_of(ent)
        if (body or near) and (not item or name:find(item, 1, true)) then
          local b = body or bodies[1]
          local rho_b = po:GetMass() / rec.shape.volume
          local fl, draft = C.floats(rho_b, b.liquid)
          if fl then floating = floating + 1 else sinking = sinking + 1 end
          lines[#lines + 1] = string.format("%s: %s kg over %s m3 = %s kg/m3 against %s kg/m3 of %s: %s (mass %s, volume %s).", name, C.g3(po:GetMass()), C.g3(rec.shape.volume),
            C.g3(rho_b), C.g3(b.L.density_kg_m3), b.liquid, fl and ("floats, " .. C.g3(draft * 100) .. " percent under") or "sinks", rec.mass_basis, rec.shape.basis)
        end
      end
    end
  end
  if #lines == 0 then
    local msg = item and ("I do not see a " .. item .. " near the water.") or "Nothing loose is near the water."
    return { msg }, msg
  end
  local summary = string.format("Near the water, %d float and %d sink, each by its density against the liquid's.", floating, sinking)
  lines[#lines + 1] = CAVEAT
  return lines, summary
end

-- ------------------------------------------------------------------------------------------------ the words
local function register_kind(kind)
  local H = Hd()
  local id, why = H.AddFoundKind(kind)
  return id, why
end

local function reply(ply, text, res)
  say(ply, text)
  return res or { action = "water", say = text }
end

local function liquid_or_refuse(ply, word)
  if not word then return "water" end
  local key, bad = C.find_liquid(word)
  if key then return key end
  say(ply, C.refusal(word))
  note(who(ply) .. " asked for " .. tostring(word) .. ": declined")
  return nil
end

W.pending_notes = W.pending_notes or {}

local function do_add(ply, text, it, opts, inner)
  local liq = liquid_or_refuse(ply, it.liquid)
  if not liq then return { action = "water", refused = true, say = "refused" } end
  if (it.count or 1) > 1 then
    return reply(ply, "One body of water at a time. Say it again for the next one.")
  end
  local base = it.noun
  local d = C.dims_for(base, { l = it.l, w = it.w, depth = it.depth })
  -- "deep", "huge": multipliers on the usual numbers (only when the player gave no number)
  local l, w, depth, notes = d.l, d.w, d.depth, d.notes
  if it.factor_size and not (it.l or it.w) then l, w = l * it.factor_size, w * it.factor_size end
  if it.factor_depth and not it.depth then depth = depth * it.factor_depth end
  local shape = it.shape or d.shape
  if shape == "round" and not (it.w) and (it.l == nil) then w = l end
  local cl, cw, cd2
  cl, cw, cd2, notes = C.clamp_dims(l, w, depth)
  l, w, depth = cl, cw, cd2
  local vol = C.volume(shape, l, w, depth)
  local ok, why = C.room_for(W.entries(), vol)
  if not ok then return reply(ply, why) end
  local kind, id = W.make_kind(base == "ocean" and "sea" or base, shape, l, w, depth, liq)
  local kid, kwhy = register_kind(kind)
  if not kid then return reply(ply, "I could not make that: " .. tostring(kwhy) .. ".") end
  local rewritten = Intent.rewrite_add(text, it, kid)
  W.last_made = { key = key_of(ply), id = kid, t = CurTime() }
  note(who(ply) .. " said \"" .. tostring(text) .. "\": " .. base .. " " .. shape .. " " .. string.format("%.3g x %.3g x %.3g", l, w, depth) .. " m of " .. liq)
  local res = inner(ply, rewritten, opts)
  local extra = {}
  for _, a in ipairs(it.assumed or {}) do extra[#extra + 1] = a end
  for _, n in ipairs(notes or {}) do extra[#extra + 1] = "I limited it to " .. n end
  local made = string.format("%s of %s, %s. %s", base, liq, fmt_dims({ shape = shape, l_m = l, w_m = w, depth_m = depth }), "Real buoyancy and drag act on whatever goes in.")
  W.after_add = { text = made, extra = extra, liq = liq, ply = ply, t = CurTime() }
  if res and res.placements and #res.placements > 0 then
    tell(ply, made)
    for _, e in ipairs(extra) do tell(ply, e) end
    tell(ply, liquid_line(liq) .. " " .. CAVEAT)
    tell(ply, liquid_source(liq))
  end
  return res
end

local function set_variant_water(entry, patch)
  local w = entry.variant.water
  for k, v in pairs(patch) do w[k] = v end
end

--- Resize / refill / change liquid of an existing body.  `how`: { depth_factor=, size_factor=, depth_m=, ... } | { fill = } | { liquid = }.
local function modify(ply, it, how)
  local tg = W.target(ply, it.target)
  if not tg then return reply(ply, "There is no water to change. Ask for a pool, a pond, a lake or a sea first.") end
  if not can_touch(ply, tg) then return reply(ply, "That water is someone else's.") end
  local entry, ent, w = tg.entry, tg.ent, tg.water
  if not IsValid(ent) then return reply(ply, "That water is still arriving. Try again in a moment.") end
  if how.fill ~= nil then
    local was = ent:GetFill()
    ent:SetLevel(how.fill)
    set_variant_water(entry, { fill = how.fill })
    if how.fill == 0 then
      if was == 0 then return reply(ply, "It is already empty.") end
      note(who(ply) .. " drained a " .. (w.base or "pool"))
      return reply(ply, "Drained. The basin stands empty; the things in it settle onto the floor.")
    end
    if was >= 1 then return reply(ply, "It is already full.") end
    note(who(ply) .. " refilled a " .. (w.base or "pool"))
    return reply(ply, "Filled to the brim with " .. w.liquid .. ".")
  end
  if how.liquid then
    local key = liquid_or_refuse(ply, how.liquid)
    if not key then return { action = "water", refused = true, say = "refused" } end
    if key == w.liquid then return reply(ply, "It is already " .. key .. ".") end
    set_variant_water(entry, { liquid = key })
    ent:SetLiquidKey(key)
    ent:SetLevel(max(ent:GetFill(), 1))
    note(who(ply) .. " changed a " .. (w.base or "pool") .. " to " .. key)
    return reply(ply, "It is " .. key .. " now. " .. liquid_line(key))
  end
  -- resize: a new basin, so a new kind (the old kind stays a stand-in only for whoever still stands as it)
  local b0 = ent:GetBasin()
  local nb = Basin.resized(b0, how)
  local ok, why = C.room_for(W.entries(), nb.volume_m3, nil)
  if not ok then
    -- the volume cap counts this body at its old size
    local others = {}
    for _, o in ipairs(W.entries()) do if o.entry ~= entry then others[#others + 1] = o end end
    ok, why = C.room_for(others, nb.volume_m3)
    if not ok then return reply(ply, why) end
  end
  local kind, id = W.make_kind(w.base or "pool", nb.shape, nb.l_m, nb.w_m, nb.depth_m, w.liquid)
  kind.variants[1].water.fill = w.fill
  local H = Hd()
  -- the entry keeps its place: only its kind, model and size change
  entry.kind = id
  entry.model = kind.variants[1].model
  local kid, kwhy = H.AddFoundKind(kind)
  if not kid then return reply(ply, "I could not resize it: " .. tostring(kwhy) .. ".") end
  local v = H.registry[entry.id] and H.registry[entry.id].variant or kind.variants[1]
  entry.variant = v
  entry.w, entry.d, entry.h = v.w, v.d, v.h
  ent:Resize(nb.shape, nb.l_m, nb.w_m, nb.depth_m)
  local sp = ent:GetPhysicsObject()
  -- bodies in the water are woken so they notice the new depth
  for _, o in ipairs(objects()) do
    local p = IsValid(o) and o:GetPhysicsObject()
    if p and IsValid(p) and p.Wake then p:Wake() end
  end
  note(who(ply) .. " resized a " .. (w.base or "pool") .. " to " .. string.format("%.3g x %.3g x %.3g", nb.l_m, nb.w_m, nb.depth_m) .. " m")
  local txt = string.format("It is %s now. I did not check it against what stands next to it; the walls grow outward from the middle.", fmt_dims({ shape = nb.shape, l_m = nb.l_m, w_m = nb.w_m, depth_m = nb.depth_m }))
  return reply(ply, txt)
end

local function do_query(ply, it)
  local tg = W.target(ply, nil)
  if it.what == "density" or it.what == "liquid" then
    if not tg then return reply(ply, "There is no water here.") end
    local w = tg.water
    tell(ply, liquid_source(w.liquid))
    return reply(ply, liquid_line(w.liquid) .. (it.what == "density" and " A body floats in it when its own density is lower." or ""))
  end
  if it.what == "depth" then
    if not tg then return reply(ply, "There is no water here.") end
    return reply(ply, "The " .. (tg.water.base or "pool") .. " is " .. fmt_dims(tg.water) .. ".")
  end
  local lines, summary = W.floats_report(ply, it.item)
  for _, l in ipairs(lines) do tell(ply, l) end
  return reply(ply, summary)
end

--- Try a sentence as water.  `inner` is the next handler (the world words, then the object parser).  Returns nil when it is not water.
function W.Handle(ply, text, opts, inner)
  local ok, it = pcall(Intent.parse, text, C)
  if not ok then ErrorNoHalt("[water] intent: " .. tostring(it) .. "\n"); return nil end
  if not it then return nil end
  -- "make it bigger", "drain it", "make it mercury" with no water word: only when this player was just talking about water
  W.ctx = W.ctx or {}
  local who_key = key_of(ply)
  if (it.kind == "resize" or it.kind == "liquid" or it.kind == "level" or it.kind == "remove") and not it.explicit then
    if CurTime() - (W.ctx[who_key] or -1e9) > 120 or #W.entries() == 0 then return nil end      -- the object parser gets the sentence
  end
  W.ctx[who_key] = CurTime()
  if it.kind == "add" then
    return do_add(ply, text, it, opts, inner)
  elseif it.kind == "drop" then
    local tg = W.target(ply, it.target)
    if not tg then return reply(ply, "There is no water to put that in. Ask for a pool, a pond, a lake or a sea first.") end
    local rewritten = Intent.rewrite_drop(text, it, tg.entry.kind)
    local res = inner(ply, rewritten, opts)
    return res
  elseif it.kind == "remove" then
    local tg = W.target(ply, it.target)
    if not tg then return nil end                                        -- "remove the water" with no water: the object parser says what it says
    if not can_touch(ply, tg) then return reply(ply, "That water is someone else's.") end
    return inner(ply, "remove the " .. tg.entry.kind, opts)
  elseif it.kind == "level" then
    return modify(ply, it, { fill = it.level })
  elseif it.kind == "liquid" then
    return modify(ply, it, { liquid = it.liquid })
  elseif it.kind == "resize" then
    local how = {}
    if it.absolute then how.depth_m = it.absolute
    elseif it.dim == "depth" then how.depth_factor = it.factor
    elseif it.dim == "width" then how.width_factor = it.factor
    elseif it.dim == "length" then how.length_factor = it.factor
    else how.size_factor = it.factor end
    return modify(ply, it, how)
  elseif it.kind == "query" then
    return do_query(ply, it)
  end
  return nil
end

local function install()
  local H = Hd()
  if not (H and H.Handle) or H.Handle == W.wrapped then return false end
  H.wrappers = H.wrappers or {}
  if H.wrappers.water then return false end                 -- (other modules wrap Handle too: never wrap twice)
  H.wrappers.water = true
  local inner = H.Handle
  local function handle(ply, text, opts)
    if type(text) == "string" and not (opts and opts.found_done) then
      local ok, res = pcall(W.Handle, ply, text, opts, inner)
      if ok and res then
        if opts and opts.done then opts.done(res) end
        return res
      elseif not ok then
        ErrorNoHalt("[water] " .. tostring(res) .. "\n")
      end
    end
    return inner(ply, text, opts)
  end
  H.Handle = handle
  W.wrapped, W.inner = handle, inner
  return true
end
W.install = install

concommand.Add("mentat_water", function(ply, _, args)
  local text = table.concat(args, " ")
  if text == "" then
    for _, l in ipairs(W.status_lines()) do tell(ply, l) end
    return
  end
  local H = Hd()
  if H and H.Handle then H.Handle(ply, text) end
end, nil, "Ask about or change the construct's water in plain words")

function W.status_lines()
  local lines = {}
  local es = W.entries()
  lines[#lines + 1] = #es .. " water bodies, enabled=" .. tostring(cv_enabled:GetBool()) .. ", g=" .. C.g3(gravity()) .. " m/s2"
  for _, it in ipairs(es) do lines[#lines + 1] = W.describe_body(it) end
  local c = W.counters
  lines[#lines + 1] = string.format("ticks %d, objects processed %d, mean %.1f us per tick, splashes sent %d, woken %d, slept %d", c.ticks, c.processed,
    c.ticks > 0 and c.us / c.ticks or 0, c.splashes, c.woken, c.slept)
  return lines
end

concommand.Add("mentat_water_status", function(ply)
  for _, l in ipairs(W.status_lines()) do print("[water] " .. l) end
end, nil, "What the water is doing")

-- ------------------------------------------------------------------------------------------------ hooks
hook.Add("Tick", "MentatWater.Tick", W.on_tick)
Move.install()
install()
timer.Simple(0, install)                         -- once more after every addon file has loaded, in case Holodeck came later
