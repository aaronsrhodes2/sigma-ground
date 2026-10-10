-- server.lua -- scenario playback: load, play/pause/scrub/rate, release-to-Havok, console UI.
--
-- Doctrine: the scenario is FROZEN sigma-ground output. REPLAY only displays it (poses are
-- set from the recorded track, nothing is simulated). RELEASE hands a body to Havok at its
-- current pose and velocity; from then on Havok, not sigma-ground, is moving it.
local U, I = Mentat.units, Mentat.interp

local st = {
  scn = nil, name = nil, ents = {},
  t = 0, t0 = 0, tend = 0, rate = 1,
  playing = false, ended = false,
  origin = Vector(0, 0, 0), scale = U.HU_PER_M,
  saved_gravity = nil,
}
Mentat.state = st

-- ------------------------------------------------------------------ HUD publishing
local function released_count()
  local n = 0
  for _, e in ipairs(st.ents) do
    if IsValid(e) and e.MentatReleased then n = n + 1 end
  end
  return n
end

local function state_string()
  if not st.scn then return "EMPTY" end
  if st.playing then return "PLAYING" end
  if st.ended then return "ENDED" end
  if st.t > st.t0 then return "PAUSED" end
  return "READY"
end

local function publish()
  SetGlobal2String("mentat_name", st.name or "")
  SetGlobal2String("mentat_state", state_string())
  SetGlobal2Float("mentat_t", st.t or 0)
  SetGlobal2Float("mentat_t0", st.t0 or 0)
  SetGlobal2Float("mentat_tend", st.tend or 0)
  SetGlobal2Float("mentat_rate", st.rate or 1)
  SetGlobal2Int("mentat_released", released_count())
  SetGlobal2Int("mentat_bodies", #st.ents)
end

-- ------------------------------------------------------------------ pose application
--- World position (Source HU) of a sigma-frame pose.
function Mentat.WorldPos(pos)
  local scn = st.scn
  local x, y, z = U.world_pos(scn.up, pos, st.scale, scn.floor_m)
  return Vector(st.origin.x + x, st.origin.y + y, st.origin.z + z)
end

--- Inverse of WorldPos: Source world Vector -> sigma-frame {x,y,z} metres.
function Mentat.SigmaPos(v)
  local scn = st.scn
  local x, y, z = U.sigma_pos(scn.up, v.x - st.origin.x, v.y - st.origin.y, v.z - st.origin.z,
                              st.scale, scn.floor_m)
  return { x, y, z }
end

--- Place every non-released body at its recorded pose for sim time t.
function Mentat.Apply(t)
  local scn = st.scn
  if not scn then return end
  for i, ent in ipairs(st.ents) do
    if IsValid(ent) and not ent.MentatReleased then
      local pose = I.sample_body(scn, scn.bodies[i], t)
      ent:SetPos(Mentat.WorldPos(pose.pos))
      local p, y, r = U.angles_from_quat(scn.up, pose.quat)
      ent:SetAngles(Angle(p, y, r))
      ent.MentatPoseT = t                  -- the sim time this pose belongs to (autotest uses it)
    end
  end
end

-- ------------------------------------------------------------------ lifecycle
function Mentat.Clear()
  for _, e in ipairs(st.ents) do
    if IsValid(e) then e:Remove() end
  end
  st.ents = {}
  if st.saved_gravity then
    physenv.SetGravity(st.saved_gravity)
    st.saved_gravity = nil
  end
  st.scn, st.name = nil, nil
  st.playing, st.ended = false, false
  st.t, st.t0, st.tend = 0, 0, 0
  publish()
end

--- Load a scenario with its stage floor at `anchor` (Source world Vector; anchor.z is the floor).
function Mentat.Load(name, anchor)
  Mentat.Clear()
  local scn, err = Mentat.GetScenario(name)
  if not scn then return false, err end
  if (scn.stage_scale or 1) ~= 1 then return false, "stage_scale ~= 1 is not supported in Phase 0" end

  st.scn, st.name = scn, name
  st.scale = Mentat.Scale(scn)
  st.origin = Vector(anchor.x, anchor.y, anchor.z)
  st.t0, st.tend = scn.times[1], scn.times[#scn.times]
  st.t, st.rate = st.t0, scn.suggested_rate or 1
  st.playing, st.ended = false, false

  -- Gravity is a global of the physics environment: set it for released bodies, restore on Clear.
  st.saved_gravity = physenv.GetGravity()
  physenv.SetGravity(Vector(0, 0, -(scn.gravity_m_s2 or 9.80665) * U.HU_PER_M))

  for i = 1, #scn.bodies do
    local ent = ents.Create("mentat_body")
    if not IsValid(ent) then
      Mentat.Clear()
      return false, "ents.Create('mentat_body') failed: is the addon mounted?"
    end
    ent:SetScenario(name)
    ent:SetBodyIndex(i)
    ent:SetPos(st.origin)
    ent:Spawn()
    ent:Activate()
    if not IsValid(ent) then
      Mentat.Clear()
      return false, "body " .. i .. " failed to initialise (see console)"
    end
    st.ents[i] = ent
  end
  Mentat.Apply(st.t)
  publish()
  return true
end

function Mentat.Play()
  if not st.scn then return false, "no scenario loaded" end
  if st.ended or st.t >= st.tend then Mentat.Reset() end
  st.playing, st.ended = true, false
  publish()
  return true
end

function Mentat.Pause()
  st.playing = false
  publish()
end

function Mentat.SetRate(r)
  r = tonumber(r)
  if not r or r <= 0 or r > 1000 then return false, "rate must be in (0, 1000]" end
  st.rate = r
  publish()
  return true
end

function Mentat.Scrub(t)
  if not st.scn then return false, "no scenario loaded" end
  t = tonumber(t)
  if not t then return false, "t must be a number" end
  st.t = math.Clamp(t, st.t0, st.tend)
  st.ended = st.t >= st.tend
  Mentat.Apply(st.t)
  publish()
  return true
end

--- Back to frame 0 with every body kinematic again.
function Mentat.Reset()
  if not st.scn then return false, "no scenario loaded" end
  st.playing, st.ended = false, false
  st.t = st.t0
  for _, e in ipairs(st.ents) do
    if IsValid(e) then
      e.MentatReleased = false
      e:SetReleased(false)
      local phys = e:GetPhysicsObject()
      if IsValid(phys) then
        phys:EnableMotion(false)
        phys:SetVelocity(vector_origin)
        phys:SetAngleVelocityInstantaneous(vector_origin)
      end
    end
  end
  Mentat.Apply(st.t)
  publish()
  return true
end

-- ------------------------------------------------------------------ release
local function release_entity(ent, i)
  local scn = st.scn
  local body = scn.bodies[i]
  -- m/s in the sigma frame -> Source world HU/s
  local vx, vy, vz = I.velocity(scn, body, st.t)
  local v = U.world_vec(scn.up, { vx, vy, vz })
  local vel = Vector(v[1], v[2], v[3]) * st.scale
  -- rad/s in the sigma world frame -> Source world deg/s
  local wx, wy, wz = I.omega(scn, body, st.t, U.angular_velocity)
  local w = U.world_vec(scn.up, { wx, wy, wz })
  local k = 180 / math.pi
  local angvel = Vector(w[1] * k, w[2] * k, w[3] * k)
  return ent:MentatRelease(vel, angvel)
end

--- which: nil/"all" = every body; number = body index; string = label substring.
function Mentat.Release(which)
  if not st.scn then return 0, "no scenario loaded" end
  local n = 0
  for i, ent in ipairs(st.ents) do
    if IsValid(ent) and not ent.MentatReleased then
      local label = tostring(st.scn.bodies[i].label or "")
      local hit = (which == nil or which == "" or which == "all")
        or (tonumber(which) and tonumber(which) == i)
        or (not tonumber(which) and label:lower():find(tostring(which):lower(), 1, true))
      if hit and release_entity(ent, i) then n = n + 1 end
    end
  end
  publish()
  return n
end

-- Grabbing a replay body with the physgun hands it to Havok first. Untested until used in-game.
hook.Add("PhysgunPickup", "MentatPhysgunRelease", function(ply, ent)
  if not (IsValid(ent) and ent:GetClass() == "mentat_body" and st.scn) then return end
  if ent.MentatReleased then return true end
  for i, e in ipairs(st.ents) do
    if e == ent then
      release_entity(ent, i)
      publish()
      return true
    end
  end
end)

-- ------------------------------------------------------------------ the replay clock
hook.Add("Tick", "MentatReplay", function()
  if not (st.scn and st.playing) then return end
  st.t = st.t + engine.TickInterval() * st.rate
  if st.t >= st.tend then
    st.t = st.tend
    st.playing, st.ended = false, true
    Mentat.Apply(st.t)
    publish()
    hook.Run("MentatEnded", st.name)
    return
  end
  Mentat.Apply(st.t)
  SetGlobal2Float("mentat_t", st.t)
end)

-- ------------------------------------------------------------------ console UI
local function say(ply, ...)
  local msg = table.concat({ ... }, "")
  if IsValid(ply) then ply:PrintMessage(HUD_PRINTCONSOLE, msg .. "\n") else print(msg) end
end

local function admin_only(ply)
  return not IsValid(ply) or ply:IsAdmin() or game.SinglePlayer()
end

local function cmd(name, fn)
  concommand.Add(name, function(ply, _, args)
    if not admin_only(ply) then return end
    fn(ply, args)
  end)
end

local function anchor_for(ply, args)
  local x, y, z = tonumber(args[2]), tonumber(args[3]), tonumber(args[4])
  if x and y and z then return Vector(x, y, z) end
  if IsValid(ply) then
    local fwd = ply:EyeAngles():Forward()
    fwd.z = 0
    fwd:Normalize()
    return ply:GetPos() + fwd * (4.5 * U.HU_PER_M)     -- 4.5 m ahead, floor = under the player's feet
  end
  return Vector(0, 0, 0)
end

cmd("mentat_list", function(ply)
  local names = Mentat.ListScenarios()
  say(ply, "[mentat] ", #names, " scenario(s): ", table.concat(names, ", "))
end)

cmd("mentat_load", function(ply, args)
  if not args[1] then return say(ply, "usage: mentat_load <scenario> [x y z]   (see mentat_list)") end
  local ok, err = Mentat.Load(args[1], anchor_for(ply, args))
  if not ok then return say(ply, "[mentat] load failed: ", tostring(err)) end
  local scn = st.scn
  say(ply, string.format("[mentat] loaded %s: %d body(ies), t %.3f..%.3f s, up=%s (%s), default rate %.3gx. mentat_play to start.",
    st.name, #st.ents, st.t0, st.tend, scn.up, scn.up_source, st.rate))
  for _, n in ipairs(scn.notes or {}) do say(ply, "[mentat] note: ", n) end
end)

cmd("mentat_play", function(ply) local ok, err = Mentat.Play(); if not ok then say(ply, "[mentat] ", err) end end)
cmd("mentat_pause", function() Mentat.Pause() end)
cmd("mentat_reset", function(ply) local ok, err = Mentat.Reset(); if not ok then say(ply, "[mentat] ", err) end end)
cmd("mentat_clear", function() Mentat.Clear() end)
cmd("mentat_reload", function(ply) Mentat.ReloadScenarios(); say(ply, "[mentat] scenario cache cleared") end)

cmd("mentat_rate", function(ply, args)
  local ok, err = Mentat.SetRate(args[1])
  if not ok then say(ply, "[mentat] ", err) end
end)

cmd("mentat_scrub", function(ply, args)
  local ok, err = Mentat.Scrub(args[1])
  if not ok then say(ply, "[mentat] ", err) end
end)

cmd("mentat_release", function(ply, args)
  local n, err = Mentat.Release(args[1])
  if err then return say(ply, "[mentat] ", err) end
  say(ply, "[mentat] released ", n, " body(ies) to Havok at t=", string.format("%.3f", st.t))
end)

cmd("mentat_status", function(ply)
  say(ply, string.format("[mentat] %s  scenario=%s  t=%.3f/%.3f  rate=%.3gx  released=%d/%d",
    state_string(), st.name or "-", st.t, st.tend, st.rate, released_count(), #st.ents))
end)

publish()
