-- autotest.lua -- headless-ish verification run: records what the ENGINE actually did.
--
-- Trigger: a JSON file  garrysmod/data/mentat/autotest.txt  such as
--   {"scenario": "fall_skillet", "release_at": 0.31, "rate": 1.0, "shots": true, "run_id": "r1"}
-- On the first player spawn it loads the scenario, plays it, hands every body to Havok at
-- `release_at` (sim seconds), keeps recording until the bodies settle, and writes
--   garrysmod/data/mentat/autotest_<scenario>.json
-- The game is NOT closed from here: GMod blocks RunConsoleCommand("quit"). The launcher
-- (stage/gmod/gmod_ctl.sh stop) closes it; a harness should wait for the result file.
--
-- The recorded numbers are RAW engine values (Source-world positions in HU, axis vectors from
-- Entity:GetForward/GetRight/GetUp, the physics object's own pose, a floor trace). Judging them
-- against the bundle is done independently in Python (stage/gmod/verify_autotest.py), NOT here,
-- so a mistake in the Lua frame mapping cannot grade itself.
if not SERVER then return end

local CFG_PATH = "mentat/autotest.txt"
if not file.Exists(CFG_PATH, "DATA") then return end

local raw = file.Read(CFG_PATH, "DATA") or ""
file.Delete(CFG_PATH)                                   -- consumed: a config can never loop
local cfg = util.JSONToTable(raw)

local function bad(msg)
  print("[mentat-autotest] " .. msg)
  file.CreateDir("mentat")
  file.Write("mentat/autotest_error.json", util.TableToJSON({ ok = false, error = msg, raw_config = raw }, true))
end

if type(cfg) ~= "table" or type(cfg.scenario) ~= "string" or not cfg.scenario:match("^[%w_%-]+$") then
  return bad("unreadable config (need a JSON object with a scenario name)")
end
local release_at = cfg.release_at ~= nil and tonumber(cfg.release_at) or nil
local rate = tonumber(cfg.rate) or 1
local lift_hu = tonumber(cfg.anchor_lift_hu) or 0          -- raise the stage so a low scenario has room to fall
if (cfg.release_at ~= nil and (release_at == nil or release_at ~= release_at)) or rate ~= rate or rate <= 0 then
  return bad("release_at and rate must be numbers (rate > 0)")
end

local U = Mentat.units
local RESULT_PATH = "mentat/autotest_" .. cfg.scenario .. ".json"
file.CreateDir("mentat")
file.Delete(RESULT_PATH)                                -- a stale file must never pass for a fresh run
local started = false

local function vec(v) return { v.x, v.y, v.z } end

-- Names a scenario may ask for as a surface material; which of them this build actually knows.
local SURFACE_CANDIDATES = { "default", "metal", "ceramic", "porcelain", "pottery", "glass", "glassbottle", "wood",
  "wood_solid", "concrete", "rubber", "plastic", "brick", "rock", "flesh", "paper", "cardboard", "foliage" }

local function finish(result)
  result.finished = true
  file.CreateDir("mentat")
  file.Write(RESULT_PATH, util.TableToJSON(result, true))
  print("[mentat-autotest] wrote data/" .. RESULT_PATH .. " (" .. #result.samples .. " samples)")
  pcall(Mentat.Clear)                                   -- put gravity and the map back
end

local function run(ply)
  if started then return end
  started = true
  local st = Mentat.state
  local anchor = ply:GetPos() + Vector(300, 0, lift_hu)  -- fixed and known, so Python can check the mapping
  local ok, err = Mentat.Load(cfg.scenario, anchor)
  local result = {
    scenario = cfg.scenario, run_id = cfg.run_id, ok = ok, error = err, samples = {}, shots = {},
    anchor = vec(anchor), anchor_lift_hu = lift_hu, hu_per_m = U.HU_PER_M, release_at = release_at,
    rate = rate, tick_interval = engine.TickInterval(), gravity = vec(physenv.GetGravity()),
    map = game.GetMap(), version = Mentat.VERSION, surface_index = {},
  }
  for _, nm in ipairs(SURFACE_CANDIDATES) do result.surface_index[nm] = util.GetSurfaceIndex(nm) end
  if not ok then return finish(result) end

  Mentat.SetRate(rate)
  result.t0, result.tend = st.t0, st.tend
  result.air_density = physenv.GetAirDensity()
  local ph0 = IsValid(st.ents[1]) and st.ents[1]:GetPhysicsObject()
  local okd, drag = pcall(function() return ph0:IsDragEnabled() end)   -- existence unverified: never let it abort the run
  result.drag_enabled = okd and drag or (okd == false and "unknown" or false)
  result.mass_kg = IsValid(ph0) and ph0:GetMass() or nil
  local okg, dlin, dang = pcall(function() return ph0:GetDamping() end)
  if okg then result.damping_linear, result.damping_angular = dlin, dang end

  -- the true floor under the stage, by trace, instead of trusting the player's feet height
  local tr = util.TraceLine({ start = anchor + Vector(0, 0, 64 - lift_hu), endpos = anchor - Vector(0, 0, 512 + lift_hu),
                              mask = MASK_SOLID_BRUSHONLY })
  result.floor_z = tr.Hit and tr.HitPos.z or nil

  local function shot(tag)
    if not cfg.shots then return end
    local ent = st.ents[1]
    if IsValid(ent) then                                 -- aim the camera at the body, wherever it is
      ply:SetEyeAngles((ent:GetPos() - ply:EyePos()):Angle())
    end
    Mentat.Shot(tag)
    result.shots[#result.shots + 1] = tag
  end

  -- phases: "play" (replaying, waiting for the release time) -> "settle" (Havok has the body) ->
  -- "final" (grace period so the last screenshot is written) -> result file.
  local phase, released_ct, still_ticks, started_ct, final_at = "play", nil, 0, CurTime(), nil
  local took_mid = false
  shot("loaded")
  Mentat.Play()

  local function sample()
    local ent = st.ents[1]
    if not IsValid(ent) then error("body entity vanished") end
    local phys = ent:GetPhysicsObject()
    local vel = IsValid(phys) and phys:GetVelocity() or Vector(0, 0, 0)
    local row = {
      ct = CurTime(), sim_t = st.t, pose_t = ent.MentatPoseT, released = ent.MentatReleased and true or false,
      pos = vec(ent:GetPos()), fwd = vec(ent:GetForward()), right = vec(ent:GetRight()), up = vec(ent:GetUp()),
      vel = vec(vel), asleep = IsValid(phys) and phys:IsAsleep() or false, phys_valid = IsValid(phys),
    }
    if IsValid(phys) then
      row.phys_pos = vec(phys:GetPos())
      local pa = phys:GetAngles()                       -- does a frozen physics object follow SetAngles?
      row.phys_fwd, row.phys_up = vec(pa:Forward()), vec(pa:Up())
    end
    result.samples[#result.samples + 1] = row
    return vel
  end

  local function step()
    if CurTime() - started_ct > 40 then
      result.error = "timed out"
      phase, final_at = "final", CurTime()
    end

    if phase == "play" then
      local mid_t = st.t0 + ((release_at or st.tend) - st.t0) * 0.5
      if cfg.shots and not took_mid and st.t >= mid_t then
        took_mid = true
        shot("mid")
      end
      if release_at and st.t >= release_at then
        Mentat.Release("all")
        released_ct = CurTime()
        result.released_ct, result.released_sim_t = released_ct, st.t
        phase = "settle"
      elseif not release_at and st.ended then          -- replay-only run
        phase, final_at = "final", CurTime() + (cfg.shots and 1 or 0.25)
      end
      sample()
    elseif phase == "settle" then
      local vel = sample()
      still_ticks = (vel:Length() < 1) and (still_ticks + 1) or 0
      if still_ticks >= 30 or CurTime() - released_ct > 6 then
        result.settled = still_ticks >= 30
        shot("landed")
        phase, final_at = "final", CurTime() + (cfg.shots and 1 or 0.25)
      end
    elseif phase == "final" then
      if CurTime() >= final_at then
        hook.Remove("Tick", "MentatAutotest")
        finish(result)
      end
    end
  end

  hook.Add("Tick", "MentatAutotest", function()
    local okp, perr = pcall(step)
    if not okp then                                    -- never throw every tick: record and stop
      result.error = tostring(perr)
      hook.Remove("Tick", "MentatAutotest")
      finish(result)
    end
  end)
end

hook.Add("PlayerInitialSpawn", "MentatAutotest", function(ply)
  timer.Simple(5, function()
    if IsValid(ply) then run(ply) end
  end)
end)
print("[mentat-autotest] armed for '" .. tostring(cfg.scenario) .. "' (run_id " .. tostring(cfg.run_id) .. ")")
