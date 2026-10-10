-- autotest.lua -- headless-ish verification run: records what the ENGINE actually did.
--
-- Trigger: a JSON file  garrysmod/data/mentat/autotest.txt  such as
--   {"scenario": "fall_skillet", "release_at": 0.31, "rate": 1.0}
-- On the first player spawn it loads the scenario, plays it, hands every body to Havok at
-- `release_at` (sim seconds), keeps recording until the bodies settle, writes
--   garrysmod/data/mentat/autotest_<scenario>.json
-- and quits the game. The file is consumed (deleted) so it can never loop.
--
-- The recorded numbers are RAW engine values (Source-world positions in HU, axis vectors
-- from Entity:GetForward/GetRight/GetUp). Judging them against the bundle is done
-- independently in Python (stage/gmod/verify_autotest.py), NOT here, so a mistake in the
-- Lua frame mapping cannot grade itself.
if not SERVER then return end

local CFG_PATH = "mentat/autotest.txt"
if not file.Exists(CFG_PATH, "DATA") then return end

local cfg = util.JSONToTable(file.Read(CFG_PATH, "DATA") or "")
file.Delete(CFG_PATH)
if not (cfg and cfg.scenario) then
  print("[mentat-autotest] unreadable config; ignoring")
  return
end

local U = Mentat.units
local started = false

local function vec(v) return { v.x, v.y, v.z } end

local function finish(result)
  file.CreateDir("mentat")
  local path = "mentat/autotest_" .. cfg.scenario .. ".json"
  file.Write(path, util.TableToJSON(result, true))
  print("[mentat-autotest] wrote data/" .. path .. " (" .. #result.samples .. " samples)")
  timer.Simple(0.5, function() RunConsoleCommand("quit") end)
end

local function run(ply)
  if started then return end
  started = true
  local st = Mentat.state
  local anchor = ply:GetPos() + Vector(300, 0, 0)       -- a fixed, known anchor: easy to check in Python
  local ok, err = Mentat.Load(cfg.scenario, anchor)
  local result = {
    scenario = cfg.scenario, ok = ok, error = err, samples = {},
    anchor = vec(anchor), hu_per_m = U.HU_PER_M, release_at = cfg.release_at,
    rate = cfg.rate or 1, tick_interval = engine.TickInterval(), gravity = vec(physenv.GetGravity()),
    map = game.GetMap(), version = Mentat.VERSION,
  }
  if not ok then return finish(result) end

  Mentat.SetRate(cfg.rate or 1)
  result.t0, result.tend = st.t0, st.tend
  result.floor_probe = nil
  Mentat.Play()

  local released_ct, still_ticks, started_ct = nil, 0, CurTime()
  hook.Add("Tick", "MentatAutotest", function()
    local ent = st.ents[1]
    if not IsValid(ent) then
      hook.Remove("Tick", "MentatAutotest")
      result.error = "body entity vanished"
      return finish(result)
    end
    local phys = ent:GetPhysicsObject()

    if not released_ct and cfg.release_at and st.t >= cfg.release_at then
      Mentat.Release("all")
      released_ct = CurTime()
      result.released_ct, result.released_sim_t = released_ct, st.t
    end

    local f, r, u = ent:GetForward(), ent:GetRight(), ent:GetUp()
    local vel = IsValid(phys) and phys:GetVelocity() or Vector(0, 0, 0)
    result.samples[#result.samples + 1] = {
      ct = CurTime(), sim_t = st.t, pose_t = ent.MentatPoseT, released = ent.MentatReleased and true or false,
      pos = vec(ent:GetPos()), fwd = vec(f), right = vec(r), up = vec(u), vel = vec(vel),
      asleep = IsValid(phys) and phys:IsAsleep() or nil,
    }

    if released_ct then
      still_ticks = (vel:Length() < 1) and (still_ticks + 1) or 0
      if still_ticks >= 30 or CurTime() - released_ct > 6 then
        hook.Remove("Tick", "MentatAutotest")
        result.settled = still_ticks >= 30
        result.wall_seconds = CurTime() - started_ct
        return finish(result)
      end
    elseif CurTime() - started_ct > 30 then
      hook.Remove("Tick", "MentatAutotest")
      result.error = "timed out before release"
      return finish(result)
    end
  end)
end

hook.Add("PlayerInitialSpawn", "MentatAutotest", function(ply)
  timer.Simple(5, function()
    if IsValid(ply) then run(ply) end
  end)
end)
print("[mentat-autotest] armed for '" .. tostring(cfg.scenario) .. "'; waiting for a player to spawn")
