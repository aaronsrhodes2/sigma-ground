-- interp.lua -- sample a Mentat scenario at a sim time. Lua twin of
-- sigma_ground.radiance.trajectory.poses_at (binary search; lerp pos; shortest-path
-- nlerp quat; lerp temperature). PURE Lua 5.1 / LuaJIT (offline-testable).
--
-- Scenario layout (see sigma_ground/radiance/export_gmod.py):
--   scn.times[k]            sim time of frame k  (k = 1..n)
--   scn.stride              numbers per body per frame: 7 (x y z qx qy qz qw) or 8 (+ T)
--   scn.bodies[i].track     flat array, frame k starts at (k-1)*stride + 1
local M = {}
local sqrt = math.sqrt

--- Index pair bracketing t. Returns lo, hi, u with u in [0,1]; clamped at both ends
-- (lo == hi == 1 before the start, lo == hi == n after the end).
function M.bracket(times, t)
  local n = #times
  if n == 0 then return 1, 1, 0 end
  if t <= times[1] then return 1, 1, 0 end
  if t >= times[n] then return n, n, 0 end
  local lo, hi = 1, n
  while hi - lo > 1 do
    local mid = math.floor((lo + hi) / 2)
    if times[mid] <= t then lo = mid else hi = mid end
  end
  return lo, hi, (t - times[lo]) / math.max(1e-9, times[hi] - times[lo])
end

local function frame(track, stride, k)
  local o = (k - 1) * stride
  return track[o + 1], track[o + 2], track[o + 3], track[o + 4], track[o + 5], track[o + 6], track[o + 7], track[o + 8]
end

--- Pose of one body at sim time t.
-- Returns {pos = {x,y,z}, quat = {x,y,z,w}, temperature_k = T or nil} in the SIGMA frame (metres).
function M.sample_body(scn, body, t)
  local stride, track = scn.stride, body.track
  local lo, hi, u = M.bracket(scn.times, t)
  local ax, ay, az, aqx, aqy, aqz, aqw, aT = frame(track, stride, lo)
  if lo == hi then
    local r = { pos = { ax, ay, az }, quat = { aqx, aqy, aqz, aqw } }
    if stride >= 8 then r.temperature_k = aT end
    return r
  end
  local bx, by, bz, bqx, bqy, bqz, bqw, bT = frame(track, stride, hi)
  local d = aqx * bqx + aqy * bqy + aqz * bqz + aqw * bqw
  local s = (d < 0) and -1 or 1
  local qx = aqx + (bqx * s - aqx) * u
  local qy = aqy + (bqy * s - aqy) * u
  local qz = aqz + (bqz * s - aqz) * u
  local qw = aqw + (bqw * s - aqw) * u
  local n = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
  if n == 0 then n = 1 end
  local r = {
    pos = { ax + (bx - ax) * u, ay + (by - ay) * u, az + (bz - az) * u },
    quat = { qx / n, qy / n, qz / n, qw / n },
  }
  if stride >= 8 then r.temperature_k = aT + (bT - aT) * u end
  return r
end

--- Poses of every body at t (list parallel to scn.bodies).
function M.sample_all(scn, t)
  local out = {}
  for i = 1, #scn.bodies do out[i] = M.sample_body(scn, scn.bodies[i], t) end
  return out
end

--- Segment (lo, hi) whose slope describes motion at t. Inside the recording it is the
-- bracketing segment; before the first / after the last frame it is the first / last
-- segment (ballistic continuation), so a body released at the end of a recording that
-- stopped mid-flight keeps moving instead of freezing dead.
function M.segment(scn, t)
  local times, n = scn.times, #scn.times
  local lo, hi = M.bracket(times, t)
  if lo == hi and n >= 2 then
    if lo == 1 then return 1, 2 end
    return n - 1, n
  end
  return lo, hi
end

--- Piecewise-constant velocity (m/s, SIGMA frame) of a body at t: the exact
-- derivative of the linear interpolation the replay uses.
function M.velocity(scn, body, t)
  local lo, hi = M.segment(scn, t)
  if lo == hi then return 0, 0, 0 end
  local stride, track = scn.stride, body.track
  local ax, ay, az = frame(track, stride, lo)
  local bx, by, bz = frame(track, stride, hi)
  local dt = scn.times[hi] - scn.times[lo]
  if dt <= 0 then return 0, 0, 0 end
  return (bx - ax) / dt, (by - ay) / dt, (bz - az) / dt
end

--- Angular velocity (rad/s, SIGMA world frame) of a body at t over the same segment.
-- angular_velocity is units.angular_velocity (passed in to keep this module standalone).
function M.omega(scn, body, t, angular_velocity)
  local lo, hi = M.segment(scn, t)
  if lo == hi then return 0, 0, 0 end
  local stride, track = scn.stride, body.track
  local _, _, _, ax, ay, az, aw = frame(track, stride, lo)
  local _, _, _, bx, by, bz, bw = frame(track, stride, hi)
  return angular_velocity({ ax, ay, az, aw }, { bx, by, bz, bw }, scn.times[hi] - scn.times[lo])
end

return M
