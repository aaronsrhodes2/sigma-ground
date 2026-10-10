-- units.lua -- frame + unit conversions between sigma-ground and Garry's Mod (Source).
--
-- PURE Lua 5.1 / LuaJIT: no GMod globals, so the exact same file is unit-tested
-- offline (lupa) against the Python originals. Do not add GLua-only syntax here.
--
-- Conventions
--   sigma-ground: metres; right-handed; quaternions [x,y,z,w], ACTIVE rotation
--                 (world = R * local); the world "up" axis is a per-bundle fact
--                 ("y" for the three.js-style viewer bundles, "z" for the theaters).
--   Source/GMod : Hammer units (1 HU = 0.75 in = 0.01905 m); right-handed with
--                 X forward, Y left, Z up. Angle(pitch,yaw,roll) in degrees.
--
-- A body's ENTITY-LOCAL frame is its sigma body-local frame translated so the
-- origin sits on the body's pivot (= centre of mass). Local axes are NOT remapped;
-- only the WORLD frame is converted (sigma world -> Source world, via WORLD[up]).
local M = {}

M.HU_PER_M = 1 / 0.01905            -- 52.4934...
local sqrt, sin, cos = math.sqrt, math.sin, math.cos
local atan2 = math.atan2 or math.atan   -- LuaJIT/5.1 have atan2; 5.3+ folds it into atan
local pi = math.pi
local DEG, RAD = 180 / pi, pi / 180

-- Source world = WORLD[up] * sigma world.  Both are proper rotations (det = +1).
--   up "y": Source X = x, Y = -z, Z = y      (rotate +90 deg about X)
--   up "z": identity
M.WORLD = {
  y = { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } },
  z = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
}

local function world_matrix(up)
  local W = M.WORLD[up]
  if not W then error("units: unknown up axis '" .. tostring(up) .. "' (expected 'y' or 'z')", 3) end
  return W
end

-- 3x3 helpers (row-major tables of rows) -------------------------------------
local function mat_mul(A, B)
  local C = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
  for i = 1, 3 do
    for j = 1, 3 do
      C[i][j] = A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j]
    end
  end
  return C
end
M.mat_mul = mat_mul

local function mat_vec(A, v)
  return {
    A[1][1] * v[1] + A[1][2] * v[2] + A[1][3] * v[3],
    A[2][1] * v[1] + A[2][2] * v[2] + A[2][3] * v[3],
    A[3][1] * v[1] + A[3][2] * v[2] + A[3][3] * v[3],
  }
end
M.mat_vec = mat_vec

--- Source-world vector for a sigma-world vector (rotation only, no scale/offset).
function M.world_vec(up, v)
  return mat_vec(world_matrix(up), v)
end

--- Rotation matrix of quaternion q = {x,y,z,w} (ACTIVE; q is normalised here).
function M.quat_to_mat(q)
  local x, y, z, w = q[1], q[2], q[3], q[4]
  local n = sqrt(x * x + y * y + z * z + w * w)
  if n == 0 then x, y, z, w = 0, 0, 0, 1 else x, y, z, w = x / n, y / n, z / n, w / n end
  return {
    { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w) },
    { 2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
    { 2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y) },
  }
end

function M.quat_conj(q) return { -q[1], -q[2], -q[3], q[4] } end

-- Hamilton product a*b (apply b first, then a), components {x,y,z,w}.
function M.quat_mul(a, b)
  local ax, ay, az, aw = a[1], a[2], a[3], a[4]
  local bx, by, bz, bw = b[1], b[2], b[3], b[4]
  return {
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  }
end

--- Entity axes in the Source world for a body whose sigma-world orientation is q.
-- Returns forward (local +x), left (local +y), up (local +z), each {x,y,z}.
function M.entity_axes(up, q)
  local A = mat_mul(world_matrix(up), M.quat_to_mat(q))
  return { A[1][1], A[2][1], A[3][1] }, { A[1][2], A[2][2], A[3][2] }, { A[1][3], A[2][3], A[3][3] }
end

--- Source MatrixAngles(): (forward, left, up) -> pitch, yaw, roll in DEGREES.
function M.angles_from_axes(f, l, u)
  local xy = sqrt(f[1] * f[1] + f[2] * f[2])
  local pitch = atan2(-f[3], xy) * DEG
  if xy > 0.001 then
    return pitch, atan2(f[2], f[1]) * DEG, atan2(l[3], u[3]) * DEG
  end
  return pitch, atan2(-l[1], l[2]) * DEG, 0
end

--- Source AngleMatrix(): pitch, yaw, roll (DEGREES) -> forward, left, up. (Test oracle.)
function M.axes_from_angles(pitch, yaw, roll)
  local sy, cy = sin(yaw * RAD), cos(yaw * RAD)
  local sp, cp = sin(pitch * RAD), cos(pitch * RAD)
  local sr, cr = sin(roll * RAD), cos(roll * RAD)
  local crcy, crsy, srcy, srsy = cr * cy, cr * sy, sr * cy, sr * sy
  return
    { cp * cy, cp * sy, -sp },
    { sp * srcy - crsy, sp * srsy + crcy, sr * cp },
    { sp * crcy + srsy, sp * crsy - srcy, cr * cp }
end

--- Convenience: sigma quaternion -> Source pitch, yaw, roll (degrees) for a given up axis.
function M.angles_from_quat(up, q)
  local f, l, u = M.entity_axes(up, q)
  return M.angles_from_axes(f, l, u)
end

--- Source-world position (HU) of a sigma-world point (metres).
-- scale = HU per metre (HU_PER_M * stage_scale); floor_m = sigma height of the
-- bundle's floor along its up axis, subtracted so the floor lands at Z = 0.
function M.world_pos(up, p, scale, floor_m)
  local v = mat_vec(world_matrix(up), p)
  return v[1] * scale, v[2] * scale, (v[3] - (floor_m or 0)) * scale
end

--- Inverse of world_pos: Source-world HU back to sigma-world metres (test/analysis aid).
function M.sigma_pos(up, x, y, z, scale, floor_m)
  local W = world_matrix(up)
  local v = { x / scale, y / scale, z / scale + (floor_m or 0) }
  -- W is orthonormal, so its inverse is its transpose
  return W[1][1] * v[1] + W[2][1] * v[2] + W[3][1] * v[3],
         W[1][2] * v[1] + W[2][2] * v[2] + W[3][2] * v[3],
         W[1][3] * v[1] + W[2][3] * v[2] + W[3][3] * v[3]
end

--- World-frame angular velocity (rad/s, SIGMA frame) taking q0 -> q1 over dt seconds.
function M.angular_velocity(q0, q1, dt)
  if dt <= 0 then return 0, 0, 0 end
  local dq = M.quat_mul(q1, M.quat_conj(q0))
  if dq[4] < 0 then dq = { -dq[1], -dq[2], -dq[3], -dq[4] } end   -- shortest arc
  local s = sqrt(dq[1] * dq[1] + dq[2] * dq[2] + dq[3] * dq[3])
  if s < 1e-12 then return 0, 0, 0 end
  local angle = 2 * atan2(s, dq[4])
  local k = angle / s / dt
  return dq[1] * k, dq[2] * k, dq[3] * k
end

return M
