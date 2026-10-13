-- world/aero.lua -- real aerodynamic drag on one object: F = 1/2 rho Cd A v^2, opposing the velocity.  PURE Lua (offline-tested
-- against sigma_ground.field.interface.projectile.drag_force).
--
-- rho   air density, from the library (constants_data.lua; ISA sea level).
-- Cd    by shape class (sphere, box, flat_plate, cylinder): the library has no Cd table, so the textbook values in constants_data.lua
--       are labelled "estimated, owner-should-confirm".
-- A     projected area of the object's REAL bounding box along its velocity (a sphere: the circle of its mean diameter).  For a box
--       with side lengths a, b, c along its own axes and a unit velocity direction (nx, ny, nz) in those axes, the shadow is exactly
--       A = |nx| b c + |ny| a c + |nz| a b.   A cylinder is approximated by its bounding box (overstates its area by up to 4/pi).
-- m     the object's real mass.
--
-- The engine integrates the object itself; each physics tick this module gives the velocity after drag.  With k = rho Cd A / (2 m)
-- (1/m) the speed after a step dt is  v / (1 + k v dt)  (an implicit step of dv/dt = -k v^2: it never reverses the motion and
-- converges to the exact solution as dt -> 0, error O(dt)).
local M = {}

local sqrt, abs, pi, min, max = math.sqrt, math.abs, math.pi, math.min, math.max

local SPHERE = { "ball", "sphere", "globe", "orb", "bowling", "basketball", "football", "beachball", "melon", "pumpkin", "marble" }
local CYLINDER = { "barrel", "drum", "can", "bottle", "pole", "pipe", "log", "cylinder", "tube", "pillar", "column", "bucket", "pot" }

local function contains(name, list)
  for _, w in ipairs(list) do if name:find(w, 1, true) then return true end end
  return false
end

--- Shape class from the object's name (kind or model path) and its side lengths (any units, any order).
function M.shape_class(name, dims)
  name = tostring(name or ""):lower()
  local a, b, c = dims[1], dims[2], dims[3]
  local s = { a, b, c }
  table.sort(s)                                            -- s[1] thinnest
  if s[3] <= 0 then return "box" end
  if contains(name, SPHERE) and not name:find("boxball", 1, true) then
    if s[1] > 0.8 * s[3] then return "sphere" end                                  -- a round thing that is nearly round
  end
  if s[1] < 0.1 * s[2] then return "flat_plate" end
  if contains(name, CYLINDER) then return "cylinder" end
  return "box"
end

--- Projected area (same units as dims, squared) of the object along the unit direction n given in the object's own axes.
function M.area(class, dims, n)
  local a, b, c = dims[1], dims[2], dims[3]
  if class == "sphere" then
    local d = (a + b + c) / 3
    return pi / 4 * d * d
  end
  return abs(n[1]) * b * c + abs(n[2]) * a * c + abs(n[3]) * a * b
end

--- k = rho Cd A / (2 m)   (1/m).  The drag deceleration is k v^2.
function M.k(rho, cd, area, mass)
  return 0.5 * rho * cd * area / mass
end

--- The drag force in newtons: the library's formula, written out (the tests compare it with the library's).
function M.force(rho, cd, area, speed)
  return 0.5 * rho * cd * area * speed * speed
end

--- Speed after one step of drag.
function M.speed_after(speed, k, dt)
  return speed / (1 + k * speed * dt)
end

--- Terminal speed of a falling object (drag = weight): sqrt(2 m g / (rho Cd A)).
function M.terminal(mass, g, rho, cd, area)
  return sqrt(2 * mass * g / (rho * cd * area))
end

return M
