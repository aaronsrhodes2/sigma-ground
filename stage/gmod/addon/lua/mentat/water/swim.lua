-- water/swim.lua -- a person in the water.  The pure step (tested offline) and the Move hook that applies it (both realms: the client
-- predicts with the same numbers).
--
-- A swimmer is a body of the human density and mass (water/data.lua "human": ESTIMATED, owner-should-confirm; the library has no body
-- density) in the standing hull (32 x 32 x 72 HU), with Archimedes on the part of the hull below the surface, the same drag law as every
-- other body (Cd of a box, the liquid's own viscosity) and a swimming thrust (also estimated).  The keys: forward/back/sides swim along
-- the view, JUMP swims up (treading water) and, at an edge you can reach, pulls you out onto it; DUCK dives.  No drowning and no damage: this
-- is not a game of survival.  Footsteps need no sound.
local M = {}
local inc = include or function(p) return dofile(p) end
local C = inc("mentat/water/constants.lua")
local Hydro = inc("mentat/water/hydro.lua")
local sqrt, abs, min, max = math.sqrt, math.abs, math.min, math.max

local H = C.data.human
M.HULL_M = { H.hull_hu.value[1] * C.HU_M, H.hull_hu.value[2] * C.HU_M, H.hull_hu.value[3] * C.HU_M }
M.CD_BOX = 1.05                   -- same Cd as world/constants_data.lua cd.box (set by the caller from the construct's table when there is one)
M.MIN_FRACTION = 0.5              -- swimming starts when the hull is at least half under (waist deep); shallower is wading: the engine's own walking

local function clamp(x, a, b) return x < a and a or (x > b and b or x) end

--- Fraction of the hull below the surface.
function M.fraction(z_feet, surface)
  return clamp((surface - z_feet) / M.HULL_M[3], 0, 1)
end

--- One step of a swimmer.
--- S = { z_feet, surface (m), vel = {x,y,z} (m/s), wish = {x,y,z} (0..1 each, world axes), up (bool), down (bool), g, rho (kg/m3), eta (Pa s), cooldown (s),
---       cd }; dt in s.  Returns the new velocity {x,y,z}, a table of info {frac, a_buoy, cooldown}.
function M.step(S, dt)
  local rho_b, mass, thrust = H.density_kg_m3.value, H.mass_kg.value, H.swim_thrust_n.value
  local f = M.fraction(S.z_feet, S.surface)
  local v = { S.vel[1], S.vel[2], S.vel[3] }
  -- Archimedes on the immersed part, minus the weight: the mass cancels (V = m / rho_body)
  local a_z = S.g * (S.rho * f / rho_b - 1)
  -- the swimmer's own effort
  local fx, fy, fz = S.wish[1], S.wish[2], S.wish[3]
  if S.up then fz = fz + 1 end
  if S.down then fz = fz - 1 end
  local mag = sqrt(fx * fx + fy * fy + fz * fz)
  if mag > 1 then fx, fy, fz = fx / mag, fy / mag, fz / mag end
  v[1] = v[1] + fx * thrust / mass * dt
  v[2] = v[2] + fy * thrust / mass * dt
  v[3] = v[3] + (a_z + fz * thrust / mass) * dt
  -- a person piercing the surface radiates waves as they move up and down: the same estimated heave damping as every floating body
  if f > 0 and f < 1 then
    local c = Hydro.heave_radiation_c(S.rho, S.g, M.HULL_M[1] * M.HULL_M[2], f * M.HULL_M[3], mass)
    v[3] = v[3] / (1 + c * dt)
  end
  local cooldown = max(0, (S.cooldown or 0) - dt)
  -- drag on the hull (immersed part of its projected area)
  local sp = sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
  if sp > 1e-6 and f > 0 then
    local hw, hl, hh = M.HULL_M[1], M.HULL_M[2], M.HULL_M[3]
    local nx, ny, nz = v[1] / sp, v[2] / sp, v[3] / sp
    local area = (abs(nx) * hl * hh + abs(ny) * hw * hh + abs(nz) * hw * hl) * f
    local k2, k1 = Hydro.drag_k(S.rho, S.cd or M.CD_BOX, area, mass, S.eta)
    local sp2 = Hydro.speed_after(sp, k2, k1, dt)
    local s = sp2 / sp
    v[1], v[2], v[3] = v[1] * s, v[2] * s, v[3] * s
  end
  return v, { frac = f, a_buoy = a_z, cooldown = cooldown }
end

--- Can a swimmer whose feet are at z_feet (m) pull themselves onto a ledge whose top is at ledge_top (m)?  Their hands reach as high as the
--- hull is tall; a ledge below their feet is not a climb.
function M.reach_ok(z_feet, ledge_top)
  return ledge_top >= z_feet and ledge_top - z_feet <= M.HULL_M[3]
end

--- The terminal speed of a swimmer pushing along one axis with full thrust (for the tests and the readback).
function M.top_speed(axis, rho, g, eta)
  local mass, thrust = H.mass_kg.value, H.swim_thrust_n.value
  local hw, hl, hh = M.HULL_M[1], M.HULL_M[2], M.HULL_M[3]
  local area = axis == "z" and hw * hl or hl * hh
  local k2, k1 = Hydro.drag_k(rho, M.CD_BOX, area, mass, eta)
  -- thrust/mass = k2 v^2 + k1 v
  local a = thrust / mass
  return (-k1 + sqrt(k1 * k1 + 4 * k2 * a)) / (2 * k2)
end

return M
