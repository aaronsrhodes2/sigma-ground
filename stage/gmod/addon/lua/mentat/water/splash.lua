-- water/splash.lua -- the size of a splash from the real energy of the entry.  PURE Lua (offline-tested).
--
-- E = 1/2 m v^2 of the body as it crosses the surface (v = its speed into the liquid, m its real mass).  The cavity it digs scales as
--   l = (E / (rho g))^(1/4)         (gravity-regime impact-crater scaling; Engel 1966; the prefactor 1 is an ESTIMATE)
-- and everything DRAWN follows from l and g: the droplets rise to about l (launch speed sqrt(2 g l), never more than the entry speed),
-- one droplet per centimetre of l (capped), the ring spreads at the group velocity of a wave of wavelength 2 pi l.  This sizes a
-- picture of a splash; no fluid is simulated (that is Phase C).
local M = {}
local sqrt, min, max, ceil, pi = math.sqrt, math.min, math.max, math.ceil, math.pi

M.MAX_PARTICLES = 48
M.MIN_ENERGY_J = 1e-4          -- below this the entry makes no visible splash (a falling leaf's last millimetre)

function M.energy(mass, speed) return 0.5 * mass * speed * speed end

--- The cavity length scale (m).
function M.length_scale(E, rho, g)
  if E <= 0 or rho <= 0 or g <= 0 then return 0 end
  return (E / (rho * g)) ^ 0.25
end

--- What to draw: { E, ell, n, v_up, ring_lambda, ring_amp, drop_r } or nil when the entry is too gentle.
function M.plan(mass, speed, rho, g, max_particles)
  local E = M.energy(mass, speed)
  if E < M.MIN_ENERGY_J then return nil end
  local ell = M.length_scale(E, rho, g)
  local n = min(max_particles or M.MAX_PARTICLES, max(1, ceil(ell / 0.01)))
  local v_up = min(speed, sqrt(2 * g * ell))
  return {
    E = E, ell = ell, n = n, v_up = v_up,
    ring_lambda = max(2 * pi * ell, 0.02),
    ring_amp = min(ell / 4, 0.25),
    drop_r = max(0.004, min(0.05, ell / 12)),
  }
end

return M
