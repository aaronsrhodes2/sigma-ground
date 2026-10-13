-- water/waves.lua -- the surface waves of a water body, from the real dispersion relation.  PURE Lua (offline-tested).
--
-- Linear capillary-gravity waves (Kelvin 1871):   omega^2 = (g k + (sigma / rho) k^3) tanh(k d)
--   deep water (k d >> 1)   omega^2 = g k   (phase speed sqrt(g / k), longer waves faster)
--   shallow water (k d << 1)  c = sqrt(g d)  (all wavelengths the same speed)
-- g is the construct's own gravity (the Moon's water is slower and its waves longer), rho and sigma the liquid's own row of the
-- library, d the body's depth.  The DRAWN surface is a sum of a few such waves (amplitude chosen by steepness a*k, a visual choice far
-- below the breaking limit, so linear theory holds); ripples from impacts spread at the group velocity d(omega)/dk.
-- No wind, no currents, no breaking, no interaction of waves with objects: what is drawn is the dispersion relation, not a simulation.
local M = {}
local sqrt, tanh, pi, sin, cos, max, min, abs, floor = math.sqrt, function(x) if x > 20 then return 1 end local e = math.exp(2 * x) return (e - 1) / (e + 1) end, math.pi, math.sin, math.cos, math.max, math.min, math.abs, math.floor

--- omega (rad/s) of a wave of wavenumber k (rad/m).
function M.omega(k, g, depth, sigma_over_rho)
  sigma_over_rho = sigma_over_rho or 0
  local w2 = (g * k + sigma_over_rho * k ^ 3) * tanh(k * depth)
  return sqrt(max(w2, 0))
end

function M.wavenumber(lambda) return 2 * pi / lambda end

--- Phase speed (m/s) of wavelength lambda.
function M.phase_speed(lambda, g, depth, sigma_over_rho)
  local k = 2 * pi / lambda
  return M.omega(k, g, depth, sigma_over_rho) / k
end

--- Group speed (m/s): d omega / d k (centred difference, exact to the step).
function M.group_speed(lambda, g, depth, sigma_over_rho)
  local k = 2 * pi / lambda
  local dk = k * 1e-4
  return (M.omega(k + dk, g, depth, sigma_over_rho) - M.omega(k - dk, g, depth, sigma_over_rho)) / (2 * dk)
end

--- The shallow-water limit, sqrt(g d), and the deep-water phase speed, sqrt(g lambda / 2 pi): the two asymptotes (for the tests).
function M.shallow_speed(g, depth) return sqrt(g * depth) end
function M.deep_speed(lambda, g) return sqrt(g * lambda / (2 * pi)) end

--- Is this wavelength deep water for this depth (d > lambda / 2)?
function M.is_deep(lambda, depth) return depth > lambda / 2 end

-- a cheap deterministic hash in [0,1)
local function hash(n)
  local s = sin(n * 12.9898 + 78.233) * 43758.5453
  return s - floor(s)
end
M.hash = hash

--- The wave components drawn on a body: n of them from lambda_min up by a factor `ratio` each, with directions spread over a
--- half-plane around `heading` (radians) and amplitude from steepness a*k.  Each: { k, omega, amp, dx, dy, phase, lambda }.
--- Wavelengths longer than half the body's smaller side are dropped (they would not fit).
function M.components(g, depth, sigma_over_rho, lambda_min, side, steepness, n, ratio, heading)
  n = n or 4
  ratio = ratio or 1.7
  heading = heading or 0.4
  local out = {}
  local lam = lambda_min
  for i = 1, n do
    if lam <= max(side / 2, lambda_min) then
      local k = 2 * pi / lam
      local ang = heading + (hash(i * 7.3) - 0.5) * 1.6
      out[#out + 1] = {
        k = k, lambda = lam, omega = M.omega(k, g, depth, sigma_over_rho), amp = steepness / k,
        dx = cos(ang), dy = sin(ang), phase = hash(i * 3.7) * 2 * pi,
      }
    end
    lam = lam * ratio
  end
  if #out == 0 then
    local lam0 = max(lambda_min, 1e-3)
    local k = 2 * pi / lam0
    out[1] = { k = k, lambda = lam0, omega = M.omega(k, g, depth, sigma_over_rho), amp = steepness / k, dx = 1, dy = 0, phase = 0 }
  end
  return out
end

--- Surface height (m, relative to the still level) at (x, y) and time t.
function M.height(comps, x, y, t)
  local h = 0
  for i = 1, #comps do
    local c = comps[i]
    h = h + c.amp * sin(c.k * (c.dx * x + c.dy * y) - c.omega * t + c.phase)
  end
  return h
end

--- Height and slope (dh/dx, dh/dy), for the shading.
function M.height_slope(comps, x, y, t)
  local h, sx, sy = 0, 0, 0
  for i = 1, #comps do
    local c = comps[i]
    local arg = c.k * (c.dx * x + c.dy * y) - c.omega * t + c.phase
    h = h + c.amp * sin(arg)
    local d = c.amp * c.k * cos(arg)
    sx, sy = sx + d * c.dx, sy + d * c.dy
  end
  return h, sx, sy
end

--- A ring from an impact: radius at age t (m) and its amplitude there.  ell = the splash's length scale, lambda = the ring's wavelength.
function M.ring(ell, lambda, age, g, depth, sigma_over_rho)
  local cg = M.group_speed(lambda, g, depth, sigma_over_rho)
  local r = cg * age
  return r, cg
end

return M
