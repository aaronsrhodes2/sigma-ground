-- water/fx.lua -- the state of the drawn splashes, ripples and droplets of a water body.  PURE Lua (offline-tested): the client file feeds
-- it events from the server and asks it for heights and droplet positions; it never touches the engine.
--
-- Sizes follow water/splash.lua (the entry's real energy); ripples spread at the group velocity of water/waves.lua (the construct's g, the
-- liquid's rho and sigma); droplets fly the real ballistics under the construct's g.  Counts are capped (drawing budget, not physics).
local M = {}
local inc = include or function(p) return dofile(p) end
local Splash = inc("mentat/water/splash.lua")
local Waves = inc("mentat/water/waves.lua")
local sqrt, exp, cos, sin, pi, min, max, floor = math.sqrt, math.exp, math.cos, math.sin, math.pi, math.min, math.max, math.floor

M.MAX_RIPPLES = 10
M.MAX_DROPS = 160
M.RIPPLE_LIFE = 6.0          -- seconds a ring is kept (it has faded long before; a drawing budget)

function M.new() return { ripples = {}, drops = {}, seed = 1 } end

local function rnd(st)
  st.seed = (st.seed * 16807) % 2147483647
  return st.seed / 2147483647
end

--- A ring from an impact: plan from Splash.plan; (x, y) world metres... any consistent length unit; t0 the clock.
function M.add_ripple(st, x, y, plan, t0, g, depth, sigma_over_rho)
  local lam = plan.ring_lambda
  local cg = Waves.group_speed(lam, g, depth, sigma_over_rho)
  if #st.ripples >= M.MAX_RIPPLES then table.remove(st.ripples, 1) end
  st.ripples[#st.ripples + 1] = { x = x, y = y, t0 = t0, amp = plan.ring_amp, lam = lam, cg = cg, k = 2 * pi / lam, ell = plan.ell }
end

--- A splash: the ring plus the droplets (drop_r radius, thrown up at v_up with a spread of directions).  Heights are metres above the surface.
function M.add_splash(st, x, y, plan, t0, g, depth, sigma_over_rho)
  M.add_ripple(st, x, y, plan, t0, g, depth, sigma_over_rho)
  local n = plan.n
  for i = 1, n do
    if #st.drops >= M.MAX_DROPS then table.remove(st.drops, 1) end
    local a = rnd(st) * 2 * pi
    local spread = 0.25 + 0.6 * rnd(st)                   -- how far sideways relative to straight up
    local sp = plan.v_up * (0.45 + 0.55 * rnd(st))
    st.drops[#st.drops + 1] = {
      x = x + cos(a) * plan.ell * 0.1, y = y + sin(a) * plan.ell * 0.1, z = 0,
      vx = cos(a) * sp * spread, vy = sin(a) * sp * spread, vz = sp, r = plan.drop_r * (0.6 + 0.8 * rnd(st)), t0 = t0,
    }
  end
end

--- Advance the droplets by dt under gravity g (m/s^2); a droplet that falls back to the surface becomes a tiny ripple (returned as a count).
function M.update(st, dt, g, now)
  local landed = 0
  local out = {}
  for _, d in ipairs(st.drops) do
    d.vz = d.vz - g * dt
    d.x, d.y, d.z = d.x + d.vx * dt, d.y + d.vy * dt, d.z + d.vz * dt
    if d.z > 0 or d.vz > 0 then out[#out + 1] = d else landed = landed + 1 end
  end
  st.drops = out
  local keep = {}
  for _, r in ipairs(st.ripples) do
    if now - r.t0 < M.RIPPLE_LIFE then keep[#keep + 1] = r end
  end
  st.ripples = keep
  return landed
end

--- The height (same unit as the ripples) the ripples add to the surface at (x, y) and time t.
--- A packet: a cosine of the ring's wavelength under a Gaussian envelope riding on the front r = cg * age; amplitude falls as sqrt(r0 / r)
--- (energy spread round a growing circle) and as exp(-age / 3 s) (viscous and radiative loss: a visual decay, not a measurement).
function M.height(st, x, y, t)
  local h = 0
  for i = 1, #st.ripples do
    local r = st.ripples[i]
    local age = t - r.t0
    if age >= 0 then
      local dx, dy = x - r.x, y - r.y
      local rr = sqrt(dx * dx + dy * dy)
      local front = r.cg * age
      local sigma = 0.5 * r.lam
      local u = (rr - front) / sigma
      if u > -3 and u < 3 then
        local r0 = max(r.ell, 0.02)
        local a = r.amp * sqrt(r0 / (r0 + front)) * exp(-age / 3)
        h = h + a * exp(-u * u) * cos(r.k * (rr - front))
      end
    end
  end
  return h
end

return M
