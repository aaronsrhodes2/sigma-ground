-- water/hydro.lua -- real hydrostatics and hydrodynamics of one body in a liquid.  PURE Lua 5.1 / LuaJIT: no engine calls, so it is
-- tested offline against the sigma-ground library's formulas (fluid_surface.submerged_fraction, projectile.drag_force).
--
-- LAWS (and nothing else):
--   Archimedes    F_b = rho_liquid * g * V_submerged, upward, applied at the centroid of the submerged volume (the centre of buoyancy).
--   Drag          F = 1/2 rho Cd A v^2 + 12 eta A v / L  (opposing the velocity).  The first term is the same quadratic law as the air drag
--                 (world/aero.lua; the same Cd shape classes and the same area estimates).  The second is Stokes' law: for a sphere of
--                 diameter L, A = pi L^2 / 4, it is exactly 3 pi eta L v; it is the part that matters below Reynolds number ~1 and is
--                 below 2.5 percent of the quadratic term above Re 1000, so it is always included.  eta is the liquid's own viscosity.
--   Angular drag  the same quadratic law on the body's area at a lever arm of L/4:  tau = 1/2 rho Cd A (L/4)^3 w^2   (an estimate, stated).
--   NOT modelled  added mass, lift, surface tension on the body, free-surface drag (wave making), currents.
--
-- GEOMETRY is exact for the shapes it knows: a box at any orientation (the volume of its part below a horizontal plane, by clipping its
-- six faces; the centroid by tetrahedra), a sphere (the spherical cap, V = pi h^2 (3r - h) / 3), a cylinder (the circular segment
-- integrated along its axis: closed forms for a vertical and a horizontal axis, Simpson quadrature of the exact segment area otherwise).
-- All lengths in metres, all forces in newtons, all speeds in m/s.
local M = {}

local sqrt, abs, pi, min, max, acos, asin, atan2, sin, cos = math.sqrt, math.abs, math.pi, math.min, math.max, math.acos, math.asin, math.atan2, math.sin, math.cos
atan2 = atan2 or function(y, x) return math.atan(y, x) end

local function clamp(x, a, b) return x < a and a or (x > b and b or x) end

-- ------------------------------------------------------------------------------------------------ box (any orientation)
local function polygon_clip(poly, level)
  -- keep the part with z <= level (Sutherland-Hodgman); returns the polygon and the points that lie on the plane
  local out, onplane = {}, {}
  local n = #poly
  if n == 0 then return out, onplane end
  for i = 1, n do
    local a, b = poly[i], poly[i % n + 1]
    local da, db = a[3] - level, b[3] - level
    local a_in, b_in = da <= 0, db <= 0
    if a_in then
      out[#out + 1] = a
      if da == 0 then onplane[#onplane + 1] = a end
    end
    if a_in ~= b_in then
      local t = da / (da - db)
      local p = { a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, level }
      out[#out + 1] = p
      onplane[#onplane + 1] = p
    end
  end
  return out, onplane
end

--- The part of an oriented box below the horizontal plane z = level.
--- dims = { a, b, c } full side lengths along the unit axes ax, ay, az ({x,y,z} tables, right-handed), centre = {x,y,z}.
--- Returns volume (m^3), centroid {x,y,z}, and the volume of the whole box.
function M.box_submerged(dims, ax, ay, az, centre, level)
  local axes = { ax, ay, az }
  local h = { dims[1] / 2, dims[2] / 2, dims[3] / 2 }
  local total = dims[1] * dims[2] * dims[3]
  local c = centre
  -- quick outs: the lowest and highest corner
  local ext = 0
  for i = 1, 3 do ext = ext + h[i] * abs(axes[i][3]) end
  if c[3] + ext <= level then return total, { c[1], c[2], c[3] }, total end
  if c[3] - ext >= level then return 0, { c[1], c[2], c[3] }, total end
  local function corner(si, sj, sk)
    local p = { c[1], c[2], c[3] }
    local s = { si, sj, sk }
    for i = 1, 3 do for k = 1, 3 do p[k] = p[k] + s[i] * h[i] * axes[i][k] end end
    return p
  end
  local faces = {}
  for i = 1, 3 do
    local j, k = i % 3 + 1, (i + 1) % 3 + 1
    for _, s in ipairs({ 1, -1 }) do
      local function v(sj, sk)
        local sv = { 0, 0, 0 }
        sv[i], sv[j], sv[k] = s, sj, sk
        return corner(sv[1], sv[2], sv[3])
      end
      local q = { v(-1, -1), v(1, -1), v(1, 1), v(-1, 1) }
      if s < 0 then q = { q[4], q[3], q[2], q[1] } end
      faces[#faces + 1] = q
    end
  end
  local o = { c[1], c[2], c[3] }
  local vol, mx, my, mz = 0, 0, 0, 0
  local cap = {}
  local function add_poly(poly)
    for i = 2, #poly - 1 do
      local a, b, d = poly[1], poly[i], poly[i + 1]
      local ax_, ay_, az_ = a[1] - o[1], a[2] - o[2], a[3] - o[3]
      local bx_, by_, bz_ = b[1] - o[1], b[2] - o[2], b[3] - o[3]
      local dx_, dy_, dz_ = d[1] - o[1], d[2] - o[2], d[3] - o[3]
      local t = (ax_ * (by_ * dz_ - bz_ * dy_) - ay_ * (bx_ * dz_ - bz_ * dx_) + az_ * (bx_ * dy_ - by_ * dx_)) / 6
      vol = vol + t
      mx = mx + t * (a[1] + b[1] + d[1] + o[1]) / 4
      my = my + t * (a[2] + b[2] + d[2] + o[2]) / 4
      mz = mz + t * (a[3] + b[3] + d[3] + o[3]) / 4
    end
  end
  for _, q in ipairs(faces) do
    local poly, on = polygon_clip(q, level)
    if #poly >= 3 then add_poly(poly) end
    for _, p in ipairs(on) do cap[#cap + 1] = p end
  end
  -- the cap on the plane: distinct points, ordered counter-clockwise seen from above
  local pts = {}
  for _, p in ipairs(cap) do
    local dup = false
    for _, q in ipairs(pts) do
      if abs(p[1] - q[1]) < 1e-9 and abs(p[2] - q[2]) < 1e-9 then dup = true; break end
    end
    if not dup then pts[#pts + 1] = p end
  end
  if #pts >= 3 then
    local cx, cy = 0, 0
    for _, p in ipairs(pts) do cx, cy = cx + p[1], cy + p[2] end
    cx, cy = cx / #pts, cy / #pts
    table.sort(pts, function(p, q) return atan2(p[2] - cy, p[1] - cx) < atan2(q[2] - cy, q[1] - cx) end)
    add_poly(pts)
  end
  if vol <= 1e-15 then return 0, { c[1], c[2], c[3] }, total end
  return vol, { mx / vol, my / vol, mz / vol }, total
end

-- ------------------------------------------------------------------------------------------------ sphere
--- Spherical cap below the plane.  cz = centre height.  Returns volume, centroid {x,y,z}, total volume.
function M.sphere_submerged(r, centre, level)
  local total = 4 / 3 * pi * r ^ 3
  local h = clamp(level - (centre[3] - r), 0, 2 * r)             -- depth of the cap
  if h <= 0 then return 0, { centre[1], centre[2], centre[3] }, total end
  local v = pi * h * h * (3 * r - h) / 3
  local below = 0                                                  -- how far under the centre the cap's centroid lies
  if h < 2 * r then below = 3 * (2 * r - h) ^ 2 / (4 * (3 * r - h)) end
  return v, { centre[1], centre[2], centre[3] - below }, total
end

-- ------------------------------------------------------------------------------------------------ cylinder
-- area of the part of a disc of radius r with u <= d, and its first moment about the disc's centre along u
local function seg_area(r, d)
  if d <= -r then return 0, 0 end
  if d >= r then return pi * r * r, 0 end
  local s = sqrt(r * r - d * d)
  return r * r * (pi / 2 + asin(d / r)) + d * s, -(2 / 3) * s ^ 3
end
M.segment_area = seg_area

--- Cylinder of radius r and length L, axis = unit vector `axis` (either sign), centre = {x,y,z}.
--- Returns volume, centroid, total volume.
function M.cylinder_submerged(r, L, axis, centre, level)
  local total = pi * r * r * L
  local az = axis[3]
  local c = centre
  local half = L / 2
  local sinth = sqrt(max(0, 1 - az * az))                          -- |horizontal part of the axis|
  if sinth < 1e-9 then                                              -- vertical axis: a disc by a depth
    local h = clamp(level - (c[3] - half), 0, L)
    local v = pi * r * r * h
    return v, { c[1], c[2], c[3] - half + h / 2 }, total
  end
  if abs(az) < 1e-9 then                                            -- horizontal axis: a circular segment along the length
    local d = level - c[3]
    local a, m1 = seg_area(r, d)
    if a <= 0 then return 0, { c[1], c[2], c[3] }, total end
    return a * L, { c[1], c[2], c[3] + m1 / a }, total
  end
  -- tilted: integrate the exact segment area along the axis.  The "up" direction in the disc's plane has length sinth.  Where the
  -- waterline is beyond the rim of the disc the section is whole (or empty): those stretches are exact, and Simpson runs only over the
  -- band where the waterline cuts the disc.
  local ux, uy, uz = -az * axis[1], -az * axis[2], 1 - az * az
  local ul = sqrt(ux * ux + uy * uy + uz * uz)
  ux, uy, uz = ux / ul, uy / ul, uz / ul
  local s_full = (level - c[3] - r * sinth) / az                    -- the section is whole on one side of this...
  local s_none = (level - c[3] + r * sinth) / az                    -- ...and empty beyond this
  local lo_s, hi_s = s_full, s_none
  local whole_below = az > 0                                         -- az > 0: whole for s < s_full; az < 0: whole for s > s_full
  if lo_s > hi_s then lo_s, hi_s = hi_s, lo_s end
  local sumV, sumX, sumY, sumZ = 0, 0, 0, 0
  local disc = pi * r * r
  local function whole(s0, s1)
    if s1 <= s0 then return end
    local len = s1 - s0
    local mid = (s0 + s1) / 2
    local vv = disc * len
    sumV = sumV + vv
    sumX = sumX + vv * (c[1] + mid * axis[1])
    sumY = sumY + vv * (c[2] + mid * axis[2])
    sumZ = sumZ + vv * (c[3] + mid * az)
  end
  if whole_below then whole(-half, min(half, lo_s)) else whole(max(-half, hi_s), half) end
  local band0, band1 = max(-half, lo_s), min(half, hi_s)
  if band1 > band0 then
    local n = 128
    local hstep = (band1 - band0) / n
    local bV, bX, bY, bZ = 0, 0, 0, 0
    for i = 0, n do
      local sp = band0 + i * hstep
      local dd = (level - c[3] - sp * az) / sinth
      local a, m1 = seg_area(r, dd)
      local w = (i == 0 or i == n) and 1 or ((i % 2 == 1) and 4 or 2)
      local px, py, pz = c[1] + sp * axis[1], c[2] + sp * axis[2], c[3] + sp * az
      bV = bV + w * a
      bX = bX + w * (a * px + m1 * ux)
      bY = bY + w * (a * py + m1 * uy)
      bZ = bZ + w * (a * pz + m1 * uz)
    end
    local k = hstep / 3
    sumV, sumX, sumY, sumZ = sumV + bV * k, sumX + bX * k, sumY + bY * k, sumZ + bZ * k
  end
  if sumV <= 1e-15 then return 0, { c[1], c[2], c[3] }, total end
  return sumV, { sumX / sumV, sumY / sumV, sumZ / sumV }, total
end

-- ------------------------------------------------------------------------------------------------ a body: its volume and its displacement
--- The shape the card's size is read as.  dims = { a, b, c } (m, the body's own axes), class from world/aero.lua's shape_class.
--- Returns a shape table: { kind = "box"|"sphere"|"cylinder", ... , volume = m^3, basis = "measured"|"estimated", note = text }.
--- `measured_volume` (m^3), when the card has a real one (e.g. a watertight mesh), replaces the bounding shape's volume as the body's
--- displaced volume: the submerged FRACTION is still the bounding shape's, scaled by it (stated: the mass is assumed spread like the box).
function M.shape(class, dims, measured_volume)
  local a, b, c = dims[1], dims[2], dims[3]
  local shape
  if class == "sphere" then
    local d = (a + b + c) / 3
    shape = { kind = "sphere", r = d / 2, volume = 4 / 3 * pi * (d / 2) ^ 3, note = "a sphere of the mean diameter (exact cap geometry)" }
  elseif class == "cylinder" then
    -- the axis is the side that differs from the other two (a standing barrel: its height)
    local ax, rad_a, rad_b, len
    local dab, dac, dbc = abs(a - b), abs(a - c), abs(b - c)
    if dab <= dac and dab <= dbc then ax, rad_a, rad_b, len = 3, a, b, c
    elseif dac <= dbc then ax, rad_a, rad_b, len = 2, a, c, b
    else ax, rad_a, rad_b, len = 1, b, c, a end
    local r = (rad_a + rad_b) / 4
    shape = { kind = "cylinder", r = r, L = len, axis = ax, volume = pi * r * r * len, note = "a cylinder of the mean width (exact segment geometry)" }
  else
    shape = { kind = "box", dims = { a, b, c }, volume = a * b * c,
              note = "its bounding box taken as solid (shape factor 1; a hollow or lumpy object is lighter than this says)" }
  end
  shape.bbox_volume = a * b * c
  shape.shape_volume = shape.volume
  if measured_volume and measured_volume > 0 then
    shape.volume = measured_volume
    shape.basis = "measured"
    shape.note = "the card's measured volume; the immersion is the bounding shape's, scaled"
  else
    shape.basis = "estimated"
  end
  return shape
end

--- Displaced volume of a posed body under the plane z = level.
--- pose = { centre = {x,y,z}, ax = {..}, ay = {..}, az = {..} } (the body's own unit axes in the world, right-handed).
--- Returns submerged volume (m^3), centroid {x,y,z}, and the fraction (0..1) of the body below the plane.
function M.submerged(shape, pose, level)
  local v, cen, total
  if shape.kind == "sphere" then
    v, cen, total = M.sphere_submerged(shape.r, pose.centre, level)
  elseif shape.kind == "cylinder" then
    local axis = ({ pose.ax, pose.ay, pose.az })[shape.axis]
    v, cen, total = M.cylinder_submerged(shape.r, shape.L, axis, pose.centre, level)
  else
    v, cen, total = M.box_submerged(shape.dims, pose.ax, pose.ay, pose.az, pose.centre, level)
  end
  local frac = total > 0 and clamp(v / total, 0, 1) or 0
  return frac * shape.volume, cen, frac
end

-- ------------------------------------------------------------------------------------------------ the forces
--- Archimedes: the buoyant force (N).
function M.buoyancy(rho_liquid, g, v_sub) return rho_liquid * g * v_sub end

--- The submerged fraction a free-floating body settles at: rho_body / rho_liquid (1 = it sinks).  Matches the library's
--- fluid_surface.submerged_fraction.
function M.draft_fraction(rho_body, rho_liquid)
  local f = rho_body / rho_liquid
  return f < 1 and f or 1
end

--- Drag coefficients of a body: the deceleration is k2 v^2 + k1 v  (m/s^2).  area in m^2, mass in kg, eta in Pa s.
function M.drag_k(rho, cd, area, mass, eta, area_equiv_length)
  local L = area_equiv_length or sqrt(4 * area / pi)
  local k2 = 0.5 * rho * cd * area / mass
  local k1 = (eta and eta > 0 and L > 0) and (12 * eta * area / (L * mass)) or 0
  return k2, k1
end

--- The drag force itself (N), for the tests against the library.
function M.drag_force(rho, cd, area, speed, eta)
  local L = sqrt(4 * area / pi)
  return 0.5 * rho * cd * area * speed * speed + ((eta and eta > 0) and (12 * eta * area / L * speed) or 0)
end

--- Speed after one step dt of dv/dt = -(k2 v^2 + k1 v), implicit: never reverses, converges as dt -> 0.
function M.speed_after(v, k2, k1, dt)
  return v / (1 + (k2 * v + k1) * dt)
end

--- Terminal speed of a body of density rho_b and volume V falling (or rising, with the sign) through the liquid:
--- (rho_b - rho_l) V g = 1/2 rho_l Cd A v^2 + 12 eta A v / L.   Returns speed (>= 0) and the direction (-1 sinks, +1 rises).
function M.terminal_speed(rho_b, V, rho_l, g, cd, area, eta)
  local net = (rho_b - rho_l) * V * g
  local dir = net > 0 and -1 or 1
  net = abs(net)
  local L = sqrt(4 * area / pi)
  local a = 0.5 * rho_l * cd * area
  local b = (eta and eta > 0) and (12 * eta * area / L) or 0
  if a <= 0 then return (b > 0) and net / b or math.huge, dir end
  return (-b + sqrt(b * b + 4 * a * net)) / (2 * a), dir
end

--- Reynolds number of a body: rho v L / eta with L the equivalent diameter.
function M.reynolds(rho, speed, area, eta)
  return rho * speed * sqrt(4 * area / pi) / eta
end

--- The angular drag coefficient: omega' = omega / (1 + c |omega| dt), c = 1/2 rho Cd A (L/4)^3 / I.
function M.ang_drag_c(rho, cd, area, L, inertia)
  return 0.5 * rho * cd * area * (L / 4) ^ 3 / inertia
end

--- Wave-radiation damping of a body moving up and down through the surface (estimated, small-body limit; see data.lua "radiation"):
--- b = 1/2 rho omega k A_w^2 exp(-2 k T) with omega = sqrt(rho g A_w / m) (its own heave frequency), k = omega^2 / g.
--- Returns the damping per unit mass b/m (1/s).  a_w = waterplane area (m^2), draft = mean immersion depth (m).
function M.heave_radiation_c(rho, g, a_w, draft, mass)
  if g <= 0 or a_w <= 0 or mass <= 0 then return 0 end
  local omega = sqrt(rho * g * a_w / mass)
  local k = omega * omega / g
  local b = 0.5 * rho * omega * k * a_w * a_w * math.exp(-2 * k * draft)
  return b / mass
end

--- Moment of inertia of a solid box of mass m about each of its own axes (kg m^2).
function M.box_inertia(mass, dims)
  local a, b, c = dims[1], dims[2], dims[3]
  return mass * (b * b + c * c) / 12, mass * (a * a + c * c) / 12, mass * (a * a + b * b) / 12
end

return M
