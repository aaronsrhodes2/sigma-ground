-- water/render.lua -- how a water body looks (CLIENT).  Everything is drawn with the mesh and render APIs: no model, no material file, no
-- sound of the game is used (a server and a client without Half-Life 2 or Counter-Strike: Source show it all).  One flat vertex-coloured
-- translucent material of our own, built from the engine's white texture.
--
--   the basin      the tiled floor and walls seen through the water (colour attenuated by the library's absorption, Beer-Lambert), the
--                  rim and the outside of the walls.  A static mesh, rebuilt only when the basin or its level changes.
--   the surface    a grid of a few thousand triangles, rebuilt each frame: height = the sum of a few waves obeying the real dispersion
--                  relation (water/waves.lua; g from the construct) + the ripples of impacts (water/fx.lua); per-vertex colour and
--                  opacity from the Fresnel reflectance (Schlick's approximation with the library's refractive index): clear looking
--                  down, mirror-like at grazing angles, sky-tinted; from below a bright window and total internal reflection past the
--                  critical angle.  Only the part near the camera is gridded on a big body (a window), the rest is flat.
--   the edge       a thin foam line where the surface meets the walls.
--   droplets       small sprites flying real ballistics.
--   underwater     a tint over the screen and distance fog when the eye is below the surface (Beer-Lambert).
--
-- Cost is bounded: at most CELLS_MAX cells (2 triangles each) per body per frame, MAX_BODIES_DRAWN bodies, droplets and rings capped.
-- Everything physical comes from water/data.lua (the library); the colours and the turbidity are labelled "visual" there.
Mentat = Mentat or {}
local M = Mentat.WaterRender or {}                 -- (include re-runs a file for every includer: the state must be shared through one table)
Mentat.WaterRender = M
local inc = include or function(p) return dofile(p) end
local C = inc("mentat/water/constants.lua")
local Basin = inc("mentat/water/basin.lua")
local Waves = inc("mentat/water/waves.lua")
local Splash = inc("mentat/water/splash.lua")
local FxState = inc("mentat/water/fx.lua")
local HU_M = C.HU_M
local sqrt, exp, floor, ceil, min, max, abs, sin, cos, pi = math.sqrt, math.exp, math.floor, math.ceil, math.min, math.max, math.abs, math.sin, math.cos, math.pi

M.CELLS_MAX = 48                  -- cells along one axis of the gridded part of a surface
M.TARGET_CELL_M = 0.35            -- the cell size wanted on a small body
M.WINDOW_M = 40                   -- on a big body, the gridded window around the camera is this wide
M.MAX_BODIES_DRAWN = 4
M.TURBIDITY_PER_M = 0.15          -- visual: scattering that gives the water some body (the library's absorption alone is nearly clear)
M.DRAW_DISTANCE_HU = 9000

local O = C.data.optics
local N_WATER = O.refractive_index.value
local R0 = O.fresnel_r0.value
local ABS = O.absorption_rgb_per_m.value
local VIS = C.data.visual
local SKY = VIS.sky_rgb
local TILE = VIS.tile_rgb
local FOAM = VIS.foam_rgb
local SIN_CRIT = 1 / N_WATER                                  -- total internal reflection past this (Snell)
M.n_water, M.r0 = N_WATER, R0

-- ------------------------------------------------------------------------------------------------ pure helpers (tested offline)
--- Schlick's approximation of the Fresnel reflectance at the cosine of the angle of incidence.
function M.fresnel(cos_theta)
  local c = cos_theta < 0 and 0 or (cos_theta > 1 and 1 or cos_theta)
  local m = 1 - c
  return R0 + (1 - R0) * m * m * m * m * m
end

--- Beer-Lambert transmission through a path of `d` metres, per channel (0..1).
function M.transmit(d)
  return exp(-ABS[1] * d), exp(-ABS[2] * d), exp(-ABS[3] * d)
end

--- Distance (HU) at which the view through water falls to 5 percent: absorption (the green channel) plus the visual turbidity.
function M.visible_distance_hu()
  return 3 / (ABS[2] + M.TURBIDITY_PER_M) / HU_M
end

--- The colour (r, g, b 0..255) and opacity (0..1) of one vertex of the surface seen from above (eye above the water).
--- nx, ny = the slope (dh/dx, dh/dy) of the surface there; vx, vy, vz the unit vector from the surface point to the eye; depth_m the water depth below.
function M.surface_above(nx, ny, vx, vy, vz, depth_m, sun)
  local inv = 1 / sqrt(nx * nx + ny * ny + 1)
  local ax, ay, az = -nx * inv, -ny * inv, inv                  -- the surface normal
  local cosi = ax * vx + ay * vy + az * vz
  local R = M.fresnel(cosi)
  -- sun glint: Blinn-Phong highlight of the normal against the half vector
  local hx, hy, hz = vx + sun[1], vy + sun[2], vz + sun[3]
  local hl = sqrt(hx * hx + hy * hy + hz * hz)
  local nh = (ax * hx + ay * hy + az * hz) / hl
  local glint = nh > 0 and nh ^ 300 or 0
  local turb = 1 - exp(-M.TURBIDITY_PER_M * depth_m * 0.5)
  local alpha = R + (1 - R) * turb * 0.35
  local r, g, b = SKY[1], SKY[2], SKY[3]
  -- the water body colour (the light that went down and came back): the sky through the library's absorption over a few metres
  local tr, tg, tb = M.transmit(min(depth_m, 6) * 2)
  local k = (1 - R) * turb * 0.35 / max(alpha, 1e-6)           -- how much of the colour is the body colour rather than the reflected sky
  r = r * (1 - k) + SKY[1] * tr * 0.55 * k
  g = g * (1 - k) + SKY[2] * tg * 0.7 * k
  b = b * (1 - k) + SKY[3] * tb * 0.8 * k
  r, g, b = r * (1 - glint) + 255 * glint, g * (1 - glint) + 255 * glint, b * (1 - glint) + 255 * glint
  alpha = max(alpha, glint)
  return r, g, b, alpha < 0 and 0 or (alpha > 0.97 and 0.97 or alpha)
end

--- The same seen from below: a bright window straight up, total internal reflection (a dark mirror) past the critical angle.
function M.surface_below(nx, ny, vx, vy, vz)
  local inv = 1 / sqrt(nx * nx + ny * ny + 1)
  local ax, ay, az = -nx * inv, -ny * inv, inv
  local cosi = abs(ax * vx + ay * vy + az * vz)               -- angle from the normal, seen from underneath
  local sini = sqrt(max(0, 1 - cosi * cosi))
  if sini >= SIN_CRIT then
    return 20, 70, 95, 0.78                                   -- total internal reflection
  end
  local R = M.fresnel(cosi)
  return SKY[1] * 0.9, SKY[2] * 0.95, SKY[3], 0.25 + 0.4 * R
end

--- Colour of a basin surface point at `depth_m` under the water (0 = dry): the tile colour through the library's absorption, there and back.
function M.tile_colour(depth_m, shade)
  shade = shade or 1
  if depth_m <= 0 then return TILE[1] * shade, TILE[2] * shade, TILE[3] * shade end
  local tr, tg, tb = M.transmit(depth_m * 2)
  local tur = exp(-M.TURBIDITY_PER_M * depth_m * 0.6)         -- the visual turbidity fades the bottom toward the body colour
  local br, bg, bb = SKY[1] * 0.25, SKY[2] * 0.4, SKY[3] * 0.5
  return (TILE[1] * tr * tur + br * (1 - tur)) * shade, (TILE[2] * tg * tur + bg * (1 - tur)) * shade, (TILE[3] * tb * tur + bb * (1 - tur)) * shade
end

--- Cell counts and the window for a rectangular surface of half sizes hx, hy (HU) seen from local eye (ex, ey): returns x0, x1, y0, y1 (HU), nx, ny.
function M.grid_window(hx, hy, ex, ey)
  local cell_hu = M.TARGET_CELL_M / HU_M
  local win_hu = M.WINDOW_M / HU_M
  local x0, x1, y0, y1 = -hx, hx, -hy, hy
  if 2 * hx > win_hu then x0 = max(-hx, ex - win_hu / 2); x1 = min(hx, x0 + win_hu); x0 = max(-hx, x1 - win_hu) end
  if 2 * hy > win_hu then y0 = max(-hy, ey - win_hu / 2); y1 = min(hy, y0 + win_hu); y0 = max(-hy, y1 - win_hu) end
  local nx = max(2, min(M.CELLS_MAX, ceil((x1 - x0) / cell_hu)))
  local ny = max(2, min(M.CELLS_MAX, ceil((y1 - y0) / cell_hu)))
  return x0, x1, y0, y1, nx, ny
end

-- ------------------------------------------------------------------------------------------------ materials
M.mats = {}
local function material(name, translucent, additive)
  local m = M.mats[name]
  if m then return m end
  local params = { ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 1, ["$nocull"] = 1 }
  if translucent then params["$translucent"] = 1 end
  if additive then params["$additive"] = 1 end
  m = CreateMaterial("mentat_water_" .. name .. "_v1", "UnlitGeneric", params)
  M.mats[name] = m
  return m
end

-- ------------------------------------------------------------------------------------------------ the basin (a static mesh)
local function emit_tri(a, b, c, ca, cb, cc)
  mesh.Position(a); mesh.Color(ca[1], ca[2], ca[3], 255); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
  mesh.Position(b); mesh.Color(cb[1], cb[2], cb[3], 255); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
  mesh.Position(c); mesh.Color(cc[1], cc[2], cc[3], 255); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
end

local function clamp255(x) return x < 0 and 0 or (x > 255 and 255 or floor(x + 0.5)) end

--- All the triangles of the basin as a list { {x,y,z}, {x,y,z}, {x,y,z}, {r,g,b}, {r,g,b}, {r,g,b} } in local HU.  Pure geometry (tested offline).
function M.basin_triangles(b, fill)
  local tris = {}
  local zs = Basin.surface_z(b, fill)
  local depth_m_at = function(z) return max(0, (zs - z) * HU_M) end
  local function col(z, shade)
    local r, g, bl = M.tile_colour(fill > 0 and depth_m_at(z) or 0, shade)
    return { clamp255(r), clamp255(g), clamp255(bl) }
  end
  local function quad(p1, p2, p3, p4, shade)
    -- p = { x, y, z }; colours by each vertex's own depth
    local c1, c2, c3, c4 = col(p1[3], shade), col(p2[3], shade), col(p3[3], shade), col(p4[3], shade)
    tris[#tris + 1] = { p1, p2, p3, c1, c2, c3 }
    tris[#tris + 1] = { p1, p3, p4, c1, c3, c4 }
  end
  local hx, hy, w, fz, top = b.hx, b.hy, b.wall, b.floor_z, b.top_z
  local strips = 8
  local function wall_strips(pa, pb, outward_shade)
    -- a vertical wall from pa to pb (x, y), from the floor to the rim, cut into strips so the depth tint has a gradient
    for s = 0, strips - 1 do
      local z0, z1 = fz + (top - fz) * s / strips, fz + (top - fz) * (s + 1) / strips
      quad({ pa[1], pa[2], z0 }, { pb[1], pb[2], z0 }, { pb[1], pb[2], z1 }, { pa[1], pa[2], z1 }, outward_shade)
    end
  end
  if b.shape == "rect" then
    -- the floor in tiles
    local tile_hu = max(1 / HU_M, max(hx, hy) * 2 / 24)
    local nx, ny = max(1, ceil(2 * hx / tile_hu)), max(1, ceil(2 * hy / tile_hu))
    for i = 0, nx - 1 do
      for j = 0, ny - 1 do
        local x0, x1 = -hx + 2 * hx * i / nx, -hx + 2 * hx * (i + 1) / nx
        local y0, y1 = -hy + 2 * hy * j / ny, -hy + 2 * hy * (j + 1) / ny
        local shade = ((i + j) % 2 == 0) and 1.0 or 0.9
        quad({ x0, y0, fz }, { x1, y0, fz }, { x1, y1, fz }, { x0, y1, fz }, shade)
      end
    end
    wall_strips({ -hx, -hy }, { hx, -hy }, 0.85)                -- inner walls
    wall_strips({ hx, -hy }, { hx, hy }, 0.75)
    wall_strips({ hx, hy }, { -hx, hy }, 0.85)
    wall_strips({ -hx, hy }, { -hx, -hy }, 0.75)
    -- the rim and the outside
    local X, Y = hx + w, hy + w
    quad({ -X, -Y, top }, { X, -Y, top }, { X, -hy, top }, { -X, -hy, top }, 1.0)
    quad({ -X, hy, top }, { X, hy, top }, { X, Y, top }, { -X, Y, top }, 1.0)
    quad({ -X, -hy, top }, { -hx, -hy, top }, { -hx, hy, top }, { -X, hy, top }, 1.0)
    quad({ hx, -hy, top }, { X, -hy, top }, { X, hy, top }, { hx, hy, top }, 1.0)
    quad({ -X, -Y, 0 }, { X, -Y, 0 }, { X, -Y, top }, { -X, -Y, top }, 0.8)
    quad({ X, -Y, 0 }, { X, Y, 0 }, { X, Y, top }, { X, -Y, top }, 0.7)
    quad({ X, Y, 0 }, { -X, Y, 0 }, { -X, Y, top }, { X, Y, top }, 0.8)
    quad({ -X, Y, 0 }, { -X, -Y, 0 }, { -X, -Y, top }, { -X, Y, top }, 0.7)
  else
    local n = 40
    local function pt(r_x, r_y, a, z) return { r_x * cos(a), r_y * sin(a), z } end
    for i = 0, n - 1 do
      local a0, a1 = 2 * pi * i / n, 2 * pi * (i + 1) / n
      local shade = (i % 2 == 0) and 1.0 or 0.93
      -- floor wedge
      local cen = { 0, 0, fz }
      local p0, p1 = pt(hx, hy, a0, fz), pt(hx, hy, a1, fz)
      local cc, c0, c1 = col(fz, shade), col(fz, shade), col(fz, shade)
      tris[#tris + 1] = { cen, p0, p1, cc, c0, c1 }
      wall_strips({ hx * cos(a0), hy * sin(a0) }, { hx * cos(a1), hy * sin(a1) }, 0.8)
      -- rim and outside
      local q0, q1 = pt(hx + w, hy + w, a0, top), pt(hx + w, hy + w, a1, top)
      local i0, i1 = pt(hx, hy, a0, top), pt(hx, hy, a1, top)
      quad(i0, i1, q1, q0, 1.0)
      quad(pt(hx + w, hy + w, a0, 0), pt(hx + w, hy + w, a1, 0), q1, q0, 0.75)
    end
  end
  return tris
end

local function build_basin_mesh(b, fill)
  local tris = M.basin_triangles(b, fill)
  local m = Mesh()
  mesh.Begin(m, MATERIAL_TRIANGLES, #tris)
  for _, t in ipairs(tris) do
    emit_tri(Vector(t[1][1], t[1][2], t[1][3]), Vector(t[2][1], t[2][2], t[2][3]), Vector(t[3][1], t[3][2], t[3][3]), t[4], t[5], t[6])
  end
  mesh.End()
  return m, #tris
end

-- ------------------------------------------------------------------------------------------------ per-body client state
M.bodies = M.bodies or setmetatable({}, { __mode = "k" })      -- ent -> { fx = FxState, basin_mesh, seq, fill, comps }
M.stats = M.stats or { draw_ms = 0, tris = 0, frames = 0, worst_ms = 0, bodies = 0 }
local function state_of(ent)
  local s = M.bodies[ent]
  if not s then s = { fx = FxState.new(), seq = -1, t_last = 0 }; M.bodies[ent] = s end
  return s
end
M.state_of = state_of

--- The wave components for a body at the construct's current g.
local function components_for(st, rec, g, cell_m, side_m)
  local key = string.format("%.4f|%.3f|%.3f|%s", g, rec.b.depth_m, cell_m, rec.liquid)
  if st.comps_key == key then return st.comps end
  local L = rec.L
  local sigma_over_rho = L.surface_tension_n_m / L.density_kg_m3
  local lam_min = max(0.9, 2.6 * cell_m)
  local steep = C.data.waves.steepness.value
  st.comps = Waves.components(max(g, 1e-6), rec.b.depth_m * max(rec.fill, 0.05), sigma_over_rho, lam_min, side_m, steep, 4, 1.7, 0.5)
  st.comps_key = key
  return st.comps
end

-- ------------------------------------------------------------------------------------------------ the surface
local SUN = { 0.35, 0.25, 0.9 }
do local l = sqrt(SUN[1] ^ 2 + SUN[2] ^ 2 + SUN[3] ^ 2); SUN[1], SUN[2], SUN[3] = SUN[1] / l, SUN[2] / l, SUN[3] / l end

--- Draw the wavy surface.  `rec` is the volumes.lua record, `eye` the local eye position (HU), t the clock, g the construct's gravity (m/s2).
--- Returns the triangle count.  All positions are local to the entity (a model matrix is pushed by the caller).
function M.draw_surface(ent, rec, st, eye, t, g, below)
  local b = rec.b
  local zs = Basin.surface_z(b, rec.fill)
  if rec.fill <= 0 then return 0 end
  local hx, hy = b.hx, b.hy
  local x0, x1, y0, y1, nx, ny = M.grid_window(hx, hy, eye.x, eye.y)
  if b.shape == "round" then x0, x1, y0, y1 = -hx, hx, -hy, hy end
  local cell_m = max((x1 - x0) / nx, (y1 - y0) / ny) * HU_M
  local comps = components_for(st, rec, g, cell_m, min(2 * hx, 2 * hy) * HU_M)
  local fx = st.fx
  local depth_m = (zs - b.floor_z) * HU_M
  local still = g <= 0                                           -- free fall: no restoring force, no waves
  local ex, ey, ez = eye.x, eye.y, eye.z
  local tris = 0

  local function vertex(x, y)
    local xm, ym = x * HU_M, y * HU_M
    local h, sx, sy = 0, 0, 0
    if not still then h, sx, sy = Waves.height_slope(comps, xm, ym, t) end
    local hr = FxState.height(fx, rec.wx + xm * rec.c - ym * rec.s, rec.wy + xm * rec.s + ym * rec.c, t)
    h = h + hr
    local z = zs + h / HU_M
    local vx, vy, vz = ex - x, ey - y, ez - z
    local vl = sqrt(vx * vx + vy * vy + vz * vz)
    if vl < 1e-6 then vl = 1e-6 end
    vx, vy, vz = vx / vl, vy / vl, vz / vl
    local r, gg, bl, a
    if below then r, gg, bl, a = M.surface_below(sx, sy, vx, vy, vz)
    else r, gg, bl, a = M.surface_above(sx, sy, vx, vy, vz, depth_m, SUN) end
    return x, y, z, clamp255(r), clamp255(gg), clamp255(bl), clamp255(a * 255)
  end

  render.SetMaterial(material("surface", true))
  if b.shape == "rect" then
    local n_tri = nx * ny * 2
    mesh.Begin(MATERIAL_TRIANGLES, n_tri)
    local prev = {}
    for j = 0, ny do
      local y = y0 + (y1 - y0) * j / ny
      local row = {}
      for i = 0, nx do
        local x = x0 + (x1 - x0) * i / nx
        row[i] = { vertex(x, y) }
      end
      if j > 0 then
        for i = 0, nx - 1 do
          local a, bb, c, d = prev[i], prev[i + 1], row[i + 1], row[i]
          for _, v in ipairs({ a, bb, c, a, c, d }) do
            mesh.Position(Vector(v[1], v[2], v[3])); mesh.Color(v[4], v[5], v[6], v[7]); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
          end
        end
      end
      prev = row
    end
    mesh.End()
    tris = n_tri
    -- the flat remainder of a big body, outside the window
    local function flat(xa, xb, ya, yb)
      if xb - xa < 1e-3 or yb - ya < 1e-3 then return end
      local r, gg, bl, a = M.surface_above(0, 0, 0.3, 0.3, 0.9, depth_m, SUN)
      mesh.Begin(MATERIAL_QUADS, 1)
      for _, p in ipairs({ { xa, ya }, { xb, ya }, { xb, yb }, { xa, yb } }) do
        mesh.Position(Vector(p[1], p[2], zs)); mesh.Color(clamp255(r), clamp255(gg), clamp255(bl), clamp255(a * 255)); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
      end
      mesh.End()
      tris = tris + 2
    end
    flat(-hx, x0, -hy, hy); flat(x1, hx, -hy, hy); flat(x0, x1, -hy, y0); flat(x0, x1, y1, hy)
  else
    local rings, segs = 14, 44
    mesh.Begin(MATERIAL_TRIANGLES, segs * (2 * rings - 1))
    local function pv(r_frac, a) return { vertex(hx * r_frac * cos(a), hy * r_frac * sin(a)) } end
    for s = 0, segs - 1 do
      local a0, a1 = 2 * pi * s / segs, 2 * pi * (s + 1) / segs
      for r = 0, rings - 1 do
        local f0, f1 = r / rings, (r + 1) / rings
        local list
        if r == 0 then list = { pv(0, a0), pv(f1, a0), pv(f1, a1) }
        else list = { pv(f0, a0), pv(f1, a0), pv(f1, a1), pv(f0, a0), pv(f1, a1), pv(f0, a1) } end
        for _, v in ipairs(list) do
          mesh.Position(Vector(v[1], v[2], v[3])); mesh.Color(v[4], v[5], v[6], v[7]); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
        end
      end
    end
    mesh.End()
    tris = segs * (2 * rings - 1)
  end
  return tris
end

--- The foam line where the surface meets the walls.
function M.draw_foam(rec, t)
  local b = rec.b
  if rec.fill <= 0 then return 0 end
  local zs = Basin.surface_z(b, rec.fill) + 0.15
  local wdt = 0.045 / HU_M
  local hx, hy = b.hx, b.hy
  render.SetMaterial(material("foam", true))
  local n = 0
  local function strip(x0, y0, x1, y1, nxn, nyn, phase)
    local pulse = 0.55 + 0.25 * sin(t * 1.3 + phase)
    local a = clamp255(255 * pulse)
    mesh.Begin(MATERIAL_QUADS, 1)
    for _, p in ipairs({ { x0, y0 }, { x1, y1 }, { x1 - nxn * wdt, y1 - nyn * wdt }, { x0 - nxn * wdt, y0 - nyn * wdt } }) do
      mesh.Position(Vector(p[1], p[2], zs)); mesh.Color(FOAM[1], FOAM[2], FOAM[3], a); mesh.TexCoord(0, 0, 0); mesh.AdvanceVertex()
    end
    mesh.End()
    n = n + 2
  end
  if b.shape == "rect" then
    strip(-hx, -hy, hx, -hy, 0, -1, 0); strip(hx, -hy, hx, hy, 1, 0, 1.5); strip(hx, hy, -hx, hy, 0, 1, 3); strip(-hx, hy, -hx, -hy, -1, 0, 4.5)
  else
    local segs = 36
    for i = 0, segs - 1 do
      local a0, a1 = 2 * pi * i / segs, 2 * pi * (i + 1) / segs
      local x0, y0, x1, y1 = hx * cos(a0), hy * sin(a0), hx * cos(a1), hy * sin(a1)
      strip(x0, y0, x1, y1, cos((a0 + a1) / 2), sin((a0 + a1) / 2), i * 0.4)
    end
  end
  return n
end

--- Droplets (sprites) of one body.  Positions are in world metres relative to the surface (z above it).
function M.draw_drops(rec, st)
  local drops = st.fx.drops
  if #drops == 0 then return 0 end
  render.SetMaterial(material("drops", true, true))
  local zs_world = rec.surf
  local col = Color(220, 240, 255, 200)
  local n = 0
  for _, d in ipairs(drops) do
    local sz = max(1.2, d.r * 2 / HU_M)
    render.DrawSprite(Vector(d.x / HU_M, d.y / HU_M, zs_world + d.z / HU_M), sz, sz, col)
    n = n + 1
  end
  return n
end

-- is the world point below the surface and inside the footprint of this body (for the underwater test)?
local function eye_inside(rec, p)
  local dx, dy = p.x - rec.x, p.y - rec.y
  local lx, ly = dx * rec.c + dy * rec.s, -dx * rec.s + dy * rec.c
  local b = rec.b
  local inside
  if b.shape == "rect" then inside = abs(lx) <= b.hx and abs(ly) <= b.hy else inside = (lx / b.hx) ^ 2 + (ly / b.hy) ^ 2 <= 1 end
  return inside and p.z <= rec.surf and p.z >= rec.floor
end
M.eye_inside = eye_inside

-- ------------------------------------------------------------------------------------------------ one body, all of it
--- Draw a whole water body.  Called from the entity's DrawTranslucent.
function M.draw_body(ent, rec, eye_world, t, g)
  local t0 = SysTime()
  local st = state_of(ent)
  local b = rec.b
  -- the basin (static)
  if st.seq ~= ent:GetSeq() or st.fill ~= rec.fill or not st.basin_mesh then
    if st.basin_mesh and st.basin_mesh.Destroy then st.basin_mesh:Destroy() end
    st.basin_mesh, st.basin_tris = build_basin_mesh(b, rec.fill)
    st.seq, st.fill = ent:GetSeq(), rec.fill
  end
  rec.wx, rec.wy = rec.x * HU_M, rec.y * HU_M
  local eye_local = ent:WorldToLocal(eye_world)
  local below = rec.fill > 0 and eye_inside(rec, eye_world)
  local tris = st.basin_tris or 0
  cam.PushModelMatrix(ent:GetWorldTransformMatrix())
  render.SetMaterial(material("basin", false))
  st.basin_mesh:Draw()
  tris = tris + M.draw_surface(ent, rec, st, eye_local, t, g, below)
  tris = tris + M.draw_foam(rec, t)
  cam.PopModelMatrix()
  tris = tris + M.draw_drops(rec, st)
  local ms = (SysTime() - t0) * 1000
  local s = M.stats
  s.frames = s.frames + 1
  s.draw_ms = s.draw_ms * 0.95 + ms * 0.05
  s.worst_ms = max(s.worst_ms * 0.999, ms)
  s.tris = tris
  return tris
end

-- ------------------------------------------------------------------------------------------------ underwater: tint and fog
--- The screen tint (r, g, b, a 0..255) and the fog (r, g, b, end_hu) for an eye `depth_m` under the surface.
function M.underwater_look(depth_m)
  local tr, tg, tb = M.transmit(5)
  local r, g, b = SKY[1] * tr * 0.55, SKY[2] * tg * 0.7, SKY[3] * tb * 0.8
  local a = 0.30 + 0.30 * (1 - exp(-0.5 * max(depth_m, 0)))
  return clamp255(r), clamp255(g), clamp255(b), clamp255(a * 255), M.visible_distance_hu()
end

return M
