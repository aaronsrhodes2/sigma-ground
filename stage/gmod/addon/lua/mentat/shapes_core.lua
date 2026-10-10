-- shapes_core.lua -- primitive geometry for Mentat bodies: a convex HULL (for
-- Entity:PhysicsInitMultiConvex) and a render TRIANGLE LIST (for clientside IMesh)
-- built from the SAME dimensions, so what you see is what collides.
-- PURE Lua 5.1 / LuaJIT (no GMod globals): offline-testable.
--
-- Primitive conventions mirror sigma_ground.kernel.shapes:
--   Cylinder: radius, FULL height, axis = local +Z, centred at its own origin
--   Box:      FULL extents x, y, z
--   Sphere:   radius
-- Fidelity notes (Phase 0): the cylinder is a 24-gon prism and the sphere is a
-- latitude/longitude polytope, so physics volume is slightly UNDER the analytic
-- value (cylinder ~1.1 %, sphere ~3.6 %; measured in test_gmod_stage.py). Subtract/CSG leaves are not modelled.
local M = {}
local pi, sin, cos, sqrt = math.pi, math.sin, math.cos, math.sqrt

M.CYL_SEGMENTS = 24
M.SPHERE_RINGS = 10      -- latitude rings between the poles
M.SPHERE_SEGMENTS = 20   -- longitude divisions

-- verts is a flat list; every 3 consecutive entries form one triangle.
-- Winding: (b-a) x (c-a) points OUTWARD. Vertex layout {x, y, z, nx, ny, nz}.
local function tri(verts, a, b, c)
  verts[#verts + 1] = a
  verts[#verts + 1] = b
  verts[#verts + 1] = c
end

local function flat(verts, p, q, r, n)
  tri(verts, { p[1], p[2], p[3], n[1], n[2], n[3] },
             { q[1], q[2], q[3], n[1], n[2], n[3] },
             { r[1], r[2], r[3], n[1], n[2], n[3] })
end

--- Cylinder about +Z. Returns { hull = {{x,y,z}...}, verts = {...} } in metres, origin-centred.
function M.cylinder(radius, height, seg)
  seg = seg or M.CYL_SEGMENTS
  local hh = height / 2
  local hull, verts = {}, {}
  local ring = {}
  for i = 0, seg - 1 do
    local a = 2 * pi * i / seg
    ring[i] = { radius * cos(a), radius * sin(a), cos(a), sin(a) }
    hull[#hull + 1] = { ring[i][1], ring[i][2], -hh }
    hull[#hull + 1] = { ring[i][1], ring[i][2], hh }
  end
  for i = 0, seg - 1 do
    local j = (i + 1) % seg
    local p0, p1 = ring[i], ring[j]
    -- side quad (smooth radial normals)
    local v0 = { p0[1], p0[2], -hh, p0[3], p0[4], 0 }
    local v1 = { p1[1], p1[2], -hh, p1[3], p1[4], 0 }
    local v2 = { p1[1], p1[2], hh, p1[3], p1[4], 0 }
    local v3 = { p0[1], p0[2], hh, p0[3], p0[4], 0 }
    tri(verts, v0, v1, v2)
    tri(verts, v0, v2, v3)
    -- top cap (+Z)
    flat(verts, { 0, 0, hh }, { p0[1], p0[2], hh }, { p1[1], p1[2], hh }, { 0, 0, 1 })
    -- bottom cap (-Z)
    flat(verts, { 0, 0, -hh }, { p1[1], p1[2], -hh }, { p0[1], p0[2], -hh }, { 0, 0, -1 })
  end
  return { hull = hull, verts = verts }
end

--- Axis-aligned box with FULL extents (sx, sy, sz).
function M.box(sx, sy, sz)
  local h = { sx / 2, sy / 2, sz / 2 }
  local hull, verts = {}, {}
  for _, x in ipairs({ -h[1], h[1] }) do
    for _, y in ipairs({ -h[2], h[2] }) do
      for _, z in ipairs({ -h[3], h[3] }) do hull[#hull + 1] = { x, y, z } end
    end
  end
  for a = 1, 3 do
    local b, c = a % 3 + 1, (a + 1) % 3 + 1          -- cyclic: e_b x e_c = e_a
    for _, s in ipairs({ 1, -1 }) do
      local function pt(u, v)
        local p = {}
        p[a], p[b], p[c] = s * h[a], u * h[b], v * h[c]
        return p
      end
      local n = { 0, 0, 0 }
      n[a] = s
      local q0, q1, q2, q3 = pt(-1, -1), pt(1, -1), pt(1, 1), pt(-1, 1)
      if s > 0 then
        flat(verts, q0, q1, q2, n); flat(verts, q0, q2, q3, n)
      else
        flat(verts, q0, q2, q1, n); flat(verts, q0, q3, q2, n)
      end
    end
  end
  return { hull = hull, verts = verts }
end

--- Latitude/longitude sphere polytope, poles on +/-Z.
function M.sphere(radius, rings, seg)
  rings, seg = rings or M.SPHERE_RINGS, seg or M.SPHERE_SEGMENTS
  local P = {}                                      -- P[j][i], j = 0..rings+1 (north -> south)
  local hull, verts = {}, {}
  for j = 0, rings + 1 do
    P[j] = {}
    local th = pi * j / (rings + 1)
    local st, ct = sin(th), cos(th)
    if j == 0 or j == rings + 1 then
      local p = { 0, 0, radius * ct, 0, 0, ct }
      for i = 0, seg - 1 do P[j][i] = p end
      hull[#hull + 1] = { p[1], p[2], p[3] }
    else
      for i = 0, seg - 1 do
        local ph = 2 * pi * i / seg
        local nx, ny, nz = st * cos(ph), st * sin(ph), ct
        P[j][i] = { radius * nx, radius * ny, radius * nz, nx, ny, nz }
        hull[#hull + 1] = { radius * nx, radius * ny, radius * nz }
      end
    end
  end
  for j = 0, rings do
    for i = 0, seg - 1 do
      local ip = (i + 1) % seg
      local A, B, C, D = P[j][i], P[j][ip], P[j + 1][ip], P[j + 1][i]
      -- Row j==0: A==B (north pole), so (A,C,B) is degenerate -> keep only (A,D,C).
      -- Row j==rings: C==D (south pole), so (A,D,C) is degenerate -> keep only (A,C,B).
      if j < rings then tri(verts, A, D, C) end
      if j > 0 then tri(verts, A, C, B) end
    end
  end
  return { hull = hull, verts = verts }
end

--- Signed volume enclosed by a closed outward-wound triangle list (divergence theorem).
function M.mesh_volume(verts)
  local v = 0
  for k = 1, #verts, 3 do
    local a, b, c = verts[k], verts[k + 1], verts[k + 2]
    v = v + (a[1] * (b[2] * c[3] - b[3] * c[2])
           - a[2] * (b[1] * c[3] - b[3] * c[1])
           + a[3] * (b[1] * c[2] - b[2] * c[1])) / 6
  end
  return v
end

local BUILDERS = {
  Cylinder = function(s) return M.cylinder(s.radius, s.height) end,
  Box = function(s) return M.box(s.size[1], s.size[2], s.size[3]) end,
  Sphere = function(s) return M.sphere(s.radius) end,
}

local function analytic_volume(s)
  if s.type == "Cylinder" then return pi * s.radius * s.radius * s.height end
  if s.type == "Box" then return s.size[1] * s.size[2] * s.size[3] end
  if s.type == "Sphere" then return 4 / 3 * pi * s.radius ^ 3 end
  return 0
end

--- Build the render + physics geometry of one scenario body, in entity-local HU.
-- body.pivot is subtracted from each shape centre so the entity origin == pivot (CoM).
-- scale = HU per metre. Returns
--   { shapes = { { hull, verts, color, material, surfaceprop, type } ... },
--     mins = {x,y,z}, maxs = {x,y,z}, volume_m3 = analytic volume of the shapes }
function M.build_body(body, scale)
  local piv = body.pivot or { 0, 0, 0 }
  local out = { shapes = {}, volume_m3 = 0 }
  local mins = { math.huge, math.huge, math.huge }
  local maxs = { -math.huge, -math.huge, -math.huge }
  for _, s in ipairs(body.shapes or {}) do
    local builder = BUILDERS[s.type]
    if not builder then error("shapes_core: unsupported shape type '" .. tostring(s.type) .. "'", 2) end
    local prim = builder(s)
    local ox, oy, oz = s.center[1] - piv[1], s.center[2] - piv[2], s.center[3] - piv[3]
    local hull, verts = {}, {}
    for i, p in ipairs(prim.hull) do
      local x, y, z = (p[1] + ox) * scale, (p[2] + oy) * scale, (p[3] + oz) * scale
      hull[i] = { x, y, z }
      if x < mins[1] then mins[1] = x end; if x > maxs[1] then maxs[1] = x end
      if y < mins[2] then mins[2] = y end; if y > maxs[2] then maxs[2] = y end
      if z < mins[3] then mins[3] = z end; if z > maxs[3] then maxs[3] = z end
    end
    for i, v in ipairs(prim.verts) do
      verts[i] = { (v[1] + ox) * scale, (v[2] + oy) * scale, (v[3] + oz) * scale, v[4], v[5], v[6] }
    end
    out.shapes[#out.shapes + 1] = {
      type = s.type, hull = hull, verts = verts,
      color = s.color or { 0.7, 0.7, 0.7 },
      material = s.material, surfaceprop = s.surfaceprop,
    }
    out.volume_m3 = out.volume_m3 + analytic_volume(s)
  end
  out.mins, out.maxs = mins, maxs
  return out
end

return M
