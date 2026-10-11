-- avatar/geometry.lua -- Skippy's body as triangles, in plain Lua: a personified floating beer can wearing a giant Napoleonic
-- admiral's bicorn. No model, no texture, no file of any kind: the client turns these triangles into a mesh (avatar_client.lua),
-- so it looks the same on a server with no Half-Life 2 content.
--
-- Units: Source units (HU). The origin is the middle of the can; +x is the way he faces; +z is up. A can 0.5 m tall (26 HU) is a
-- believable can scaled up so it reads from across a room; the hat is much wider than the can, as an admiral's is.
--
-- build() returns { parts = { can =, label =, face =, mouth =, hat = }, pivots = {...} }: each part is a list of triangles
--   { a = {x,y,z}, b =, c =,  n = {x,y,z} unit face normal,  color = {r,g,b},  kind = "side" | "up" | "down" | "inner" | "decor" }
-- wound counter-clockwise seen from outside (the client draws two-sided, so the engine's winding rule cannot hide a face).
-- Parts move separately: the label glows with his voice, the mouth opens, the hat tilts about pivots.hat.
local G = {}

G.HU_PER_M = 52.49
G.CAN_H = 26                    -- HU tall (about 0.5 m)
G.CAN_R = 7.2                   -- HU radius (a can is about 1.8 : 1, so it is slim)
G.HAT_HALF_WIDTH = 23.5         -- HU: the hat is 47 HU (0.9 m) from point to point
G.PIVOT_HAT = { x = 0, y = 0, z = 12 }
G.PIVOT_MOUTH = { x = 7.3, y = 0, z = -4.2 }

-- the project palette (HudPalette.kt / ChatPalette.kt) and a few can colours
G.COLORS = {
  green = { 0, 255, 136 },      -- #00FF88  label band and glow
  amber = { 255, 170, 0 },      -- #FFAA00  hat trim
  cyan = { 0, 204, 255 },       -- #00CCFF
  red = { 255, 68, 68 },        -- #FF4444
  white = { 255, 255, 255 },
  silver = { 176, 184, 192 },   -- the can, a little dim
  silver_dark = { 112, 120, 130 },
  hat = { 34, 38, 58 },         -- dark felt (a little blue so it reads against the sky and the grass)
  ink = { 12, 12, 20 },
}
local C = G.COLORS

local function sub(a, b) return { x = a.x - b.x, y = a.y - b.y, z = a.z - b.z } end
local function cross(a, b) return { x = a.y * b.z - a.z * b.y, y = a.z * b.x - a.x * b.z, z = a.x * b.y - a.y * b.x } end
local function len(v) return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) end

--- Face normal (unit) of a triangle by its winding, and its area.
function G.normal(a, b, c)
  local n = cross(sub(b, a), sub(c, a))
  local l = len(n)
  if l < 1e-12 then return { x = 0, y = 0, z = 1 }, 0 end
  return { x = n.x / l, y = n.y / l, z = n.z / l }, l / 2
end

local function P(x, y, z) return { x = x, y = y, z = z } end

-- a builder for one part
local function new_part() return { tris = {}, pivot = nil } end

--- Add a triangle (skipped when degenerate). `ref`: a point inside the solid; when given, the triangle is wound to face away from it.
local function tri(part, a, b, c, color, kind, ref)
  local n, area = G.normal(a, b, c)
  if area < 1e-6 then return end
  if ref then
    local m = P((a.x + b.x + c.x) / 3 - ref.x, (a.y + b.y + c.y) / 3 - ref.y, (a.z + b.z + c.z) / 3 - ref.z)
    if n.x * m.x + n.y * m.y + n.z * m.z < 0 then b, c = c, b; n = G.normal(a, b, c) end
  end
  part.tris[#part.tris + 1] = { a = a, b = b, c = c, n = n, color = color, kind = kind or "side" }
end

local function quad(part, a, b, c, d, color, kind, ref)
  tri(part, a, b, c, color, kind, ref)
  tri(part, a, c, d, color, kind, ref)
end

local function ring_pt(i, N, r, z)
  local t = 2 * math.pi * (i % N) / N
  return P(r * math.cos(t), r * math.sin(t), z)
end

-- a band of the cylinder / cone between (z0, r0) and (z1, r1), z1 > z0, facing out
local function band(part, N, z0, r0, z1, r1, color, kind)
  for i = 0, N - 1 do
    local l0, l1 = ring_pt(i, N, r0, z0), ring_pt(i + 1, N, r0, z0)
    local u0, u1 = ring_pt(i, N, r1, z1), ring_pt(i + 1, N, r1, z1)
    tri(part, l0, l1, u1, color, kind or "side")
    tri(part, l0, u1, u0, color, kind or "side")
  end
end

-- a flat disc at z (faces up when `up`)
local function disc(part, N, z, r, color, up)
  local c = P(0, 0, z)
  for i = 0, N - 1 do
    local a, b = ring_pt(i, N, r, z), ring_pt(i + 1, N, r, z)
    if up then tri(part, c, a, b, color, "up") else tri(part, c, b, a, color, "down") end
  end
end

-- a flat ring between radii at z, facing up
local function annulus(part, N, z, r_in, r_out, color)
  for i = 0, N - 1 do
    local i0, i1 = ring_pt(i, N, r_in, z), ring_pt(i + 1, N, r_in, z)
    local o0, o1 = ring_pt(i, N, r_out, z), ring_pt(i + 1, N, r_out, z)
    tri(part, i0, o0, o1, color, "up")
    tri(part, i0, o1, i1, color, "up")
  end
end

-- an axis-aligned box, every face wound outward
local function box(part, x0, x1, y0, y1, z0, z1, color, kind)
  kind = kind or "decor"
  quad(part, P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1), color, kind)       -- +x
  quad(part, P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0), color, kind)       -- -x
  quad(part, P(x1, y1, z0), P(x0, y1, z0), P(x0, y1, z1), P(x1, y1, z1), color, kind)       -- +y
  quad(part, P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1), color, kind)       -- -y
  quad(part, P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1), color, kind)       -- +z
  quad(part, P(x0, y0, z0), P(x0, y1, z0), P(x1, y1, z0), P(x1, y0, z0), color, kind)       -- -z
end

-- a small disc standing in the y-z plane, facing +x, centred at (x, y, z)
local function face_disc(part, N, x, y, z, r, color)
  local c = P(x, y, z)
  for i = 0, N - 1 do
    local t0, t1 = 2 * math.pi * i / N, 2 * math.pi * (i + 1) / N
    tri(part, c, P(x, y + r * math.cos(t0), z + r * math.sin(t0)), P(x, y + r * math.cos(t1), z + r * math.sin(t1)), color, "decor")
  end
end

-- ------------------------------------------------------------------------------------------------ the can
local function build_can(can, label)
  local N, R, h = 24, G.CAN_R, G.CAN_H / 2
  local zb, zt = -h, h
  -- bottom: a flat base, then the chime flaring out to the body
  disc(can, N, zb, R * 0.86, C.silver_dark, false)
  band(can, N, zb, R * 0.86, zb + 1.2, R, C.silver, "side")
  -- body: silver below, the label band, silver above
  local lz0, lz1 = -6.8, 4.8
  band(can, N, zb + 1.2, R, lz0, R, C.silver, "side")
  band(label, N, lz0, R * 1.012, lz1, R * 1.012, C.green, "side")
  band(can, N, lz1, R, 8.4, R, C.silver, "side")
  -- shoulder, neck, lip
  band(can, N, 8.4, R, 11.2, R * 0.86, C.silver, "side")
  band(can, N, 11.2, R * 0.86, 12.0, R * 0.82, C.silver, "side")
  band(can, N, 12.0, R * 0.82, zt, R * 0.82, C.silver_dark, "side")
  annulus(can, N, zt, R * 0.72, R * 0.82, C.silver_dark)
  -- the lid sits inside the lip: its wall faces the axis
  for i = 0, N - 1 do
    local l0, l1 = ring_pt(i, N, R * 0.72, 12.2), ring_pt(i + 1, N, R * 0.72, 12.2)
    local u0, u1 = ring_pt(i, N, R * 0.72, zt), ring_pt(i + 1, N, R * 0.72, zt)
    tri(can, l0, u1, l1, C.silver_dark, "inner")
    tri(can, l0, u0, u1, C.silver_dark, "inner")
  end
  disc(can, N, 12.2, R * 0.72, C.silver, true)
  -- pull tab: a flat tongue with a darker rivet and finger hole
  box(can, -1.0, 4.2, -1.3, 1.3, 12.2, 12.7, C.silver_dark, "decor")
  box(can, 2.4, 3.8, -0.6, 0.6, 12.7, 12.72, C.ink, "decor")
  box(can, -1.2, -0.2, -0.5, 0.5, 12.7, 12.8, C.silver, "decor")
end

-- ------------------------------------------------------------------------------------------------ the face
local function build_face(face, mouth)
  local x = G.CAN_R * 1.012                 -- the label band is at 1.012 R: the face must stand in front of it, not inside it
  for _, y in ipairs({ -3.0, 3.0 }) do
    face_disc(face, 14, x + 0.15, y, 0.6, 2.3, C.white)
    face_disc(face, 12, x + 0.3, y, 0.6, 1.2, C.ink)
  end
  -- a slight scowl: the knowing look of a being much cleverer than the room
  box(face, x + 0.1, x + 0.5, -5.0, -1.2, 3.4, 4.0, C.ink, "decor")
  box(face, x + 0.1, x + 0.5, 1.2, 5.0, 3.4, 4.0, C.ink, "decor")
  -- the mouth: the client scales it with his voice
  local c = G.PIVOT_MOUTH
  box(mouth, c.x - 0.1, c.x + 0.5, -3.0, 3.0, c.z - 0.5, c.z + 0.5, C.amber, "decor")
  mouth.pivot = c
end

-- ------------------------------------------------------------------------------------------------ the hat
-- A bicorn worn athwartships: a long thin crescent whose two points rise at the left and right, folded from a dark felt body with an
-- amber braid along its edges and a tricolour cockade pinned at the front. Built as a lens-shaped tube along y: at each station
-- the cross-section is an ellipse in (x, z), narrowing to a point at each end while the centre line curves up.
function G.hat_station(u)
  local W = G.HAT_HALF_WIDTH
  local au = math.abs(u)
  local d = 11.0 * (1 - u * u) ^ 0.7                 -- half depth (front to back)
  local t = 8.0 * (1 - u * u) ^ 0.5                  -- half height of the cross-section
  local zc = 12.0 + 11.0 * au ^ 2.2                  -- the centre line, curling up at both ends; low in the middle so the crown comes down over the can
  return W * u, d, t, zc
end
local hat_station = G.hat_station

local function build_hat(hat)
  local NU, NR = 22, 16
  hat.pivot = G.PIVOT_HAT
  local rings = {}
  for i = 0, NU do
    local u = -1 + 2 * i / NU
    local y, d, t, zc = hat_station(u)
    local r = { centre = P(0, y, zc), pts = {} }
    for j = 0, NR - 1 do
      local th = 2 * math.pi * j / NR
      r.pts[j] = P(d * math.cos(th), y, zc + t * math.sin(th))
    end
    rings[i] = r
  end
  for i = 0, NU - 1 do
    local a, b = rings[i], rings[i + 1]
    local tip = (i == 0 or i == NU - 1)
    for j = 0, NR - 1 do
      local j1 = (j + 1) % NR
      -- gold lace along the top crest and the lower edge (the folded brim), and at the two points; the faces stay dark felt
      local edge = (j == NR / 4 - 1 or j == NR / 4 or j == 3 * NR / 4 - 1 or j == 3 * NR / 4)
      local color = (edge or tip) and C.amber or C.hat
      local ref = P(0, (a.centre.y + b.centre.y) / 2, (a.centre.z + b.centre.z) / 2)
      quad(hat, a.pts[j], a.pts[j1], b.pts[j1], b.pts[j], color, "side", ref)
    end
  end
  -- the cockade on the front, a little off the middle, standing on the front edge: blue, white, red from the outside in
  local R_COCK = 5.4
  local y, d, _, zc = hat_station(0.38)
  local _, d_in = hat_station(0.38 - R_COCK / G.HAT_HALF_WIDTH)            -- the hat is deeper nearer the middle: stand clear of its fullest part under the disc
  d = math.max(d, d_in) + 0.2
  face_disc(hat, 14, d + 0.15, y, zc, 5.4, C.cyan)
  face_disc(hat, 14, d + 0.3, y, zc, 3.7, C.white)
  face_disc(hat, 14, d + 0.45, y, zc, 2.0, C.red)
end

-- ------------------------------------------------------------------------------------------------ the whole
local built

--- The triangles of every part (built once).
function G.build()
  if built then return built end
  local can, label, face, mouth, hat = new_part(), new_part(), new_part(), new_part(), new_part()
  build_can(can, label)
  build_face(face, mouth)
  build_hat(hat)
  built = { parts = { can = can, label = label, face = face, mouth = mouth, hat = hat },
            order = { "can", "label", "face", "mouth", "hat" },
            pivots = { hat = G.PIVOT_HAT, mouth = G.PIVOT_MOUTH } }
  return built
end

function G.triangle_count()
  local b = G.build()
  local n = 0
  for _, p in pairs(b.parts) do n = n + #p.tris end
  return n
end

--- Axis-aligned bounding box of some parts (names, or all when nil): min, max as {x,y,z}.
function G.bbox(names)
  local b = G.build()
  local lo, hi = P(math.huge, math.huge, math.huge), P(-math.huge, -math.huge, -math.huge)
  for _, name in ipairs(names or b.order) do
    for _, t in ipairs(b.parts[name].tris) do
      for _, v in ipairs({ t.a, t.b, t.c }) do
        lo.x, lo.y, lo.z = math.min(lo.x, v.x), math.min(lo.y, v.y), math.min(lo.z, v.z)
        hi.x, hi.y, hi.z = math.max(hi.x, v.x), math.max(hi.y, v.y), math.max(hi.z, v.z)
      end
    end
  end
  return lo, hi
end

-- the fixed light the baked shading uses (from above, a little to the front-left)
local LX, LY, LZ = 0.35, 0.30, 0.89
local LL = math.sqrt(LX * LX + LY * LY + LZ * LZ)

--- Brightness 0.4..1 of a surface with this normal under the baked light (two-sided: a face seen from behind is lit by its far side).
function G.shade(n)
  local d = math.abs(n.x * LX + n.y * LY + n.z * LZ) / LL
  return 0.4 + 0.6 * d
end

--- The vertex colour (r, g, b 0..255) of a triangle after shading.
function G.vertex_color(t)
  local s = G.shade(t.n)
  return math.floor(t.color[1] * s + 0.5), math.floor(t.color[2] * s + 0.5), math.floor(t.color[3] * s + 0.5)
end

return G
