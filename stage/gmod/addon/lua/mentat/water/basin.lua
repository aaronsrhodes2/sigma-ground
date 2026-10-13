-- water/basin.lua -- the geometry of a water body's basin: the shell that holds the water (floor and walls, as convex collision
-- pieces), the box the holodeck's planner sees, and the volume the water fills.  PURE Lua: no engine calls (offline-tested).
--
-- A water body is a basin standing ON the floor (the game cannot dig terrain): a floor slab and walls of wall_m thickness around an
-- inside of l x w x depth metres, the walls standing freeboard_m above the water.  Local frame: x along the length, y along the width
-- (the planner's footprint: x extent = d, y extent = w), z up from the floor the basin stands on.  Hammer units throughout (1 HU =
-- 0.01905 m) unless a name ends in _m.
local M = {}
local inc = include or function(p) return dofile(p) end
local C = inc("mentat/water/constants.lua")
local HU_M = C.HU_M
local pi, cos, sin, max = math.pi, math.cos, math.sin, math.max
M.SEGMENTS = 24

--- The basin of an inside l_m x w_m x depth_m.  shape = "rect" | "round" (round: an ellipse of those axes).
--- Returns { shape, mins = {x,y,z}, maxs = {x,y,z}, hx, hy (inside half extents), floor_z, depth, top_z, wall, volume_m3 }.
function M.new(shape, l_m, w_m, depth_m)
  local L = C.LIMITS
  local wall = L.wall_m.value / HU_M
  local free = L.freeboard_m.value / HU_M
  local hx, hy = l_m / HU_M / 2, w_m / HU_M / 2
  local depth = depth_m / HU_M
  local b = {
    shape = shape == "round" and "round" or "rect",
    hx = hx, hy = hy, wall = wall, floor_z = wall, depth = depth, free = free,
    top_z = wall + depth + free,
    l_m = l_m, w_m = w_m, depth_m = depth_m,
    mins = { -(hx + wall), -(hy + wall), 0 },
    maxs = { hx + wall, hy + wall, wall + depth + free },
  }
  b.volume_m3 = C.volume(b.shape, l_m, w_m, depth_m)
  return b
end

--- The footprint the planner needs (its w is the y extent, d the x extent, h the height), in HU.
function M.footprint(b)
  return { w = b.maxs[2] - b.mins[2], d = b.maxs[1] - b.mins[1], h = b.maxs[3] - b.mins[3] }
end

local function box(x0, x1, y0, y1, z0, z1)
  return { { x0, y0, z0 }, { x1, y0, z0 }, { x1, y1, z0 }, { x0, y1, z0 }, { x0, y0, z1 }, { x1, y0, z1 }, { x1, y1, z1 }, { x0, y1, z1 } }
end

--- The convex collision pieces of the shell: a list of lists of {x,y,z}.
function M.parts(b)
  local out = {}
  local hx, hy, w, H = b.hx, b.hy, b.wall, b.top_z
  if b.shape == "rect" then
    out[#out + 1] = box(-hx - w, hx + w, -hy - w, hy + w, 0, w)                          -- floor
    out[#out + 1] = box(-hx - w, hx + w, -hy - w, -hy, w, H)                             -- walls
    out[#out + 1] = box(-hx - w, hx + w, hy, hy + w, w, H)
    out[#out + 1] = box(-hx - w, -hx, -hy, hy, w, H)
    out[#out + 1] = box(hx, hx + w, -hy, hy, w, H)
  else
    local n = M.SEGMENTS
    local floor_pts, top_pts = {}, {}
    for i = 0, n - 1 do
      local a = 2 * pi * i / n
      floor_pts[#floor_pts + 1] = { (hx + w) * cos(a), (hy + w) * sin(a), 0 }
      floor_pts[#floor_pts + 1] = { (hx + w) * cos(a), (hy + w) * sin(a), w }
    end
    out[#out + 1] = floor_pts
    for i = 0, n - 1 do
      local a0, a1 = 2 * pi * i / n, 2 * pi * (i + 1) / n
      local seg = {}
      for _, z in ipairs({ w, H }) do
        seg[#seg + 1] = { hx * cos(a0), hy * sin(a0), z }
        seg[#seg + 1] = { hx * cos(a1), hy * sin(a1), z }
        seg[#seg + 1] = { (hx + w) * cos(a0), (hy + w) * sin(a0), z }
        seg[#seg + 1] = { (hx + w) * cos(a1), (hy + w) * sin(a1), z }
      end
      out[#out + 1] = seg
    end
  end
  return out
end

--- Is the local point (x, y) inside the water's footprint?  (z is not looked at.)
function M.inside_xy(b, x, y)
  if b.shape == "rect" then return x >= -b.hx and x <= b.hx and y >= -b.hy and y <= b.hy end
  local u, v = x / b.hx, y / b.hy
  return u * u + v * v <= 1
end

--- Resized copy: factor on the depth and/or the footprint (or absolute metres).  Returns a new basin.
function M.resized(b, o)
  local l, w, d = b.l_m, b.w_m, b.depth_m
  if o.depth_factor then d = d * o.depth_factor end
  if o.size_factor then l, w = l * o.size_factor, w * o.size_factor end
  if o.length_factor then l = l * o.length_factor end
  if o.width_factor then w = w * o.width_factor end
  if o.depth_m then d = o.depth_m end
  l, w, d = C.clamp_dims(l, w, d)
  return M.new(b.shape, l, w, d)
end

--- Water surface height (local z, HU) for a fill fraction 0..1.
function M.surface_z(b, fill)
  return b.floor_z + b.depth * math.max(0, math.min(1, fill))
end

return M
