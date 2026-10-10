-- layout.lua -- the geometry of arrangement. Pure Lua (no GMod globals), unit-tested offline.
--
-- Conventions (the same as the engine's): X east, Y north, yaw in degrees counter-clockwise from +X, Hammer units (HU).
-- A footprint is a rectangle: centre (x, y), `d` along the direction the thing FACES (depth), `w` across it (width), and
-- `theta`, the yaw of the facing direction. "right" of a facing is theta - 90 degrees.
--
-- A group is laid out in a FRAME {x, y, theta}: `theta` points from the group toward its audience (its open side), local
-- coordinates are u (forward, along theta) and v (right of it). Every generator returns WORLD poses {x, y, theta, z}
-- in the order of the items it was given, so callers never juggle local coordinates.
local Layout = {}

local rad, deg = math.rad, math.deg
local cos, sin, sqrt, abs, floor, ceil, max, min = math.cos, math.sin, math.sqrt, math.abs, math.floor, math.ceil, math.max, math.min
local atan2 = math.atan2 or math.atan
local pi = math.pi

-- ------------------------------------------------------------------------------------------------ basics
function Layout.wrap(a) return a % 360 end

function Layout.fwd(theta)
  local r = rad(theta)
  return cos(r), sin(r)
end

function Layout.right(theta)
  local fx, fy = Layout.fwd(theta)
  return fy, -fx
end

function Layout.angle_of(dx, dy) return deg(atan2(dy, dx)) % 360 end

--- Smallest signed difference between two yaws, in (-180, 180].
function Layout.angle_diff(a, b)
  local d = (a - b) % 360
  if d > 180 then d = d - 360 end
  return d
end

--- Deterministic random numbers (Park-Miller): the same request always lays out the same room.
function Layout.rng(seed)
  local state = floor(abs(seed or 1)) % 2147483646 + 1
  local self = {}
  function self.next()
    state = (state * 16807) % 2147483647
    return state / 2147483647
  end
  function self.range(a, b) return a + (b - a) * self.next() end
  function self.pick(list) return list[floor(self.next() * #list) + 1] end
  return self
end

function Layout.hash(str)
  local h = 5381
  for i = 1, #str do h = (h * 33 + str:byte(i)) % 2147483647 end
  return h
end

-- ------------------------------------------------------------------------------------------------ rectangles
function Layout.rect(x, y, w, d, theta)
  local fx, fy = Layout.fwd(theta)
  return { x = x, y = y, w = w, d = d, theta = theta, fx = fx, fy = fy, rx = fy, ry = -fx }
end

function Layout.corners(r)
  local out = {}
  for _, sd in ipairs({ { 1, 1 }, { 1, -1 }, { -1, -1 }, { -1, 1 } }) do
    out[#out + 1] = { r.x + sd[1] * r.d / 2 * r.fx + sd[2] * r.w / 2 * r.rx, r.y + sd[1] * r.d / 2 * r.fy + sd[2] * r.w / 2 * r.ry }
  end
  return out
end

--- Do two footprints overlap (or come closer than `pad` HU)? Separating-axis test; `pad` widens `a` on every side.
function Layout.overlap(a, b, pad)
  pad = pad or 0
  local dx, dy = b.x - a.x, b.y - a.y
  local axes = { { a.fx, a.fy }, { a.rx, a.ry }, { b.fx, b.fy }, { b.rx, b.ry } }
  for i = 1, 4 do
    local ux, uy = axes[i][1], axes[i][2]
    local dist = abs(dx * ux + dy * uy)
    local ra = (a.d / 2 + pad) * abs(a.fx * ux + a.fy * uy) + (a.w / 2 + pad) * abs(a.rx * ux + a.ry * uy)
    local rb = (b.d / 2) * abs(b.fx * ux + b.fy * uy) + (b.w / 2) * abs(b.rx * ux + b.ry * uy)
    if dist >= ra + rb then return false end
  end
  return true
end

--- How far two footprints interpenetrate along the best separating axis (0 when apart): for tests and for scoring.
function Layout.penetration(a, b)
  local dx, dy = b.x - a.x, b.y - a.y
  local axes = { { a.fx, a.fy }, { a.rx, a.ry }, { b.fx, b.fy }, { b.rx, b.ry } }
  local best = math.huge
  for i = 1, 4 do
    local ux, uy = axes[i][1], axes[i][2]
    local dist = abs(dx * ux + dy * uy)
    local ra = (a.d / 2) * abs(a.fx * ux + a.fy * uy) + (a.w / 2) * abs(a.rx * ux + a.ry * uy)
    local rb = (b.d / 2) * abs(b.fx * ux + b.fy * uy) + (b.w / 2) * abs(b.rx * ux + b.ry * uy)
    local pen = ra + rb - dist
    if pen <= 0 then return 0 end
    if pen < best then best = pen end
  end
  return best
end

--- Is the point inside the footprint (grown by `pad`)?
function Layout.contains(r, px, py, pad)
  pad = pad or 0
  local dx, dy = px - r.x, py - r.y
  return abs(dx * r.fx + dy * r.fy) <= r.d / 2 + pad and abs(dx * r.rx + dy * r.ry) <= r.w / 2 + pad
end

function Layout.to_world(frame, u, v, phi)
  local fx, fy = Layout.fwd(frame.theta)
  return { x = frame.x + u * fx + v * fy, y = frame.y + u * fy - v * fx, theta = Layout.wrap(frame.theta + (phi or 0)), z = 0 }
end

--- Extent of a footprint turned by `phi` degrees relative to the frame, along u and along v.
local function extent(item, phi)
  local q = floor((phi % 180) / 90 + 0.5) % 2
  if q == 0 then return item.d, item.w end
  return item.w, item.d
end

-- ------------------------------------------------------------------------------------------------ generators
--- Items side by side across the frame (along v), all facing the audience (or `opts.phi` away from it).
function Layout.row(frame, items, opts)
  opts = opts or {}
  local gap, phi = opts.gap or 6, opts.phi or 0
  local total, deepest = 0, 0
  for i, it in ipairs(items) do
    local eu, ev = extent(it, phi)
    total, deepest = total + ev + (i > 1 and gap or 0), max(deepest, eu)
  end
  local poses, v = {}, -total / 2
  for i, it in ipairs(items) do
    local eu, ev = extent(it, phi)
    -- align_back: the frame sits half the deepest item in front of the shared back line, so every back edge is flush
    local u = (opts.u or 0) + (opts.align_back and (eu - deepest) / 2 or 0)
    poses[i] = Layout.to_world(frame, u, v + ev / 2, phi)
    v = v + ev + gap
  end
  return poses
end

--- Items on the sides of an inner clear rectangle, every one facing in. sides = {back, left, right, front} lists of
--- items; `back` items stand at -u facing +u, `front` at +u facing -u, `left` (at -v) face +v, `right` face -v.
--- opts: gap between neighbours, clear (inner margin), inner_w / inner_d (minimum clear size), arm_align "back"|"center".
--- Returns poses keyed by item (poses[item] = pose) plus the inner rectangle {w, d}.
function Layout.perimeter(frame, sides, opts)
  opts = opts or {}
  local gap, clear = opts.gap or 6, opts.clear or 0
  local function total(list)
    local s = 0
    for i, it in ipairs(list or {}) do s = s + it.w + (i > 1 and gap or 0) end
    return s
  end
  local win = max(opts.inner_w or 0, total(sides.back), total(sides.front))
  local din = max(opts.inner_d or 0, total(sides.left), total(sides.right))
  local poses = {}
  local function along(list, fixed, vertical, phi, base)
    -- vertical: positions vary along u (arms); else along v. `fixed` = the fixed coordinate of the side's centre line
    local len = total(list)
    local pos = vertical and (opts.arm_align == "back" and -din / 2 or -len / 2) or -len / 2
    for i, it in ipairs(list or {}) do
      local c = pos + it.w / 2
      local depth_off = it.d / 2 + clear
      local p
      if vertical then p = Layout.to_world(frame, c, fixed * (win / 2 + depth_off), phi)
      else p = Layout.to_world(frame, fixed * (din / 2 + depth_off), c, phi) end
      poses[it] = p
      pos = pos + it.w + gap
    end
  end
  along(sides.back, -1, false, 0)
  along(sides.front, 1, false, 180)
  along(sides.left, -1, true, -90)
  along(sides.right, 1, true, 90)
  return poses, { w = win, d = din }
end

--- Items on a circle around (cx, cy), facing the centre (or away with opts.outward). The radius grows until no two
--- footprints touch. opts: gap, min_r, start (yaw of the first item as seen from the centre), arc (degrees, default 360).
function Layout.circle(cx, cy, items, opts)
  opts = opts or {}
  local gap, n = opts.gap or 6, #items
  local arc = opts.arc or 360
  local start = opts.start or 0
  local sum = 0
  for _, it in ipairs(items) do sum = sum + it.w + gap end
  local r = max(opts.min_r or 0, sum / rad(arc), 12)
  local function build(radius)
    local poses, rects = {}, {}
    local step = (n > 1) and (arc >= 360 and 360 / n or arc / (n - 1)) or 0
    for i, it in ipairs(items) do
      local a = start + (i - 1) * step - ((arc < 360 and n > 1) and arc / 2 or 0)
      local px, py = cx + radius * cos(rad(a)), cy + radius * sin(rad(a))
      local face = opts.outward and a or a + 180
      poses[i] = { x = px, y = py, theta = Layout.wrap(face), z = 0 }
      rects[i] = Layout.rect(px, py, it.w, it.d, face)
    end
    return poses, rects
  end
  for _ = 1, 600 do
    local poses, rects = build(r)
    local clash = false
    for i = 1, n do
      for j = i + 1, n do
        if Layout.overlap(rects[i], rects[j], gap / 2) then clash = true; break end
      end
      if clash then break end
    end
    if not clash or n < 2 then return poses, r end
    r = r + 2
  end
  local poses = build(r)
  return poses, r
end

--- Rows and columns (cols across, rows along u), all facing the audience. opts: gap_u, gap_v, cols.
function Layout.grid(frame, items, opts)
  opts = opts or {}
  local n = #items
  local cols = opts.cols or ceil(sqrt(n))
  local rows = ceil(n / cols)
  local cw, cd = 0, 0
  for _, it in ipairs(items) do cw, cd = max(cw, it.w), max(cd, it.d) end
  local gv, gu = opts.gap_v or opts.gap or 10, opts.gap_u or opts.gap or 10
  local poses = {}
  for i, it in ipairs(items) do
    local r, c = floor((i - 1) / cols), (i - 1) % cols
    local in_row = (r == rows - 1) and (n - r * cols) or cols
    local v = (c - (in_row - 1) / 2) * (cw + gv)
    local u = (r - (rows - 1) / 2) * (cd + gu)
    poses[i] = Layout.to_world(frame, u, v, 0)
  end
  return poses, { cols = cols, rows = rows, w = cols * (cw + gv) - gv, d = rows * (cd + gu) - gu }
end

--- A pyramid / pile: bottom layer of as many as fit across `base`, each layer above one fewer, centred on the join.
--- Returns poses with z = height above the floor of the pile's own base.
function Layout.stack(frame, items, opts)
  opts = opts or {}
  local n = #items
  local gap = opts.gap or 1
  local base = opts.base
  if not base then base = 1; while base * (base + 1) / 2 < n do base = base + 1 end end
  local poses, i, z, layer = {}, 1, 0, 0
  while i <= n do
    local count = min(base - layer, n - i + 1)
    if count < 1 then count = n - i + 1 end
    local lh, lw = 0, 0
    for k = i, i + count - 1 do lh, lw = max(lh, items[k].h or 0), max(lw, items[k].w) end
    for k = 0, count - 1 do
      local it = items[i + k]
      local v = (k - (count - 1) / 2) * (lw + gap)
      local pose = Layout.to_world(frame, 0, v, 0)
      pose.z = z
      poses[i + k] = pose
    end
    i = i + count
    z = z + lh
    layer = layer + 1
  end
  return poses
end

--- Scatter by rejection sampling inside a disc; deterministic for a given rng. `spin`: random yaws (else face the frame).
function Layout.scatter(frame, items, rng, opts)
  opts = opts or {}
  local gap = opts.gap or 10
  local radius = opts.radius or 60
  local poses, rects = {}, {}
  for i, it in ipairs(items) do
    local ok
    for attempt = 1, 400 do
      local rr = radius * sqrt(rng.next()) * (1 + attempt / 200)
      local a = rng.range(0, 2 * pi)
      local u, v = rr * cos(a), rr * sin(a)
      local theta = opts.spin and rng.range(0, 360) or frame.theta + rng.range(-25, 25)
      local pose = Layout.to_world(frame, u, v, 0)
      local rect = Layout.rect(pose.x, pose.y, it.w, it.d, theta)
      local clash = false
      for j = 1, #rects do
        if Layout.overlap(rects[j], rect, gap / 2) then clash = true; break end
      end
      if not clash then
        pose.theta = Layout.wrap(theta)
        poses[i], rects[i], ok = pose, rect, true
        break
      end
    end
    if not ok then
      local pose = Layout.to_world(frame, radius * 2 + i * 20, 0, 0)
      poses[i] = pose
      rects[i] = Layout.rect(pose.x, pose.y, it.w, it.d, frame.theta)
    end
  end
  return poses
end

--- Items standing on the flat top of something (`top` = {x, y, w, d, theta, z} of the usable top), spread along its long
--- axis and set toward its front edge. Returns poses with z = the top height.
function Layout.on_top(top, items, opts)
  opts = opts or {}
  local gap = opts.gap or 4
  local frame = { x = top.x, y = top.y, theta = top.theta }
  local n = #items
  local along_v = top.w >= top.d
  local poses = {}
  local total = 0
  for i, it in ipairs(items) do total = total + (along_v and it.w or it.d) + (i > 1 and gap or 0) end
  local pos = -total / 2
  for i, it in ipairs(items) do
    local len = along_v and it.w or it.d
    local c = pos + len / 2
    local u, v = 0, 0
    if along_v then v = c else u = c end
    -- bias: -1 sets things toward the back edge (a monitor on a desk), +1 toward the front, 0 centres
    if opts.bias and opts.bias ~= 0 then
      local room = (along_v and top.d - it.d or top.w - it.w) / 2
      local shift = opts.bias * max(0, room - 2) * 0.8
      if along_v then u = shift else v = shift end
    end
    local pose = Layout.to_world(frame, u, v, opts.phi or 0)
    pose.z = top.z or 0
    poses[i] = pose
    pos = pos + len + gap
  end
  return poses
end

-- ------------------------------------------------------------------------------------------------ walls and space
--- Nearest wall around (x, y): casts `rays` rays out to `maxd` HU through env.wall_probe(x, y, dx, dy, maxd), which
--- returns nil or {dist, nx, ny} (distance to the wall and its unit normal pointing into the room). Returns
--- {dist (perpendicular), nx, ny, px, py (the foot point on the wall)} or nil.
function Layout.nearest_wall(env, x, y, maxd, rays)
  if not (env and env.wall_probe) then return nil end
  rays = rays or 16
  local best
  for k = 0, rays - 1 do
    local a = k * 360 / rays
    local dx, dy = cos(rad(a)), sin(rad(a))
    local hit = env.wall_probe(x, y, dx, dy, maxd)
    if hit then
      local perp = hit.dist * max(0, -(dx * hit.nx + dy * hit.ny))
      if perp > 0 and (not best or perp < best.dist) then
        best = { dist = perp, nx = hit.nx, ny = hit.ny, px = x - hit.nx * perp, py = y - hit.ny * perp }
      end
    end
  end
  return best
end

--- Free space (HU) straight ahead of (x, y) along theta, up to `maxd`.
function Layout.free_ahead(env, x, y, theta, maxd)
  if not (env and env.wall_probe) then return maxd end
  local dx, dy = Layout.fwd(theta)
  local hit = env.wall_probe(x, y, dx, dy, maxd)
  return hit and hit.dist or maxd
end

--- The room's rough shape from rays around a point: the distances to the walls in the four axis-aligned-to-the-nearest-wall
--- directions. Returns {theta0 (yaw of the nearest wall's normal), ahead, behind, left, right} measured from (x, y).
function Layout.room_box(env, x, y, maxd)
  local wall = Layout.nearest_wall(env, x, y, maxd)
  if not wall then return nil end
  local t0 = Layout.angle_of(wall.nx, wall.ny)
  local box = { theta0 = t0 }
  for i, k in ipairs({ "along", "right", "behind", "left" }) do
    local dx, dy = Layout.fwd(t0 + (i - 1) * 90)
    local hit = env.wall_probe(x, y, dx, dy, maxd)
    box[k] = hit and hit.dist or maxd
  end
  return box
end

--- Turn a pose so its back (the side opposite its facing) sits `gap` HU off `wall`, keeping the item's centre line.
--- `size` = {w, d}; the item then faces along the wall normal, into the room.
function Layout.snap_to_wall(pose, size, wall, gap)
  local theta = Layout.angle_of(wall.nx, wall.ny)
  -- distance of the centre from the wall plane, measured along the normal
  local off = (pose.x - wall.px) * wall.nx + (pose.y - wall.py) * wall.ny
  local want = size.d / 2 + (gap or 1)
  return { x = pose.x + (want - off) * wall.nx, y = pose.y + (want - off) * wall.ny, theta = theta, z = pose.z or 0 }
end

--- Corner positions of the room around (x, y): the four wall intersections of room_box, pulled in by `inset`.
function Layout.room_corners(env, x, y, maxd, inset)
  local box = Layout.room_box(env, x, y, maxd)
  if not box then return nil end
  local out = {}
  for i = 0, 3 do
    local ax, ay = Layout.fwd(box.theta0 + i * 90)
    local bx, by = Layout.fwd(box.theta0 + (i + 1) * 90)
    local da = ({ box.along, box.right, box.behind, box.left })[i + 1]
    local db = ({ box.along, box.right, box.behind, box.left })[(i + 1) % 4 + 1]
    local cx, cy = x + ax * (da - inset) + bx * (db - inset), y + ay * (da - inset) + by * (db - inset)
    out[#out + 1] = { x = cx, y = cy, theta = Layout.wrap(Layout.angle_of(x - cx, y - cy)) }
  end
  return out, box
end

return Layout
