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
-- (a search builds hundreds of rectangles at the same few yaws: the last yaw's direction is remembered)
local last_theta, last_fx, last_fy
function Layout.rect(x, y, w, d, theta)
  local fx, fy
  if theta == last_theta then fx, fy = last_fx, last_fy
  else
    fx, fy = Layout.fwd(theta)
    last_theta, last_fx, last_fy = theta, fx, fy
  end
  return { x = x, y = y, w = w, d = d, theta = theta, fx = fx, fy = fy, rx = fy, ry = -fx, rad = sqrt(w * w + d * d) / 2 }
end

function Layout.corners(r)
  local out = {}
  for _, sd in ipairs({ { 1, 1 }, { 1, -1 }, { -1, -1 }, { -1, 1 } }) do
    out[#out + 1] = { r.x + sd[1] * r.d / 2 * r.fx + sd[2] * r.w / 2 * r.rx, r.y + sd[1] * r.d / 2 * r.fy + sd[2] * r.w / 2 * r.ry }
  end
  return out
end

local SQRT2 = 1.4143

-- Do the projections of a (half extents ad along its front, aw across) and b (bd, bw) onto the axis (ux, uy) leave a gap?
local function separated(ux, uy, dx, dy, a, b, ad, aw, bd, bw)
  return abs(dx * ux + dy * uy) >= ad * abs(a.fx * ux + a.fy * uy) + aw * abs(a.rx * ux + a.ry * uy)
                                   + bd * abs(b.fx * ux + b.fy * uy) + bw * abs(b.rx * ux + b.ry * uy)
end

--- Do two footprints overlap (or come closer than `pad` HU)? A circle test throws out the far pairs (most of them) before the
--- separating-axis test; `pad` widens `a` on every side.
function Layout.overlap(a, b, pad)
  pad = pad or 0
  local dx, dy = b.x - a.x, b.y - a.y
  local reach = (a.rad or sqrt(a.w * a.w + a.d * a.d) / 2) + pad * SQRT2 + (b.rad or sqrt(b.w * b.w + b.d * b.d) / 2)
  if dx * dx + dy * dy >= reach * reach then return false end
  local ad, aw, bd, bw = a.d / 2 + pad, a.w / 2 + pad, b.d / 2, b.w / 2
  return not (separated(a.fx, a.fy, dx, dy, a, b, ad, aw, bd, bw) or separated(a.rx, a.ry, dx, dy, a, b, ad, aw, bd, bw)
              or separated(b.fx, b.fy, dx, dy, a, b, ad, aw, bd, bw) or separated(b.rx, b.ry, dx, dy, a, b, ad, aw, bd, bw))
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
--- footprints touch. opts: gap, min_r, start (yaw of the first item as seen from the centre), arc (degrees, default 360),
--- stop (a function that says the caller has run out of time: the best radius so far is returned).
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
    if opts.stop and opts.stop() then return poses, r end
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
--- Each item rests on the highest of the (up to two) items of the layer below that it overlaps. Items of unlike widths (a mixed
--- pile) go widest first at the bottom, the base packed edge to edge, and every upper item stands centred on a support of its own,
--- so no layer is wider than the one below and nothing hangs over a gap. opts.max_h widens the base until the pile is no taller
--- than that. Returns poses (keyed by the caller's own item indices) with z = height above the floor of the pile's own base.
function Layout.stack(frame, items, opts)
  opts = opts or {}
  local n = #items
  local gap = opts.gap or 1
  local base = opts.base
  if not base then base = 1; while base * (base + 1) / 2 < n do base = base + 1 end end
  -- layer sizes for a given base (the last layer takes whatever is left)
  local function layers_of(b)
    local out, i, layer = {}, 1, 0
    while i <= n do
      local count = min(b - layer, n - i + 1)
      if count < 1 then count = n - i + 1 end
      out[#out + 1] = count
      i, layer = i + count, layer + 1
    end
    return out
  end
  local tallest = 0
  for _, it in ipairs(items) do tallest = max(tallest, it.h or 0) end
  if opts.max_h and tallest > 0 then
    while base < n and #layers_of(base) * tallest > opts.max_h do base = base + 1 end
  end
  -- the widest go lowest (a stable sort, poses written back to the caller's own indices), so no layer is wider than the one under
  -- it and the base layer bounds the whole pile: the caller's wall and floor checks on the base then cover every layer
  local order = {}
  for k = 1, n do order[k] = k end
  table.sort(order, function(a, b)
    if items[a].w ~= items[b].w then return items[a].w > items[b].w end
    return a < b
  end)
  -- unlike widths (a mixed pile): the base is packed edge to edge and every upper item stands centred on a support of its own (the
  -- k-th widest on the k-th widest support below), so nothing hangs over a gap; equal widths keep the pyramid, each over two
  local wmin, wmax = 1e9, -1e9
  for _, it in ipairs(items) do wmin, wmax = min(wmin, it.w), max(wmax, it.w) end
  local mixed = wmax - wmin > 0.5
  local poses, i = {}, 1
  local below                              -- the layer under this one: { {v, half, top} ... }
  for _, count in ipairs(layers_of(base)) do
    local lw = 0
    for k = i, i + count - 1 do lw = max(lw, items[order[k]].w) end
    local row = {}
    local supports
    if mixed and below then
      supports = {}
      for _, b in ipairs(below) do supports[#supports + 1] = b end
      table.sort(supports, function(a, b) if a.half ~= b.half then return a.half > b.half end return a.v < b.v end)
    end
    local total = -gap
    if mixed and not below then
      for k = 0, count - 1 do total = total + items[order[i + k]].w + gap end
    end
    local cursor = -total / 2
    for k = 0, count - 1 do
      local it = items[order[i + k]]
      local v = (k - (count - 1) / 2) * (lw + gap)
      local z = 0
      if mixed and not below then
        v = cursor + it.w / 2
        cursor = cursor + it.w + gap
      elseif mixed and supports[k + 1] then
        v, z = supports[k + 1].v, supports[k + 1].top
      elseif below then
        local seen = false
        for _, b in ipairs(below) do
          if abs(b.v - v) < b.half + it.w / 2 then z, seen = max(z, b.top), true end
        end
        if not seen then for _, b in ipairs(below) do z = max(z, b.top) end end
      end
      local pose = Layout.to_world(frame, 0, v, 0)
      pose.z = z
      poses[order[i + k]] = pose
      row[#row + 1] = { v = v, half = it.w / 2, top = z + (it.h or 0) }
    end
    below = row
    i = i + count
  end
  return poses
end

--- Scatter by rejection sampling inside a disc; deterministic for a given rng. `spin`: random yaws (else face the frame);
--- `stop`: a function that says the caller is out of time (what is not placed by then is parked out of the way).
function Layout.scatter(frame, items, rng, opts)
  opts = opts or {}
  local gap = opts.gap or 10
  local radius = opts.radius or 60
  local poses, rects = {}, {}
  local stopped = false
  for i, it in ipairs(items) do
    local ok
    for attempt = 1, 400 do
      if stopped or (opts.stop and attempt % 16 == 1 and opts.stop()) then stopped = true; break end
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

--- Items standing on the flat top of something (`top` = {x, y, w, d, theta, z} of the usable top), in rows along its long
--- axis and set toward its back (bias -1) or front (+1) edge. Nothing overhangs by more than `opts.overhang` (default 2 HU)
--- a side; an item that does not fit gets no pose. A symmetric item (`it.sym`) may turn a quarter to fit. The gap between
--- neighbours shrinks (4, 2, then 1 HU) before anything is left off. Returns poses (poses[i] = nil for an item left off,
--- else with z = the top height) and the number left off.
function Layout.on_top(top, items, opts)
  opts = opts or {}
  local tol = opts.overhang or 2
  local frame = { x = top.x, y = top.y, theta = top.theta }
  local along_v = top.w >= top.d
  local long, short = (along_v and top.w or top.d) + 2 * tol, (along_v and top.d or top.w) + 2 * tol
  -- extents of an item along the top's long axis and across it, turned a quarter or not
  local function ext(it, turned)
    local a, b = it.w, it.d                                    -- along v, along u
    if turned then a, b = b, a end
    if along_v then return a, b end
    return b, a
  end
  local function pack(gap)
    local rows, dropped = {}, 0
    for i, it in ipairs(items) do
      local turn = false
      local a, b = ext(it, false)
      if (a > long or b > short) and it.sym then
        local a2, b2 = ext(it, true)
        if a2 <= long and b2 <= short then a, b, turn = a2, b2, true end
      end
      local at
      if a <= long and b <= short then
        for _, row in ipairs(rows) do
          if row.len + gap + a <= long then
            -- a deeper item deepens its row: the rows together must still fit the depth of the top
            local others = 0
            for _, r2 in ipairs(rows) do if r2 ~= row then others = others + r2.depth + gap end end
            if others + max(row.depth, b) <= short then at = row; break end
          end
        end
        if not at then
          local used = 0
          for _, row in ipairs(rows) do used = used + row.depth + gap end
          if used + b <= short then
            at = { len = -gap, depth = 0, list = {} }
            rows[#rows + 1] = at
          end
        end
      end
      if at then
        at.len = at.len + gap + a
        at.depth = max(at.depth, b)
        at.list[#at.list + 1] = { i = i, a = a, b = b, turn = turn }
      else
        dropped = dropped + 1
      end
    end
    return rows, dropped
  end
  local gap0 = opts.gap or 4
  local rows, dropped = pack(gap0)
  local gap = gap0
  for _, g in ipairs({ 2, 1 }) do
    if dropped == 0 or g >= gap0 then break end
    local r2, d2 = pack(g)
    if d2 < dropped then rows, dropped, gap = r2, d2, g end
  end
  -- the block of rows, centred across the top and pushed toward the biased edge
  local height = -gap
  for _, row in ipairs(rows) do height = height + gap + row.depth end
  local poses = {}
  local cross = -height / 2
  for _, row in ipairs(rows) do
    local pos = -row.len / 2
    for _, e in ipairs(row.list) do
      local it = items[e.i]
      local c = pos + e.a / 2
      local lat = cross + row.depth / 2                         -- the row's centre line across the top
      if opts.bias and opts.bias ~= 0 then
        -- one row: each item sets toward the edge by what it has to spare; several rows move as a block
        local room = (#rows == 1) and (short - 2 * tol - e.b) / 2 or (short - 2 * tol - height) / 2
        lat = lat + opts.bias * max(0, room - 2) * 0.8
      end
      local u, v
      if along_v then u, v = lat, c else u, v = c, lat end
      local pose = Layout.to_world(frame, u, v, (opts.phi or 0) + (e.turn and 90 or 0))
      pose.z = top.z or 0
      poses[e.i] = pose
      pos = pos + e.a + gap
    end
    cross = cross + row.depth + gap
  end
  return poses, dropped
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
