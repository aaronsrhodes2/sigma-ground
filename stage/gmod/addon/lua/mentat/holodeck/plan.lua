-- plan.lua -- turns a parsed request into poses. Pure Lua (no GMod globals): the world is reached only through ctx.env,
-- so the whole planner is unit-tested offline against synthetic rooms, then driven by the engine glue (holodeck.lua).
--
--   local plan = Plan.new(catalog)
--   local res  = plan:run(request, ctx)
--
-- ctx = { pos = {x,y,z}, yaw = degrees, aim = {x,y,z} or nil, seed = n,
--         env = { floor_z(x,y) -> z|nil,  clear(rect, z, h) -> bool,  wall_probe(x,y,dx,dy,maxd) -> {dist,nx,ny}|nil },
--         registry = { {id, kind, vi, x, y, z, theta, w, d, h, color, group, seq}, ... }   -- what is already standing }
-- res = { action, placements = { {kind, vi, model, x, y, z, theta, w, d, h, origin = {x,y,z}, yaw, ...} },
--         moves = { {id, x, y, z, theta} }, removals = { id, ... }, notes = { ... }, say = "one sentence" }
--
-- It never asks a question: whatever is unclear resolves to the most sensible reading, and the reading is reported.
local Layout = include("mentat/holodeck/layout.lua")

local Plan = {}
Plan.__index = Plan

local rad, deg = math.rad, math.deg
local cos, sin, sqrt, abs, floor, ceil, max, min = math.cos, math.sin, math.sqrt, math.abs, math.floor, math.ceil, math.max, math.min
local rect, to_world, fwd = Layout.rect, Layout.to_world, Layout.fwd

local MAX_PER_KIND, MAX_TOTAL = 60, 150

-- gaps between neighbours of a kind (HU; one HU is 1.905 cm)
local GAP = { seating = 6, table = 12, bed = 24, storage = 1, container = 3, lighting = 40, appliance = 1, bath = 6, electronics = 14, decor = 24 }

local COLORS = {
  red = { 200, 45, 45 }, blue = { 55, 95, 215 }, green = { 65, 165, 75 }, yellow = { 230, 205, 60 }, orange = { 235, 140, 45 },
  pink = { 235, 135, 180 }, purple = { 140, 75, 190 }, violet = { 140, 75, 190 }, brown = { 125, 85, 45 }, black = { 35, 35, 35 },
  white = { 255, 255, 255 }, grey = { 150, 150, 150 }, gray = { 150, 150, 150 }, cyan = { 60, 190, 190 }, teal = { 50, 160, 160 },
  gold = { 212, 175, 55 }, silver = { 192, 192, 192 }, beige = { 215, 190, 150 }, tan = { 210, 180, 140 }, navy = { 25, 35, 105 },
  maroon = { 115, 25, 35 }, turquoise = { 60, 200, 190 }, lime = { 150, 220, 60 }, crimson = { 190, 25, 55 },
}
local SIZE_UP = { big = true, large = true, huge = true, giant = true, long = true, wide = true, bigger = true, larger = true,
  biggest = true, largest = true, massive = true, enormous = true, king = true }
local SIZE_DOWN = { small = true, little = true, tiny = true, short = true, mini = true, narrow = true, compact = true, smaller = true,
  smallest = true, petite = true }
local STYLE = { office = "office", antique = "antique", old = "antique", vintage = "antique", kitchen = "kitchen", lab = "lab",
  garden = "tides", outdoor = "tides", patio = "tides", military = "militia", army = "militia", prison = "prison", jail = "prison",
  hospital = "hospital", train = "train", station = "train", cafeteria = "cafeteria", wooden = "wood", wood = "wood",
  combine = "combine", wasteland = "wasteland" }

-- ---------------------------------------------------------------------------------------------- small helpers
local function plural(kind, n)
  if n == 1 then return kind end
  local last = kind:match("(%S+)$")
  local base = kind:sub(1, #kind - #last)
  if last:match("[sxz]$") or last:match("ch$") or last:match("sh$") then return base .. last .. "es" end
  if last:match("[^aeiou]y$") then return base .. last:sub(1, -2) .. "ies" end
  return base .. last .. "s"
end

local function article(word) return (word:match("^[aeiou]") and "an " or "a ") .. word end

local NUMWORD = { "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve" }
local function count_word(n) return NUMWORD[n] or tostring(n) end

local function color_of(adjs)
  for _, a in ipairs(adjs or {}) do
    if COLORS[a] then return COLORS[a], a end
  end
end

--- Entity position and yaw that put a model's footprint centre at (x, y), facing `theta`, its bottom at `zbot`.
function Plan.entity_pose(v, x, y, zbot, theta)
  local yaw = (theta - v.front) % 360
  local r = rad(yaw)
  local ox = x - (v.cx * cos(r) - v.cy * sin(r))
  local oy = y - (v.cx * sin(r) + v.cy * cos(r))
  return { x = ox, y = oy, z = zbot - v.mins[3] }, yaw
end

--- The usable flat top of a variant in its footprint frame: {u, v, w, d, z} (u along its front), or nil.
local function top_frame(v)
  local t = v.top
  if not t then return nil end
  local f = rad(v.front)
  local cf, sf = cos(f), sin(f)
  local u0, u1, v0, v1 = 1e9, -1e9, 1e9, -1e9
  for _, c in ipairs({ { t.x0, t.y0 }, { t.x0, t.y1 }, { t.x1, t.y0 }, { t.x1, t.y1 } }) do
    local dx, dy = c[1] - v.cx, c[2] - v.cy
    local u, w = dx * cf + dy * sf, dx * sf - dy * cf
    u0, u1, v0, v1 = min(u0, u), max(u1, u), min(v0, w), max(v1, w)
  end
  return { u = (u0 + u1) / 2, v = (v0 + v1) / 2, d = u1 - u0, w = v1 - v0, z = t.z }
end

local function make_inst(self, kind_id, vi, adjs)
  local k = self.kinds[kind_id]
  local v = k.variants[vi]
  local color, color_name = color_of(adjs)
  return { kind = kind_id, k = k, vi = vi, v = v, w = v.w, d = v.d, h = v.h, adjectives = adjs or {}, color = color, color_name = color_name }
end

--- Which variants of a kind to use for `n` of it, with the adjectives that narrow or steer the choice.
function Plan:pick_variants(kind_id, n, adjs, rng)
  local k = self.kinds[kind_id]
  local pool = {}
  for i = 1, #k.variants do pool[i] = i end
  local function area(i) return k.variants[i].w * k.variants[i].d end
  for _, a in ipairs(adjs or {}) do
    if STYLE[a] then
      local narrowed = {}
      for _, i in ipairs(pool) do
        if k.variants[i].model:find(STYLE[a], 1, true) then narrowed[#narrowed + 1] = i end
      end
      if #narrowed > 0 then pool = narrowed end
    end
  end
  table.sort(pool, function(a, b) if area(a) ~= area(b) then return area(a) < area(b) end return a < b end)
  local up, down = false, false
  for _, a in ipairs(adjs or {}) do
    if SIZE_UP[a] then up = true elseif SIZE_DOWN[a] then down = true end
  end
  -- a television is a screen you can see across a room: skip the little tabletop set unless asked for a small or old one
  if kind_id == "tv" and not down then
    local big, wants_old = {}, false
    for _, a in ipairs(adjs or {}) do if a == "old" or a == "vintage" or a == "crt" or a == "antique" then wants_old = true end end
    for _, i in ipairs(pool) do if k.variants[i].h >= 30 or wants_old then big[#big + 1] = i end end
    if #big > 0 then pool = big end
  end
  if up then pool = { pool[#pool] } elseif down then pool = { pool[1] } end
  local out = {}
  local consistent = ({ seating = true, table = true, bed = true, storage = true, appliance = true, bath = true, lighting = false })[k.category]
  if kind_id == "tv" or kind_id == "computer" or kind_id == "crate" or kind_id == "box" then consistent = true end
  if consistent or #pool == 1 then
    local pick = pool[floor(rng.next() * #pool) + 1]
    for i = 1, n do out[i] = pick end
  else
    local order = {}
    for i, v in ipairs(pool) do order[i] = v end
    for i = #order, 2, -1 do
      local j = floor(rng.next() * i) + 1
      order[i], order[j] = order[j], order[i]
    end
    for i = 1, n do out[i] = order[(i - 1) % #order + 1] end
  end
  return out
end

-- ---------------------------------------------------------------------------------------------- world checks
local function floor_ok(env, r, tol)
  local fz = env.floor_z(r.x, r.y)
  if not fz then return nil end
  for _, c in ipairs(Layout.corners(r)) do
    local cz = env.floor_z(c[1], c[2])
    if not cz or abs(cz - fz) > (tol or 8) then return nil end
  end
  return fz
end

--- Violations of a placement list: off the floor, into the world, into each other, into what already stands.
local function judge(ctx, pls, ignore)
  local env, viol, pen, hard = ctx.env, 0, 0, 0
  local rects = {}
  for i, p in ipairs(pls) do
    local r = rect(p.x, p.y, p.w, p.d, p.theta)
    rects[i] = r
    if (p.zoff or 0) <= 0.01 then
      local fz = floor_ok(env, r)
      if not fz then viol, hard = viol + 1, hard + 1
      else
        p.fz = fz
        if not env.clear(r, fz, p.h) then viol, hard = viol + 1, hard + 1 end
      end
    else
      p.fz = env.floor_z(p.x, p.y) or ctx.pos.z
    end
  end
  if ctx.keep_out ~= false then
    local me = rect(ctx.pos.x, ctx.pos.y, 40, 40, 0)
    for i, p in ipairs(pls) do
      if (p.zoff or 0) <= 0.01 and Layout.overlap(me, rects[i], 4) then viol, hard = viol + 1, hard + 1 end
    end
  end
  for i = 1, #pls do
    for j = i + 1, #pls do
      local a, b = pls[i], pls[j]
      local skip = (a.zoff or 0) > 0.01 or (b.zoff or 0) > 0.01
      local tucked = a.tuck == b or b.tuck == a
      if tucked and not skip and Layout.overlap(rects[i], rects[j], 0.4) then
        -- allowed only as far as the chair's front edge goes under the table
        local allow = (a.tuck == b and a.tuck_depth or b.tuck_depth or 9) + 1.5
        if Layout.penetration(rects[i], rects[j]) > allow then viol = viol + 1; pen = pen + Layout.penetration(rects[i], rects[j]) end
        skip = true
      end
      if not skip and Layout.overlap(rects[i], rects[j], 0.4) then
        viol = viol + 1
        pen = pen + Layout.penetration(rects[i], rects[j])
      end
    end
  end
  for i, p in ipairs(pls) do
    if (p.zoff or 0) <= 0.01 then
      for _, e in ipairs(ctx.registry or {}) do
        if not (ignore and ignore[e.id]) then
          local er = rect(e.x, e.y, e.w, e.d, e.theta)
          if Layout.overlap(er, rects[i], 0.4) and not (p.sits_on == e.id) then
            viol, hard = viol + 1, hard + 1
            pen = pen + Layout.penetration(er, rects[i])
          end
        end
      end
    end
  end
  return viol, pen, hard
end

-- ---------------------------------------------------------------------------------------------- finding a place
local function player_frame_theta(ctx) return Layout.wrap(ctx.yaw + 180) end

--- Walls near the player, nearest first, the one in view ahead of the rest: {px,py (foot), nx,ny, perp, hx,hy (view hit)}.
local function nearby_walls(ctx, maxd)
  local env, pos = ctx.env, ctx.pos
  if not env.wall_probe then return {} end
  local found = {}
  local function ray(a, tag)
    local dx, dy = fwd(a)
    local hit = env.wall_probe(pos.x, pos.y, dx, dy, maxd)
    if not hit then return end
    local perp = hit.dist * max(0, -(dx * hit.nx + dy * hit.ny))
    if perp < 8 then return end
    local ang = Layout.angle_of(hit.nx, hit.ny)
    for _, w in ipairs(found) do
      if abs(Layout.angle_diff(w.ang, ang)) < 8 and abs(w.perp - perp) < 10 then
        if tag == "view" then w.view = true; w.hx, w.hy = pos.x + dx * hit.dist, pos.y + dy * hit.dist end
        return
      end
    end
    found[#found + 1] = { ang = ang, nx = hit.nx, ny = hit.ny, perp = perp, px = pos.x - hit.nx * perp, py = pos.y - hit.ny * perp,
                          view = (tag == "view"), hx = pos.x + dx * hit.dist, hy = pos.y + dy * hit.dist }
  end
  ray(ctx.yaw, "view")
  for k = 0, 23 do ray(k * 15, "ring") end
  table.sort(found, function(a, b)
    if a.view ~= b.view then return a.view end
    return a.perp < b.perp
  end)
  return found
end

--- Frames to try for a group of radius R: along the player's view, widening outward and sideways.
local function open_frames(ctx, R, theta_for)
  local pos, yaw = ctx.pos, ctx.yaw
  local frames = {}
  local function add(cx, cy)
    frames[#frames + 1] = { x = cx, y = cy, theta = theta_for(cx, cy) }
  end
  if ctx.aim and ctx.aim_first then add(ctx.aim.x, ctx.aim.y) end
  -- straight ahead first, then sweeping round the player, so a player in the middle of a small room still finds a spot
  for _, bearing in ipairs({ 0, 35, -35, 70, -70, 105, -105, 140, -140, 180 }) do
    local fx, fy = fwd(yaw + bearing)
    local rx, ry = fy, -fx
    for k = 0, 12 do
      local dist = R + 62 + k * 28
      for _, lat in ipairs({ 0, 1, -1, 2, -2 }) do
        add(pos.x + fx * dist + rx * lat * 55, pos.y + fy * dist + ry * lat * 55)
      end
    end
  end
  return frames
end

--- Frames flush against walls: the group's back line on the wall, slid along it. `depth` is the group's depth (HU).
local function wall_frames(ctx, depth, maxd)
  local frames = {}
  for wi, w in ipairs(nearby_walls(ctx, maxd or 900)) do
    if wi > 3 then break end
    local cx0, cy0 = w.view and w.hx or w.px, w.view and w.hy or w.py
    local tx, ty = -w.ny, w.nx
    for _, s in ipairs({ 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8 }) do
      local sx, sy = cx0 + tx * s * 40, cy0 + ty * s * 40
      frames[#frames + 1] = { x = sx + w.nx * (depth / 2 + 1), y = sy + w.ny * (depth / 2 + 1), theta = Layout.angle_of(w.nx, w.ny), wall = w }
    end
  end
  return frames
end

--- The room the player stands in, when the four directions along the nearest wall all hit a wall within `maxd`:
--- {frame = {x, y, theta}, w, d} in the room builder's convention (back wall at -u, theta pointing out from the nearest wall).
local function real_room(ctx, maxd)
  local env, pos = ctx.env, ctx.pos
  if not env.wall_probe then return nil end
  local box = Layout.room_box(env, pos.x, pos.y, maxd)
  if not box then return nil end
  if box.along >= maxd or box.behind >= maxd or box.right >= maxd or box.left >= maxd then return nil end
  local w, d = box.right + box.left, box.along + box.behind
  local fx, fy = fwd(box.theta0)
  local rx, ry = fy, -fx
  local u_off, v_off = (box.along - box.behind) / 2, (box.left - box.right) / 2
  return { frame = { x = pos.x + fx * u_off + rx * v_off, y = pos.y + fy * u_off + ry * v_off, theta = box.theta0 }, w = w, d = d }
end

--- Build with each candidate frame; the first that violates nothing wins, otherwise the least-bad.
local function fit(ctx, builders, frames, ignore)
  local best, best_score
  -- a request for a hundred chairs in a full room must not stall a shared server: stop searching after a moment
  -- (one allowance for the whole request, shared by every fallback it tries)
  ctx.started = ctx.started or os.clock()
  local budget = ctx.budget or 0.6
  for _, frame in ipairs(frames) do
    if os.clock() - ctx.started > budget then break end
    for bi, b in ipairs(builders) do
      local pls = b.build(frame)
      local viol, pen, hard = judge(ctx, pls, ignore)
      if viol == 0 then return pls, frame, b, 0 end
      -- a thing in a wall or on no floor is never acceptable; things merely brushing each other are the last resort
      if hard == 0 then
        local score = viol * 1000 + pen
        if not best_score or score < best_score then best, best_score = { pls, frame, b }, score end
      end
    end
  end
  if not best then return nil end
  return best[1], best[2], best[3], best_score
end

-- ---------------------------------------------------------------------------------------------- arrangement builders
local function to_items(insts)
  local items = {}
  for i, inst in ipairs(insts) do items[i] = { w = inst.w, d = inst.d, h = inst.h, inst = inst } end
  return items
end

local function P(inst, pose, role, extra)
  local p = { inst = inst, kind = inst.kind, x = pose.x, y = pose.y, theta = pose.theta, zoff = pose.z or 0, role = role, w = inst.w, d = inst.d, h = inst.h }
  if extra then for k2, v2 in pairs(extra) do p[k2] = v2 end end
  return p
end

local function radius_of(pls)
  local r = 0
  for _, p in ipairs(pls) do
    for _, c in ipairs(Layout.corners(rect(p.x, p.y, p.w, p.d, p.theta))) do r = max(r, sqrt(c[1] * c[1] + c[2] * c[2])) end
  end
  return r
end

--- Split items over named sides, in order, for the sided arrangements.
local function split_sides(items, plan)
  local sides, idx = { back = {}, left = {}, right = {}, front = {} }, 1
  for _, s in ipairs(plan) do
    for _ = 1, s[2] do
      if items[idx] then table.insert(sides[s[1]], items[idx]); idx = idx + 1 end
    end
  end
  while items[idx] do table.insert(sides.back, items[idx]); idx = idx + 1 end
  return sides
end

local function wants_wall(insts)
  for _, inst in ipairs(insts) do
    if inst.k.wall ~= "prefer" then return false end
  end
  return true
end

--- A named arrangement of `insts` as a builder: build(frame) -> placements. `name` as the parser says it.
function Plan:arrangement(name, insts, rng, ctx, opts)
  opts = opts or {}
  local items = to_items(insts)
  local n = #items
  local cat = insts[1].k.category
  local gap = opts.gap or GAP[cat] or 8
  local wide = insts[1].w >= 40
  local desc
  local build
  local function from_poses(poses, role)
    local out = {}
    for i, it in ipairs(items) do out[#out + 1] = P(it.inst, poses[i] or poses[it], role) end
    return out
  end

  if name == "single" or name == "row" then
    local wall = opts.wall or wants_wall(insts)
    desc = n == 1 and "" or "in a row"
    build = function(frame) return from_poses(Layout.row(frame, items, { gap = gap, align_back = wall })) end
  elseif name == "circle" then
    desc = "in a circle, facing the middle"
    build = function(frame)
      return from_poses((Layout.circle(frame.x, frame.y, items, { gap = gap, start = frame.theta + 180, min_r = opts.min_r or 14 })))
    end
  elseif name == "grid" then
    local cols = opts.cols or (insts[1].k.seat and ceil(sqrt(n * 1.6)) or ceil(sqrt(n)))
    desc = "in a grid"
    build = function(frame) return from_poses((Layout.grid(frame, items, { gap = max(gap, 8), cols = cols }))) end
  elseif name == "cluster" then
    desc = "in a tight group"
    build = function(frame) return from_poses((Layout.grid(frame, items, { gap = 2 }))) end
  elseif name == "u_shape" then
    desc = "in a U shape, open toward you"
    local arm = max(1, floor(n / 3 + 0.5))
    local sides = split_sides(items, { { "back", n - 2 * arm }, { "left", arm }, { "right", arm } })
    build = function(frame)
      local poses = Layout.perimeter(frame, sides, { gap = gap, clear = 2, arm_align = "back", inner_w = opts.inner_w or 50 })
      return from_poses(poses)
    end
  elseif name == "l_shape" then
    desc = "in an L"
    local back = ceil(n / 2)
    local sides = split_sides(items, { { "back", back }, { "left", n - back } })
    build = function(frame)
      return from_poses((Layout.perimeter(frame, sides, { gap = gap, clear = 2, arm_align = "back" })))
    end
  elseif name == "square" then
    desc = "on four sides, facing the middle"
    local base, extra = floor(n / 4), n % 4
    local plan = {}
    for i, s in ipairs({ "back", "front", "left", "right" }) do plan[i] = { s, base + (i <= extra and 1 or 0) } end
    local sides = split_sides(items, plan)
    build = function(frame)
      return from_poses((Layout.perimeter(frame, sides, { gap = gap, clear = 2, inner_w = opts.inner_w or 50, inner_d = opts.inner_d or 50 })))
    end
  elseif name == "facing_each_other" or name == "two_rows" then
    desc = "in two rows facing each other"
    if n == 2 then desc = "facing each other" end
    local half = ceil(n / 2)
    local sides = split_sides(items, { { "left", half }, { "right", n - half } })
    local sep = opts.sep or (wide and 56 or 34)
    build = function(frame)
      return from_poses((Layout.perimeter(frame, sides, { gap = gap, clear = 0, arm_align = "center", inner_w = sep })))
    end
  elseif name == "stacked" then
    desc = n > 2 and "stacked in a pile" or "stacked"
    build = function(frame) return from_poses((Layout.stack(frame, items, { gap = 1 }))) end
  elseif name == "scattered" then
    desc = "scattered about"
    local area = 0
    for _, it in ipairs(items) do area = area + it.w * it.d end
    local radius = max(40, sqrt(area) * 1.15 + 12)
    local seed = rng.next() * 1e6
    build = function(frame)
      local r2 = Layout.rng(seed)
      return from_poses((Layout.scatter(frame, items, r2, { gap = 12, radius = radius, spin = insts[1].k.front_none })))
    end
  else
    return self:arrangement("row", insts, rng, ctx, opts)
  end
  local b = { desc = desc, name = name, build = build, wall = (name == "single" or name == "row") and (opts.wall or wants_wall(insts)) }
  b.radius = radius_of(b.build({ x = 0, y = 0, theta = 0 }))
  return b
end

--- The arrangement a bunch of one kind falls into when nobody says.
function Plan:default_arrangement(insts)
  local k = insts[1].k
  local n = #insts
  local wide = insts[1].w >= 40
  if n == 1 then return "single" end
  if k.seat then
    if n == 2 then return "facing_each_other" end
    if wide then
      if n == 3 then return "u_shape" elseif n == 4 then return "square" elseif n <= 12 then return "two_rows" end
      return "grid"
    end
    if n <= 12 then return "circle" end
    return "grid"
  end
  local c = k.category
  if c == "table" then
    if n <= 3 then return "row" end
    return "grid"
  elseif c == "bed" then
    return n <= 6 and "row" or "grid"
  elseif c == "storage" or c == "appliance" or c == "bath" then
    return "row"
  elseif c == "container" then
    if k.kind == "crate" or k.kind == "box" then return n <= 2 and "row" or "stacked" end
    if k.kind == "barrel" then return "cluster" end
    return "row"
  elseif c == "lighting" then
    return "row"
  elseif c == "decor" then
    return "scattered"
  end
  return "row"
end

-- ---------------------------------------------------------------------------------------------- relative placement
local function ref_of(e)
  return { x = e.x, y = e.y, theta = e.theta, w = e.w, d = e.d, h = e.h, z = e.z or 0, id = e.id, kind = e.kind, top = e.top, inst = e.inst }
end

local function seat_counts(sides_len, seat_w, gap, n, order)
  local cap, count = {}, {}
  for _, s in ipairs(order) do
    -- a thing wider than the side it stands at (a couch by a coffee table) still goes there, centred and overhanging
    cap[s] = max(1, floor((sides_len[s] + gap) / (seat_w + gap)))
    count[s] = 0
  end
  local placed = 0
  for _ = 1, n do
    local best, best_rel
    for _, s in ipairs(order) do
      if count[s] < cap[s] then
        local rel = (cap[s] - count[s]) / cap[s]
        if not best_rel or rel > best_rel + 1e-9 then best, best_rel = s, rel end
      end
    end
    if not best then break end
    count[best] = count[best] + 1
    placed = placed + 1
  end
  return count, placed, cap
end

--- Items around a reference footprint, facing it. Seats tuck their front edge under a table.
local function around_builder(ref, insts, opts)
  opts = opts or {}
  local items = to_items(insts)
  local seat = insts[1].k.seat
  local gap = opts.gap or 5
  local sides_len = { back = ref.w, front = ref.w, left = ref.d, right = ref.d }
  local order = opts.order or { "left", "right", "back", "front" }
  local counts, placed = seat_counts(sides_len, insts[1].w, gap, #items, order)
  if placed < #items then
    counts, placed = seat_counts(sides_len, insts[1].w, 1, #items, order)
    gap = 1
  end
  local sides, idx = { back = {}, left = {}, right = {}, front = {} }, 1
  for _, s in ipairs(order) do
    for _ = 1, counts[s] do
      if items[idx] then table.insert(sides[s], items[idx]); idx = idx + 1 end
    end
  end
  local dropped = #items - placed
  -- a chair slides its front edge under a table top; a couch or armchair at a low coffee table stands clear of it
  local topf = ref.topf or (ref.inst and top_frame(ref.inst.v))
  local chairlike = insts[1].kind == "chair" or insts[1].kind == "office chair" or insts[1].kind == "stool"
  local tuck = (seat and chairlike and topf and topf.z >= 28) and min(9, 0.35 * insts[1].d) or 0
  local b = { desc = "around the " .. (ref.kind or "middle"), dropped = dropped }
  b.build = function()
    local frame = { x = ref.x, y = ref.y, theta = ref.theta }
    local poses = Layout.perimeter(frame, sides, { gap = gap, clear = (tuck > 0) and -tuck or 3, inner_w = ref.w, inner_d = ref.d })
    local out = {}
    for _, it in ipairs(items) do
      if poses[it] then out[#out + 1] = P(it.inst, poses[it], "around", { tuck = ref.placement or ref.id, tuck_group = true, sits_on = ref.id, tuck_depth = tuck }) end
    end
    return out
  end
  b.radius = 0
  return b
end

--- Items beside / in front of / behind a reference, set off from it and aligned with it.
local function offset_builder(ref, insts, relation, opts)
  opts = opts or {}
  local items = to_items(insts)
  local gap = opts.gap or 6
  local desc = ({ beside = "beside the ", in_front_of = "in front of the ", behind = "behind the " })[relation] .. (ref.kind or "it")
  local b = { desc = desc }
  b.build = function(frame)
    local out = {}
    local rf = { x = ref.x, y = ref.y, theta = ref.theta }
    if relation == "beside" then
      local used = { [1] = 0, [-1] = 0 }
      for i, it in ipairs(items) do
        local side = (i % 2 == 1) and 1 or -1
        if opts.flip then side = -side end
        local v = side * (ref.w / 2 + gap + used[side] + it.w / 2)
        used[side] = used[side] + it.w + gap
        local pose = to_world(rf, 0, v, 0)
        pose.theta = ref.theta
        out[#out + 1] = P(it.inst, pose, "beside")
      end
    elseif relation == "in_front_of" then
      local dist = opts.distance
      local total = 0
      for i, it in ipairs(items) do total = total + it.w + (i > 1 and gap or 0) end
      local v = -total / 2
      for _, it in ipairs(items) do
        local u = ref.d / 2 + (dist or gap) + it.d / 2
        local pose = to_world(rf, u, v + it.w / 2, 180)
        out[#out + 1] = P(it.inst, pose, "in_front_of")
        v = v + it.w + gap
      end
    else
      local total = 0
      for i, it in ipairs(items) do total = total + it.w + (i > 1 and gap or 0) end
      local v = -total / 2
      for _, it in ipairs(items) do
        local u = -(ref.d / 2 + gap + it.d / 2)
        local pose = to_world(rf, u, v + it.w / 2, 0)
        out[#out + 1] = P(it.inst, pose, "behind")
        v = v + it.w + gap
      end
    end
    return out
  end
  b.radius = 0
  return b
end

--- Items standing on a reference's flat top.
local function on_builder(ref, insts, opts)
  opts = opts or {}
  local items = to_items(insts)
  local top = ref.topf
  local b = { desc = "on the " .. (ref.kind or "surface") }
  b.build = function()
    local rf = { x = ref.x, y = ref.y, theta = ref.theta }
    local c = to_world(rf, top.u, top.v, 0)
    local frame = { x = c.x, y = c.y, theta = ref.theta, w = top.w, d = top.d, z = top.z + (ref.zoff or 0) }
    local poses = Layout.on_top({ x = c.x, y = c.y, w = top.w, d = top.d, theta = ref.theta, z = frame.z }, items, { gap = 4, bias = opts.bias or -1 })
    local out = {}
    for i, it in ipairs(items) do out[#out + 1] = P(it.inst, poses[i], "on", { sits_on = ref.id }) end
    return out
  end
  b.radius = 0
  return b
end

-- ---------------------------------------------------------------------------------------------- rooms
local ROLE_ALIAS = { on_desk = "on", on_wall_unit = "on", beside_bed = "beside_bed", at_desk = "at_desk", row_wall = "row_wall" }
local SURFACE_FOR = { computer = "desk", phone = "desk", ["desk lamp"] = "side table", mug = "coffee table", microwave = "table" }

local function infer_role(k, has)
  local kind = k.kind
  if kind == "tv" then return "focus" end
  if kind == "desk" then return "desk" end
  if k.category == "bed" then return "wall_head" end
  if kind == "side table" then return has.bed and "beside_bed" or "beside_main" end
  if k.category == "table" then return "center" end
  if k.seat then
    if kind == "office chair" and has.desk then return "at_desk" end
    if kind == "chair" or kind == "stool" or kind == "office chair" then
      if has.center then return "around" end
      if has.desk then return "at_desk" end
      return "side_seat"
    end
    if kind == "armchair" then return "side_seat" end
    return "main_seat"
  end
  if kind == "computer" or kind == "phone" or kind == "desk lamp" or kind == "mug" then return "on" end
  if kind == "microwave" and (has.center or has.desk) then return "on" end
  if kind == "floor lamp" or kind == "plant" or kind == "trash can" then return "corner" end
  return "wall"
end

local function bbox_extent(pls)
  local x0, x1, y0, y1 = 1e9, -1e9, 1e9, -1e9
  for _, p in ipairs(pls) do
    for _, c in ipairs(Layout.corners(rect(p.x, p.y, p.w, p.d, p.theta))) do
      x0, x1, y0, y1 = min(x0, c[1]), max(x1, c[1]), min(y0, c[2]), max(y1, c[2])
    end
  end
  return x0, x1, y0, y1
end

--- A furnished room from role-tagged entries: walls (back, left, right; the front faces the viewer) carry the wall
--- things, a cluster in the middle carries the table and the seats, corners take the lamps and plants.
--- entries = { {inst=, role=}, ... }.  Returns builder with .build(frame) and .size = {w, d} (the room).
function Plan:room(entries, rng, ctx, opts)
  opts = opts or {}
  local by = {}
  for _, e in ipairs(entries) do
    by[e.role] = by[e.role] or {}
    table.insert(by[e.role], e)
  end
  local function list(role) return by[role] or {} end
  local function insts_of(role)
    local t = {}
    for _, e in ipairs(list(role)) do t[#t + 1] = e.inst end
    return t
  end
  local has_focus, has_main = #list("focus") > 0, #list("main_seat") > 0

  -- -- the centre cluster: the table (or coffee table), the seats round it
  local centre = list("center")[1]
  local cluster = {}                                    -- builder(frame) -> placements, in the room frame
  local conv_seats = {}
  for _, e in ipairs(list("main_seat")) do conv_seats[#conv_seats + 1] = e end
  for _, e in ipairs(list("side_seat")) do conv_seats[#conv_seats + 1] = e end
  local around = list("around")
  local desk = list("desk")[1]

  local function cluster_build(frame)
    local out = {}
    local centre_p
    local ph = has_focus and 90 or 0                    -- with a screen on the left wall the table runs along u
    if centre then
      centre_p = P(centre.inst, to_world(frame, 0, 0, ph), "center")
      out[#out + 1] = centre_p
    end
    local ref = centre_p
    -- conversation seats: main seats take the right side (opposite the screen), side seats the back and front
    if #conv_seats > 0 then
      local items = {}
      for i, e in ipairs(conv_seats) do items[i] = { w = e.inst.w, d = e.inst.d, inst = e.inst, role = e.role } end
      local order = has_focus and { "right", "back", "front" } or { "back", "left", "right", "front" }
      local sides = { back = {}, left = {}, right = {}, front = {} }
      local mains, sides_s = {}, {}
      for _, it in ipairs(items) do if it.role == "main_seat" then mains[#mains + 1] = it else sides_s[#sides_s + 1] = it end end
      local oi = 1
      local function put(it, prefer)
        local s = prefer or order[oi]
        table.insert(sides[s], it)
        if not prefer then oi = oi % #order + 1 end
      end
      for i, it in ipairs(mains) do put(it, (i == 1) and order[1] or nil) end
      if has_focus then
        -- side seats alternate back / front so they flank the table
        local flip = { "back", "front" }
        for i, it in ipairs(sides_s) do put(it, flip[(i - 1) % 2 + 1]) end
      else
        -- no screen: main seats on the back, armchairs on the sides facing in
        for _, it in ipairs(sides_s) do put(it, (#sides.left <= #sides.right) and "left" or "right") end
      end
      local iw = ref and (ph == 90 and ref.d or ref.w) or 36
      local id = ref and (ph == 90 and ref.w or ref.d) or 36
      local poses = Layout.perimeter(frame, sides, { gap = 8, clear = 4, arm_align = "center", inner_w = iw + 10, inner_d = id + 10 })
      for _, it in ipairs(items) do
        if poses[it] then out[#out + 1] = P(it.inst, poses[it], it.role) end
      end
    end
    -- chairs round a table
    if #around > 0 and ref then
      local seats = {}
      for i, e in ipairs(around) do seats[i] = e.inst end
      local ab = around_builder({ x = ref.x, y = ref.y, theta = frame.theta + ph, w = (ph == 90 and ref.d or ref.w) * 0 + ref.w, d = ref.d, kind = ref.inst.kind, placement = ref, inst = ref.inst },
        seats, { gap = 5 })
      if ph == 90 then
        -- the table was turned by 90 degrees: express its footprint in the turned frame
        ab = around_builder({ x = ref.x, y = ref.y, theta = Layout.wrap(frame.theta + ph), w = ref.inst.w, d = ref.inst.d, kind = ref.inst.kind, placement = ref, inst = ref.inst },
          seats, { gap = 5 })
      end
      for _, p in ipairs(ab.build()) do out[#out + 1] = p end
    end
    -- a desk and the chair at it
    if desk then
      local dp = P(desk.inst, to_world(frame, 0, 0, 0), "desk")
      out[#out + 1] = dp
      for _, e in ipairs(list("at_desk")) do
        local dtop = top_frame(desk.inst.v)
        local tuck = (dtop and dtop.z >= 28) and min(9, 0.35 * e.inst.d) or 0
        local u = desk.inst.d / 2 + e.inst.d / 2 - tuck
        local pose = to_world({ x = dp.x, y = dp.y, theta = dp.theta }, u, 0, 180)
        out[#out + 1] = P(e.inst, pose, "at_desk", { tuck = dp, tuck_group = true, sits_on = nil, tuck_depth = tuck })
      end
    end
    return out
  end

  -- -- sizing: measure the cluster once in a neutral frame
  local probe = cluster_build({ x = 0, y = 0, theta = 0 })
  local cx0, cx1, cy0, cy1 = bbox_extent(probe)
  local cu, cv = (cx1 - cx0), (cy1 - cy0)               -- theta 0: world x is u, world y is -v
  local cv_left = cy1                                    -- how far the cluster reaches toward -v (v = -y in this probe)

  -- -- wall residents
  local back, left, right = {}, {}, {}
  local function to_wall(wall, e) table.insert(wall, e) end
  for _, e in ipairs(list("wall_head")) do to_wall(back, e) end
  for _, e in ipairs(list("row_wall")) do to_wall(back, e) end
  if has_focus then for _, e in ipairs(list("focus")) do to_wall(left, e) end
  else for _, e in ipairs(list("focus")) do to_wall(back, e) end end
  if not has_focus and not centre then
    for _, e in ipairs(list("main_seat")) do to_wall(back, e) end
  end
  local wall_items = list("wall")
  local function wall_len(w)
    local s = 0
    for i, e in ipairs(w) do s = s + e.inst.w + (i > 1 and 2 or 0) end
    return s
  end
  local back_cap = max(330, cv + 80)
  for _, e in ipairs(wall_items) do
    if wall_len(back) + e.inst.w + 2 <= back_cap then to_wall(back, e)
    elseif wall_len(right) <= wall_len(left) then to_wall(right, e) else to_wall(left, e) end
  end
  local function depth_of(w)
    local d = 0
    for _, e in ipairs(w) do d = max(d, e.inst.d) end
    return d
  end
  local db, dl, dr = depth_of(back), depth_of(left), depth_of(right)
  local W = max(opts.min_w or 230, wall_len(back) + 60, dl + 34 + cv + 34 + dr, 120 + cv)
  local D = max(opts.min_d or 190, db + 34 + cu + 40, wall_len(left) + 60, wall_len(right) + 60)
  if #list("wall_head") + (desk and 1 or 0) > 0 then D = max(D, db + 34 + cu + 60) end
  do
    local cw = 0
    for _, e in ipairs(list("corner")) do cw = max(cw, e.inst.w, e.inst.d) end
    if cw > 0 and wall_len(back) > 0 then W = max(W, wall_len(back) + 2 * cw + 12) end
  end
  if opts.room then W, D = opts.room.w, opts.room.d end

  local b = { desc = opts.desc or "as a room", size = { w = W, d = D } }

  -- things that slide along their wall to get out of each other's way
  local MOVABLE = { wall = true, corner = true, beside_bed = true, beside_main = true, focus = true }
  local function repair(out, frame)
    local fx0, fy0 = fwd(frame.theta)
    local function clashes(p, i)
      local r = rect(p.x, p.y, p.w, p.d, p.theta)
      for j, q in ipairs(out) do
        if j ~= i and (q.zoff or 0) <= 0.01 and (p.zoff or 0) <= 0.01
           and not (p.tuck == q or q.tuck == p or (p.tuck and p.tuck == q.tuck and p.tuck_group)) then
          if Layout.overlap(r, rect(q.x, q.y, q.w, q.d, q.theta), 0.4) then return true end
        end
      end
      return false
    end
    local function inside(p)
      for _, c in ipairs(Layout.corners(rect(p.x, p.y, p.w, p.d, p.theta))) do
        local dx, dy = c[1] - frame.x, c[2] - frame.y
        if abs(dx * fx0 + dy * fy0) > D / 2 + 0.5 or abs(dx * fy0 - dy * fx0) > W / 2 + 0.5 then return false end
      end
      return true
    end
    for i, p in ipairs(out) do
      if MOVABLE[p.role] and (p.zoff or 0) <= 0.01 and clashes(p, i) then
        local tx, ty = Layout.right(p.theta)
        local dirs = { { tx, ty } }
        if p.role == "corner" then
          local gx, gy = Layout.right(frame.theta)
          dirs = { { gx, gy }, { fx0, fy0 } }
        end
        local x0, y0 = p.x, p.y
        local moved = false
        for k = 1, 30 do
          for _, dir in ipairs(dirs) do
            for _, sgn in ipairs({ 1, -1 }) do
              p.x, p.y = x0 + dir[1] * sgn * k * 8, y0 + dir[2] * sgn * k * 8
              if inside(p) and not clashes(p, i) then moved = true; break end
            end
            if moved then break end
          end
          if moved then break end
        end
        if not moved then p.x, p.y = x0, y0 end
      end
    end
  end
  -- the same room, reflected left to right
  local function mirror(out, frame)
    local fx0, fy0 = fwd(frame.theta)
    for _, p in ipairs(out) do
      local dx, dy = p.x - frame.x, p.y - frame.y
      local np = to_world(frame, dx * fx0 + dy * fy0, -(dx * fy0 - dy * fx0), -Layout.angle_diff(p.theta, frame.theta))
      p.x, p.y, p.theta = np.x, np.y, np.theta
    end
  end

  local function raw_build(frame)
    local out = {}
    local cu_c = (db - 0) / 2                            -- the cluster sits between the back wall's depth and the front
    -- the cluster sits in the middle of the free floor between the back wall's things and the front edge
    local c_lo = -D / 2 + db + 34 + cu / 2
    local c_hi = D / 2 - cu / 2 - 20
    local c_u = max(c_lo, min(c_hi, db / 2))
    if c_u > c_hi then c_u = c_hi end
    -- with a screen on the left wall the cluster moves toward it, to a viewing distance, instead of floating mid-room
    local c_v = 0
    if has_focus then
      local want = -W / 2 + dl + 90 + cv_left
      c_v = min(0, want)
    end
    local cf = to_world(frame, c_u + (opts.shift and opts.shift[1] or 0), c_v + (opts.shift and opts.shift[2] or 0), 0)
    for _, p in ipairs(cluster_build({ x = cf.x, y = cf.y, theta = frame.theta })) do out[#out + 1] = p end
    -- the back wall: a row, flush
    local function run(wall, fixed_axis, sign, phi, centre_on)
      local total = wall_len(wall)
      local pos = -total / 2
      if centre_on then pos = centre_on - total / 2 end
      for _, e in ipairs(wall) do
        local c = pos + e.inst.w / 2
        local pose
        if fixed_axis == "u" then pose = to_world(frame, -D / 2 + e.inst.d / 2 + 1, c, phi)
        else pose = to_world(frame, c, sign * (W / 2 - e.inst.d / 2 - 1), phi) end
        out[#out + 1] = P(e.inst, pose, e.role)
        pos = pos + e.inst.w + 2
      end
    end
    run(back, "u", 0, 0)
    run(left, "v", -1, -90, c_u)
    run(right, "v", 1, 90, c_u)
    -- corners: back-left, back-right, front-left, front-right
    local corner_items = list("corner")
    local slots = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } }
    for i, e in ipairs(corner_items) do
      local s = slots[(i - 1) % 4 + 1]
      -- a thing that looks the same from every side keeps the room's own alignment; the rest turn to face the middle
      local sym = e.inst.v.symmetric
      local r = (sym and max(e.inst.w, e.inst.d) / 2 or sqrt(e.inst.w ^ 2 + e.inst.d ^ 2) / 2) + 2
      local pose = to_world(frame, s[1] * (D / 2 - r), s[2] * (W / 2 - r), 0)
      pose.theta = sym and frame.theta or Layout.angle_of(frame.x - pose.x, frame.y - pose.y)
      out[#out + 1] = P(e.inst, pose, "corner")
    end
    -- beside things: end tables by the bed head, by the main seat
    for _, e in ipairs(list("beside_bed")) do
      local bed
      for _, p in ipairs(out) do if p.role == "wall_head" then bed = p; break end end
      if bed then
        local used = 0
        for _, q in ipairs(out) do if q.role == "beside_bed" and q.bedside == bed then used = used + 1 end end
        local side = (used % 2 == 0) and 1 or -1
        local v = side * (bed.w / 2 + e.inst.w / 2 + 4 + floor(used / 2) * (e.inst.w + 4))
        local pose = to_world({ x = bed.x, y = bed.y, theta = bed.theta }, -(bed.d / 2 - e.inst.d / 2), v, 0)
        pose.theta = bed.theta
        out[#out + 1] = P(e.inst, pose, "beside_bed", { bedside = bed })
      else
        local pose = to_world(frame, 0, 0, 0)
        out[#out + 1] = P(e.inst, pose, "beside_bed")
      end
    end
    for _, e in ipairs(list("beside_main")) do
      local main
      for _, p in ipairs(out) do if p.role == "main_seat" then main = p; break end end
      local pose
      if main then pose = to_world({ x = main.x, y = main.y, theta = main.theta }, 0, main.w / 2 + e.inst.w / 2 + 4, 0); pose.theta = main.theta
      else pose = to_world(frame, 0, 0, 0) end
      out[#out + 1] = P(e.inst, pose, "beside_main")
    end
    -- things standing on surfaces
    local surfaces = {}
    for _, p in ipairs(out) do
      local tf = top_frame(p.inst.v)
      if tf and (p.zoff or 0) == 0 then surfaces[#surfaces + 1] = { p = p, tf = tf } end
    end
    local groups = {}
    for _, e in ipairs(list("on")) do
      local want = SURFACE_FOR[e.inst.kind]
      local pick
      for _, s in ipairs(surfaces) do
        if s.p.kind == want then pick = s; break end
      end
      if not pick then
        for _, s in ipairs(surfaces) do
          if s.p.role == "desk" or s.p.role == "center" then pick = s; break end
        end
      end
      pick = pick or surfaces[1]
      if pick then
        local g = groups[pick] or {}
        groups[pick] = g
        g[#g + 1] = e
      else
        out[#out + 1] = P(e.inst, to_world(frame, 0, 0, 0), "on")
      end
    end
    for s, es in pairs(groups) do
      local ins = {}
      for i, e in ipairs(es) do ins[i] = e.inst end
      local ob = on_builder({ x = s.p.x, y = s.p.y, theta = s.p.theta, kind = s.p.kind, topf = s.tf, id = nil, zoff = s.p.zoff }, ins,
        { bias = (s.p.role == "desk") and -1 or 0 })
      for _, p in ipairs(ob.build()) do p.on_placement = s.p; out[#out + 1] = p end
    end
    return out
  end
  b.build = function(frame)
    local out = raw_build(frame)
    repair(out, frame)
    if opts.mirror then mirror(out, frame) end
    return out
  end
  b.radius = sqrt((W / 2) ^ 2 + (D / 2) ^ 2)
  return b
end

-- ---------------------------------------------------------------------------------------------- the planner proper
function Plan.new(catalog)
  local self = setmetatable({}, Plan)
  self.catalog = catalog
  self.kinds = catalog.kinds
  return self
end

local function finish(self, ctx, pls, res, group_id)
  for _, p in ipairs(pls) do
    local v = p.inst.v
    local zbot = (p.fz or ctx.pos.z) + (p.zoff or 0)
    local origin, yaw = Plan.entity_pose(v, p.x, p.y, zbot, p.theta)
    res.placements[#res.placements + 1] = {
      kind = p.kind, vi = p.inst.vi, model = v.model, x = p.x, y = p.y, z = zbot, theta = Layout.wrap(p.theta),
      w = p.w, d = p.d, h = p.h, origin = origin, yaw = yaw, mass = v.mass, material = v.material, color = p.inst.color,
      color_name = p.inst.color_name, role = p.role, group = group_id, skins = v.skins, top = v.top,
      on = ((p.zoff or 0) > 0.01 or p.on_placement) and true or nil, symmetric = v.symmetric,
    }
  end
end

local function aim_or_front(ctx)
  return ctx.aim or ctx.pos
end

--- Everything standing that matches a spec (kind + colour), nearest to `from` first.
local function registry_matches(ctx, kind, adjs, from, list)
  local out = {}
  local _, color_name = color_of(adjs)
  for _, e in ipairs(list or ctx.registry or {}) do
    if e.kind == kind and (not color_name or not e.color_name or e.color_name == color_name) then out[#out + 1] = e end
  end
  table.sort(out, function(a, b)
    local da = (a.x - from.x) ^ 2 + (a.y - from.y) ^ 2
    local db = (b.x - from.x) ^ 2 + (b.y - from.y) ^ 2
    if da ~= db then return da < db end
    return (a.seq or 0) > (b.seq or 0)
  end)
  return out
end

local function say_list(parts)
  if #parts == 0 then return "" end
  if #parts == 1 then return parts[1] end
  return table.concat(parts, ", ", 1, #parts - 1) .. " and " .. parts[#parts]
end

--- The things a request wants, expanded (scenes become their furniture) and checked against the library.
function Plan:wanted(req, notes)
  local want, seen_unknown = {}, {}
  for _, it in ipairs(req.items or {}) do
    local kind = it.kind
    if tostring(kind):match("^UNKNOWN") then
      local name = it.phrase or kind:sub(9)
      if not seen_unknown[name] then
        seen_unknown[name] = true
        notes[#notes + 1] = "I don't have " .. article(name) .. " in my library"
      end
    elseif kind == "@last" then
      -- only meaningful for management verbs
    else
      local n = it.count
      if n == 0 then n = 1 end
      if n > MAX_PER_KIND then
        notes[#notes + 1] = "I limited " .. plural(kind, 2) .. " to " .. MAX_PER_KIND
        n = MAX_PER_KIND
      end
      if it.approx then notes[#notes + 1] = "closest I have to a " .. tostring(it.word) .. " is a " .. kind end
      want[#want + 1] = { kind = kind, count = n, adjectives = it.adjectives or {}, word = it.word, plural = it.plural }
    end
  end
  if req.scene and req.scene ~= "" then
    for _, line in ipairs(self.catalog.recipes[req.scene] or {}) do
      want[#want + 1] = { kind = line[2], count = line[1], adjectives = {}, role = line[3], scene = req.scene }
    end
  end
  return want
end

function Plan:add(req, ctx)
  local res = { action = "add", placements = {}, notes = {}, groups = {} }
  local notes = res.notes
  local rng = Layout.rng(ctx.seed or Layout.hash(req.raw or ""))
  local want = self:wanted(req, notes)
  if #want == 0 then
    res.say = (#notes > 0) and (notes[1] .. ", so I added nothing.") or "There was nothing in that to add."
    return res
  end

  -- merge the same kind asked for twice; keep scene roles
  local total = 0
  for _, w in ipairs(want) do total = total + w.count end
  if total > MAX_TOTAL then
    local f = MAX_TOTAL / total
    for _, w in ipairs(want) do w.count = max(1, floor(w.count * f)) end
    notes[#notes + 1] = "that was a lot, so I scaled it down to about " .. MAX_TOTAL .. " things"
  end

  -- the relation (if any) decides the mode
  local arr, target = req.arrangement or "unspecified", req.target or ""
  local kinds_present, order = {}, {}
  for _, w in ipairs(want) do
    if not kinds_present[w.kind] then kinds_present[w.kind] = true; order[#order + 1] = w.kind end
  end

  -- instances
  local insts_by_want = {}
  for wi, w in ipairs(want) do
    local vis = self:pick_variants(w.kind, w.count, w.adjectives, rng)
    local list = {}
    for i = 1, w.count do list[i] = make_inst(self, w.kind, vis[i], w.adjectives) end
    insts_by_want[wi] = list
  end
  local function all_insts()
    local t = {}
    for _, l in ipairs(insts_by_want) do for _, i in ipairs(l) do t[#t + 1] = i end end
    return t
  end
  local group_desc, builders, frames, wallish

  local function wide_radius(bs)
    local r = 0
    for _, b in ipairs(bs) do r = max(r, b.radius or 0) end
    return r
  end

  local function theta_for_player(cx, cy) return Layout.angle_of(ctx.pos.x - cx, ctx.pos.y - cy) end
  local function theta_square(cx, cy) return player_frame_theta(ctx) end

  local insts = all_insts()
  local spec_by_inst = {}
  for wi, l in ipairs(insts_by_want) do for _, i in ipairs(l) do spec_by_inst[i] = want[wi] end end

  -- a mode for relative-to-something requests
  local referent
  local relational = (arr == "around" or arr == "beside" or arr == "in_front_of" or arr == "behind" or arr == "on" or arr == "facing_me")
                     and target ~= "" and target ~= "me" and target ~= "here" and target ~= "wall" and target ~= "corner"
                     and target ~= "window" and target ~= "door" and target ~= "ceiling" and target ~= "floor" and target ~= "room"
  local on_me = (target == "me" or target == "here")

  local function auto_referent(kind_id, n_around)
    -- reuse what stands near the player, else make one
    local near = registry_matches(ctx, kind_id, {}, aim_or_front(ctx))
    if near[1] and ((near[1].x - ctx.pos.x) ^ 2 + (near[1].y - ctx.pos.y) ^ 2) < 1600 ^ 2 then
      local e = near[1]
      local ref = ref_of(e)
      ref.inst = nil
      ref.topf = e.vi and self.kinds[e.kind].variants[e.vi] and top_frame(self.kinds[e.kind].variants[e.vi]) or nil
      ref.exists = true
      return ref
    end
    return nil
  end

  local notes_after = {}
  local desc_parts = {}

  if relational and kinds_present[target] and #order == 1 then
    -- "add a table around a table": nonsense; treat as no target
    relational = false
    arr = "unspecified"
  end

  if relational then
    -- items that are not the target itself
    local rel_insts, target_insts = {}, {}
    for _, i in ipairs(insts) do
      if i.kind == target then target_insts[#target_insts + 1] = i else rel_insts[#rel_insts + 1] = i end
    end
    local ref = auto_referent(target)
    local created
    if not ref and #target_insts > 0 then
      -- the target is in the request: create it first, as the main thing
      local tb = self:arrangement("single", { target_insts[1] }, rng, ctx, {})
      created = target_insts[1]
      ref = { kind = target, inst = created, w = created.w, d = created.d, h = created.h, theta = 0, x = 0, y = 0, topf = top_frame(created.v), created = true }
    elseif not ref then
      -- the target is not in the request: make one (a table for chairs "around the table")
      local kind_id = target
      local k = self.kinds[kind_id]
      if not k then
        notes[#notes + 1] = "I don't know what '" .. target .. "' is, so I used the middle of the room"
        relational = false
      else
        local n_rel = #rel_insts
        local pick
        if k.category == "table" and (arr == "around") then
          -- the smallest table that seats them all
          local best_cap, best
          for vi, v in ipairs(k.variants) do
            local len = 2 * (v.w + v.d)
            local cap = floor(len / (rel_insts[1].w + 5))
            if cap >= n_rel and (not best or v.w * v.d < best_cap) then best, best_cap = vi, v.w * v.d end
          end
          if not best and kind_id == "table" and self.kinds["meeting table"] then
            kind_id, k = "meeting table", self.kinds["meeting table"]
            best = 1
          end
          pick = best or 1
        else
          pick = self:pick_variants(kind_id, 1, {}, rng)[1]
        end
        created = make_inst(self, kind_id, pick, {})
        ref = { kind = kind_id, inst = created, w = created.w, d = created.d, h = created.h, theta = 0, x = 0, y = 0, topf = top_frame(created.v), created = true }
        notes[#notes + 1] = "there was no " .. target .. ", so I added one"
      end
    end
    if relational then
      local rel = (arr == "facing_me") and "in_front_of" or arr
      -- the builders work in the reference's own frame; create a builder over (ref pose) that recomputes per frame
      local function make_builders()
        local bs = {}
        local function with_ref(frame)
          local r = {}
          for k2, v2 in pairs(ref) do r[k2] = v2 end
          if ref.created then
            r.x, r.y, r.theta = frame.x, frame.y, frame.theta
          end
          return r
        end
        local function mk(kind_rel, opts2)
          local b = { desc = nil, radius = 0 }
          local probe_ref = { x = 0, y = 0, theta = 0, w = ref.w, d = ref.d, kind = ref.kind, inst = ref.inst, topf = ref.topf, h = ref.h, id = ref.id }
          local inner
          if rel == "around" then inner = around_builder(probe_ref, rel_insts, opts2)
          elseif rel == "on" and ref.topf then inner = on_builder(probe_ref, rel_insts, opts2)
          elseif rel == "on" then inner = offset_builder(probe_ref, rel_insts, "beside", opts2)
          else
            local dist = opts2 and opts2.distance
            if rel == "in_front_of" and ref.kind == "tv" then dist = max(dist or 0, 70, 1.6 * ref.w) end
            inner = offset_builder(probe_ref, rel_insts, rel, { flip = opts2 and opts2.flip, distance = dist })
          end
          b.desc = inner.desc
          b.dropped = inner.dropped
          b.build = function(frame)
            local out = {}
            local r = with_ref(frame)
            local ib
            if rel == "around" then ib = around_builder(r, rel_insts, opts2)
            elseif rel == "on" and r.topf then ib = on_builder(r, rel_insts, opts2)
            elseif rel == "on" then ib = offset_builder(r, rel_insts, "beside", opts2)
            else
              local dist = opts2 and opts2.distance
              if rel == "in_front_of" and r.kind == "tv" then dist = max(dist or 0, 70, 1.6 * r.w) end
              ib = offset_builder(r, rel_insts, rel, { flip = opts2 and opts2.flip, distance = dist })
            end
            if ref.created then
              local rp = P(ref.inst, { x = r.x, y = r.y, theta = r.theta, z = 0 }, "referent")
              out[#out + 1] = rp
              r.placement = rp
            end
            for _, p in ipairs(ib.build(frame)) do
              if p.tuck == nil and rel == "around" then p.tuck = out[1] end
              out[#out + 1] = p
            end
            return out
          end
          local probe = b.build({ x = 0, y = 0, theta = 0 })
          b.radius = radius_of(probe)
          return b
        end
        bs[#bs + 1] = mk(rel, {})
        if rel == "beside" then bs[#bs + 1] = mk(rel, { flip = true }) end
        if rel == "in_front_of" then
          for _, d in ipairs({ 30, 60, 100, 160 }) do bs[#bs + 1] = mk(rel, { distance = d }) end
        end
        return bs
      end
      builders = make_builders()
      group_desc = builders[1].desc
      if builders[1].dropped and builders[1].dropped > 0 then
        notes[#notes + 1] = "only " .. (#rel_insts - builders[1].dropped) .. " fit around it"
      end
      if ref.exists then
        -- an existing referent fixes the frame: try its own pose, not the player's view
        frames = { { x = ref.x, y = ref.y, theta = ref.theta } }
        local pls, frame, b, score = fit(ctx, builders, frames)
        if pls and score == 0 then
          finish(self, ctx, pls, res, 1)
          res.say = say_list({ count_word(#rel_insts) .. " " .. plural(rel_insts[1].kind, #rel_insts) }) .. " " .. (b.desc or group_desc or "") .. "."
          if #notes > 0 then res.say = res.say .. " " .. table.concat(notes, "; ") .. "." end
          return res
        end
        -- no clean room against it: the same things, free-standing nearby (never on top of what stands)
        notes[#notes + 1] = "there was no room " .. (arr == "around" and "around" or arr:gsub("_", " ")) .. " the " .. target .. ", so I put them nearby"
        relational = false
        arr, target = "unspecified", ""
      end
      if relational then
      frames = open_frames(ctx, wide_radius(builders), theta_square)
      local wallf
      if #rel_insts > 0 and ref.kind and self.kinds[ref.kind] and self.kinds[ref.kind].wall == "prefer" and ref.created then
        wallf = wall_frames(ctx, wide_radius(builders) * 2, 900)
      end
      if wallf and #wallf > 0 then
        for _, f in ipairs(frames) do wallf[#wallf + 1] = f end
        frames = wallf
      end
      local pls, frame, b, score = fit(ctx, builders, frames)
      if pls then
        finish(self, ctx, pls, res, 1)
        local parts = {}
        local counts, corder = {}, {}
        for _, p in ipairs(res.placements) do
          if p.role ~= "referent" then
            counts[p.kind] = (counts[p.kind] or 0) + 1
            if not corder[p.kind] then corder[#corder + 1] = p.kind; corder[p.kind] = true end
          end
        end
        for _, kname in ipairs(corder) do parts[#parts + 1] = count_word(counts[kname]) .. " " .. plural(kname, counts[kname]) end
        res.say = say_list(parts) .. " " .. (group_desc or "") .. "."
        if score and score > 0 then notes[#notes + 1] = "it was a tight squeeze" end
        if #notes > 0 then res.say = res.say .. " " .. table.concat(notes, "; ") .. "." end
        return res
      end
      res.say = "I could not find clear floor for that."
      return res
      end
    end
  end

  -- -- relative to the player
  if on_me and (arr == "around" or arr == "in_front_of" or arr == "behind" or arr == "beside" or arr == "facing_me" or arr == "unspecified") then
    local pos = ctx.pos
    local b
    if arr == "around" then
      b = { desc = "around you" }
      local items = to_items(insts)
      b.build = function()
        local poses = Layout.circle(pos.x, pos.y, items, { gap = 8, min_r = 46, start = ctx.yaw })
        local out = {}
        for i, it in ipairs(items) do out[#out + 1] = P(it.inst, poses[i], "around_me") end
        return out
      end
      b.radius = 0
    else
      local base_arr = self:default_arrangement(insts)
      if base_arr == "single" then base_arr = "row" end
      b = self:arrangement(base_arr, insts, rng, ctx, {})
      group_desc = b.desc
      local inner = b.build
      local side = (arr == "behind") and -1 or 1
      local dist_off = 0
      b.build = function(frame)
        return inner(frame)
      end
      desc_parts[#desc_parts + 1] = ({ in_front_of = "in front of you", behind = "behind you", beside = "beside you" })[arr] or "facing you"
    end
    builders = { b }
    local pls, frame, bb, score
    if arr == "around" then
      pls, frame, bb, score = fit(ctx, builders, { { x = pos.x, y = pos.y, theta = ctx.yaw } })
    else
      local r = b.radius or 40
      local fx, fy = fwd(ctx.yaw + ((arr == "behind") and 180 or 0))
      local rx, ry = fy, -fx
      local frames2 = {}
      for k = 0, 14 do
        for _, lat in ipairs((arr == "beside") and { 1, -1, 2, -2 } or { 0, 1, -1, 2, -2 }) do
          local dist = (arr == "beside") and 0 or (r + 60 + k * 28)
          local latd = (arr == "beside") and (r + 50 + k * 28) * lat or lat * 55
          local cx, cy = pos.x + fx * dist + rx * latd, pos.y + fy * dist + ry * latd
          frames2[#frames2 + 1] = { x = cx, y = cy, theta = Layout.angle_of(pos.x - cx, pos.y - cy) }
        end
      end
      pls, frame, bb, score = fit(ctx, builders, frames2)
    end
    if pls then
      finish(self, ctx, pls, res, 1)
      local counts, corder = {}, {}
      for _, p in ipairs(res.placements) do
        counts[p.kind] = (counts[p.kind] or 0) + 1
        if not corder[p.kind] then corder[#corder + 1] = p.kind; corder[p.kind] = true end
      end
      local parts = {}
      for _, kname in ipairs(corder) do parts[#parts + 1] = count_word(counts[kname]) .. " " .. plural(kname, counts[kname]) end
      res.say = say_list(parts) .. " " .. (desc_parts[1] or b.desc or "") .. "."
      if #notes > 0 then res.say = res.say .. " " .. table.concat(notes, "; ") .. "." end
      return res
    end
    res.say = "I could not find clear floor around you."
    return res
  end

  -- -- the surface a small thing needs
  local function needs_surface(kind_id)
    return SURFACE_FOR[kind_id] ~= nil and kind_id ~= "microwave"
  end
  local function has_surface_kind()
    for _, w in ipairs(want) do
      local k = self.kinds[w.kind]
      if k.surface then return true end
    end
    return false
  end
  if arr == "unspecified" and not (req.scene and req.scene ~= "") and not has_surface_kind() then
    local only_small = true
    for _, w in ipairs(want) do if not needs_surface(w.kind) then only_small = false end end
    if only_small then
      -- put the things on a surface that I add under them
      local first = want[1].kind
      local sk = SURFACE_FOR[first]
      if sk == "coffee table" and #insts > 6 then sk = "table" end
      local pick = self:pick_variants(sk, 1, {}, rng)[1]
      local surf = make_inst(self, sk, pick, {})
      table.insert(want, 1, { kind = sk, count = 1, adjectives = {} })
      table.insert(insts_by_want, 1, { surf })
      table.insert(insts, 1, surf)
      kinds_present[sk] = true
      order[#order + 1] = sk
      notes[#notes + 1] = "I put " .. (#insts > 2 and "them" or "it") .. " on a " .. sk .. " I added"
    end
  end

  -- -- an explicit arrangement on everything
  local explicit_arr = arr ~= "unspecified" and arr ~= "facing_me" and arr ~= "around" and arr ~= "beside" and arr ~= "in_front_of"
                       and arr ~= "behind" and arr ~= "on" and arr ~= "stacked_on"
  local room_mode = (not explicit_arr) and ((req.scene and req.scene ~= "") or #order >= 2)
  if arr == "against_wall" or arr == "corner" then explicit_arr = true; room_mode = false end
  if arr == "facing_me" and #order >= 2 then room_mode = true end

  if room_mode then
    local has = {}
    for _, w in ipairs(want) do
      local k = self.kinds[w.kind]
      if w.kind == "bed" or k.category == "bed" then has.bed = true end
      if w.kind == "desk" then has.desk = true end
      if k.category == "table" and w.kind ~= "desk" and w.kind ~= "side table" then has.center = true end
      if k.surface then has.surface = true end
    end
    local entries = {}
    for wi, w in ipairs(want) do
      local role = w.role and (ROLE_ALIAS[w.role] or w.role) or infer_role(self.kinds[w.kind], has)
      for _, inst in ipairs(insts_by_want[wi]) do entries[#entries + 1] = { inst = inst, role = role, spec = w } end
    end
    -- the surfaces that were asked for by name need a table-ish role even when asked for as a list
    local rb = self:room(entries, rng, ctx, { desc = nil })
    builders = { rb, self:room(entries, rng, ctx, { mirror = true }) }
    local radius = rb.radius
    local pls, frame, bb, score
    -- standing in a real room of about the right size: furnish that room, against its own walls
    local rr = real_room(ctx, 900)
    if rr and rr.w <= 720 and rr.d <= 720 and rr.w >= 0.75 * rb.size.w and rr.d >= 0.75 * rb.size.d then
      local rbs = {}
      for _, sh in ipairs({ { 0, 0 }, { 0, 70 }, { 0, -70 }, { 70, 0 }, { -70, 0 }, { 70, 70 }, { 70, -70 }, { -70, 70 }, { -70, -70 }, { 0, 140 }, { 0, -140 } }) do
        rbs[#rbs + 1] = self:room(entries, rng, ctx, { room = { w = rr.w, d = rr.d }, shift = sh })
        rbs[#rbs + 1] = self:room(entries, rng, ctx, { room = { w = rr.w, d = rr.d }, shift = sh, mirror = true })
      end
      local flipped = { x = rr.frame.x, y = rr.frame.y, theta = Layout.wrap(rr.frame.theta + 180) }
      local p2, f2, b2, s2 = fit(ctx, rbs, { rr.frame, flipped })
      if ctx.debug then ctx.debug(string.format("real room %.0fx%.0f, natural %.0fx%.0f -> fit=%s score=%s", rr.w, rr.d, rb.size.w, rb.size.d, tostring(p2 ~= nil), tostring(s2))) end
      if p2 and s2 == 0 then pls, frame, bb, score = p2, f2, b2, s2; notes[#notes + 1] = "I furnished the room you are standing in" end
    end
    if not pls then
      local frames3 = open_frames(ctx, radius, theta_square)
      pls, frame, bb, score = fit(ctx, builders, frames3)
    end
    if not pls then
      -- a snug version: sized to its contents, no minimum room
      local snug = self:room(entries, rng, ctx, { min_w = 0, min_d = 0 })
      pls, frame, bb, score = fit(ctx, { snug }, open_frames(ctx, snug.radius, theta_square))
    end
    if not pls then
      res.say = "I could not find room for that."
      return res
    end
    finish(self, ctx, pls, res, 1)
    local counts, corder = {}, {}
    for _, p in ipairs(res.placements) do
      counts[p.kind] = (counts[p.kind] or 0) + 1
      if not corder[p.kind] then corder[#corder + 1] = p.kind; corder[p.kind] = true end
    end
    local parts = {}
    for _, kname in ipairs(corder) do parts[#parts + 1] = count_word(counts[kname]) .. " " .. plural(kname, counts[kname]) end
    local what = (req.scene and req.scene ~= "") and (article(req.scene) .. ": ") or ""
    res.say = what .. say_list(parts) .. ", arranged as a room."
    if score and score > 0 then notes[#notes + 1] = "it was a tight squeeze" end
    if #notes > 0 then res.say = res.say .. " " .. table.concat(notes, "; ") .. "." end
    return res
  end

  -- -- a single arrangement, with fallbacks when it does not fit
  local name = explicit_arr and arr or self:default_arrangement(insts)
  if arr == "facing_me" or arr == "unspecified" then name = self:default_arrangement(insts) end
  local wall_pref = false
  if name == "against_wall" then name, wall_pref = "row", true end
  if name == "in_front_of" or name == "beside" or name == "around" or name == "behind" or name == "on" then name = self:default_arrangement(insts) end
  local FALLBACK = {
    row = { { "grid", 2 }, { "grid" } }, circle = { { "grid" } }, u_shape = { { "two_rows" }, { "row" }, { "grid" } },
    square = { { "two_rows" }, { "row" }, { "grid" } }, two_rows = { { "row" }, { "grid" } }, facing_each_other = { { "row" } },
    l_shape = { { "row" }, { "grid" } }, stacked = { { "row" }, { "grid" } }, scattered = { { "cluster" }, { "row" } },
    cluster = { { "row" }, { "grid" } }, grid = { { "row" } }, single = { { "row" } },
  }

  local function attempt(nm, wall_flag, cols)
    local bb
    if nm == "corner" then
      local rr = real_room(ctx, 900)
      if not rr then return nil end
      bb = { desc = "in the corner", radius = 0 }
      local its = to_items(insts)
      bb.build = function()
        local out = {}
        local order = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } }
        -- nearest to where the player is looking first
        local lx, ly = fwd(ctx.yaw)
        local tx, ty = ctx.pos.x + lx * 260, ctx.pos.y + ly * 260
        table.sort(order, function(a, b)
          local function dist(c)
            local pc = to_world(rr.frame, c[1] * rr.d / 2, c[2] * rr.w / 2, 0)
            return (pc.x - tx) ^ 2 + (pc.y - ty) ^ 2
          end
          return dist(a) < dist(b)
        end)
        for i, it in ipairs(its) do
          local c = order[(i - 1) % 4 + 1]
          local ring = floor((i - 1) / 4)
          local back_u = c[1] < 0
          local u = c[1] * (rr.d / 2 - it.d / 2 - 1)
          local v = c[2] * (rr.w / 2 - it.w / 2 - 1 - ring * (it.w + 2))
          out[#out + 1] = P(it.inst, to_world(rr.frame, u, v, back_u and 0 or 180), "corner")
        end
        return out
      end
      local pls, frame, _, score = fit(ctx, { bb }, { rr.frame })
      return bb, pls, frame, score
    end
    bb = self:arrangement(nm, insts, rng, ctx, { wall = wall_flag or nil, cols = cols })
    local pls, frame, score
    if (bb.wall or wall_flag) and (nm == "row" or nm == "single") then
      local depth = 0
      for _, i in ipairs(insts) do depth = max(depth, i.d) end
      local wf = wall_frames(ctx, depth, 900)
      if #wf > 0 then
        pls, frame, _, score = fit(ctx, { bb }, wf)
        if pls and score and score > 0 then pls = nil end
      elseif wall_flag then
        notes[#notes + 1] = "there is no wall in reach, so I stood them in the open"
      end
      if not pls then
        pls, frame, _, score = fit(ctx, { bb }, open_frames(ctx, bb.radius, (#insts == 1) and theta_for_player or theta_square))
      end
    else
      pls, frame, _, score = fit(ctx, { bb }, open_frames(ctx, bb.radius, (#insts == 1) and theta_for_player or theta_square))
    end
    return bb, pls, frame, score
  end

  local b, pls, frame, score
  if name == "corner" then
    b, pls, frame, score = attempt("corner")
    if not pls then
      notes[#notes + 1] = "there is no corner in reach, so I grouped them in the open"
      name = "cluster"
    end
  end
  if not pls then b, pls, frame, score = attempt(name, wall_pref) end
  if not pls then
    for _, fb in ipairs(FALLBACK[name] or { { "row" } }) do
      local b2, p2, f2, s2 = attempt(fb[1], false, fb[2])
      if p2 then
        b, pls, frame, score = b2, p2, f2, s2
        notes[#notes + 1] = "it would not fit as asked, so I laid it out " .. (b2.desc ~= "" and b2.desc or "in a row")
        break
      end
    end
  end
  if not pls then
    res.say = "I could not find clear floor for that."
    return res
  end
  group_desc = b.desc
  finish(self, ctx, pls, res, 1)
  local counts, corder = {}, {}
  for _, p in ipairs(res.placements) do
    counts[p.kind] = (counts[p.kind] or 0) + 1
    if not corder[p.kind] then corder[#corder + 1] = p.kind; corder[p.kind] = true end
  end
  local parts = {}
  for _, kname in ipairs(corder) do parts[#parts + 1] = count_word(counts[kname]) .. " " .. plural(kname, counts[kname]) end
  res.say = say_list(parts) .. ((group_desc and group_desc ~= "") and (" " .. group_desc) or "") .. "."
  if (b.wall or wall_pref) and frame and frame.wall then res.say = res.say:gsub("%.$", ", against the wall.") end
  if score and score > 0 then notes[#notes + 1] = "it was a tight squeeze" end
  if #notes > 0 then res.say = res.say .. " " .. table.concat(notes, "; ") .. "." end
  return res
end

-- ---------------------------------------------------------------------------------------------- management verbs
--- Pick the standing things a spec means.
local function select_entries(ctx, spec, from)
  -- what a player may change: ctx.mine (their own things) when the game supplies it, else everything standing
  local list = ctx.mine or ctx.registry or {}
  if spec.kind == "@last" then
    local last
    for _, e in ipairs(list) do
      if not last or (e.seq or 0) > (last.seq or 0) then last = e end
    end
    if not last then return {} end
    local out = {}
    for _, e in ipairs(list) do if e.group == last.group then out[#out + 1] = e end end
    return out
  end
  local pool = registry_matches(ctx, spec.kind, spec.adjectives, from, list)
  if spec.count == 0 or #pool == 0 then return pool end
  local n = spec.count
  if spec.definite and n == 1 then n = 1 end
  local out = {}
  for i = 1, min(n, #pool) do out[i] = pool[i] end
  return out
end

function Plan:remove(req, ctx)
  local res = { action = "remove", removals = {}, notes = {}, placements = {} }
  local from = aim_or_front(ctx)
  local taken = {}
  local parts = {}
  for _, it in ipairs(req.items or {}) do
    if tostring(it.kind):match("^UNKNOWN") then
      res.notes[#res.notes + 1] = "there is no " .. (it.phrase or it.kind:sub(9)) .. " here"
    else
      local sel = select_entries(ctx, it, from)
      local n = 0
      for _, e in ipairs(sel) do
        if not taken[e.id] then taken[e.id] = true; res.removals[#res.removals + 1] = e.id; n = n + 1 end
      end
      if n > 0 then parts[#parts + 1] = count_word(n) .. " " .. plural(sel[1].kind, n)
      elseif it.kind ~= "@last" then res.notes[#res.notes + 1] = "there is no " .. it.kind .. " to remove" end
    end
  end
  if #parts > 0 then res.say = "Removed " .. say_list(parts) .. "."
  else res.say = (res.notes[1] or "There was nothing to remove") .. "." end
  return res
end

--- Turn standing things: half a turn, to face the player or another thing, or by an angle in the words.
function Plan:rotate(req, ctx)
  local res = { action = "rotate", moves = {}, notes = {}, placements = {} }
  local from = aim_or_front(ctx)
  local text = req.normalised or ""
  local sel = {}
  for _, it in ipairs(req.items or {}) do
    for _, e in ipairs(select_entries(ctx, it, from)) do sel[#sel + 1] = e end
  end
  if #sel == 0 then res.say = "There was nothing to turn."; return res end
  local face_to
  if req.arrangement == "facing_me" or req.facing then
    local tg = req.facing or req.target
    if tg == "me" or tg == "" or tg == nil then face_to = { x = ctx.pos.x, y = ctx.pos.y }
    else
      local m = registry_matches(ctx, tg, {}, sel[1])
      if m[1] then face_to = { x = m[1].x, y = m[1].y } end
    end
  end
  local angle = tonumber(text:match("(%d+)%s*degree")) or tonumber(text:match("(%d+)%s*deg"))
  if text:find("quarter") then angle = angle or 90 end
  if text:find("half") or req.half_turn or text:find("around") or text:find("flip") or text:find("backward") then angle = angle or 180 end
  local sign = (text:find("clockwise") and not text:find("counter") and not text:find("anti")) and -1 or 1
  if text:find(" right") and not text:find("left") then sign = -1 end
  for _, e in ipairs(sel) do
    local theta
    if face_to then theta = Layout.angle_of(face_to.x - e.x, face_to.y - e.y)
    else theta = Layout.wrap(e.theta + sign * (angle or 90)) end
    res.moves[#res.moves + 1] = { id = e.id, x = e.x, y = e.y, z = e.z, theta = theta, rotate_only = true }
  end
  res.say = "Turned " .. count_word(#sel) .. " " .. plural(sel[1].kind, #sel) .. "."
  return res
end

function Plan:move(req, ctx)
  local res = { action = "move", moves = {}, notes = {}, placements = {} }
  local from = aim_or_front(ctx)
  local sel, kinds = {}, {}
  for _, it in ipairs(req.items or {}) do
    if tostring(it.kind):match("^UNKNOWN") then
      res.notes[#res.notes + 1] = "there is no " .. (it.phrase or it.kind:sub(9)) .. " here"
    else
      for _, e in ipairs(select_entries(ctx, it, from)) do sel[#sel + 1] = e end
    end
  end
  if #sel == 0 then res.say = (res.notes[1] or "There was nothing to move") .. "."; return res end
  local ignore = {}
  for _, e in ipairs(sel) do ignore[e.id] = true end
  local text = req.normalised or ""
  local pos, yaw = ctx.pos, ctx.yaw
  local fx, fy = fwd(yaw)
  local rx, ry = fy, -fx

  local insts = {}
  for i, e in ipairs(sel) do
    local inst = make_inst(self, e.kind, e.vi, {})
    inst.entry = e
    insts[i] = inst
  end
  local target = req.target or ""
  local arr = req.arrangement or "unspecified"
  local frames, builders
  local rng = Layout.rng(ctx.seed or Layout.hash(req.raw or ""))
  local function commit(pls)
    for i, p in ipairs(pls) do
      local e = p.inst.entry
      res.moves[#res.moves + 1] = { id = e.id, x = p.x, y = p.y, z = (p.fz or pos.z) + (p.zoff or 0), theta = Layout.wrap(p.theta), zoff = p.zoff, fz = p.fz }
    end
    res.say = "Moved " .. count_word(#sel) .. " " .. plural(sel[1].kind, #sel) .. "."
    return res
  end

  if (arr == "beside" or arr == "in_front_of" or arr == "behind" or arr == "on" or arr == "around" or arr == "facing_me")
     and target ~= "" and target ~= "me" and target ~= "here" and self.kinds[target] then
    local m = registry_matches(ctx, target, {}, from)
    local ref
    for _, e in ipairs(m) do if not ignore[e.id] then ref = e; break end end
    if ref then
      local r = ref_of(ref)
      local k = self.kinds[ref.kind]
      r.topf = ref.vi and top_frame(k.variants[ref.vi]) or nil
      r.kind = ref.kind
      local rel = (arr == "facing_me") and "in_front_of" or arr
      local b
      if rel == "around" then b = around_builder(r, insts, {})
      elseif rel == "on" and r.topf then b = on_builder(r, insts, {})
      else b = offset_builder(r, insts, (rel == "on") and "beside" or rel, { distance = (r.kind == "tv") and max(70, 1.6 * r.w) or nil }) end
      local alt = (rel == "beside" or rel == "on") and offset_builder(r, insts, "beside", { flip = true }) or nil
      local bs = { b }
      if alt then bs[2] = alt end
      local pls, frame = fit(ctx, bs, { { x = r.x, y = r.y, theta = r.theta } }, ignore)
      if pls then return commit(pls) end
      res.say = "There was no clear spot for that."
      return res
    end
    res.notes[#res.notes + 1] = "there is no " .. target .. " to move them to"
  end

  -- loose directions in the words
  local dx, dy
  if text:find("closer") or text:find("toward me") or text:find("towards me") or text:find("over here") then
    local e = sel[1]
    local ux, uy = pos.x - e.x, pos.y - e.y
    local len = sqrt(ux * ux + uy * uy)
    if len > 1 then dx, dy = ux / len * min(80, max(0, len - 70)), uy / len * min(80, max(0, len - 70)) end
  elseif text:find("away") or text:find("farther") or text:find("further") then
    local e = sel[1]
    local ux, uy = e.x - pos.x, e.y - pos.y
    local len = max(1, sqrt(ux * ux + uy * uy))
    dx, dy = ux / len * 80, uy / len * 80
  elseif text:find(" left") then dx, dy = -rx * 60, -ry * 60
  elseif text:find(" right") then dx, dy = rx * 60, ry * 60
  elseif text:find("forward") then dx, dy = fx * 60, fy * 60
  elseif text:find(" back") or text:find("backward") then dx, dy = -fx * 60, -fy * 60
  end
  if dx then
    local pls = {}
    for i, e in ipairs(sel) do pls[i] = P(insts[i], { x = e.x + dx, y = e.y + dy, theta = e.theta, z = 0 }, "moved") end
    local viol = judge(ctx, pls, ignore)
    if viol == 0 then return commit(pls) end
    -- blocked: try half the shift
    for i, e in ipairs(sel) do pls[i] = P(insts[i], { x = e.x + dx / 2, y = e.y + dy / 2, theta = e.theta, z = 0 }, "moved") end
    viol = judge(ctx, pls, ignore)
    if viol == 0 then return commit(pls) end
    res.say = "Something was in the way."
    return res
  end

  -- otherwise to where the player is looking, laid out as they were (or in a row)
  local name = (arr ~= "unspecified" and arr ~= "facing_me" and arr ~= "around" and arr ~= "beside" and arr ~= "in_front_of" and arr ~= "behind" and arr ~= "on")
               and arr or self:default_arrangement(insts)
  if name == "against_wall" then name = "row" end
  if name == "corner" then name = "cluster" end
  local b = self:arrangement(name, insts, rng, ctx, {})
  local at = ctx.aim or { x = pos.x + fx * (b.radius + 90), y = pos.y + fy * (b.radius + 90) }
  local frames2 = {}
  frames2[1] = { x = at.x, y = at.y, theta = Layout.angle_of(pos.x - at.x, pos.y - at.y) }
  for _, f in ipairs(open_frames(ctx, b.radius, function(cx, cy) return Layout.angle_of(pos.x - cx, pos.y - cy) end)) do frames2[#frames2 + 1] = f end
  local pls = fit(ctx, { b }, frames2, ignore)
  if pls then return commit(pls) end
  res.say = "I could not find clear floor for that."
  return res
end

function Plan:rearrange(req, ctx)
  local res = { action = "rearrange", moves = {}, notes = {}, placements = {} }
  local from = aim_or_front(ctx)
  local sel = {}
  for _, it in ipairs(req.items or {}) do
    for _, e in ipairs(select_entries(ctx, it, from)) do sel[#sel + 1] = e end
  end
  if #sel == 0 then
    res.say = "There was nothing to rearrange."
    return res
  end
  local ignore = {}
  local insts = {}
  for i, e in ipairs(sel) do
    ignore[e.id] = true
    local inst = make_inst(self, e.kind, e.vi, {})
    inst.entry = e
    insts[i] = inst
  end
  local rng = Layout.rng(ctx.seed or Layout.hash(req.raw or ""))
  local arr = req.arrangement or "unspecified"
  local name = (arr ~= "unspecified" and arr ~= "facing_me" and arr ~= "around" and arr ~= "beside" and arr ~= "in_front_of" and arr ~= "behind" and arr ~= "on")
               and arr or self:default_arrangement(insts)
  if name == "against_wall" then name = "row" end
  if name == "corner" then name = "cluster" end
  -- the centroid of what is there, facing the player
  local cx, cy = 0, 0
  for _, e in ipairs(sel) do cx, cy = cx + e.x, cy + e.y end
  cx, cy = cx / #sel, cy / #sel
  local b = self:arrangement(name, insts, rng, ctx, {})
  local frames = { { x = cx, y = cy, theta = Layout.angle_of(ctx.pos.x - cx, ctx.pos.y - cy) } }
  for _, f in ipairs(open_frames(ctx, b.radius, function(px, py) return Layout.angle_of(ctx.pos.x - px, ctx.pos.y - py) end)) do frames[#frames + 1] = f end
  local pls = fit(ctx, { b }, frames, ignore)
  if not pls then res.say = "I could not find clear floor for that."; return res end
  for _, p in ipairs(pls) do
    local e = p.inst.entry
    res.moves[#res.moves + 1] = { id = e.id, x = p.x, y = p.y, z = (p.fz or ctx.pos.z) + (p.zoff or 0), theta = Layout.wrap(p.theta), zoff = p.zoff, fz = p.fz }
  end
  res.say = "Rearranged " .. count_word(#sel) .. " " .. plural(sel[1].kind, #sel) .. (b.desc and b.desc ~= "" and (" " .. b.desc) or "") .. "."
  return res
end

--- Entry point.
function Plan:run(req, ctx)
  ctx.registry = ctx.registry or {}
  local a = req.action
  local res
  if a == "add" then res = self:add(req, ctx)
  elseif a == "remove" then res = self:remove(req, ctx)
  elseif a == "rotate" then res = self:rotate(req, ctx)
  elseif a == "move" then res = self:move(req, ctx)
  elseif a == "rearrange" then res = self:rearrange(req, ctx)
  elseif a == "clear" then res = { action = "clear", placements = {}, notes = {}, say = "Cleared everything." }
  elseif a == "undo" then res = { action = "undo", placements = {}, notes = {}, say = "Undone." }
  else res = { action = "none", placements = {}, notes = {}, say = "" } end
  if res.say and res.say ~= "" then res.say = res.say:sub(1, 1):upper() .. res.say:sub(2) end
  return res
end

-- exposed for the offline tests
Plan._judge, Plan._fit, Plan._real_room, Plan._top_frame = judge, fit, real_room, top_frame

return Plan
