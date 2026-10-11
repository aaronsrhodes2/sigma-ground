-- plan.lua -- turns a parsed request into poses. Pure Lua (no GMod globals): the world is reached only through ctx.env,
-- so the whole planner is unit-tested offline against synthetic rooms, then driven by the engine glue (holodeck.lua).
--
--   local plan = Plan.new(catalog)
--   local res  = plan:run(request, ctx)
--
-- ctx = { pos = {x,y,z}, yaw = degrees, aim = {x,y,z} or nil, seed = n,
--         env = { floor_z(x,y) -> z|nil,  clear(rect, z, h) -> bool,  wall_probe(x,y,dx,dy,maxd) -> {dist,nx,ny}|nil,
--                 ceiling_z(x,y,z) -> z|nil   (optional: how high the room is above z) },
--         registry = { {id, kind, vi, x, y, z, theta, w, d, h, color_name, group, seq, ...}, ... }   -- what is already standing
--         mine     = the same rows for the requester's own things (nil: may change anything),
--         last     = { id, ... }   the requester's most recent group (what "it" and "them" mean),
--         players  = { {x,y,z}, ... }   every OTHER player (nobody gets a prop on top of them),
--         now = function() -> seconds   (monotonic; default os.clock), budget = seconds one request may spend planning (0.25),
--         max_total, max_per_kind = limits on one request (150 and 60),
--         exhaustive / explain = test hooks (judge every candidate to the end / be told why a placement is refused) }
-- res = { action, placements = { {kind, vi, model, x, y, z, theta, w, d, h, origin = {x,y,z}, yaw, ...} },
--         moves = { {id, x, y, z, theta} }, removals = { id, ... }, notes = { ... }, say = "one sentence",
--         timed_out = true when the search ran out of time, clamped = true when a limit cut the request down,
--         unknown_words = { "dragon" }  (sanitised; the only place a player's own word appears: never in say or notes),
--         fixed = true when `say` is a constant sentence }
--
-- Request fields the planner reads besides the parser's core ones, when the parser sends them:
--   req.facing_away  'me' | 'wall' | '' | <a kind>: the group turns its back on that (a kind that stands near: each one looks along
--                    the line from it; one that does not stand: the group turns from the player and the reply says so)
--   req.side         'other' | 'opposite' | 'left' | 'right': for a move beside a target, which side of it ('other' is the side
--                    facing away from the player and not the one the thing stands at; left/right are the player's own)
--   req.every_mode   'per' | 'total': with req.every = 'corner', whether the count is for each corner or in all (absent: a count that
--                    is a multiple of four is the total)
--   req.extra_scenes the scenes named beside req.scene (one room is built at a time; the reply names the one left for next)
--
-- It never asks a question: whatever is unclear resolves to the most sensible reading, and the reading is reported.
local Layout = include("mentat/holodeck/layout.lua")

local Plan = {}
Plan.__index = Plan

local rad, deg = math.rad, math.deg
local cos, sin, sqrt, abs, floor, ceil, max, min = math.cos, math.sin, math.sqrt, math.abs, math.floor, math.ceil, math.max, math.min
local rect, to_world, fwd = Layout.rect, Layout.to_world, Layout.fwd

-- what one request may ask for when the engine says nothing (ctx.max_per_kind, ctx.max_total), and how long it may think (ctx.budget)
local MAX_PER_KIND, MAX_TOTAL, DEFAULT_BUDGET = 60, 150, 0.25
-- the footprint kept clear round every player (HU), and by how much it is padded
local KEEP_OUT, KEEP_OUT_PAD = 40, 4
-- two new things further into each other than this (HU) are inside each other, not touching; the seats at one table may touch
-- at the corners (a chair is not its whole bounding box) up to TUCK_PEN
local DEEP, TUCK_PEN = 3, 14
-- a pile is no taller than this (HU): two of the tallest common crate (65.65 HU) with a margin; a wall within WALL_NEAR is where
-- a wall-loving thing goes unasked, WALL_FAR when asked
local MAX_PILE_H, WALL_NEAR, WALL_FAR = 134, 420, 900
-- what a chair may slide its front edge under: open tops with knee room. A dresser or a side table is a drawer unit, a coffee table
-- is too low: a seat stands clear of those (tucked in it would stand through the drawers)
local TUCKS_UNDER = { table = true, desk = true, ["meeting table"] = true, ["patio table"] = true }
-- how far a thing may overhang the edge of what it stands on (HU)
local ON_OVERHANG = 2

-- gaps between neighbours of a kind (HU; one HU is 1.905 cm)
local GAP = { seating = 6, table = 12, bed = 24, storage = 1, container = 3, lighting = 40, appliance = 1, bath = 6, electronics = 14, decor = 24 }

-- sentences that never contain a word the player chose (what Skippy says goes to everybody)
local T = {
  library = "I don't have that in my library",
  nothing_recent = "There is nothing recent to go by",
  too_much = "That is more than I can fit here",
  out_of_time = "I ran out of time, so this is the best I found",
  no_floor = "I could not find clear floor for that",
  no_room = "I could not find room for that",
  nothing_to_add = "There was nothing in that to add",
}

local COLORS = {
  red = { 200, 45, 45 }, blue = { 55, 95, 215 }, green = { 65, 165, 75 }, yellow = { 230, 205, 60 }, orange = { 235, 140, 45 },
  pink = { 235, 135, 180 }, purple = { 140, 75, 190 }, violet = { 140, 75, 190 }, brown = { 125, 85, 45 }, black = { 35, 35, 35 },
  white = { 255, 255, 255 }, grey = { 150, 150, 150 }, gray = { 150, 150, 150 }, cyan = { 60, 190, 190 }, teal = { 50, 160, 160 },
  gold = { 212, 175, 55 }, silver = { 192, 192, 192 }, beige = { 215, 190, 150 }, tan = { 210, 180, 140 }, navy = { 25, 35, 105 },
  maroon = { 115, 25, 35 }, turquoise = { 60, 200, 190 }, lime = { 150, 220, 60 }, crimson = { 190, 25, 55 },
}
-- colour names that mean one colour, so "grey" matches "gray"
local COLOR_CANON = { gray = "grey", violet = "purple" }
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
  if last:match("[lr]f$") then return base .. last:sub(1, -2) .. "ves" end              -- shelf, bookshelf
  if last:match("[^f]fe$") then return base .. last:sub(1, -3) .. "ves" end
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

local function same_color(a, b) return (COLOR_CANON[a] or a) == (COLOR_CANON[b] or b) end

--- A player's word made safe to keep: lower case letters, digits and spaces, three words and 24 characters at most.
local function clean_word(s)
  s = tostring(s or ""):lower()
  s = s:gsub("[^a-z0-9 ]", " ")
  s = s:gsub("%s+", " ")
  local out, n = {}, 0
  for w in s:gmatch("%S+") do
    n = n + 1
    if n > 3 then break end
    out[n] = w
  end
  s = table.concat(out, " ")
  if #s > 24 then s = s:sub(1, 24):gsub("%s+$", "") end
  return s
end

--- Remember a word the library lacks, for the requester's eyes only (res.unknown_words); it never enters a sentence.
local function note_unknown(res, word)
  local w = clean_word(word)
  if w == "" then return end
  res.unknown_words = res.unknown_words or {}
  if #res.unknown_words >= 4 then return end
  for _, x in ipairs(res.unknown_words) do if x == w then return end end
  res.unknown_words[#res.unknown_words + 1] = w
end

local function add_note(res, text)
  for _, n in ipairs(res.notes) do if n == text then return end end
  res.notes[#res.notes + 1] = text
end

--- A note that says what the layout DID ("there was no X, so I added one"): it is true only if something was placed, so a refusal
--- drops it (request-level notes, such as an unknown word or a scaled-down count, stay true either way).
function Plan:effect_note(res, text)
  add_note(res, text)
  self._effects = self._effects or setmetatable({}, { __mode = "k" })             -- (weak: a finished reply is not kept alive)
  self._effects[res] = self._effects[res] or {}
  self._effects[res][text] = true
end

local function say_list(parts)
  if #parts == 0 then return "" end
  if #parts == 1 then return parts[1] end
  return table.concat(parts, ", ", 1, #parts - 1) .. " and " .. parts[#parts]
end

--- "three couches and a plant" from a list of things (anything with a .kind), in the order first seen. More than `cap` kinds
--- read as "N things", which is what a person says of a whole room.
local function describe(list, cap)
  local counts, order = {}, {}
  for _, e in ipairs(list) do
    if not counts[e.kind] then counts[e.kind] = 0; order[#order + 1] = e.kind end
    counts[e.kind] = counts[e.kind] + 1
  end
  if #order > (cap or 4) then return count_word(#list) .. " things" end
  local parts = {}
  for _, k in ipairs(order) do parts[#parts + 1] = count_word(counts[k]) .. " " .. plural(k, counts[k]) end
  return say_list(parts)
end

--- Notes joined onto a sentence, each one starting a sentence of its own.
local function with_notes(say, notes)
  if #notes == 0 then return say end
  local out = {}
  for i, n in ipairs(notes) do out[i] = n:sub(1, 1):upper() .. n:sub(2) end
  return say .. " " .. table.concat(out, ". ") .. "."
end

--- A sentence that does not change with the request: the engine can skip the persona for it.
local function fixed(res, text)
  res.say, res.fixed = text, true
  return res
end

-- ---------------------------------------------------------------------------------------------- time
-- One deadline per request (Plan:run sets it); the places the CPU goes ask whether it has passed.
local function expired(ctx, every)
  if ctx.timed_out then return true end
  local dl = ctx.deadline
  if not dl then return false end
  if every and every > 1 then
    ctx._tick = (ctx._tick or 0) + 1
    if ctx._tick % every ~= 0 then return false end
  end
  if ctx.clock() > dl then ctx.timed_out, ctx.cut_short = true, true; return true end
  return false
end

--- The function Layout's generators call to learn that the time is up.
local function stopper(ctx)
  ctx._stop = ctx._stop or function() return expired(ctx, 4) end
  return ctx._stop
end

--- What is left of the request's time for one stage: `share` of it (a stage may not eat the others' allowance).
local function slice_deadline(ctx, share)
  if not ctx.deadline then return nil end
  local now = ctx.clock()
  return min(ctx.deadline, now + max(0, ctx.deadline - now) * (share or 1))
end

-- ---------------------------------------------------------------------------------------------- the entities
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
  local t = v and v.top
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

--- Which variants of a kind to use for `n` of it, with the adjectives that narrow or steer the choice. `uniform`: one variant
--- for all of them (a pile of mixed barrels would float).
function Plan:pick_variants(kind_id, n, adjs, rng, uniform)
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
  if consistent or uniform or #pool == 1 then
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

--- Is this standing thing still in the library (the kind and the variant it was made from)?
local function in_library(self, e)
  local k = self.kinds[e.kind]
  return k ~= nil and e.vi ~= nil and k.variants[e.vi] ~= nil
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

--- What already stands, in a grid of cells, so a placement meets only its neighbours (built once per request).
local CELL = 128
local function reg_index(ctx)
  local reg = ctx.registry or {}
  local idx = ctx._ridx
  if idx and idx.src == reg and idx.n == #reg then return idx end
  idx = { src = reg, n = #reg, cells = {}, stamp = 0 }
  for _, e in ipairs(reg) do
    local r = rect(e.x, e.y, e.w, e.d, e.theta)
    r.id = e.id
    local rr = r.rad
    for cx = floor((e.x - rr) / CELL), floor((e.x + rr) / CELL) do
      for cy = floor((e.y - rr) / CELL), floor((e.y + rr) / CELL) do
        local key = cx * 100003 + cy
        local c = idx.cells[key]
        if not c then c = {}; idx.cells[key] = c end
        c[#c + 1] = r
      end
    end
  end
  ctx._ridx = idx
  return idx
end

--- Everything standing that a footprint overlaps: calls fn(entry_rect) for each.
local function each_standing(ctx, r, pad, fn)
  local idx = reg_index(ctx)
  if idx.n == 0 then return end
  idx.stamp = idx.stamp + 1
  local stamp = idx.stamp
  local rr = r.rad + pad
  for cx = floor((r.x - rr) / CELL), floor((r.x + rr) / CELL) do
    for cy = floor((r.y - rr) / CELL), floor((r.y + rr) / CELL) do
      local c = idx.cells[cx * 100003 + cy]
      if c then
        for _, er in ipairs(c) do
          if er.seen ~= stamp then
            er.seen = stamp
            fn(er)
          end
        end
      end
    end
  end
end

--- The people a prop must never land on: the requester and everybody else (the same keep-out for each).
local function keep_outs(ctx)
  if ctx.keep_out == false then return {} end
  local cached = ctx._people                              -- (the same people for the whole request)
  if cached then return cached end
  local out = { rect(ctx.pos.x, ctx.pos.y, KEEP_OUT, KEEP_OUT, 0) }
  for _, p in ipairs(ctx.players or {}) do
    if abs((p.z or ctx.pos.z or 0) - (ctx.pos.z or 0)) < 200 then out[#out + 1] = rect(p.x, p.y, KEEP_OUT, KEEP_OUT, 0) end
  end
  ctx._people = out
  return out
end

--- Violations of a placement list: off the floor, into the world, into each other, into what already stands, onto a person.
--- Returns viol (how many), pen (how deep, summed), hard (how many can never be accepted: floor, walls, people, what stands,
--- things far inside each other). o.quick stops at the first hard one; o.limit gives up once viol*1000 reaches it (the caller
--- already has something at least as good); o.rigid skips the pairs among the placements (a group moved as it stood).
local function judge(ctx, pls, ignore, o)
  o = o or {}
  local env, viol, pen, hard = ctx.env, 0, 0, 0
  local n = #pls
  local rects = {}
  for i, p in ipairs(pls) do rects[i] = rect(p.x, p.y, p.w, p.d, p.theta) end
  local limit = o.limit
  local explain = ctx.explain                      -- a test hook: told why a placement is refused
  local function done() return (o.quick and hard > 0) or (limit and viol * 1000 >= limit) end

  -- people first: the cheapest question
  local people = keep_outs(ctx)
  for i, p in ipairs(pls) do
    if (p.zoff or 0) <= 0.01 then
      for _, me in ipairs(people) do
        if Layout.overlap(me, rects[i], KEEP_OUT_PAD) then
          viol, hard = viol + 1, hard + 1
          if explain then explain("person", p.kind, p.role) end
          break
        end
      end
      if done() then return viol, pen, hard end
    end
  end
  -- what already stands
  for i, p in ipairs(pls) do
    if (p.zoff or 0) <= 0.01 then
      local ri = rects[i]
      each_standing(ctx, ri, 0.4, function(er)
        if not (ignore and ignore[er.id]) and not (p.sits_on == er.id) and Layout.overlap(er, ri, 0.4) then
          viol, hard = viol + 1, hard + 1
          pen = pen + Layout.penetration(er, ri)
          if explain then explain("standing", p.kind, p.role, er.id) end
        end
      end)
      if done() then return viol, pen, hard end
    end
  end
  -- what the moved things carry (a load on a crate that goes to a corner) must stand clear of the walls too
  if ctx.carry_check and ctx.carry_check(pls) then
    viol, hard = viol + 1, hard + 1
    ctx._carry_blocked = true
    if explain then explain("carried load", pls[1] and pls[1].kind, "carry") end
    if done() then return viol, pen, hard end
  end
  -- the world itself (these are the engine's traces: ask last, and not at all once the time is up)
  -- (a candidate that fails does so at its outermost piece: those are asked first, so a refusal costs one trace, not one per piece)
  local wi = {}
  for k = 1, n do wi[k] = k end
  if n > 3 and o.quick then
    local cx, cy, far = 0, 0, {}
    for _, p in ipairs(pls) do cx, cy = cx + p.x, cy + p.y end
    cx, cy = cx / n, cy / n
    for k, p in ipairs(pls) do far[k] = (p.x - cx) ^ 2 + (p.y - cy) ^ 2 end
    table.sort(wi, function(a, b) if far[a] ~= far[b] then return far[a] > far[b] end return a < b end)
  end
  for _, i in ipairs(wi) do
    local p = pls[i]
    if expired(ctx, 16) then return viol + 1, pen, hard + 1 end
    if (p.zoff or 0) <= 0.01 then
      local fz = floor_ok(env, rects[i])
      if not fz then
        viol, hard = viol + 1, hard + 1
        if explain then explain("no floor", p.kind, p.role, p.x, p.y) end
      else
        p.fz = fz
        if not env.clear(rects[i], fz, p.h) then
          viol, hard = viol + 1, hard + 1
          if explain then explain("blocked", p.kind, p.role, p.x, p.y) end
        end
      end
    else
      p.fz = env.floor_z(p.x, p.y) or ctx.pos.z
      -- an upper layer is held to the world too: there must be floor under all of it (a wall or an edge is not a place to stack)
      if p.pile and not floor_ok(env, rects[i]) then
        viol, hard = viol + 1, hard + 1
        if explain then explain("no floor", p.kind, p.role, p.x, p.y) end
      end
      -- a pile that would reach the ceiling is not a pile (the engine tells where the ceiling is, when it can)
      if p.pile and env.ceiling_z then
        local ceil_z = env.ceiling_z(p.x, p.y, p.fz)
        if ceil_z and p.fz + p.zoff + p.h > ceil_z - 2 then viol, hard = viol + 1, hard + 1 end
      end
    end
    if done() then return viol, pen, hard end
  end
  if o.rigid then return viol, pen, hard end
  -- the placements among themselves: a sweep along x, so only the near pairs are looked at
  local order = {}
  for i = 1, n do order[i] = i end
  table.sort(order, function(a, b) return rects[a].x - rects[a].rad < rects[b].x - rects[b].rad end)
  for oi = 1, n do
    local i = order[oi]
    local ri = rects[i]
    local reach = ri.x + ri.rad
    for oj = oi + 1, n do
      local j = order[oj]
      local rj = rects[j]
      if rj.x - rj.rad > reach then break end
      local lo, hi = i, j
      if lo > hi then lo, hi = hi, lo end
      local a, b = pls[lo], pls[hi]
      local skip = (a.zoff or 0) > 0.01 or (b.zoff or 0) > 0.01
      if not skip and Layout.overlap(rects[lo], rects[hi], 0.4) then
        local tucked = a.tuck == b or b.tuck == a
        local pn = Layout.penetration(rects[lo], rects[hi])
        if tucked then
          -- allowed only as far as the chair's front edge goes under the table
          local allow = (a.tuck == b and a.tuck_depth or b.tuck_depth or 9) + 1.5
          if pn > allow then viol = viol + 1; pen = pen + pn end
        else
          viol = viol + 1
          pen = pen + pn
          local siblings = a.tuck_group and b.tuck_group and a.tuck ~= nil and a.tuck == b.tuck
          if pn > (siblings and TUCK_PEN or DEEP) then
            hard = hard + 1
            if explain then explain("inside each other", a.kind .. "/" .. (a.role or "?"), b.kind .. "/" .. (b.role or "?"), pn, floor(a.x) .. "," .. floor(a.y) .. "," .. floor(a.w) .. "x" .. floor(a.d) .. "@" .. floor(a.theta) .. " vs " .. floor(b.x) .. "," .. floor(b.y) .. "," .. floor(b.w) .. "x" .. floor(b.d) .. "@" .. floor(b.theta)) end
          end
        end
      end
    end
    if done() then return viol, pen, hard end
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

--- Frames to try for a group of radius R: along the player's view, widening outward and sideways. The ones nearer than R+62
--- come last: they are for a player standing in the middle of a small room, and cost nothing when the others have worked.
local function open_frames(ctx, R, theta_for)
  local pos, yaw = ctx.pos, ctx.yaw
  local frames = {}
  local function add(cx, cy)
    frames[#frames + 1] = { x = cx, y = cy, theta = theta_for(cx, cy), open = true }
  end
  if ctx.aim and ctx.aim_first then add(ctx.aim.x, ctx.aim.y) end
  -- straight ahead first, then sweeping round the player, so a player in the middle of a small room still finds a spot
  local bearings = { 0, 35, -35, 70, -70, 105, -105, 140, -140, 180 }
  for _, bearing in ipairs(bearings) do
    local fx, fy = fwd(yaw + bearing)
    local rx, ry = fy, -fx
    for k = 0, 12 do
      local dist = R + 62 + k * 28
      for _, lat in ipairs({ 0, 1, -1, 2, -2 }) do
        add(pos.x + fx * dist + rx * lat * 55, pos.y + fy * dist + ry * lat * 55)
      end
    end
  end
  for _, bearing in ipairs(bearings) do
    local fx, fy = fwd(yaw + bearing)
    local rx, ry = fy, -fx
    for dist = R + 34, 24, -28 do
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
  for wi, w in ipairs(nearby_walls(ctx, maxd or WALL_FAR)) do
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

local NO_BUILDERS = {}
--- Is there floor under (x, y)? Asked once per place for the whole request (the engine's trace is the dearest thing in a search).
local function floor_at(ctx, x, y)
  local cache = ctx._floorc
  if not cache then cache = {}; ctx._floorc = cache end
  local key = floor(x * 4 + 0.5) * 4194304 + floor(y * 4 + 0.5)
  local v = cache[key]
  if v == nil then v = ctx.env.floor_z(x, y) or false; cache[key] = v end
  return v
end

--- Build with each candidate frame; the first that violates nothing wins, otherwise the least-bad. `share` is the part of the
--- request's remaining time this search may spend (the stages after it keep the rest); the first build always happens.
--- A frame of the open sweep (frame.open) is the CENTRE of the group: when that stands on no floor it lies outside the room (the
--- sweep reaches well past the walls of a small room) and nothing built round it can stand, so it is passed over without building.
--- Returns placements, frame, builder, score (nil when nothing was acceptable).
local function fit(ctx, builders, frames, ignore, share)
  local best, best_score
  local local_dl = slice_deadline(ctx, share)
  local tried = 0
  local function out_of_time()
    if tried == 0 then return false end
    if expired(ctx) then return true end
    if local_dl and ctx.clock() > local_dl then ctx.cut_short = true; return true end
    return false
  end
  for _, frame in ipairs(frames) do
    if out_of_time() then break end
    local outside = frame.open and tried > 0 and not ctx.exhaustive and not floor_at(ctx, frame.x, frame.y)
    for _, b in ipairs(outside and NO_BUILDERS or builders) do
      if out_of_time() then break end
      tried = tried + 1
      local pls = b.build(frame)
      -- (ctx.exhaustive judges every candidate to the end: the tests use it to show the early exits change no answer)
      local viol, pen, hard = judge(ctx, pls, ignore, (not ctx.exhaustive) and { quick = true, limit = best_score } or nil)
      if viol == 0 then return pls, frame, b, 0 end
      -- a thing in a wall or on no floor is never acceptable; things merely brushing each other are the last resort
      if hard == 0 then
        local score = viol * 1000 + pen
        -- a tie (to within noise) keeps the earlier, nearer frame
        if not best_score or score < best_score - 0.5 then best, best_score = { pls, frame, b }, score end
      end
    end
  end
  if not best then return nil end
  return best[1], best[2], best[3], best_score
end

-- ---------------------------------------------------------------------------------------------- arrangement builders
local function to_items(insts)
  local items = {}
  for i, inst in ipairs(insts) do items[i] = { w = inst.w, d = inst.d, h = inst.h, inst = inst, sym = inst.v and inst.v.symmetric or nil } end
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
  local stop = stopper(ctx)
  local function from_poses(poses, role, extra)
    local out = {}
    for i, it in ipairs(items) do out[#out + 1] = P(it.inst, poses[i] or poses[it], role, extra) end
    return out
  end

  if name == "single" or name == "row" then
    local wall = opts.wall or wants_wall(insts)
    desc = n == 1 and "" or "in a row"
    build = function(frame) return from_poses(Layout.row(frame, items, { gap = gap, align_back = wall })) end
  elseif name == "circle" then
    desc = "in a circle, facing the middle"
    -- the radius that clears every footprint does not depend on where the circle stands: found once, then reused (a search builds
    -- the same circle round hundreds of frames, and each radius search is quadratic)
    local known_r
    build = function(frame)
      local poses, r = Layout.circle(frame.x, frame.y, items, { gap = gap, start = frame.theta + 180, min_r = max(opts.min_r or 14, known_r or 0), stop = stop })
      if not known_r and not ctx.timed_out then known_r = r end
      return from_poses(poses)
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
    -- the words describe the result: a pile the height cap lays flat (tall crates) is a row, not a pile
    local flat = true
    for _, pz in pairs(Layout.stack({ x = 0, y = 0, theta = 0 }, items, { gap = 1, max_h = MAX_PILE_H })) do
      if (pz.z or 0) > 0.01 then flat = false end
    end
    desc = flat and (n == 1 and "" or "in a row") or (n > 2 and "stacked in a pile" or "stacked")
    build = function(frame) return from_poses((Layout.stack(frame, items, { gap = 1, max_h = MAX_PILE_H })), nil, { pile = true }) end
  elseif name == "scattered" then
    desc = "scattered about"
    local area = 0
    for _, it in ipairs(items) do area = area + it.w * it.d end
    local radius = max(40, sqrt(area) * 1.15 + 12)
    local seed = rng.next() * 1e6
    build = function(frame)
      local r2 = Layout.rng(seed)
      return from_poses((Layout.scatter(frame, items, r2, { gap = 12, radius = radius, spin = insts[1].k.front_none, stop = stop })))
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
    -- a pile is for crates and boxes only: unlike kinds are not stacked under each other (beds on crates, through walls)
    local piles = true
    for _, i in ipairs(insts) do if i.kind ~= "crate" and i.kind ~= "box" then piles = false end end
    if (k.kind == "crate" or k.kind == "box") and piles then return n <= 2 and "row" or "stacked" end
    if k.kind == "crate" or k.kind == "box" then return n <= 6 and "row" or "grid" end
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
  local tuck = (seat and chairlike and topf and topf.z >= 28 and TUCKS_UNDER[ref.kind or ""]) and min(9, 0.35 * insts[1].d) or 0
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

--- Items in a ring round a reference, facing it: for when the sides of the reference are too short for them all. opts.arc: only
--- that many degrees of the ring, centred on the reference's front (a screen against a wall has no back to stand behind).
local function ring_builder(ref, insts, opts)
  opts = opts or {}
  local items = to_items(insts)
  local inner = 0.5 * sqrt(ref.w * ref.w + ref.d * ref.d)
  local deepest = 0
  for _, it in ipairs(items) do deepest = max(deepest, it.d) end
  local b = { desc = (opts.arc and "in an arc around the " or "in a ring around the ") .. (ref.kind or "middle"), dropped = 0, ring = true }
  b.build = function()
    local poses = Layout.circle(ref.x, ref.y, items, { gap = opts.gap or 6, min_r = inner + deepest / 2 + 3, start = ref.theta, stop = opts.stop,
                                                        arc = opts.arc })
    local out = {}
    for i, it in ipairs(items) do out[#out + 1] = P(it.inst, poses[i], "around", { sits_on = ref.id }) end
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
        local side = opts.only_side or ((i % 2 == 1) and 1 or -1)
        if opts.flip and not opts.only_side then side = -side end
        local v = side * (ref.w / 2 + gap + used[side] + it.w / 2)
        used[side] = used[side] + it.w + gap
        -- opts.align: the backs line up with the reference's (a deeper thing beside a wall-flush one would stick out behind the wall)
        local pose = to_world(rf, opts.align and (it.d - ref.d) / 2 or 0, v, 0)
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

--- Items standing on a reference's flat top. b.dropped counts the ones that do not fit on it (they are left off, never
--- left hanging over the edge).
local function on_builder(ref, insts, opts)
  opts = opts or {}
  local items = to_items(insts)
  local top = ref.topf
  local b = { desc = "on the " .. (ref.kind or "surface"), dropped = 0 }
  local function lay(rf)
    local c = to_world(rf, top.u, top.v, 0)
    return Layout.on_top({ x = c.x, y = c.y, w = top.w, d = top.d, theta = ref.theta, z = top.z + (ref.zoff or 0) }, items,
      { gap = 4, bias = opts.bias or -1, overhang = ON_OVERHANG })
  end
  local _, dropped = lay({ x = 0, y = 0, theta = 0 })
  b.dropped = dropped
  b.build = function()
    local poses = lay({ x = ref.x, y = ref.y, theta = ref.theta })
    local out = {}
    for i, it in ipairs(items) do
      if poses[i] then out[#out + 1] = P(it.inst, poses[i], "on", { sits_on = ref.id }) end
    end
    return out
  end
  b.radius = 0
  return b
end

--- How many of `insts` stand on this top (Layout.on_top's own answer).
local function top_holds(topf, insts)
  local _, dropped = Layout.on_top({ x = 0, y = 0, w = topf.w, d = topf.d, theta = 0, z = 0 }, to_items(insts), { gap = 4, overhang = ON_OVERHANG })
  return #insts - dropped
end

-- the surface a small thing wants, and the kinds to climb through when it does not hold them all
local SURFACE_FOR = { computer = "desk", phone = "desk", ["desk lamp"] = "side table", mug = "coffee table", microwave = "table" }
local SURFACE_LADDER = { "side table", "coffee table", "table", "desk", "meeting table" }

--- The smallest surface (of the kind the thing wants, else of any; `only` restricts it to one kind) that holds all of `insts`.
--- Returns kind, variant index, how many it holds. When none holds them all, the one that holds most.
function Plan:surface_for(first_kind, insts, only)
  local pref = only or SURFACE_FOR[first_kind] or "table"
  local cands = {}
  local function consider(kind_id)
    local k = self.kinds[kind_id]
    if not k then return end
    for vi, v in ipairs(k.variants) do
      local tf = top_frame(v)
      if tf then cands[#cands + 1] = { kind = kind_id, vi = vi, cap = top_holds(tf, insts), area = v.w * v.d, pref = (kind_id == pref) } end
    end
  end
  consider(pref)
  if not only then
    for _, kid in ipairs(SURFACE_LADDER) do if kid ~= pref then consider(kid) end end
  end
  local best
  for _, c in ipairs(cands) do
    local all = c.cap >= #insts
    if not best then best = c
    else
      local b_all = best.cap >= #insts
      if all ~= b_all then if all then best = c end
      elseif all then
        if c.pref ~= best.pref then if c.pref then best = c end
        elseif c.area < best.area then best = c end
      else
        if c.cap > best.cap or (c.cap == best.cap and c.pref and not best.pref) then best = c end
      end
    end
  end
  if not best then return nil end
  return best.kind, best.vi, best.cap
end

--- How well `n` of `seat` sit round a table variant: lost seats, and whether any two seats touch.
local function seating_of(kind_id, v, seat, n)
  local insts = {}
  for i = 1, n do insts[i] = seat end
  local probe = { x = 0, y = 0, theta = 0, w = v.w, d = v.d, kind = kind_id, topf = top_frame(v), inst = { v = v, kind = kind_id } }
  local ab = around_builder(probe, insts, { gap = 5 })
  local pls = ab.build()
  local rects, tight = {}, false
  for i, p in ipairs(pls) do rects[i] = rect(p.x, p.y, p.w, p.d, p.theta) end
  for i = 1, #rects do
    for j = i + 1, #rects do
      if Layout.overlap(rects[i], rects[j], 0.4) then tight = true; break end
    end
    if tight then break end
  end
  return ab.dropped, tight
end

--- The smallest table (of this kind, else the meeting table) that seats `n` of `seat` without touching, found by seating them.
--- Returns kind, variant index, lost seats.
function Plan:table_for_seats(kind_id, seat, n)
  local function best_of(kid)
    local k = self.kinds[kid]
    if not k then return nil end
    local best
    for vi, v in ipairs(k.variants) do
      local lost, tight = seating_of(kid, v, seat, n)
      local rank = (lost == 0) and (tight and 2 or 1) or 3
      local area = v.w * v.d
      local c = { kind = kid, vi = vi, lost = lost, rank = rank, area = area }
      if not best or rank < best.rank or (rank == best.rank and ((rank == 3 and (lost < best.lost or (lost == best.lost and area < best.area)))
                                                                     or (rank ~= 3 and area < best.area))) then
        best = c
      end
    end
    return best
  end
  local best = best_of(kind_id)
  if best and best.rank > 1 and kind_id == "table" then
    local alt = best_of("meeting table")
    if alt and (alt.rank < best.rank or (alt.rank == best.rank and alt.lost < best.lost)) then best = alt end
  end
  if not best then return nil end
  return best.kind, best.vi, best.lost
end

-- ---------------------------------------------------------------------------------------------- rooms
local ROLE_ALIAS = { on_desk = "on", on_wall_unit = "on", beside_bed = "beside_bed", at_desk = "at_desk", row_wall = "row_wall" }
local STATION_GAP = 30                                  -- between the tables (and desks) of a room that has several

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

--- Hand each small thing to the first surface that still has room for it, the kind it wants first. surfaces = { {tf, kind,
--- role}, ... }. Returns assigned (surface index -> list of insts) and the insts no surface holds.
local function assign_on(on_insts, surfaces)
  local assigned, leftover = {}, {}
  for _, inst in ipairs(on_insts) do
    local want = SURFACE_FOR[inst.kind]
    local order = {}
    for si, s in ipairs(surfaces) do if s.kind == want then order[#order + 1] = si end end
    for si, s in ipairs(surfaces) do if s.kind ~= want and (s.role == "desk" or s.role == "center") then order[#order + 1] = si end end
    for si, s in ipairs(surfaces) do if s.kind ~= want and not (s.role == "desk" or s.role == "center") then order[#order + 1] = si end end
    local done = false
    for _, si in ipairs(order) do
      local trial = {}
      for i, x in ipairs(assigned[si] or {}) do trial[i] = x end
      trial[#trial + 1] = inst
      if top_holds(surfaces[si].tf, trial) == #trial then assigned[si] = trial; done = true; break end
    end
    if not done then leftover[#leftover + 1] = inst end
  end
  return assigned, leftover
end

--- Chairs at a desk: side by side along its front, the overflow at its sides. Returns placements and how many did not fit.
local function desk_chairs(dp, chairs)
  local out = {}
  if #chairs == 0 then return out, 0 end
  local dtop = top_frame(dp.inst.v)
  local per = max(1, floor((dp.w + 4) / (chairs[1].inst.w + 4)))
  local row = min(per, #chairs)
  local frame = { x = dp.x, y = dp.y, theta = dp.theta }
  for k = 1, row do
    local inst = chairs[k].inst
    local tuck = (dtop and dtop.z >= 28 and TUCKS_UNDER[dp.inst.kind or ""]) and min(9, 0.35 * inst.d) or 0
    local u = dp.inst.d / 2 + inst.d / 2 - tuck
    local v = (k - (row + 1) / 2) * (inst.w + 4)
    out[#out + 1] = P(inst, to_world(frame, u, v, 180), "at_desk", { tuck = dp, tuck_group = true, sits_on = nil, tuck_depth = tuck })
  end
  local lost = 0
  if #chairs > row then
    local rest = {}
    for k = row + 1, #chairs do rest[#rest + 1] = chairs[k].inst end
    local ab = around_builder({ x = dp.x, y = dp.y, theta = dp.theta, w = dp.inst.w, d = dp.inst.d, kind = dp.inst.kind, placement = dp, inst = dp.inst },
      rest, { gap = 5, order = { "left", "right", "back" } })
    for _, p in ipairs(ab.build()) do p.role = "at_desk"; out[#out + 1] = p end
    lost = ab.dropped or 0
  end
  return out, lost
end

--- A furnished room from role-tagged entries: walls (back, left, right; the front faces the viewer) carry the wall
--- things, a cluster in the middle carries the table and the seats, corners take the lamps and plants.
--- entries = { {inst=, role=}, ... }.  Returns builder with .build(frame) and .size = {w, d} (the room); b.dropped counts the
--- seats that found no place at their table, b.on_left the small things no surface could hold.
function Plan:room(entries, rng, ctx, opts)
  opts = opts or {}
  -- a seat for a desk needs a desk, a chair "round the table" needs a table: without one it is just a seat
  local has_desk_kind, has_table, has_dining, has_coffee, n_conv = false, false, false, false, 0
  for _, e in ipairs(entries) do
    if e.inst.kind == "desk" and e.role ~= "at_desk" then has_desk_kind = true end
    if e.role == "center" then
      has_table = true
      if e.inst.kind == "coffee table" then has_coffee = true else has_dining = true end
    end
    if e.role == "main_seat" or e.role == "focus" then n_conv = n_conv + 1 end
  end
  local by = {}
  for _, e in ipairs(entries) do
    local role = e.role
    if role == "at_desk" and not has_desk_kind then role = "side_seat" end
    if role == "around" and not has_table then role = "side_seat" end
    -- chairs "round" a coffee table that couches already stand at: they are conversation seats, not a second ring of seats
    if role == "around" and not has_dining and n_conv > 0 then role = "side_seat" end
    by[role] = by[role] or {}
    table.insert(by[role], role == e.role and e or { inst = e.inst, role = role, spec = e.spec })
  end
  local function list(role) return by[role] or {} end
  local has_focus = #list("focus") > 0

  -- -- the stations: every table (with the seats round it) and every desk (with its chairs) stands as a unit in the middle
  local centers = list("center")
  local conv_tbl
  for _, e in ipairs(centers) do if e.inst.kind == "coffee table" then conv_tbl = e; break end end
  -- a dining table with chairs round it is the chairs' table: couches never share its sides with them
  if not conv_tbl and #list("around") == 0 then conv_tbl = centers[1] end
  local dining = {}
  for _, e in ipairs(centers) do if e.inst.kind ~= "coffee table" then dining[#dining + 1] = e end end
  if #dining == 0 then dining = centers end
  -- couches and armchairs stand round the coffee table; with nothing to look at and nothing to gather round, a couch is a
  -- wall thing (each seat is placed once, in one place)
  local mains_on_wall = (not has_focus) and conv_tbl == nil and #list("side_seat") == 0
  local conv_seats = {}
  if not mains_on_wall then for _, e in ipairs(list("main_seat")) do conv_seats[#conv_seats + 1] = e end end
  for _, e in ipairs(list("side_seat")) do conv_seats[#conv_seats + 1] = e end

  local stations, station_of = {}, {}
  for _, e in ipairs(centers) do
    local st = { kind = "center", entry = e, seats = {}, conv = (e == conv_tbl) }
    stations[#stations + 1] = st
    station_of[e] = st
  end
  if not conv_tbl and #conv_seats > 0 then table.insert(stations, 1, { kind = "center", seats = {}, conv = true }) end
  for i, e in ipairs(list("around")) do
    local st = station_of[dining[(i - 1) % #dining + 1]]
    table.insert(st.seats, e)
  end
  local desk_stations = {}
  for _, e in ipairs(list("desk")) do
    local st = { kind = "desk", entry = e, chairs = {} }
    stations[#stations + 1] = st
    desk_stations[#desk_stations + 1] = st
  end
  for i, e in ipairs(list("at_desk")) do
    if #desk_stations > 0 then table.insert(desk_stations[(i - 1) % #desk_stations + 1].chairs, e) end
  end

  local function station_build(st, frame)
    local out = {}
    st.lost = 0
    if st.kind == "center" then
      local centre_p
      local ph = (st.conv and has_focus) and 90 or 0     -- with a screen on the left wall the table runs along u
      if st.entry then
        centre_p = P(st.entry.inst, to_world(frame, 0, 0, ph), "center")
        out[#out + 1] = centre_p
      end
      local ref = centre_p
      -- conversation seats: main seats take the right side (opposite the screen), side seats the back and front
      if st.conv and #conv_seats > 0 then
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
      if #st.seats > 0 and ref then
        local seats = {}
        for i, e in ipairs(st.seats) do seats[i] = e.inst end
        local ab = around_builder({ x = ref.x, y = ref.y, theta = Layout.wrap(frame.theta + ph), w = ref.inst.w, d = ref.inst.d, kind = ref.inst.kind, placement = ref, inst = ref.inst },
          seats, { gap = 5 })
        st.lost = ab.dropped or 0
        for _, p in ipairs(ab.build()) do out[#out + 1] = p end
      end
    else
      -- a desk and the chairs at it
      local dp = P(st.entry.inst, to_world(frame, 0, 0, 0), "desk")
      out[#out + 1] = dp
      local ps, lost = desk_chairs(dp, st.chairs)
      st.lost = lost
      for _, p in ipairs(ps) do out[#out + 1] = p end
    end
    return out
  end

  -- the stations in a grid of their own (one row up to three of them), each measured alone in a neutral frame
  local offsets = {}
  if #stations <= 1 then
    offsets[1] = { u = 0, v = 0 }
  else
    local ext = {}
    for i, st in ipairs(stations) do
      local x0, x1, y0, y1 = bbox_extent(station_build(st, { x = 0, y = 0, theta = 0 }))
      ext[i] = { u0 = x0, u1 = x1, v0 = -y1, v1 = -y0 }                -- theta 0: world x is u, world y is -v
    end
    local cols = (#stations <= 3) and #stations or ceil(sqrt(#stations))
    local cursor_u, rows = 0, {}
    for i = 1, #stations do
      local r = floor((i - 1) / cols) + 1
      rows[r] = rows[r] or {}
      table.insert(rows[r], i)
    end
    local lo_u, hi_u = 1e9, -1e9
    for _, row in ipairs(rows) do
      local u0, u1 = 1e9, -1e9
      for _, i in ipairs(row) do u0, u1 = min(u0, ext[i].u0), max(u1, ext[i].u1) end
      local ou = cursor_u - u0
      lo_u, hi_u = min(lo_u, cursor_u), max(hi_u, ou + u1)
      local vcur, vlo, vhi = 0, 1e9, -1e9
      local places = {}
      for _, i in ipairs(row) do
        local ov = vcur - ext[i].v0
        places[i] = ov
        vlo, vhi = min(vlo, vcur), max(vhi, ov + ext[i].v1)
        vcur = ov + ext[i].v1 + STATION_GAP
      end
      for _, i in ipairs(row) do offsets[i] = { u = ou, v = places[i] - (vlo + vhi) / 2 } end
      cursor_u = ou + u1 + STATION_GAP
    end
    -- one row keeps every table and desk on the cluster's own centre line; several rows are centred as a block
    for i = 1, #stations do offsets[i].u = (#rows > 1) and (offsets[i].u - (lo_u + hi_u) / 2) or 0 end
  end

  local function cluster_build(frame)
    local out = {}
    for i, st in ipairs(stations) do
      local f = to_world(frame, offsets[i].u, offsets[i].v, 0)
      f.theta = frame.theta
      for _, p in ipairs(station_build(st, f)) do out[#out + 1] = p end
    end
    return out
  end

  -- -- sizing: measure the cluster once in a neutral frame
  local probe = cluster_build({ x = 0, y = 0, theta = 0 })
  local cx0, cx1, cy0, cy1 = bbox_extent(probe)
  local cu, cv = (cx1 - cx0), (cy1 - cy0)               -- theta 0: world x is u, world y is -v
  if #probe == 0 then cu, cv = 0, 0 end
  local cv_left = cy1                                    -- how far the cluster reaches toward -v (v = -y in this probe)
  local around_lost, around_n = 0, #list("around")
  for _, st in ipairs(stations) do around_lost = around_lost + (st.lost or 0) end

  -- -- wall residents
  local back, left, right = {}, {}, {}
  local function to_wall(wall, e) table.insert(wall, e) end
  for _, e in ipairs(list("wall_head")) do to_wall(back, e) end
  for _, e in ipairs(list("row_wall")) do to_wall(back, e) end
  if has_focus then for _, e in ipairs(list("focus")) do to_wall(left, e) end
  else for _, e in ipairs(list("focus")) do to_wall(back, e) end end
  if mains_on_wall then
    for _, e in ipairs(list("main_seat")) do to_wall(back, e) end
  end
  local wall_items = list("wall")
  -- bath fixtures keep a hand's width between them (a toilet 2 HU from the tub tipped over in the engine); other things stand 2 apart
  local function wall_gap(a, b)
    return ((a.inst.k and a.inst.k.category == "bath") or (b.inst.k and b.inst.k.category == "bath")) and 8 or 2
  end
  local function wall_len(w)
    local s = 0
    for i, e in ipairs(w) do s = s + e.inst.w + (i > 1 and wall_gap(w[i - 1], e) or 0) end
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
  local beside_w = 0                                        -- the end tables stand at the ends of the row of beds
  if #list("wall_head") > 0 then
    for _, e in ipairs(list("beside_bed")) do beside_w = beside_w + e.inst.w + 4 end
  end
  -- the cluster stands in the middle (not against the left wall, as it does with a screen there), so the deeper side row counts on both sides
  local side_w = has_focus and (dl + dr) or (2 * max(dl, dr))
  -- side tables by seats that stand on a wall take the two ends of the row, alternately: the room is wide enough for them
  local side_end = { [1] = 0, [-1] = 0 }
  if mains_on_wall then
    for i, e in ipairs(list("beside_main")) do
      local s = (i % 2 == 1) and 1 or -1
      side_end[s] = side_end[s] + e.inst.w + 4
    end
  end
  local W = max(opts.min_w or 230, wall_len(back) + 60 + beside_w, wall_len(back) + 2 * max(30, side_end[1] + 4, side_end[-1] + 4),
                side_w + 34 + cv + 34, 120 + cv)
  -- a side row that shares a corner with the back row starts beyond the back row's depth, and the room is deep enough for that
  local function corner_shared(w) return #back > 0 and #w > 0 and wall_len(back) / 2 + depth_of(w) + 2 > W / 2 end
  local function side_need(w) return #w == 0 and 0 or (wall_len(w) + 60 + (corner_shared(w) and db or 0)) end
  local D = max(opts.min_d or 190, db + 34 + cu + 40, side_need(left), side_need(right))
  if #list("wall_head") + #desk_stations > 0 then D = max(D, db + 34 + cu + 60) end
  do
    local cw = 0
    for _, e in ipairs(list("corner")) do cw = max(cw, e.inst.w, e.inst.d) end
    if cw > 0 and wall_len(back) > 0 then W = max(W, wall_len(back) + 2 * cw + 12) end
  end
  if opts.room then W, D = opts.room.w, opts.room.d end

  local b = { desc = opts.desc or "as a room", size = { w = W, d = D }, dropped = around_lost, around_n = around_n, on_left = 0 }

  -- things that slide along their wall to get out of each other's way
  local MOVABLE = { wall = true, corner = true, beside_bed = true, beside_main = true, focus = true }
  local function repair(out, frame)
    local fx0, fy0 = fwd(frame.theta)
    local rects = {}
    for i, p in ipairs(out) do rects[i] = rect(p.x, p.y, p.w, p.d, p.theta) end
    local function clashes(p, i)
      local r = rects[i]
      for j, q in ipairs(out) do
        if j ~= i and (q.zoff or 0) <= 0.01 and (p.zoff or 0) <= 0.01
           and not (p.tuck == q or q.tuck == p or (p.tuck and p.tuck == q.tuck and p.tuck_group)) then
          if Layout.overlap(r, rects[j], 0.4) then return true end
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
        if expired(ctx) then break end
        local tx, ty = Layout.right(p.theta)
        local dirs = { { tx, ty } }
        if p.role == "corner" then
          local gx, gy = Layout.right(frame.theta)
          dirs = { { gx, gy }, { fx0, fy0 } }
        end
        local x0, y0 = p.x, p.y
        local moved = false
        local r = rects[i]
        for k = 1, 30 do
          for _, dir in ipairs(dirs) do
            for _, sgn in ipairs({ 1, -1 }) do
              p.x, p.y = x0 + dir[1] * sgn * k * 8, y0 + dir[2] * sgn * k * 8
              r.x, r.y = p.x, p.y
              if inside(p) and not clashes(p, i) then moved = true; break end
            end
            if moved then break end
          end
          if moved then break end
        end
        if not moved then
          p.x, p.y = x0, y0
          r.x, r.y = x0, y0
        else
          -- whatever stands on a surface that slid goes with it
          local ddx, ddy = p.x - x0, p.y - y0
          for _, q in ipairs(out) do if q.on_placement == p then q.x, q.y = q.x + ddx, q.y + ddy end end
        end
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
    -- the cluster sits in the middle of the free floor between the back wall's things and the front edge
    local c_lo = -D / 2 + db + 34 + cu / 2
    local c_hi = D / 2 - cu / 2 - 20
    local c_u = max(c_lo, min(c_hi, db / 2))
    if c_u > c_hi then c_u = c_hi end
    -- the cluster's own origin is not its middle (seats on one side only): it is the middle that is centred in the room
    local mid_u, mid_v = (cx0 + cx1) / 2, -(cy0 + cy1) / 2
    if #probe == 0 then mid_u, mid_v = 0, 0 end
    -- with a screen on the left wall the cluster moves toward it, to a viewing distance, instead of floating mid-room
    local c_v = 0
    if has_focus then
      local want = -W / 2 + dl + 90 + cv_left
      c_v = min(0, want)
    end
    local cf = to_world(frame, c_u - mid_u + (opts.shift and opts.shift[1] or 0), c_v - ((c_v == 0) and mid_v or 0) + (opts.shift and opts.shift[2] or 0), 0)
    for _, p in ipairs(cluster_build({ x = cf.x, y = cf.y, theta = frame.theta })) do out[#out + 1] = p end
    -- the back wall: a row, flush
    local function run(wall, fixed_axis, sign, phi, centre_on)
      local total = wall_len(wall)
      local pos = -total / 2
      if centre_on then pos = centre_on - total / 2 end
      if fixed_axis == "v" and corner_shared(wall) then pos = max(pos, -D / 2 + db + 2) end
      for wi, e in ipairs(wall) do
        if wi > 1 then pos = pos + wall_gap(wall[wi - 1], e) - 2 end
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
    -- chairs for a desk that stands on a wall (an office): in front of it
    if #desk_stations == 0 and #list("at_desk") > 0 then
      local wall_desks = {}
      for _, p in ipairs(out) do if p.inst.kind == "desk" and (p.zoff or 0) == 0 then wall_desks[#wall_desks + 1] = p end end
      local shares = {}
      for i, e in ipairs(list("at_desk")) do
        if #wall_desks > 0 then
          local k = (i - 1) % #wall_desks + 1
          shares[k] = shares[k] or {}
          table.insert(shares[k], e)
        end
      end
      for k, dp in ipairs(wall_desks) do
        local ps = desk_chairs(dp, shares[k] or {})
        for _, p in ipairs(ps) do out[#out + 1] = p end
      end
    end
    -- corners: back-left, back-right, front-left, front-right; a second thing in a corner goes along the wall, and a wall
    -- that is used up starts a second layer
    local corner_items = list("corner")
    local slots = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } }
    local c_used, c_layer, c_depth = { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }
    for i, e in ipairs(corner_items) do
      local slot = (i - 1) % 4 + 1
      local s = slots[slot]
      -- a thing that looks the same from every side keeps the room's own alignment; the rest turn to face the middle
      local sym = e.inst.v.symmetric
      local r = (sym and max(e.inst.w, e.inst.d) / 2 or sqrt(e.inst.w ^ 2 + e.inst.d ^ 2) / 2) + 2
      if c_used[slot] > 0 and c_used[slot] + 2 * r > W / 2 - 4 then
        c_layer[slot], c_used[slot], c_depth[slot] = c_layer[slot] + c_depth[slot], 0, 0
      end
      local pose = to_world(frame, s[1] * (D / 2 - r - c_layer[slot]), s[2] * (W / 2 - r - c_used[slot]), 0)
      c_used[slot] = c_used[slot] + 2 * r
      c_depth[slot] = max(c_depth[slot], 2 * r)
      pose.theta = sym and frame.theta or Layout.angle_of(frame.x - pose.x, frame.y - pose.y)
      out[#out + 1] = P(e.inst, pose, "corner")
    end
    -- beside things: end tables by the bed head (at the two ends of a row of beds), by the main seat
    local beds = {}
    for _, p in ipairs(out) do if p.role == "wall_head" then beds[#beds + 1] = p end end
    local rx0, ry0 = Layout.right(frame.theta)
    table.sort(beds, function(a, c) return (a.x - frame.x) * rx0 + (a.y - frame.y) * ry0 < (c.x - frame.x) * rx0 + (c.y - frame.y) * ry0 end)
    local at_end, n_beside = { [1] = 0, [-1] = 0 }, 0
    for _, e in ipairs(list("beside_bed")) do
      if #beds > 0 then
        local side = (n_beside % 2 == 0) and 1 or -1
        local bed = (side == 1) and beds[#beds] or beds[1]
        local v = side * (bed.w / 2 + e.inst.w / 2 + 4 + at_end[side])
        at_end[side] = at_end[side] + e.inst.w + 4
        n_beside = n_beside + 1
        local pose = to_world({ x = bed.x, y = bed.y, theta = bed.theta }, -(bed.d / 2 - e.inst.d / 2), v, 0)
        pose.theta = bed.theta
        out[#out + 1] = P(e.inst, pose, "beside_bed", { bedside = bed })
      else
        local pose = to_world(frame, 0, 0, 0)
        out[#out + 1] = P(e.inst, pose, "beside_bed")
      end
    end
    -- side tables by the main seats: on a wall the tables stand at the two ENDS of the row of seats (alternating, each further one
    -- beyond the last), so they never share a place with a seat; round a table they keep to the first seat's right
    local mains = {}
    for _, p in ipairs(out) do if p.role == "main_seat" then mains[#mains + 1] = p end end
    local first_main = mains[1]
    if mains_on_wall then
      table.sort(mains, function(a, c) return (a.x - frame.x) * rx0 + (a.y - frame.y) * ry0 < (c.x - frame.x) * rx0 + (c.y - frame.y) * ry0 end)
    end
    local main_end, n_main = { [1] = 0, [-1] = 0 }, 0
    for _, e in ipairs(list("beside_main")) do
      local pose
      if first_main then
        local side = (mains_on_wall and n_main % 2 == 1) and -1 or 1
        local main = first_main
        if mains_on_wall then main = (side == 1) and mains[#mains] or mains[1] end
        local v = side * (main.w / 2 + e.inst.w / 2 + 4 + main_end[side])
        main_end[side] = main_end[side] + e.inst.w + 4
        n_main = n_main + 1
        pose = to_world({ x = main.x, y = main.y, theta = main.theta }, 0, v, 0)
        pose.theta = main.theta
      else pose = to_world(frame, 0, 0, 0) end
      out[#out + 1] = P(e.inst, pose, "beside_main")
    end
    -- things standing on surfaces: each goes to a surface with room for it
    local surfaces = {}
    for _, p in ipairs(out) do
      local tf = top_frame(p.inst.v)
      if tf and (p.zoff or 0) == 0 then surfaces[#surfaces + 1] = { p = p, tf = tf, kind = p.kind, role = p.role } end
    end
    local on_insts = {}
    for i, e in ipairs(list("on")) do on_insts[i] = e.inst end
    local assigned, leftover = assign_on(on_insts, surfaces)
    for si, s in ipairs(surfaces) do
      if assigned[si] then
        local ob = on_builder({ x = s.p.x, y = s.p.y, theta = s.p.theta, kind = s.p.kind, topf = s.tf, id = nil, zoff = s.p.zoff }, assigned[si],
          { bias = (s.p.role == "desk") and -1 or 0 })
        for _, p in ipairs(ob.build()) do p.on_placement = s.p; out[#out + 1] = p end
      end
    end
    b.on_left = #leftover
    return out
  end
local function full_build(frame)    local out = raw_build(frame)    repair(out, frame)    if opts.mirror then mirror(out, frame) end    return out  end
  -- things in the corners must not stand in the cluster: the room grows (the cluster stays as it is) until they do not. The cluster
  -- is measured alone, so a crowded middle (several tables) used to reach the corner plants whatever the room's size.
  if not opts.room and #list("corner") > 0 and #probe > 0 then
    for _ = 1, 24 do
      local out = {}
      for _, p in ipairs(full_build({ x = 0, y = 0, theta = 0 })) do out[#out + 1] = p end
      local rects, clash = {}, false
      for i, p in ipairs(out) do rects[i] = rect(p.x, p.y, p.w, p.d, p.theta) end
      for i, p in ipairs(out) do
        if p.role == "corner" then
          for j, q in ipairs(out) do
            if q.role ~= "corner" and (q.zoff or 0) <= 0.01 and Layout.overlap(rects[i], rects[j], 0.4) then clash = true; break end
          end
        end
        if clash then break end
      end
      if not clash then break end
      W, D = W + 16, D + 16
      b.size = { w = W, d = D }
    end
  end
  -- The room is the same wherever it stands: built once in a neutral frame, then moved rigidly to each frame a search asks for (a
  -- search builds the same room hundreds of times, and laying out its stations, walls and corners is most of what a candidate costs).
  -- References between its pieces (a seat tucked into its table, a load on its surface) follow the copies.
  local neutral
  b.build = function(frame)
    if not neutral then neutral = full_build({ x = 0, y = 0, theta = 0 }) end
    local out, copy = {}, {}
    for i, p in ipairs(neutral) do
      local q = {}
      for k, v in pairs(p) do q[k] = v end
      local pose = to_world(frame, p.x, -p.y, 0)
      q.x, q.y, q.theta = pose.x, pose.y, Layout.wrap(frame.theta + p.theta)
      copy[p], out[i] = q, q
    end
    for _, q in ipairs(out) do
      if q.tuck and copy[q.tuck] then q.tuck = copy[q.tuck] end
      if q.on_placement and copy[q.on_placement] then q.on_placement = copy[q.on_placement] end
      if q.bedside and copy[q.bedside] then q.bedside = copy[q.bedside] end
    end
    return out
  end
  b.radius = sqrt((W / 2) ^ 2 + (D / 2) ^ 2)
  return b
end

-- ---------------------------------------------------------------------------------------------- the planner proper
-- Words a parser note may use besides the library's own names: a note with any other word could carry a player's word.
local NOTE_WORDS = {}
for w in ("i read that as a an the it this for to of and or mean meant understood heard took taken used so guessed maybe probably "
          .. "think thought assumed closest nearest you your what"):gmatch("%a+") do NOTE_WORDS[w] = true end

function Plan.new(catalog)
  local self = setmetatable({}, Plan)
  self.catalog = catalog
  self.kinds = catalog.kinds
  -- the library's own words, for vetting notes the parser wrote
  self.vocab = {}
  for id, k in pairs(catalog.kinds) do
    for w in tostring(id):lower():gmatch("%a+") do self.vocab[w] = true end
    for _, nm in ipairs(k.names or {}) do
      for w in tostring(nm):lower():gmatch("%a+") do self.vocab[w] = true end
    end
  end
  return self
end

--- May a note the parser wrote be shown? Only if every word in it is the library's or a plain connecting word.
function Plan:safe_note(s)
  if type(s) ~= "string" or #s > 160 then return false end
  for w in s:lower():gmatch("%a+") do
    local one = w:gsub("ves$", "f"):gsub("ies$", "y"):gsub("es$", ""):gsub("s$", "")          -- plurals of the library's words
    if not (self.vocab[w] or self.vocab[one] or self.vocab[w:gsub("s$", "")] or NOTE_WORDS[w]) then return false end
  end
  return true
end

local function finish(self, ctx, pls, res, group_id)
  -- "facing away from the tv": each one looks along the line from that thing to itself
  local apt, apx, apy = ctx._away_pt, nil, nil
  if apt == "referent" then
    for _, q in ipairs(pls) do if q.role == "referent" then apx, apy = q.x, q.y end end
  elseif type(apt) == "table" then apx, apy = apt.x, apt.y end
  for _, p in ipairs(pls) do
    local v = p.inst.v
    local zbot = (p.fz or ctx.pos.z) + (p.zoff or 0)
    local theta = p.theta
    if res.away and p.role ~= "referent" then                                       -- "facing away from ..."
      if apx and (p.x - apx) ^ 2 + (p.y - apy) ^ 2 > 1 then theta = Layout.angle_of(p.x - apx, p.y - apy) else theta = theta + 180 end
    end
    local origin, yaw = Plan.entity_pose(v, p.x, p.y, zbot, theta)
    res.placements[#res.placements + 1] = {
      kind = p.kind, vi = p.inst.vi, model = v.model, x = p.x, y = p.y, z = zbot, theta = Layout.wrap(theta),
      w = p.w, d = p.d, h = p.h, origin = origin, yaw = yaw, mass = v.mass, material = v.material, color = p.inst.color,
      color_name = p.inst.color_name, role = p.role, group = group_id, skins = v.skins, top = v.top,
      on = ((p.zoff or 0) > 0.01 or p.on_placement) and true or nil, symmetric = v.symmetric,
    }
  end
end

local function aim_or_front(ctx)
  return ctx.aim or ctx.pos
end

--- Everything standing that matches a spec (kind + colour), nearest to `from` first. A named colour matches only that colour;
--- the second result lists the uncoloured ones of the kind, for a verb that may fall back to them when none has the colour.
local function registry_matches(self, ctx, kind, adjs, from, list)
  local out, plain = {}, {}
  local _, color_name = color_of(adjs)
  for _, e in ipairs(list or ctx.registry or {}) do
    if e.kind == kind and in_library(self, e) then
      if not color_name or (e.color_name and same_color(e.color_name, color_name)) then out[#out + 1] = e
      elseif not e.color_name then plain[#plain + 1] = e end
    end
  end
  local function by_distance(a, b)
    local da = (a.x - from.x) ^ 2 + (a.y - from.y) ^ 2
    local db = (b.x - from.x) ^ 2 + (b.y - from.y) ^ 2
    if da ~= db then return da < db end
    return (a.seq or 0) > (b.seq or 0)
  end
  table.sort(out, by_distance)
  table.sort(plain, by_distance)
  return out, plain
end

--- The standing thing that is the heart of a group (the biggest), for "next to it".
local function group_anchor(entries)
  local best
  for _, e in ipairs(entries) do
    if not best or e.w * e.d > best.w * best.d + 1e-6 then best = e end
  end
  return best
end

--- The requester's most recent group, as standing things (ctx.last holds their ids).
local function last_entries(self, ctx)
  local ids = {}
  for _, id in ipairs(ctx.last or {}) do ids[id] = true end
  local out = {}
  for _, e in ipairs(ctx.mine or ctx.registry or {}) do
    if ids[e.id] and in_library(self, e) then out[#out + 1] = e end
  end
  return out
end

--- What was placed, set against what was asked for: "only 4 of the 6 chairs fit".
local function note_missing(res, insts, pls)
  local placed = {}
  for _, p in ipairs(pls) do placed[p.inst] = true end
  local want, got, order = {}, {}, {}
  for _, inst in ipairs(insts) do
    if not want[inst.kind] then want[inst.kind] = 0; got[inst.kind] = 0; order[#order + 1] = inst.kind end
    want[inst.kind] = want[inst.kind] + 1
    if placed[inst] then got[inst.kind] = got[inst.kind] + 1 end
  end
  local lost = 0
  for _, k in ipairs(order) do
    if got[k] < want[k] then
      lost = lost + want[k] - got[k]
      add_note(res, "only " .. got[k] .. " of the " .. want[k] .. " " .. plural(k, want[k]) .. " fit")
    end
  end
  return lost
end

--- The things a request wants, expanded (scenes become their furniture) and checked against the library.
function Plan:wanted(req, res, ctx)
  local want, seen_unknown = {}, {}
  local limit = (ctx and ctx.max_per_kind) or MAX_PER_KIND
  local function clamp(n, kind)
    if n > limit then
      add_note(res, "I limited " .. plural(kind, 2) .. " to " .. limit)
      res.clamped = true
      return limit
    end
    return n
  end
  for _, it in ipairs(req.items or {}) do
    local kind = it.kind
    if tostring(kind):match("^UNKNOWN") then
      local name = it.phrase or kind:sub(9)
      if not seen_unknown[name] then
        seen_unknown[name] = true
        note_unknown(res, name)
        add_note(res, T.library)
      end
    elseif kind == "@last" or kind == "@all" then
      -- only meaningful for management verbs
    elseif not self.kinds[kind] then
      -- a kind the library lacks (an agent's request, or a library that lost it since)
      note_unknown(res, kind)
      add_note(res, T.library)
    else
      local n = it.count
      if n == 0 then n = 1 end
      -- "a lamp on the desk" is a desk lamp, not the standing one
      if kind == "floor lamp" and req.arrangement == "on" and (it.word == "lamp" or it.word == "lamps") and self.kinds["desk lamp"] then
        kind = "desk lamp"
      end
      n = clamp(n, kind)
      if it.approx then add_note(res, "the closest I have to that is " .. article(kind)) end
      want[#want + 1] = { kind = kind, count = n, adjectives = it.adjectives or {}, word = it.word, plural = it.plural }
    end
  end
  -- "a table for six": the seats the table has to have
  local seats = tonumber(req.seats)
  if seats and seats >= 1 and #want > 0 then
    seats = clamp(floor(seats), "chair")
    local has_table, chair
    for _, w in ipairs(want) do
      local k = self.kinds[w.kind]
      if k.category == "table" and w.kind ~= "desk" and w.kind ~= "side table" and w.kind ~= "coffee table" then has_table = true end
      if w.kind == "chair" or w.kind == "stool" or w.kind == "office chair" then chair = w end
    end
    if chair then chair.count = seats
    elseif has_table and self.kinds["chair"] then want[#want + 1] = { kind = "chair", count = seats, adjectives = {} } end
  end
  if req.scene and req.scene ~= "" then
    -- an item named beside the scene takes the place of the scene's own (a living room with a red couch has one couch, red;
    -- "another" and "more" mean extra ones)
    local more = false
    for _, w in ipairs({ "another", "more", "extra", "additional", "also" }) do
      if (" " .. (req.normalised or "") .. " "):find(" " .. w .. " ", 1, true) then more = true end
    end
    local lines = {}
    -- "without the tv": the scene leaves that kind out; a kind asked for by name in the same request is still made
    local excluded = {}
    for _, k in ipairs(req.exclude or {}) do excluded[k] = true end
    for _, line in ipairs(self.catalog.recipes[req.scene] or {}) do
      local w = { kind = line[2], count = line[1], adjectives = {}, role = line[3], scene = req.scene }
      local asked = false
      for _, x in ipairs(want) do if x.kind == w.kind and not x.role then asked = true end end
      if self.kinds[w.kind] and not (excluded[w.kind] and not asked) then
        if not more then
          for _, x in ipairs(want) do
            if x.kind == w.kind and not x.role and not x.merged then
              w.count, w.adjectives, w.word, w.plural = x.count, x.adjectives, x.word, x.plural
              x.merged = true
              break
            end
          end
        end
        lines[#lines + 1] = w
      end
    end
    local kept = {}
    for _, x in ipairs(want) do if not x.merged then kept[#kept + 1] = x end end
    for _, w in ipairs(lines) do kept[#kept + 1] = w end
    want = kept
  end
  return want
end

-- ---------------------------------------------------------------------------------------------- one group, somewhere sensible
--- The corners of a room, for things asked to stand in them, as builders (nearest the view first / diagonal first).
local function corner_builders(ctx, insts, rr)
  local its = to_items(insts)
  local order = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } }
  local lx, ly = fwd(ctx.yaw)
  local tx, ty = ctx.pos.x + lx * 260, ctx.pos.y + ly * 260
  table.sort(order, function(a, b)
    local function dist(c)
      local pc = to_world(rr.frame, c[1] * rr.d / 2, c[2] * rr.w / 2, 0)
      return (pc.x - tx) ^ 2 + (pc.y - ty) ^ 2
    end
    return dist(a) < dist(b)
  end)
  local diagonal = { order[1], { -order[1][1], -order[1][2] } }
  for _, c in ipairs(order) do
    local seen = false
    for _, d in ipairs(diagonal) do if d[1] == c[1] and d[2] == c[2] then seen = true end end
    if not seen then diagonal[#diagonal + 1] = c end
  end
  local function mk(corners, k)
    local b = { desc = (#insts > 1) and "in the corners" or "in the corner", radius = 0 }
    b.build = function()
      local out = {}
      -- each corner keeps a running length along its wall, and starts a second layer when the wall is used up
      local used, layer, layer_d = { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }
      for i, it in ipairs(its) do
        local slot = (i - 1 + k) % 4 + 1
        local c = corners[slot]
        if used[slot] > 0 and used[slot] + it.w > rr.w / 2 - 4 then
          layer[slot] = layer[slot] + layer_d[slot] + 2
          used[slot], layer_d[slot] = 0, 0
        end
        local u = c[1] * (rr.d / 2 - it.d / 2 - 1 - layer[slot])
        local v = c[2] * (rr.w / 2 - it.w / 2 - 1 - used[slot])
        used[slot] = used[slot] + it.w + 2
        layer_d[slot] = max(layer_d[slot], it.d)
        out[#out + 1] = P(it.inst, to_world(rr.frame, u, v, c[1] < 0 and 0 or 180), "corner")
      end
      return out
    end
    return b
  end
  -- nearest corner first; when a corner is taken (by a plant, say) the next one is tried, so a second "in the corner" finds a free one
  local out = {}
  for k = 0, 3 do
    out[#out + 1] = mk(order, k)
    if #insts > 1 then out[#out + 1] = mk(diagonal, k) end
  end
  return out
end

local FALLBACK = {
  row = { { "grid", 2 }, { "grid" } }, circle = { { "grid" } }, u_shape = { { "two_rows" }, { "row" }, { "grid" } },
  square = { { "two_rows" }, { "row" }, { "grid" } }, two_rows = { { "row" }, { "grid" } }, facing_each_other = { { "row" } },
  l_shape = { { "row" }, { "grid" } }, stacked = { { "row" }, { "grid" } }, scattered = { { "cluster" }, { "row" } },
  cluster = { { "row" }, { "grid" } }, grid = { { "row" } }, single = { { "row" } },
}

--- Lay `insts` out as arrangement `name` somewhere sensible: against a wall for what likes one, in a corner when asked, else in
--- the open near the player, and in a simpler shape when the one asked for does not fit. Returns builder, placements, frame,
--- score (placements nil when nothing fits). o: wall_pref (they were asked to go against a wall), ignore (ids standing there
--- that do not count as in the way), strict (a move to a wall or corner: stay put rather than stand in the open),
--- frames_first (frames to try before the usual ones), theta_for.
function Plan:place_group(ctx, insts, name, rng, res, o)
  o = o or {}
  local ignore = o.ignore
  local function pre(frames)
    if not o.frames_first then return frames end
    local all = {}
    for _, f in ipairs(o.frames_first) do all[#all + 1] = f end
    for _, f in ipairs(frames) do all[#all + 1] = f end
    return all
  end
  local theta_for = o.theta_for or ((#insts == 1) and function(cx, cy) return Layout.angle_of(ctx.pos.x - cx, ctx.pos.y - cy) end
                                                    or function() return player_frame_theta(ctx) end)

  local corner_reason
  local function attempt(nm, wall_flag, cols)
    if nm == "corner" then
      local rr = real_room(ctx, WALL_FAR)
      if not rr then corner_reason = "none"; return nil end
      local builders = corner_builders(ctx, insts, rr)
      local pls, frame, bb, score = fit(ctx, builders, { rr.frame }, ignore, 0.6)
      if not pls then corner_reason = "full" end
      return bb, pls, frame, score
    end
    local bb = self:arrangement(nm, insts, rng, ctx, { wall = wall_flag or nil, cols = cols })
    local pls, frame, score, _won
    -- the same frames with the group turned a quarter: a long thing (a meeting table) fits across a room it cannot fit along
    local function turned(frames)
      local out = {}
      for i, f in ipairs(frames) do out[i] = { x = f.x, y = f.y, theta = Layout.wrap(f.theta + 90), open = f.open } end
      return out
    end
    if (bb.wall or wall_flag) and (nm == "row" or nm == "single") then
      local depth = 0
      for _, i in ipairs(insts) do depth = max(depth, i.d) end
      local wf = wall_frames(ctx, depth, wall_flag and WALL_FAR or WALL_NEAR)
      if #wf > 0 then
        pls, frame, _won, score = fit(ctx, { bb }, wf, ignore, 0.35)
        if pls and score and score > 0 then pls = nil end
      elseif wall_flag and not o.strict then
        self:effect_note(res, "there is no wall in reach, so I stood them in the open")
      end
      if not pls and not (o.strict and wall_flag) then
        local frames = pre(open_frames(ctx, bb.radius, theta_for))
        pls, frame, _won, score = fit(ctx, { bb }, frames, ignore, 0.6)
        if not pls then pls, frame, _won, score = fit(ctx, { bb }, turned(frames), ignore, 0.7) end
      end
    else
      local frames = pre(open_frames(ctx, bb.radius, theta_for))
      pls, frame, _won, score = fit(ctx, { bb }, frames, ignore, 0.6)
      if not pls then pls, frame, _won, score = fit(ctx, { bb }, turned(frames), ignore, 0.7) end
    end
    return bb, pls, frame, score
  end

  local b, pls, frame, score
  if name == "corner" then
    b, pls, frame, score = attempt("corner")
    if not pls then
      if o.strict then return nil end
      self:effect_note(res, corner_reason == "full" and "it would not fit in the corners, so I grouped them in the open"
                                              or "there is no corner in reach, so I grouped them in the open")
      name = "cluster"
    end
  end
  if not pls then b, pls, frame, score = attempt(name, o.wall_pref, o.cols) end
  if not pls and not (o.strict and (o.wall_pref or name == "corner")) then
    for _, fb in ipairs(FALLBACK[name] or { { "row" } }) do
      local b2, p2, f2, s2 = attempt(fb[1], false, fb[2])
      if p2 then
        b, pls, frame, score = b2, p2, f2, s2
        if ctx.cut_short then
          -- a stage's slice of the time ran out, so "it would not fit" is not known: say that time ran out instead
          ctx._cut_fallback = true
        else
          self:effect_note(res, (o.asked == false and "it would not fit in one place, so I laid it out " or "it would not fit as asked, so I laid it out ")
                                .. (b2.desc ~= "" and b2.desc or "in a row"))
        end
        break
      end
    end
  end
  return b, pls, frame, score
end

-- ---------------------------------------------------------------------------------------------- adding
local LAYOUT_NAMES = { row = true, circle = true, grid = true, u_shape = true, l_shape = true, square = true, facing_each_other = true,
  two_rows = true, stacked = true, scattered = true, cluster = true, single = true }
local REL_ARR = { around = true, beside = true, in_front_of = true, behind = true, on = true, facing_me = true, flank = true }
local NOT_A_THING = { [""] = true, me = true, here = true, wall = true, corner = true, window = true, door = true, ceiling = true,
  floor = true, room = true }
local KEEP_R = (KEEP_OUT / 2 + KEEP_OUT_PAD) * 1.4143 + 2          -- clear of the corner of the keep-out square, from the player

--- The columns a grid is asked to have ("in 3 rows", "in 2 rows of 3"), or nil for the usual shape.
function Plan.grid_cols(req, n)
  local c, r = tonumber(req.cols), tonumber(req.rows)
  if c and c >= 1 then return min(floor(c), max(1, n)) end
  if r and r >= 1 then return max(1, ceil(n / min(floor(r), max(1, n)))) end
  return nil
end

--- One variant for the whole pile.
local function uniformize(insts)
  local first = insts[1]
  for _, i in ipairs(insts) do if i.kind ~= first.kind then return end end
  for _, i in ipairs(insts) do i.vi, i.v, i.w, i.d, i.h = first.vi, first.v, first.w, first.d, first.h end
end

--- Where a bearing from the player lies, in words ("to your left").
local function side_words(ctx, bearing)
  local a = Layout.angle_diff(bearing, ctx.yaw)
  if abs(a) <= 45 then return "in front of you" elseif abs(a) > 135 then return "behind you" end
  return (a > 0) and "to your left" or "to your right"
end

function Plan:add(req, ctx)
  local res = { action = "add", placements = {}, notes = {}, groups = {} }
  local rng = Layout.rng(ctx.seed or Layout.hash(req.raw or ""))
  local want = self:wanted(req, res, ctx)
  -- "a living room and a kitchen": one room at a time is built; say which one was left for the next request (the parser lists
  -- the others in req.extra_scenes; only names of the library's own scenes are spoken)
  if type(req.extra_scenes) == "table" and req.scene and req.scene ~= "" then
    for _, nm in ipairs(req.extra_scenes) do
      if type(nm) == "string" and nm ~= req.scene and self.catalog.recipes[nm] then
        add_note(res, "I built the " .. req.scene .. " only, so ask for the " .. nm .. " next")
        break
      end
    end
  end
  if #want == 0 then
    res.say = ((#res.notes > 0) and (res.notes[1] .. ", so I added nothing") or T.nothing_to_add) .. "."
    res.fixed = true
    return res
  end

  -- "a plant in each corner": one set of the things in every corner
  if req.every == "corner" then
    local limit = ctx.max_per_kind or MAX_PER_KIND
    for _, w in ipairs(want) do
      -- the parser says whether the count is per corner ("each", "every") or the total ("all four corners"); without that
      -- word, a multiple of four is read as the total, as before
      local mode = req.every_mode
      local total = (mode == "total") or (mode ~= "per" and w.count >= 4 and w.count % 4 == 0)
      if not total then w.count = min(limit, w.count * 4) end
    end
  end
  -- what one request may ask for in all
  local max_total = ctx.max_total or MAX_TOTAL
  local total = 0
  for _, w in ipairs(want) do total = total + w.count end
  if total > max_total then
    local f = max_total / total
    for _, w in ipairs(want) do w.count = max(1, floor(w.count * f)) end
    total = 0
    for _, w in ipairs(want) do total = total + w.count end
    while total > max_total do
      local big
      for _, w in ipairs(want) do if w.count > 1 and (not big or w.count > big.count) then big = w end end
      if not big then break end
      big.count, total = big.count - 1, total - 1
    end
    add_note(res, "that was a lot, so I scaled it down to about " .. max_total .. " things")
    res.clamped = true
  end

  -- the relation (if any) decides the mode
  local arr, target = req.arrangement or "unspecified", req.target or ""
  if arr == "facing_away" then
    -- "facing away from X / outward": arranged as it would be facing, then every one turned about
    res.away = true
    if target ~= "" and target ~= "me" and target ~= "here" then arr = "facing_me"
    elseif target == "me" or target == "here" then arr = "facing_me"; target = "me"
    else arr, target = "unspecified", "" end
  end
  local kinds_present, order = {}, {}
  for _, w in ipairs(want) do
    if not kinds_present[w.kind] then kinds_present[w.kind] = true; order[#order + 1] = w.kind end
  end
  -- a second relation the parser kept: "six chairs in a circle facing away from the table" is the circle, every one turned about;
  -- "a row of chairs with their backs to the wall" is a row against the wall
  local fa = req.facing_away
  local away_from                                   -- the standing thing the group turns its back on ("facing away from the tv")
  if type(fa) == "string" and arr ~= "facing_away" then
    if fa == "wall" then
      if arr == "row" or arr == "single" or arr == "unspecified" then arr, target = "against_wall", "" end
    elseif self.kinds[fa] and not kinds_present[fa] and #order == 1 and not REL_ARR[arr] and arr ~= "corner" and arr ~= "against_wall"
           and (req.facing == nil or req.facing == "") then
      -- a NAMED thing, not the player: the group stands in front of the one that stands near, and every one looks away from it;
      -- with none standing nothing is made up for it to turn from: they turn away from the player, and the reply says so
      res.away = true
      local stands = false
      for _, list in ipairs({ ctx.mine or {}, ctx.registry or {} }) do
        local near = registry_matches(self, ctx, fa, {}, aim_or_front(ctx), list)
        if near[1] and (near[1].x - ctx.pos.x) ^ 2 + (near[1].y - ctx.pos.y) ^ 2 < 700 ^ 2 then stands = true; break end
      end
      if stands then away_from = fa
      else add_note(res, "there was no " .. fa .. " to turn away from") end
    elseif #order == 1 and not res.away then
      res.away = true
    end
  end

  -- instances
  local uniform = (arr == "stacked")
  local insts_by_want = {}
  for wi, w in ipairs(want) do
    local vis = self:pick_variants(w.kind, w.count, w.adjectives, rng, uniform)
    local list = {}
    for i = 1, w.count do list[i] = make_inst(self, w.kind, vis[i], w.adjectives) end
    insts_by_want[wi] = list
  end
  local function all_insts()
    local t = {}
    for _, l in ipairs(insts_by_want) do for _, i in ipairs(l) do t[#t + 1] = i end end
    return t
  end
  local insts = all_insts()

  local function wide_radius(bs)
    local r = 0
    for _, b in ipairs(bs) do r = max(r, b.radius or 0) end
    return r
  end
  local function theta_square() return player_frame_theta(ctx) end
  local function theta_for_player(cx, cy) return Layout.angle_of(ctx.pos.x - cx, ctx.pos.y - cy) end

  -- how the work ends: the sentence, the notes, and what ran out of time
  local conclude, refuse
  function conclude(pls, desc, score, o)
    o = o or {}
    -- a referent standing alone is not an answer: nothing that was asked for found a place
    local anything = false
    for _, p in ipairs(pls) do if p.role ~= "referent" then anything = true end end
    if not anything then return refuse(T.no_floor) end
    finish(self, ctx, pls, res, 1)
    local shown = {}
    for _, p in ipairs(pls) do
      if not (p.role == "referent" and o.hide_referent) then shown[#shown + 1] = p end
    end
    local what = o.prefix or ""
    if res.away then
      -- every one was turned about after the layout was made
      local d = desc or ""
      local changed = false
      d = (d:gsub(", facing the middle, facing the ", ", facing the "))                      -- (a circle in front of a thing: only the thing matters)
      if d:find("facing the middle", 1, true) then d = (d:gsub("facing the middle", "facing outward")); changed = true end
      if d:find("facing you", 1, true) then d = (d:gsub("facing you", "facing away from you")); changed = true end
      local named, n = d:gsub("facing the ", "facing away from the ")                   -- "facing the tv": the thing they turned their backs on
      if n > 0 then d, changed = named, true end
      if not changed and not d:find("away", 1, true) then d = (d ~= "" and (d .. ", ") or "") .. "facing away" end
      desc = d
    end
    local text = describe(shown, 8)
    if o.comma then res.say = what .. text .. ", " .. desc .. "."
    else res.say = what .. text .. ((desc and desc ~= "") and (" " .. desc) or "") .. "." end
    local lost = note_missing(res, o.insts or insts, pls)
    if score and score > 0 then add_note(res, "it was a tight squeeze") end
    if ctx.cut_short and ((score or 0) > 0 or lost > 0 or ctx._cut_fallback) then
      res.timed_out = true
      add_note(res, T.out_of_time)
    end
    res.say = with_notes(res.say, res.notes)
    return res
  end
  function refuse(text)
    if ctx.cut_short then
      res.timed_out = true
      text = T.too_much
    end
    -- nothing was built, so a note about what was built ("I added a table", "I put them on...") would be false
    local effects = self._effects and self._effects[res]
    if effects then
      local kept = {}
      for _, n in ipairs(res.notes) do if not effects[n] then kept[#kept + 1] = n end end
      res.notes = kept
    end
    res.say = with_notes(text .. ".", res.notes)
    res.fixed = (#res.notes == 0) or nil
    return res
  end

  -- -- "it": the last thing built is the thing to put it next to
  local pinned
  if target == "@last" and (REL_ARR[arr]) then
    pinned = group_anchor(last_entries(self, ctx))
    if pinned then target = pinned.kind
    else
      add_note(res, T.nothing_recent)
      target, arr = "", "unspecified"
    end
  end
  -- -- "in a row facing the tv": the group stands in front of that and looks at it
  local shape_name
  local face_kind = req.facing or away_from
  if face_kind == "@last" and not REL_ARR[arr] then
    pinned = group_anchor(last_entries(self, ctx))
    if pinned then face_kind = pinned.kind
    else
      add_note(res, T.nothing_recent)
      face_kind = nil
    end
  end
  if face_kind and face_kind ~= "" and not NOT_A_THING[face_kind] and not REL_ARR[arr] and arr ~= "against_wall" and arr ~= "corner" then
    if self.kinds[face_kind] then
      shape_name = LAYOUT_NAMES[arr] and arr or nil
      arr, target = "facing_group", face_kind
    else
      note_unknown(res, face_kind)
      add_note(res, T.library)
    end
  end

  -- "five chairs in a grid in front of the tv": the shape the parser kept, standing in front of that and looking at it
  if arr == "in_front_of" and not shape_name and LAYOUT_NAMES[req.shape or ""] and req.shape ~= "row" and req.shape ~= "single"
     and target ~= "" and not NOT_A_THING[target] and (self.kinds[target] or pinned) then
    shape_name, arr = req.shape, "facing_group"
  end

  -- a mode for relative-to-something requests
  local relational = (REL_ARR[arr] or arr == "facing_group") and not NOT_A_THING[target]
  local on_me = (target == "me" or target == "here")

  if relational and not pinned and kinds_present[target] and #order == 1 and arr ~= "facing_group" then
    -- "add a table around a table": nonsense; treat as no target
    relational = false
    arr = "unspecified"
  end

  --- A standing thing near the requester to put things by (the requester's own first), or nil.
  local function auto_referent(kind_id)
    local from = aim_or_front(ctx)
    local function nearest(list)
      local near = registry_matches(self, ctx, kind_id, {}, from, list)
      local e = near[1]
      if e and ((e.x - ctx.pos.x) ^ 2 + (e.y - ctx.pos.y) ^ 2) < 700 ^ 2 then return e end
    end
    local e = (ctx.mine and nearest(ctx.mine)) or nearest(ctx.registry)
    if not e then return nil end
    local ref = ref_of(e)
    ref.inst = nil
    ref.topf = top_frame(self.kinds[e.kind].variants[e.vi])
    ref.exists = true
    return ref
  end

  if relational then
    -- items that are not the target itself
    local rel_insts, target_insts = {}, {}
    for _, i in ipairs(insts) do
      if i.kind == target and not pinned and arr ~= "facing_group" then target_insts[#target_insts + 1] = i else rel_insts[#rel_insts + 1] = i end
    end
    local ref
    if pinned then
      ref = ref_of(pinned)
      ref.inst = nil
      ref.topf = top_frame(self.kinds[pinned.kind].variants[pinned.vi])
      ref.exists = true
    elseif #target_insts == 0 then
      ref = auto_referent(target)
    end
    local created
    local from_request = false
    if not ref and #target_insts > 0 then
      -- the target is in the request: it is made, as the main thing, and the rest of its kind joins the group
      created = target_insts[1]
      from_request = true
      for k = 2, #target_insts do rel_insts[#rel_insts + 1] = target_insts[k] end
    elseif not ref then
      -- the target is not in the request, and nothing of the kind stands near: make one
      local kind_id = target
      local k = self.kinds[kind_id]
      if not k then
        note_unknown(res, kind_id)
        add_note(res, T.library)
        add_note(res, "so I used the middle of the room")
        relational = false
      elseif #rel_insts > 0 then
        local pk, pv
        if arr == "around" and k.category == "table" then
          pk, pv = self:table_for_seats(kind_id, rel_insts[1], #rel_insts)
        elseif arr == "on" and k.surface then
          local sk, sv = self:surface_for(rel_insts[1].kind, rel_insts, kind_id)
          pk, pv = sk, sv
        end
        if not pk then pk, pv = kind_id, self:pick_variants(kind_id, 1, {}, rng)[1] end
        created = make_inst(self, pk, pv, {})
        self:effect_note(res, "there was no " .. pk .. ", so I added one")
      else
        relational = false
      end
    end
    if relational and created and from_request and #rel_insts > 0 then
      -- the table was asked for by name: still the smallest one that holds what goes round or on it, unless its size was chosen
      local chosen = false
      for _, a in ipairs(created.adjectives or {}) do if SIZE_UP[a] or SIZE_DOWN[a] or STYLE[a] then chosen = true end end
      if not chosen and arr == "around" and created.k.category == "table" then
        local lost = seating_of(created.kind, created.v, rel_insts[1], #rel_insts)
        if lost > 0 then
          local pk, pv = self:table_for_seats(created.kind, rel_insts[1], #rel_insts)
          if pk and (pk ~= created.kind or pv ~= created.vi) then created = make_inst(self, pk, pv, created.adjectives) end
        end
      elseif not chosen and arr == "on" and created.k.surface then
        local sk, sv = self:surface_for(rel_insts[1].kind, rel_insts, created.kind)
        if sk and sv ~= created.vi then created = make_inst(self, sk, sv, created.adjectives) end
      end
    end
    if relational and not ref and created then
      ref = { kind = created.kind, inst = created, w = created.w, d = created.d, h = created.h, theta = 0, x = 0, y = 0,
              topf = top_frame(created.v), created = true }
    end
    if relational and #rel_insts == 0 then relational = false end

    if relational then
      local rel = arr
      if arr == "facing_me" or arr == "facing_group" then rel = "in_front_of" end
      if arr == "flank" then rel = "beside" end
      if arr == "facing_group" then rel = "facing_group" end
      -- turned away from a NAMED thing (not from the player): each one looks along the line from that thing to itself
      if res.away and (rel == "in_front_of" or rel == "facing_group") then ctx._away_pt = ref.exists and { x = ref.x, y = ref.y } or "referent" end
      if arr == "flank" and #rel_insts == 1 then
        -- one on each side
        local only = rel_insts[1]
        local twin = make_inst(self, only.kind, only.vi, only.adjectives)
        rel_insts[2] = twin
        insts[#insts + 1] = twin
      end
      local grp
      if rel == "facing_group" then
        local nm = shape_name or self:default_arrangement(rel_insts)
        if nm == "single" then nm = "row" end
        if nm == "circle" then self:effect_note(res, "a circle looks at its own middle, so I stood it in front of the " .. target) end
        grp = self:arrangement(nm, rel_insts, rng, ctx, {})
      end
      local no_top = (rel == "on" and not ref.topf)
      -- nothing of this size fits on that top at all: beside it, and said
      local no_fit = false
      if rel == "on" and ref.topf and top_holds(ref.topf, rel_insts) == 0 then
        rel, no_fit = "beside", true
      end
      local function inner_for(r, o2)
        if rel == "around" then
          if o2.ring then return ring_builder(r, rel_insts, { stop = stopper(ctx), arc = o2.arc }) end
          return around_builder(r, rel_insts, o2)
        elseif rel == "on" and r.topf then return on_builder(r, rel_insts, o2)
        elseif rel == "on" then return offset_builder(r, rel_insts, "beside", o2)
        elseif rel == "facing_group" then
          local probe = grp.build({ x = 0, y = 0, theta = 0 })
          local _, x1 = bbox_extent(probe)                     -- theta 0: the group looks along +x
          local b = { desc = grp.desc, dropped = 0 }
          b.build = function()
            local fx, fy = fwd(r.theta)
            local reach = r.d / 2 + (o2.distance or 30) + x1
            return grp.build({ x = r.x + fx * reach, y = r.y + fy * reach, theta = Layout.wrap(r.theta + 180) })
          end
          return b
        end
        local dist = o2.distance
        if rel == "in_front_of" and r.kind == "tv" then dist = max(dist or 0, 70, 1.6 * r.w) end
        return offset_builder(r, rel_insts, rel, { flip = o2.flip, distance = dist, align = o2.align })
      end
      -- the builders work in the reference's own frame; one per way of standing, recomputed per frame
      local function mk(o2)
        local b = { radius = 0 }
        local probe_ref = { x = 0, y = 0, theta = 0, w = ref.w, d = ref.d, kind = ref.kind, inst = ref.inst, topf = ref.topf, h = ref.h, id = ref.id }
        local inner = inner_for(probe_ref, o2)
        b.desc, b.dropped = inner.desc, inner.dropped or 0
        b.build = function(frame)
          local out = {}
          local r = {}
          for k2, v2 in pairs(ref) do r[k2] = v2 end
          if ref.created then r.x, r.y, r.theta = frame.x, frame.y, frame.theta end
          local ib = inner_for(r, o2)
          if ref.created then
            local rp = P(ref.inst, { x = r.x, y = r.y, theta = r.theta, z = 0 }, "referent")
            out[#out + 1] = rp
            r.placement = rp
          end
          for _, p in ipairs(ib.build(frame)) do
            if p.tuck == nil and rel == "around" and not o2.ring then p.tuck = out[1] end
            out[#out + 1] = p
          end
          return out
        end
        local probe = b.build({ x = 0, y = 0, theta = 0 })
        b.radius = radius_of(probe)
        local back = 0                                          -- how far the group reaches behind the referent's centre
        for _, p in ipairs(probe) do
          for _, c in ipairs(Layout.corners(rect(p.x, p.y, p.w, p.d, p.theta))) do back = max(back, -c[1]) end
        end
        b.back = back
        return b
      end
      local builders = {}
      local first = mk({})
      -- a ring takes more than the sides do; "in a circle around the table" asks for one. A thing against a wall has no back to
      -- stand behind: only an arc in front of it fits, and that is tried before the sides.
      local arcs = {}
      if rel == "around" and #rel_insts > 1 then
        for _, a in ipairs({ 300, 240, 180, 140 }) do
          local ab = mk({ ring = true, arc = a })
          ab.radius = 0                                  -- (the sweep of frames is laid out for the usual builders, not for these)
          arcs[#arcs + 1] = ab
        end
      end
      local ring_wanted = rel == "around" and (first.dropped > 0 or req.shape == "circle")
      if ring_wanted then builders[#builders + 1] = mk({ ring = true }) end
      if ring_wanted and req.shape == "circle" then for _, a in ipairs(arcs) do builders[#builders + 1] = a end end
      builders[#builders + 1] = first
      if not (ring_wanted and req.shape == "circle") then for _, a in ipairs(arcs) do builders[#builders + 1] = a end end
      if rel == "beside" then
        builders[#builders + 1] = mk({ flip = true })
        -- a deeper thing beside a wall-flush one: backs aligned, after the centred poses (which stay first wherever they fit)
        builders[#builders + 1] = mk({ align = true })
        builders[#builders + 1] = mk({ align = true, flip = true })
      end
      if rel == "in_front_of" or rel == "facing_group" then
        for _, d in ipairs({ 30, 60, 100, 160 }) do builders[#builders + 1] = mk({ distance = d }) end
      end
      local group_desc = first.desc
      if rel == "facing_group" then group_desc = (grp.desc ~= "" and (grp.desc .. ", ") or "") .. "facing the " .. target end
      local rel_phrase = ({ around = "around", on = "on", beside = "beside", in_front_of = "in front of", behind = "behind",
                            facing_group = "in front of" })[rel] or "near"
      local o = { hide_referent = ref.created and not from_request, insts = rel_insts }
      if ref.exists then
        -- an existing referent fixes the frame: try its own pose, not the player's view
        local pls, _, b, score = fit(ctx, builders, { { x = ref.x, y = ref.y, theta = ref.theta } })
        if pls then
          if no_top then self:effect_note(res, "I can't stand things on the " .. ref.kind .. ", so I put them beside it") end
          if no_fit then self:effect_note(res, (#rel_insts > 1 and "they do not" or "it does not") .. " fit on the " .. ref.kind .. ", so I put " .. (#rel_insts > 1 and "them" or "it") .. " beside it") end
          return conclude(pls, rel == "facing_group" and group_desc or (b.desc or group_desc or ""), score, o)
        end
        -- no clean room against it: the same things, free-standing nearby (never on top of what stands)
        self:effect_note(res, "there was no room " .. rel_phrase .. " the " .. ref.kind .. ", so I put them nearby")
        relational = false
        arr, target = "unspecified", ""
        if rel == "facing_group" then arr = shape_name or "unspecified" end
      else
        local frames = open_frames(ctx, wide_radius(builders), theta_square)
        if ref.kind and self.kinds[ref.kind] and self.kinds[ref.kind].wall == "prefer" and ref.created then
          local back = 0
          for _, b in ipairs(builders) do back = max(back, b.back or 0) end
          local wallf = wall_frames(ctx, back * 2, WALL_NEAR)
          if #wallf > 0 then
            for _, f in ipairs(frames) do wallf[#wallf + 1] = f end
            frames = wallf
          end
        end
        local pls, _, b, score = fit(ctx, builders, frames)
        if pls then
          if no_top then self:effect_note(res, "I can't stand things on the " .. ref.kind .. ", so I put them beside it") end
          if no_fit then self:effect_note(res, (#rel_insts > 1 and "they do not" or "it does not") .. " fit on the " .. ref.kind .. ", so I put " .. (#rel_insts > 1 and "them" or "it") .. " beside it") end
          return conclude(pls, rel == "facing_group" and group_desc or (b.desc or group_desc or ""), score, o)
        end
        return refuse(T.no_floor)
      end
    end
  end

  -- -- relative to the player
  if on_me and (arr == "around" or arr == "in_front_of" or arr == "behind" or arr == "beside" or arr == "facing_me" or arr == "unspecified") then
    local pos = ctx.pos
    local b, pls, bb, score, frame
    if arr == "around" then
      local items = to_items(insts)
      local min_r0, sum = 46, 0
      for _, it in ipairs(items) do min_r0 = max(min_r0, it.d / 2 + KEEP_R); sum = sum + it.w + 8 end
      b = { desc = "around you", radius = 0 }
      b.build = function(f)
        local poses = Layout.circle(pos.x, pos.y, items, { gap = 8, min_r = min_r0 + (f.grow or 0), start = ctx.yaw + (f.rot or 0),
                                                            arc = f.arc, stop = stopper(ctx) })
        local out = {}
        for i, it in ipairs(items) do out[#out + 1] = P(it.inst, poses[i], "around_me") end
        return out
      end
      -- a full ring first, then wider ones, turned about the player, then a part of a ring (a wall or a prop is in the way)
      local frames = { { x = pos.x, y = pos.y, theta = ctx.yaw } }
      for g = 0, 240, 6 do
        for _, rr in ipairs({ 0, 180, 90, 270, 45, 225, 135, 315 }) do
          if g > 0 or rr > 0 then frames[#frames + 1] = { x = pos.x, y = pos.y, theta = ctx.yaw, grow = g, rot = rr } end
        end
      end
      if #items > 1 then
        local r_full = max(min_r0, sum / (2 * math.pi))
        for _, arc in ipairs({ 300, 240, 180, 140, 100, 60 }) do
          if sum / rad(arc) <= 1.8 * r_full + 40 then
            for _, k in ipairs({ 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12 }) do
              frames[#frames + 1] = { x = pos.x, y = pos.y, theta = ctx.yaw, arc = arc, rot = k * 15 }
            end
          end
        end
      end
      pls, frame, bb, score = fit(ctx, { b }, frames, nil, 0.8)
      if pls and frame.arc then self:effect_note(res, "something was in the way, so they only go part of the way round") end
      if not pls then
        -- no ring will do: the same things as a group nearby
        local nb, p2, _, s2 = self:place_group(ctx, insts, (#insts > 1) and "circle" or "single", rng, res, {})
        if p2 then
          self:effect_note(res, "there was no room around you, so I put them nearby")
          return conclude(p2, nb.desc, s2, {})
        end
      end
    else
      local base_arr = self:default_arrangement(insts)
      if base_arr == "single" then base_arr = "row" end
      b = self:arrangement(base_arr, insts, rng, ctx, {})
      local heading = (arr == "behind") and (ctx.yaw + 180) or ctx.yaw
      local r = b.radius or 40
      local fx, fy = fwd(heading)
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
      pls, frame, bb, score = fit(ctx, { b }, frames2, nil, 0.5)
      local moved_words
      if not pls then
        -- nothing on that side: every bearing round the player, the wanted side first, from close in
        local bearings = {}
        for k = 0, 23 do
          local ang = Layout.wrap(ctx.yaw + k * 15)
          local want_a = (arr == "beside") and min(abs(Layout.angle_diff(ang, ctx.yaw + 90)), abs(Layout.angle_diff(ang, ctx.yaw - 90)))
                                             or abs(Layout.angle_diff(ang, heading))
          bearings[#bearings + 1] = { a = ang, w = want_a }
        end
        table.sort(bearings, function(p, q) if p.w ~= q.w then return p.w < q.w end return p.a < q.a end)
        local frames3 = {}
        for _, bg in ipairs(bearings) do
          local dx, dy = fwd(bg.a)
          for dist = 42, r + 172, 10 do
            local cx, cy = pos.x + dx * dist, pos.y + dy * dist
            frames3[#frames3 + 1] = { x = cx, y = cy, theta = Layout.angle_of(pos.x - cx, pos.y - cy), bearing = bg.a, off = bg.w }
          end
        end
        pls, frame, bb, score = fit(ctx, { b }, frames3, nil, 1)
        if pls and frame.off and frame.off > 50 then
          local want_words = ({ in_front_of = "in front of you", behind = "behind you", beside = "beside you" })[arr] or "in front of you"
          self:effect_note(res, "there was no room " .. want_words .. ", so I put " .. (#insts > 1 and "them " or "it ") .. side_words(ctx, frame.bearing))
        end
      end
      if not pls then
        -- no bearing round the player will do: the same things wherever they fit (place_group falls back from a row to a grid)
        local nb, p2, _, s2 = self:place_group(ctx, insts, base_arr, rng, res, { asked = false })
        if p2 then
          local want_words = ({ in_front_of = "in front of you", behind = "behind you", beside = "beside you" })[arr] or "in front of you"
          self:effect_note(res, "there was no room " .. want_words .. ", so I put " .. (#insts > 1 and "them" or "it") .. " nearby")
          return conclude(p2, nb.desc, s2, {})
        end
      end
      if pls then
        local desc = ({ in_front_of = "in front of you", behind = "behind you", beside = "beside you" })[arr] or "facing you"
        if frame.off and frame.off > 50 then desc = b.desc end
        return conclude(pls, desc, score, {})
      end
    end
    if pls then return conclude(pls, b.desc, score, {}) end
    return refuse(T.no_floor)
  end

  -- -- the surface a small thing needs
  local function needs_surface(kind_id)
    return SURFACE_FOR[kind_id] ~= nil and kind_id ~= "microwave" and self:surface_for(kind_id, {}) ~= nil
  end
  local function has_surface_kind()
    for _, w in ipairs(want) do
      if self.kinds[w.kind].surface then return true end
    end
    return false
  end
  if arr == "unspecified" and not (req.scene and req.scene ~= "") and not has_surface_kind() then
    local only_small = true
    for _, w in ipairs(want) do if not needs_surface(w.kind) then only_small = false end end
    if only_small then
      -- put the things on surfaces that I add under them: as few, and as small, as hold them all
      local sk, sv, cap = self:surface_for(want[1].kind, insts)
      local need = (cap and cap >= 1) and ceil(#insts / cap) or 1
      need = min(need, 4)
      local room_for = (cap or 0) * need
      if room_for < #insts then
        -- not even four of the biggest hold them all: what fits is what is made
        local keep, left = max(1, room_for), #insts
        for wi = #want, 1, -1 do
          local list = insts_by_want[wi]
          while left > keep and #list > 0 do table.remove(list); left = left - 1 end
          want[wi].count = #list
        end
        for wi = #want, 1, -1 do
          if #insts_by_want[wi] == 0 then table.remove(want, wi); table.remove(insts_by_want, wi) end
        end
        insts = all_insts()
        self:effect_note(res, "only " .. #insts .. " fit on " .. (need == 1 and ("the " .. sk) or (count_word(need) .. " " .. plural(sk, need))))
      end
      local surfs = {}
      for i = 1, need do surfs[i] = make_inst(self, sk, sv, {}) end
      table.insert(want, 1, { kind = sk, count = need, adjectives = {} })
      table.insert(insts_by_want, 1, surfs)
      insts = all_insts()
      kinds_present[sk] = true
      order[#order + 1] = sk
      self:effect_note(res, "I put " .. (#insts > need + 1 and "them" or "it") .. " on " .. (need == 1 and article(sk) or (count_word(need) .. " " .. plural(sk, need))) .. " I added")
    end
  end

  -- -- an explicit arrangement on everything
  local explicit_arr = arr ~= "unspecified" and arr ~= "facing_me" and arr ~= "around" and arr ~= "beside" and arr ~= "in_front_of"
                       and arr ~= "behind" and arr ~= "on" and arr ~= "stacked_on" and arr ~= "flank" and arr ~= "facing_group"
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

    -- the table that seats the chairs asked for (the smallest that does, unless its size was chosen)
    local function size_tables(ents)
      local dining, seats = {}, {}
      for i, e in ipairs(ents) do
        if e.role == "center" and e.inst.kind ~= "coffee table" then dining[#dining + 1] = i end
        if e.role == "around" then seats[#seats + 1] = e end
      end
      if #dining == 0 or #seats == 0 then return ents, false end
      local out = {}
      for i, e in ipairs(ents) do out[i] = e end
      local changed = false
      for di, idx in ipairs(dining) do
        local e = out[idx]
        local share = 0
        for si = 1, #seats do if (si - 1) % #dining + 1 == di then share = share + 1 end end
        local chosen = false
        for _, a in ipairs(e.inst.adjectives or {}) do if SIZE_UP[a] or SIZE_DOWN[a] or STYLE[a] then chosen = true end end
        local lost, tight = 0, false
        if not chosen and share > 0 then lost, tight = seating_of(e.inst.kind, e.inst.v, seats[1].inst, share) end
        if not chosen and share > 0 and (lost > 0 or tight) then      -- a table where seats are lost or touch is swapped for one that seats them
          local pk, pv = self:table_for_seats(e.inst.kind, seats[1].inst, share)
          if pk and (pk ~= e.inst.kind or pv ~= e.inst.vi) then
            out[idx] = { inst = make_inst(self, pk, pv, e.inst.adjectives), role = e.role, spec = e.spec }
            changed = true
          end
        end
      end
      return out, changed
    end
    local excluded_kind = {}
    for _, k in ipairs(req.exclude or {}) do excluded_kind[k] = true end
    -- a surface for every small thing that has none to stand on
    local function ensure_surfaces(ents)
      local on_insts = {}
      for _, e in ipairs(ents) do if e.role == "on" then on_insts[#on_insts + 1] = e.inst end end
      if #on_insts == 0 then return ents end
      local out = {}
      for i, e in ipairs(ents) do out[i] = e end
      for _ = 1, 3 do
        local surfaces = {}
        for _, e in ipairs(out) do
          local tf = top_frame(e.inst.v)
          if tf and e.role ~= "on" then surfaces[#surfaces + 1] = { tf = tf, kind = e.inst.kind, role = e.role } end
        end
        local _, leftover = assign_on(on_insts, surfaces)
        if #leftover == 0 then break end
        local sk, sv = self:surface_for(leftover[1].kind, leftover)
        if not sk or excluded_kind[sk] then break end                       -- (nothing is added that the request said to leave out)
        out[#out + 1] = { inst = make_inst(self, sk, sv, {}), role = (sk == "desk") and "desk" or "center" }
        self:effect_note(res, "I added " .. article(sk) .. " to put " .. (#leftover > 1 and "them" or "it") .. " on")
      end
      return out
    end

    local sequential
    local function room_attempt(ents)
      local rb = self:room(ents, rng, ctx, {})
      local builders = { rb, self:room(ents, rng, ctx, { mirror = true }) }
      local pls, frame, bb, score
      local used_room = false
      -- standing in a real room of about the right size: furnish that room, against its own walls
      local rr = real_room(ctx, WALL_FAR)
      if rr and rr.w <= 720 and rr.d <= 720 and rr.w >= 0.75 * rb.size.w and rr.d >= 0.75 * rb.size.d then
        local rbs = {}
        local shifts = { { 0, 0 }, { 0, 70 }, { 0, -70 }, { 70, 0 }, { -70, 0 }, { 70, 70 }, { 70, -70 }, { -70, 70 }, { -70, -70 }, { 0, 140 }, { 0, -140 } }
        if #ents > 40 then shifts = { { 0, 0 } } end
        for _, sh in ipairs(shifts) do
          rbs[#rbs + 1] = self:room(ents, rng, ctx, { room = { w = rr.w, d = rr.d }, shift = sh })
          rbs[#rbs + 1] = self:room(ents, rng, ctx, { room = { w = rr.w, d = rr.d }, shift = sh, mirror = true })
        end
        local flipped = { x = rr.frame.x, y = rr.frame.y, theta = Layout.wrap(rr.frame.theta + 180) }
        local p2, f2, b2, s2 = fit(ctx, rbs, { rr.frame, flipped }, nil, 0.4)
        if ctx.debug then ctx.debug(string.format("real room %.0fx%.0f, natural %.0fx%.0f -> fit=%s score=%s", rr.w, rr.d, rb.size.w, rb.size.d, tostring(p2 ~= nil), tostring(s2))) end
        if p2 and s2 == 0 then pls, frame, bb, score = p2, f2, b2, s2; used_room = true end
      end
      if not pls then
        pls, frame, bb, score = fit(ctx, builders, open_frames(ctx, rb.radius, theta_square), nil, 0.6)
      end
      if not pls then
        -- a snug version: sized to its contents, no minimum room
        local snug = self:room(ents, rng, ctx, { min_w = 0, min_d = 0 })
        pls, frame, bb, score = fit(ctx, { snug }, open_frames(ctx, snug.radius, theta_square))
      end
      return pls, frame, bb, score, used_room
    end

    --- What is already placed counts as standing while `fn` plans the next group.
    local function with_standing(placed, fn)
      local saved = ctx.registry
      local reg = {}
      for i, e in ipairs(saved or {}) do reg[i] = e end
      for i, p in ipairs(placed) do
        if (p.zoff or 0) <= 0.01 then reg[#reg + 1] = { id = "placed" .. i, kind = p.kind, x = p.x, y = p.y, w = p.w, d = p.d, theta = p.theta } end
      end
      ctx.registry, ctx._ridx = reg, nil
      local a, b = fn()
      ctx.registry, ctx._ridx = saved, nil
      return a, b
    end
    --- The fallback when no one room holds everything: the tables and desks with their seats and loads, then the seats round
    --- nothing, then every other kind (a row, a pile, a circle: whatever its kind falls into), each placed clear of the last.
    --- Whatever finds no place is left out (and said). Returns placements, score (nil placements: nothing fit).
    function sequential(ents)
      local TABLE_ROLES = { center = true, around = true, desk = true, at_desk = true, on = true }
      local SEAT_ROLES = { main_seat = true, side_seat = true, focus = true, beside_main = true }
      local groups, by_kind = { {}, {} }, {}
      for _, e in ipairs(ents) do
        if TABLE_ROLES[e.role] then table.insert(groups[1], e)
        elseif SEAT_ROLES[e.role] then table.insert(groups[2], e)
        else
          local g = by_kind[e.inst.kind]
          if not g then g = { insts = {} }; by_kind[e.inst.kind] = g; groups[#groups + 1] = g end
          table.insert(g.insts, e.inst)
        end
      end
      -- the biggest footprint first: it is the one that needs the open floor
      local function area_of(g)
        local a = 0
        for _, e in ipairs(g.insts or g) do local inst = e.inst or e; a = a + inst.w * inst.d end
        return a
      end
      local order_g = {}
      for gi, g in ipairs(groups) do order_g[#order_g + 1] = { g = g, gi = gi, a = area_of(g) } end
      table.sort(order_g, function(p, q) if p.a ~= q.a then return p.a > q.a end return p.gi < q.gi end)
      local placed, worst = {}, 0
      --- One kind of thing in a layout of its own, clear of what is placed.
      local function place_kind(insts)
        local nm = self:default_arrangement(insts)
        if nm == "stacked" then uniformize(insts) end
        return with_standing(placed, function()
          local _, p, _, sc = self:place_group(ctx, insts, nm, rng, res, { asked = false })
          return p, sc
        end)
      end
      local function keep(gp, gs)
        if not gp then return end
        for _, p in ipairs(gp) do placed[#placed + 1] = p end
        worst = max(worst, gs or 0)
      end
      for _, og in ipairs(order_g) do
        local g, gi = og.g, og.gi
        if ctx.timed_out or ctx.cut_short then break end
        if gi > 2 then
          keep(place_kind(g.insts))
        elseif #g > 0 then
          local gp, gs = with_standing(placed, function() local p, _, _, sc = room_attempt(g); return p, sc end)
          if gp then keep(gp, gs)
          else
            -- not even this much fits as one furnished group: each of its kinds on its own
            local per, kinds_in = {}, {}
            for _, e in ipairs(g) do
              if not per[e.inst.kind] then per[e.inst.kind] = {}; kinds_in[#kinds_in + 1] = e.inst.kind end
              table.insert(per[e.inst.kind], e.inst)
            end
            for _, k in ipairs(kinds_in) do keep(place_kind(per[k])) end
          end
        end
      end
      if #placed == 0 then return nil end
      self:effect_note(res, "it would not fit as one room, so I laid each kind out on its own")
      return placed, worst
    end

    local sized, changed = size_tables(entries)
    sized = ensure_surfaces(sized)
    local pls, _frame, bb, score, used_room = room_attempt(sized)
    if not pls and not ctx.timed_out and changed then
      -- the bigger table did not fit where there is room: the one first drawn, with the seats that fit
      local fallback = ensure_surfaces(entries)
      pls, _frame, bb, score, used_room = room_attempt(fallback)
      sized = fallback
    end
    local separate = false
    if not pls and not ctx.timed_out and not (req.scene and req.scene ~= "") then
      -- no single room holds it all: each kind of thing in a group of its own, one group after the other
      pls, score = sequential(sized)
      separate = pls ~= nil
    end
    if not pls then return refuse(T.no_room) end
    if used_room then self:effect_note(res, "I furnished the room you are standing in") end
    local room_insts = {}
    for i, e in ipairs(sized) do room_insts[i] = e.inst end
    local what = (req.scene and req.scene ~= "" and self.catalog.recipes[req.scene]) and (article(req.scene) .. ": ") or ""
    return conclude(pls, separate and "in separate groups" or "arranged as a room", score, { prefix = what, comma = true, insts = room_insts })
  end

  -- -- a single arrangement, with fallbacks when it does not fit
  local name = explicit_arr and arr or self:default_arrangement(insts)
  if arr == "facing_me" or arr == "unspecified" or arr == "facing_group" then name = self:default_arrangement(insts) end
  local wall_pref = false
  if name == "against_wall" then name, wall_pref = "row", true end
  if name == "in_front_of" or name == "beside" or name == "around" or name == "behind" or name == "on" or name == "flank" then
    name = self:default_arrangement(insts)
  end
  if name == "stacked" then uniformize(insts) end
  local b, pls, frame, score = self:place_group(ctx, insts, name, rng, res,
    { wall_pref = wall_pref, asked = explicit_arr, cols = (name == "grid") and Plan.grid_cols(req, #insts) or nil })
  if not pls then return refuse(T.no_floor) end
  local desc = b.desc
  local end_text = (desc and desc ~= "") and desc or ""
  if (b.wall or wall_pref) and frame and frame.wall then
    -- "against the wall" rather than a bare full stop
    local out = conclude(pls, end_text, score, {})
    out.say = out.say:gsub("%.", ", against the wall.", 1)
    return out
  end
  return conclude(pls, end_text, score, {})
end

-- ---------------------------------------------------------------------------------------------- management verbs
local function by_seq(a, b)
  if (a.seq or 0) ~= (b.seq or 0) then return (a.seq or 0) < (b.seq or 0) end
  return (a.id or 0) < (b.id or 0)
end

-- Bare words the parser reads as one kind but players use for several: "the lamp" is the floor lamp or the desk lamp that stands
-- ("merge": any of them, nearest first); "the chair" and "the table" are the plain kind, and the others only when none stands.
local WORD_FAMILY = {}
for word, fam in pairs({
  lamp = { merge = true, "floor lamp", "desk lamp" }, lamps = { merge = true, "floor lamp", "desk lamp" },
  chair = { "chair", "office chair" }, chairs = { "chair", "office chair" },
  table = { "table", "coffee table", "side table", "meeting table", "patio table" },
  tables = { "table", "coffee table", "side table", "meeting table", "patio table" },
}) do
  fam.has = {}
  for _, k in ipairs(fam) do fam.has[k] = true end
  WORD_FAMILY[word] = fam
end

--- Pick the standing things a spec means. `plain_ok`: a named colour may fall back to the uncoloured ones when none has it.
--- Returns the entries and whether the fallback was used.
local function select_entries(self, ctx, spec, from, plain_ok, include_stale)
  -- what a player may change: ctx.mine (their own things) when the game supplies it, else everything standing
  local list = ctx.mine or ctx.registry or {}
  if spec.kind == "@last" then return last_entries(self, ctx), false end
  if spec.kind == "@all" then
    local out = {}
    for _, e in ipairs(list) do if include_stale or in_library(self, e) then out[#out + 1] = e end end
    table.sort(out, by_seq)
    return out, false
  end
  local pool, plain = registry_matches(self, ctx, spec.kind, spec.adjectives, from, list)
  -- a bare word that names several kinds ("the lamp" for a floor lamp or a desk lamp) means any of them: the parser picks one kind
  -- for the word, and a standing desk lamp would otherwise be invisible to "remove the lamp"
  local fam = spec.word and WORD_FAMILY[tostring(spec.word):lower()]
  if fam and fam.has[spec.kind] then
    for _, k in ipairs(fam) do
      if k ~= spec.kind and (fam.merge or (#pool == 0 and #plain == 0)) and self.kinds[k] then
        local p2, pl2 = registry_matches(self, ctx, k, spec.adjectives, from, list)
        for _, e in ipairs(p2) do pool[#pool + 1] = e end
        for _, e in ipairs(pl2) do plain[#plain + 1] = e end
      end
    end
    local function nearer(a, b)
      local da, db = (a.x - from.x) ^ 2 + (a.y - from.y) ^ 2, (b.x - from.x) ^ 2 + (b.y - from.y) ^ 2
      if da ~= db then return da < db end
      return (a.seq or 0) > (b.seq or 0)
    end
    table.sort(pool, nearer)
    table.sort(plain, nearer)
  end
  local used_plain = false
  if #pool == 0 and plain_ok and #plain > 0 then pool, used_plain = plain, true end
  if spec.count == 0 or #pool == 0 then return pool, used_plain end
  local out = {}
  for i = 1, min(spec.count, #pool) do out[i] = pool[i] end
  return out, used_plain
end

--- The things a management request names (minus `exclude`), each once. info: nothing_recent, plain (a colour fell back to
--- uncoloured ones), emptied (the exclusions left nothing).
function Plan:select(req, ctx, res, mode)
  local from = aim_or_front(ctx)
  local sel, seen, info = {}, {}, {}
  local exclude = {}
  for _, k in ipairs(req.exclude or {}) do exclude[k] = true end
  local verb_text = ({ remove = "to remove", move = "to move", rotate = "to turn", rearrange = "to rearrange" })[mode] or ""
  local unknown_seen = false
  for _, it in ipairs(req.items or {}) do
    if tostring(it.kind):match("^UNKNOWN") then
      note_unknown(res, it.phrase or it.kind:sub(9))
      unknown_seen = true
    else
      local got, used_plain = select_entries(self, ctx, it, from, mode ~= "remove", mode == "remove")
      if used_plain then
        local _, cname = color_of(it.adjectives)
        info.plain = true
        add_note(res, "I had no " .. (cname or "such") .. " " .. it.kind .. ", so I used the plain one")
      end
      local before = #sel
      for _, e in ipairs(got) do
        if not seen[e.id] then
          seen[e.id] = true
          if exclude[e.kind] then info.excluded = true
          else sel[#sel + 1] = e end
        end
      end
      if #got == 0 then
        if it.kind == "@last" then info.nothing_recent = true
        elseif it.kind ~= "@all" then
          local _, cname = color_of(it.adjectives)
          add_note(res, "there is no " .. (cname and (cname .. " ") or "") .. it.kind .. (verb_text ~= "" and (" " .. verb_text) or ""))
        end
      elseif #sel == before and info.excluded then
        info.emptied = true
      end
    end
  end
  -- a word the library lacks is worth a note when it was the point of the request; "turn it 45 degrees" is not
  if unknown_seen and (#sel == 0 or mode == "remove" or mode == "move") then add_note(res, T.library) end
  return sel, info
end

--- The reply for a management request that found nothing to act on.
local function none_say(res, info, default)
  if info and info.nothing_recent then
    res.say = T.nothing_recent .. "."
  elseif info and info.emptied and #res.notes == 0 then
    res.say = "That leaves nothing " .. default:gsub("^There was nothing ", "") .. "."
  elseif #res.notes == 0 then
    res.say = default .. "."
  else
    local rest = {}
    for i = 2, #res.notes do rest[#rest + 1] = res.notes[i] end
    res.say = res.notes[1] .. ((#rest > 0) and ("; " .. table.concat(rest, "; ")) or "") .. "."
  end
  res.fixed = true
  return res
end

function Plan:remove(req, ctx)
  local res = { action = "remove", removals = {}, notes = {}, placements = {} }
  local sel, info = self:select(req, ctx, res, "remove")
  if #sel == 0 then return none_say(res, info, "There was nothing to remove") end
  for _, e in ipairs(sel) do res.removals[#res.removals + 1] = e.id end
  res.say = with_notes("Removed " .. describe(sel, 4) .. ".", res.notes)
  return res
end

-- -- what stands on what: a thing moved takes its load with it
--- Standing things resting on `sel`, and on what rests on them: { {e = entry, parent = entry}, ... } (supports first).
--- `sel_may_ride`: members of `sel` that rest on another member count as riders too (to find who in a selection is only load).
local function riders_of(self, ctx, sel, sel_may_ride)
  local out, taken = {}, {}
  if not sel_may_ride then for _, e in ipairs(sel) do taken[e.id] = true end end
  local reg = ctx.registry or {}
  local queue = {}
  for i, e in ipairs(sel) do queue[i] = e end
  local qi = 1
  while qi <= #queue do
    local s = queue[qi]
    qi = qi + 1
    local k = self.kinds[s.kind]
    local v = k and s.vi and k.variants[s.vi]
    local tf = v and top_frame(v)
    local tops = { (s.z or 0) + (s.h or 0) }
    if tf then tops[#tops + 1] = (s.z or 0) + tf.z end
    local sr = rect(s.x, s.y, s.w, s.d, s.theta)
    for _, e in ipairs(reg) do
      if not taken[e.id] and (e.z or 0) > (s.z or 0) + 0.5 and Layout.contains(sr, e.x, e.y, 2) then
        for _, top in ipairs(tops) do
          if abs((e.z or 0) - top) <= 4 then
            taken[e.id] = true
            out[#out + 1] = { e = e, parent = s }
            queue[#queue + 1] = e
            break
          end
        end
      end
    end
  end
  return out
end

--- Move what stood on the moved things by the same rigid motion. moves_by_id[id] = {id, x, y, z, theta} for the supports.
local function carry_riders(moves_by_id, riders)
  for _, r in ipairs(riders) do
    local pm = moves_by_id[r.parent.id]
    if pm and not moves_by_id[r.e.id] then
      local old = r.parent
      local dth = Layout.angle_diff(pm.theta, old.theta)
      local c, s = cos(rad(dth)), sin(rad(dth))
      local dx, dy = r.e.x - old.x, r.e.y - old.y
      moves_by_id[r.e.id] = { id = r.e.id, x = pm.x + dx * c - dy * s, y = pm.y + dx * s + dy * c,
                              z = (r.e.z or 0) + (pm.z - (old.z or 0)), theta = Layout.wrap((r.e.theta or 0) + dth), carried = true }
    end
  end
end

--- res.moves from the supports' moves plus the load they carry (a support always comes before what rests on it).
local function emit_moves(res, order, moves_by_id, riders)
  carry_riders(moves_by_id, riders)
  for _, id in ipairs(order) do
    if not moves_by_id[id].virtual then res.moves[#res.moves + 1] = moves_by_id[id] end
  end
  for _, r in ipairs(riders) do
    if moves_by_id[r.e.id] then res.moves[#res.moves + 1] = moves_by_id[r.e.id] end
  end
end

--- Where would the load (riders) of the moved things stand if they went with them, and would it stand on floor? Returns a test for
--- judge (true: some carried thing would poke out of the room), or nil when nothing is carried. A placement stands for the
--- standing thing in its inst.entry.
local function carry_check_for(ctx, riders)
  if not riders or #riders == 0 then return nil end
  return function(pls)
    local moves = {}
    for _, p in ipairs(pls) do
      local e = p.inst and p.inst.entry
      if e then moves[e.id] = { id = e.id, x = p.x, y = p.y, z = e.z or 0, theta = Layout.wrap(p.theta) } end
    end
    carry_riders(moves, riders)
    for _, r in ipairs(riders) do
      local m = moves[r.e.id]
      if m and m.carried and not floor_ok(ctx.env, rect(m.x, m.y, r.e.w, r.e.d, m.theta)) then return true end
    end
    return false
  end
end

--- The same for a turn in place: would the load of `e` stand on floor when `e` is at (x, y, theta)?
local function carry_ok_for(ctx, e, riders)
  local mine = {}
  for _, r in ipairs(riders or {}) do mine[#mine + 1] = r end
  if #mine == 0 then return nil end
  return function(x, y, theta)
    local moves = { [e.id] = { id = e.id, x = x, y = y, z = e.z or 0, theta = Layout.wrap(theta) } }
    carry_riders(moves, mine)
    for _, r in ipairs(mine) do
      local m = moves[r.e.id]
      if m and m.carried and not floor_ok(ctx.env, rect(m.x, m.y, r.e.w, r.e.d, m.theta)) then return false end
    end
    return true
  end
end

local function id_set(list)
  local s = {}
  for _, e in ipairs(list) do s[e.id] = true end
  return s
end
local function rider_ids(riders)
  local s = {}
  for _, r in ipairs(riders) do s[r.e.id] = true end
  return s
end

--- The entries of `sel` that are not resting on another of them (those travel with what carries them).
--- Second result: the entries dropped because they ride on a kept one (they still count in the reply).
local function without_riders(self, ctx, sel)
  local all = riders_of(self, ctx, sel, true)
  local carried = {}
  for _, r in ipairs(all) do carried[r.e.id] = true end
  local top, dropped = {}, {}
  for _, e in ipairs(sel) do
    if carried[e.id] then dropped[#dropped + 1] = e else top[#top + 1] = e end
  end
  if #top == 0 then return sel, {} end
  return top, dropped
end

--- `list` followed by the entries of `more` it does not hold yet.
local function with_more(list, more)
  if not more or #more == 0 then return list end
  local out, seen = {}, {}
  for _, e in ipairs(list) do out[#out + 1] = e; seen[e.id] = true end
  for _, e in ipairs(more) do if not seen[e.id] then out[#out + 1] = e end end
  return out
end

--- A pile (something rests on two or more of the others) cannot be re-laid piece by piece: it becomes ONE rigid thing, a virtual
--- entry with the pile's bounding footprint, the pile's members riding on it. Returns that entry and the riders, or nil.
local function pile_of(sel, riders)
  local members, seen = {}, {}
  for _, e in ipairs(sel) do if not seen[e.id] then seen[e.id] = true; members[#members + 1] = e end end
  for _, r in ipairs(riders) do if not seen[r.e.id] then seen[r.e.id] = true; members[#members + 1] = r.e end end
  local multi = false
  for _, r in ipairs(riders) do
    local re = rect(r.e.x, r.e.y, r.e.w, r.e.d, r.e.theta)
    local n = 0
    for _, m in ipairs(members) do
      if m ~= r.e and (r.e.z or 0) > (m.z or 0) + 0.5 and abs((r.e.z or 0) - ((m.z or 0) + (m.h or 0))) <= 4
         and Layout.penetration(rect(m.x, m.y, m.w, m.d, m.theta), re) > 1 then n = n + 1 end
    end
    if n >= 2 then multi = true; break end
  end
  if not multi then return nil end
  table.sort(members, function(a, b)
    if (a.z or 0) ~= (b.z or 0) then return (a.z or 0) < (b.z or 0) end
    return tostring(a.id) < tostring(b.id)
  end)
  local base = members[1]
  local fx, fy = fwd(base.theta)
  local rx, ry = fy, -fx
  local u0, u1, v0, v1, ztop = 1e9, -1e9, 1e9, -1e9, -1e9
  for _, m in ipairs(members) do
    for _, c in ipairs(Layout.corners(rect(m.x, m.y, m.w, m.d, m.theta))) do
      local u, v = (c[1] - base.x) * fx + (c[2] - base.y) * fy, (c[1] - base.x) * rx + (c[2] - base.y) * ry
      u0, u1, v0, v1 = min(u0, u), max(u1, u), min(v0, v), max(v1, v)
    end
    ztop = max(ztop, (m.z or 0) + (m.h or 0))
  end
  local cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
  local virtual = { id = "pile", virtual = true, kind = base.kind, vi = base.vi, x = base.x + cu * fx + cv * rx, y = base.y + cu * fy + cv * ry,
                    theta = base.theta, w = v1 - v0, d = u1 - u0, h = ztop - (base.z or 0), z = base.z or 0, members = members }
  local rs = {}
  for _, m in ipairs(members) do rs[#rs + 1] = { e = m, parent = virtual } end
  return virtual, rs
end

--- `list` with each virtual pile replaced by the things it is made of.
local function expand_piles(list)
  local out = {}
  for _, e in ipairs(list) do
    if e.virtual then for _, m in ipairs(e.members) do out[#out + 1] = m end else out[#out + 1] = e end
  end
  return out
end

-- -- turning
--- A test of poses for a standing thing: ok(x, y, theta) -> fits, floor z. A pose fits when it is not worse than the present one
--- (no deeper in a neighbour, a wall or a person; still on a floor), so a chair tucked under a table can still be turned about.
--- `ignore`: ids that do not count as in the way; `decided`: rects of the others already turned in this request.
local function pose_checker(ctx, e, ignore, decided)
  local env = ctx.env
  local old_r = rect(e.x, e.y, e.w, e.d, e.theta)
  local fz_old = floor_ok(env, old_r)
  local elevated = fz_old and (e.z or 0) - fz_old > 1.5
  local clear_old = fz_old and not elevated and env.clear(old_r, fz_old, e.h)
  local people = keep_outs(ctx)
  local was_on_me = {}
  for i, me in ipairs(people) do was_on_me[i] = Layout.overlap(me, old_r, KEEP_OUT_PAD) end
  local function ok(x, y, theta)
    local r = rect(x, y, e.w, e.d, theta)
    local fz = floor_ok(env, r)
    if clear_old or (fz_old and not elevated) then
      if not fz then return false end
      if clear_old and not env.clear(r, fz, e.h) then return false end
    end
    for i, me in ipairs(people) do
      if not was_on_me[i] and Layout.overlap(me, r, KEEP_OUT_PAD) then return false end
    end
    local worse = false
    each_standing(ctx, r, 0.4, function(er)
      if worse or ignore[er.id] or er.id == e.id then return end
      if Layout.overlap(er, r, 0.4) then
        local pn = Layout.penetration(er, r)
        if pn > 0.4 and pn > Layout.penetration(er, old_r) + 0.5 then worse = true end
      end
    end)
    if worse then return false end
    for _, d in ipairs(decided or {}) do
      if Layout.overlap(d, r, 0.4) then return false end
    end
    return true, fz
  end
  return ok, { old_r = old_r, fz_old = fz_old, elevated = elevated }
end

--- Where a thing can turn to `theta` as {x, y, z}: in place when that is no worse, else eased off a wall or a neighbour by as much
--- as the turn needs; nil when no nearby spot will do (and for a thing tucked into others, which is never slid away from them).
local function settle_turn(self, ctx, e, theta, ignore, decided, carry_ok)
  local ok, st = pose_checker(ctx, e, ignore, decided)
  if carry_ok then
    -- what the thing carries turns with it: the load must stay on floor too
    local ok0 = ok
    ok = function(x, y, t)
      local g, z = ok0(x, y, t)
      if g and not carry_ok(x, y, t) then return false end
      return g, z
    end
  end
  local old_r, fz_old, elevated = st.old_r, st.fz_old, st.elevated
  -- footprint-neutral: a half turn, or a quarter turn of a square one
  local dth = abs(Layout.angle_diff(theta, e.theta))
  if dth < 0.5 or abs(dth - 180) < 0.5 or (abs(dth - 90) < 0.5 and abs(e.w - e.d) < 1) then return { x = e.x, y = e.y, z = e.z } end
  if elevated then
    -- on a surface: stay on it (the turned corners inside whatever it rests on)
    for _, s in ipairs(ctx.registry or {}) do
      if s.id ~= e.id and (s.z or 0) < (e.z or 0) - 1 and Layout.contains(rect(s.x, s.y, s.w, s.d, s.theta), e.x, e.y, 0) then
        local sr = rect(s.x, s.y, s.w, s.d, s.theta)
        for _, c in ipairs(Layout.corners(rect(e.x, e.y, e.w, e.d, theta))) do
          if not Layout.contains(sr, c[1], c[2], 2) then return nil end
        end
        break
      end
    end
    return { x = e.x, y = e.y, z = e.z }
  end
  local good, fz = ok(e.x, e.y, theta)
  if good then return { x = e.x, y = e.y, z = (e.z or 0) + ((fz or fz_old or 0) - (fz_old or 0)) } end
  -- a thing tucked into others (a table with its chairs round it) is not slid away from them
  local tucked = false
  each_standing(ctx, old_r, 0.4, function(er)
    if not ignore[er.id] and er.id ~= e.id and Layout.overlap(er, old_r, 0.4) then tucked = true end
  end)
  if tucked then return nil end
  -- the most a turned footprint can grow past the old one is its half-diagonal less its short half side (a bench turned a quarter
  -- against a wall needs its whole length less its depth); the rings widen by a quarter each, so none up to that is skipped
  local reach = (sqrt(e.w * e.w + e.d * e.d) - min(e.w, e.d)) / 2 + 8
  local ring = 0
  while ring < reach do
    ring = ring + max(3, floor(ring / 4))
    if expired(ctx, 1) then return nil end
    for k = 0, 11 do
      local a = k * 30
      local x, y = e.x + ring * cos(rad(a)), e.y + ring * sin(rad(a))
      local g, gz = ok(x, y, theta)
      if g then return { x = x, y = y, z = (e.z or 0) + ((gz or fz_old or 0) - (fz_old or 0)) } end
    end
  end
  return nil
end

--- Turn standing things: half a turn, to face the player or another thing, each other, a wall, or by an angle in the words.
function Plan:rotate(req, ctx)
  local res = { action = "rotate", moves = {}, notes = {}, placements = {} }
  local sel, info = self:select(req, ctx, res, "rotate")
  local usable = {}
  for _, e in ipairs(sel) do if in_library(self, e) then usable[#usable + 1] = e end end
  local sel_all
  sel, sel_all = without_riders(self, ctx, usable)
  if #sel == 0 then return none_say(res, info, "There was nothing to turn") end
  local text = req.normalised or ""
  local arrangement = req.arrangement
  local tg = req.facing or req.target
  local away = (arrangement == "facing_away")
  local each_other = (arrangement == "facing_each_other")
  local facing = away or each_other or arrangement == "facing_me" or (req.facing ~= nil and req.facing ~= "")
  local sel_ids = id_set(sel)

  -- where does this one look? nil: it stays as it is
  local why
  local cx, cy = 0, 0
  for _, e in ipairs(sel) do cx, cy = cx + e.x, cy + e.y end
  cx, cy = cx / #sel, cy / #sel
  local function aim_for(e)
    if each_other then
      if #sel < 2 then why = "one thing can't face another"; return nil end
      if (e.x - cx) ^ 2 + (e.y - cy) ^ 2 < 1 then return nil end
      return Layout.angle_of(cx - e.x, cy - e.y)
    end
    local pt
    if tg == nil or tg == "" or tg == "me" or tg == "here" then pt = { x = ctx.pos.x, y = ctx.pos.y }
    elseif tg == "wall" then
      local wall = Layout.nearest_wall(ctx.env, e.x, e.y, 600, 36)
      if not wall then why = "there is no wall in reach"; return nil end
      local th = Layout.angle_of(-wall.nx, -wall.ny)
      return away and Layout.wrap(th + 180) or th
    elseif tg == "@last" then
      local rest = {}
      for _, m in ipairs(last_entries(self, ctx)) do if not sel_ids[m.id] then rest[#rest + 1] = m end end
      local anchor = group_anchor(rest)
      if not anchor then why = T.nothing_recent; return nil end
      pt = { x = anchor.x, y = anchor.y }
    elseif NOT_A_THING[tg] then why = "I can't tell where that is"; return nil
    elseif self.kinds[tg] then
      local best
      for _, m in ipairs((registry_matches(self, ctx, tg, {}, e, ctx.registry))) do
        if not sel_ids[m.id] then best = m; break end
      end
      if not best then why = "there is no " .. tg .. " to face"; return nil end
      pt = { x = best.x, y = best.y }
    else
      note_unknown(res, tg)
      why = T.library
      return nil
    end
    if (pt.x - e.x) ^ 2 + (pt.y - e.y) ^ 2 < 1 then return nil end
    local th = Layout.angle_of(pt.x - e.x, pt.y - e.y)
    return away and Layout.wrap(th + 180) or th
  end

  local angle = tonumber(text:match("(%d+)%s*degree")) or tonumber(text:match("(%d+)%s*deg"))
  if text:find("quarter") then angle = angle or 90 end
  if text:find("half") or req.half_turn or text:find("around") or text:find("flip") or text:find("backward") then angle = angle or 180 end
  local sign = (text:find("clockwise") and not text:find("counter") and not text:find("anti")) and -1 or 1
  if text:find(" right") and not text:find("left") then sign = -1 end

  local riders = riders_of(self, ctx, sel)
  local ignore = rider_ids(riders)
  local decided, moves_by_id, order = {}, {}, {}
  local turned, stuck, unresolved = {}, 0, 0
  -- "turn it around" of a group that was made together: the group turns as one body, about its middle (a table and its chairs
  -- keep their places round it), or not at all when there is no room for that
  local one_body = false
  if not facing and #sel > 1 then
    local named_last = false
    for _, it in ipairs(req.items or {}) do if it.kind == "@last" then named_last = true end end
    one_body = named_last
  end
  if one_body then
    local turn = Layout.wrap((angle or 90) * sign)
    local c, sn = cos(rad(turn)), sin(rad(turn))
    local all_ids = id_set(sel)
    for id in pairs(ignore) do all_ids[id] = true end
    local fits, plan = true, {}
    for _, e in ipairs(sel) do
      local nx = cx + (e.x - cx) * c - (e.y - cy) * sn
      local ny = cy + (e.x - cx) * sn + (e.y - cy) * c
      local th = Layout.wrap(e.theta + turn)
      local ok = pose_checker(ctx, e, all_ids, nil)
      local good, fz = ok(nx, ny, th)
      if not good then fits = false; break end
      plan[#plan + 1] = { e = e, x = nx, y = ny, theta = th, dz = 0 }
    end
    if not fits then
      res.say = "There was no room to turn them there."
      res.fixed = true
      return res
    end
    for _, q in ipairs(plan) do
      moves_by_id[q.e.id] = { id = q.e.id, x = q.x, y = q.y, z = q.e.z, theta = q.theta }
      order[#order + 1] = q.e.id
      turned[#turned + 1] = q.e
    end
    sel = {}
  end
  for _, e in ipairs(sel) do
    local theta
    if facing then
      theta = aim_for(e)
      if not theta then unresolved = unresolved + 1 end
    else
      theta = Layout.wrap(e.theta + sign * (angle or 90))
    end
    if theta then
      local at = settle_turn(self, ctx, e, theta, ignore, decided, carry_ok_for(ctx, e, riders))
      if at then
        moves_by_id[e.id] = { id = e.id, x = at.x, y = at.y, z = at.z, theta = Layout.wrap(theta), rotate_only = (at.x == e.x and at.y == e.y) or nil }
        order[#order + 1] = e.id
        decided[#decided + 1] = rect(at.x, at.y, e.w, e.d, theta)
        turned[#turned + 1] = e
      else
        stuck = stuck + 1
        decided[#decided + 1] = rect(e.x, e.y, e.w, e.d, e.theta)
      end
    end
  end
  if why then add_note(res, why) end
  if #turned == 0 then
    if unresolved == #sel then
      res.say = with_notes("I could not tell what to face, so I left " .. ((#sel > 1) and "them" or "it") .. " as " .. ((#sel > 1) and "they were" or "it was") .. ".", res.notes)
    else
      res.say = "There was no room to turn " .. ((#sel > 1) and "them" or "it") .. " there."
    end
    res.fixed = true
    return res
  end
  -- what rests on a turned thing turns with it
  local turned_riders = {}
  for _, r in ipairs(riders) do if moves_by_id[r.parent.id] or moves_by_id[r.e.id] then turned_riders[#turned_riders + 1] = r end end
  emit_moves(res, order, moves_by_id, turned_riders)
  local say = "Turned " .. describe(with_more(turned, sel_all), 4)
  if stuck > 0 then add_note(res, count_word(stuck) .. " would not turn there, so " .. (stuck > 1 and "they stayed" or "it stayed")) end
  res.say = with_notes(say .. ".", res.notes)
  return res
end

-- -- moving
local MOVE_MAKES = { put = true, place = true, set = true, position = true, bring = true, stick = true }

function Plan:move(req, ctx)
  local res = { action = "move", moves = {}, notes = {}, placements = {} }
  local from = aim_or_front(ctx)
  local sel, info = self:select(req, ctx, res, "move")
  local usable = {}
  for _, e in ipairs(sel) do if in_library(self, e) then usable[#usable + 1] = e end end
  local dropped
  sel, dropped = without_riders(self, ctx, usable)
  local target = req.target or ""
  local arr = req.arrangement or "unspecified"
  if #sel == 0 then
    -- "put the lamp next to the couch" with no lamp standing: a lamp is made, there
    local definite_missing = false
    for _, it in ipairs(req.items or {}) do
      if it.kind ~= "@last" and it.kind ~= "@all" and not tostring(it.kind):match("^UNKNOWN") then definite_missing = true end
    end
    if definite_missing and MOVE_MAKES[req.verb or ""] and arr ~= "unspecified" then
      local areq = {}
      for k2, v2 in pairs(req) do areq[k2] = v2 end
      areq.action, areq.items = "add", {}
      for _, it in ipairs(req.items) do
        local copy = {}
        for k2, v2 in pairs(it) do copy[k2] = v2 end
        if (copy.count or 0) == 0 then copy.count = copy.plural and 3 or 1 end
        areq.items[#areq.items + 1] = copy
      end
      local added = self:add(areq, ctx)
      if #added.placements > 0 then
        added.notes[#added.notes + 1] = "there was nothing of yours to move, so I made it"
        added.say = with_notes(added.say, { added.notes[#added.notes] })
        return added
      end
    end
    return none_say(res, info, "There was nothing to move")
  end
  local riders = riders_of(self, ctx, sel)
  do
    local pile, pile_riders = pile_of(sel, riders)
    if pile then sel, riders = { pile }, pile_riders end
  end
  local ignore = id_set(sel)
  ctx.carry_check = carry_check_for(ctx, riders)            -- the load that goes with them must stay in the room too
  local carry_say = "What rests on " .. ((#sel > 1) and "them" or "it") .. " would not fit there."
  for id in pairs(rider_ids(riders)) do ignore[id] = true end
  local text = req.normalised or ""
  local pos, yaw = ctx.pos, ctx.yaw
  local fx, fy = fwd(yaw)
  local rx, ry = fy, -fx

  local insts = {}
  for i, e in ipairs(sel) do
    local inst = make_inst(self, e.kind, e.vi, {})
    inst.entry = e
    if e.virtual then inst.w, inst.d, inst.h = e.w, e.d, e.h end
    insts[i] = inst
  end
  local rng = Layout.rng(ctx.seed or Layout.hash(req.raw or ""))
  --- Write the moves for placements `pls` (each with .inst.entry), carry the load, and report.
  local function commit(pls)
    local moves_by_id, order = {}, {}
    local shifted = false
    for _, p in ipairs(pls) do
      local e = p.inst.entry
      local z = (p.fz or pos.z) + (p.zoff or 0)
      moves_by_id[e.id] = { id = e.id, x = p.x, y = p.y, z = z, theta = Layout.wrap(p.theta), zoff = p.zoff, fz = p.fz, virtual = e.virtual }
      order[#order + 1] = e.id
      if abs(p.x - e.x) > 1.5 or abs(p.y - e.y) > 1.5 or abs(Layout.angle_diff(p.theta, e.theta)) > 1 then shifted = true end
    end
    if not shifted then
      res.say = "It is already there."
      res.fixed = true
      return res
    end
    emit_moves(res, order, moves_by_id, riders)
    local moved = {}
    for _, p in ipairs(pls) do moved[#moved + 1] = p.inst.entry end
    res.say = with_notes("Moved " .. describe(with_more(expand_piles(moved), dropped), 4) .. ".", res.notes)
    return res
  end

  -- "to the other side of the couch", "on the left of the couch" (req.side): just the thing named moves, to that side of the reference
  local side_req = (type(req.side) == "string" and req.side ~= "") and req.side or nil
  if side_req and arr == "unspecified" and target ~= "" then arr = "beside" end
  -- -- beside / in front of / on / around a standing thing
  local ref_entry
  if target == "@last" and (arr == "beside" or arr == "in_front_of" or arr == "behind" or arr == "on" or arr == "around" or arr == "facing_me" or arr == "flank") then
    local rest = {}
    for _, e in ipairs(last_entries(self, ctx)) do if not ignore[e.id] then rest[#rest + 1] = e end end
    ref_entry = group_anchor(rest)
    if not ref_entry then return fixed(res, T.nothing_recent .. ".") end
  elseif (arr == "beside" or arr == "in_front_of" or arr == "behind" or arr == "on" or arr == "around" or arr == "facing_me" or arr == "flank")
         and target ~= "" and not NOT_A_THING[target] and self.kinds[target] then
    for _, e in ipairs((registry_matches(self, ctx, target, {}, from))) do
      if not ignore[e.id] then ref_entry = e; break end
    end
    if not ref_entry then
      -- nothing to move them to: they stay where they are
      return fixed(res, "There is no " .. target .. " to move " .. ((#sel > 1) and "them" or "it") .. " to.")
    end
  end
  if ref_entry then
    local r = ref_of(ref_entry)
    local k = self.kinds[ref_entry.kind]
    r.topf = ref_entry.vi and k and top_frame(k.variants[ref_entry.vi]) or nil
    r.kind = ref_entry.kind
    local rel = (arr == "facing_me") and "in_front_of" or arr
    if rel == "flank" then rel = "beside" end
    local b
    if rel == "around" then b = around_builder(r, insts, {})
    elseif rel == "on" and r.topf then b = on_builder(r, insts, {})
    else b = offset_builder(r, insts, (rel == "on") and "beside" or rel, { distance = (r.kind == "tv") and max(70, 1.6 * r.w) or nil }) end
    local bs = { b }
    if rel == "around" and (b.dropped or 0) > 0 then table.insert(bs, 1, ring_builder(r, insts, { stop = stopper(ctx) })) end
    if rel == "around" and #insts > 1 then
      for _, a in ipairs({ 300, 240, 180, 140 }) do bs[#bs + 1] = ring_builder(r, insts, { stop = stopper(ctx), arc = a }) end
    end
    if rel == "beside" or rel == "on" then
      bs[#bs + 1] = offset_builder(r, insts, "beside", { flip = true })
      bs[#bs + 1] = offset_builder(r, insts, "beside", { align = true })
      bs[#bs + 1] = offset_builder(r, insts, "beside", { align = true, flip = true })
    end
    if side_req then
      -- the side asked for: "left"/"right" are the player's own; "other"/"opposite" is the side facing away from the player, and not
      -- the one the thing stands at now (a lamp by a couch's right end goes to its left end when the back is a wall). The other sides
      -- follow, the nearest to the wanted direction first, for when a wall or a neighbour fills the first.
      local rfx, rfy = fwd(r.theta)
      local rrx, rry = Layout.right(r.theta)
      local sides = { front = { rfx, rfy }, back = { -rfx, -rfy }, right = { rrx, rry }, left = { -rrx, -rry } }
      local wx, wy                                                    -- the direction the wanted side faces in the world
      local strict = false
      if side_req == "left" or side_req == "right" then
        local px, py = Layout.right(yaw)
        if side_req == "left" then px, py = -px, -py end
        wx, wy = px, py
        strict = true
      else
        wx, wy = r.x - pos.x, r.y - pos.y
        if wx * wx + wy * wy < 1 then wx, wy = fwd(yaw) end
      end
      local here
      do
        local e1, best_dot = sel[1], -1e9
        for nm, n in pairs(sides) do
          local dot = (e1.x - r.x) * n[1] + (e1.y - r.y) * n[2]
          if dot > best_dot then here, best_dot = nm, dot end
        end
      end
      local ranked = {}
      for _, nm in ipairs({ "front", "back", "right", "left" }) do
        ranked[#ranked + 1] = { nm = nm, dot = sides[nm][1] * wx + sides[nm][2] * wy }
      end
      table.sort(ranked, function(p, q) if p.dot ~= q.dot then return p.dot > q.dot end return p.nm < q.nm end)
      bs = {}
      for ri, rk in ipairs(ranked) do
        if (ri == 1 or not strict) and (strict or rk.nm ~= here or ri == #ranked) then
          for _, al in ipairs({ false, true }) do
            local o3 = { align = al or nil, distance = (r.kind == "tv") and max(70, 1.6 * r.w) or nil }
            if rk.nm == "right" or rk.nm == "left" then
              o3.only_side = (rk.nm == "right") and 1 or -1
              bs[#bs + 1] = offset_builder(r, insts, "beside", o3)
            else
              bs[#bs + 1] = offset_builder(r, insts, (rk.nm == "front") and "in_front_of" or "behind", o3)
            end
          end
        end
      end
    end
    local pls, _, bb = fit(ctx, bs, { { x = r.x, y = r.y, theta = r.theta } }, ignore)
    if pls then
      if rel == "on" and not r.topf then add_note(res, "I can't stand things on the " .. ref_entry.kind .. ", so I put them beside it") end
      return commit(pls)
    end
    return fixed(res, ctx._carry_blocked and carry_say or "There was no clear spot for that.")
  end

  -- -- against a wall, in a corner
  if arr == "against_wall" or arr == "corner" then
    local name = (arr == "corner") and "corner" or "row"
    local b, pls = self:place_group(ctx, insts, name, rng, res, { wall_pref = (arr == "against_wall"), ignore = ignore, strict = true })
    if pls then return commit(pls) end
    if ctx.cut_short then res.timed_out = true; return fixed(res, T.too_much .. ".") end
    return fixed(res, ctx._carry_blocked and carry_say or ((arr == "corner") and "There is no corner in reach." or "There is no wall in reach."))
  end

  -- loose directions in the words
  local dx, dy
  local closer = false
  if text:find("closer") or text:find("toward me") or text:find("towards me") or text:find("over here") then
    local e = sel[1]
    local ux, uy = pos.x - e.x, pos.y - e.y
    local len = sqrt(ux * ux + uy * uy)
    closer = true
    if len > 1 then dx, dy = ux / len * min(80, max(0, len - 70)), uy / len * min(80, max(0, len - 70)) end
  elseif text:find("away") or text:find("farther") or text:find("further") then
    local e = sel[1]
    -- "away from the couch": away from that, not from the player
    local ox, oy = pos.x, pos.y
    if req.away_from and self.kinds[req.away_from] then
      local others = {}
      for _, r in ipairs(ctx.registry or {}) do if not ignore[r.id] then others[#others + 1] = r end end
      local near = registry_matches(self, ctx, req.away_from, {}, e, others)
      if near[1] then ox, oy = near[1].x, near[1].y
      else add_note(res, "there was no " .. req.away_from .. " to go by, so I moved " .. (#sel > 1 and "them" or "it") .. " away from you") end
    end
    local ux, uy = e.x - ox, e.y - oy
    local len = max(1, sqrt(ux * ux + uy * uy))
    dx, dy = ux / len * 80, uy / len * 80
  elseif text:find(" left") then dx, dy = -rx * 60, -ry * 60
  elseif text:find(" right") then dx, dy = rx * 60, ry * 60
  elseif text:find("forward") then dx, dy = fx * 60, fy * 60
  elseif text:find(" back") or text:find("backward") then dx, dy = -fx * 60, -fy * 60
  end
  if dx then
    if closer and abs(dx) + abs(dy) < 2 then return fixed(res, "It is already as close as I can get it.") end
    local function shifted_by(k)
      local pls = {}
      for i, e in ipairs(sel) do pls[i] = P(insts[i], { x = e.x + dx * k, y = e.y + dy * k, theta = e.theta, z = 0 }, "moved") end
      return pls
    end
    -- the group keeps its shape, so only what is outside it can be in the way
    local pls = shifted_by(1)
    local viol = judge(ctx, pls, ignore, { rigid = true })
    if viol == 0 then return commit(pls) end
    -- blocked: try half the shift
    pls = shifted_by(0.5)
    viol = judge(ctx, pls, ignore, { rigid = true })
    if viol == 0 then return commit(pls) end
    return fixed(res, ctx._carry_blocked and carry_say or "Something was in the way.")
  end

  -- otherwise to where the player is looking, laid out as they were (or in a row)
  local name = (arr ~= "unspecified" and arr ~= "facing_me" and arr ~= "around" and arr ~= "beside" and arr ~= "in_front_of" and arr ~= "behind"
                and arr ~= "on" and arr ~= "flank") and arr or self:default_arrangement(insts)
  local b0 = self:arrangement(name, insts, rng, ctx, {})
  local at = ctx.aim or { x = pos.x + fx * (b0.radius + 90), y = pos.y + fy * (b0.radius + 90) }
  local first = { x = at.x, y = at.y, theta = Layout.angle_of(pos.x - at.x, pos.y - at.y) }
  local b, pls = self:place_group(ctx, insts, name, rng, res, {
    ignore = ignore, frames_first = { first },
    theta_for = function(cx, cy) return Layout.angle_of(pos.x - cx, pos.y - cy) end })
  if pls then return commit(pls) end
  return fixed(res, ctx.cut_short and (T.too_much .. ".") or (T.no_floor .. "."))
end

--- The shape a set of standing things is already in, if it is one we know: a row, a circle.
local function shape_of(entries)
  local n = #entries
  if n < 3 then return nil end
  local cx, cy = 0, 0
  for _, e in ipairs(entries) do cx, cy = cx + e.x, cy + e.y end
  cx, cy = cx / n, cy / n
  -- a row: nearly all on one line through the middle
  local sxx, sxy, syy = 0, 0, 0
  for _, e in ipairs(entries) do
    local dx, dy = e.x - cx, e.y - cy
    sxx, sxy, syy = sxx + dx * dx, sxy + dx * dy, syy + dy * dy
  end
  local ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
  local ux, uy = cos(ang), sin(ang)
  local off, width = 0, 0
  for _, e in ipairs(entries) do
    off = max(off, abs(-(e.x - cx) * uy + (e.y - cy) * ux))
    width = width + e.w
  end
  if off <= 0.35 * (width / n) then return "row" end
  -- a circle: all about as far from the middle
  local mean = 0
  for _, e in ipairs(entries) do mean = mean + sqrt((e.x - cx) ^ 2 + (e.y - cy) ^ 2) end
  mean = mean / n
  local spread = 0
  for _, e in ipairs(entries) do spread = max(spread, abs(sqrt((e.x - cx) ^ 2 + (e.y - cy) ^ 2) - mean)) end
  if mean > 8 and spread <= 0.18 * mean + 4 then return "circle" end
  return nil
end

function Plan:rearrange(req, ctx)
  local res = { action = "rearrange", moves = {}, notes = {}, placements = {} }
  local sel, info = self:select(req, ctx, res, "rearrange")
  local usable = {}
  for _, e in ipairs(sel) do if in_library(self, e) then usable[#usable + 1] = e end end
  local dropped
  sel, dropped = without_riders(self, ctx, usable)
  if #sel == 0 then return none_say(res, info, "There was nothing to rearrange") end
  local riders = riders_of(self, ctx, sel)
  do
    local pile, pile_riders = pile_of(sel, riders)
    if pile then sel, riders = { pile }, pile_riders end
  end
  local ignore = id_set(sel)
  ctx.carry_check = carry_check_for(ctx, riders)
  for id in pairs(rider_ids(riders)) do ignore[id] = true end
  local insts = {}
  for i, e in ipairs(sel) do
    local inst = make_inst(self, e.kind, e.vi, {})
    inst.entry = e
    if e.virtual then inst.w, inst.d, inst.h = e.w, e.d, e.h end
    insts[i] = inst
  end
  local rng = Layout.rng(ctx.seed or Layout.hash(req.raw or ""))
  local arr = req.arrangement or "unspecified"
  local explicit = arr ~= "unspecified" and arr ~= "facing_me" and arr ~= "around" and arr ~= "beside" and arr ~= "in_front_of"
                   and arr ~= "behind" and arr ~= "on" and arr ~= "flank"
  local name
  if explicit then name = arr
  else
    -- nothing asked for: keep the shape they are in
    name = shape_of(sel) or self:default_arrangement(insts)
  end
  local wall_pref = false
  if name == "against_wall" then name, wall_pref = "row", true end
  -- the centroid of what is there, facing the player
  local cx, cy = 0, 0
  for _, e in ipairs(sel) do cx, cy = cx + e.x, cy + e.y end
  cx, cy = cx / #sel, cy / #sel
  local strict = (arr == "against_wall" or arr == "corner")
  local function toward(px, py) return Layout.angle_of(ctx.pos.x - px, ctx.pos.y - py) end
  local b, pls = self:place_group(ctx, insts, name, rng, res, {
    ignore = ignore, wall_pref = wall_pref, strict = strict, theta_for = toward, cols = (name == "grid") and Plan.grid_cols(req, #insts) or nil,
    frames_first = (not strict) and { { x = cx, y = cy, theta = toward(cx, cy) } } or nil })
  if not pls then
    if ctx.cut_short then res.timed_out = true; return fixed(res, T.too_much .. ".") end
    if strict then return fixed(res, ctx._carry_blocked and "What rests on it would not fit there." or ((arr == "corner") and "There is no corner in reach." or "There is no wall in reach.")) end
    return fixed(res, T.no_floor .. ".")
  end
  local moves_by_id, order = {}, {}
  for _, p in ipairs(pls) do
    local e = p.inst.entry
    moves_by_id[e.id] = { id = e.id, x = p.x, y = p.y, z = (p.fz or ctx.pos.z) + (p.zoff or 0), theta = Layout.wrap(p.theta), zoff = p.zoff, fz = p.fz,
                          virtual = e.virtual }
    order[#order + 1] = e.id
  end
  emit_moves(res, order, moves_by_id, riders)
  local say = "Rearranged " .. describe(with_more(expand_piles(sel), dropped), 4) .. ((b.desc and b.desc ~= "") and (" " .. b.desc) or "")
  res.say = with_notes(say .. ".", res.notes)
  return res
end

--- Entry point.
function Plan:run(req, ctx)
  ctx.registry = ctx.registry or {}
  req = req or {}
  -- one clock and one deadline for the whole request; the places the time goes ask whether it has passed
  ctx.clock = ctx.now or os.clock
  ctx.deadline = ctx.clock() + (ctx.budget or DEFAULT_BUDGET)
  ctx.timed_out, ctx.cut_short, ctx._ridx, ctx._tick = false, false, nil, 0
  ctx._cut_fallback, ctx.carry_check, ctx._carry_blocked, ctx._away_pt, ctx._floorc, ctx._people = nil, nil, nil, nil, nil, nil
  local a = req.action
  local res
  if a == "add" then res = self:add(req, ctx)
  elseif a == "remove" then res = self:remove(req, ctx)
  elseif a == "rotate" then res = self:rotate(req, ctx)
  elseif a == "move" then res = self:move(req, ctx)
  elseif a == "rearrange" then res = self:rearrange(req, ctx)
  elseif a == "clear" then res = { action = "clear", placements = {}, notes = {}, say = "Cleared everything.", fixed = true }
  elseif a == "undo" then res = { action = "undo", placements = {}, notes = {}, say = "Undone.", fixed = true }
  else res = { action = "none", placements = {}, notes = {}, say = "" } end
  res.notes = res.notes or {}
  res.timed_out = res.timed_out or false
  res.clamped = res.clamped or false
  -- what the parser changed in the player's words (a repaired spelling), when it is made of library words only
  if type(req.notes) == "table" then
    for _, n in ipairs(req.notes) do
      if self:safe_note(n) then add_note(res, n) end
    end
  end
  if req.truncated and a ~= "none" then add_note(res, "that was long, so I read only the start of it") end
  if res.say and res.say ~= "" then
    -- a note the parser wrote, or this one, that the action's own sentence has not said yet
    for _, n in ipairs(res.notes) do
      if not res.say:lower():find(n:lower(), 1, true) then res.say = res.say .. " " .. n:sub(1, 1):upper() .. n:sub(2) .. "." end
    end
    res.say = res.say:sub(1, 1):upper() .. res.say:sub(2)
  end
  return res
end

-- exposed for the offline tests
Plan._judge, Plan._fit, Plan._real_room, Plan._top_frame = judge, fit, real_room, top_frame
Plan._riders_of, Plan._shape_of = riders_of, shape_of

return Plan
