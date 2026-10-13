-- water/intent.lua -- the words for water: "add a pool", "a 10 metre pond", "make a lake", "fill it with mercury", "make the pool deeper",
-- "drain it", "remove the water", "drop a ball in the pool", "what floats", "what is the density of this water".  PURE Lua: no engine
-- calls, so it is tested offline.
--
-- The holodeck glue consults this BEFORE the object parser (water.lua wraps Holodeck.Handle, as world.lua does for physics).  It claims a
-- sentence only when it carries a water-body noun (pool, pond, lake, sea, tank) or the word water in a water sentence.  "A pool table" and
-- "swim" are never taken for a pool.  Typos are tolerated (one edit for ordinary words, two for long ones; the first letter must be right).
--
--   parse(text) -> nil | intent, one of
--     { kind = "add",    noun, shape, l, w, depth (metres, nil = the default), liquid (word) , count, assumed = {texts}, span = {s, e} }
--     { kind = "drop",   span = {s, e}, verb_span = {s, e} }            "drop a ball in the pool": rewrite = verb -> put, span -> on %WATER%
--     { kind = "resize", dim = "depth"|"size"|"width"|"length", factor = n | absolute = metres, target }
--     { kind = "liquid", liquid (word) , target }
--     { kind = "level",  level = 0 | 1, target }                       drain / fill up
--     { kind = "remove", target }
--     { kind = "query",  what = "floats"|"density"|"depth"|"liquid"|"which", item = word|nil }
local M = {}

local World = (include or function(p) return dofile(p) end)("mentat/world/intent.lua")
local lev = World.lev

local function swapped(w, v)
  if #w ~= #v or #w < 4 then return false end
  for i = 1, #w - 1 do
    if w:sub(1, i - 1) .. w:sub(i + 1, i + 1) .. w:sub(i, i) .. w:sub(i + 2) == v then return true end
  end
  return false
end

local function is(w, v)
  if w == v then return true end
  if swapped(w, v) and w:sub(1, 1) == v:sub(1, 1) then return true end
  local n = #v
  local cap = n >= 7 and 2 or (n >= 4 and 1 or 0)
  if w:sub(1, 1) ~= v:sub(1, 1) then return false end
  return cap > 0 and lev(w, v, cap) <= cap
end
local function any(w, list) for _, v in ipairs(list) do if is(w, v) then return v end end return nil end

local NOUNS = {
  pool = { "pool", "pools" }, pond = { "pond", "ponds" }, lake = { "lake", "lakes" }, sea = { "sea", "seas" },
  ocean = { "ocean", "oceans" },
}
-- ("tank" is not here: it is a military, fuel or fish-tank object for the object parser and the SkippyVerse inventory)
local NOUN_ORDER = { "pool", "pond", "lake", "ocean", "sea" }
local NOUN_NOTYPO = { sea = true }        -- short words: "see", "task", "tan" stay themselves
local NOT_AFTER = { table = true, tables = true, cue = true, cues = true, ball = true, balls = true, hall = true, party = true, boy = true, noodle = true,
                    noodles = true, cleaner = true, toy = true, toys = true, float = true, floats = true, shell = true, shells = true, horse = true, gull = true, weed = true, food = true, bed = true, sick = true, salt = true,
                    side = true, port = true, wall = true, level = true, creature = true, monster = true, bass = true }
local NOT_BEFORE = { car = true, gene = true, talent = true, motor = true, think = true, swim = true }

local ADD_VERBS = { "add", "make", "create", "build", "spawn", "summon", "place", "put", "give", "get", "dig", "need", "want", "have", "bring", "construct",
                    "conjure", "generate", "install", "form", "show", "set" }
local DROP_VERBS = { "drop", "throw", "toss", "dump", "chuck", "lob", "fling", "sink", "float", "put", "place", "add", "pour", "let" }
local IN_PREPS = { "in", "into", "inside", "within", "over", "above", "onto", "on", "to", "down" }
local ARTICLES = { a = true, an = true, the = true, some = true, one = true, another = true, this = true, that = true, my = true, our = true, new = true }
local SIZE_WORDS = {
  big = 1.5, bigger = 1.5, large = 1.5, larger = 1.5, huge = 3, giant = 3, enormous = 3, massive = 3, vast = 3, wide = 1.5, long = 1.5, immense = 3,
  small = 0.5, smaller = 0.5, little = 0.5, tiny = 0.25, mini = 0.25, narrow = 0.5, short = 0.5,
}
local DEPTH_WORDS = { deep = 2, deeper = 2, shallow = 0.5, shallower = 0.5, bottomless = 4 }
local SHAPE_WORDS = { round = "round", circular = "round", circle = "round", oval = "round", rectangular = "rect", square = "rect" }
local FLUFF = { swimming = true, paddling = true, garden = true, fish = true, koi = true, water = true, ["of"] = true, plain = true, nice = true,
                lovely = true, pretty = true, real = true, proper = true, full = true, filled = true, with = true, sized = true, olympic = false, little = true,
                lap = true, hot = false }
local UNIT_M = { m = 1, metre = 1, metres = 1, meter = 1, meters = 1, ft = 0.3048, foot = 0.3048, feet = 0.3048, cm = 0.01, centimetre = 0.01,
                 centimetres = 0.01, centimeter = 0.01, centimeters = 0.01, yd = 0.9144, yard = 0.9144, yards = 0.9144, km = 1000, kilometre = 1000,
                 kilometres = 1000, inch = 0.0254, inches = 0.0254 }
local NUMBER_WORDS = { one = 1, two = 2, three = 3, four = 4, five = 5, six = 6, seven = 7, eight = 8, nine = 9, ten = 10, twenty = 20, thirty = 30,
                       fifty = 50, hundred = 100 }

-- ------------------------------------------------------------------------------------------------ tokens
--- { { w = word, s = first byte, e = last byte, num = number|nil, unit = metres per unit|nil } ... } of the lower-cased text
local function tokenize(text)
  local low = text:lower()
  local out = {}
  for s, w, e in low:gmatch("()([%w%.'%-]+)()") do
    w = w:gsub("^[%.'%-]+", ""):gsub("[%.'%-]+$", "")
    if w ~= "" then
      e = e - 1
      local n, u = w:match("^(%d+%.?%d*)(%a+)$")
      if n and (UNIT_M[u] or u == "x") then
        out[#out + 1] = { w = n, s = s, e = e, num = tonumber(n) }
        if UNIT_M[u] then out[#out + 1] = { w = u, s = s, e = e, unit = UNIT_M[u] } end
      elseif w:match("^%d+%.?%d*x%d") then                                            -- 10x5x2
        for part in w:gmatch("[^x]+") do out[#out + 1] = { w = part, s = s, e = e, num = tonumber(part) } end
      elseif tonumber(w) then
        out[#out + 1] = { w = w, s = s, e = e, num = tonumber(w) }
      else
        out[#out + 1] = { w = w, s = s, e = e, unit = UNIT_M[w], numword = NUMBER_WORDS[w] }
      end
    end
  end
  return out
end
M.tokenize = tokenize

local function noun_of(w)
  if NOT_AFTER[w] then return nil end
  for _, n in ipairs(NOUN_ORDER) do
    for _, v in ipairs(NOUNS[n]) do
      if w == v then return n end
      if not NOUN_NOTYPO[n] and #v >= 4 and is(w, v) then return n end
    end
  end
  return nil
end

--- A liquid word at tokens[i] (one or two words): returns the word as the constants table knows it, and how many tokens it used.
local function liquid_at(toks, i, C)
  local a, b = toks[i] and toks[i].w, toks[i + 1] and toks[i + 1].w
  if a and b then
    local two = a .. " " .. b
    local key, bad = C.find_liquid(two)
    if key or bad then return two, 2 end
  end
  if a then
    local key, bad = C.find_liquid(a)
    if key or bad then return a, 1 end
    -- typos in the longer liquid words
    if #a >= 6 then
      for _, w in ipairs({ "seawater", "saltwater", "mercury", "glycerin", "glycerol", "ethanol", "alcohol" }) do
        if is(a, w) then return w, 1 end
      end
    end
  end
  return nil, 0
end

local function find(toks, pred, from, to)
  for i = from or 1, to or #toks do if pred(toks[i], i) then return i end end
  return nil
end

local BREAKERS = { ["and"] = true, ["then"] = true, ["also"] = true, ["but"] = true, ["plus"] = true }
local PLACE_PREPS = { ["in"] = true, on = true, near = true, over = true, next = true, behind = true, by = true, at = true, there = true, here = true,
                      beside = true, ["in"] = true, front = true, left = true, right = true, between = true, under = true, against = true, along = true,
                      close = true, around = true, ["to"] = true, ["for"] = true, outside = true, inside = true, across = true, opposite = true }

-- ------------------------------------------------------------------------------------------------ numbers and sizes
--- Read the size words and numbers in tokens[from..to] (modifiers of one water phrase).
--- Returns { l, w, depth, shape, factor_size, factor_depth, assumed = {...} }
local function read_size(toks, from, to, noun_i)
  local r = { assumed = {} }
  local nums = {}
  local i = from
  while i <= to do
    local t = toks[i]
    local nxt = toks[i + 1]
    local val = t.num or (t.numword and nxt and nxt.unit and t.numword)
    if val and i == noun_i - 1 and not (nxt and nxt.unit) then val = nil end         -- "two pools": a count, not a size
    if val and i ~= noun_i then
      local unit = nxt and nxt.unit
      local consumed = 0
      if unit then consumed = 1 end
      nums[#nums + 1] = { v = val * (unit or 1), unit = unit, at = i, deep = false }
      -- "3 metres deep" / "3 deep"
      local after = toks[i + 1 + consumed]
      if after and (after.w == "deep" or after.w == "depth" or after.w == "high") then nums[#nums].deep = true end
      if after and (after.w == "wide" or after.w == "across") then nums[#nums].role = "w" end
      if after and (after.w == "long" or after.w == "diameter") then nums[#nums].role = "l" end
      if after and (after.w == "wide" or after.w == "long" or after.w == "deep" or after.w == "high") then nums[#nums].consumed_adj = true end
      -- a bare number without a unit is metres only when it is followed by a size word or "by"/"x"
      nums[#nums].bare = unit == nil
      i = i + 1 + consumed
    else
      local w = t.w
      local prev = toks[i - 1]
      local unit_adj = (w == "long" or w == "wide" or w == "deep" or w == "high" or w == "short") and prev and (prev.unit or prev.num)
      if unit_adj then
        -- a measure ("30 feet long"), not an adjective
      elseif SIZE_WORDS[w] then r.factor_size = (r.factor_size or 1) * SIZE_WORDS[w]; r.size_word = w
      elseif DEPTH_WORDS[w] then r.factor_depth = (r.factor_depth or 1) * DEPTH_WORDS[w]; r.depth_word = w
      elseif SHAPE_WORDS[w] then r.shape = SHAPE_WORDS[w] end
      i = i + 1
    end
  end
  -- numbers: "N by M [by D]" / "N x M" -> length, width, depth; a lone number is the length (or depth if marked deep)
  local plain = {}
  for _, n in ipairs(nums) do
    if n.deep then r.depth = n.v
    elseif n.role == "w" then r.w = n.v
    elseif n.role == "l" then r.l = n.v
    elseif not (n.bare and n.v > 1000) then plain[#plain + 1] = n.v end
  end
  if #plain >= 1 and not r.l then r.l = plain[1]; table.remove(plain, 1) end
  if #plain >= 1 and not r.w then r.w = plain[1]; table.remove(plain, 1) end
  if #plain >= 1 and not r.depth then r.depth = plain[1] end
  if r.depth then r.factor_depth = nil end
  if r.l or r.w then r.factor_size = nil end
  if r.factor_size then r.assumed[#r.assumed + 1] = "I read '" .. tostring(r.size_word) .. "' as " .. string.format("%.3g", r.factor_size) .. " times the usual size" end
  if r.factor_depth then r.assumed[#r.assumed + 1] = "I read '" .. tostring(r.depth_word) .. "' as " .. string.format("%.3g", r.factor_depth) .. " times the usual depth" end
  return r
end

-- ------------------------------------------------------------------------------------------------ the sentence
local function is_add_verb(w) return any(w, ADD_VERBS) ~= nil or w == "ad" or w == "adds" or w == "addd" end

--- Clause words before index i that make it a request for the thing (add/make/...), or "I want a pool" style.
local function wants_thing(toks, i)
  for k = 1, i - 1 do
    local w = toks[k].w
    if is_add_verb(w) or w == "like" or w == "gimme" or w == "dig" or w == "lets" or w == "let's" then return true end
    if BREAKERS[w] then
      -- "add a couch and a pool": the verb carries over
    end
  end
  return false
end

local function target_of(toks, from)
  for i = from or 1, #toks do
    local n = noun_of(toks[i].w)
    if n then return n end
  end
  return nil
end

local function has_word(toks, list, from, to)
  for i = from or 1, math.min(to or #toks, #toks) do if any(toks[i].w, list) then return i end end
  return nil
end

function M.parse(text, C)
  if type(text) ~= "string" or #text > 240 then return nil end
  C = C or (include or function(p) return dofile(p) end)("mentat/water/constants.lua")
  local toks = tokenize(text)
  if #toks == 0 or #toks > 40 then return nil end
  local W = {}
  for i, t in ipairs(toks) do W[i] = t.w end

  -- never "pool table", "pool party", "car pool"
  local noun_i, noun
  for i, t in ipairs(toks) do
    local n = noun_of(t.w)
    if n and not (toks[i + 1] and NOT_AFTER[toks[i + 1].w]) and not (toks[i - 1] and NOT_BEFORE[toks[i - 1].w]) then noun_i, noun = i, n; break end
  end
  local water_i = find(toks, function(t) return t.w == "water" or is(t.w, "water") and #t.w >= 5 end)
  local first = W[1]

  -- ---- questions (what floats, density, depth, liquid)
  local asked = (first == "what" or first == "whats" or first == "which" or first == "how" or first == "does" or first == "do" or first == "is" or first == "will"
    or first == "can" or first == "tell" or first == "show" or first == "list" or first == "are" or first == "would")
  if asked and (water_i or noun_i or has_word(toks, { "float", "floats", "floating", "sink", "sinks", "sinking", "buoyant", "buoyancy" })) then
    local floatw = has_word(toks, { "float", "floats", "floating", "buoyant", "buoyancy" })
    local sinkw = has_word(toks, { "sink", "sinks", "sinking" })
    local dens = has_word(toks, { "density", "dense", "heavy", "heavier", "viscosity", "viscous", "thick" })
    local deep = has_word(toks, { "deep", "depth" }) and (first == "how" or W[2] == "deep")
    local liq = has_word(toks, { "liquid", "fluid", "filled", "full" }) or (find(toks, function(t, i) return liquid_at(toks, i, C) ~= nil end) and true)
    if floatw or sinkw then
      -- "does the ball float" -> item = the word between does/will/do and float
      local item
      for i = 2, #toks do
        local w = toks[i].w
        if any(w, { "float", "floats", "sink", "sinks", "floating", "sinking" }) then
          for k = i - 1, 2, -1 do
            local cand = toks[k].w
            if not ARTICLES[cand] and cand ~= "everything" and cand ~= "anything" and cand ~= "things" and cand ~= "objects" and cand ~= "here" and cand ~= "there"
               and cand ~= "and" and cand ~= "or" and cand ~= "in" and cand ~= "the" and cand ~= "what" and cand ~= "which" and cand ~= "will" and cand ~= "does" and cand ~= "do" then
              item = cand; break
            end
          end
          break
        end
      end
      if first == "what" or first == "which" or first == "list" or first == "show" then
        if not item or item == "of" then return { kind = "query", what = "which" } end
      end
      return { kind = "query", what = "floats", item = item }
    end
    if dens and (water_i or noun_i or liq) then return { kind = "query", what = "density" } end
    if deep and noun_i then return { kind = "query", what = "depth" } end
    if (first == "what" or first == "whats") and (liq or water_i) and (W[2] == "is" or W[2] == "liquid" or W[2] == "whats" or W[2] == "in") then
      return { kind = "query", what = "liquid" }
    end
    if first == "what" and has_word(toks, { "floats", "float" }) then return { kind = "query", what = "which" } end
    return nil
  end

  -- ---- drain / empty / fill up
  if has_word(toks, { "drain", "empty", "deplete", "evaporate", "dry" }, 1, 3) and (noun_i or water_i or W[2] == "it" or W[3] == "it" or W[2] == "them") then
    return { kind = "level", level = 0, target = noun, explicit = (noun ~= nil or water_i ~= nil) }
  end
  if (first == "let" or first == "get") and has_word(toks, { "out" }) and water_i then return { kind = "level", level = 0, target = noun, explicit = (noun ~= nil or water_i ~= nil) } end
  if has_word(toks, { "refill", "replenish", "top" }, 1, 2) then return { kind = "level", level = 1, target = noun, explicit = (noun ~= nil or water_i ~= nil) } end

  -- ---- fill it with X / change the water to X / make it mercury
  local fill_i = has_word(toks, { "fill", "filling", "replace", "change", "switch", "swap", "turn", "convert" }, 1, 3)
  if fill_i and not (noun_i and wants_thing(toks, noun_i) and not water_i) then
    local lq, lq_n
    for i = fill_i + 1, #toks do
      local w, n = liquid_at(toks, i, C)
      if w and not (w == "water" and toks[i - 1] and (toks[i - 1].w == "the" or toks[i - 1].w == "of") and not (toks[i + 1])) then
        -- "fill it with water" is a liquid change to water; "fill the water" is nonsense; both fine
        lq, lq_n = w, n
      end
      if lq then break end
    end
    local what = W[fill_i + 1]
    if lq then return { kind = "liquid", liquid = lq, target = noun, explicit = (noun ~= nil or water_i ~= nil) } end
    if (what == "it" or what == "up" or what == "the" or what == "them") and (not lq) and (noun_i or what == "it" or what == "up") then
      return { kind = "level", level = 1, target = noun, explicit = (noun ~= nil or water_i ~= nil) }
    end
    -- "fill it with <unknown>": a liquid the vocabulary does not know -> pass the word on so it is refused by name
    local with_i = find(toks, function(t) return t.w == "with" or t.w == "to" or t.w == "into" end, fill_i + 1)
    if with_i and toks[with_i + 1] and not noun_of(toks[with_i + 1].w) then
      local k = with_i + 1
      while toks[k] and ARTICLES[toks[k].w] do k = k + 1 end
      if toks[k] and not noun_of(toks[k].w) and (noun_i or W[fill_i + 1] == "it" or W[fill_i + 1] == "the") then
        return { kind = "liquid", liquid = toks[k].w, target = noun, explicit = noun ~= nil }
      end
    end
  end
  if first == "make" and (W[2] == "it" or W[2] == "the") then
    -- "make it mercury" / "make the pool deeper"
    local lq
    for i = 3, #toks do
      local w = liquid_at(toks, i, C)
      if w then lq = w; break end
    end
    if lq and not has_word(toks, { "deeper", "bigger", "larger", "wider", "longer", "shallower", "smaller" }) then return { kind = "liquid", liquid = lq, target = noun, explicit = (noun ~= nil or water_i ~= nil) } end
  end

  -- ---- resize: deeper / bigger / shallower ...
  if noun_i or W[2] == "it" or W[3] == "it" or water_i then
    local deeper = has_word(toks, { "deeper", "deepen", "shallower" }, 1, #toks)
    local sizew = has_word(toks, { "bigger", "larger", "wider", "longer", "smaller", "expand", "enlarge", "grow", "shrink", "widen", "lengthen" }, 1, #toks)
    if (deeper or sizew) and not wants_thing(toks, noun_i or 1) or ((deeper or sizew) and first == "make") then
      local idx = deeper or sizew
      local w = W[idx]
      local shrink = is(w, "shallower") or is(w, "smaller") or w == "shrink"
      local f = shrink and (1 / 1.5) or 1.5
      local lead = toks[idx - 1] and toks[idx - 1].w
      if lead == "much" or lead == "far" or lead == "a" and toks[idx - 2] and toks[idx - 2].w == "lot" then f = shrink and (1 / 2.25) or 2.25 end
      if lead == "twice" or lead == "double" then f = shrink and 0.25 or 4 end
      if lead == "slightly" or lead == "bit" or lead == "little" then f = shrink and (1 / 1.2) or 1.2 end
      local dim = deeper and "depth" or ((is(w, "wider") or is(w, "widen")) and "width" or ((is(w, "longer") or is(w, "lengthen")) and "length" or "size"))
      -- "N metres deeper" is not a thing; "to N metres deep" handled below
      return { kind = "resize", dim = dim, factor = f, target = noun, explicit = (noun ~= nil or water_i ~= nil) }
    end
    -- "make the pool 3 metres deep" / "set the depth to 4 m"
    local di = find(toks, function(t) return t.w == "deep" or t.w == "depth" end)
    if di then
      local val
      for k = 1, #toks do
        if toks[k].num then
          local unit = toks[k + 1] and toks[k + 1].unit or 1
          val = toks[k].num * unit
          break
        end
      end
      if val and not (noun_i and wants_thing(toks, noun_i) and first ~= "make" and first ~= "set") then
        return { kind = "resize", dim = "depth", absolute = val, target = noun, explicit = (noun ~= nil or water_i ~= nil) }
      end
    end
  end

  -- ---- drop X in the pool
  do
    local drop_i = has_word(toks, DROP_VERBS, 1, 2)
    if drop_i and (noun_i or water_i) then
      local tgt_i = noun_i or water_i
      -- a preposition right before the (article +) water word
      local k = tgt_i - 1
      while k > 0 and (ARTICLES[toks[k].w]) do k = k - 1 end
      local prep = k > drop_i and toks[k] and any(toks[k].w, IN_PREPS)
      -- "add a pool" has no preposition and never lands here; "put a boat on the lake" does
      if prep and k > drop_i + 0 and not (drop_i == k) then
        -- the object between the verb and the preposition must not itself be the water noun
        local between_noun = false
        for m = drop_i + 1, k - 1 do if noun_of(toks[m].w) then between_noun = true end end
        if not between_noun and k - drop_i >= 1 then
          local first_tok = toks[drop_i]
          local phrase_start = toks[k].s
          local phrase_end = toks[tgt_i].e
          -- take "pool" and a following "water"
          if noun_i and water_i and water_i == noun_i + 1 then phrase_end = toks[water_i].e end
          return { kind = "drop", span = { phrase_start, phrase_end }, verb_span = { first_tok.s, first_tok.e }, target = noun }
        end
      end
    end
  end

  -- ---- remove / delete / clear / get rid of
  do
    local rm = has_word(toks, { "remove", "delete", "destroy", "clear", "erase", "banish", "dismiss", "undo", "dispose" }, 1, 3)
    local rid = first == "get" and W[2] == "rid"
    if (rm or rid) and (noun_i or water_i) then
      return { kind = "remove", target = noun, explicit = true }
    end
  end

  -- ---- add a pool
  if noun_i and (wants_thing(toks, noun_i) or ARTICLES[W[1]] or noun_of(W[1]) or first == "pool" or first == "lake" or first == "pond") then
    -- the phrase: walk left over modifiers
    local s_i = noun_i
    local k = noun_i - 1
    while k >= 1 do
      local t = toks[k]
      local w = t.w
      local ok = ARTICLES[w] or SIZE_WORDS[w] or DEPTH_WORDS[w] or SHAPE_WORDS[w] or FLUFF[w] or t.num or t.unit or w == "by" or w == "x" or w == "deep"
        or (w ~= "and" and liquid_at(toks, k, C) ~= nil) or NUMBER_WORDS[w] ~= nil
        or any(w, { "bigger", "larger", "deeper", "smaller" })
      if not ok then break end
      s_i = k
      k = k - 1
    end
    -- the size/liquid words on the right: "pool 10 metres long", "pool of mercury", "pool filled with oil", "pool that is 3 metres deep"
    local e_i = noun_i
    local liquid, liquid_i
    local j = noun_i + 1
    while j <= #toks do
      local t = toks[j]
      local w = t.w
      if BREAKERS[w] or PLACE_PREPS[w] and w ~= "to" then break end
      local lw, ln = liquid_at(toks, j, C)
      if lw and (W[j - 1] == "of" or W[j - 1] == "with" or W[j - 1] == "full" or W[j - 1] == "filled" or W[j - 1] == "in") then
        liquid, liquid_i = lw, j
        e_i = j + ln - 1
        j = j + ln
      elseif w == "of" or w == "with" or w == "filled" or w == "full" or w == "that" or w == "which" or w == "is" or w == "thats" or w == "being" or w == "a" or w == "an" or w == "the" then
        e_i = j
        j = j + 1
      elseif t.num or t.unit or w == "deep" or w == "long" or w == "wide" or w == "by" or w == "x" or w == "across" or w == "diameter" or w == "high" then
        e_i = j
        j = j + 1
      else
        break
      end
    end
    -- trim trailing glue ("of", "with", "a", "the", "is") that carried nothing
    while e_i > noun_i do
      local w = W[e_i]
      if w == "of" or w == "with" or w == "filled" or w == "full" or w == "that" or w == "which" or w == "is" or w == "thats" or w == "a" or w == "an" or w == "the" or w == "being" then
        e_i = e_i - 1
      else
        break
      end
    end
    -- the liquid before the noun ("a mercury pool", "a water pool")
    if not liquid then
      for m = s_i, noun_i - 1 do
        local lw = liquid_at(toks, m, C)
        if lw then liquid = lw; break end
      end
    end
    local sz = read_size(toks, s_i, e_i, noun_i)
    -- a count: "two pools"
    local count = 1
    for m = s_i, noun_i - 1 do
      local t = toks[m]
      if m == noun_i - 1 and (t.numword or t.num) then count = t.numword or t.num end
    end
    local liquid_defaulted = liquid == nil
    if liquid_at(toks, s_i, C) == "water" and noun == nil then liquid = "water" end
    -- one more thing: the player's word for "sea"/"ocean"
    return {
      kind = "add", noun = noun, shape = sz.shape, l = sz.l, w = sz.w, depth = sz.depth, factor_size = sz.factor_size, factor_depth = sz.factor_depth,
      liquid = liquid, count = count, assumed = sz.assumed, span = { toks[s_i].s, toks[e_i].e },
    }
  end
  return nil
end

--- Rewrite a claimed sentence for the object parser: the water phrase becomes `noun` ("add a 10 metre pool next to me" -> "add a poolaa next to me").
function M.rewrite_add(text, intent, noun)
  local s, e = intent.span[1], intent.span[2]
  return text:sub(1, s - 1) .. "a " .. noun .. text:sub(e + 1)
end

--- "drop a ball in the pool" -> "put a ball on <noun>".
function M.rewrite_drop(text, intent, noun)
  local vs, ve = intent.verb_span[1], intent.verb_span[2]
  local s, e = intent.span[1], intent.span[2]
  local out = text:sub(1, vs - 1) .. "put" .. text:sub(ve + 1, s - 1) .. "on the " .. noun .. text:sub(e + 1)
  return out
end

return M
