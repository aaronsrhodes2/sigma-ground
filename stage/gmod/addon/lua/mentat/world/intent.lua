-- world/intent.lua -- the words that change or ask about the construct's physics ("set gravity to the moon", "zero gravity",
-- "what are the world's physics").  PURE Lua: no engine calls, so it is tested offline.
--
-- The holodeck glue consults this BEFORE the object parser (world.lua wraps Holodeck.Handle).  It claims a sentence only when
-- EVERY clause of it is a world-physics clause, so "add a couch and zero gravity" still goes to the object parser untouched, and
-- "drag the couch" or "gravity gun" are never taken for physics.  Typos are tolerated: a word within one edit (two for long words)
-- of the vocabulary counts.
--
-- Real-world physics only: the only things that can be asked for are real conditions of real places (Earth, the Moon, Mars,
-- Jupiter, free fall).  "Half gravity", "more air drag" and "no air drag" are recognised so Skippy can say plainly that the world
-- does not do that, and nothing changes.
--
--   parse(text) -> nil | { intents... }   each: { kind = "gravity_preset", name = "moon" }
--       | { kind = "not_real", what = "gravity" | "drag" }   (a gameplay knob: refused)
--       | { kind = "query", what = "world" | "gravity" | "drag" }
local M = {}

local function lev(a, b, cap)
  if a == b then return 0 end
  local la, lb = #a, #b
  if math.abs(la - lb) > cap then return cap + 1 end
  local prev, cur = {}, {}
  for j = 0, lb do prev[j] = j end
  for i = 1, la do
    cur[0] = i
    local rowmin = cur[0]
    for j = 1, lb do
      local cost = (a:byte(i) == b:byte(j)) and 0 or 1
      cur[j] = math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
      if cur[j] < rowmin then rowmin = cur[j] end
    end
    if rowmin > cap then return cap + 1 end
    prev, cur = cur, prev
  end
  return prev[lb]
end

--- Does token `w` match the vocabulary word `v`, allowing typos (none for short words; the first letter must be right)?
local function is(w, v)
  if w == v then return true end
  local n = #v
  local cap = n >= 7 and 2 or (n >= 4 and 1 or 0)
  if w:sub(1, 1) ~= v:sub(1, 1) then return false end
  return cap > 0 and lev(w, v, cap) <= cap
end

local function any(w, list) for _, v in ipairs(list) do if is(w, v) then return true end end return false end
local function has(ws, list) for _, w in ipairs(ws) do if any(w, list) then return w end end return nil end

local PRESET_WORDS = {
  moon = { "moon", "moons", "lunar", "luna" },
  mars = { "mars", "martian" },
  jupiter = { "jupiter", "jovian" },
  earth = { "earth", "earths", "terra" },
}
local ZERO = { "zero", "weightless", "none", "nothing", "null", "off", "no", "freefall" }
local NORMAL = { "normal", "default", "usual", "reset", "regular", "ordinary", "restore", "original", "standard", "real", "realistic", "actual" }
local MORE = { "more", "higher", "increase", "stronger", "heavier", "thicker", "greater", "extra", "double", "twice", "triple" }
local LESS = { "less", "lower", "reduce", "weaker", "lighter", "thinner", "decrease", "fewer", "half", "quarter" }
local QUERY_FIRST = { what = true, whats = true, which = true, show = true, tell = true, read = true, list = true, describe = true,
  status = true, report = true, explain = true, check = true, current = true, does = true, ["do"] = true, is = true }
local NOT_WORDS = { gravy = true, gravel = true, graves = true, grave = true }        -- near the vocabulary but never meant

local function words(text)
  text = tostring(text or ""):lower()
  text = text:gsub("([%a])'s%f[%A]", "%1"):gsub("'", "")                       -- world's -> world
  text = text:gsub("%%", " percent "):gsub("m/s%^?2", " mps "):gsub("m/s", " mps ")
  local out = {}
  for w in text:gmatch("[%a%d%.]+") do
    w = w:gsub("^%.+", ""):gsub("%.+$", "")
    if w ~= "" and w ~= "skippy" and w ~= "please" and w ~= "hey" then out[#out + 1] = w end
  end
  return out
end

--- Split a sentence into clauses at "and", "then", "also", commas, and "but".
local function clauses(text)
  local t = " " .. tostring(text or ""):lower():gsub("[,;]", " , ") .. " "
  local out, cur = {}, {}
  for w in t:gmatch("%S+") do
    if w == "," or w == "and" or w == "then" or w == "also" or w == "but" or w == "plus" then
      if #cur > 0 then out[#out + 1] = table.concat(cur, " "); cur = {} end
    else
      cur[#cur + 1] = w
    end
  end
  if #cur > 0 then out[#out + 1] = table.concat(cur, " ") end
  return out
end

local function is_gravity(w) return is(w, "gravity") or w == "grav" or w == "gravitation" end
local function is_drag_word(w) return is(w, "drag") or is(w, "friction") or is(w, "resistance") end

local function parse_clause(c)
  local ws = words(c)
  if #ws == 0 or #ws > 14 then return nil end
  for _, w in ipairs(ws) do if NOT_WORDS[w] then return nil end end
  local gi, di, pi
  for i, w in ipairs(ws) do
    if is_gravity(w) then gi = gi or i end
    if is(w, "physics") or is(w, "constants") then pi = pi or i end
    if is_drag_word(w) then di = di or i end
  end
  local air = false
  for _, w in ipairs(ws) do if w == "air" or w == "wind" then air = true end end
  local asked = QUERY_FIRST[ws[1]] == true
  if ws[1] == "how" and (ws[2] == "strong" or ws[2] == "much" or ws[2] == "heavy" or ws[2] == "is" or ws[2] == "big") then asked = true end
  if ws[1] == "can" or ws[1] == "could" or ws[1] == "would" or ws[1] == "will" then           -- "can you set gravity ..." is a request
    local k = 2
    while ws[k] and (ws[k] == "you" or ws[k] == "we" or ws[k] == "please" or ws[k] == "i") do k = k + 1 end
    if ws[k] == "tell" or ws[k] == "show" or ws[k] == "read" then asked = true end
  end
  local has_number = false
  for _, w in ipairs(ws) do if tonumber(w) or w:match("^%d+%.?%d*x$") or w:match("^x%d") then has_number = true end end

  -- ---- gravity
  if gi then
    for _, w in ipairs(ws) do
      if w == "gun" or w == "guns" or w == "well" or w == "wells" then return nil end          -- "gravity gun" is a weapon
    end
    local preset
    for name, list in pairs(PRESET_WORDS) do if has(ws, list) then preset = name end end
    local z, nrm = has(ws, ZERO), has(ws, NORMAL)
    if asked then return { kind = "query", what = "gravity" } end
    if has_number or has(ws, MORE) or has(ws, LESS) then return { kind = "not_real", what = "gravity" } end
    if z and not preset then return { kind = "gravity_preset", name = "zero" } end
    if preset then return { kind = "gravity_preset", name = preset } end
    if nrm then return { kind = "gravity_preset", name = "earth" } end
    return nil
  end

  -- ---- air drag (never plain "drag": "drag the couch" is a move)
  if di and (air or has(ws, ZERO) or has(ws, MORE) or has(ws, LESS) or has(ws, NORMAL) or has_number or asked) then
    if (not air) and ws[di] == "drag" then
      local near = ws[di - 1] or ""
      local qualified = any(near, ZERO) or any(near, MORE) or any(near, LESS) or any(near, NORMAL) or near == "percent"
      if not qualified and not (asked and #ws <= 5) then return nil end
      for _, w in ipairs(ws) do if w == "couch" or w == "chair" or w == "table" or w == "it" or w == "them" or w == "there" then return nil end end
    end
    if asked then return { kind = "query", what = "drag" } end
    if has(ws, ZERO) or has(ws, MORE) or has(ws, LESS) or has_number or has(ws, { "without", "remove", "vacuum", "disable" }) then
      return { kind = "not_real", what = "drag" }
    end
    if has(ws, NORMAL) then return { kind = "query", what = "drag" } end
    return nil
  end

  -- ---- "what are the world's physics"
  if pi then
    local w = has(ws, { "world", "construct", "holodeck", "game", "room", "stage" })
    if w and (asked or ws[1] == "world" or ws[1] == "physics") then return { kind = "query", what = "world" } end
  end
  return nil
end

--- The intents a sentence carries, or nil when any of it is something else (the object parser then gets the whole sentence).
function M.parse(text)
  if type(text) ~= "string" or #text > 200 then return nil end
  local out = {}
  for _, c in ipairs(clauses(text)) do
    if #words(c) > 0 then                                  -- ("Skippy," alone is no clause)
      local it = parse_clause(c)
      if not it then return nil end
      out[#out + 1] = it
    end
    if #out > 3 then return nil end
  end
  if #out == 0 then return nil end
  return out
end

M.lev = lev
return M
