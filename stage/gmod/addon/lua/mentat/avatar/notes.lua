-- avatar/notes.lua -- the summariser: raw journal events -> a handful of compact NOTES. PURE and deterministic (same events, same time -> same notes),
-- tested offline (test_gmod_director.py).
--
-- THE NOTES ARE WHAT A MODEL PROVIDER SEES, so they are built to say as little as will do:
--   * never a SteamID, a player's name, a wake name or anything a player typed except an object noun that is a plain word or two
--   * at most MAX_NOTES strings of at most MAX_CHARS characters, plain letters, digits and spaces
--   * facts (a small table of the same numbers) go to the bridge too, for the template callbacks; same rule
-- A journal event is a flat table { t = epoch seconds, k = kind, name?, n?, why?, cmd?, preset?, liquid?, tag?, src? }. Kinds:
--   add (name, n)       remove, undo, clear (n)       beam_ok (name, src)       beam_fail (name, why)
--   water (cmd, liquid) gravity (preset)              splash (n)                fall_water
--   session_start, session_end                        name_change, roast (level)   joke (tag)
local N = {}

N.MAX_NOTES = 8
N.MAX_CHARS = 100
N.ORDINALS = { "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth" }
N.NUMBERS = { [1] = "one", [2] = "two", [3] = "three", [4] = "four", [5] = "five", [6] = "six", [7] = "seven", [8] = "eight", [9] = "nine", [10] = "ten" }

--- An object noun that may be shown to a model: one to three plain words, letters and spaces only, not (part of) any name in `banned` (the player's nick,
--- their wake names, anyone's), else "something". Lower-cased.
function N.safe_noun(noun, banned)
  local s = tostring(noun or ""):lower():gsub("^%s+", ""):gsub("%s+$", "")
  if s == "" or #s > 24 or not s:match("^[a-z][a-z ]*$") then return "something" end
  local words = 0
  for _ in s:gmatch("%a+") do words = words + 1 end
  if words > 3 then return "something" end
  for _, b in ipairs(banned or {}) do
    b = tostring(b or ""):lower()
    if #b >= 3 and (s:find(b, 1, true) or b:find(s, 1, true)) then return "something" end
  end
  return s
end

local function plural(name, n)
  if n == 1 then return name end
  if name:match("[sx]$") or name:match("ch$") or name:match("sh$") then return name .. "es" end
  if name:match("[^aeiou]y$") then return name:sub(1, -2) .. "ies" end
  return name .. "s"
end

local function a_or_an(name) return (name:match("^[aeiou]") and "an " or "a ") .. name end
local function number_word(n) return N.NUMBERS[n] or tostring(n) end
local function times(n) return n == 1 and "once" or n == 2 and "twice" or (number_word(n) .. " times") end

local function day_key(t, date_fn) return date_fn("%Y-%m-%d", t) end

local function part_of_day(hour)
  if hour < 5 then return "night" elseif hour < 12 then return "morning" elseif hour < 18 then return "afternoon" end
  return "evening"
end

local function top_by_count(counts, order)
  local best, bn
  for _, name in ipairs(order) do
    if not bn or counts[name] > bn then best, bn = name, counts[name] end
  end
  return best, bn
end

--- Summarise `events` (oldest first). opts = { banned = { names to keep out }, date_fn = os.date-like (default os.date), tz = unused,
--- window = seconds of history that counts as "today" (default: since local midnight) }.
--- Returns { notes = { strings }, facts = { ... } }.
function N.summarise(events, now, opts)
  opts = opts or {}
  local date_fn = opts.date_fn or os.date
  local banned = opts.banned or {}
  local today = day_key(now, date_fn)
  local evs = {}
  for _, e in ipairs(events or {}) do
    if type(e) == "table" and type(e.t) == "number" and e.t <= now + 5 and day_key(e.t, date_fn) == today then evs[#evs + 1] = e end
  end
  local notes, facts = {}, {}
  if #evs == 0 then return { notes = notes, facts = facts } end

  local built, built_order = {}, {}          -- kind -> items placed today
  local asks = {}                            -- kind -> how many separate requests
  local fails, fail_order = {}, {}
  local undo_n, builds, streak, best_streak, last_build = 0, 0, 0, 0, nil
  local gravity, water_fill, pool_built, splashes, falls = nil, nil, false, 0, 0
  local session_start, last_action = nil, nil
  local jokes, joke_order = {}, {}
  local function add_built(name, n)
    if not built[name] then built[name] = 0; built_order[#built_order + 1] = name end
    built[name] = built[name] + n
  end
  for _, e in ipairs(evs) do
    local k = e.k
    if k == "session_start" then session_start = session_start or e.t
    elseif k == "add" or k == "beam_ok" then
      local name = N.safe_noun(e.name, banned)
      if name ~= "something" then
        add_built(name, math.max(1, math.floor(tonumber(e.n) or 1)))
        if k == "add" then asks[name] = (asks[name] or 0) + 1 end
        last_build = name
        builds = builds + 1
        streak = streak + 1
        best_streak = math.max(best_streak, streak)
        if name == "pool" or name == "lake" or name == "pond" then pool_built = true end
      end
      last_action = e.t
    elseif k == "beam_fail" then
      local name = N.safe_noun(e.name, banned)
      if name ~= "something" then
        if not fails[name] then fails[name] = 0; fail_order[#fail_order + 1] = name end
        fails[name] = fails[name] + 1
      end
      last_action = e.t
    elseif k == "undo" or k == "remove" or k == "clear" then
      undo_n = undo_n + 1
      streak = 0
      last_action = e.t
    elseif k == "water" then
      if e.cmd == "pool" or e.cmd == "lake" or e.cmd == "pond" or e.cmd == "sea" then pool_built = true end
      if e.liquid and tostring(e.liquid):match("^%a+$") and #tostring(e.liquid) <= 12 then water_fill = tostring(e.liquid):lower() end
      last_action = e.t
    elseif k == "gravity" then
      if e.preset and tostring(e.preset):match("^[%a ]+$") and #tostring(e.preset) <= 14 then gravity = tostring(e.preset):lower() end
      last_action = e.t
    elseif k == "splash" then splashes = splashes + math.max(1, math.floor(tonumber(e.n) or 1))
    elseif k == "fall_water" then falls = falls + 1
    elseif k == "joke" then
      local tag = tostring(e.tag or "")
      if tag:match("^%a+$") and #tag <= 12 then
        if not jokes[tag] then jokes[tag] = 0; joke_order[#joke_order + 1] = tag end
        jokes[tag] = jokes[tag] + 1
      end
    end
  end

  -- what was built today
  local top, top_n = top_by_count(built, built_order)
  if top then
    local parts = {}
    local shown = {}
    for _, name in ipairs(built_order) do shown[#shown + 1] = name end
    table.sort(shown, function(a, b) if built[a] ~= built[b] then return built[a] > built[b] end return a < b end)
    for i = 1, math.min(3, #shown) do
      local nm = shown[i]
      parts[#parts + 1] = built[nm] == 1 and a_or_an(nm) or (number_word(built[nm]) .. " " .. plural(nm, built[nm]))
    end
    local list = #parts == 1 and parts[1] or (#parts == 2 and (parts[1] .. " and " .. parts[2]) or (parts[1] .. ", " .. parts[2] .. " and " .. parts[3]))
    notes[#notes + 1] = "built " .. list .. " today"
    facts.top_item, facts.top_n, facts.last_build, facts.builds_n = shown[1], built[shown[1]], last_build, builds
    top, top_n = shown[1], built[shown[1]]
    if top_n >= 2 then
      local ord = N.ORDINALS[top_n] or (tostring(top_n) .. "th")
      notes[#notes + 1] = ord .. " " .. top .. " of the " .. part_of_day(tonumber(date_fn("%H", now)) or 12)
    end
    if (asks[top] or 0) >= 3 then notes[#notes + 1] = "asked for " .. a_or_an(top) .. " " .. number_word(asks[top]) .. " times" end
  end
  if pool_built then facts.pool_built = "yes" end
  if water_fill then facts.water_fill = water_fill; notes[#notes + 1] = "filled water with " .. water_fill end
  if gravity then facts.gravity = gravity; notes[#notes + 1] = "set gravity to " .. gravity end
  -- failures
  table.sort(fail_order, function(a, b) if fails[a] ~= fails[b] then return fails[a] > fails[b] end return a < b end)
  for i = 1, math.min(2, #fail_order) do
    local nm = fail_order[i]
    notes[#notes + 1] = nm .. " failed " .. times(fails[nm])
    if i == 1 then facts.fail_item, facts.fail_n = nm, fails[nm] end
  end
  if undo_n > 0 then facts.undo_n = undo_n; notes[#notes + 1] = "undid " .. number_word(undo_n) .. (undo_n == 1 and " thing" or " things") end
  -- what he already joked about (a callback must be fresh)
  if #joke_order > 0 then
    table.sort(joke_order, function(a, b) if jokes[a] ~= jokes[b] then return jokes[a] > jokes[b] end return a < b end)
    local parts = {}
    for i = 1, math.min(3, #joke_order) do parts[#parts + 1] = joke_order[i] end
    notes[#notes + 1] = "already joked about " .. table.concat(parts, ", ")
  end
  if best_streak >= 4 then notes[#notes + 1] = number_word(best_streak) .. " builds in a row without an undo" end
  if falls > 0 then notes[#notes + 1] = "jumped into water " .. times(falls) end
  if splashes >= 3 then notes[#notes + 1] = "made " .. number_word(splashes) .. " splashes" end
  -- time
  if last_action then
    local mins = math.max(0, math.floor((now - last_action) / 60))
    facts.minutes_since = mins
    if mins >= 5 then notes[#notes + 1] = "last action " .. mins .. " minutes ago" end
  end
  if session_start then
    local mins = math.max(0, math.floor((now - session_start) / 60))
    facts.session_min = mins
    if mins >= 10 then notes[#notes + 1] = "playing for " .. mins .. " minutes" end
  end
  -- caps and last clean: plain characters only
  local out = {}
  for _, s in ipairs(notes) do
    s = s:gsub("[^%w ,]", " "):gsub("%s+", " ")
    if #s > N.MAX_CHARS then s = s:sub(1, N.MAX_CHARS) end
    out[#out + 1] = s
    if #out >= N.MAX_NOTES then break end
  end
  return { notes = out, facts = facts }
end

--- The "what do you remember" answer when no model is reachable: a plain sentence from the notes (the bridge's callback templates make it Skippy-ish).
function N.plain_recall(res)
  if not res or #res.notes == 0 then return "Nothing yet. Build something and I'll make sure to remember it badly." end
  return "From my notes: " .. table.concat(res.notes, "; ") .. "."
end

return N
