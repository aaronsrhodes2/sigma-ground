-- parse.lua -- turns what a person says to the holodeck into a structured request. Pure Lua (no GMod globals), so it is
-- unit-tested offline. It NEVER asks a question: every sentence resolves to the most sensible reading, and what it did
-- not understand is reported in `request.unknown` so the caller can say so while still acting on the rest.
--
--   local parser = Parse.new(catalog);  local req = parser:parse("computer, add three red couches facing me")
--
-- Result:
--   { action = "add"|"remove"|"clear"|"undo"|"move"|"rotate"|"rearrange"|"none",
--     items = { { kind="couch", count=3, adjectives={"red"}, word="couches", definite=false, approx=nil } },
--     arrangement = "unspecified"|"row"|"circle"|"around"|"facing_each_other"|"facing_me"|"u_shape"|"l_shape"|"grid"|
--                   "against_wall"|"beside"|"in_front_of"|"behind"|"scattered"|"stacked"|"corner"|"on",
--     target = "" | "me" | <kind> | "wall"|"window"|"door"|"here"|"last",
--     facing = nil | <target>   (a second relation: "in a row facing the tv"),
--     scene = "" | <recipe name>, unknown = { "dragon", ... }, notes = { ... }, raw = "<the sentence>" }
-- count 0 means "everything of that kind that exists".
local Parse = {}
Parse.__index = Parse

-- ---------------------------------------------------------------------------------------------- vocabulary
local NUMBER_WORDS = {
  zero = 0, one = 1, two = 2, three = 3, four = 4, five = 5, six = 6, seven = 7, eight = 8, nine = 9, ten = 10,
  eleven = 11, twelve = 12, thirteen = 13, fourteen = 14, fifteen = 15, sixteen = 16, seventeen = 17, eighteen = 18,
  nineteen = 19, twenty = 20, thirty = 30, forty = 40, fifty = 50, sixty = 60, seventy = 70, eighty = 80, ninety = 90,
  hundred = 100,
}

-- Quantity phrases (matched on the words BEFORE a noun, longest first). count = integer; 0 = everything that exists.
local QUANT_PHRASES = {
  { "half a dozen of", 6 }, { "half a dozen", 6 }, { "a couple of dozen", 24 }, { "a couple of", 2 }, { "a couple", 2 },
  { "a pair of", 2 }, { "a pair", 2 }, { "a duo of", 2 }, { "a few", 3 }, { "a handful of", 5 }, { "a handful", 5 },
  { "a bunch of", 6 }, { "a bunch", 6 }, { "a lot of", 6 }, { "lots of", 6 }, { "loads of", 6 }, { "a load of", 6 },
  { "plenty of", 6 }, { "a ton of", 6 }, { "tons of", 6 }, { "a dozen of", 12 }, { "a dozen", 12 }, { "dozens of", 12 },
  { "a hundred", 100 }, { "one hundred", 100 }, { "a number of", 4 }, { "several", 4 }, { "some", 3 }, { "both of the", 2 }, { "both the", 2 }, { "both", 2 }, { "many", 6 },
  { "numerous", 6 }, { "all of the", 0 }, { "all the", 0 }, { "every single", 0 }, { "every", 0 }, { "each of the", 0 },
  { "each", 0 }, { "all of them", 0 }, { "all", 0 },
  { "another", 1 }, { "one more", 1 }, { "two more", 2 }, { "three more", 3 }, { "an extra", 1 }, { "one extra", 1 },
  { "a single", 1 }, { "single", 1 }, { "an", 1 }, { "a", 1 },
}
table.sort(QUANT_PHRASES, function(a, b)
  if #a[1] ~= #b[1] then return #a[1] > #b[1] end
  return a[1] < b[1]
end)
local DEFINITE_QUANT = { ["both"] = true, ["both of the"] = true, ["both the"] = true, ["all"] = true, ["all the"] = true,
  ["all of the"] = true, ["every"] = true, ["every single"] = true, ["each"] = true, ["each of the"] = true, ["all of them"] = true }

local ARTICLES = { the = true, this = true, that = true, these = true, those = true, my = true, your = true, our = true, its = true }
local DEFINITE_WORDS = { the = true, this = true, that = true, these = true, those = true, my = true, your = true, our = true, them = true, it = true }

-- Relations: phrase -> arrangement, and whether a target noun follows.
local RELATIONS = {
  { "in front of", "in_front_of", true }, { "in the front of", "in_front_of", true }, { "ahead of", "in_front_of", true },
  { "on top of each other", "stacked", false }, { "on top of one another", "stacked", false },
  { "on top of", "on", true }, { "on the top of", "on", true }, { "up on", "on", true }, { "on", "on", true }, { "onto", "on", true },
  { "next to", "beside", true }, { "right next to", "beside", true }, { "beside", "beside", true }, { "alongside", "beside", true },
  { "close to", "beside", true }, { "near to", "beside", true }, { "near", "beside", true }, { "by the side of", "beside", true },
  { "at the side of", "beside", true }, { "to the left of", "beside", true }, { "to the right of", "beside", true },
  { "behind", "behind", true }, { "in back of", "behind", true }, { "at the back of", "behind", true },
  { "around", "around", true }, { "surrounding", "around", true }, { "encircling", "around", true },
  { "facing each other", "facing_each_other", false }, { "face each other", "facing_each_other", false },
  { "to face each other", "facing_each_other", false }, { "to face one another", "facing_each_other", false },
  { "facing one another", "facing_each_other", false }, { "face one another", "facing_each_other", false },
  { "opposite each other", "facing_each_other", false }, { "across from each other", "facing_each_other", false },
  { "face to face", "facing_each_other", false }, { "facing opposite", "facing_each_other", false },
  { "facing towards", "facing_me", true }, { "facing toward", "facing_me", true }, { "turned towards", "facing_me", true },
  { "turned toward", "facing_me", true }, { "pointing at", "facing_me", true }, { "pointed at", "facing_me", true },
  { "to face", "facing_me", true }, { "facing", "facing_me", true }, { "face", "facing_me", true }, { "towards", "facing_me", true },
  { "toward", "facing_me", true }, { "looking at", "facing_me", true }, { "look at", "facing_me", true },
  { "against the wall", "against_wall", false }, { "against a wall", "against_wall", false }, { "against the walls", "against_wall", false },
  { "along the wall", "against_wall", false }, { "along a wall", "against_wall", false }, { "along the walls", "against_wall", false },
  { "by the wall", "against_wall", false }, { "on the wall", "against_wall", false }, { "up against the wall", "against_wall", false },
  { "to the wall", "against_wall", false }, { "to the walls", "against_wall", false }, { "against", "against_wall", true },
  { "in the corner", "corner", false }, { "in a corner", "corner", false }, { "in the corners", "corner", false },
  { "in each corner", "corner", false }, { "in every corner", "corner", false }, { "in corners", "corner", false },
  { "to the corner", "corner", false }, { "to a corner", "corner", false }, { "in the room's corner", "corner", false },
  { "in a row", "row", false }, { "in rows", "row", false }, { "in a line", "row", false }, { "in line", "row", false },
  { "in a straight line", "row", false }, { "in a single row", "row", false }, { "in one row", "row", false },
  { "side by side", "row", false }, { "next to each other", "row", false }, { "beside each other", "row", false },
  { "lined up", "row", false }, { "in a queue", "row", false }, { "in a single file", "row", false }, { "in a long row", "row", false },
  { "in a circle", "circle", false }, { "in a ring", "circle", false }, { "in a semicircle", "circle", false },
  { "in a semi circle", "circle", false }, { "in a round", "circle", false }, { "in a circular", "circle", false },
  { "in a circular arrangement", "circle", false }, { "in a loop", "circle", false }, { "in a big circle", "circle", false },
  { "in a u shape", "u_shape", false }, { "in a u", "u_shape", false }, { "in a u formation", "u_shape", false },
  { "u shaped", "u_shape", false }, { "u shape", "u_shape", false }, { "in a horseshoe", "u_shape", false }, { "horseshoe", "u_shape", false },
  { "in an l shape", "l_shape", false }, { "in an l", "l_shape", false }, { "l shaped", "l_shape", false }, { "l shape", "l_shape", false },
  { "in a grid", "grid", false }, { "in a square", "grid", false }, { "in a block", "grid", false }, { "in columns", "grid", false },
  { "in rows and columns", "grid", false }, { "in a matrix", "grid", false }, { "in a lattice", "grid", false }, { "grid", "grid", false },
  { "a circle of", "circle", false }, { "a ring of", "circle", false }, { "a row of", "row", false }, { "a line of", "row", false },
  { "a grid of", "grid", false }, { "a stack of", "stacked", false }, { "a pile of", "stacked", false }, { "a pyramid of", "stacked", false },
  { "a u shape of", "u_shape", false }, { "a semicircle of", "circle", false }, { "a horseshoe of", "u_shape", false },
  { "everywhere", "scattered", false }, { "anywhere", "scattered", false }, { "all around the room", "scattered", false },
  { "scattered", "scattered", false }, { "randomly", "scattered", false }, { "at random", "scattered", false },
  { "all over the place", "scattered", false }, { "all over", "scattered", false }, { "around the room", "scattered", false },
  { "about the room", "scattered", false }, { "throughout the room", "scattered", false }, { "here and there", "scattered", false },
  { "scatter", "scattered", false },
  { "stacked", "stacked", false }, { "stack", "stacked", false }, { "stack up", "stacked", false }, { "piled up", "stacked", false },
  { "piled", "stacked", false }, { "in a pile", "stacked", false }, { "in a stack", "stacked", false }, { "in a pyramid", "stacked", false },
  { "in a heap", "stacked", false },
}
table.sort(RELATIONS, function(a, b)
  if #a[1] ~= #b[1] then return #a[1] > #b[1] end
  return a[1] < b[1]
end)

local WORLD_TARGETS = { wall = "wall", walls = "wall", window = "window", windows = "window", door = "door", doors = "door",
  corner = "corner", corners = "corner", ceiling = "ceiling", floor = "floor", room = "room", fire = "fire" }
local SPEAKER_WORDS = { me = true, myself = true, us = true, ourselves = true, here = true, player = true }
local PRONOUN_TARGETS = { it = true, them = true, that = true, this = true, those = true, these = true, one = true, ones = true }

-- Words that are never a thing's name or description.
local STOP = {
  ["and"] = true, ["with"] = true, ["plus"] = true, ["of"] = true, ["to"] = true, ["in"] = true, ["on"] = true, ["at"] = true,
  ["for"] = true, ["from"] = true, ["into"] = true, ["onto"] = true, ["up"] = true, ["it"] = true, ["them"] = true,
  ["then"] = true, ["also"] = true, ["too"] = true, ["please"] = true, ["thanks"] = true, ["thank"] = true, ["you"] = true,
  ["there"] = true, ["over"] = true, ["right"] = true, ["just"] = true, ["now"] = true, ["here"] = true, ["so"] = true,
  ["is"] = true, ["are"] = true, ["be"] = true, ["we"] = true, ["i"] = true, ["me"] = true, ["my"] = true, ["out"] = true,
  ["as"] = true, ["or"] = true, ["but"] = true, ["if"] = true, ["when"] = true, ["while"] = true, ["where"] = true,
  ["around"] = true, ["next"] = true, ["by"] = true, ["off"] = true, ["down"] = true, ["back"] = true, ["away"] = true,
}
-- nouns that name places or parts of a scene, never things to summon (kept out of "unknown thing" detection)
local SCENE_NOUNS = { room = true, middle = true, center = true, centre = true, floor = true, left = true, right = true, back = true,
  front = true, side = true, area = true, space = true, scene = true, program = true, holodeck = true, world = true, simulation = true,
  rest = true, others = true, other = true, same = true, way = true, spot = true, place = true, position = true, distance = true,
  direction = true, ground = true, air = true, top = true, bottom = true, edge = true, ["end"] = true, line = true, row = true,
  rows = true, circle = true, ring = true, grid = true, pile = true, stack = true, group = true, set = true, bunch = true, lot = true,
  lots = true, pair = true, couple = true, few = true, wall = true, walls = true, corner = true, corners = true, window = true,
  windows = true, door = true, doors = true, ceiling = true, layout = true, arrangement = true, formation = true, order = true,
  thing = true, things = true, stuff = true, it = true, them = true, this = true, that = true, time = true, moment = true,
  sec = true, second = true, minute = true, bit = true, little = true, more = true, less = true, much = true, enough = true,
  look = true, view = true, idea = true, favor = true, favour = true, step = true, mess = true, size = true, one = true,
  ones = true, everything = true, anything = true, something = true, nothing = true, all = true, each = true, both = true,
}

local VERB_ADD = { add = true, spawn = true, summon = true, create = true, make = true, put = true, place = true, give = true,
  get = true, bring = true, build = true, generate = true, materialize = true, materialise = true, conjure = true, set = true,
  throw = true, drop = true, show = true, install = true, insert = true, produce = true, fetch = true, offer = true,
  provide = true, load = true, populate = true, furnish = true, stick = true, plop = true, pop = true, whip = true, rustle = true,
  grab = true, lay = true, render = true, deliver = true, need = true, want = true, like = true, drag = true, pull = true }
-- verbs that mean "move something that already exists" when the thing is a definite one ("put THE lamp next to ...")
local VERB_MOVEISH = { put = true, bring = true, get = true, take = true, drag = true, pull = true, push = true, set = true,
  place = true, shift = true, carry = true, drop = true, stick = true, lay = true, move = true, slide = true, shove = true }
local VERB_TWO = { ["set up"] = true, ["throw in"] = true, ["drop in"] = true, ["bring in"] = true, ["put in"] = true, ["pull up"] = true,
  ["whip up"] = true, ["rustle up"] = true, ["give me"] = true, ["get me"] = true, ["bring me"] = true, ["make me"] = true,
  ["pop in"] = true, ["stick in"] = true, ["pull in"] = true, ["fit out"] = true, ["kit out"] = true, ["line up"] = true }

local REMOVE_PHRASES = { "get rid of", "take away", "take out", "throw away", "throw out", "do away with", "clear away", "pick up",
  "remove", "delete", "erase", "destroy", "dismiss", "banish", "dematerialize", "dematerialise", "despawn", "vanish", "kill",
  "discard", "ditch", "lose", "scrap", "eliminate", "zap", "disintegrate", "evaporate", "clear", "wipe out", "get rid" }
local CLEAR_STRONG = { "end program", "end the program", "end simulation", "end the simulation", "end holodeck", "end the holodeck",
  "arch", "clear the room", "clear the area", "clear the scene", "clear the floor", "clear the space", "clear it all", "clear everything",
  "clear all", "clear out everything", "clean up everything", "clean the room", "start over", "start again", "empty the room",
  "wipe the room", "wipe everything", "wipe it all", "wipe all", "reset the room", "reset the scene", "reset everything", "clean slate",
  "remove everything", "remove all", "remove it all", "delete everything", "delete all", "delete it all", "get rid of everything",
  "get rid of it all", "get rid of all of it", "dismiss everything", "dismiss all", "erase everything", "erase all", "erase it all",
  "nuke everything", "nuke it all", "destroy everything", "destroy it all", "clear it out", "clear everything out", "that's enough",
  "thats enough", "enough already", "get rid of all", "clean up all", "take everything away", "take it all away", "scrap everything",
  "scrap it all", "make it all go away", "make everything go away", "dematerialize everything", "dematerialise everything" }
local CLEAR_WEAK = { "clear out", "clear up", "clean up", "clear", "wipe", "reset", "wipe out", "clean out" }
local UNDO_PHRASES = { "undo", "take that back", "take it back", "scratch that", "never mind", "nevermind", "revert", "put it back",
  "put them back", "that was a mistake", "go back", "oops", "cancel that", "cancel", "ctrl z", "undo that", "undo it", "undo the last" }
local MOVE_VERBS = { "move", "shift", "slide", "relocate", "nudge", "reposition", "push", "pull closer", "bring closer",
  "bring over", "move over", "scoot", "drag", "carry", "transfer", "shove" }
local ROTATE_VERBS = { "rotate", "turn", "spin", "swivel", "flip", "pivot", "twist", "orient", "angle", "reorient", "swing" }
local REARRANGE_VERBS = { "rearrange", "re arrange", "reorganize", "reorganise", "reorder", "tidy up", "tidy", "regroup",
  "straighten", "line up", "organize", "organise", "realign", "re align", "shuffle", "redo the layout", "redo", "neaten", "arrange" }

-- ---------------------------------------------------------------------------------------------- helpers
local function trim(s) return (s:gsub("^%s+", ""):gsub("%s+$", "")) end

local function words_of(s)
  local out = {}
  for w in s:gmatch("[^%s]+") do out[#out + 1] = w end
  return out
end

local function join(t, i, j) return table.concat(t, " ", i, j) end

--- Regular English plurals of a (possibly multi-word) name.
local function plural_forms(name)
  local words = words_of(name)
  local last = words[#words]
  local base = (#words > 1) and (join(words, 1, #words - 1) .. " ") or ""
  local forms = {}
  local function add(p) forms[#forms + 1] = base .. p end
  if last:match("[sxz]$") or last:match("ch$") or last:match("sh$") then add(last .. "es")
  elseif last:match("[^aeiou]y$") then add(last:sub(1, -2) .. "ies")
  elseif last:match("fe$") then add(last:sub(1, -3) .. "ves"); add(last .. "s")
  elseif last:match("[^f]f$") then add(last:sub(1, -2) .. "ves"); add(last .. "s")
  else add(last .. "s") end
  local head = name:match("^(%S+) of ")                       -- "chest of drawers" -> "chests of drawers"
  if head then forms[#forms + 1] = plural_forms(head)[1] .. name:sub(#head + 1) end
  return forms
end

local function starts_with_phrase(tokens, i, phrase)
  local pw = words_of(phrase)
  for k = 1, #pw do
    if tokens[i + k - 1] ~= pw[k] then return nil end
  end
  return #pw
end

local function number_of(tok)
  if tok == nil then return nil end
  local n = tonumber(tok)
  if n and n >= 0 and n == math.floor(n) and n <= 200 then return n end
  return NUMBER_WORDS[tok]
end

local function singular_guess(w)
  if w:match("ies$") and #w > 4 then return w:sub(1, -4) .. "y" end
  if w:match("ves$") and #w > 4 then return w:sub(1, -4) .. "f" end
  if w:match("ses$") or w:match("xes$") or w:match("ches$") or w:match("shes$") then return w:sub(1, -3) end
  if w:match("[^s]s$") and #w > 3 then return w:sub(1, -2) end
  return w
end

--- Edit distance where swapping two neighbouring letters costs one ("barrles" is one slip from "barrels").
local function distance(a, b, limit)
  local la, lb = #a, #b
  if math.abs(la - lb) > limit then return limit + 1 end
  local rows = {}
  for i = 0, la do
    rows[i] = { [0] = i }
  end
  for j = 0, lb do rows[0][j] = j end
  for i = 1, la do
    local best = i
    for j = 1, lb do
      local cost = (a:byte(i) == b:byte(j)) and 0 or 1
      local v = math.min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
      if i > 1 and j > 1 and a:byte(i) == b:byte(j - 1) and a:byte(i - 1) == b:byte(j) then
        v = math.min(v, rows[i - 2][j - 2] + 1)
      end
      rows[i][j] = v
      if v < best then best = v end
    end
    if best > limit then return limit + 1 end
  end
  return rows[la][lb]
end

-- ---------------------------------------------------------------------------------------------- construction
function Parse.new(catalog)
  local self = setmetatable({}, Parse)
  self.catalog = catalog
  self.noun = {}                      -- word string -> { kind = id, plural = bool, approx = bool }
  self.max_len = 1
  local function add_noun(str, kind, plural, approx)
    str = trim(str:lower())
    if str == "" then return end
    local w = words_of(str)
    if #w > self.max_len then self.max_len = #w end
    local old = self.noun[str]
    -- an exact name always beats an approximate one; a singular beats a plural
    if not old or (old.approx and not approx) or (old.plural and not plural and (not approx or old.approx)) then
      self.noun[str] = { kind = kind, plural = plural, approx = approx or nil }
    end
    local nospace = str:gsub(" ", "")
    if #w > 1 and not self.noun[nospace] then self.noun[nospace] = { kind = kind, plural = plural, approx = approx or nil } end
  end
  for _, kind in ipairs(catalog.order) do
    local k = catalog.kinds[kind]
    add_noun(kind, kind, false)
    for _, n in ipairs(k.names) do
      add_noun(n, kind, false)
      for _, p in ipairs(plural_forms(n:lower())) do add_noun(p, kind, true) end
    end
    for _, p in ipairs(plural_forms(kind:lower())) do add_noun(p, kind, true) end
  end
  -- approximate names: the nearest thing the library has ("fireplace" -> wood stove); added after every exact name
  for _, kind in ipairs(catalog.order) do
    for _, n in ipairs(catalog.kinds[kind].approx or {}) do
      add_noun(n, kind, false, true)
      for _, p in ipairs(plural_forms(n:lower())) do add_noun(p, kind, true, true) end
    end
  end
  local extra = { ["bookshelves"] = "bookshelf", ["shelves"] = "shelf", ["loos"] = "toilet" }
  for p, kind in pairs(extra) do if catalog.kinds[kind] then add_noun(p, kind, true) end end
  -- words usable for typo repair: single-word nouns of at least 5 letters
  self.typo_words = {}
  for str, nd in pairs(self.noun) do
    if not str:find(" ", 1, true) and #str >= 5 then self.typo_words[#self.typo_words + 1] = { str, nd } end
  end
  table.sort(self.typo_words, function(a, b) return a[1] < b[1] end)
  -- scenes
  self.scene = {}
  for name, _ in pairs(catalog.recipes or {}) do self.scene[name] = name end
  local scene_alias = {
    ["lounge"] = "living room", ["sitting room"] = "living room", ["front room"] = "living room", ["den"] = "living room",
    ["family room"] = "living room", ["parlor"] = "living room", ["parlour"] = "living room", ["living area"] = "living room",
    ["livingroom"] = "living room", ["lounge room"] = "living room", ["dining area"] = "dining room", ["diningroom"] = "dining room",
    ["diner"] = "dining room", ["canteen"] = "dining room", ["study"] = "office", ["workplace"] = "office", ["work room"] = "office",
    ["home office"] = "office", ["cubicle"] = "office", ["bed room"] = "bedroom", ["sleeping area"] = "bedroom", ["dorm"] = "bedroom",
    ["dorm room"] = "bedroom", ["reception"] = "waiting room", ["lobby"] = "waiting room", ["waiting area"] = "waiting room",
    ["waitingroom"] = "waiting room", ["restroom"] = "bathroom", ["washroom"] = "bathroom", ["bath room"] = "bathroom",
    ["toilets"] = "bathroom", ["kitchenette"] = "kitchen", ["galley"] = "kitchen",
  }
  for alias, name in pairs(scene_alias) do if catalog.recipes and catalog.recipes[name] then self.scene[alias] = name end end
  self.scene_max = 3
  return self
end

-- ---------------------------------------------------------------------------------------------- normalisation
local PREFIXES = {
  "^hey skippy[,:%s]+", "^ok skippy[,:%s]+", "^okay skippy[,:%s]+", "^skippy[,:%s]+", "^skippy$",
  "^computer[,:%s]+", "^holodeck[,:%s]+", "^hey computer[,:%s]+", "^hey holodeck[,:%s]+", "^hey[,%s]+", "^ok[,%s]+", "^okay[,%s]+",
  "^alright[,%s]+", "^all right[,%s]+", "^please[,%s]+", "^um+[,%s]+", "^uh+[,%s]+", "^er+[,%s]+", "^hmm+[,%s]+", "^well[,%s]+",
  "^so[,%s]+", "^now[,%s]+", "^and[,%s]+", "^then[,%s]+", "^actually[,%s]+", "^first[,%s]+", "^next[,%s]+", "^also[,%s]+",
  "^can you[%s]+", "^could you[%s]+", "^would you[%s]+", "^will you[%s]+", "^can u[%s]+", "^could u[%s]+", "^can i have[%s]+",
  "^can we have[%s]+", "^could i have[%s]+", "^could we have[%s]+", "^may i have[%s]+", "^i want you to[%s]+",
  "^i would like you to[%s]+", "^id like you to[%s]+", "^i want[%s]+", "^id like[%s]+", "^i would like[%s]+", "^i need[%s]+",
  "^i wanna[%s]+", "^i gotta have[%s]+", "^lets[%s]+", "^let us[%s]+", "^we need[%s]+", "^we want[%s]+", "^we should have[%s]+",
  "^go ahead and[%s]+", "^just[%s]+", "^maybe[%s]+", "^perhaps[%s]+", "^try[%s]+", "^how about[%s]+", "^what about[%s]+",
  "^i think i need[%s]+", "^how bout[%s]+", "^show me[%s]+", "^there should be[%s]+", "^there needs to be[%s]+", "^we have[%s]+",
  "^i have[%s]+", "^need[%s]+", "^gimme[%s]+", "^pls[,%s]+", "^plz[,%s]+", "^kindly[%s]+", "^yo[,%s]+", "^dude[,%s]+", "^man[,%s]+",
  "^quickly[%s]+", "^real quick[,%s]+", "^thats enough[,%s]+", "^that is enough[,%s]+",
}
local SUFFIXES = { "[,%s]+please$", "[,%s]+thanks$", "[,%s]+thank you$", "[,%s]+pls$", "[,%s]+plz$", "[,%s]+thx$", "[,%s]+ty$",
  "[,%s]+if you can$", "[,%s]+for me$", "[,%s]+right now$", "[,%s]+now$", "[,%s]+quickly$", "[,%s]+asap$", "[,%s]+okay$", "[,%s]+ok$",
  "[,%s]+real quick$", "[,%s]+thanks a lot$", "[,%s]+thank you very much$", "[,%s]+lol$", "[,%s]+lmao$", "[,%s]+haha+$",
  "[,%s]+xd$", "[,%s]+hehe$", "[,%s]+if possible$", "[,%s]+if thats ok$", "[,%s]+thx$" }

local function normalise(s)
  s = tostring(s or ""):lower()
  s = s:gsub("'", ""):gsub("\226\128\153", "")
  s = s:gsub("[%.!?;:]+", " , ")
  s = s:gsub("%-", " ")
  s = s:gsub("%s*,%s*", " , ")
  s = s:gsub("(%d)x%f[%A]", "%1 x"):gsub("%f[%w]x(%d)", "x %1")
  s = trim(s:gsub("%s+", " "))
  local changed = true
  while changed do
    changed = false
    for _, p in ipairs(PREFIXES) do
      local n = s:gsub(p, "", 1)
      if n ~= s then s = trim(n:gsub("^,%s*", "")); changed = true end
    end
    for _, p in ipairs(SUFFIXES) do
      local n = s:gsub(p, "", 1)
      if n ~= s then s = trim(n); changed = true end
    end
  end
  s = s:gsub("%s*,%s*$", ""):gsub("^%s*,%s*", "")
  -- spoken variants of the same thing
  s = (" " .. s .. " "):gsub(" into ", " in "):gsub(" onto ", " on "):gsub(" u shaped ", " u shape "):gsub(" tv's ", " tvs ")
  s = s:gsub(" towards ", " toward ")
  return trim(s)
end
Parse.normalise = normalise

-- ---------------------------------------------------------------------------------------------- the parse
local function contains_phrase(tokens, list)
  for i = 1, #tokens do
    for _, p in ipairs(list) do
      local n = starts_with_phrase(tokens, i, p)
      if n then return i, n, p end
    end
  end
end

--- The earliest match of any phrase of `list` at or before token index `limit` (default: anywhere).
local function first_phrase(tokens, list, limit)
  local bi, bn, bp
  for i = 1, math.min(#tokens, limit or #tokens) do
    for _, p in ipairs(list) do
      local n = starts_with_phrase(tokens, i, p)
      if n and (not bi or n > bn) then bi, bn, bp = i, n, p end
    end
    if bi then break end
  end
  return bi, bn, bp
end

-- ---------------------------------------------------------------------------------------------- the wake word
-- Players with a live mic talk to each other normally; only a sentence that STARTS with the wake word is a request.
-- Speech recognition spells the name several ways, so the accepted spellings are a list.
Parse.WAKE_WORDS = { "skippy", "skipy", "skippie", "skipi", "skippi", "skipee", "skippy's" }
local WAKE_LEAD = { hey = true, ok = true, okay = true, yo = true, hi = true, hello = true, oi = true, so = true }

--- Is this sentence addressed to the holodeck? Returns (true, the request without the wake word) or (false, text).
function Parse.strip_wake(text, words)
  local raw = tostring(text or "")
  local s = raw:lower():gsub("^%s+", "")
  local rest_from = 1
  local lead = s:match("^(%a+)[,%s]+")
  local after_lead = 1
  if lead and WAKE_LEAD[lead] then after_lead = #s:match("^%a+[,%s]+") + 1 end
  for _, w in ipairs(words or Parse.WAKE_WORDS) do
    if s:sub(after_lead, after_lead + #w - 1) == w then
      local nxt = s:sub(after_lead + #w, after_lead + #w)
      if nxt == "" or not nxt:match("[%a%d]") then
        local lead_len = raw:len() - raw:gsub("^%s+", ""):len()
        local rest = raw:sub(lead_len + after_lead + #w):gsub("^[%s,:;%.!%-]+", "")
        return true, rest
      end
    end
  end
  return false, raw
end

function Parse:parse(text)
  local raw = tostring(text or "")
  local s = normalise(raw)
  local req = { raw = raw, normalised = s, action = "add", items = {}, arrangement = "unspecified", target = "", scene = "", unknown = {},
                notes = {} }
  if s == "" then req.action = "none"; return req end
  local tokens = words_of(s)
  local text_s = " " .. s .. " "

  -- ---- whole-sentence commands
  for _, p in ipairs(UNDO_PHRASES) do
    if text_s:find(" " .. p .. " ", 1, true) and #tokens <= #words_of(p) + 4 then req.action = "undo"; return req end
  end
  local has_noun_word = false
  for i = 1, #tokens do
    for len = math.min(self.max_len, #tokens - i + 1), 1, -1 do
      if self.noun[join(tokens, i, i + len - 1)] then has_noun_word = true; break end
    end
    if has_noun_word then break end
  end
  for _, p in ipairs(CLEAR_STRONG) do
    if text_s:find(" " .. p .. " ", 1, true) and not (has_noun_word and p:match("^remove all$")) then
      -- "get rid of all the lamps" names a kind: that is a removal of that kind, not a clear
      if not (has_noun_word and (p == "remove all" or p == "delete all" or p == "clear all" or p == "dismiss all" or p == "erase all"
                                 or p == "get rid of all" or p == "clean up all")) then
        req.action = "clear"
        return req
      end
    end
  end
  do
    local wi, wn = first_phrase(tokens, CLEAR_WEAK, 3)
    if wi and not has_noun_word then
      local rest = join(tokens, wi + wn, #tokens)
      if rest == "" or rest:match("^the room") or rest:match("^everything") or rest:match("^all$") or rest:match("^it all")
         or rest:match("^out$") or rest:match("^up$") or rest:match("^the area") or rest:match("^the scene") or rest:match("^the holodeck")
         or rest:match("^the floor") or rest:match("^the space") or rest:match("^it out") or rest:match("^house") then
        req.action = "clear"
        return req
      end
    end
  end

  -- ---- the verb (first few words), and where the wanted things start
  local verb, start = nil, 1
  do
    local ri, rn = first_phrase(tokens, REMOVE_PHRASES, 3)
    local ti, tn = first_phrase(tokens, ROTATE_VERBS, 3)
    local ai, an = first_phrase(tokens, REARRANGE_VERBS, 3)
    local mi, mn = first_phrase(tokens, MOVE_VERBS, 3)
    if ri then
      verb, start, req.action = tokens[ri], ri + rn, "remove"
      if tokens[start] == "all" and #tokens == start then req.action = "clear" end
    elseif ti then verb, start, req.action = tokens[ti], ti + tn, "rotate"
    elseif ai then verb, start, req.action = tokens[ai], ai + an, "rearrange"
    elseif mi then verb, start, req.action = tokens[mi], mi + mn, "move"
    else
      local first = tokens[1]
      if VERB_ADD[first] then
        verb, start = first, 2
        local two = first .. " " .. (tokens[2] or "")
        if VERB_TWO[two] then start = 3 end
        if tokens[start] == "me" or tokens[start] == "us" then start = start + 1 end
        if tokens[start] == "in" or tokens[start] == "up" or tokens[start] == "out" then start = start + 1 end
      end
    end
  end
  req.verb = verb

  local mask = {}
  local function mark(i, j) for k = i, j do mask[k] = true end end

  -- ---- relations and their targets
  local relation_hits = {}
  do
    local i = start
    while i <= #tokens do
      local hit
      for _, r in ipairs(RELATIONS) do
        local n = starts_with_phrase(tokens, i, r[1])
        if n then hit = { phrase = r[1], arr = r[2], needs = r[3], n = n }; break end
      end
      if hit then
        hit.i = i
        relation_hits[#relation_hits + 1] = hit
        mark(i, i + hit.n - 1)
        i = i + hit.n
      else
        i = i + 1
      end
    end
  end
  for _, hit in ipairs(relation_hits) do
    if hit.needs then
      local j = hit.i + hit.n
      local k = j
      while tokens[k] and (ARTICLES[tokens[k]] or tokens[k] == "a" or tokens[k] == "an") do k = k + 1 end
      local found
      if tokens[k] and SPEAKER_WORDS[tokens[k]] then found = { target = "me", i = j, j = k }
      elseif tokens[k] == "where" and tokens[k + 1] == "i" then found = { target = "here", i = j, j = k + 3 }
      elseif tokens[k] and PRONOUN_TARGETS[tokens[k]] and not ARTICLES[tokens[j]] then found = { target = "@prev", i = j, j = k }
      elseif tokens[k] and WORLD_TARGETS[tokens[k]] then found = { target = WORLD_TARGETS[tokens[k]], i = j, j = k }
      else
        for kk = k, math.min(#tokens, k + 2) do
          for len = math.min(self.max_len, #tokens - kk + 1), 1, -1 do
            local nd = self.noun[join(tokens, kk, kk + len - 1)]
            if nd and not mask[kk] then found = { target = nd.kind, i = j, j = kk + len - 1, plural = nd.plural }; break end
          end
          if found then break end
        end
      end
      if found then
        hit.target, hit.target_plural = found.target, found.plural
        mark(found.i, found.j)
      end
    end
  end

  -- ---- the library's own things (longest match first, left to right)
  local spans = {}
  do
    local i = start
    while i <= #tokens do
      if mask[i] then
        i = i + 1
      else
        local matched
        for len = math.min(self.max_len, #tokens - i + 1), 1, -1 do
          local ok = true
          for k = i, i + len - 1 do if mask[k] then ok = false; break end end
          if ok then
            local nd = self.noun[join(tokens, i, i + len - 1)]
            if nd then matched = { i = i, j = i + len - 1, kind = nd.kind, plural = nd.plural, approx = nd.approx, word = join(tokens, i, i + len - 1) }; break end
          end
        end
        if matched then spans[#spans + 1] = matched; mark(matched.i, matched.j); i = matched.j + 1 else i = i + 1 end
      end
    end
  end

  -- ---- whole rooms ("a living room"), on what is left
  for i = start, #tokens do
    if not mask[i] then
      for len = self.scene_max, 1, -1 do
        if i + len - 1 <= #tokens then
          local ok = true
          for k = i, i + len - 1 do if mask[k] then ok = false; break end end
          local phrase = join(tokens, i, i + len - 1)
          if ok and self.scene[phrase] then
            req.scene = self.scene[phrase]
            mark(i, i + len - 1)
            if i > start and (tokens[i - 1] == "a" or tokens[i - 1] == "an" or tokens[i - 1] == "the" or tokens[i - 1] == "my") then mark(i - 1, i - 1) end
            break
          end
        end
      end
    end
  end

  -- ---- quantity and adjectives for each kind: scan backwards from the noun
  local function scan_back(i_noun, lower_bound)
    local adjs, count, definite, ate = {}, nil, false, i_noun
    local k = i_noun - 1
    local guard = 0
    while k >= lower_bound and guard < 7 do
      guard = guard + 1
      local tok = tokens[k]
      if mask[k] and not (tok == "a" or tok == "an" or tok == "the") then break end
      if STOP[tok] and tok ~= "of" then break end
      local took
      for _, q in ipairs(QUANT_PHRASES) do
        local pw = words_of(q[1])
        local from = k - #pw + 1
        if from >= lower_bound then
          local ok = true
          for m = 1, #pw do if tokens[from + m - 1] ~= pw[m] then ok = false; break end end
          if ok then
            count = q[2]; ate = from; k = from - 1; took = true
            if DEFINITE_QUANT[q[1]] then definite = true end
            break
          end
        end
      end
      if took then break end
      local num = number_of(tok)
      if num then count = num; ate = k; k = k - 1; break end
      if ARTICLES[tok] then
        definite = DEFINITE_WORDS[tok] or false
        ate = k; k = k - 1
        break
      end
      if tok == "of" then k = k - 1
      elseif tok == "," then break
      else table.insert(adjs, 1, tok); ate = k; k = k - 1 end
    end
    return adjs, count, definite, ate
  end

  for idx, sp in ipairs(spans) do
    local prev_end = (idx > 1) and spans[idx - 1].j + 1 or start
    local adjs, count, definite, ate = scan_back(sp.i, prev_end)
    if count == nil and tokens[sp.j + 1] == "x" and number_of(tokens[sp.j + 2]) then
      count = number_of(tokens[sp.j + 2]); mark(sp.j + 1, sp.j + 2)
    end
    if count == nil and tokens[sp.i - 1] == "x" and number_of(tokens[sp.i - 2]) then count = number_of(tokens[sp.i - 2]) end
    if count == nil then
      -- "the chairs" means the ones that stand there (all of them); a bare "chairs" means a handful
      if sp.plural and definite then count = 0 else count = sp.plural and 3 or 1 end
    end
    local clean = {}
    for _, a in ipairs(adjs) do
      if not STOP[a] and a ~= "x" and not number_of(a) and not ARTICLES[a] then clean[#clean + 1] = a end
    end
    for k = ate, sp.i - 1 do mask[k] = true end
    req.items[#req.items + 1] = { kind = sp.kind, count = count, adjectives = clean, word = sp.word, definite = definite, plural = sp.plural,
                                   position = sp.i, approx = sp.approx or nil }
  end

  -- ---- things named but not in the library: article / quantity + word(s) left over
  local function quant_at(i)
    for _, q in ipairs(QUANT_PHRASES) do
      local n = starts_with_phrase(tokens, i, q[1])
      if n then return n, q[2], DEFINITE_QUANT[q[1]] end
    end
    local num = number_of(tokens[i])
    if num then return 1, num, false end
    if ARTICLES[tokens[i]] then return 1, 1, DEFINITE_WORDS[tokens[i]] or false end
    return nil
  end
  do
    local i = start
    while i <= #tokens do
      if not mask[i] then
        local n, count, definite = quant_at(i)
        if n then
          local j = i + n
          local run = {}
          while tokens[j] and not mask[j] and not STOP[tokens[j]] and tokens[j] ~= "," and not quant_at(j) do
            run[#run + 1] = tokens[j]; j = j + 1
          end
          if #run > 0 then
            local word = run[#run]
            if not number_of(word) and not VERB_ADD[word] and not SCENE_NOUNS[word] and not SCENE_NOUNS[singular_guess(word)] then
              -- a typo of something we have? ("barrles")
              local repaired
              if #word >= 4 then
                local limit = (#word >= 8) and 2 or 1
                local best, best_d, tie = nil, limit + 1, false
                for _, tw in ipairs(self.typo_words) do
                  local d = distance(word, tw[1], limit)
                  if d < best_d then best, best_d, tie = tw, d, false elseif d == best_d and best and best[2].kind ~= tw[2].kind then tie = true end
                end
                if best and not tie and best_d <= limit then repaired = best end
              end
              local c = count
              if repaired then
                local adjs = {}
                for k = 1, #run - 1 do adjs[#adjs + 1] = run[k] end
                if c == 1 and (repaired[2].plural or (word:match("s$") and not word:match("ss$"))) and n == 1 and not definite and not number_of(tokens[i]) and tokens[i] ~= "a" and tokens[i] ~= "an" then c = 3 end
                req.items[#req.items + 1] = { kind = repaired[2].kind, count = c, adjectives = adjs, word = word, definite = definite,
                                               plural = repaired[2].plural, position = i, approx = repaired[2].approx, typo = true }
                req.notes[#req.notes + 1] = "read '" .. word .. "' as '" .. repaired[1] .. "'"
              else
                local sing = singular_guess(word)
                req.unknown[#req.unknown + 1] = sing
                if c == 1 and word:match("s$") and not word:match("ss$") and #word > 3 and tokens[i] ~= "a" and tokens[i] ~= "an" and not number_of(tokens[i]) then c = 3 end
                local adjs = {}
                for k = 1, #run - 1 do adjs[#adjs + 1] = run[k] end
                local name = (#run > 1) and (join(run, 1, #run - 1) .. " " .. sing) or sing
                req.items[#req.items + 1] = { kind = "UNKNOWN:" .. sing, count = c, adjectives = {}, word = word, definite = definite,
                                               position = i, phrase = name }
              end
              mark(i, j - 1)
            end
          end
          i = math.max(j, i + 1)
        else
          i = i + 1
        end
      else
        i = i + 1
      end
    end
  end
  table.sort(req.items, function(a, b) return (a.position or 0) < (b.position or 0) end)

  -- ---- arrangement and target
  local facing_target
  for _, hit in ipairs(relation_hits) do
    if hit.arr == "facing_me" then facing_target = hit.target or "me" end
  end
  local governing                               -- the item just before the first non-facing relation (for "it")
  for _, hit in ipairs(relation_hits) do
    if hit.arr ~= "facing_me" then
      req.arrangement = hit.arr
      req.target = hit.target or ""
      for _, it in ipairs(req.items) do if it.position < hit.i then governing = it end end
      break
    end
  end
  if facing_target then
    if req.arrangement == "unspecified" then req.arrangement, req.target = "facing_me", facing_target
    else req.facing = facing_target end
  end
  -- "it" / "them": the nearest EARLIER item of a different kind, else whatever was created last
  local function resolve_pronoun(t, gov)
    if t ~= "@prev" then return t end
    local pick
    for _, it in ipairs(req.items) do
      if (not gov or it ~= gov) and not tostring(it.kind):match("^UNKNOWN") and it.position < (gov and gov.position or 1e9) and (not gov or it.kind ~= gov.kind) then pick = it end
    end
    return pick and pick.kind or "last"
  end
  req.target = resolve_pronoun(req.target, governing)
  if req.facing then req.facing = resolve_pronoun(req.facing, governing) end
  -- "against" followed by a thing means beside that thing
  if req.arrangement == "against_wall" and req.target ~= "" and req.target ~= "wall" then req.arrangement = "beside" end
  -- "turn the couch around" is a half turn, not "around something"
  if req.action == "rotate" and req.arrangement == "around" and req.target == "" then
    req.arrangement = "unspecified"; req.half_turn = true
  end
  -- a relation target that is also an item in the same sentence: the item IS the target
  for _, it in ipairs(req.items) do
    if it.kind == req.target then it.is_target = true end
  end

  -- ---- words that carried meaning but that nothing above explained ("somewhere to relax"): the caller may want help
  req.leftover = {}
  for i = start, #tokens do
    local tok = tokens[i]
    if not mask[i] and tok ~= "," and not STOP[tok] and not number_of(tok) and not ARTICLES[tok] and not VERB_ADD[tok]
       and not SCENE_NOUNS[tok] and not SCENE_NOUNS[singular_guess(tok)] and not quant_at(i) and not WORLD_TARGETS[tok]
       and not PRONOUN_TARGETS[tok] and not SPEAKER_WORDS[tok] and tok ~= "x" then
      req.leftover[#req.leftover + 1] = tok
    end
  end

  -- ---- settle the action from the whole sentence
  local function definite_items()
    if #req.items == 0 then return false end
    for _, it in ipairs(req.items) do
      if not (it.definite or it.count == 0) then return false end
    end
    return true
  end
  local function explicit_new_items()
    for _, it in ipairs(req.items) do if it.count and it.count > 0 and not it.definite then return true end end
    return false
  end
  if req.action == "rearrange" or req.action == "move" or req.action == "rotate" then
    -- "line up 4 chairs", "arrange six chairs in a circle": new things, not existing ones
    if (verb == "line" or verb == "arrange" or verb == "organize" or verb == "organise" or verb == "set" or verb == "put")
       and explicit_new_items() and not definite_items() then
      req.action = "add"
      if verb == "line" and req.arrangement == "unspecified" then req.arrangement = "row" end
    end
  elseif req.action == "add" then
    if (req.arrangement == "facing_me" or req.arrangement == "facing_each_other" or req.half_turn)
       and VERB_MOVEISH[verb or ""] == nil and (verb == "make" or verb == "have" or verb == "let" or verb == "get") and definite_items() then
      req.action = "rotate"
    elseif VERB_MOVEISH[verb or ""] and definite_items() and (req.arrangement ~= "unspecified" or s:find(" over ") or s:find(" closer") or s:find(" back ")) then
      req.action = "move"
    elseif verb == "make" and definite_items() and (req.arrangement == "facing_me" or req.arrangement == "facing_each_other") then
      req.action = "rotate"
    elseif (req.arrangement == "stacked" or req.arrangement == "row" or req.arrangement == "circle" or req.arrangement == "grid"
            or req.arrangement == "scattered") and #req.items > 0 and definite_items() and not verb then
      req.action = "rearrange"
    end
    -- "stack all the crates up": everything-that-exists plus an arrangement re-lays them out
    if req.action == "add" and #req.items > 0 and req.arrangement ~= "unspecified" and definite_items() and not (verb and VERB_ADD[verb] and explicit_new_items()) then
      local all_zero = true
      for _, it in ipairs(req.items) do if it.count ~= 0 then all_zero = false end end
      if all_zero then req.action = "rearrange" end
    end
  end
  if req.action == "remove" and #req.items == 0 and req.scene == "" then
    -- "get rid of it", "remove that": the last thing created
    req.items[1] = { kind = "@last", count = 1, adjectives = {}, word = "it", definite = true, position = 1 }
  end
  return req
end

return Parse
