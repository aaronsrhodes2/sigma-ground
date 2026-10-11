-- parse.lua -- turns what a person says to the holodeck into structured requests. Pure Lua (no GMod globals), so it is
-- unit-tested offline. It NEVER asks a question: every sentence resolves to the most sensible reading, and what it did
-- not understand is reported in `request.unknown` so the caller can say so while still acting on the rest.
--
--   local parser = Parse.new(catalog)
--   local req  = parser:parse("computer, add three red couches facing me")       -- the first request in the words
--   local reqs = parser:parse_all("add a table. put four chairs around it")      -- every request, in order
--
-- parse_all cuts the words at sentence ends ("." "!" "?" ";" and new lines), at "then" / "and then" / "after that" / "afterwards",
-- and at an "and" or a comma that is followed by a verb of its own or by a second thing with its own relation ("put a mug on the desk
-- and a plant on the shelf"). Parts that only list things stay one request ("remove the lamp and take out three stools"; "add a table
-- and put four chairs around it" is one build). "it" / "them" in a request with nothing earlier to point at means the group made last
-- ("@last"). The second argument of parse / parse_all is the request before the whole utterance, used by "another one" / "again".
-- Input is read up to 400 characters (req.truncated); control characters, quotes, brackets and Unicode spaces are folded away.
--
-- Result:
--   { action = "add"|"remove"|"clear"|"undo"|"move"|"rotate"|"rearrange"|"none",
--     items = { { kind="couch", count=3, adjectives={"red"}, word="couches", definite=false, approx=nil } },
--       kind is a library id, "UNKNOWN:<word>" (a thing the library lacks), "@last" (the most recently made group: "it",
--       "them", "the last one", "what I just made") or "@all" (everything standing: "everything", "the furniture").
--       "@all" appears only for remove / move / rotate / rearrange ("remove everything" alone is action "clear").
--     arrangement = "unspecified"|"row"|"circle"|"around"|"facing_each_other"|"facing_me"|"facing_away"|"u_shape"|"l_shape"|
--                   "grid"|"against_wall"|"beside"|"in_front_of"|"behind"|"scattered"|"stacked"|"corner"|"on"|"flank",
--     target = "" | "me" | <kind> | "wall"|"window"|"door"|"here"|"center"|"@last",
--     facing = nil | <target>   (a second relation: "in a row facing the tv"),   facing_away = nil | <target> | ""
--     shape = nil | <arrangement> (a shape that a targeted relation displaced: "in a circle around the table" -> around + circle),
--     exclude = nil | { kind ids }  ("everything except the couch", "a living room without the tv"),
--     seats = nil | N ("a table for six"),  every = nil | "corner" ("a plant in each corner"),  rows / cols = nil | N,
--     scene = "" | <recipe name>, unknown = { "dragon", ... },
--     notes = { strings for the player: "read 'barrles' as 'barrels'", "that sounded like a no, ...", ... },
--     unknown_verb = nil | "<verb>" (a verb for something the holodeck cannot do: action is "none", nothing is added),
--     away_from = nil | <kind> ("move the lamp away from the couch"), again = nil | true ("another one": see prev),
--     truncated = nil | true, verb, leftover (words nothing explained), half_turn, normalised, raw }
-- count 0 means "everything of that kind that exists". Counts are capped at 1000 (the planner applies its own, lower cap).
local Parse = {}
Parse.__index = Parse

local MAX_INPUT = 400        -- characters of a request that are read; the rest is dropped (req.truncated)
local MAX_REQUESTS = 8       -- requests read from one utterance
local COUNT_CAP = 1000

-- ---------------------------------------------------------------------------------------------- small helpers
local function set(list)
  local t = {}
  for _, v in ipairs(list) do t[v] = true end
  return t
end

local function trim(s) return (s:gsub("^%s+", ""):gsub("%s+$", "")) end

local function words_of(s)
  local out = {}
  for w in s:gmatch("[^%s]+") do out[#out + 1] = w end
  return out
end

local function join(t, i, j) return table.concat(t, " ", i, j) end

--- Index phrases by their first word (and by their last), longest first. Each entry gets .w (the words) and .n.
local function index_phrases(list, text_of)
  local by_first, by_last = {}, {}
  for _, e in ipairs(list) do
    e.w = words_of(text_of(e))
    e.n = #e.w
    by_first[e.w[1]] = by_first[e.w[1]] or {}
    table.insert(by_first[e.w[1]], e)
    by_last[e.w[e.n]] = by_last[e.w[e.n]] or {}
    table.insert(by_last[e.w[e.n]], e)
  end
  local function longest_first(a, b)
    if a.n ~= b.n then return a.n > b.n end
    return text_of(a) < text_of(b)
  end
  for _, l in pairs(by_first) do table.sort(l, longest_first) end
  for _, l in pairs(by_last) do table.sort(l, longest_first) end
  return by_first, by_last
end

--- Does the phrase (an entry with .w) start at tokens[i]?  "into" counts as "in" for the first word.
local function phrase_at(tokens, i, e)
  local w = e.w
  for k = 1, e.n do
    local t = tokens[i + k - 1]
    if t ~= w[k] and not (k == 1 and t == "into" and w[1] == "in") then return false end
  end
  return true
end

--- The first (longest) entry of a first-word index that starts at tokens[i].
local function match_at(index, tokens, i)
  local t = tokens[i]
  if t == nil then return nil end
  local l = index[t]
  if t == "into" and not l then l = index["in"] end
  if not l then return nil end
  for _, e in ipairs(l) do
    if phrase_at(tokens, i, e) then return e end
  end
end

-- ---------------------------------------------------------------------------------------------- vocabulary
local NUMBER_WORDS = {
  zero = 0, one = 1, two = 2, three = 3, four = 4, five = 5, six = 6, seven = 7, eight = 8, nine = 9, ten = 10,
  eleven = 11, twelve = 12, thirteen = 13, fourteen = 14, fifteen = 15, sixteen = 16, seventeen = 17, eighteen = 18,
  nineteen = 19, twenty = 20, thirty = 30, forty = 40, fourty = 40, fifty = 50, sixty = 60, seventy = 70, eighty = 80, ninety = 90,
  hundred = 100, thousand = 1000,
}
local UNITS = { one = 1, two = 2, three = 3, four = 4, five = 5, six = 6, seven = 7, eight = 8, nine = 9 }
local TEENS = { zero = 0, ten = 10, eleven = 11, twelve = 12, thirteen = 13, fourteen = 14, fifteen = 15, sixteen = 16,
  seventeen = 17, eighteen = 18, nineteen = 19 }
local TENS = { twenty = 20, thirty = 30, forty = 40, fourty = 40, fifty = 50, sixty = 60, seventy = 70, eighty = 80, ninety = 90 }

-- Quantity phrases (matched on the words BEFORE a noun). count = integer; 0 = everything that exists.
local QUANT_SRC = {
  { "half a dozen of", 6 }, { "half a dozen", 6 }, { "a couple of dozen", 24 }, { "a couple dozen", 24 }, { "a few dozen", 36 },
  { "a couple of", 2 }, { "a couple", 2 },
  { "a pair of", 2 }, { "a pair", 2 }, { "a trio of", 3 }, { "a trio", 3 }, { "a quartet of", 4 }, { "a quartet", 4 }, { "a duo of", 2 }, { "a few", 3 }, { "quite a few", 3 }, { "a handful of", 5 }, { "a handful", 5 },
  { "a bunch of", 6 }, { "a bunch", 6 }, { "a lot of", 6 }, { "lots of", 6 }, { "loads of", 6 }, { "a load of", 6 },
  { "plenty of", 6 }, { "a ton of", 6 }, { "tons of", 6 }, { "a dozen of", 12 }, { "a dozen", 12 }, { "dozens of", 12 },
  { "a hundred", 100 }, { "one hundred", 100 }, { "hundreds of", 100 }, { "a thousand", 1000 }, { "thousands of", 1000 },
  { "a number of", 4 }, { "several", 4 }, { "some", 3 }, { "both of the", 2 }, { "both the", 2 }, { "both", 2 }, { "many", 6 },
  { "numerous", 6 }, { "all of the", 0 }, { "all the", 0 }, { "every single", 0 }, { "every", 0 }, { "each of the", 0 },
  { "each", 0 }, { "all of them", 0 }, { "all", 0 },
  { "another", 1 }, { "one more", 1 }, { "two more", 2 }, { "three more", 3 }, { "an extra", 1 }, { "one extra", 1 },
  { "an additional", 1 }, { "one additional", 1 }, { "yet another", 1 },
  { "a single", 1 }, { "single", 1 }, { "an", 1 }, { "a", 1 },
  -- intensified quantity nouns: "a whole bunch of", "a large number of", "a good few", "loads and loads of"
  { "quite a bunch of", 6 }, { "quite a lot of", 6 }, { "a whole lot of", 6 }, { "a whole load of", 6 }, { "a whole bunch of", 6 },
  { "a good few", 4 }, { "a fair few", 4 }, { "a good many", 6 }, { "a great many", 6 }, { "a good number of", 4 },
  { "a fair number of", 4 }, { "a decent number of", 4 }, { "a large number of", 6 }, { "a huge number of", 6 },
  { "a great number of", 6 }, { "a great deal of", 6 }, { "a good bunch of", 6 }, { "a big bunch of", 6 },
  { "loads and loads of", 6 }, { "lots and lots of", 6 }, { "tons and tons of", 6 }, { "heaps of", 6 }, { "masses of", 6 },
}
local QUANT_FIRST, QUANT_LAST = index_phrases((function()
  local l = {}
  for _, q in ipairs(QUANT_SRC) do l[#l + 1] = { text = q[1], count = q[2] } end
  return l
end)(), function(e) return e.text end)
local DEFINITE_QUANT = { ["both"] = true, ["both of the"] = true, ["both the"] = true, ["all"] = true, ["all the"] = true,
  ["all of the"] = true, ["every"] = true, ["every single"] = true, ["each"] = true, ["each of the"] = true, ["all of them"] = true }

local ARTICLES = { the = true, this = true, that = true, these = true, those = true, my = true, your = true, our = true, its = true }
local DEFINITE_WORDS = { the = true, this = true, that = true, these = true, those = true, my = true, your = true, our = true, them = true, it = true }
local DETERMINERS = set{ "the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "its", "each", "every", "all", "both",
  "either", "any", "some", "another", "other", "same" }

-- Relations: phrase -> arrangement, whether a target noun follows, and optional flags.
--   flag "prev": the target is the thing named just before ("a sofa with a lamp on each side"); "rev": the items follow the target
--   ("a couch flanked by two lamps"); "each": a distributive phrase ("in each corner"); "none": changes nothing, just not a thing.
local RELATION_SRC = {
  { "in front of", "in_front_of", true }, { "in the front of", "in_front_of", true }, { "ahead of", "in_front_of", true },
  { "on top of each other", "stacked", false }, { "on top of one another", "stacked", false },
  { "on top of", "on", true }, { "on the top of", "on", true }, { "up on", "on", true }, { "on", "on", true }, { "onto", "on", true },
  { "next to", "beside", true }, { "right next to", "beside", true }, { "beside", "beside", true }, { "alongside", "beside", true },
  { "close to", "beside", true }, { "near to", "beside", true }, { "near", "beside", true }, { "by the side of", "beside", true },
  { "at the side of", "beside", true }, { "to the left of", "beside", true }, { "to the right of", "beside", true },
  { "left of", "beside", true }, { "right of", "beside", true }, { "on the left of", "beside", true }, { "on the right of", "beside", true },
  { "on the left side of", "beside", true }, { "on the right side of", "beside", true }, { "to the side of", "beside", true },
  { "on the left hand side of", "beside", true }, { "on the right hand side of", "beside", true },
  { "on the other side of", "beside", true }, { "on the opposite side of", "beside", true }, { "on the far side of", "beside", true },
  { "on the near side of", "beside", true }, { "on this side of", "beside", true }, { "on that side of", "beside", true },
  { "at the other end of", "beside", true }, { "on the other end of", "beside", true }, { "at the far end of", "beside", true },
  { "on the left side", "none", false }, { "on the right side", "none", false }, { "on the left", "none", false },
  { "on the right", "none", false }, { "to the left", "none", false }, { "to the right", "none", false },
  { "on the front side", "none", false }, { "on the back side", "none", false },
  { "behind", "behind", true }, { "in back of", "behind", true }, { "at the back of", "behind", true },
  { "around", "around", true }, { "surrounding", "around", true }, { "encircling", "around", true },
  { "facing each other", "facing_each_other", false }, { "face each other", "facing_each_other", false },
  { "faces each other", "facing_each_other", false },
  { "to face each other", "facing_each_other", false }, { "to face one another", "facing_each_other", false },
  { "facing one another", "facing_each_other", false }, { "face one another", "facing_each_other", false },
  { "opposite each other", "facing_each_other", false }, { "across from each other", "facing_each_other", false },
  { "face to face", "facing_each_other", false }, { "facing opposite", "facing_each_other", false },
  { "facing away from", "facing_away", true }, { "face away from", "facing_away", true }, { "faces away from", "facing_away", true },
  { "turned away from", "facing_away", true }, { "looking away from", "facing_away", true }, { "pointing away from", "facing_away", true },
  { "pointed away from", "facing_away", true }, { "to face away from", "facing_away", true },
  { "with their backs to", "facing_away", true }, { "with its back to", "facing_away", true }, { "with the back to", "facing_away", true },
  { "with their back to", "facing_away", true }, { "with backs to", "facing_away", true }, { "with back to", "facing_away", true },
  { "backs to", "facing_away", true }, { "their backs to", "facing_away", true },
  { "facing away", "facing_away", false }, { "face away", "facing_away", false }, { "faces away", "facing_away", false },
  { "facing outward", "facing_away", false }, { "facing outwards", "facing_away", false }, { "facing out", "facing_away", false },
  { "face outward", "facing_away", false }, { "face outwards", "facing_away", false }, { "face out", "facing_away", false },
  { "faces outward", "facing_away", false }, { "looking outward", "facing_away", false }, { "looking outwards", "facing_away", false },
  { "pointing outward", "facing_away", false }, { "pointing outwards", "facing_away", false }, { "outward facing", "facing_away", false },
  { "facing inward", "facing_me", false, "center" }, { "facing inwards", "facing_me", false, "center" }, { "face inward", "facing_me", false, "center" },
  { "face inwards", "facing_me", false, "center" }, { "inward", "facing_me", false, "center" }, { "inwards", "facing_me", false, "center" },
  { "facing the inside", "facing_me", false, "center" }, { "facing in", "facing_me", false, "center" },
  { "facing towards", "facing_me", true }, { "facing toward", "facing_me", true }, { "turned towards", "facing_me", true },
  { "turned toward", "facing_me", true }, { "pointing at", "facing_me", true }, { "pointed at", "facing_me", true },
  { "pointing towards", "facing_me", true }, { "pointing toward", "facing_me", true },
  { "to face", "facing_me", true }, { "facing", "facing_me", true }, { "face", "facing_me", true }, { "faces", "facing_me", true },
  { "faced", "facing_me", true }, { "towards", "facing_me", true },
  { "toward", "facing_me", true }, { "looking at", "facing_me", true }, { "look at", "facing_me", true }, { "looks at", "facing_me", true },
  { "looking toward", "facing_me", true }, { "looking towards", "facing_me", true }, { "points at", "facing_me", true },
  { "on both sides of", "flank", true }, { "on either side of", "flank", true }, { "on each side of", "flank", true },
  { "at both sides of", "flank", true }, { "at either side of", "flank", true }, { "at each side of", "flank", true },
  { "on both sides", "flank", false, "prev" }, { "on either side", "flank", false, "prev" }, { "on each side", "flank", false, "prev" },
  { "at both sides", "flank", false, "prev" }, { "at either side", "flank", false, "prev" }, { "at each side", "flank", false, "prev" },
  { "either side of", "flank", true }, { "each side of", "flank", true }, { "both sides of", "flank", true },
  { "at both ends of", "flank", true }, { "at either end of", "flank", true }, { "at each end of", "flank", true },
  { "on both ends of", "flank", true }, { "on either end of", "flank", true }, { "on each end of", "flank", true },
  { "flanking", "flank", true }, { "flank", "flank", true }, { "to flank", "flank", true },
  { "flanked by", "flank", false, "rev" }, { "bookended by", "flank", false, "rev" }, { "bracketed by", "flank", false, "rev" },
  { "against the wall", "against_wall", false }, { "against a wall", "against_wall", false }, { "against the walls", "against_wall", false },
  { "along the wall", "against_wall", false }, { "along a wall", "against_wall", false }, { "along the walls", "against_wall", false },
  { "by the wall", "against_wall", false }, { "on the wall", "against_wall", false }, { "up against the wall", "against_wall", false },
  { "to the wall", "against_wall", false }, { "to the walls", "against_wall", false }, { "against", "against_wall", true },
  { "in the corner", "corner", false }, { "in a corner", "corner", false }, { "in the corners", "corner", false },
  { "in each corner", "corner", false, "each" }, { "in every corner", "corner", false, "each" }, { "in corners", "corner", false },
  { "to the corner", "corner", false }, { "to a corner", "corner", false }, { "in the room's corner", "corner", false },
  { "in a row", "row", false }, { "in rows", "row", false }, { "in a line", "row", false }, { "in line", "row", false },
  { "in a straight line", "row", false }, { "in a single row", "row", false }, { "in one row", "row", false },
  { "side by side", "row", false }, { "next to each other", "row", false }, { "beside each other", "row", false },
  { "lined up", "row", false }, { "in a queue", "row", false }, { "in a single file", "row", false }, { "in a long row", "row", false },
  { "in a circle", "circle", false }, { "in a ring", "circle", false }, { "in a semicircle", "circle", false },
  { "in a semi circle", "circle", false }, { "in a half circle", "circle", false }, { "in a round", "circle", false },
  { "in a circular", "circle", false },
  { "in a circular arrangement", "circle", false }, { "in a loop", "circle", false }, { "in a big circle", "circle", false },
  { "in a u shape", "u_shape", false }, { "in a u", "u_shape", false }, { "in a u formation", "u_shape", false },
  { "u shaped", "u_shape", false }, { "u shape", "u_shape", false }, { "in a horseshoe", "u_shape", false }, { "horseshoe", "u_shape", false },
  { "in an l shape", "l_shape", false }, { "in an l", "l_shape", false }, { "l shaped", "l_shape", false }, { "l shape", "l_shape", false },
  { "in a grid", "grid", false }, { "in a square", "grid", false }, { "in a block", "grid", false }, { "in columns", "grid", false },
  { "in rows and columns", "grid", false }, { "in a matrix", "grid", false }, { "in a lattice", "grid", false }, { "grid", "grid", false },
  { "a circle of", "circle", false }, { "a ring of", "circle", false }, { "a row of", "row", false }, { "a line of", "row", false },
  { "a grid of", "grid", false }, { "a stack of", "stacked", false }, { "a pile of", "stacked", false }, { "a pyramid of", "stacked", false },
  { "a u shape of", "u_shape", false }, { "a semicircle of", "circle", false }, { "a horseshoe of", "u_shape", false },
  { "a heap of", "stacked", false }, { "a tower of", "stacked", false }, { "a column of", "stacked", false },
  { "a wall of", "row", false }, { "straight", "row", false }, { "a circle with", "circle", false }, { "a ring with", "circle", false }, { "a row with", "row", false },
  { "a line with", "row", false }, { "pile up", "stacked", false }, { "heap up", "stacked", false },
  { "im looking at", "none", false }, { "i am looking at", "none", false }, { "youre looking at", "none", false },
  { "were looking at", "none", false }, { "i m looking at", "none", false },
  { "everywhere", "scattered", false }, { "anywhere", "scattered", false }, { "all around the room", "scattered", false },
  { "scattered", "scattered", false }, { "randomly", "scattered", false }, { "at random", "scattered", false },
  { "all over the place", "scattered", false }, { "all over", "scattered", false }, { "around the room", "scattered", false },
  { "about the room", "scattered", false }, { "throughout the room", "scattered", false }, { "here and there", "scattered", false },
  { "spread out", "scattered", false }, { "spaced out", "scattered", false }, { "spread around", "scattered", false },
  { "scatter", "scattered", false },
  { "stacked", "stacked", false }, { "stack", "stacked", false }, { "stack up", "stacked", false }, { "piled up", "stacked", false },
  { "piled", "stacked", false }, { "in a pile", "stacked", false }, { "in a stack", "stacked", false }, { "in a pyramid", "stacked", false },
  { "in a heap", "stacked", false },
}
-- soft relations: they only count when a thing (or window / door) follows ("put it by the couch" but "by the way", "over there")
local SOFT_SRC = {
  { "over by", "beside" }, { "over to", "beside" }, { "up to", "beside" }, { "close by", "beside" }, { "over near", "beside" },
  { "to", "beside" }, { "at", "beside" }, { "by", "beside" }, { "under", "beside" }, { "underneath", "beside" }, { "beneath", "beside" },
  { "below", "beside" }, { "inside", "beside" }, { "in", "beside" }, { "within", "beside" }, { "between", "beside", "between" },
  { "for", "beside", "for" }, { "over", "on" }, { "above", "on" }, { "atop", "on" },
  { "opposite", "facing_me" }, { "across from", "facing_me" }, { "across the table from", "facing_me" },
  { "at the foot of", "in_front_of" }, { "at the end of", "in_front_of" }, { "at the head of", "behind" },
  { "at the bottom of", "in_front_of" }, { "at the top of", "behind" }, { "across", "beside" },
  { "away from", "away", "away" }, { "off of", "away", "away" }, { "off", "away", "away" }, { "back to", "facing_away", "backto" },
  { "from", "away", "away" }, { "out of", "away", "away" },
  { "to the front of", "in_front_of" }, { "to the back of", "behind" }, { "at the rear of", "behind" }, { "in the rear of", "behind" },
}
local REL_FIRST = index_phrases((function()
  local l = {}
  for _, r in ipairs(RELATION_SRC) do l[#l + 1] = { text = r[1], arr = r[2], needs = r[3], flag = r[4] } end
  return l
end)(), function(e) return e.text end)
local SOFT_FIRST = index_phrases((function()
  local l = {}
  for _, r in ipairs(SOFT_SRC) do l[#l + 1] = { text = r[1], arr = r[2], flag = r[3], needs = true, soft = true } end
  return l
end)(), function(e) return e.text end)

-- walls and corners ("against the back wall", "in the far corner", "in each corner")
local WALL_PREPS = set{ "against", "along", "by", "on", "to", "at" }
local CORNER_PREPS = set{ "in", "into", "to", "at" }
local PLACE_DETS = set{ "the", "a", "an", "that", "this", "my", "our", "each", "every", "all", "both" }
local PLACE_MODS = set{ "far", "back", "rear", "left", "right", "front", "north", "south", "east", "west", "northern", "southern",
  "eastern", "western", "nearest", "nearby", "near", "opposite", "other", "side", "closest", "farthest", "furthest", "whole", "long",
  "short", "empty", "northeast", "northwest", "southeast", "southwest", "upper", "lower", "top", "bottom", "very", "hand", "four", "4",
  "two", "2", "three", "3", "free", "spare", "dark", "bright", "sunny", "remote", "distant" }

local WORLD_TARGETS = { wall = "wall", walls = "wall", window = "window", windows = "window", door = "door", doors = "door",
  corner = "corner", corners = "corner", ceiling = "ceiling", floor = "floor", ground = "floor", room = "room", fire = "fire",
  center = "center", centre = "center", middle = "center" }
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
  ["would"] = true, ["could"] = true, ["can"] = true, ["will"] = true, ["should"] = true, ["shall"] = true, ["might"] = true,
  ["may"] = true, ["must"] = true, ["do"] = true, ["does"] = true, ["did"] = true, ["was"] = true, ["were"] = true, ["am"] = true,
  ["has"] = true, ["had"] = true, ["which"] = true, ["who"] = true, ["whom"] = true, ["what"] = true, ["than"] = true,
  ["about"] = true, ["us"] = true, ["let"] = true, ["lets"] = true, ["whose"] = true, ["its"] = true, ["been"] = true,
  ["uh"] = true, ["um"] = true, ["umm"] = true, ["uhh"] = true, ["er"] = true, ["erm"] = true, ["hmm"] = true, ["hm"] = true,
  ["oh"] = true, ["ah"] = true, ["lol"] = true, ["haha"] = true,
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
  furniture = true, lot = true, house = true, place = true, people = true, again = true, anymore = true, persons = true, guests = true, diners = true, folks = true,
  versions = true, version = true, copy = true, copies = true, kind = true, kinds = true, sort = true, type = true, types = true,
  shape = true, pattern = true, style = true, fashion = true, manner = true, form = true,
}
-- measurement words: "3 feet to the left", "rotate 90 degrees" -- never things to summon
local UNIT_WORDS = set{ "high", "tall", "deep", "layers", "layer", "levels", "level", "tiers", "tier", "turn", "turns", "quarter", "half", "full", "revolution", "rotation", "clockwise", "counterclockwise",
  "anticlockwise", "counter", "anti", "notch", "notches", "little", "bit", "feet", "foot", "ft", "meter", "meters", "metre", "metres", "m", "cm", "centimeter", "centimeters",
  "inch", "inches", "yard", "yards", "unit", "units", "hu", "degree", "degrees", "deg", "percent", "steps", "step", "mm", "km" }
local COLORS = set{ "red", "blue", "green", "yellow", "orange", "pink", "purple", "violet", "brown", "black", "white", "grey", "gray",
  "cyan", "teal", "gold", "silver", "beige", "tan", "navy", "maroon", "turquoise", "lime", "crimson", "golden", "magenta", "cream",
  "ivory", "indigo", "olive", "coral", "khaki", "scarlet" }
local SIZE_WORDS = set{ "big", "large", "huge", "giant", "long", "wide", "bigger", "larger", "biggest", "largest", "massive", "enormous",
  "king", "small", "little", "tiny", "short", "mini", "narrow", "compact", "smaller", "smallest", "petite", "tall", "taller", "tallest",
  "wider", "longer", "shorter", "thin", "thick", "fat", "medium", "normal", "regular", "standard" }
local STYLE_WORDS = set{ "office", "antique", "old", "vintage", "kitchen", "lab", "garden", "outdoor", "patio", "military", "army",
  "prison", "jail", "hospital", "train", "station", "cafeteria", "wooden", "wood", "combine", "wasteland", "nice", "comfy",
  "comfortable", "cozy", "cosy", "fancy", "modern", "cheap", "plain", "simple", "new", "newer", "used", "broken", "shiny", "clean", "dirty",
  "dusty", "lovely", "beautiful", "ugly", "cute", "rusty", "fresh", "round", "rectangular", "soft", "hard", "heavy", "bright", "dark", "warm",
  "cold", "empty", "full", "spare", "extra", "metallic", "elegant", "stylish", "classic", "rustic", "sturdy", "tidy", "neat" }
local INTENSIFIERS = set{ "quite", "very", "really", "extremely", "pretty", "fairly", "rather", "totally", "super", "kinda", "sorta",
  "somewhat", "slightly", "reasonably", "truly" }
local NOISE_ADJ = set{ "more", "extra", "additional", "other", "another", "whole", "entire", "brand", "new" }
local ADJECTIVE_WORDS = {}
for _, t in ipairs({ COLORS, SIZE_WORDS, STYLE_WORDS, INTENSIFIERS }) do for w in pairs(t) do ADJECTIVE_WORDS[w] = true end end

local VERB_ADD = { add = true, spawn = true, summon = true, create = true, make = true, put = true, place = true, give = true,
  get = true, bring = true, build = true, generate = true, materialize = true, materialise = true, conjure = true, set = true,
  throw = true, drop = true, show = true, install = true, insert = true, produce = true, fetch = true, offer = true,
  provide = true, load = true, populate = true, furnish = true, stick = true, plop = true, pop = true, whip = true, rustle = true,
  grab = true, lay = true, render = true, deliver = true, need = true, want = true, like = true, drag = true, pull = true,
  duplicate = true, copy = true, clone = true, hand = true, have = true, let = true, take = true, send = true, fill = true, stock = true,
  equip = true, outfit = true }
-- verbs that mean "move something that already exists" when the thing is a definite one ("put THE lamp next to ...")
local VERB_MOVEISH = { put = true, bring = true, get = true, take = true, drag = true, pull = true, push = true, set = true,
  place = true, shift = true, carry = true, drop = true, stick = true, lay = true, move = true, slide = true, shove = true,
  send = true, hand = true }
-- "make the chairs face the table": the verb only causes a change, so definite things plus a facing are turned in place
local VERB_CAUSE = set{ "make", "have", "let", "get", "want", "should", "keep" }
local VERB_TWO = set{ "set up", "throw in", "drop in", "bring in", "put in", "pull up", "whip up", "rustle up", "give me", "get me",
  "bring me", "make me", "pop in", "stick in", "pull in", "fit out", "kit out", "show me", "hand me", "fetch me", "build me",
  "put me", "grab me", "spawn me", "summon me", "conjure me", "create me", "add me", "cook up", "knock up" }

-- verb phrases by class. Order of the classes below is their priority when two phrases of equal length start at the same word.
local VERB_SRC = {
  remove = { "get rid of", "get rid", "take away", "take out", "throw away", "throw out", "do away with", "clear away", "pick up",
    "remove", "delete", "erase", "destroy", "dismiss", "banish", "dematerialize", "dematerialise", "despawn", "vanish", "kill",
    "discard", "ditch", "lose", "scrap", "eliminate", "zap", "disintegrate", "evaporate", "clear", "wipe out", "wipe", "scratch",
    "nix", "toss", "chuck", "dispose of", "clean up", "clean out", "sweep away", "send away", "purge", "nuke", "put away",
    "tidy away", "clear out", "pack away", "take off", "undo", "cancel", "uncreate", "unspawn", "bin", "trash" },
  rotate = { "rotate", "turn", "spin", "swivel", "flip", "pivot", "twist", "orient", "angle", "reorient", "swing", "point", "aim" },
  rearrange = { "rearrange", "re arrange", "reorganize", "reorganise", "reorder", "tidy up", "tidy", "regroup", "straighten up",
    "straighten", "line up", "organize", "organise", "realign", "re align", "shuffle", "redo the layout", "redo", "neaten", "arrange",
    "sort out", "sort" },
  move = { "move", "shift", "slide", "relocate", "nudge", "reposition", "push", "pull closer", "bring closer", "bring over",
    "move over", "scoot", "drag", "carry", "transfer", "shove", "gather", "pull back", "push back", "bump", "inch", "haul" },
  -- things the holodeck cannot do (yet): the answer is "none", with the verb, never an add
  unsupported = { "paint", "repaint", "recolor", "recolour", "color", "colour", "dye", "tint", "resize", "enlarge", "shrink", "grow",
    "widen", "lengthen", "swap", "switch", "replace", "exchange", "change", "convert", "transform", "group", "ungroup", "open",
    "close", "lock", "unlock", "fix", "repair", "break", "smash", "light", "ignite", "extinguish", "use", "power", "plug",
    "unplug", "hide", "rename", "label", "scale", "stretch", "lift", "raise", "lower", "tilt", "lean", "wash", "sit", "revert",
    "turn off", "turn on", "turn up", "turn down", "switch off", "switch on", "power off", "power on", "shut", "shut down",
    "polish", "upgrade", "name", "duplicate it", "tie", "attach", "glue", "weld", "heat", "cool",
    "freeze", "melt", "burn", "shatter", "knock over", "tip over", "flatten", "bend", "inflate", "deflate", "dust", "center", "centre",
    "align", "even", "balance", "equalize", "distribute", "hold", "carry on", "pause", "play", "stop", "zoom", "look", "reset", "restore", "skip",
    "refresh", "revive", "reload" },
  add = {},
}
-- forms of the verbs that people actually say: "removing the lamp", "putting a lamp"
local DOUBLE_LAST = set{ "put", "get", "set", "drag", "drop", "plop", "pop", "zap", "whip", "grab", "scrap", "shop", "sit", "nix", "tip",
  "bump", "trim", "chop", "flip", "swap" }
local function ing_form(w)
  if w:match("ie$") then return w:sub(1, -3) .. "ying" end
  if w:match("[^e]e$") then return w:sub(1, -2) .. "ing" end
  if DOUBLE_LAST[w] then return w .. w:sub(-1) .. "ing" end
  return w .. "ing"
end
local VERB_CLASSES = { "remove", "rotate", "rearrange", "move", "unsupported", "add" }
local VERB_PRIO = {}
for i, c in ipairs(VERB_CLASSES) do VERB_PRIO[c] = i end
local VERB_ENTRIES = {}
do
  local seen = {}
  local function add_entry(text, class, base)
    local key = text
    if seen[key] and VERB_PRIO[seen[key]] <= VERB_PRIO[class] then return end
    seen[key] = class
    VERB_ENTRIES[#VERB_ENTRIES + 1] = { text = text, class = class, base = base }
  end
  for class, list in pairs(VERB_SRC) do
    for _, p in ipairs(list) do
      add_entry(p, class, p)
      local w = words_of(p)
      if class ~= "unsupported" or #w == 1 then
        w[1] = ing_form(w[1])
        add_entry(table.concat(w, " "), class, p)
      end
    end
  end
  for v in pairs(VERB_ADD) do
    add_entry(v, "add", v)
    add_entry(ing_form(v), "add", v)
  end
  -- "put away" etc. are in the remove list already; "give me a couch" etc.
  for p in pairs(VERB_TWO) do add_entry(p, "add", p) end
end
-- the table above is a list; resolve duplicates by priority (a word listed as remove and add is a remove verb)
local VERB_FIRST = index_phrases(VERB_ENTRIES, function(e) return e.text end)
for _, l in pairs(VERB_FIRST) do
  table.sort(l, function(a, b)
    if a.n ~= b.n then return a.n > b.n end
    if VERB_PRIO[a.class] ~= VERB_PRIO[b.class] then return VERB_PRIO[a.class] < VERB_PRIO[b.class] end
    return a.text < b.text
  end)
end

-- words that may come before the verb without being part of the request
local LEAD = set{ "to", "you", "please", "could", "can", "would", "will", "should", "just", "now", "then", "also", "and", "lets", "us",
  "i", "we", "id", "im", "want", "wanna", "need", "think", "maybe", "perhaps", "kindly", "really", "actually", "so", "ok", "okay",
  "well", "nice", "cool", "great", "sure", "yes", "yeah", "oh", "hmm", "um", "uh", "thanks", "alright", "right", ",", "gonna",
  "if", "able", "mind", "possible", "ahead", "go", "try" }

local UNDO_SRC = { "undo", "take that back", "take it back", "scratch that", "never mind", "nevermind", "nvm", "revert", "put it back",
  "put them back", "that was a mistake", "go back", "oops", "whoops", "cancel that", "cancel", "ctrl z", "undo that", "undo it",
  "undo the last", "undo last", "roll back", "rollback", "take back", "forget that", "forget it", "ignore that", "my bad",
  "that was wrong", "that was the wrong one", "wrong one", "not that", "put that back", "step back" }
local UNDO_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(UNDO_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
local UNDO_FILLER = set{ "that", "it", "them", "the", "last", "one", "thing", "things", "again", "please", "just", "now", "i", "mean",
  "meant", "sorry", "this", "those", "these", "all", "and", "then", "so", "actually", "wait", "uh", "um", "hmm", "a", "there", "thanks",
  "was", "previous", "ok", "okay", "my", "ones", "stuff", "change", "changes", "action", "move", "step", "oh", "no", "yes", "yeah",
  "sorry" }

-- A retraction at the start of a sentence ("oops, I meant a table", "scratch that, add a lamp") takes the last thing back; what
-- follows it, if it asks for something, is a request of its own.
local RETRACT_SRC = { "never mind", "nevermind", "nvm", "scratch that", "oops", "whoops", "forget that", "forget it", "cancel that",
  "take that back", "ignore that", "my bad", "i meant", "i mean", "make that", "no wait", "wait no", "no i meant", "no make that",
  "sorry i meant", "oh i meant", "or rather", "sorry", "wait", "actually no" }
local RETRACT_STRICT = set{ "make that" }                       -- "make that couch red" is a recolour: only "make that two chairs"
local RETRACT_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(RETRACT_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)

--- Where the retraction phrases at the start of tokens end (the index after them), and whether one of them was a strict one.
local function retraction_end(tokens)
  local i, any, strict = 1, false, false
  while true do
    local e = match_at(RETRACT_FIRST, tokens, i)
    if e then
      any = true
      if RETRACT_STRICT[e.text] then strict = true end
      i = i + e.n
    elseif tokens[i] == "," and any then i = i + 1
    else break end
  end
  if not any then return nil end
  return i, strict
end

-- Whole-sentence clears. A clear is verb + whole-room object + politeness, nothing else: "clear the floor lamp" is a removal.
local CLEAR_VERBS = set{ "clear", "remove", "delete", "erase", "wipe", "dismiss", "destroy", "nuke", "scrap", "empty", "reset", "clean",
  "dematerialize", "dematerialise", "purge", "banish", "zap", "kill", "trash", "bin", "ditch", "discard", "lose", "eliminate", "obliterate",
  "annihilate", "vanish", "disintegrate", "evaporate" }
local CLEAR_VERB_PHRASES = { "get rid of", "take away", "take out", "throw away", "throw out", "clear away", "clear out", "wipe out",
  "clean up", "clean out", "do away with", "sweep away", "pack away", "make go away" }
local CLEAR_OBJECTS = { "everything", "all", "it all", "all of it", "everything out", "everything away", "the lot", "the whole lot",
  "the room", "the whole room", "the area", "the scene", "the holodeck", "the floor", "the space", "the stage", "the place", "the world",
  "the simulation", "the program", "it out", "the furniture", "all the furniture", "all furniture", "all my stuff", "all the stuff",
  "my stuff", "the stuff", "everything here", "everything in here", "everything in the room", "everything out of here",
  "all of the furniture", "everything there", "the house", "the whole place", "the whole thing", "everything up", "this room",
  "this place", "the whole area", "the whole scene", "the whole floor", "the entire room", "the entire area", "the entire scene",
  "all the things", "all things" }
local CLEAR_ALONE = { "end program", "end the program", "end simulation", "end the simulation", "end holodeck", "end the holodeck",
  "arch", "start over", "start again", "start from scratch", "begin again", "clean slate", "a clean slate", "blank slate",
  "make it all go away", "make everything go away", "thats enough", "that is enough", "enough already", "reset",
  "clear", "wipe", "clean up", "clear up", "clear out", "clean out", "wipe out", "empty it", "empty everything", "restart", "reboot",
  "wipe the slate clean", "start fresh", "start anew", "begin anew" }
local CLEAR_POLITE = set{ "out", "away", "now", "please", "already", "again", "here", "up", "thanks", "right", "quickly", "then", "also",
  "and", "for", "me", "just", "there", "completely", "entirely", "totally", "altogether", "all", "ok", "okay", "pls", "plz" }
local CLEAR_OBJ_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(CLEAR_OBJECTS) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
local CLEAR_ALONE_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(CLEAR_ALONE) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
local CLEAR_VERBP_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(CLEAR_VERB_PHRASES) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)

-- "everything" as the OBJECT of a management verb: every standing thing the player may manage
local ALL_SRC = { "everything", "all of it", "it all", "the lot", "the whole lot", "all the furniture", "all of the furniture",
  "all furniture", "the furniture", "all my furniture", "my furniture", "all my stuff", "all the stuff", "all of the stuff",
  "all of my stuff", "my stuff", "everything here", "everything in here", "everything in the room", "everything there",
  "all the things", "all things", "all the items", "all items", "the whole room", "the room", "all the props", "all props",
  "everything standing", "everything out here", "all the objects", "the entire room", "everything around", "everything around here",
  "each and everything", "the whole lot of it", "anything", "all", "everything else", "all else" }
local ALL_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(ALL_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
local REST_SRC = { "the rest", "the others", "all the others", "all the rest", "the rest of it", "the rest of them", "the remaining ones",
  "the remaining", "everything else", "all else", "whatever else", "whatever is left", "whatevers left", "all of the others",
  "the other stuff", "the other things", "the other ones", "the rest of the stuff", "the rest of the furniture", "all the other stuff",
  "all the other things", "all other stuff", "all others", "everything left", "anything else", "all of the rest" }
local REST_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(REST_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)

-- the exception markers of "everything except the couch"
local EXCEPT_SRC = { "except for", "except", "other than", "apart from", "aside from", "save for", "excluding", "not counting",
  "with the exception of", "besides", "barring", "but not", "minus", "all but", "everything but", "anything but", "none but",
  "but keep", "but leave", "but save", "and keep", "and leave", "leaving", "just not", "not including", "but" }
local EXCEPT_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(EXCEPT_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)

-- negators inside an add request: the noun phrase right after them is NOT wanted
local NEGATORS = { "without", "but no", "but not", "instead of", "rather than", "not", "no", "minus", "sans", "excluding", "and no",
  "with no", "and not", "nor", "free of", "lacking", "no more" }
local NEG_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(NEGATORS) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)

-- "the last one", "what I just made": the most recently made group
local LAST_HEADS = set{ "one", "ones", "thing", "things", "stuff", "group", "set", "batch", "addition", "additions", "creation", "creations" }
local LAST_MODS = set{ "last", "previous", "newest", "latest", "new", "recent", "prior", "final" }
local MADE_VERBS = set{ "made", "built", "added", "put", "created", "spawned", "placed", "summoned", "asked", "requested", "conjured",
  "generated", "set", "dropped", "produced", "got", "gave", "did" }

-- ---------------------------------------------------------------------------------------------- numbers
local function number_of(tok)
  if tok == nil then return nil end
  if tok:match("^%d+$") then
    if #tok > 4 then return COUNT_CAP end
    return math.min(tonumber(tok), COUNT_CAP)
  end
  return NUMBER_WORDS[tok]
end

--- "twenty five" -> "25", "three hundred" -> "300", "two dozen" -> "24" (one token per number, so every later step sees one).
local function fold_numbers(tokens)
  local out, i, n = {}, 1, #tokens
  local function below_hundred(j)                       -- value, next index (nil when no number starts at j)
    local t = tokens[j]
    if TENS[t] then
      local u = UNITS[tokens[j + 1]]
      if u then return TENS[t] + u, j + 2, true end
      return TENS[t], j + 1, false
    end
    if TEENS[t] then return TEENS[t], j + 1, false end
    if UNITS[t] then return UNITS[t], j + 1, false end
    if t and t:match("^%d%d?$") then return tonumber(t), j + 1, false end
  end
  while i <= n do
    local t = tokens[i]
    local v, j, compound = below_hundred(i)
    if t and t:match("^%d%d%d?%d?$") and not v then v, j = tonumber(t), i + 1 end
    if v then
      local used = compound
      if tokens[j] == "hundred" and v >= 1 and v <= 99 then
        v = v * 100; j = j + 1; used = true
        local k = j
        if tokens[k] == "and" then k = k + 1 end
        local rest, nj = below_hundred(k)
        if rest and rest > 0 then v = v + rest; j = nj end
      end
      if tokens[j] == "thousand" and v >= 1 and v <= 999 then
        v = v * 1000; j = j + 1; used = true
        local rest, nj = below_hundred(j)
        if rest and rest > 0 then v = v + rest; j = nj end
      end
      if tokens[j] == "dozen" and v >= 1 and v <= 99 and not (tokens[i - 1] == "a" and i == j - 1) then
        v = v * 12; j = j + 1; used = true
      end
      if used then
        out[#out + 1] = tostring(math.min(v, COUNT_CAP))
        i = j
      else
        out[#out + 1] = t
        i = i + 1
      end
    else
      out[#out + 1] = t
      i = i + 1
    end
  end
  return out
end

-- ---------------------------------------------------------------------------------------------- text hygiene
local function cut_utf8(s, n)
  if #s <= n then return s end
  s = s:sub(1, n)
  local i, back = #s, 0
  while i > 0 and back < 4 do                                       -- drop a character cut in half
    local b = s:byte(i)
    if b >= 192 then
      local need = (b >= 240) and 4 or ((b >= 224) and 3 or 2)
      if #s - i + 1 < need then s = s:sub(1, i - 1) end
      break
    elseif b >= 128 then i = i - 1; back = back + 1
    else break end
  end
  return s
end

--- Fold what people paste and speech engines emit into plain ASCII words: Unicode spaces, quotes, dashes, ellipses, full-width letters.
local function fold_text(s)
  s = s:gsub("\r?\n", " . ")
  s = s:gsub("%c", " ")
  if s:find("[\128-\255]") then
    s = s:gsub("\239([\188\189])([\128-\191])", function(a, b)             -- full-width ASCII (U+FF01 .. U+FF5E)
      local cp = 0xFF00 + (a:byte() - 0xBC) * 64 + (b:byte() - 0x80)
      local ascii = cp - 0xFEE0
      if ascii >= 0x21 and ascii <= 0x7E then return string.char(ascii) end
      return " "
    end)
    s = s:gsub("\226\128[\152-\155]", "")                                  -- curly single quotes: dropped like an apostrophe
    s = s:gsub("\226\128[\156-\159]", " ")                                 -- curly double quotes
    s = s:gsub("\226\128[\144-\149]", " ")                                 -- dashes
    s = s:gsub("\226\128\166", " , ")                                      -- ellipsis
    s = s:gsub("\226\128[\128-\143]", " ")                                 -- Unicode spaces and zero-width marks
    s = s:gsub("\226\128[\168-\175]", " ")
    s = s:gsub("\226\129[\159-\175]", " ")
    s = s:gsub("\194[\160\173\171\187]", " ")                              -- no-break space, soft hyphen, guillemets
    s = s:gsub("\195\151", " x ")                                          -- the multiplication sign
    s = s:gsub("\227\128\128", " ")                                        -- ideographic space
    s = s:gsub("\227\128[\129\130]", " , ")                                -- CJK comma and full stop
    s = s:gsub("\239\187\191", " ")                                        -- byte order mark
    s = s:gsub("[\240-\247][\128-\191]*", " ")                             -- emoji and other astral characters
  end
  return s
end

local PREFIX_SRC = {
  "hey skippy", "ok skippy", "okay skippy", "skippy", "skipy", "skippie", "skipi", "skippi", "skipee", "skippey", "skippee", "skeepy",
  "computer", "holodeck", "hey computer", "hey holodeck", "hey", "ok", "okay", "alright", "all right", "please", "um", "umm", "uh", "uhh",
  "er", "erm", "hmm", "hm", "well", "so", "now", "and", "then", "actually", "first", "next", "also", "can you", "could you", "would you",
  "will you", "can u", "could u", "can i have", "can we have", "could i have", "could we have", "may i have", "i want you to",
  "i would like you to", "id like you to", "i want", "id like", "i would like", "i need", "i wanna", "i gotta have", "lets", "let us",
  "we need", "we want", "we should have", "go ahead and", "just", "maybe", "perhaps", "try", "how about", "what about",
  "i think i need", "how bout", "show me", "there should be", "there needs to be", "we have", "i have", "need", "gimme", "pls", "plz",
  "kindly", "yo", "dude", "man", "quickly", "real quick", "if you could", "if you can", "if you would",
  "if you dont mind", "i think you should", "i think we should", "i think", "you should", "you could", "you can", "would it be possible to",
  "is it possible to", "would you be able to", "could you be able to", "are you able to", "can you please", "could you please",
  "do you mind", "would you mind", "do you think you could", "i was wondering if you could", "i wonder if you could", "feel free to",
  "i need you to", "we need you to", "its time to", "time to", "lets go ahead and", "i want to", "i would like to", "id like to",
  "gonna", "were gonna", "im going to", "im gonna", "i guess", "ok so", "now lets", "nice", "cool", "great", "awesome", "perfect",
  "sweet", "sure", "yeah", "yes", "yep", "oh", "ah", "wait", "hold on", "hang on", "one sec", "one moment", "ok now", "hi", "hello",
  "oi", "hiya", "howdy", "thanks", "thank you", "good", "i mean", "like", "ok then", "okay then", "right then", "alright then",
  "hey there", "excuse me", "no worries", "ta", "cheers", "ooh", "ooh la la", "whoa", "wow", "listen", "look", "see", "anyway",
  "anyways", "okey", "my bad", "thx", "ty", "on second thought", "come to think of it", "cant you", "couldnt you", "wouldnt you", "why dont you", "why not",
  "dont forget to", "do not forget to", "please dont forget to", "dont forget", "by the way", "oh and", "actually wait",
  "ooh", "so um", "hm okay", "right so", "okay so", "ok so then",
}
local SUFFIX_SRC = { "please", "thanks", "thank you", "pls", "plz", "thx", "ty", "if you can", "for me", "right now", "now", "quickly",
  "asap", "okay", "ok", "real quick", "thanks a lot", "thank you very much", "lol", "lmao", "haha", "hahaha", "hehe", "xd", "if possible",
  "if thats ok", "i guess", "or something", "or whatever", "would be great", "that would be great", "if you dont mind", "ok then",
  "okay then", "for now", "thank you so much", "thanks so much", "much appreciated", "appreciate it", "yeah", "mate", "buddy",
  "dude", "bro", "skippy", "skipy", "skippie", "kindly", "will you", "would you", "can you", "could you",
  "and thanks", "and thank you", "and please", "cheers", "ta", "ok thanks", "okay thanks" }
local PREFIX_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(PREFIX_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
local _, SUFFIX_LAST = index_phrases((function()
  local l = {}
  for _, p in ipairs(SUFFIX_SRC) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
-- a prefix word that is really the start of a request: "next to the wall", "computer desk", "like a circle"
local PREFIX_NOT_BEFORE = { next = set{ "to" }, computer = set{ "desk", "chair", "monitor", "table", "and", "on", "in", "is", "next" },
  like = set{ "a", "an", "the", "this", "that", "so", "i" }, so = set{ "that" }, look = set{ "at" }, see = set{ "that" },
  good = set{ "chair", "couch", "table" }, well = set{ "lit" }, first = set{ "chair", "couch", "table" }, wait = set{ "room" },
  man = set{ "cave" } }
-- a clause-opening word that is also a noun: only a prefix when a comma follows ("computer, add three couches")
local PREFIX_NEEDS_COMMA = set{ "look", "see", "like", "good", "sorry", "listen", "man", "sure", "wait" }

--- Lower-case words (commas are words of their own) with the polite padding stripped from both ends. Linear in the length.
local function tokenise(raw)
  local s = fold_text(tostring(raw or ""))
  s = s:lower()
  for _ = 1, 3 do                                                           -- 1,000 and 1,000,000 (bounded: never quadratic)
    local n
    s, n = s:gsub("(%d),(%d%d%d)%f[%D]", "%1%2")
    if n == 0 then break end
  end
  s = s:gsub("(%d+)%.%d+", "%1")
  s = s:gsub("'", "")
  s = s:gsub("[%.!?;:]+", " , ")
  s = s:gsub("[&+]", " and ")
  s = s:gsub("[^%w%s,\128-\255]", " ")
  s = s:gsub("(%d)x%f[%A]", "%1 x"):gsub("%f[%w]x(%d)", "x %1")
  s = s:gsub("(%d)(%a%a%a)", "%1 %2")
  s = s:gsub("%s+", " ")
  s = s:gsub("[%s,]*,[%s,]*", " , ")
  local tokens = {}
  for w in s:gmatch("%S+") do tokens[#tokens + 1] = w end
  -- strip politeness from both ends (one pass forward, one index at each end: never rescans)
  local i, j = 1, #tokens
  local progress = true
  while progress and i <= j do
    progress = false
    while i <= j and tokens[i] == "," do i = i + 1; progress = true end
    local e = match_at(PREFIX_FIRST, tokens, i)
    if e and i <= j then
      local nxt = tokens[i + e.n]
      local blocked = PREFIX_NOT_BEFORE[e.text] and nxt and PREFIX_NOT_BEFORE[e.text][nxt]
      if PREFIX_NEEDS_COMMA[e.text] and nxt ~= "," and not (nxt == nil) then blocked = true end
      if e.text == "computer" or e.text == "holodeck" then
        -- "computer add a lamp" / "computer, ..." greets the machine; "computer desk" names a thing
        if nxt ~= "," and nxt ~= "" and not match_at(VERB_FIRST, tokens, i + 1) then blocked = true end
        if nxt == nil then blocked = (e.text == "computer") end
      end
      if not blocked and i + e.n - 1 <= j then i = i + e.n; progress = true end
    end
    while j >= i and tokens[j] == "," do j = j - 1; progress = true end
    if j >= i then
      local l = SUFFIX_LAST[tokens[j]]
      if l then
        for _, e2 in ipairs(l) do
          local from = j - e2.n + 1
          if from > i then                                                   -- something must be left of the politeness
            local ok = true
            for k = 1, e2.n do if tokens[from + k - 1] ~= e2.w[k] then ok = false; break end end
            if ok then j = from - 1; progress = true; break end
          end
        end
      end
    end
  end
  local out = {}
  for k = i, j do
    local t = tokens[k]
    if t == "towards" then t = "toward" end
    if t == "shaped" and (out[#out] == "u" or out[#out] == "l") then t = "shape" end
    out[#out + 1] = t
  end
  return out
end

local function normalise(s)
  return table.concat(tokenise(s), " ")
end
Parse.normalise = normalise

-- ---------------------------------------------------------------------------------------------- the wake word
-- Players with a live mic talk to each other normally; only a sentence that STARTS with the wake word is a request.
-- Speech recognition spells the name several ways, so the accepted spellings are a list; a few filler words may come first.
Parse.WAKE_WORDS = { "skippy", "skipy", "skippie", "skipie", "skipi", "skippi", "skipee", "skippee", "skippey", "skipey", "skeepy",
  "skeepi", "skeepie", "skippy's" }
local WAKE_LEAD = set{ "hey", "hi", "hello", "hiya", "howdy", "yo", "oi", "ok", "okay", "so", "um", "umm", "uh", "uhh", "er", "erm", "hmm",
  "hm", "well", "alright", "all", "right", "there", "please", "excuse", "me", "oh", "ah", "now", "then", "and", "but", "yes", "yeah", "sure" }

--- Is this sentence addressed to the holodeck? Returns (true, the request without the wake word) or (false, text).
function Parse.strip_wake(text, words)
  local raw = tostring(text or "")
  local from = raw:find("%S") or 1
  local head = raw:sub(from, from + 160)                                    -- the wake word is at the start; never scan a whole flood
  local list = {}
  for _, w in ipairs(words or Parse.WAKE_WORDS) do list[w] = true end
  local pos, lead = 1, 0
  for _ = 1, 5 do
    local s, e = head:find("[%w']+", pos)
    if not s then return false, raw end
    local word = head:sub(s, e):lower()
    if list[word] then
      local nxt = head:sub(e + 1, e + 1)
      if nxt ~= "_" then
        local rest = raw:sub(from + e):gsub("^[^%w]+", "")
        return true, rest
      end
      return false, raw
    end
    if not WAKE_LEAD[word] or lead >= 3 then return false, raw end
    lead = lead + 1
    pos = e + 1
  end
  return false, raw
end

-- ---------------------------------------------------------------------------------------------- helpers over a library
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

--- A word a player typed, fit to be quoted in a note that may be shown to everyone: plain letters, short, else nil.
local function safe_word(w)
  if w and #w <= 20 and w:match("^%a+$") then return w end
end

local function singular_guess(w)
  if w:match("ies$") and #w > 4 then return w:sub(1, -4) .. "y" end
  if w:match("ves$") and #w > 4 then return w:sub(1, -4) .. "f" end
  if w:match("ses$") or w:match("xes$") or w:match("ches$") or w:match("shes$") then return w:sub(1, -3) end
  if w:match("[^s]s$") and #w > 3 then return w:sub(1, -2) end
  return w
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
  -- compound names that are one thing, so the second word is not a second thing ("computer desk", "sofa bed")
  local compounds = {
    { "computer desk", "desk" }, { "sofa bed", "couch" }, { "desk phone", "phone" }, { "bed side table", "side table" },
    { "kitchen counter", "table" }, { "gaming pc", "computer" }, { "pool table", "table" }, { "ice box", "fridge" },
    { "tv stand", "dresser" }, { "tv cabinet", "dresser" }, { "tv unit", "dresser" }, { "tv table", "dresser" },
    { "television stand", "dresser" }, { "gaming chair", "office chair" }, { "bedside lamp", "desk lamp" }, { "trash", "trash can" },

  }
  for _, c in ipairs(compounds) do
    if catalog.kinds[c[2]] and not self.noun[c[1]] then
      add_noun(c[1], c[2], false, true)
      for _, p in ipairs(plural_forms(c[1])) do add_noun(p, c[2], true, true) end
    end
  end
  -- words usable for typo repair: single-word nouns of 4+ letters, and the words of multi-word names ("coffee" of "coffee table")
  self.typo_words = {}
  local seen_typo, known = {}, {}
  local function add_typo(str, nd, whole, nextw)
    if #str >= 4 then
      local e = seen_typo[str]
      if not e then
        e = { str, nd, false, {} }
        seen_typo[str] = e
        self.typo_words[#self.typo_words + 1] = e
      end
      if whole then e[3], e[2] = true, nd end
      if nextw then e[4][nextw] = true end
    end
  end
  for str, nd in pairs(self.noun) do
    if not str:find(" ", 1, true) then add_typo(str, nd, true)
    else
      -- the first word of a longer name ("coffee" of "coffee table") is repaired only when the rest of the name follows it
      local ws = words_of(str)
      if not self.noun[ws[1]] and not STOP[ws[1]] and not ARTICLES[ws[1]] then add_typo(ws[1], nd, false, ws[2]) end
    end
    for _, w in ipairs(words_of(str)) do known[w] = true end
  end
  self.known_word = known                  -- every word of every name: never "repaired" into something else
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
  -- the room words can be mistyped too ("dinng room", "bedrom")
  for name, canonical in pairs(self.scene) do
    local ws = words_of(name)
    local nd = { kind = "SCENE:" .. canonical }
    if #ws == 1 then add_typo(ws[1], nd, true) elseif #ws > 1 then add_typo(ws[1], nd, false, ws[2]) end
  end
  table.sort(self.typo_words, function(a, b) return a[1] < b[1] end)
  return self
end

--- The longest library noun starting at tokens[i] (not touching a masked token): nd, length.
--- A word that is a noun plus an "s" is that noun's plural ("boxs", "tellys").
function Parse:noun_at(tokens, i, mask)
  for len = math.min(self.max_len, #tokens - i + 1), 1, -1 do
    local ok = true
    if mask then for k = i, i + len - 1 do if mask[k] then ok = false; break end end end
    if ok then
      local nd = self.noun[join(tokens, i, i + len - 1)]
      if nd then return nd, len end
    end
  end
  local w = tokens[i]
  if w and #w >= 4 and w:sub(-1) == "s" and not (mask and mask[i]) then
    local nd = self.noun[w:sub(1, -2)]
    if nd and not nd.plural then return { kind = nd.kind, plural = true, approx = nd.approx }, 1 end
  end
end

function Parse:is_surface(kind)
  local k = self.catalog.kinds[kind]
  return k and (k.surface or k.category == "table") and true or false
end

function Parse:is_table_like(kind)
  local k = self.catalog.kinds[kind]
  return k and k.category == "table" and true or false
end

function Parse:is_seat(kind)
  local k = self.catalog.kinds[kind]
  return k and k.seat and true or false
end

-- ---------------------------------------------------------------------------------------------- verbs and sentence shapes
local function verb_at(tokens, i)
  return match_at(VERB_FIRST, tokens, i)
end

--- Index of the first token that is not lead-in padding ("to", "you", ",", ...), at most 6 in.
local function skip_lead(tokens, from)
  local p = from or 1
  while p < (from or 1) + 6 and LEAD[tokens[p]] and not verb_at(tokens, p) do p = p + 1 end
  return p
end

--- Is the whole sentence just undo phrases (and filler)?  "undo", "oops, put them back" -- but not "never mind, add a lamp".
local function is_bare_undo(tokens)
  local i, found = 1, false
  while i <= #tokens do
    local e = match_at(UNDO_FIRST, tokens, i)
    if e then found = true; i = i + e.n
    elseif tokens[i] == "," or UNDO_FILLER[tokens[i]] then i = i + 1
    else return false end
  end
  return found
end

--- Is tokens[a..b] a request to wipe everything?  verb + whole-room object + politeness, or a stock phrase ("start over").
local function is_clear_clause(tokens, a, b)
  while a <= b and (tokens[a] == "," or LEAD[tokens[a]] and not CLEAR_VERBS[tokens[a]]) do a = a + 1 end
  if a > b then return false end
  do                                                                 -- a stock phrase with its own politeness: "enough already"
    local whole = {}
    for k = a, b do whole[#whole + 1] = tokens[k] end
    local stock = match_at(CLEAR_ALONE_FIRST, whole, 1)
    if stock and stock.n == #whole then return true end
  end
  while b >= a and (tokens[b] == "," or CLEAR_POLITE[tokens[b]]) and not (tokens[b] == "all" and a == b) do b = b - 1 end
  if a > b then return false end
  local sub = {}
  for k = a, b do sub[#sub + 1] = tokens[k] end
  local n = #sub
  local alone = match_at(CLEAR_ALONE_FIRST, sub, 1)
  if alone and alone.n == n then return true end
  local vn
  local vp = match_at(CLEAR_VERBP_FIRST, sub, 1)
  if vp then vn = vp.n elseif CLEAR_VERBS[sub[1]] then vn = 1 end
  if not vn then
    local first = sub[1]
    if first and first:match("ing$") then
      -- "clearing everything"
      local base = first:sub(1, -4)
      if CLEAR_VERBS[base] or CLEAR_VERBS[base .. "e"] then vn = 1 end
    end
  end
  if not vn then return false end
  if vn == n then return alone ~= nil or sub[1] == "clear" or sub[1] == "reset" or sub[1] == "wipe" end
  local obj = match_at(CLEAR_OBJ_FIRST, sub, vn + 1)
  if obj and vn + obj.n == n then return true end
  -- "wipe everything out", "clear it all out"
  if obj then
    local rest = vn + obj.n + 1
    while rest <= n and (CLEAR_POLITE[sub[rest]]) do rest = rest + 1 end
    if rest == n + 1 then return true end
  end
  return false
end

--- Split tokens at "," / "and" / "then" and ask whether EVERY clause is a clear (start over, wipe everything).
local function is_clear(tokens, last)
  last = last or #tokens
  local a, found = 1, false
  for i = 1, last + 1 do
    local t = tokens[i]
    if i == last + 1 or t == "," or t == "and" or t == "then" or t == "also" then
      if i - 1 >= a then
        if not is_clear_clause(tokens, a, i - 1) then
          -- "clear the room , please" leaves a pure-politeness clause: ignore those
          local polite = true
          for k = a, i - 1 do if not CLEAR_POLITE[tokens[k]] and tokens[k] ~= "," then polite = false end end
          if not polite then return false end
        else
          found = true
        end
      end
      a = i + 1
    end
  end
  return found
end

--- The index of a "the last one" / "what I just made" reference at tokens[i] and its length.
local function last_ref_at(tokens, i)
  local t = tokens[i]
  if t == nil then return nil end
  if (t == "the" or t == "that" or t == "those" or t == "these" or t == "my" or t == "our" or t == "this") and LAST_MODS[tokens[i + 1]]
     and LAST_HEADS[tokens[i + 2]] then
    return 3
  end
  if (t == "the" or t == "that" or t == "those" or t == "these" or t == "this") and LAST_HEADS[tokens[i + 1]] and tokens[i + 1] ~= "one"
     and tokens[i + 1] ~= "ones" and tokens[i + 1] ~= "set" and tokens[i + 1] ~= "group" and tokens[i + 1] ~= "batch" then
    local k = i + 2
    if (tokens[k] == "i" or tokens[k] == "we" or tokens[k] == "you") then
      k = k + 1
      if tokens[k] == "just" or tokens[k] == "recently" then k = k + 1 end
      if MADE_VERBS[tokens[k]] then return k - i + 1 end
    end
    if tokens[i + 1] == "thing" or tokens[i + 1] == "things" then
      -- the bare "that thing" / "those things" only when nothing else follows to describe them
      if t == "that" or t == "those" or t == "the" then
        local nxt = tokens[i + 2]
        if nxt == nil or nxt == "," or nxt == "to" or nxt == "next" or nxt == "closer" or nxt == "around" or nxt == "back" or nxt == "over" then
          return 2
        end
      end
    end
  end
  if (t == "the" or t == "that" or t == "those") and (tokens[i + 1] == "one" or tokens[i + 1] == "ones") and (tokens[i + 2] == "i" or tokens[i + 2] == "we") then
    local k = i + 3
    if tokens[k] == "just" or tokens[k] == "recently" then k = k + 1 end
    if MADE_VERBS[tokens[k]] then return k - i + 1 end
  end
  if (t == "what" or t == "everything" or t == "whatever" or t == "all") and (tokens[i + 1] == "i" or tokens[i + 1] == "we" or tokens[i + 1] == "you") then
    local k = i + 2
    if tokens[k] == "just" or tokens[k] == "recently" then k = k + 1 end
    if MADE_VERBS[tokens[k]] then return k - i + 1 end
  end
  return nil
end

--- "i just made", "you just built": noise after a noun that says which one ("the couch i just made").
local function made_tail_at(tokens, i)
  local t = tokens[i]
  if (t == "i" or t == "we" or t == "you") then
    local k = i + 1
    if tokens[k] == "just" or tokens[k] == "recently" or tokens[k] == "had" then k = k + 1 end
    if MADE_VERBS[tokens[k]] then return k - i + 1 end
  end
  return nil
end

local function is_numeric_token(t) return t ~= nil and (t:match("^%d+$") ~= nil or (NUMBER_WORDS[t] ~= nil)) end

local GRID_WORDS = set{ "grid", "array", "matrix", "lattice", "block", "square", "rectangle", "pattern", "formation", "layout" }
local ROW_UNITS = { rows = "rows", columns = "cols", cols = "cols", lines = "rows" }
local GROUP_UNITS = set{ "groups", "sets", "piles", "stacks", "clusters", "batches", "bunches", "pairs" }

--- "two rows of five chairs" -> a count of 10 (rows 2, cols 5, a grid); "a 3 by 3 grid of crates" -> 9 crates in a grid.
function Parse:fold_products(tokens)
  local out, i, n = {}, 1, #tokens
  local dims, hint = nil, false
  while i <= n do
    local took = false
    local a = number_of(tokens[i])
    if a and a > 0 then
      local t2 = tokens[i + 1]
      local b = number_of(tokens[i + 2])
      if (t2 == "by" or t2 == "x" or t2 == "times") and b and b > 0 then
        local j = i + 2
        local gridw = false
        while GRID_WORDS[tokens[j + 1]] do j = j + 1; gridw = true end
        local has_of = tokens[j + 1] == "of"
        if has_of then j = j + 1 end
        local nd = false
        for k = j + 1, math.min(n, j + 3) do if self:noun_at(tokens, k) then nd = true; break end end
        if gridw or has_of or nd then
          out[#out + 1] = tostring(math.min(a * b, COUNT_CAP))
          dims = dims or { rows = a, cols = b }
          hint = hint or gridw or has_of or (a > 1 and b > 1)
          i = j + 1
          took = true
        end
      elseif (ROW_UNITS[t2] or GROUP_UNITS[t2]) and tokens[i + 2] == "of" then
        local b2 = number_of(tokens[i + 3])
        if b2 and b2 > 0 then
          out[#out + 1] = tostring(math.min(a * b2, COUNT_CAP))
          if ROW_UNITS[t2] == "rows" then dims = dims or { rows = a, cols = b2 }; hint = true
          elseif ROW_UNITS[t2] == "cols" then dims = dims or { rows = b2, cols = a }; hint = true end
          i = i + 4
          took = true
        elseif t2 == "pairs" then
          out[#out + 1] = tostring(math.min(a * 2, COUNT_CAP))
          i = i + 3
          took = true
        end
      end
    end
    if not took then
      out[#out + 1] = tokens[i]
      i = i + 1
    end
  end
  if hint then
    out[#out + 1] = "in"; out[#out + 1] = "a"; out[#out + 1] = "grid"
  end
  return out, dims
end

-- ---------------------------------------------------------------------------------------------- what the parser itself knows
-- Words that are never "repaired" into a library noun: everything the parser understands, plus ordinary words that sit one slip
-- away from a library noun without being a typo of it ("tablet", "stone", "bridge").
local KNOWN_WORDS = {}
do
  for _, t in ipairs({ STOP, ARTICLES, DEFINITE_WORDS, NUMBER_WORDS, UNITS, TEENS, TENS, WORLD_TARGETS, SPEAKER_WORDS, PRONOUN_TARGETS,
                       SCENE_NOUNS, UNIT_WORDS, ADJECTIVE_WORDS, LEAD, UNDO_FILLER, CLEAR_POLITE, DETERMINERS, MADE_VERBS, LAST_MODS,
                       LAST_HEADS, VERB_ADD, VERB_MOVEISH, CLEAR_VERBS, PLACE_MODS }) do
    for w in pairs(t) do KNOWN_WORDS[w] = true end
  end
  for _, list in ipairs({ QUANT_SRC, RELATION_SRC, SOFT_SRC }) do
    for _, e in ipairs(list) do for _, w in ipairs(words_of(e[1])) do KNOWN_WORDS[w] = true end end
  end
  for _, list in ipairs({ UNDO_SRC, CLEAR_OBJECTS, CLEAR_ALONE, CLEAR_VERB_PHRASES, ALL_SRC, REST_SRC, EXCEPT_SRC, NEGATORS, PREFIX_SRC,
                          SUFFIX_SRC }) do
    for _, p in ipairs(list) do for _, w in ipairs(words_of(p)) do KNOWN_WORDS[w] = true end end
  end
  for _, e in ipairs(VERB_ENTRIES) do for _, w in ipairs(words_of(e.text)) do KNOWN_WORDS[w] = true end end
  for w in ([[tablet tablets stable stables planet planets plane planes plank planks shell shells chain chains coach coaches crane cranes
    truck trucks bridge bridges orange oranges beach beaches choir choirs store stores stone stones cable cables piano pianos dragon
    dragons rug rugs carpet carpets lamb clock clocks painting paintings picture pictures poster posters mirror mirrors curtain
    curtains fireplace sofa boat boats ship ships bike bikes horse horses tree trees rock rocks fence fences pool pools stairs stair
    statue statues pillar pillars column columns arch tower towers building buildings castle tent tents fountain guitar guitars
    drums radio radios speaker speakers camera cameras printer printers scanner laptop laptops keyboard keyboards mouse plate plates
    bowl bowls fork forks knife spoon bottle bottles glass glasses book books pencil paper bag bags basket baskets vase vases candle
    candles clothes shirt shirts bins ovens cooker hood garage garages plan plans self tale tales pant pants plat chai stud studs dinner dinners ladder ladders swing swings slide slides piano telescope globe globes
    flag flags banner banners sign signs balloon balloons cake cakes pizza pizzas apple apples banana bananas bread table tables
    carry carries cart carts wagon wagons wheel wheels tire tires tyre tyres sandbag sandbags barricade barricades wall walls
    maze mazes fort forts pyramid pyramids tunnel tunnels ramp ramps stair stairs platform platforms stage stages podium podiums
    cabinet cabinets counter counters couple couples friend friends people persons person guest guests]]):gmatch("%S+") do
    KNOWN_WORDS[w] = true
  end
end

-- QWERTY neighbours: a typing slip hits the key next to the one meant
local NEIGHBOUR_SRC = { q = "wa", w = "qeas", e = "wrsd", r = "etdf", t = "ryfg", y = "tugh", u = "yihj", i = "uojk", o = "ipkl", p = "ol",
  a = "qwsz", s = "awedxz", d = "serfcx", f = "drtgvc", g = "ftyhbv", h = "gyujnb", j = "huikmn", k = "jiolm", l = "kop",
  z = "asx", x = "zsdc", c = "xdfv", v = "cfgb", b = "vghn", n = "bhjm", m = "njk" }
local NEIGHBOUR = {}
for ch, list in pairs(NEIGHBOUR_SRC) do
  NEIGHBOUR[ch] = {}
  for k = 1, #list do NEIGHBOUR[ch][list:sub(k, k)] = true end
end

--- Is `word` one typing slip away from `target`? Slips: two neighbouring letters swapped, one letter dropped, one letter doubled,
--- one neighbouring key hit. Long words (8+) also forgive any one extra, missing or wrong letter. The first letter must be right:
--- people rarely mistype it, and "bridge" -> "fridge", "truck" -> "chest" and "orange" -> "stove" are other words, not typos.
local function one_slip(word, target, in_name)
  local lw, lt = #word, #target
  if word == target or lw < 4 or word:sub(1, 1) ~= target:sub(1, 1) then return false end
  local long = lt >= 8
  if lw == lt then
    local first, count = nil, 0
    local second
    for i = 1, lw do
      if word:byte(i) ~= target:byte(i) then
        count = count + 1
        if count == 1 then first = i elseif count == 2 then second = i end
      end
    end
    if count == 2 and second == first + 1 and word:byte(first) == target:byte(second) and word:byte(second) == target:byte(first) then
      return true                                                           -- a swap
    end
    if count == 1 and lw >= 5 then
      local typed, meant = word:sub(first, first), target:sub(first, first)
      if long then return true end
      return (NEIGHBOUR[meant] and NEIGHBOUR[meant][typed]) and true or false
    end
    return false
  end
  if lw == lt - 1 and (lw >= 5 or lw == 4) then                              -- one letter dropped
    for i = 2, lt do
      if word == target:sub(1, i - 1) .. target:sub(i + 1) then return true end
    end
    return false
  end
  if lw == lt + 1 and lw >= 5 then                                          -- one letter extra
    for i = 2, lw do
      if target == word:sub(1, i - 1) .. word:sub(i + 1) then
        local ch = word:sub(i, i)
        if long or word:sub(i - 1, i - 1) == ch or word:sub(i + 1, i + 1) == ch then return true end
        return false
      end
    end
  end
  return false
end

-- single-word verbs, for repairing a mistyped verb ("remvoe the lamp")
local VERB_WORDS = {}
do
  local seen = {}
  for _, e in ipairs(VERB_ENTRIES) do
    if e.n == 1 and #e.text >= 4 and not seen[e.text] then seen[e.text] = true; VERB_WORDS[#VERB_WORDS + 1] = e.text end
  end
  table.sort(VERB_WORDS)
end

-- ---------------------------------------------------------------------------------------------- relations and their targets
local TARGET_BREAK = set{ "and", "with", "plus", "then", "or", "but", "also", "to", "in", "on", "at", "for", "from", "as", "if", "when",
  "while", "where", "of", "by", "that", "which", "who", "whose", "so", "into", "onto", "than", "is", "are", "should", "would", "could" }
local PEOPLE_WORDS = set{ "people", "persons", "person", "guests", "guest", "diners", "folks", "kids", "adults", "children", "friends",
  "players", "seats", "seat", "pax", "of" }
local SEAT_VERBS = set{ "seats", "seat", "sits", "sleeps", "fits", "holds", "accommodates", "accomodates", "hosts", "seating" }
local WALL_NOUNS = set{ "wall", "walls" }
local CORNER_NOUNS = set{ "corner", "corners" }
local LOC_PREPS = set{ "in", "into", "inside", "within", "to", "for", "at", "from", "of", "near", "by", "throughout", "across", "onto",
  "on", "outside", "around" }
local PRON_OBJ = set{ "it", "them", "that", "those", "these", "this" }
local MANAGE = { remove = true, move = true, rotate = true, rearrange = true }

--- A wall or corner phrase with its adjectives: "against the back wall", "in the far corner", "in each corner", "in all four corners".
--- Returns the number of tokens it covers, the arrangement, and "each" / "four" when it names every corner.
local function place_at(tokens, i)
  local t = tokens[i]
  local j = i
  if t == "up" and tokens[i + 1] == "against" then j = i + 1; t = "against" end
  if not (WALL_PREPS[t] or CORNER_PREPS[t]) then return nil end
  local k = j + 1
  local each_kind, four = nil, false
  local guard = 0
  while tokens[k] and guard < 8 do
    guard = guard + 1
    local w = tokens[k]
    if w == "each" or w == "every" or w == "all" or w == "both" then each_kind = each_kind or "each"; k = k + 1
    elseif PLACE_DETS[w] or w == "single" then k = k + 1
    elseif w == "of" and each_kind then k = k + 1
    elseif w == "one" and each_kind and tokens[k + 1] == "of" then k = k + 1
    elseif w == "four" or w == "4" then four = true; k = k + 1
    else break end
  end
  local mods = 0
  while tokens[k] and PLACE_MODS[tokens[k]] and not WALL_NOUNS[tokens[k]] and not CORNER_NOUNS[tokens[k]] and mods < 3 do
    k = k + 1; mods = mods + 1
  end
  local noun = tokens[k]
  local arr
  if WALL_NOUNS[noun] and WALL_PREPS[t] then arr = "against_wall"
  elseif CORNER_NOUNS[noun] and CORNER_PREPS[t] then arr = "corner" end
  if not arr then return nil end
  local last = k
  if tokens[k + 1] == "of" and (tokens[k + 2] == "the" or tokens[k + 2] == "this" or tokens[k + 2] == "my") then
    local r = tokens[k + 3]
    if r == "room" or r == "place" or r == "space" or r == "area" or r == "house" or r == "holodeck" then last = k + 3 end
  end
  local every
  if arr == "corner" then
    if each_kind then every = "each" elseif four then every = "four" end
  end
  return last - i + 1, arr, every
end

--- "around 10 crates" means "about 10 crates", not "around the crates".
local function is_approximator(tokens, i, start)
  local nt = tokens[i + 1]
  if nt == nil then return false end
  local numeric = is_numeric_token(nt) or nt == "several" or nt == "some" or nt == "many" or nt == "half"
  if nt == "a" then
    local n2 = tokens[i + 2]
    numeric = n2 == "dozen" or n2 == "couple" or n2 == "few" or n2 == "handful" or n2 == "hundred" or n2 == "thousand" or n2 == "bunch" or n2 == "pair"
  end
  if not numeric then return false end
  local prev = tokens[i - 1]
  if i <= start or prev == nil then return true end
  return prev == "," or prev == "and" or prev == "with" or prev == "plus" or prev == "of" or prev == "then" or prev == "also"
    or prev == "need" or prev == "want" or VERB_ADD[prev] == true or LEAD[prev] == true
end

--- A relation's target: the thing (or "me", a wall, "the last one") named from tokens[j]. Returns { target, i, j, plural } or nil.
function Parse:find_target(tokens, j, mask, opts)
  opts = opts or {}
  local n = #tokens
  local first = tokens[j]
  if first == nil then return nil end
  if first == "it" or first == "them" then return { target = "@prev", i = j, j = j } end
  local ln = last_ref_at(tokens, j)
  if ln then return { target = "@last", i = j, j = j + ln - 1 } end
  local k = j
  while tokens[k] and DETERMINERS[tokens[k]] and k - j < 3 do k = k + 1 end
  local tk = tokens[k]
  if tk == nil then
    if PRONOUN_TARGETS[first] then return { target = "@prev", i = j, j = j } end
    return nil
  end
  if SPEAKER_WORDS[tk] and (not opts.soft or opts.rotating) then return { target = "me", i = j, j = k } end
  if tk == "where" and tokens[k + 1] == "i" then return { target = "here", i = j, j = k + 2 } end
  for kk = k, math.min(n, k + 6) do
    local t = tokens[kk]
    if kk > k then
      if t == "," or TARGET_BREAK[t] then break end
      if opts.soft and is_numeric_token(t) then break end
      if match_at(REL_FIRST, tokens, kk) then break end
    end
    if not mask[kk] then
      local nd, len = self:noun_at(tokens, kk, mask)
      if nd then return { target = nd.kind, i = j, j = kk + len - 1, plural = nd.plural } end
      local w = WORLD_TARGETS[t]
      if w and (not opts.soft or w == "window" or w == "door" or (opts.rotating and w == "center")) then return { target = w, i = j, j = kk } end
    end
  end
  if PRONOUN_TARGETS[first] then return { target = "@prev", i = j, j = j } end
  return nil
end

--- The relation phrase that starts at tokens[i], if any: { i, n, arr, needs, flag, soft, found }.
function Parse:relation_at(tokens, i, mask, start, ctx, other_thing)
  ctx = ctx or {}
  local rotating = ctx.rotating
  local pn, parr, pevery = place_at(tokens, i)
  if pn then return { i = i, n = pn, arr = parr, needs = false, every = pevery, phrase = "place" } end
  local rows_n = tokens[i] == "in" and is_numeric_token(tokens[i + 1]) and ROW_UNITS[tokens[i + 2]]
  if rows_n then                                                              -- "in two rows [of 3]"
    local n = 3
    if tokens[i + 3] == "of" and number_of(tokens[i + 4]) then n = 5 end
    return { i = i, n = n, arr = "grid", needs = false, phrase = "rows", rows = number_of(tokens[i + 1]), by_cols = ROW_UNITS[tokens[i + 2]] == "cols",
             per = (n == 5) and number_of(tokens[i + 4]) or nil }
  end
  if tokens[i] == "where" then
    -- "where the couch was": the spot of something gone or elsewhere, not a thing to place
    local k = i + 1
    while tokens[k] and DETERMINERS[tokens[k]] and k - i < 4 do k = k + 1 end
    local nd, len = self:noun_at(tokens, k, mask)
    local after = nd and tokens[k + len]
    if after == "was" or after == "is" or after == "were" or after == "are" or after == "stood" or after == "stands" then
      return { i = i, n = k + len - i + 1, arr = "none", needs = false, phrase = "where" }
    end
  end
  if ctx.moving and (tokens[i] == "to" or tokens[i] == "toward") and (tokens[i + 1] == "me" or tokens[i + 1] == "us") then
    return { i = i, n = 2, arr = "in_front_of", needs = false, target = "me", phrase = "to me" }          -- "bring the chairs to me"
  end
  local e = match_at(REL_FIRST, tokens, i)
  if e then
    if e.text == "around" and is_approximator(tokens, i, start) then return nil end
    if e.text == "face" or e.text == "faces" or e.text == "faced" then
      -- "turn the couch face down" / "face the room": only a relation when a thing, the speaker or "each other" follows
      if not (tokens[i + 1]) then return nil end
    end
    return { i = i, n = e.n, arr = e.arr, needs = e.needs, flag = e.flag, phrase = e.text, target = (e.flag == "center") and "center" or nil }
  end
  local s = match_at(SOFT_FIRST, tokens, i)
  -- "build a wall out of crates" / "make it from crates" name a material, not a place to take things from
  if s and s.flag == "away" and ctx.creating then s = nil end
  if s then
    local found = self:find_target(tokens, i + s.n, mask, { soft = true, rotating = rotating })
    -- "make room for chairs": when the place is the only thing said, it is the thing
    if found and other_thing and not other_thing(i, found.j) then found = nil end
    if found and found.target ~= "@prev" or (found and (tokens[i + s.n] == "it" or tokens[i + s.n] == "them")) then
      local arr = s.arr
      if rotating and arr == "beside" then arr = "facing_me" end
      if rotating and arr == "away" then arr = "facing_away" end
      if s.flag == "between" and (found.target == "@prev" or tokens[i + s.n] == "them") then return nil end
      return { i = i, n = s.n, arr = arr, needs = true, flag = s.flag, phrase = s.text, soft = true, found = found }
    end
  end
end

-- ---------------------------------------------------------------------------------------------- idioms
local REMOVE_TAILS = {
  "has to go", "have to go", "had to go", "needs to go", "need to go", "ought to go", "got to go", "gotta go", "must go", "can go",
  "could go", "should go", "may go", "will go", "is not needed", "isnt needed", "are not needed", "arent needed", "is not wanted",
  "isnt wanted", "are not wanted", "arent wanted", "is no longer needed", "are no longer needed", "can be removed", "should be removed",
  "must be removed", "needs to be removed", "has to be removed", "can be deleted", "should be deleted", "needs to be gone", "to go away",
  "go away", "to go", "gone", "removed", "deleted", "erased", "cleared", "taken away", "taken out", "thrown away", "thrown out",
  "to disappear", "disappear", "to vanish", "vanish", "disappeared", "vanished", "to be removed", "to be gone", "to be deleted",
  "to be cleared", "out of here" }
local _, REMOVE_TAIL_LAST = index_phrases((function()
  local l = {}
  for _, p in ipairs(REMOVE_TAILS) do l[#l + 1] = { text = p } end
  return l
end)(), function(e) return e.text end)
local CAUSE_LEADS = set{ "make", "have", "get", "let", "want", "see", "watch", "id" }
local PARTICLE_VERBS = { take = set{ "away", "out", "off" }, throw = set{ "away", "out" }, send = set{ "away" }, put = set{ "away" },
  get = set{ "out", "gone", "away" }, kick = set{ "out", "away" }, haul = set{ "away" }, carry = set{ "away", "out" }, pick = set{ "up" },
  clear = set{ "away", "out" }, wipe = set{ "away", "out" }, sweep = set{ "away" }, toss = set{ "away", "out" }, chuck = set{ "away", "out" },
  pack = set{ "away" }, tidy = set{ "away" } }
local QUESTION_WORDS = set{ "how", "what", "whats", "why", "who", "whos", "when", "where", "wheres", "which", "whose", "is", "are", "does", "did", "hows" }
local COPULA = set{ "is", "are", "was", "were", "isnt", "arent", "wasnt", "werent", "looks", "seems", "feels", "looked", "seemed", "needs",
  "wants", "smells", "sounds" }
local TRASH_WORDS = set{ "trash", "bin", "garbage", "rubbish", "dumpster", "dump", "skip", "wastebasket", "trashcan", "litter", "waste" }
local NEG_VERB_START = set{ "dont", "never", "stop", "quit", "no", "not", "nope" }
local ANYMORE = { "anymore", "any more", "any longer", "no longer", "at all", "anymore please" }

--- Is tokens[a..b] one of the given trailing phrases (given as a last-word index)? Returns its length.
local function tail_phrase(tokens, last, from)
  local l = REMOVE_TAIL_LAST[tokens[last]]
  if not l then return nil end
  for _, e in ipairs(l) do
    local f = last - e.n + 1
    if f >= from then
      local ok = true
      for k = 1, e.n do if tokens[f + k - 1] ~= e.w[k] then ok = false; break end end
      if ok then return e.n end
    end
  end
end

--- Sentences that mean a plain verb said another way, rewritten to it: "the lamp has to go" -> "remove the lamp",
--- "take the chairs away" -> "remove the chairs", "keep the couch and remove the rest" -> "remove everything except the couch".
--- Returns the (possibly new) tokens and "none" (a refusal: "dont add a couch"), "undo", or nil.
function Parse:rewrite_idioms(tokens, p)
  local n = #tokens
  local t = tokens[p]
  if t == nil then return tokens, nil end
  local function sub(a, b)
    local out = {}
    for k = a, b do out[#out + 1] = tokens[k] end
    return out
  end
  local function cat(...)
    local out = {}
    for _, part in ipairs({ ... }) do for _, w in ipairs(part) do out[#out + 1] = w end end
    return out
  end
  -- "i no longer want X" is "i dont want X anymore"
  if (t == "no" and tokens[p + 1] == "longer" and (tokens[p + 2] == "want" or tokens[p + 2] == "need" or tokens[p + 2] == "like")) then
    local rest = sub(p + 2, n)
    local obj = sub(p + 3, n)
    return self:rewrite_idioms(cat({ "dont" }, rest, { "anymore" }), 1)
  end
  -- refusals: "dont add a couch", "no more chairs", "stop adding chairs"; "i dont want the couch anymore" is a removal
  local neg_at = p
  if t == "do" and tokens[p + 1] == "not" then neg_at = p + 1; t = "not" end
  if (t == "dont" or t == "cant") or (t == "not" and neg_at ~= p) then
    local rest = neg_at + 1
    if tokens[rest] == "you" then rest = rest + 1 end
    local w = tokens[rest]
    if w == "want" or w == "need" or w == "like" or w == "care" then
      local obj_from = rest + 1
      if tokens[obj_from] == "to" or tokens[obj_from] == "have" then return tokens, "none" end
      local anymore = false
      local obj_to = n
      for _, a in ipairs(ANYMORE) do
        local aw = words_of(a)
        local f = n - #aw + 1
        if f > obj_from then
          local ok = true
          for k = 1, #aw do if tokens[f + k - 1] ~= aw[k] then ok = false; break end end
          if ok then anymore = true; obj_to = f - 1; break end
        end
      end
      local definite = DEFINITE_WORDS[tokens[obj_from] or ""] and tokens[obj_from] ~= "it" and true or false
      if (anymore or definite or tokens[obj_from] == "it" or tokens[obj_from] == "them") and obj_to >= obj_from then
        return cat({ "remove" }, sub(obj_from, obj_to)), nil
      end
    end
    return tokens, "none"
  end
  if t == "i" or t == "we" then
    local nxt = tokens[p + 1]
    if (nxt == "dont" or nxt == "no" or nxt == "never") and (tokens[p + 2] == "want" or tokens[p + 2] == "need" or tokens[p + 2] == "longer"
       or tokens[p + 2] == "like") then
      local q = p + 2
      if tokens[q] == "longer" then q = q + 1 end
      return self:rewrite_idioms(cat({ "dont" }, sub(q, n)), 1)
    end
  end
  if NEG_VERB_START[t] and not (t == "never" and tokens[p + 1] == "mind") then
    if t == "no" or t == "not" or t == "nope" or t == "never" or t == "stop" or t == "quit" then return tokens, "none" end
  end
  -- "keep the couch and remove the rest" / "leave the couch, delete everything else"
  if t == "keep" or t == "leave" or t == "save" then
    for q = p + 2, n - 1 do
      if tokens[q] == "and" or tokens[q] == "," or tokens[q] == "then" then
        local ve = verb_at(tokens, q + 1)
        if ve and ve.class == "remove" then
          local obj = q + 1 + ve.n
          if tokens[obj] == "the" and tokens[obj + 1] == "rest" or match_at(REST_FIRST, tokens, obj) or match_at(ALL_FIRST, tokens, obj) then
            return cat({ "remove", "everything", "except" }, sub(p + 1, q - 1)), nil
          end
        end
      end
    end
  end
  -- "put it back" / "put the chairs back": undo, unless a place follows
  if (t == "put" or t == "place" or t == "set" or t == "bring" or t == "take" or t == "get") and n >= p + 2 then
    local last = n
    local back_at
    for q = p + 1, n do
      if tokens[q] == "back" then back_at = q end
    end
    if back_at then
      local after = back_at + 1
      local tail_ok = after > n
      if not tail_ok then
        local rest = join(tokens, after, n)
        if rest == "where it was" or rest == "where they were" or rest == "again" or rest == "there" or rest == "like before"
           or rest == "as it was" or rest == "to how it was" or rest == "how it was" or rest == "where it belongs" or rest == "to where it was"
           or rest == "where they belong" or rest == "to where they were" then tail_ok = true end
      end
      if tail_ok and (t == "put" or t == "place" or t == "set") then
        -- a thing, "it" or "them", and nothing else: no destination was named
        local only = true
        for q = p + 1, back_at - 1 do
          local w = tokens[q]
          if not (PRON_OBJ[w] or w == "all" or DETERMINERS[w] or ADJECTIVE_WORDS[w] or self:noun_at(tokens, q) or number_of(w) or w == "of" or w == ",") then only = false end
        end
        if only then return tokens, "undo" end
      end
    end
  end
  -- "face the couch to the tv" is "turn the couch to face the tv"
  if t == "face" and n >= p + 3 then
    for q = p + 2, math.min(n, p + 6) do
      local w = tokens[q]
      if w == "to" or w == "toward" or w == "at" then
        local out = {}
        for k = 1, p - 1 do out[#out + 1] = tokens[k] end
        out[#out + 1] = "turn"
        for k = p + 1, q - 1 do out[#out + 1] = tokens[k] end
        out[#out + 1] = "to"; out[#out + 1] = "face"
        for k = q + 1, n do out[#out + 1] = tokens[k] end
        return out, nil
      end
    end
  end
  -- "spread the chairs out", "space the chairs out": scattered about
  if (t == "spread" or t == "space") and n >= p + 2 then
    for q = p + 2, math.min(n, p + 7) do
      if tokens[q] == "out" then
        local out = {}
        for k = 1, p - 1 do out[#out + 1] = tokens[k] end
        out[#out + 1] = "scatter"
        for k = p + 1, n do if k ~= q then out[#out + 1] = tokens[k] end end
        return out, nil
      end
    end
  end
  -- "turn the tv off", "switch the lamp on": the verb is "turn off" / "turn on", which the holodeck cannot do
  if (t == "turn" or t == "switch" or t == "power") and n >= p + 2 and (tokens[n] == "off" or tokens[n] == "on") and tokens[p + 1] ~= "off"
     and tokens[p + 1] ~= "on" and tokens[p + 1] ~= "up" and tokens[p + 1] ~= "down" then
    local out = {}
    for k = 1, p - 1 do out[#out + 1] = tokens[k] end
    out[#out + 1] = "turn"; out[#out + 1] = tokens[n]
    for k = p + 1, n - 1 do out[#out + 1] = tokens[k] end
    return out, nil
  end
  -- "put the crates in the trash / bin / dumpster": they go
  if (t == "put" or t == "throw" or t == "toss" or t == "chuck" or t == "drop" or t == "dump" or t == "shove" or t == "stick" or t == "place")
     and n >= p + 3 then
    for q = p + 2, n - 1 do
      if tokens[q] == "in" or tokens[q] == "into" then
        local k = q + 1
        while tokens[k] and DETERMINERS[tokens[k]] do k = k + 1 end
        local last = k
        if tokens[k + 1] == "can" or tokens[k + 1] == "bin" then last = k + 1 end
        if TRASH_WORDS[tokens[k]] and last == n then return cat({ "remove" }, sub(p + 1, q - 1)), nil end
        break
      end
    end
  end
  -- "line them up", "tidy the chairs up", "straighten them out"
  if (t == "line" or t == "tidy" or t == "straighten" or t == "neaten") and tokens[p + 1] ~= "up" and n >= p + 2 then
    for q = p + 2, math.min(n, p + 7) do
      if tokens[q] == "," then break end
      if tokens[q] == "up" or tokens[q] == "out" then
        local out = {}
        for k = 1, p - 1 do out[#out + 1] = tokens[k] end
        out[#out + 1] = t; out[#out + 1] = "up"
        for k = p + 1, n do if k ~= q then out[#out + 1] = tokens[k] end end
        return out, nil
      end
    end
  end
  -- "take the chairs away": the verb and a particle later on
  local pv = PARTICLE_VERBS[t]
  if pv and n >= p + 2 then
    for q = p + 2, math.min(n, p + 9) do
      local w = tokens[q]
      if w == "," then break end
      if pv[w] and not (w == "away" and tokens[q + 1] == "from") and not (t == "put" and w == "away" and tokens[q + 1] == "from")
         and not (w == "out" and tokens[q + 1] == "of" and tokens[q + 2] == "the" and tokens[q + 3] == "way") then
        if t == "pick" and tokens[p + 1] == "up" then break end
        -- the thing(s) between; keep "off the table" / "out of here" so the place stays a place
        local drop = (w == "away" or w == "gone" or w == "up" or (w == "out" and tokens[q + 1] ~= "of")) and q or nil
        local out = { "remove" }
        for k = p + 1, n do if k ~= drop then out[#out + 1] = tokens[k] end end
        return out, nil
      end
    end
  end
  -- "X has to go", "I want X gone", "make X disappear", "have X removed"
  local ve0 = verb_at(tokens, p)
  local lead_cause = CAUSE_LEADS[t] and (t ~= "get" or not (tokens[p + 1] == "rid"))
  if not ve0 or lead_cause then
    local from = p
    if lead_cause then from = p + 1 end
    if from <= n then
      -- the tail may be followed by politeness already stripped; find a removal tail ending at the last token
      local tl = tail_phrase(tokens, n, from + 1)
      if tl then
        local obj_to = n - tl
        if tokens[obj_to] == "go" and false then obj_to = obj_to - 1 end
        if obj_to >= from then
          local obj = sub(from, obj_to)
          -- only when the object really is a thing (a noun, a pronoun, a determiner): "gone" alone is not a request
          local thing = false
          for _, w in ipairs(obj) do if PRON_OBJ[w] or DETERMINERS[w] or number_of(w) then thing = true end end
          if not thing then
            for q = 1, #obj do if self:noun_at(obj, q) or last_ref_at(obj, q) then thing = true; break end end
          end
          if thing then
            return cat({ "remove" }, obj), nil
          end
        end
      end
    end
  end
  -- "clear the room of barrels", "clear the floor of crates": the barrels go, not the room
  if (t == "clear" or t == "empty" or t == "clean") and n >= p + 3 then
    local k = p + 1
    if tokens[k] == "the" or tokens[k] == "this" then k = k + 1 end
    if tokens[k] == "whole" or tokens[k] == "entire" then k = k + 1 end
    local w = tokens[k]
    if (w == "room" or w == "floor" or w == "area" or w == "space" or w == "place" or w == "scene" or w == "house") and tokens[k + 1] == "of" then
      return cat({ "remove" }, sub(k + 2, n)), nil
    end
  end
  return tokens, nil
end

--- The first exception marker ("except", "but", "other than", "and keep") that makes the rest of the sentence things to keep.
function Parse:find_exclusion(tokens, from)
  local universal = false
  for i = from, #tokens do
    local t = tokens[i]
    if t == "all" or t == "everything" or t == "every" or t == "anything" or t == "each" or t == "whatever" or t == "lot" or t == "furniture" or t == "stuff" then
      universal = true
    end
    local m = match_at(EXCEPT_FIRST, tokens, i)
    if m and i > from then
      local ok
      if m.text == "but" or m.text == "besides" or m.text == "minus" or m.text == "leaving" or m.text == "just not" or m.text == "and keep"
         or m.text == "and leave" then
        ok = universal
      else
        ok = true
      end
      if m.text == "all but" or m.text == "everything but" or m.text == "anything but" or m.text == "none but" then ok = true end
      if ok then return { i = i, n = m.n, text = m.text, implied_all = (m.text == "all but" or m.text == "everything but" or m.text == "anything but") } end
    elseif m and i == from and (m.text == "all but" or m.text == "everything but" or m.text == "anything but") then
      return { i = i, n = m.n, text = m.text, implied_all = true }
    end
  end
end

-- ---------------------------------------------------------------------------------------------- one request
local CREATE_VERBS = set{ "build", "make", "create", "generate", "produce", "craft", "construct", "form", "assemble", "conjure", "materialize",
  "materialise", "fashion", "devise", "design", "spawn", "summon" }
local SIZE_COMPARATIVE = set{ "bigger", "larger", "smaller", "taller", "shorter", "wider", "longer", "narrower", "thinner", "thicker",
  "biggest", "largest", "smallest", "tallest", "lower", "higher", "huge", "tiny", "bigger", "massive", "giant" }

--- One request out of one clause. The phases, in order (each is marked "-- ----" below):
---   words -> numbers folded -> undo / retraction -> idioms rewritten ("the lamp has to go") -> the verb -> exceptions ("except X") ->
---   clear -> homophones -> typos -> seats ("a table for six") -> negators ("without") -> relations and their targets -> "I just made" ->
---   things (it / everything / the last one / library nouns) -> rooms -> quantity and adjectives -> unknown things -> split counts ->
---   arrangement, target, a second relation -> pronoun targets -> negated and excepted items -> the action settled from the whole sentence.
--- Tokens are masked as they are explained; what is left over is `leftover`. `opts.prev` is the request before this one.
function Parse:parse_clause(raw, opts)
  opts = opts or {}
  local req = { raw = raw, normalised = "", action = "add", items = {}, arrangement = "unspecified", target = "", scene = "",
                unknown = {}, notes = {}, leftover = {} }
  local tokens = tokenise(opts.text or raw)
  if #tokens == 0 then req.action = "none"; return req end
  tokens = fold_numbers(tokens)
  local grid_dims
  tokens, grid_dims = self:fold_products(tokens)
  tokens = self:strip_greeting(tokens)
  req.normalised = table.concat(tokens, " ")
  if is_bare_undo(tokens) then req.action = "undo"; return req end
  do
    local after, strict = retraction_end(tokens)
    if after then
      local nt = tokens[after]
      if nt == nil then req.action = "undo"; return req end
      local item_start = nt == "a" or nt == "an" or number_of(nt) or match_at(QUANT_FIRST, tokens, after)
                         or (not strict and not ARTICLES[nt] and self:noun_at(tokens, after))
      if not self:starts_request(tokens, after) and not item_start then req.action = "undo"; return req end
    end
  end

  -- ---- sentences that are a plain verb said another way, and refusals
  local p = skip_lead(tokens, 1)
  local idiom
  tokens, idiom = self:rewrite_idioms(tokens, p)
  if idiom == "none" then
    req.action = "none"
    req.notes[1] = "that sounded like a no, so I changed nothing"
    return req
  elseif idiom == "undo" then
    req.action = "undo"
    return req
  end
  p = skip_lead(tokens, 1)

  -- ---- the verb
  local ve = verb_at(tokens, p)
  if ve then
    -- "swivel chair", "trash can": the longer name wins over the verb that starts it
    local _, name_len = self:noun_at(tokens, p)
    if name_len and name_len > ve.n then ve = nil end
  end
  if ve and ve.n == 1 and self.noun[tokens[p]] then
    -- "bin", "trash", "skip" are things too: a verb only when a thing follows ("trash the barrels"), else the thing itself
    local nx = tokens[p + 1]
    if not (nx and (DETERMINERS[nx] or PRON_OBJ[nx] or number_of(nx))) then ve = nil end
  end
  local typo_note = {}
  if not ve and tokens[p] and tokens[p]:match("^%a+$") and #tokens[p] >= 4 and not KNOWN_WORDS[tokens[p]] and not self.noun[tokens[p]]
     and not self.known_word[tokens[p]] and not self.scene[tokens[p]] then
    local best, tie
    for _, vw in ipairs(VERB_WORDS) do
      if one_slip(tokens[p], vw) then
        if best and best ~= vw then tie = true end
        best = best or vw
      end
    end
    if best and not tie then
      typo_note[#typo_note + 1] = "read '" .. tokens[p] .. "' as '" .. best .. "'"
      tokens[p] = best
      ve = verb_at(tokens, p)
    end
  end
  local class = ve and ve.class
  local verb, start = nil, p
  if ve then
    verb = words_of(ve.base)[1]
    start = p + ve.n
    if class == "remove" then req.action = "remove"
    elseif class == "rotate" then req.action = "rotate"
    elseif class == "rearrange" then req.action = "rearrange"
    elseif class == "move" then req.action = "move"
    elseif class == "add" then
      if tokens[start] == "me" or tokens[start] == "us" then start = start + 1 end
      if tokens[start] == "in" or tokens[start] == "up" or tokens[start] == "out" then start = start + 1 end
    end
  end
  req.verb = verb
  for _, nt in ipairs(typo_note) do req.notes[#req.notes + 1] = nt end

  -- ---- "everything except the couch": the markers and their tail
  local excl
  if class == "remove" or class == "move" or class == "rotate" or class == "rearrange" or (class == "add" and VERB_MOVEISH[verb or ""]) or class == nil then
    excl = self:find_exclusion(tokens, start)
  end
  if not excl and is_clear(tokens) then req.action = "clear"; return req end

  local mask = {}
  local function mark(i, j) for k = i, j do mask[k] = true end end
  local ntok = #tokens
  if excl then mark(excl.i, excl.i + excl.n - 1) end

  -- ---- speech-recognition homophones of number words: "add to chairs" is two chairs
  local HOMO_WORD = { ["1"] = "one", ["2"] = "two", ["3"] = "three", ["4"] = "four", ["6"] = "six", ["8"] = "eight" }
  local HOMO = { to = "2", too = "2", ["for"] = "4", fore = "4", ate = "8", tree = "3", free = "3", won = "1", sick = "6" }
  for i = start, ntok do
    local h = HOMO[tokens[i]]
    if h and (i == start or tokens[i - 1] == "," or tokens[i - 1] == "and" or tokens[i - 1] == "plus"
              or (tokens[i - 1] == "with" and tokens[i] ~= "to" and tokens[i] ~= "for" and tokens[i] ~= "too")) then
      for look = 1, 3 do
        local nd = self:noun_at(tokens, i + look)
        if nd then
          if nd.plural or tokens[i] == "won" then
            req.notes[#req.notes + 1] = "read '" .. tokens[i] .. "' as '" .. HOMO_WORD[h] .. "'"
            tokens[i] = h
          end
          break
        end
        local w = tokens[i + look]
        if not w or STOP[w] or w == "," or number_of(w) then break end
      end
    end
  end

  -- ---- typos: a word that is not a word of ours but one typing slip from a thing's name ("couhc", "coffe table")
  local typo_at = {}
  do
    local tries = 0
    for i = start, ntok do
      local w = tokens[i]
      if tries < 14 and w and #w >= 4 and #w <= 16 and w:match("^%a+$") and not KNOWN_WORDS[w] and not self.noun[w] and not self.known_word[w]
         and not self.scene[w] and not self:noun_at(tokens, i) then
        tries = tries + 1
        local best, tie
        for _, tw in ipairs(self.typo_words) do
          if math.abs(#tw[1] - #w) <= 1 and one_slip(w, tw[1], not tw[3]) and (tw[3] or tw[4][tokens[i + 1] or ""]) then
            if best and best[2].kind ~= tw[2].kind then tie = true end
            best = best or tw
          end
        end
        if best and not tie then
          req.notes[#req.notes + 1] = "read '" .. w .. "' as '" .. best[1] .. "'"
          tokens[i] = best[1]
          typo_at[i] = true
        end
      end
    end
  end

  -- ---- "a table for six", "that seats 8", "big enough for ten": the seats the table must have
  local seats, for_count
  do
    local function noun_ending_at(b)
      for len = 1, self.max_len do
        if b - len + 1 >= start then
          local nd = self.noun[join(tokens, b - len + 1, b)]
          if nd then return nd, b - len + 1 end
        end
      end
    end
    local i = start
    while i <= ntok do
      local t = tokens[i]
      local num = number_of(tokens[i + 1])
      if t == "for" and num and num >= 1 and num <= 99 and not mask[i] then
        local nd = nil
        for b = i - 1, math.max(start, i - 6), -1 do
          local w = tokens[b]
          if not (w == "enough" or w == "big" or w == "large" or w == "long" or w == "wide" or w == "roomy" or w == "room" or w == "space"
                  or w == "seating" or w == "with" or w == ",") then
            nd = noun_ending_at(b)
            break
          end
        end
        local after = i + 2
        while PEOPLE_WORDS[tokens[after]] do after = after + 1 end
        if nd and self:is_table_like(nd.kind) then
          seats = num
          mark(i, after - 1)
        elseif nd and self:is_seat(nd.kind) then
          for_count = { kind = nd.kind, n = num }
          mark(i, after - 1)
        elseif not nd and tokens[i - 1] == "seating" then
          for_count = { kind = "chair", n = num, make = true }
          mark(i - 1, after - 1)
        end
        i = after
      elseif SEAT_VERBS[t] and num and num >= 1 and num <= 99 and not mask[i] and (tokens[i - 1] == "that" or tokens[i - 1] == "which" or tokens[i - 1] == "who"
             or tokens[i - 1] == "it" or tokens[i - 1] == "can" or tokens[i - 1] == "to" or tokens[i - 1] == "and" or i == start or tokens[i - 1] == "will") then
        local has_table = false
        for b = start, i - 1 do
          local nd = self:noun_at(tokens, b)
          if nd and (self:is_table_like(nd.kind) or nd.kind == "bed" or nd.kind == "couch" or nd.kind == "loveseat" or nd.kind == "bench") then has_table = true end
        end
        if has_table then
          local nd0
          for b = start, i - 1 do
            local nd = self:noun_at(tokens, b)
            if nd and self:is_table_like(nd.kind) then nd0 = nd end
          end
          if nd0 then seats = num end
          local after = i + 2
          while PEOPLE_WORDS[tokens[after]] do after = after + 1 end
          local from = i
          if tokens[i - 1] == "that" or tokens[i - 1] == "which" or tokens[i - 1] == "who" or tokens[i - 1] == "it" or tokens[i - 1] == "can" or tokens[i - 1] == "will" then from = i - 1 end
          mark(from, after - 1)
          i = after
        else
          i = i + 1
        end
      else
        i = i + 1
      end
    end
  end

  -- ---- negators: "a table without chairs", "a bedroom but no bed", "a couch not a chair"
  local neg_scopes = {}
  do
    local i = start
    while i <= ntok do
      local e = match_at(NEG_FIRST, tokens, i)
      if e and not mask[i] and not (e.text == "no" and i == start and ntok == i) then
        -- an add request only: in "remove X" a "not" belongs to an exception
        local s0 = i + e.n
        local k = s0
        while tokens[k] and k - s0 < 8 do
          local w = tokens[k]
          if w == "," or w == "and" or w == "with" or w == "plus" or w == "then" or w == "also" or w == "but" then
            break
          elseif match_at(REL_FIRST, tokens, k) and w ~= "around" then break
          else k = k + 1 end
        end
        if k > s0 then
          neg_scopes[#neg_scopes + 1] = { from = s0, to = k - 1 }
          mark(i, i + e.n - 1)
        end
        i = math.max(k, i + e.n)
      else
        i = i + 1
      end
    end
  end

  -- ---- relations and their targets
  local relation_hits = {}
  -- is there a thing in the clause other than tokens a..b?  (a soft "by the couch" needs something to put by it)
  local function other_thing(a, b)
    for k = start, ntok do
      if k < a or k > b then
        local w = tokens[k]
        if PRON_OBJ[w] and w ~= "that" and w ~= "this" then return true end
        if self:noun_at(tokens, k) or last_ref_at(tokens, k) or w == "everything" or match_at(REST_FIRST, tokens, k) then return true end
        local al = match_at(ALL_FIRST, tokens, k)
        if al and al.text ~= "all" then return true end
        local prev_w = tokens[k - 1]
        if k > start and (prev_w == "a" or prev_w == "an" or number_of(prev_w)) and not ADJECTIVE_WORDS[w] and not STOP[w] and not number_of(w)
           and not SCENE_NOUNS[w] and not UNIT_WORDS[w] and w ~= "," and not match_at(REL_FIRST, tokens, k) then return true end
        if self.scene[w] then return true end
      end
    end
    return false
  end
  local rel_ctx = { rotating = req.action == "rotate", creating = CREATE_VERBS[verb or ""], moving = class == "move" or VERB_MOVEISH[verb or ""] }
  do
    local i = start
    while i <= ntok do
      local hit = (not mask[i]) and self:relation_at(tokens, i, mask, start, rel_ctx, other_thing) or nil
      if hit then
        relation_hits[#relation_hits + 1] = hit
        mark(i, i + hit.n - 1)
        if hit.found then
          hit.target, hit.target_plural = hit.found.target, hit.found.plural
          mark(hit.found.i, hit.found.j)
          hit.resolved = true
          i = math.max(i + hit.n, hit.found.j + 1)
        else
          i = i + hit.n
        end
      else
        i = i + 1
      end
    end
  end
  for _, hit in ipairs(relation_hits) do
    if hit.needs and not hit.resolved then
      local found = self:find_target(tokens, hit.i + hit.n, mask, {})
      if found then
        hit.target, hit.target_plural = found.target, found.plural
        mark(found.i, found.j)
      else
        -- "onto the rug": a thing the library lacks named as the place. It is not something to add or move.
        local from = hit.i + hit.n
        local k = from
        while tokens[k] and DETERMINERS[tokens[k]] and k - from < 3 do k = k + 1 end
        local m, run = k, {}
        while tokens[m] and not mask[m] and not STOP[tokens[m]] and tokens[m] ~= "," and not is_numeric_token(tokens[m])
              and not match_at(REL_FIRST, tokens, m) and #run < 3 do
          run[#run + 1] = tokens[m]; m = m + 1
        end
        local word = run[#run]
        if word and #word > 2 and not SCENE_NOUNS[word] and not SCENE_NOUNS[singular_guess(word)] and not SPEAKER_WORDS[word]
           and not WORLD_TARGETS[word] and not PRONOUN_TARGETS[word] and not UNIT_WORDS[word] and not ADJECTIVE_WORDS[word]
           and not PLACE_MODS[word] and not verb_at(tokens, m - 1) then
          hit.unknown_ref = singular_guess(word)
          mark(from, m - 1)
        end
      end
    end
    if hit.flag == "between" and hit.target then
      -- "between the couch and the tv": the first is the target, the second is only a place, never a new thing
      local j = hit.found and hit.found.j or (hit.i + hit.n)
      if tokens[j + 1] == "and" then
        local f2 = self:find_target(tokens, j + 2, mask, { soft = true })
        if f2 and f2.target ~= "@prev" then
          hit.between = f2.target
          mark(j + 1, f2.j)
        end
      end
    end
  end
  -- a soft "back to" or "for" that found nothing is not a relation; "off" / "away from" keep the place away from the items
  local kept = {}
  for _, hit in ipairs(relation_hits) do
    if hit.flag == "backto" and not hit.target then
      for k = hit.i, hit.i + hit.n - 1 do mask[k] = nil end
    else
      kept[#kept + 1] = hit
    end
  end
  relation_hits = kept

  -- ---- "i just made", "you just built": which one is said, nothing to place
  do
    local i = start
    while i <= ntok do
      if not mask[i] then
        local mt = made_tail_at(tokens, i)
        if mt and tokens[i - 1] and (self:noun_at(tokens, i - 1) or PRON_OBJ[tokens[i - 1]]) then mark(i, i + mt - 1); i = i + mt else i = i + 1 end
      else
        i = i + 1
      end
    end
  end

  -- ---- the library's own things (longest match first, left to right), "it", "everything", "the last one"
  local spans = {}
  local anaphor_pending
  do
    local i = start
    local manage_context = (class == "remove" or class == "move" or class == "rotate" or class == "rearrange" or class == "unsupported")
                           or VERB_MOVEISH[verb or ""] or VERB_CAUSE[verb or ""] or class == nil
    while i <= ntok do
      if mask[i] then
        i = i + 1
      else
        local matched
        local tok = tokens[i]
        local ln = last_ref_at(tokens, i)
        if ln then
          matched = { i = i, j = i + ln - 1, pseudo = "@last", word = join(tokens, i, i + ln - 1) }
        elseif tok == "all" and tokens[i + 1] == "of" and (tokens[i + 2] == "them") then
          matched = { i = i, j = i + 2, pseudo = "@last", word = "all of them" }
        elseif (tok == "it" or tok == "them") and manage_context then
          local j = i
          if tokens[i + 1] == "all" then j = i + 1 end
          local named_before = false
          for _, sp0 in ipairs(spans) do if not sp0.pseudo then named_before = true end end
          if named_before then
            matched = { i = i, j = j, skip = true }     -- "turn the couch so it faces the tv": it is the couch
          elseif tok == "it" and tokens[i + 1] == "all" then
            matched = { i = i, j = j, pseudo = "@all", word = "it all" }
          else
            matched = { i = i, j = j, pseudo = "@last", word = tok }
          end
        elseif (tok == "that" or tok == "this" or tok == "those" or tok == "these") and manage_context then
          local nx = tokens[i + 1]
          local determiner = nx and (self:noun_at(tokens, i + 1) or ADJECTIVE_WORDS[nx] or number_of(nx) or (not STOP[nx] and nx ~= "," and not match_at(REL_FIRST, tokens, i + 1)
                                       and not match_at(SOFT_FIRST, tokens, i + 1) and not SCENE_NOUNS[nx] and not verb_at(tokens, i + 1)))
          if not determiner then matched = { i = i, j = i, pseudo = "@last", word = tok } end
        end
        if not matched and manage_context then
          local al = match_at(ALL_FIRST, tokens, i)
          if al then
            local ok = true
            local nx = tokens[i + al.n]
            if al.text == "all" then
              ok = (nx == nil or nx == "," or match_at(REL_FIRST, tokens, i + 1) ~= nil or match_at(SOFT_FIRST, tokens, i + 1) ~= nil or (excl and i + 1 == excl.i))
              if ok and i > start and (tokens[i - 1] == "of") then ok = false end
            elseif al.text == "the room" or al.text == "the whole room" or al.text == "the entire room" or al.text == "everything around" then
              ok = (i == start or tokens[i - 1] == "," or tokens[i - 1] == "and") and not (nx == "of")
            end
            if al.text == "the furniture" or al.text == "my furniture" or al.text == "the lot" then ok = ok and nx ~= "of" end
            if ok then matched = { i = i, j = i + al.n - 1, pseudo = "@all", word = al.text } end
          end
          if not matched then
            local rs = match_at(REST_FIRST, tokens, i)
            if rs then matched = { i = i, j = i + rs.n - 1, pseudo = "@rest", word = rs.text } end
          end
        end
        if not matched and (tok == "one" or tok == "ones") and i > start then
          -- "two blue ones": the kind of the thing named before. "four chairs one in each corner" is a number, not "a chair".
          local ant
          for s = #spans, 1, -1 do if not spans[s].pseudo then ant = spans[s]; break end end
          local counting = ant and spans[#spans] and spans[#spans].j == i - 1 and tok == "one"
          if tok == "one" then
            -- "one table" is a number; "a red one" stands for a thing
            for look = 1, 3 do
              local w = tokens[i + look]
              if not w or STOP[w] or w == "," then break end
              if self:noun_at(tokens, i + look) then counting = true; break end
            end
          end
          if ant and not counting then
            matched = { i = i, j = i, kind = ant.kind, plural = (tok == "ones"), approx = ant.approx, word = ant.word, anaphor = true }
          elseif not ant and not counting and (ADJECTIVE_WORDS[tokens[i - 1] or ""] or tokens[i - 1] == "a" or tokens[i - 1] == "an") then
            -- "add a red one": which kind is only known from what was asked before
            anaphor_pending = { adjectives = (ADJECTIVE_WORDS[tokens[i - 1] or ""]) and { tokens[i - 1] } or {}, count = (tok == "ones") and 3 or 1 }
            mark(i - 1, i)
            i = i + 1
          end
        end
        if not matched then
          local nd, len = self:noun_at(tokens, i, mask)
          if nd then
            matched = { i = i, j = i + len - 1, kind = nd.kind, plural = nd.plural, approx = nd.approx, word = join(tokens, i, i + len - 1) }
          end
        end
        if matched then
          if not matched.skip then spans[#spans + 1] = matched end
          mark(matched.i, matched.j); i = matched.j + 1
        else
          i = i + 1
        end
      end
    end
  end

  -- ---- whole rooms ("a living room"), on what is left; a room word that only says where ("in the kitchen") builds nothing
  local function thing_named_before(upto)                  -- "add a dragon in the kitchen": the dragon is the thing being placed
    for k = start, upto - 1 do
      local w = tokens[k]
      if (w == "a" or w == "an" or number_of(w)) and k + 1 < upto then return true end
    end
    return false
  end
  local real_before_scene = 0
  for _, sp in ipairs(spans) do if not sp.pseudo then real_before_scene = real_before_scene + 1 end end
  do
    local span_at = {}
    for _, sp in ipairs(spans) do span_at[sp.i] = true end
    local i = start
    while i <= ntok do
      local took = false
      if not mask[i] then
        for len = self.scene_max, 1, -1 do
          if i + len - 1 <= ntok then
            local ok = true
            for k = i, i + len - 1 do if mask[k] then ok = false; break end end
            local phrase = join(tokens, i, i + len - 1)
            if ok and self.scene[phrase] then
              local j = i + len - 1
              local first_det = i
              if i > start and (tokens[i - 1] == "a" or tokens[i - 1] == "an" or tokens[i - 1] == "the" or tokens[i - 1] == "my" or tokens[i - 1] == "this") then first_det = i - 1 end
              -- a room word right before a thing is an adjective of it ("kitchen stool", "living room couch")
              local modifier = span_at[j + 1] ~= nil
              -- a room word after "in / for / to / at the" says where, when something is being placed
              local k = first_det - 1
              while k >= start and (DETERMINERS[tokens[k]] or tokens[k] == "whole") do k = k - 1 end
              local prep = tokens[k]
              local where = prep and LOC_PREPS[prep] and (real_before_scene > 0 or req.scene ~= "" or thing_named_before(first_det))
              if modifier then
                -- left as an adjective
              elseif where then
                mark(i, j)
                if first_det < i then mark(first_det, first_det) end
              else
                if req.scene == "" then req.scene = self.scene[phrase] end
                mark(i, j)
                if first_det < i then mark(first_det, first_det) end
              end
              i = j + 1
              took = true
              break
            end
          end
        end
      end
      if not took then i = i + 1 end
    end
  end

  -- ---- quantity and adjectives for each kind: scan backwards from the noun
  local function scan_back(i_noun, lower_bound)
    local adjs, count, definite, ate = {}, nil, false, i_noun
    local k = i_noun - 1
    local guard = 0
    while k >= lower_bound and guard < 8 do
      guard = guard + 1
      local tok = tokens[k]
      if mask[k] and not (tok == "a" or tok == "an" or tok == "the") then break end
      if STOP[tok] and tok ~= "of" then break end
      local took
      local cands = QUANT_LAST[tok]
      if cands then
        for _, q in ipairs(cands) do
          local from = k - q.n + 1
          if from >= lower_bound then
            local ok = true
            for m = 1, q.n do if tokens[from + m - 1] ~= q.w[m] then ok = false; break end end
            if ok then
              count = q.count; ate = from; k = from - 1; took = true
              if DEFINITE_QUANT[q.text] then definite = true end
              break
            end
          end
        end
      end
      if took then break end
      local num = number_of(tok)
      if num then
        count = num; ate = k; k = k - 1
        if DEFINITE_WORDS[tokens[k] or ""] and ARTICLES[tokens[k]] and k >= lower_bound then definite = true; ate = k end
        break
      end
      if ARTICLES[tok] then
        definite = DEFINITE_WORDS[tok] or false
        ate = k; k = k - 1
        if tokens[k] == "of" and k - 1 >= lower_bound and number_of(tokens[k - 1]) then      -- "two of the chairs"
          count = number_of(tokens[k - 1]); ate = k - 1; k = k - 2
        elseif tokens[k] == "of" and tokens[k - 1] == "half" then
          count = -1; ate = k - 1; k = k - 2                                                 -- "half of the crates": not expressible
        end
        break
      end
      if tok == "of" then k = k - 1
      elseif tok == "," then break
      else table.insert(adjs, 1, tok); ate = k; k = k - 1 end
    end
    return adjs, count, definite, ate
  end

  local explicit = {}                                  -- items whose count was spoken (not defaulted)
  for idx, sp in ipairs(spans) do
    local prev_end = (idx > 1) and spans[idx - 1].j + 1 or start
    if sp.pseudo then
      local item
      if sp.pseudo == "@all" or sp.pseudo == "@rest" then
        item = { kind = "@all", count = 0, adjectives = {}, word = sp.word, definite = true, position = sp.i }
        if sp.pseudo == "@rest" then item.rest = true end
      else
        item = { kind = "@last", count = 1, adjectives = {}, word = sp.word, definite = true, position = sp.i }
      end
      req.items[#req.items + 1] = item
    else
      local adjs, count, definite, ate = scan_back(sp.i, prev_end)
      if sp.anaphor then
        -- "blue ones": the adjectives before "ones", the count before them
        definite = definite or false
      end
      local spoken = count ~= nil
      if count == nil and tokens[sp.j + 1] == "x" and number_of(tokens[sp.j + 2]) then
        count = number_of(tokens[sp.j + 2]); mark(sp.j + 1, sp.j + 2); spoken = true
      end
      if count == nil and tokens[sp.i - 1] == "x" and number_of(tokens[sp.i - 2]) then count = number_of(tokens[sp.i - 2]); spoken = true end
      if count == nil then
        -- "the chairs" means the ones that stand there (all of them); a bare "chairs" means a handful
        if sp.plural and definite then count = 0 else count = sp.plural and 3 or 1 end
      end
      local clean = {}
      for _, a in ipairs(adjs) do
        if not STOP[a] and a ~= "x" and not number_of(a) and not ARTICLES[a] and not INTENSIFIERS[a] and not NOISE_ADJ[a] then clean[#clean + 1] = a end
      end
      for k = ate, sp.i - 1 do mask[k] = true end
      local item = { kind = sp.kind, count = count, adjectives = clean, word = sp.word, definite = definite, plural = sp.plural,
                     position = sp.i, approx = sp.approx or nil }
      if typo_at[sp.i] or typo_at[sp.j] then item.typo = true end
      req.items[#req.items + 1] = item
      if spoken then explicit[item] = true end
      sp.item = item
    end
  end

  -- ---- things named but not in the library: article / quantity + word(s) left over
  local function quant_at(i)
    local e = match_at(QUANT_FIRST, tokens, i)
    if e then return e.n, e.count, DEFINITE_QUANT[e.text] end
    local num = number_of(tokens[i])
    if num then return 1, num, false end
    if ARTICLES[tokens[i]] then return 1, 1, DEFINITE_WORDS[tokens[i]] or false end
    return nil
  end
  do
    local i = start
    while i <= ntok do
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
            if n == 1 and i > start and (tokens[i] == "a" or tokens[i] == "an") and (tokens[i - 1] == "in" or tokens[i - 1] == "into") and #run <= 2
               and not number_of(word) then
              -- "in a zigzag": a way of arranging the things that the holodeck does not know
              local sw = safe_word(word)
              req.notes[#req.notes + 1] = sw and ("I don't know how to arrange things in " .. ((sw:match("^[aeiou]") and "an ") or "a ") .. sw .. ", so I used the usual layout")
                                          or "I don't know that way of arranging things, so I used the usual layout"
              mark(i - 1, j - 1)
            elseif not number_of(word) and not VERB_ADD[word] and not SCENE_NOUNS[word] and not SCENE_NOUNS[singular_guess(word)]
               and not UNIT_WORDS[word] and not ADJECTIVE_WORDS[word] and not INTENSIFIERS[word] and #word > 1 and word ~= "x" then
              local sing = singular_guess(word)
              req.unknown[#req.unknown + 1] = sing
              local c = count
              if c == 1 and word:match("s$") and not word:match("ss$") and #word > 3 and tokens[i] ~= "a" and tokens[i] ~= "an" and not number_of(tokens[i]) then c = 3 end
              local name = (#run > 1) and (join(run, 1, #run - 1) .. " " .. sing) or sing
              req.items[#req.items + 1] = { kind = "UNKNOWN:" .. sing, count = c, adjectives = {}, word = word, definite = definite,
                                             position = i, phrase = name }
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

  -- ---- a colour said after the noun: "a couch in red", "3 chairs colored red", "3 chairs, red"
  do
    for _, it in ipairs(req.items) do
      if it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") and it.position then
        local sp
        for _, s in ipairs(spans) do if s.item == it then sp = s end end
        if sp then
          local k = sp.j + 1
          local w = tokens[k]
          if w == "," or w == "in" or w == "colored" or w == "coloured" or w == "painted" or w == "that" or w == "which" or w == "thats" or w == "is" or w == "are" or w == "being" then
            local m = k + 1
            while tokens[m] == "is" or tokens[m] == "are" or tokens[m] == "colored" or tokens[m] == "coloured" or tokens[m] == "painted" or tokens[m] == "in" or tokens[m] == "be" do m = m + 1 end
            if COLORS[tokens[m]] and not mask[m] then
              it.adjectives[#it.adjectives + 1] = tokens[m]
              mark(k, m)
            end
          end
        end
      end
    end
  end

  -- ---- "two couches, one red and one blue": the two are told apart, not four couches
  do
    local split_items = {}
    for _, it in ipairs(req.items) do
      split_items[#split_items + 1] = it
      if it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") and it.count >= 2 then
        local sp
        for _, s0 in ipairs(spans) do if s0.item == it then sp = s0 end end
        if sp then
          local k = sp.j + 1
          if tokens[k] == "," then k = k + 1 end
          local a = number_of(tokens[k])
          if a and not mask[k] then
            local m, adj1 = k + 1, {}
            while tokens[m] and ADJECTIVE_WORDS[tokens[m]] do adj1[#adj1 + 1] = tokens[m]; m = m + 1 end
            local b = (#adj1 > 0 and (tokens[m] == "and" or tokens[m] == ",")) and number_of(tokens[m + 1]) or nil
            if b then
              local m3, adj2 = m + 2, {}
              while tokens[m3] and ADJECTIVE_WORDS[tokens[m3]] do adj2[#adj2 + 1] = tokens[m3]; m3 = m3 + 1 end
              if #adj2 > 0 and a + b == it.count and not self:noun_at(tokens, m3) then
                local twin = {}
                for key, v in pairs(it) do twin[key] = v end
                it.count, twin.count = a, b
                twin.adjectives = {}
                for _, w in ipairs(it.adjectives) do twin.adjectives[#twin.adjectives + 1] = w end
                for _, w in ipairs(adj1) do it.adjectives[#it.adjectives + 1] = w end
                for _, w in ipairs(adj2) do twin.adjectives[#twin.adjectives + 1] = w end
                twin.position = it.position + 0.5
                split_items[#split_items + 1] = twin
                explicit[twin] = explicit[it]
                mark(k, m3 - 1)
                for q = #split_items - 1, 1, -1 do                       -- the "one"s the span search took for things
                  local other = split_items[q]
                  if other ~= it and other.position and other.position >= k and other.position <= m3 - 1 then table.remove(split_items, q) end
                end
                for q = #req.items, 1, -1 do                            -- (items after this one, not yet copied)
                  local other = req.items[q]
                  if other ~= it and other.position and other.position >= k and other.position <= m3 - 1 then table.remove(req.items, q) end
                end
              end
            end
          end
        end
      end
    end
    req.items = split_items
    table.sort(req.items, function(a, b) return (a.position or 0) < (b.position or 0) end)
  end

  -- ---- the clause's relations: arrangement, target, a second relation (facing), the shapes a target displaced
  local facing_hit, facing_away_hit, shape_hit, place_hit, target_hit, other_hit
  local lost_destination
  for _, hit in ipairs(relation_hits) do
    local a = hit.arr
    if hit.unknown_ref then
      lost_destination = true
      req.unknown[#req.unknown + 1] = hit.unknown_ref
      local w = safe_word(hit.unknown_ref)
      req.notes[#req.notes + 1] = w and ("I don't have " .. ((w:match("^[aeiou]") and "an ") or "a ") .. w .. " in my library, so I ignored where it should go")
                                  or "I don't have that in my library, so I ignored where it should go"
    elseif a == "none" then
      -- not a relation at all
    elseif a == "facing_me" then facing_hit = facing_hit or hit
    elseif a == "facing_away" then facing_away_hit = facing_away_hit or hit
    elseif a == "row" or a == "circle" or a == "grid" or a == "u_shape" or a == "l_shape" or a == "stacked" or a == "scattered" then
      shape_hit = shape_hit or hit
    elseif a == "against_wall" and not hit.target or a == "corner" then place_hit = place_hit or hit
    elseif a == "facing_each_other" then other_hit = other_hit or hit
    else target_hit = target_hit or hit end
  end
  -- a wall or a corner named as the target of "next to" / "near" is the wall or corner itself
  if target_hit and (target_hit.target == "wall" or target_hit.target == "corner") and (target_hit.arr == "beside" or target_hit.arr == "against_wall") then
    local fixed = { i = target_hit.i, n = target_hit.n, arr = (target_hit.target == "wall") and "against_wall" or "corner", needs = false, phrase = "place" }
    place_hit = place_hit or fixed
    target_hit = nil
  end
  -- "on the floor", "in the middle of the room": a place, not a thing
  if target_hit and (target_hit.target == "floor" or target_hit.target == "room" or target_hit.target == "ceiling") then
    target_hit = nil
  end
  local chosen
  if target_hit then
    chosen = target_hit
    if shape_hit and target_hit.arr ~= "flank" then req.shape = shape_hit.arr end
    if shape_hit and shape_hit.arr == "circle" and target_hit.arr == "around" then req.shape = "circle" end
  else
    chosen = place_hit or shape_hit or other_hit
    if place_hit and shape_hit then req.shape = shape_hit.arr end
  end
  local governing
  if chosen then
    req.arrangement = chosen.arr
    req.target = chosen.target or ""
    if chosen.flag == "each" then req.every = "corner" end
    if chosen.every == "each" then req.every = "corner" end
    for _, it in ipairs(req.items) do if it.position < chosen.i then governing = it end end
    if chosen.phrase == "rows" then
      grid_dims = grid_dims or {}
      if chosen.by_cols then grid_dims.cols = chosen.rows else grid_dims.rows = chosen.rows end
      if chosen.per then if chosen.by_cols then grid_dims.rows = chosen.per else grid_dims.cols = chosen.per end end
    end
    if chosen.flag == "prev" then
      req.target = "@prev"
    elseif chosen.flag == "rev" then
      -- "a couch flanked by two lamps": the couch is what is flanked
      req.target = governing and governing.kind or "@prev"
    elseif chosen.flag == "for" then
      -- "two chairs for the table" stand around it; "a lamp for the couch" stands beside it
      req.arrangement = self:is_table_like(chosen.target or "") and "around" or "beside"
    end
  end
  if grid_dims then
    if grid_dims.rows then req.rows = grid_dims.rows end
    if grid_dims.cols then req.cols = grid_dims.cols end
  end
  if facing_hit then
    local ft = facing_hit.target or "me"
    if req.arrangement == "unspecified" then req.arrangement, req.target = "facing_me", ft
    else req.facing = ft end
  end
  if facing_away_hit then
    local ft = facing_away_hit.target
    if ft == nil then ft = "" end
    if ft == "wall" then
      -- "with its back to the wall" is a thing standing against the wall
      if req.arrangement == "unspecified" then req.arrangement = "against_wall"; req.target = "" else req.facing_away = "wall" end
    elseif req.arrangement == "unspecified" then req.arrangement, req.target = "facing_away", ft
    else req.facing_away = ft end
  end
  if req.arrangement == "unspecified" and other_hit then req.arrangement = other_hit.arr end

  -- "it" / "them" as a relation target: the nearest EARLIER item of a different kind, else whatever was created last
  local function resolve_pronoun(t, gov, rel)
    if t ~= "@prev" then return t end
    local pick
    local pos_limit = gov and gov.position or 1e9
    for _, it in ipairs(req.items) do
      if it ~= gov and it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") and it.position < pos_limit and (not gov or it.kind ~= gov.kind) then
        if rel == "on" then
          if self:is_surface(it.kind) then pick = it end
        elseif not it.after_with then
          pick = it
        end
      end
    end
    if not pick and rel == "on" then
      for _, it in ipairs(req.items) do
        if it ~= gov and it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") and it.position < pos_limit and (not gov or it.kind ~= gov.kind) then pick = it end
      end
    end
    return pick and pick.kind or "@last"
  end
  -- the things that a "with" introduced ("a table with 4 chairs ... it") are not what "it" means
  for _, it in ipairs(req.items) do
    local before = it.position - 1
    while before >= start and (mask[before] and not (tokens[before] == "with")) and it.position - before < 6 do before = before - 1 end
    if tokens[before] == "with" then it.after_with = true end
  end
  req.target = resolve_pronoun(req.target, governing, req.arrangement)
  local function governing_of(hit)
    local g
    if hit then for _, it in ipairs(req.items) do if it.position < hit.i then g = it end end end
    return g
  end
  if req.target == "@prev" then req.target = "@last" end
  if req.arrangement == "facing_me" and facing_hit and facing_hit.target == "@prev" then
    req.target = resolve_pronoun("@prev", governing_of(facing_hit), "facing_me")
  end
  if req.facing then req.facing = resolve_pronoun(req.facing, governing_of(facing_hit), "facing_me") end
  if req.facing_away then req.facing_away = resolve_pronoun(req.facing_away, governing_of(facing_away_hit), "facing_me") end
  local function real_place(t) return t ~= nil and t ~= "" and t:sub(1, 1) ~= "@" and t ~= "me" and t ~= "here" and t ~= "wall" end
  if req.facing == "@last" and real_place(req.target) then req.facing = req.target end            -- "chairs in front of the tv facing it"
  if req.facing_away == "@last" and real_place(req.target) then req.facing_away = req.target end
  -- "against" followed by a thing means beside that thing
  if req.arrangement == "against_wall" and req.target ~= "" and req.target ~= "wall" then req.arrangement = "beside" end
  -- "turn the couch around" is a half turn, not "around something"
  if req.action == "rotate" and req.arrangement == "around" and req.target == "" then
    req.arrangement = "unspecified"; req.half_turn = true
    if req.facing then
      req.arrangement, req.target, req.facing, req.half_turn = "facing_me", req.facing, nil, nil
    end
  end
  -- "move the lamp away from the couch": the couch is where from, never a thing to move
  for _, hit in ipairs(relation_hits) do
    if hit.arr == "away" then
      if hit.target and hit.target:sub(1, 1) ~= "@" then req.away_from = hit.target end
    end
  end
  if req.arrangement == "away" then req.arrangement, req.target = "unspecified", "" end
  if req.arrangement == "flank" then
    if req.target == "" then req.target = "@last" end
  end
  -- a relation target that is also an item in the same sentence: the item IS the target
  for _, it in ipairs(req.items) do
    if it.kind == req.target then it.is_target = true end
  end
  if req.arrangement == "flank" or req.arrangement == "around" then
    for _, it in ipairs(req.items) do if it.kind == req.target then it.is_target = true end end
  end
  -- "a plant in each corner": one in every corner
  if req.arrangement == "corner" and chosen and chosen.every then
    if chosen.every == "four" then
      local single = true
      for _, it in ipairs(req.items) do if it.kind:sub(1, 1) ~= "@" and it.count > 1 then single = false end end
      if single then req.every = "corner" end
    else
      req.every = "corner"
    end
    local before = tokens[chosen.i - 1]
    if before == "one" or before == "1" then req.every = nil end
  end
  if req.every and req.arrangement ~= "corner" then req.every = nil end

  -- ---- seats: "a table for six"
  if seats then req.seats = seats end
  if for_count then
    local done = false
    for _, it in ipairs(req.items) do
      if it.kind == for_count.kind and not explicit[it] then it.count = for_count.n; explicit[it] = true; done = true; break end
    end
    if not done and for_count.make then
      req.items[#req.items + 1] = { kind = "chair", count = for_count.n, adjectives = {}, word = "seating", definite = false, plural = true, position = for_count.n and start }
    end
  end

  -- ---- negated things are not wanted: "a table without chairs"
  if #neg_scopes > 0 then
    local kept_items, dropped_kinds = {}, {}
    for _, it in ipairs(req.items) do
      local gone = false
      for _, sc in ipairs(neg_scopes) do
        if it.position >= sc.from - 4 and it.position <= sc.to then
          -- an item belongs to the scope when its span (quantity + word) lies inside it
          local span_start = it.position
          for k = it.position - 1, sc.from, -1 do if mask[k] then span_start = k else break end end
          if span_start >= sc.from and it.position <= sc.to then gone = true end
        end
      end
      if gone then
        dropped_kinds[#dropped_kinds + 1] = it.kind
      else
        kept_items[#kept_items + 1] = it
      end
    end
    -- a scene recipe holds the thing: tell the planner to leave it out
    if #dropped_kinds > 0 then
      local ex = {}
      for _, k in ipairs(dropped_kinds) do
        if k:sub(1, 1) ~= "@" and not k:match("^UNKNOWN") then ex[#ex + 1] = k end
      end
      if #ex > 0 and req.scene ~= "" then req.exclude = ex end
    end
    req.items = kept_items
    if #kept_items == 0 and req.scene == "" and req.action == "add" and #dropped_kinds > 0 then
      req.action = "none"
      req.notes[#req.notes + 1] = "that sounded like a no, so I added nothing"
      return req
    end
    local keep_unknown = {}
    for _, u in ipairs(req.unknown) do
      local dropped = false
      for _, k in ipairs(dropped_kinds) do if k == "UNKNOWN:" .. u then dropped = true end end
      if not dropped then keep_unknown[#keep_unknown + 1] = u end
    end
    req.unknown = keep_unknown
  end

  -- ---- exceptions: the items after the marker are kept, the rest named is wanted
  local keep_note
  if excl then
    local heads, tails = {}, {}
    for _, it in ipairs(req.items) do
      if it.position >= excl.i + excl.n then tails[#tails + 1] = it else heads[#heads + 1] = it end
    end
    -- the tail may be a count ("all the chairs except one") or a subset ("except the red one"): not expressible as kinds
    local tail_has_count = false
    for k = excl.i + excl.n, ntok do
      local w = tokens[k]
      if number_of(w) or w == "one" or w == "ones" or w == "some" or w == "half" then tail_has_count = true end
    end
    local head_kinds = {}
    for _, it in ipairs(heads) do head_kinds[it.kind] = true end
    local subset = false
    for _, it in ipairs(tails) do
      if head_kinds[it.kind] then subset = true end
    end
    if #tails == 0 and tail_has_count then subset = true end
    if tail_has_count and #tails > 0 then
      for _, it in ipairs(tails) do if explicit[it] and it.count > 0 then subset = true end end
    end
    if excl.implied_all and #heads == 0 then
      heads[1] = { kind = "@all", count = 0, adjectives = {}, word = "everything", definite = true, position = excl.i }
    end
    local has_all = false
    for _, it in ipairs(heads) do if it.kind == "@all" then has_all = true end end
    if #heads == 0 then
      -- "leaving the couch" with nothing named before: everything else goes
      if req.action == "remove" then
        heads[1] = { kind = "@all", count = 0, adjectives = {}, word = "everything", definite = true, position = excl.i }
        has_all = true
      end
    end
    if subset and not has_all then
      -- keep only some of a kind
      req.items = {}
      req.action = "none"
      keep_note = "I can't keep just some of them yet, so I left everything as it was"
      req.notes[#req.notes + 1] = keep_note
    elseif subset and has_all then
      -- "everything except the chairs" is fine; "except one" is not
      if #tails == 0 then
        req.items = {}
        req.action = "none"
        req.notes[#req.notes + 1] = "I can't keep just some of them yet, so I left everything as it was"
      else
        req.items = heads
      end
    else
      req.items = heads
    end
    if req.action ~= "none" then
      local ex, seen_ex = {}, {}
      for _, it in ipairs(tails) do
        if it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") and not seen_ex[it.kind] then
          seen_ex[it.kind] = true; ex[#ex + 1] = it.kind
          if #(it.adjectives or {}) > 0 and not subset then
            req.notes[#req.notes + 1] = "I kept every " .. it.kind .. ", not only the " .. table.concat(it.adjectives, " ") .. " ones"
          end
        end
      end
      if #ex > 0 then req.exclude = ex end
    end
  end
  if req.action == "none" and keep_note then return req end

  -- ---- the rest/others without a keep or except clause: nothing to be kept, nothing safe to do
  for _, it in ipairs(req.items) do
    if it.rest and not req.exclude then
      req.items = {}
      req.action = "none"
      req.notes[#req.notes + 1] = "I could not tell which ones the rest are, so I changed nothing"
      return req
    end
  end

  -- ---- words that carried meaning but that nothing above explained ("somewhere to relax"): the caller may want help
  req.leftover = {}
  for i = start, ntok do
    local tok = tokens[i]
    if not mask[i] and tok ~= "," and not STOP[tok] and not number_of(tok) and not ARTICLES[tok] and not VERB_ADD[tok]
       and not SCENE_NOUNS[tok] and not SCENE_NOUNS[singular_guess(tok)] and not quant_at(i) and not WORLD_TARGETS[tok]
       and not PRONOUN_TARGETS[tok] and not SPEAKER_WORDS[tok] and tok ~= "x" and not UNIT_WORDS[tok] and not ADJECTIVE_WORDS[tok]
       and not DETERMINERS[tok] then
      req.leftover[#req.leftover + 1] = tok
    end
  end

  -- ---- settle the action from the whole sentence
  local function real_items()
    local t = {}
    for _, it in ipairs(req.items) do if it.kind:sub(1, 1) ~= "@" then t[#t + 1] = it end end
    return t
  end
  local function pseudo_items()
    local t = {}
    for _, it in ipairs(req.items) do if it.kind:sub(1, 1) == "@" then t[#t + 1] = it end end
    return t
  end
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
  local arr = req.arrangement
  local is_facing = (arr == "facing_me" or arr == "facing_each_other" or arr == "facing_away" or req.half_turn) and true or false
  local is_shape = (arr == "row" or arr == "circle" or arr == "grid" or arr == "u_shape" or arr == "l_shape" or arr == "stacked" or arr == "scattered")

  -- "toss a barrel behind me" throws one in; "toss the lamp" throws it out
  if req.action == "remove" and (verb == "toss" or verb == "chuck") and req.arrangement ~= "unspecified" then req.action = "add" end
  -- "half of the crates", "all of the chairs but the red ones": a share of them cannot be named
  for _, it in ipairs(req.items) do
    if it.count and it.count < 0 then
      req.action = "none"
      req.items = {}
      req.notes[#req.notes + 1] = "I can't take just a share of them yet, so I changed nothing"
      return req
    end
  end
  -- "remove the chairs near the window": the holodeck cannot pick them out by where they stand
  if req.action == "remove" and req.target ~= "" and req.target:sub(1, 1) ~= "@" and req.target ~= "me" and req.target ~= "here"
     and req.target ~= "wall" and req.target ~= "floor" and req.target ~= "room" and req.target ~= "center" then
    local many = false
    for _, it in ipairs(req.items) do if it.kind:sub(1, 1) ~= "@" and (it.count == 0 or it.count > 1) then many = true end end
    if many then
      req.action = "none"
      req.items = {}
      req.notes[#req.notes + 1] = "I can't pick out just the ones by the " .. req.target .. " yet, so I changed nothing"
      return req
    end
  end
  -- statements and questions are not requests: "the lamp is ugly", "how many chairs are there"
  if class == nil and req.action == "add" and tokens[p] then
    local first = tokens[p]
    local is_question = QUESTION_WORDS[first] and not last_ref_at(tokens, p)
    local statement = false
    if DEFINITE_WORDS[first] and first ~= "them" and first ~= "it" and req.arrangement == "unspecified" then
      for k = p + 1, ntok do if COPULA[tokens[k]] then statement = true; break end end
    end
    if statement and explicit_new_items() then statement = false end      -- "the room needs a couch" still asks for one
    if is_question or statement then
      req.action = "none"
      req.items = {}
      req.scene = ""
      req.arrangement, req.target, req.facing, req.shape = "unspecified", "", nil, nil
      return req
    end
  end
  -- unsupported verbs: "paint the couch red", "turn off the tv", "replace the chair with a stool"
  if class == "unsupported" then
    local trans = (verb == "change" or verb == "convert" or verb == "transform" or verb == "switch" or verb == "turn")
    if trans and req.scene ~= "" and #real_items() == 0 then
      req.action = "add"
    else
      req.action = "none"
      req.unknown_verb = ve.base
      req.items = {}
      req.arrangement, req.target, req.facing, req.shape = "unspecified", "", nil, nil
      return req
    end
  end
  -- "make the couch blue", "make it bigger": a change this holodeck cannot make
  if (verb == "make" or verb == "turn" or verb == "have" or verb == "get") and #req.items > 0 and #pseudo_items() + #real_items() > 0 then
    local last_w = tokens[ntok]
    if last_w and (COLORS[last_w] or SIZE_COMPARATIVE[last_w]) and not mask[ntok] and req.scene == "" and definite_items() then
      req.action = "none"
      req.unknown_verb = COLORS[last_w] and "recolour" or "resize"
      req.items = {}
      return req
    end
  end
  if req.action == "rotate" and req.scene ~= "" and #real_items() == 0 and not req.half_turn then
    -- "turn this into a waiting room"
    req.action = "add"
    req.items = {}
  end
  -- a verb we do not know followed by a definite thing: never a duplicate
  if class == nil and tokens[p] and ntok >= p + 1 and not KNOWN_WORDS[tokens[p]] and not self.noun[tokens[p]] and not self.known_word[tokens[p]]
     and not self.scene[tokens[p]] and tokens[p]:match("^%a+$") and not number_of(tokens[p]) then
    local nx = tokens[p + 1]
    if nx == "the" or nx == "this" or nx == "that" or nx == "these" or nx == "those" or nx == "my" or nx == "it" or nx == "them"
       or nx == "out" or nx == "up" or nx == "down" or nx == "off" or nx == "away" then
      if #req.items > 0 and definite_items() and req.scene == "" then
        req.action = "none"
        req.unknown_verb = safe_word(tokens[p]) or "that"
        req.items = {}
        return req
      end
    end
  end

  if lost_destination and req.arrangement == "unspecified"
     and (req.action == "move" or req.action == "rotate" or (VERB_MOVEISH[verb or ""] and definite_items())) then
    -- "put the lamp onto the rug" with no rug in the library: nowhere to put it
    req.action = "none"
    req.items = {}
    return req
  end
  if req.action == "rearrange" or req.action == "move" or req.action == "rotate" then
    -- "line up 4 chairs", "arrange six chairs in a circle": new things, not existing ones
    if (verb == "line" or verb == "arrange" or verb == "organize" or verb == "organise" or verb == "set" or verb == "put")
       and explicit_new_items() and not definite_items() and #real_items() > 0 then
      req.action = "add"
      if verb == "line" and req.arrangement == "unspecified" then req.arrangement = "row" end
    elseif req.action == "rearrange" and verb == "line" and (req.arrangement == "unspecified" or req.arrangement == "facing_me")
           and #real_items() + #pseudo_items() > 0 then
      req.arrangement = "row"
    elseif req.action == "rearrange" and verb == "line" and not req.shape and not is_shape and req.arrangement ~= "unspecified" then
      req.shape = "row"
    end
  elseif req.action == "add" then
    local manage_pron = #real_items() == 0 and #pseudo_items() > 0 and req.scene == ""
    if manage_pron then
      -- "put it next to the tv", "put them in a circle", "make it face the tv", "bring it here"
      if is_facing then req.action = "rotate"
      elseif is_shape then req.action = "rearrange"
      elseif arr ~= "unspecified" then req.action = "move"
      elseif (verb == "put" or verb == "bring" or verb == "get" or verb == "move" or verb == "take" or verb == "drag" or verb == "pull") and (req.normalised:find(" over ") or req.normalised:find(" closer")
             or req.normalised:find(" here") or req.normalised:find(" back") or req.normalised:find(" there")) then
        req.action = "move"
      else
        -- nothing to do with them: no pseudo item as something to add
        req.items = {}
      end
    elseif is_facing and definite_items() and #real_items() > 0 and (verb == nil or VERB_CAUSE[verb] or verb == "want") and not (verb and VERB_MOVEISH[verb] and verb ~= "get" and verb ~= "let" and verb ~= "have") then
      req.action = "rotate"
    elseif (is_facing or req.half_turn) and VERB_MOVEISH[verb or ""] == nil and (verb == "make" or verb == "have" or verb == "let" or verb == "get") and definite_items() then
      req.action = "rotate"
    elseif VERB_MOVEISH[verb or ""] and definite_items() and is_shape and not target_hit then
      req.action = "rearrange"
    elseif VERB_MOVEISH[verb or ""] and definite_items() and (req.arrangement ~= "unspecified" or req.normalised:find(" over ") or req.normalised:find(" closer")
           or req.normalised:find(" back ") or req.normalised:find(" farther") or req.normalised:find(" further")) then
      req.action = "move"
    elseif verb == "make" and definite_items() and (req.arrangement == "facing_me" or req.arrangement == "facing_each_other") then
      req.action = "rotate"
    elseif (req.arrangement == "stacked" or req.arrangement == "row" or req.arrangement == "circle" or req.arrangement == "grid"
            or req.arrangement == "scattered") and #req.items > 0 and definite_items() and not verb then
      req.action = "rearrange"
    elseif (verb == "have" or verb == "let" or verb == "get" or verb == "make") and is_shape and definite_items() then
      req.action = "rearrange"
    end
    -- "stack all the crates up": everything-that-exists plus an arrangement re-lays them out
    if req.action == "add" and #req.items > 0 and req.arrangement ~= "unspecified" and definite_items() and not (verb and VERB_ADD[verb] and explicit_new_items()) then
      local all_zero = true
      for _, it in ipairs(req.items) do if it.count ~= 0 then all_zero = false end end
      if all_zero then req.action = "rearrange" end
    end
  end
  -- an "@all" or "@last" is only for the verbs that manage what stands; for an add it names nothing
  if req.action == "add" then
    local kept_items = {}
    for _, it in ipairs(req.items) do if it.kind:sub(1, 1) ~= "@" then kept_items[#kept_items + 1] = it end end
    req.items = kept_items
  end

  -- ---- management verbs with no object: "move closer", "turn around", "tidy up", "rearrange the room"
  if (req.action == "move" or req.action == "rotate" or req.action == "rearrange") and #req.items == 0 and req.scene == "" then
    local everything = (req.action == "rearrange" and (verb == "tidy" or verb == "organize" or verb == "organise" or verb == "neaten" or verb == "straighten"
                        or verb == "rearrange" or verb == "reorganize" or verb == "reorganise" or verb == "sort"))
    if everything then
      req.items[1] = { kind = "@all", count = 0, adjectives = {}, word = "everything", definite = true, position = start }
    else
      req.items[1] = { kind = "@last", count = 1, adjectives = {}, word = "it", definite = true, position = start }
    end
  end
  if req.action == "remove" and #req.items == 0 and req.scene == "" then
    -- "get rid of it", "remove that": the last thing created
    req.items[1] = { kind = "@last", count = 1, adjectives = {}, word = "it", definite = true, position = 1 }
  end
  -- "clear everything off the table": only what stands on / near something, which the holodeck cannot tell
  if req.action == "remove" and #req.items == 1 and req.items[1].kind == "@all" and not req.exclude then
    local restricted = req.away_from or (req.target ~= "" and req.target:sub(1, 1) ~= "@" and req.target ~= "me" and req.target ~= "here")
    local any_place = false
    for _, hit in ipairs(relation_hits) do if hit.arr ~= "facing_me" and hit.arr ~= "facing_away" then any_place = true end end
    if restricted or any_place then
      req.action = "none"
      req.items = {}
      req.notes[#req.notes + 1] = "I can't pick out just part of the room yet, so I changed nothing"
      return req
    end
  end
  -- "remove everything" with no exception is a clear
  if req.action == "remove" and #req.items == 1 and req.items[1].kind == "@all" and not req.exclude and not req.items[1].rest then
    req.action = "clear"
    req.items = {}
    req.arrangement, req.target = "unspecified", ""
    return req
  end
  -- "everything" with no way to manage it
  if req.action == "remove" and #req.items == 1 and req.items[1].kind == "@all" and req.items[1].rest then
    req.items[1].rest = nil
  end

  -- ---- again / another one: nothing was named, but the words say "the same"
  if req.action == "add" and #real_items() == 0 and req.scene == "" then
    local again = anaphor_pending ~= nil
    for i = start, ntok do
      local w = tokens[i]
      if w == "again" or w == "another" or w == "same" or (w == "more" and (i == start or tokens[i - 1] == "one" or number_of(tokens[i - 1]))) then again = true end
    end
    if again then
      req.again = true
      local prev = opts.prev
      if prev and prev.action == "add" and #(prev.items or {}) > 0 then
        local count_override
        for i = start, ntok do
          if tokens[i] == "another" or (tokens[i] == "more" and tokens[i - 1] == "one") then count_override = 1
          elseif tokens[i] == "more" and number_of(tokens[i - 1]) then count_override = number_of(tokens[i - 1]) end
        end
        req.items = {}
        if anaphor_pending then
          local last
          for _, it in ipairs(prev.items) do if it.kind:sub(1, 1) ~= "@" then last = it end end
          if last then
            req.items[1] = { kind = last.kind, count = anaphor_pending.count, adjectives = anaphor_pending.adjectives, word = last.word, definite = false,
                             plural = last.plural, position = 1, approx = last.approx }
          end
        end
        for _, it in ipairs(anaphor_pending and {} or prev.items) do
          if it.kind:sub(1, 1) ~= "@" then
            local adjs = {}
            for _, a in ipairs(it.adjectives or {}) do adjs[#adjs + 1] = a end
            req.items[#req.items + 1] = { kind = it.kind, count = count_override or it.count, adjectives = adjs, word = it.word, definite = false,
                                           plural = it.plural, position = #req.items + 1, approx = it.approx, phrase = it.phrase }
          end
        end
      end
    end
  end

  -- ---- small corrections that need the whole picture
  -- a bare "lamp" placed on a table is the desk lamp
  if req.action == "add" and req.arrangement == "on" and req.target ~= "" and self.catalog.kinds["desk lamp"] then
    local tk = self.catalog.kinds[req.target]
    if tk and (tk.category == "table") then
      for _, it in ipairs(req.items) do
        if it.kind == "floor lamp" and not it.definite and (it.word == "lamp" or it.word == "lamps") then it.kind = "desk lamp" end
      end
    end
  end
  -- "a lamp on each side of the couch": one on each side
  if req.arrangement == "flank" then
    local movable = {}
    for _, it in ipairs(req.items) do
      if it.kind:sub(1, 1) ~= "@" and not it.is_target and it.kind ~= req.target then movable[#movable + 1] = it end
    end
    if #movable == 1 then
      -- a bare "chairs on both sides" is one per side; "a lamp on each side" stays one (the planner puts one on each side)
      local it = movable[1]
      if not explicit[it] and it.count > 2 then it.count = 2 end
    end
  end
  -- an item whose count the player never gave but a "for N" did
  -- a zero asked for is nothing at all, never "everything that exists"
  do
    local kept_items, zero = {}, false
    for _, it in ipairs(req.items) do
      if it.count == 0 and explicit[it] and not it.definite then zero = true
      else kept_items[#kept_items + 1] = it end
    end
    if zero then
      req.items = kept_items
      if #kept_items == 0 and req.scene == "" and (req.action == "remove" or req.action == "add" or req.action == "move") then
        req.action = "none"
        req.notes[#req.notes + 1] = "zero of something is nothing, so I did nothing"
        return req
      end
    end
  end
  if req.action == "none" then return req end
  -- the counts that were explicit zero in "all": numbers are capped
  for _, it in ipairs(req.items) do
    if it.count and it.count > COUNT_CAP then it.count = COUNT_CAP end
  end
  if req.every and #real_items() == 0 then req.every = nil end
  return req
end

-- ---------------------------------------------------------------------------------------------- several requests in one utterance
local JOINER_SRC = { "and then", "and also", "and after that", "after that", "afterwards", "afterward", "and finally", "and next", "then",
  "also", "next", "finally", "lastly", "and lastly", "after which", "and now", "plus also" }
local SOFT_JOINERS = set{ "and also", "also", "plus also" }
local JOINER_FIRST = index_phrases((function()
  local l = {}
  for _, p in ipairs(JOINER_SRC) do l[#l + 1] = { text = p, soft = SOFT_JOINERS[p] } end
  return l
end)(), function(e) return e.text end)

--- "computr, add a lamp": a mistyped greeting is a greeting, not a computer.
function Parse:strip_greeting(tokens)
  if #tokens >= 3 and tokens[2] == "," and #tokens[1] >= 5 and not self.noun[tokens[1]] and not KNOWN_WORDS[tokens[1]]
     and (one_slip(tokens[1], "computer") or one_slip(tokens[1], "holodeck") or one_slip(tokens[1], "skippy")) then
    table.remove(tokens, 1)
    table.remove(tokens, 1)
  end
  return tokens
end

--- Does a new request start at tokens[k]? (a verb, after any politeness)
function Parse:starts_request(tokens, k)
  local i = k
  local guard = 0
  while guard < 6 do
    guard = guard + 1
    local e = match_at(PREFIX_FIRST, tokens, i)
    if e and tokens[i] ~= "computer" and tokens[i] ~= "like" and tokens[i] ~= "need" and tokens[i] ~= "want" and tokens[i] ~= "next" and tokens[i] ~= "man" then
      i = i + e.n
    elseif tokens[i] == "," then
      i = i + 1
    else
      break
    end
  end
  i = skip_lead(tokens, i)
  local ve = verb_at(tokens, i)
  if ve then
    local base = ve.base
    if ve.class == "add" then
      local w = words_of(base)[1]
      if w == "like" or w == "need" or w == "want" then return false end
      if w == "set" and tokens[i + 1] == "of" then return false end
      if w == "pop" or w == "hand" or w == "copy" then return tokens[i + 1] ~= nil and (tokens[i + 1] == "a" or tokens[i + 1] == "the") end
    end
    if ve.class == "remove" and (words_of(base)[1] == "trash" or words_of(base)[1] == "bin") then return false end
    if ve.class == "unsupported" then
      local nx = tokens[i + ve.n]
      return nx == "the" or nx == "this" or nx == "that" or nx == "these" or nx == "those" or nx == "my" or nx == "it" or nx == "them"
             or nx == "all" or nx == "every" or nx == "each"
    end
    return true
  end
  if match_at(UNDO_FIRST, tokens, i) then return true end
  local t = tokens[i]
  if t == "start" and (tokens[i + 1] == "over" or tokens[i + 1] == "again") then return true end
  if t == "keep" or t == "leave" then return false end
  return false
end

--- Cut an utterance (already tokenised) at its sentence-like joints. Returns a list of { tokens, soft } where soft says that the
--- joint was only "and" / a comma (a list may continue), not "then" / a verb of its own.
function Parse:split_tokens(tokens)
  local n = #tokens
  local covered = {}
  local no_mask = {}
  for i = 1, n do
    local v2 = (i > 1) and verb_at(tokens, i - 1)
    if not (v2 and v2.n == 2 and v2.w[2] == tokens[i]) then            -- "bring in two couches": the "in" belongs to the verb
      local hit = self:relation_at(tokens, i, no_mask, 1, nil)
      if hit then for k = i, i + hit.n - 1 do covered[k] = true end end
    end
  end
  local cuts = {}                                   -- { from = first connector index, to = last connector index, soft = bool }
  local first_word = tokens[skip_lead(tokens, 1)]
  local keep_first = (first_word == "keep" or first_word == "leave" or first_word == "save")
  local i = 2
  do
    -- "never mind, add a lamp": the retraction is one request, the lamp another
    local after, strict = retraction_end(tokens)
    if after and after <= n then
      local nt = tokens[after]
      local item_start = nt == "a" or nt == "an" or number_of(nt) or match_at(QUANT_FIRST, tokens, after)
                         or (not strict and not ARTICLES[nt] and self:noun_at(tokens, after))
      local rest = {}
      for k = after, n do rest[#rest + 1] = tokens[k] end
      if (self:starts_request(tokens, after) or item_start) and not is_bare_undo(rest) then      -- "oops, put them back" is one undo
        cuts[1] = { from = after, to = after - 1, soft = false }
        i = math.max(2, after)
      end
    end
  end
  while i <= n - 1 do
    local t = tokens[i]
    local run_end, hard = nil, false
    if (t == "," or t == "and") and not covered[i] then
      run_end = i
      -- "and then", ", then", "and also": the joiner words that follow belong to the same joint
      local k = i + 1
      while true do
        local jn = match_at(JOINER_FIRST, tokens, k)
        if jn and not (jn.text == "next" and tokens[k + 1] == "to") then
          if not jn.soft then hard = true end
          run_end = k + jn.n - 1; k = run_end + 1
        elseif tokens[k] == "," or tokens[k] == "and" then run_end = k; k = k + 1
        else break end
      end
    else
      local jn = match_at(JOINER_FIRST, tokens, i)
      if jn and not (jn.text == "next" and tokens[i + 1] == "to") and not covered[i] then
        hard = not jn.soft
        run_end = i + jn.n - 1
        local k = run_end + 1
        while tokens[k] == "," or tokens[k] == "and" do run_end = k; k = k + 1 end
      end
    end
    if run_end then
      local after = run_end + 1
      local ok = after <= n
      -- protect "loads and loads of"
      if ok and tokens[i] == "and" and tokens[i - 1] == tokens[i + 1] and run_end == i then ok = false end
      if ok and not hard and keep_first then ok = false end
      if ok and not hard then
        -- a soft joint splits only when a verb or a whole relation-bearing phrase follows
        local prev_cut = (#cuts > 0) and cuts[#cuts].to + 1 or 1
        if self:starts_request(tokens, after) then
          ok = true
        else
          local a_rel, b_rel = false, false
          for k = prev_cut, i - 1 do if covered[k] then a_rel = true; break end end
          local next_cut = n
          for k = after, n do
            if (tokens[k] == "," or tokens[k] == "and") and not covered[k] then next_cut = k - 1; break end
          end
          for k = after, next_cut do if covered[k] then b_rel = true; break end end
          local first = tokens[after]
          local noun_start = first and (ARTICLES[first] or first == "a" or first == "an" or number_of(first) or match_at(QUANT_FIRST, tokens, after)
                                         or self:noun_at(tokens, after) or ADJECTIVE_WORDS[first] or DETERMINERS[first]) and true or false
          ok = a_rel and b_rel and noun_start
        end
      end
      if ok then
        -- both sides must say something
        local left_has, right_has = false, false
        local prev_cut = (#cuts > 0) and cuts[#cuts].to + 1 or 1
        for k = prev_cut, i - 1 do if tokens[k] ~= "," then left_has = true; break end end
        for k = after, n do if tokens[k] ~= "," then right_has = true; break end end
        if left_has and right_has then
          cuts[#cuts + 1] = { from = i, to = run_end, soft = not hard }
        end
      end
      i = run_end + 1
    else
      i = i + 1
    end
  end
  local parts = {}
  local from = 1
  local soft_before = false
  for _, c in ipairs(cuts) do
    local part = {}
    for k = from, c.from - 1 do part[#part + 1] = tokens[k] end
    parts[#parts + 1] = { tokens = part, soft = soft_before }
    soft_before = c.soft
    from = c.to + 1
  end
  local last = {}
  for k = from, n do last[#last + 1] = tokens[k] end
  parts[#parts + 1] = { tokens = last, soft = soft_before }
  return parts
end

--- Join two neighbouring requests that were only listed ("remove the lamp and take out three stools"): the planner gets one request.
local function merge_requests(a, b, is_surface)
  if a.action == "add" and b.action == "rearrange" and #b.items == 1 and b.items[1].kind == "@last" and a.arrangement == "unspecified"
     and a.scene == "" and not a.every and #a.items > 0 and b.target == "" and b.arrangement ~= "unspecified" and not b.exclude then
    -- "add four chairs and arrange them in a circle": the chairs are made in the circle
    for _, it in ipairs(a.items) do if it.kind:sub(1, 1) == "@" then return false end end
    a.arrangement, a.rows, a.cols = b.arrangement, b.rows, b.cols
    return true
  end
  if a.action ~= b.action then return false end
  if b.exclude or a.exclude or b.unknown_verb or a.unknown_verb or b.again or a.again then return false end
  if a.action == "remove" then
    for _, req in ipairs({ a, b }) do
      for _, it in ipairs(req.items) do if it.kind == "@all" then return false end end
    end
    if b.arrangement ~= "unspecified" or a.arrangement ~= "unspecified" then return false end
  elseif a.action == "add" then
    -- "add a table and put four chairs around it": one build, the chairs around the table of the first clause
    if b.target == "@last" and a.arrangement == "unspecified" and a.scene == "" and not a.facing and not a.every and not a.seats
       and b.scene == "" and #a.items > 0 then
      local head
      for _, it in ipairs(a.items) do
        if it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") then
          if not head then head = it end
          if b.arrangement == "on" and is_surface(it.kind) then head = it; break end
        end
      end
      local all_real = head ~= nil
      for _, it in ipairs(b.items) do if it.kind:sub(1, 1) == "@" then all_real = false end end
      if all_real then
        for _, it in ipairs(b.items) do a.items[#a.items + 1] = it end
        head.is_target = true
        a.arrangement, a.target, a.facing, a.shape, a.facing_away = b.arrangement, head.kind, b.facing, b.shape, b.facing_away
        a.every, a.rows, a.cols, a.seats = b.every, b.rows, b.cols, b.seats
        for _, u in ipairs(b.unknown) do a.unknown[#a.unknown + 1] = u end
        for _, nt in ipairs(b.notes) do a.notes[#a.notes + 1] = nt end
        return true
      end
    end
    for _, req in ipairs({ a, b }) do
      if req.arrangement ~= "unspecified" or req.scene ~= "" or req.facing or req.target ~= "" or req.every or req.seats or req.facing_away or req.shape then return false end
      for _, it in ipairs(req.items) do if it.kind:sub(1, 1) == "@" then return false end end
    end
  elseif a.action == "clear" or a.action == "undo" then
    return true
  else
    return false
  end
  for _, it in ipairs(b.items) do a.items[#a.items + 1] = it end
  for _, u in ipairs(b.unknown) do a.unknown[#a.unknown + 1] = u end
  for _, nt in ipairs(b.notes) do a.notes[#a.notes + 1] = nt end
  for _, w in ipairs(b.leftover or {}) do a.leftover[#a.leftover + 1] = w end
  return true
end

--- "move the couch next to the table and turn it to face the tv": "it" is what the first part moved, not the group made last.
local function bind_it(r, prev, is_surface)
  if not prev or not (prev.action == "move" or prev.action == "rotate" or prev.action == "rearrange") or r.action == "undo" then return end
  local moved = {}
  for _, it in ipairs(prev.items) do if it.kind:sub(1, 1) ~= "@" and not it.kind:match("^UNKNOWN") then moved[#moved + 1] = it end end
  if #moved == 0 then return end
  local items, replaced = {}, false
  for _, it in ipairs(r.items) do
    if it.kind == "@last" and it.word ~= "the last one" then
      for _, m in ipairs(moved) do
        local copy = {}
        for key, v in pairs(m) do copy[key] = v end
        copy.position = it.position
        items[#items + 1] = copy
      end
      replaced = true
    else
      items[#items + 1] = it
    end
  end
  if replaced then r.items = items end
  if r.target == "@last" then
    if r.arrangement == "on" and prev.target and is_surface(prev.target) then r.target = prev.target
    else r.target = moved[1].kind end
  end
  if r.facing == "@last" then r.facing = moved[1].kind end
end

--- Every request in the words, in order: sentences ("." "!" "?" ";") and "then" / "and then" / "after that" / "also" cut them apart,
--- and so does an "and" or a comma followed by a verb of its own. The pronoun "it" in a later request means what the earlier made.
--- `prev` (optional) is the request that came before the whole utterance, used for "another one" / "the same again".
function Parse:parse_all(text, prev)
  local raw = tostring(text or "")
  local truncated = false
  if #raw > MAX_INPUT then raw = cut_utf8(raw, MAX_INPUT); truncated = true end
  local reqs = {}
  local folded = fold_text(raw):gsub("(%d)%.(%d)", "%1\1%2")
  local sentences = {}
  for s in (folded .. "."):gmatch("([^%.!?;]*)[%.!?;]+") do
    s = s:gsub("\1", ".")
    sentences[#sentences + 1] = s
  end
  local pieces = {}
  for si, s in ipairs(sentences) do
    local tokens = self:strip_greeting(tokenise(s))
    if #tokens > 0 then
      local parts = self:split_tokens(tokens)
      for pi, part in ipairs(parts) do
        pieces[#pieces + 1] = { text = table.concat(part.tokens, " "), raw = (#parts == 1) and trim(s) or nil, soft = part.soft and #pieces > 0,
                                single = (#parts == 1) }
      end
    end
  end
  if #pieces == 1 then pieces[1].raw = raw end
  local last_prev = prev
  for idx, pc in ipairs(pieces) do
    if #reqs >= MAX_REQUESTS then
      local last = reqs[#reqs]
      last.notes[#last.notes + 1] = "I only did the first " .. MAX_REQUESTS .. " things you asked for"
      break
    end
    local r = self:parse_clause(pc.raw or pc.text, { text = pc.text, prev = last_prev })
    bind_it(r, last_prev, function(k) return self:is_surface(k) end)
    local skip = (r.action == "none" and r.normalised == "" and #r.notes == 0 and not r.unknown_verb)
    if not skip then
      if truncated then r.truncated = true end
      local merged = false
      if pc.soft and #reqs > 0 then merged = merge_requests(reqs[#reqs], r, function(k) return self:is_surface(k) end) end
      if not merged then reqs[#reqs + 1] = r end
      last_prev = reqs[#reqs]
    end
  end
  do
    -- "I'm feeling hungry, add a kitchen": the part that asked for nothing is dropped when something else was asked
    local useful = {}
    for _, r in ipairs(reqs) do
      if not (r.action == "add" and #r.items == 0 and r.scene == "" and not r.again) then useful[#useful + 1] = r end
    end
    if #useful > 0 then reqs = useful end
  end
  if truncated and reqs[1] then reqs[1].notes[#reqs[1].notes + 1] = "I only read the first " .. MAX_INPUT .. " characters" end
  if #reqs == 0 then
    local r = { raw = raw, normalised = "", action = "none", items = {}, arrangement = "unspecified", target = "", scene = "", unknown = {},
                notes = {}, leftover = {} }
    if truncated then r.truncated = true end
    reqs[1] = r
  end
  return reqs
end

--- The first request in the words (see parse_all for the rest).
function Parse:parse(text, prev)
  return self:parse_all(text, prev)[1]
end

return Parse
