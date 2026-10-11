-- avatar/logic.lua -- the arithmetic behind Skippy's in-world avatar. Pure Lua (plain {x=,y=,z=} tables, no GMod globals except
-- `include` for the voice numbers), shared by the server (avatar.lua) and the client (avatar_client.lua), unit-tested offline.
--
--   * match_command   what a player said -> "summon" | "dismiss" | nil  (typos tolerated; "add a couch" is never a command)
--   * summon_points   where to try putting him when called (beside the caller, slightly above, to the right first)
--   * follow_target / step   the hover point behind the owner's shoulder, and the smooth drift toward it (snapping when far)
--   * hear            Skippy's voice at a listener: the SAME numbers as proximity voice (voice/core.lua)
--   * hat_tilt / speech_envelope   what the client does with the hat and the glow
-- Units: Source units (HU). 1 m is about 52.5 HU. Angles: degrees; yaw 0 faces +x, forward = (cos, sin), right = (sin, -cos).
local Core = include("mentat/voice/core.lua")

local L = {}
L.Core = Core

L.DEFAULTS = {
  side = 79,         -- HU beside the caller on arrival (1.5 m)
  summon_up = 12,    -- HU above the caller's eye on arrival
  hover_right = 46,  -- follow: HU to the owner's right ...
  hover_back = 26,   -- ... and behind ...
  hover_up = 16,     -- ... and above the eye
  bob = 3,           -- HU of gentle bobbing
  bob_rate = 1.7,    -- rad/s
  smooth = 4.5,      -- 1/s: the drift toward the target (exponential)
  max_speed = 900,   -- HU/s: the drift never goes faster than this
  snap_dist = 420,   -- HU (8 m): farther than this from the owner and he teleports to the target
  jump_dist = 300,   -- HU: an owner who moved this far in one tick has teleported; he follows at once
  turn_rate = 360,   -- deg/s
}

local function merged(o)
  local m = {}
  for k, v in pairs(L.DEFAULTS) do m[k] = v end
  for k, v in pairs(o or {}) do m[k] = v end
  return m
end

-- ------------------------------------------------------------------------------------------------ words
local function osa(a, b)                       -- edit distance with adjacent swaps counted once ("cmoe" is one slip from "come")
  local la, lb = #a, #b
  local d = {}
  for i = 0, la do d[i] = { [0] = i } end
  for j = 0, lb do d[0][j] = j end
  for i = 1, la do
    for j = 1, lb do
      local c = (a:sub(i, i) == b:sub(j, j)) and 0 or 1
      local v = math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
      if i > 1 and j > 1 and a:sub(i, i) == b:sub(j - 1, j - 1) and a:sub(i - 1, i - 1) == b:sub(j, j) then v = math.min(v, d[i - 2][j - 2] + 1) end
      d[i][j] = v
    end
  end
  return d[la][lb]
end
L.edit_distance = osa

local function set(list) local s = {}; for _, w in ipairs(list) do s[w] = true end; return s end

-- words that carry no meaning for these commands (dropped before reading the sentence)
local FILLER = set({ "please", "pls", "plz", "hey", "hi", "ok", "okay", "yo", "um", "uh", "so", "now", "thanks", "thank", "buddy", "mate",
                     "pal", "can", "could", "would", "will", "you", "just", "quickly", "i", "want", "need", "lets", "let", "then", "right" })
-- the words that may follow a summoning verb
local SUMMON_REST = set({ "here", "over", "to", "me", "closer", "near", "back", "on", "by", "with", "along", "hither", "up", "us", "my",
                          "side", "yourself", "in", "down", "next", "beside", "the", "a" })
local SUMMON_NEEDS = set({ "here", "over", "me", "us", "yourself", "near", "closer", "by", "side", "beside", "next", "along" })
-- verb -> true when it may stand alone ("Skippy, come."), false when it needs one of SUMMON_NEEDS
local SUMMON_VERBS = { come = true, appear = true, materialize = true, materialise = true, teleport = false, hover = false, follow = false,
                       join = false, show = false, get = false, fly = false, beam = false }
local DISMISS_ALONE = set({ "dismiss", "disappear", "vanish", "begone", "scram", "bye", "goodbye", "shoo", "depart", "leave", "hide" })
local DISMISS_REST = set({ "away", "off", "yourself", "home", "back", "out", "of", "here", "sight", "me", "for", "a", "while", "bit",
                           "alone", "this", "place", "room", "be", "sleep", "now", "us", "again" })
local GO_NEEDS = set({ "away", "home", "off", "back", "sleep", "hide", "bye", "out" })

local function words(text)
  local out = {}
  local t = tostring(text or ""):lower():gsub("[^%a%s]", " ")
  for w in t:gmatch("%a+") do out[#out + 1] = w end
  return out
end

local function is_wake(w) return w == "skippy" or (#w >= 5 and osa(w, "skippy") <= 2) end

-- the key of `tbl` that `w` is, tolerating one slip in words of four letters or more
local function fuzzy(w, tbl, minlen)
  minlen = minlen or 4
  if tbl[w] ~= nil then return w end
  if #w < minlen then return nil end
  for v in pairs(tbl) do
    if #v >= minlen and v:sub(1, 1) == w:sub(1, 1) and math.abs(#v - #w) <= 1 and osa(w, v) <= 1 then return v end
  end
  return nil
end
local function in_set(w, s)
  if s[w] then return true end
  if #w < 4 then return false end
  for v in pairs(s) do if #v >= 4 and osa(w, v) <= 1 then return true end end
  return false
end

--- What was said, as a command to the avatar: "summon", "dismiss", or nil (anything else belongs to the holodeck).
--- The whole sentence must be the command: a verb at the start and nothing after it but the usual little words.
function L.match_command(text)
  local kept = {}
  for _, w in ipairs(words(text)) do
    if not (is_wake(w) or FILLER[w]) then kept[#kept + 1] = w end
  end
  if kept[1] == "to" then table.remove(kept, 1) end
  if #kept == 0 or #kept > 6 then return nil end
  local first = kept[1]

  -- dismiss: "go away", "go home", "get lost", and the single-word leave-takings
  if first == "go" then
    local need = false
    for k = 2, #kept do
      if not (DISMISS_REST[kept[k]] or GO_NEEDS[kept[k]] or kept[k] == "to") then return nil end
      if GO_NEEDS[kept[k]] then need = true end
    end
    return need and "dismiss" or nil
  end
  if first == "get" and kept[2] == "lost" and #kept == 2 then return "dismiss" end
  if fuzzy(first, DISMISS_ALONE, 5) then
    for k = 2, #kept do if not in_set(kept[k], DISMISS_REST) then return nil end end
    return "dismiss"
  end

  -- summon: a verb of coming, then only the little words of "here" / "over to me"
  if first == "over" and kept[2] == "here" and #kept == 2 then return "summon" end
  local v = fuzzy(first, SUMMON_VERBS)
  if v then
    local need = false
    for k = 2, #kept do
      if not in_set(kept[k], SUMMON_REST) then return nil end
      if in_set(kept[k], SUMMON_NEEDS) then need = true end
    end
    if #kept == 1 then return SUMMON_VERBS[v] and "summon" or nil end
    if need or SUMMON_VERBS[v] then return "summon" end            -- "come on", "come back"
  end
  return nil
end

-- ------------------------------------------------------------------------------------------------ geometry of following
local function rad(d) return d * math.pi / 180 end
function L.forward(yaw) return math.cos(rad(yaw)), math.sin(rad(yaw)) end
function L.right(yaw) return math.sin(rad(yaw)), -math.cos(rad(yaw)) end

local function dist3(a, b) return math.sqrt((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2 + (a.z - b.z) ^ 2) end
L.dist = dist3

--- Yaw (degrees) that faces from `from` toward `to`.
function L.yaw_to(from, to) return math.deg(math.atan2(to.y - from.y, to.x - from.x)) end

--- Turn `cur` toward `target` by at most `maxstep` degrees, the short way round.
function L.turn(cur, target, maxstep)
  local d = (target - cur + 180) % 360 - 180
  if math.abs(d) <= maxstep then return target end
  return cur + (d > 0 and maxstep or -maxstep)
end

--- The places to try on arrival, best first: beside the caller (right, then left), slightly above their eye; then further back.
--- opos = the caller's origin, eye = height of the caller's eyes above it.
function L.summon_points(opos, yaw, eye, opts)
  local P = merged(opts)
  local fx, fy = L.forward(yaw)
  local rx, ry = L.right(yaw)
  local z = opos.z + (eye or 64) + P.summon_up
  local function at(side, ahead, up)
    return { x = opos.x + rx * side + fx * ahead, y = opos.y + ry * side + fy * ahead, z = z + (up or 0) }
  end
  return { at(P.side, 12), at(-P.side, 12), at(P.side * 0.6, -30, 10), at(-P.side * 0.6, -30, 10), at(0, 0, 60) }
end

--- The hover point while following: behind the owner's right shoulder, above the eye, with a gentle bob (t = seconds).
function L.follow_target(opos, yaw, eye, t, opts)
  local P = merged(opts)
  local fx, fy = L.forward(yaw)
  local rx, ry = L.right(yaw)
  local bob = math.sin((t or 0) * P.bob_rate) * P.bob
  return { x = opos.x + rx * P.hover_right - fx * P.hover_back, y = opos.y + ry * P.hover_right - fy * P.hover_back,
           z = opos.z + (eye or 64) + P.hover_up + bob }
end

--- One tick of the drift. st = { pos = {x,y,z}, last_owner = {x,y,z} or nil }; target = where to be; opos = the owner's origin.
--- Moves st.pos toward target (exponential, speed-limited). Returns true when he snapped (teleported) instead: the owner is
--- farther than snap_dist, or moved jump_dist in one tick (they teleported), and then st.pos = target exactly.
function L.step(st, target, opos, dt, opts)
  local P = merged(opts)
  local jumped = st.last_owner and dist3(st.last_owner, opos) > P.jump_dist
  st.last_owner = { x = opos.x, y = opos.y, z = opos.z }
  if jumped or dist3(st.pos, opos) > P.snap_dist then
    st.pos = { x = target.x, y = target.y, z = target.z }
    return true
  end
  local k = 1 - math.exp(-P.smooth * math.max(dt, 0))
  local dx, dy, dz = (target.x - st.pos.x) * k, (target.y - st.pos.y) * k, (target.z - st.pos.z) * k
  local len = math.sqrt(dx * dx + dy * dy + dz * dz)
  local cap = P.max_speed * math.max(dt, 0)
  if len > cap and len > 0 then local s = cap / len; dx, dy, dz = dx * s, dy * s, dz * s end
  st.pos = { x = st.pos.x + dx, y = st.pos.y + dy, z = st.pos.z + dz }
  return false
end

-- ------------------------------------------------------------------------------------------------ his voice
--- Skippy's voice at a listener, by the rules of proximity voice. `mode` is the server's voice mode ("proximity" or anything else:
--- global / off, where distance does not matter); `wallf` the fraction of the range left through a wall. Returns (heard, gain 0..1).
--- Identical to voice/core.lua's decide(), which is what players' own voices use.
function L.hear(dist, range, blocked, wallf, edge, mode)
  if mode ~= nil and mode ~= "proximity" then return true, 1 end
  local r = blocked and range * (wallf or Core.DEFAULTS.wall) or range
  local g = Core.gain(dist, r, edge or Core.DEFAULTS.edge)
  return g > 0, g
end

--- How far his voice carries: a normal voice, or a whisper when the person he is talking to is crouching.
function L.voice_range(base, crouching) return Core.effective_range({ range = base }, { crouching = crouching }) end

-- ------------------------------------------------------------------------------------------------ the client's animation
--- Hat tilt toward the person talking. (dx, dy) = that person's offset from him in HIS frame (x forward, y left), strength 0..1.
--- Returns pitch (positive tips the front down), roll (positive tips the left down), both in degrees, at most 14 * strength.
function L.hat_tilt(dx, dy, strength)
  local len = math.sqrt(dx * dx + dy * dy)
  if len < 1e-6 then return 0, 0 end
  local s = math.max(0, math.min(1, strength or 1)) * 14
  return s * dx / len, s * dy / len
end

--- Move `cur` toward `target` at `rate` (1/s), exponentially.
function L.approach(cur, target, rate, dt) return cur + (target - cur) * (1 - math.exp(-rate * math.max(dt, 0))) end

--- A stand-in loudness 0..1 for a voice whose real level cannot be read: syllable-like pulses at about 4 Hz, with pauses.
function L.speech_envelope(t)
  local syll = 0.5 + 0.5 * math.sin(t * 25)
  local phrase = math.max(0, math.sin(t * 2.3 + 1.1) * 1.6)
  return math.max(0, math.min(1, syll * math.min(1, phrase)))
end

return L
