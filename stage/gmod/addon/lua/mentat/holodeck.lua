-- holodeck.lua -- the holodeck inside Garry's Mod (server side).
--
-- A player says "Skippy, add three couches" (in chat, or by voice through the bridge) or types `holodeck add three couches`
-- in the console. The parser turns the words into a request, the planner turns the request into poses that respect the
-- room's walls and floor and what already stands, and this file spawns real models (a catalog measured in the engine) with a
-- short materialising fade. Everything it makes is registered, so "remove the couch", "move it closer", "turn the chair around",
-- "undo" and "clear" work on exactly those things.
--
-- The parser and planner are pure Lua and tested offline (stage/gmod/holodeck/); this file is only the engine glue:
-- traces for floors and walls, entity spawning, chat, undo, and the limits that keep a shared server safe (prop caps, the
-- planner's time budget, what the bridge may be asked, who may hear what Skippy says).
if not SERVER then return end

Holodeck = Holodeck or {}
local H = Holodeck

local Parse = include("mentat/holodeck/parse.lua")
local Plan = include("mentat/holodeck/plan.lua")
local Layout = include("mentat/holodeck/layout.lua")
local Library = include("mentat/holodeck/library.lua")
local raw_catalog = include("mentat/holodeck/catalog.lua")

local cv_content = CreateConVar("holodeck_content", "all", FCVAR_ARCHIVE, "all = every model this server has | hl2 = only Half-Life 2 models (what every player owns)")

H.parse_mod, H.plan_mod, H.layout, H.library = Parse, Plan, Layout, Library

--- Where a model stands in the CURRENT variant list of a kind: its index and variant, or nil when the library no longer has it.
--- The one place a model becomes an index, so a stale index can never name a different variant.
local function find_variant(kind_id, model)
  local k = H.catalog and H.catalog.kinds[kind_id]
  if k and model then
    for vi, v in ipairs(k.variants) do
      if v.model == model then return vi, v end
    end
  end
  return nil
end

--- Does the engine on this server still have this model (whatever the library says about it)?
local function model_present(model)
  if not model then return false end
  local ok, valid = pcall(util.IsValidModel, model)
  return ok and valid and true or false
end

H.found_kinds = H.found_kinds or {}      -- kind id -> a kind record the SkippyVerse resolver found this session (see "found objects")
H.found_order = H.found_order or {}      -- their ids, oldest first

--- The library the parser and planner see: the models this server can show, then this session's found kinds (placeholder boxes).
--- A found kind never replaces one of the server's own.
local function assemble(base)
  local cat = { version = base.version, hu_m = base.hu_m, order = {}, kinds = {}, recipes = base.recipes }
  for i, id in ipairs(base.order) do cat.order[i] = id; cat.kinds[id] = base.kinds[id] end
  for _, id in ipairs(H.found_order) do
    if H.found_kinds[id] and not cat.kinds[id] then cat.kinds[id] = H.found_kinds[id]; cat.order[#cat.order + 1] = id end
  end
  return cat
end

--- Parser and planner over `cat`; and things already standing keep their model, found again in the (possibly different) variant
--- list. One whose model left the library stays LEGACY: no index (the old one would name another variant of the new list), but it
--- keeps the variant it was built from, which is all that moving, turning and removing it need.
local function install_library(cat)
  H.catalog = cat
  H.parser = Parse.new(cat)
  H.planner = Plan.new(cat)
  for _, e in pairs(H.registry or {}) do
    local vi, v = find_variant(e.kind, e.model)
    if v then e.vi, e.variant, e.legacy = vi, v, nil else e.vi, e.legacy = nil, true end
  end
end

--- (Re)build the parser and planner over the models this server can actually show.
function H.BuildLibrary()
  local filtered, report = Library.filter(raw_catalog, util.IsValidModel, { hl2_only = cv_content:GetString() == "hl2" })
  H.base_catalog, H.library_report = filtered, report
  install_library(assemble(filtered))
  MsgC(Color(0, 255, 136), "[holodeck] ", Color(255, 255, 255), report.kept .. " of " .. report.total .. " catalog models are available here, " ..
       #filtered.order .. " kinds" .. (#report.lost_kinds > 0 and (" (no model at all for: " .. table.concat(report.lost_kinds, ", ") .. ")") or "") .. "\n")
  return report
end
H.registry = H.registry or {}            -- id -> entry
H.order = H.order or {}                  -- ids in creation order
H.next_id, H.seq, H.group = H.next_id or 1, H.seq or 0, H.group or 0
H.undo_stacks = H.undo_stacks or {}      -- player key -> list of operations
H.last_group = H.last_group or {}        -- player key -> ids of what they made last
H.absent = H.absent or {}                -- player key -> CurTime() when they left
H.next_ok = H.next_ok or {}              -- player key -> CurTime() before which they may not ask again
H.agent_busy = H.agent_busy or {}        -- player key -> true while the agent is thinking about their last request
H.prev_req = H.prev_req or {}            -- player key -> their previous request (for "it" and the like)
H.last_add = H.last_add or {}            -- player key -> the last request of theirs that really added something (for "another one", "again")
local function IsValidPlayer(ply) return ply ~= nil and IsValid(ply) and ply:IsPlayer() end
H.BuildLibrary()
H.pending = 0              -- spawns still scheduled
H.pending_spawns = H.pending_spawns or {}   -- entry id -> { owner =, run = fn } for props whose turn has not come yet
H.log = H.log or {}
H.debug = true

util.AddNetworkString("holodeck_say")
util.AddNetworkString("holodeck_audio")
util.AddNetworkString("holodeck_voice")

local cv_max = CreateConVar("holodeck_max_ents", "400", FCVAR_ARCHIVE, "Most props the holodeck will keep standing at once")
local cv_admin = CreateConVar("holodeck_admin_only", "0", FCVAR_ARCHIVE, "1 = only admins may use the holodeck")
local cv_stagger = CreateConVar("holodeck_stagger", "0.05", FCVAR_ARCHIVE, "Seconds between one prop appearing and the next")
local cv_wake = CreateConVar("holodeck_need_wake", "1", FCVAR_ARCHIVE, "1 = chat and voice need the wake word (Skippy); the console never does")
local cv_bridge = CreateConVar("holodeck_bridge", "", FCVAR_ARCHIVE + FCVAR_PROTECTED, "Skippy bridge URL, e.g. http://127.0.0.1:8777 (empty = no agent and no voice)")
-- A secret: deliberately NOT FCVAR_ARCHIVE. An archived convar is written to the game's server.vdf, so a token that was rotated or
-- revoked would stay on disk for ever. Set it at each start (server.cfg, launch line or the launcher) and it lives in memory only.
local cv_token = CreateConVar("holodeck_bridge_token", "", FCVAR_PROTECTED + FCVAR_DONTRECORD, "shared token for the Skippy bridge (never saved to disk)")
local cv_agent = CreateConVar("holodeck_agent", "1", FCVAR_ARCHIVE, "0 = never ask the agent, 1 = ask when the words build nothing, 2 = ask for every request to add things")
local cv_agent_destructive = CreateConVar("holodeck_agent_destructive", "0", FCVAR_ARCHIVE, "1 = the agent may also remove, move and clear (default: it may only add)")
local cv_persona = CreateConVar("holodeck_persona", "1", FCVAR_ARCHIVE, "1 = Skippy answers in his own words (through the bridge), else a plain sentence")
local cv_cooldown = CreateConVar("holodeck_cooldown", "1.0", FCVAR_ARCHIVE, "Seconds a player must wait between requests")
local cv_per_player = CreateConVar("holodeck_max_per_player", "150", FCVAR_ARCHIVE, "Most props one player may keep standing")
local cv_voice = CreateConVar("holodeck_voice_out", "1", FCVAR_ARCHIVE, "1 = Skippy speaks his answers aloud, from where he built things")
local cv_calls = CreateConVar("holodeck_bridge_per_min", "24", FCVAR_ARCHIVE, "Most bridge calls (agent, persona, voice) the whole server may make per minute")
local cv_chars = CreateConVar("holodeck_voice_chars_per_min", "1800", FCVAR_ARCHIVE, "Most characters of speech the whole server may ask the bridge to voice per minute")
local cv_inflight = CreateConVar("holodeck_bridge_inflight", "4", FCVAR_ARCHIVE, "Most bridge calls waiting for an answer at once")
local cv_real_mass = CreateConVar("holodeck_real_mass", "1", FCVAR_ARCHIVE, "1 = props get the real-world mass the catalog documents; 0 = the engine's own gameplay masses")
local cv_orphans = CreateConVar("holodeck_orphan_minutes", "30", FCVAR_ARCHIVE, "Minutes after which the props of a player who left are taken down (0 = keep until the room is needed)")

local HEAR = 2600                -- how far from where Skippy stands his words and voice carry (HU)
local MAX_AUDIO = 160000         -- the biggest spoken clip we will relay (bytes); the bridge makes about 5 KB per second
local CHUNK = 12000              -- bytes of audio per network message
local MAX_QUEUED_CLIPS = 3
local MAX_INPUT = 400            -- characters of a request that are read at all

-- ------------------------------------------------------------------------------------------------ small helpers
--- Printable, single-spaced text of at most `n` characters: nothing a player or a model typed can carry control codes.
local function clean_text(s, n)
  s = tostring(s or "")
  s = s:gsub("[%c\127]", " "):gsub("%s+", " "):match("^%s*(.-)%s*$") or ""
  s = string.sub(s, 1, n or 240)
  if utf8 and utf8.force then s = utf8.force(s) end
  return s
end

local function player_key(ply) return IsValidPlayer(ply) and ply:SteamID64() or "console" end

--- The connected player a registry owner key names, or nil (left, or the console).
local function player_of(key)
  if not key or key == "console" then return nil end
  for _, p in ipairs(player.GetAll()) do
    if p:SteamID64() == key then return p end
  end
  return nil
end

--- A refilling allowance: `cap` units at most, `rate` units back per second.
local function bucket(cap, rate) return { cap = cap, rate = rate, tokens = cap, t = SysTime() } end
local function refill(b)
  local now = SysTime()
  b.tokens = math.min(b.cap, b.tokens + (now - b.t) * b.rate)
  b.t = now
end
local function take(b, n)
  refill(b)
  if b.tokens >= n then b.tokens = b.tokens - n; return true end
  return false
end

-- Every allowance has two layers: each player's own share (a refusal there stops that player alone) and a global ceiling for
-- the whole server. The shares are small enough that the players who stay inside theirs can never empty the ceiling.
local global_calls, global_chars, planner_cpu = bucket(12, 0.4), bucket(1800, 30), bucket(0.4, 0.1)
local PLAYER_CPU_CAP, PLAYER_CPU_RATE = 0.15, 0.02      -- seconds of planning a player may bank, and get back per second
local PLAYER_CALLS_CAP, PLAYER_CALL_SHARE = 3, 8         -- bridge calls a player may bank; a player's rate is the server's / 8
H.player_buckets = H.player_buckets or {}
H.inflight_by = H.inflight_by or {}                      -- player key -> bridge calls of theirs waiting for an answer
local function player_bucket(key, kind)
  local set = H.player_buckets[key]
  if not set then set = {}; H.player_buckets[key] = set end
  kind = kind == "cpu" and "cpu" or "bridge"
  if not set[kind] then set[kind] = kind == "cpu" and bucket(PLAYER_CPU_CAP, PLAYER_CPU_RATE) or bucket(PLAYER_CALLS_CAP, 1 / 20) end
  return set[kind]
end

-- ------------------------------------------------------------------------------------------------ who may change what
--- Everyone's things belong to whoever made them; admins, the host of a listen server, a player alone in single player and
--- the console may change anything.
local function manages_all(ply)
  if not IsValidPlayer(ply) then return true end
  return ply:IsAdmin() or game.SinglePlayer() or ply:IsListenServerHost()
end

local function can_touch(ply, entry)
  return manages_all(ply) or entry.owner == player_key(ply)
end

-- ------------------------------------------------------------------------------------------------ the bridge (agent and voice)
local function bridge_url() return cv_bridge:GetString() end

--- May this player (and the server) spend one more bridge call of this kind now? `chars` is the length to be voiced.
--- Every bridge call costs the owner a model run, a speech synthesis or credit, so they are rationed.
local function bridge_allow(kind, ply, chars)
  if bridge_url() == "" then return false end
  if H.thinking >= cv_inflight:GetInt() then return false end
  local mine
  if IsValidPlayer(ply) and not manages_all(ply) then
    local key = player_key(ply)
    if (H.inflight_by[key] or 0) >= math.max(1, math.floor(cv_inflight:GetInt() / 2)) then return false end   -- one player cannot fill the queue
    mine = player_bucket(key, "bridge")
    mine.cap, mine.rate = PLAYER_CALLS_CAP, cv_calls:GetFloat() / 60 / PLAYER_CALL_SHARE
    if not take(mine, 1) then return false end
  end
  global_calls.cap, global_calls.rate = math.max(2, cv_calls:GetFloat() / 2), cv_calls:GetFloat() / 60
  if not take(global_calls, 1) then
    if mine then mine.tokens = mine.tokens + 1 end
    return false
  end
  if kind == "speak" then
    global_chars.cap, global_chars.rate = cv_chars:GetFloat(), cv_chars:GetFloat() / 60
    if not take(global_chars, chars or 0) then
      global_calls.tokens = global_calls.tokens + 1
      if mine then mine.tokens = mine.tokens + 1 end
      return false
    end
  end
  return true
end

--- POST json to the bridge. ok(body, headers) on 200, fail(reason) otherwise. Returns false if nothing was sent.
H.thinking = 0             -- bridge calls in flight
local function bridge_post(path, tbl, ok, fail, timeout, quiet)
  if bridge_url() == "" then return false end
  if not quiet then H.thinking = H.thinking + 1 end         -- (quiet: a status poll, not a piece of work; it never crowds out real calls)
  local who = tbl and tbl.who
  if who and not quiet then H.inflight_by[who] = (H.inflight_by[who] or 0) + 1 end
  local finished = false
  local function settle()
    if finished then return end
    finished = true
    if quiet then return end
    H.thinking = math.max(0, H.thinking - 1)
    if who then H.inflight_by[who] = math.max(0, (H.inflight_by[who] or 1) - 1); if H.inflight_by[who] == 0 then H.inflight_by[who] = nil end end
  end
  local sent = HTTP({
    url = bridge_url() .. path, method = "POST", type = "application/json", body = util.TableToJSON(tbl),
    headers = { ["X-Skippy-Token"] = cv_token:GetString() }, timeout = timeout or 20,
    success = function(code, body, headers)
      settle()
      if code == 200 then ok(body, headers) else fail("HTTP " .. tostring(code) .. " " .. tostring(body):sub(1, 120)) end
    end,
    failed = function(reason) settle(); fail(tostring(reason)) end,
  })
  if sent == false then settle(); fail("the engine refused the request (is -allowlocalhttp set?)"); return true end
  return true
end
H.BridgePost = bridge_post

-- ------------------------------------------------------------------------------------------------ Skippy's voice
-- Spoken clips wait in one queue and go out a chunk at a time: a whole clip in one tick can overflow a client's reliable
-- buffer (about 256 KB) and disconnect them.
local audio_id, audio_q = 0, {}

local function audio_pump()
  local clip = audio_q[1]
  if not clip then timer.Remove("holodeck_audio_pump"); return end
  local alive = {}
  for _, p in ipairs(clip.targets) do if IsValid(p) then alive[#alive + 1] = p end end
  clip.targets = alive
  if #alive == 0 or clip.i > clip.total then table.remove(audio_q, 1); return end
  local data = clip.bytes:sub((clip.i - 1) * CHUNK + 1, clip.i * CHUNK)
  net.Start("holodeck_audio")
  net.WriteUInt(clip.id, 16)
  net.WriteUInt(clip.i, 8)
  net.WriteUInt(clip.total, 8)
  net.WriteVector(clip.pos)
  net.WriteUInt(#data, 16)
  net.WriteData(data, #data)
  net.Send(alive)
  clip.i = clip.i + 1
  if clip.i > clip.total then table.remove(audio_q, 1) end
end

--- Queue audio bytes for everyone who could hear Skippy where he stands (and the one who asked).
local function send_audio(bytes, pos, ply)
  local total = math.ceil(#bytes / CHUNK)
  if #bytes < 200 or #bytes > MAX_AUDIO or total < 1 or total > 255 then H.last_speak_error = "audio size " .. #bytes .. " not acceptable"; return end
  if #audio_q >= MAX_QUEUED_CLIPS then H.last_speak_error = "audio queue full"; return end
  local targets, at = {}, pos or (IsValidPlayer(ply) and ply:GetPos()) or Vector(0, 0, 0)
  for _, p in ipairs(player.GetAll()) do
    if p == ply or p:GetPos():DistToSqr(at) < HEAR * HEAR then targets[#targets + 1] = p end
  end
  audio_id = (audio_id % 60000) + 1
  H.last_audio = { id = audio_id, bytes = #bytes, chunks = total, to = #targets }
  if #targets == 0 then return end
  audio_q[#audio_q + 1] = { id = audio_id, bytes = bytes, total = total, i = 1, pos = at, targets = targets }
  if not timer.Exists("holodeck_audio_pump") then timer.Create("holodeck_audio_pump", 0.12, 0, audio_pump) end
end

local function looks_like_mp3(b)
  if b:sub(1, 3) == "ID3" then return true end
  local b1, b2 = b:byte(1, 2)
  return b1 == 0xFF and b2 ~= nil and b2 >= 0xE0
end

--- Skippy speaks: the bridge makes the audio, the server hands it to whoever is near.
function H.Speak(text, pos, ply)
  text = clean_text(text, 220)
  if not cv_voice:GetBool() or text == "" then return false end
  if not bridge_allow("speak", ply, #text) then return false end
  return bridge_post("/speak", { text = text, who = player_key(ply) }, function(body, headers)
    if #body > 200 and looks_like_mp3(body) then
      local voice = headers and (headers["X-Skippy-Voice"] or headers["x-skippy-voice"]) or "?"
      if voice ~= H.last_voice then print("[holodeck] Skippy's voice is now: " .. tostring(voice)) end     -- elevenlabs = his own; sapi = Windows speech
      H.last_voice = voice
      send_audio(body, pos, ply)
    else
      H.last_speak_error = "the bridge sent something that is not speech (" .. #body .. " bytes)"
    end
  end, function(reason)
    H.last_speak_error = reason
  end, 30)
end

-- ------------------------------------------------------------------------------------------------ speaking
local function name_of(ply) return IsValidPlayer(ply) and (clean_text(ply:Nick(), 32) .. " " .. ply:SteamID64()) or "console" end

local function audience(ply, pos)
  local out, seen = {}, {}
  local function add(p) if IsValid(p) and not seen[p] then seen[p] = true; out[#out + 1] = p end end
  if IsValidPlayer(ply) then add(ply) end
  local at = pos or (IsValidPlayer(ply) and ply:GetPos())
  if at then
    for _, p in ipairs(player.GetAll()) do
      if p:GetPos():DistToSqr(at) < HEAR * HEAR then add(p) end
    end
  end
  return out
end

--- A line only the requester sees (refusals, what was not understood): never broadcast, never spoken.
function H.Tell(ply, text)
  text = clean_text(text, 240)
  if text == "" then return end
  H.last_tell = text
  print("[holodeck] (to " .. name_of(ply) .. ") " .. text)
  if IsValidPlayer(ply) then
    net.Start("holodeck_say")
    net.WriteString(text)
    net.WriteString("")
    net.WriteUInt(0, 32)
    net.Send(ply)
  end
end

--- Say it in chat to the requester and to whoever is within earshot of where Skippy stands (never to the whole server), and
--- aloud if the bridge has a voice. `pos` is where the voice comes from; opts.speak = false keeps it to text.
function H.Say(ply, text, pos, opts)
  opts = opts or {}
  text = clean_text(text, 240)
  if text == "" then return end
  H.last_say = text
  H.log[#H.log + 1] = text
  if #H.log > 200 then table.remove(H.log, 1) end
  print("[holodeck] (for " .. name_of(ply) .. ") " .. text)
  local targets = audience(ply, pos)
  if #targets > 0 then
    net.Start("holodeck_say")
    net.WriteString(text)
    net.WriteString(IsValidPlayer(ply) and clean_text(ply:Nick(), 24) or "")
    -- who asked, as a number: the client decides "is this my own line" by it, never by comparing names (a name is changed by
    -- clean_text, so the two sides disagreed for double spaces, long non-ASCII names and the like)
    net.WriteUInt(IsValidPlayer(ply) and ply:UserID() or 0, 32)
    net.Send(targets)
  end
  hook.Run("HolodeckSaid", ply, text)
  if opts.speak ~= false then H.Speak(text, pos or (IsValidPlayer(ply) and ply:GetPos()) or nil, ply) end
end

-- ------------------------------------------------------------------------------------------------ the world, as the planner asks about it
local MASK = MASK_SOLID_BRUSHONLY      -- walls: the map's own brushes (see wall_probe)

--- What blocks a footprint besides the world: players (alive, walking), NPCs, ragdolls and other people's props. The holodeck's
--- own props are the registry's business, so they never block here (a chair may sit under its table).
local function blocks(ent)
  if ent.HolodeckId then return false end
  if ent:IsPlayer() then return ent:Alive() and ent:GetMoveType() ~= MOVETYPE_NOCLIP end
  return true
end

--- What counts as GROUND (and as a ceiling) for the floor and ceiling probes: the map's own brushes and brush entities, plus a
--- prop somebody froze (a platform, a floor built of props). With the old brush-only mask a requester standing ON props was
--- planned onto the world floor far below. Not ground: players, NPCs, ragdolls, loose (unfrozen) props, the holodeck's own props
--- (the registry plans those) and anything else lying around (weapons, items): a stray crate is not a floor. Deliberately NOT
--- `blocks`, which says yes to a living player, so a probe down through a person would land on their head.
local function ground(ent)
  if ent.IsWorld and ent:IsWorld() then return true end
  if ent.HolodeckId then return false end
  if ent.IsPlayer and ent:IsPlayer() then return false end
  if (ent.IsNPC and ent:IsNPC()) or (ent.IsNextBot and ent:IsNextBot()) then return false end
  local class = ent.GetClass and ent:GetClass() or ""
  if class:sub(1, 12) == "prop_physics" then
    local po = ent.GetPhysicsObject and ent:GetPhysicsObject()
    return po ~= nil and po ~= false and IsValid(po) and po.IsMotionEnabled ~= nil and not po:IsMotionEnabled()
  end
  return class:sub(1, 5) == "func_"
end

local function make_env(origin_z)
  local env = {}
  function env.floor_z(x, y)
    local tr = util.TraceLine({ start = Vector(x, y, origin_z + 48), endpos = Vector(x, y, origin_z - 700), mask = MASK_SOLID, filter = ground })
    if tr.StartSolid or not tr.Hit or tr.HitNormal.z < 0.7 or tr.HitSky then return nil end
    return tr.HitPos.z
  end
  function env.clear(r, z, h)
    -- axis-aligned box around the turned footprint, from just above the floor to the item's height (capped at a person's)
    local hx = r.d / 2 * math.abs(r.fx) + r.w / 2 * math.abs(r.rx)
    local hy = r.d / 2 * math.abs(r.fy) + r.w / 2 * math.abs(r.ry)
    local top = math.min(h or 40, 64)
    local tr = util.TraceHull({
      start = Vector(r.x, r.y, z + 3), endpos = Vector(r.x, r.y, z + 3),
      mins = Vector(-hx + 0.5, -hy + 0.5, 0), maxs = Vector(hx - 0.5, hy - 0.5, math.max(top - 6, 2)), mask = MASK_SOLID, filter = blocks,
    })
    return not tr.Hit and not tr.StartSolid
  end
  --- How high the room is above z at (x, y): the absolute height of the ceiling, or nil outdoors (the planner limits piles by it).
  function env.ceiling_z(x, y, z)
    local tr = util.TraceLine({ start = Vector(x, y, z + 3), endpos = Vector(x, y, z + 600), mask = MASK_SOLID, filter = ground })
    if not tr.Hit or tr.StartSolid or tr.HitSky then return nil end
    return tr.HitPos.z
  end
  -- Walls stay brush-only on purpose: a wall is a big, continuous surface, and the planner snaps furniture to it ("against the
  -- wall"). A frozen prop is not that (a stray frozen lamp would pull a couch to it); the clearance hull already keeps new things
  -- out of other people's props. The ray runs at the probed floor's height, so on a platform it still reaches the room's walls.
  function env.wall_probe(x, y, dx, dy, maxd)
    local fz = env.floor_z(x, y) or origin_z
    local a = Vector(x, y, fz + 30)
    local tr = util.TraceLine({ start = a, endpos = a + Vector(dx, dy, 0) * maxd, mask = MASK })
    if not tr.Hit or tr.StartSolid or tr.HitSky then return nil end
    if math.abs(tr.HitNormal.z) > 0.5 then return nil end                 -- a slope or a ledge, not a wall
    local n = Vector(tr.HitNormal.x, tr.HitNormal.y, 0)
    n:Normalize()
    return { dist = tr.Fraction * maxd, nx = n.x, ny = n.y }
  end
  return env
end

-- ------------------------------------------------------------------------------------------------ what stands, as the planner sees it
local TILT = 12              -- degrees from upright after which a prop counts as knocked over

--- Footprint pose of a variant standing as `ent` stands: the inverse of Plan.entity_pose.
local function footprint_of(v, ent)
  local o, ang = ent:GetPos(), ent:GetAngles()
  local r = math.rad(ang.yaw)
  local x = o.x + (v.cx * math.cos(r) - v.cy * math.sin(r))
  local y = o.y + (v.cx * math.sin(r) + v.cy * math.cos(r))
  return x, y, o.z + v.mins[3], (ang.yaw + v.front) % 360
end

--- Props get pushed, grabbed and knocked over after they appear: look at where they really are before planning around them.
local function resync()
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    local ent = e and e.ent
    if e and IsValid(ent) and e.variant and not e.spawning then
      local ang = ent:GetAngles()
      local moved = ent:GetPos():DistToSqr(Vector(e.origin.x, e.origin.y, e.origin.z)) > 9 or math.abs(math.AngleDifference(ang.yaw, e.yaw)) > 4
      local tilted = math.abs(math.NormalizeAngle(ang.pitch)) > TILT or math.abs(math.NormalizeAngle(ang.roll)) > TILT
      if tilted then
        local mins, maxs = ent:WorldSpaceAABB()
        e.x, e.y, e.z, e.theta = (mins.x + maxs.x) / 2, (mins.y + maxs.y) / 2, mins.z, 0
        e.w, e.d, e.h, e.tilted = maxs.y - mins.y, maxs.x - mins.x, maxs.z - mins.z, true
        e.origin = { x = ent:GetPos().x, y = ent:GetPos().y, z = ent:GetPos().z }
      elseif moved or e.tilted then
        e.x, e.y, e.z, e.theta = footprint_of(e.variant, ent)
        e.w, e.d, e.h, e.tilted = e.variant.w, e.variant.d, e.variant.h, nil
        e.origin = { x = ent:GetPos().x, y = ent:GetPos().y, z = ent:GetPos().z }
        e.yaw = ang.yaw % 360
      end
    end
  end
end

--- The variant index the PLANNER is told for a standing thing. A legacy one (its model left the library) has none of its own; the
--- planner needs some index of the kind to name it, move it and count it, and its stored footprint (x, y, w, d, h, theta) does the
--- rest, so it is given the variant of that kind closest to its size. The registry itself never takes this index.
local function planner_vi(e)
  if not e.legacy then return e.vi end
  local k = H.catalog.kinds[e.kind]
  if not k then return nil end
  local best, best_err = nil, math.huge
  for vi, v in ipairs(k.variants) do
    local err = math.abs((v.w or 0) - (e.w or 0)) + math.abs((v.d or 0) - (e.d or 0)) + math.abs((v.h or 0) - (e.h or 0))
    if err < best_err then best, best_err = vi, err end
  end
  return best
end

local function registry_list(owner)
  local out = {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e and (not owner or e.owner == owner) then
      out[#out + 1] = { id = e.id, kind = e.kind, vi = planner_vi(e), x = e.x, y = e.y, z = e.z, theta = e.theta, w = e.w, d = e.d, h = e.h,
                        color_name = e.color_name, group = e.group, seq = e.seq, owner = e.owner, tilted = e.tilted, legacy = e.legacy }
    end
  end
  return out
end

local function owner_count(key)
  local n = 0
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e and e.owner == key then n = n + 1 end
  end
  return n
end

local function last_ids(ply)
  local key, out = player_key(ply), {}
  for _, id in ipairs(H.last_group[key] or {}) do
    local e = H.registry[id]
    if e and can_touch(ply, e) then out[#out + 1] = id end
  end
  return out
end

local function context(ply, opts)
  opts = opts or {}
  resync()
  local pos, yaw
  if IsValidPlayer(ply) then
    pos = ply:GetPos()
    yaw = ply:EyeAngles().yaw
  else
    local p = opts.pos or { x = 0, y = 0, z = 0 }
    pos, yaw = Vector(p.x, p.y, p.z), opts.yaw or 0
  end
  local trusted = manages_all(ply)
  local ctx = {
    pos = { x = pos.x, y = pos.y, z = pos.z }, yaw = yaw, seed = opts.seed, registry = registry_list(),
    env = make_env(pos.z), now = SysTime, last = last_ids(ply), players = {},
    budget = opts.budget or (trusted and 0.25 or 0.05),
    max_total = opts.max_total or (trusted and 150 or 40), max_per_kind = opts.max_per_kind or (trusted and 60 or 40),
  }
  for _, p in ipairs(player.GetAll()) do
    if p ~= ply and p:Alive() then ctx.players[#ctx.players + 1] = { x = p:GetPos().x, y = p:GetPos().y, z = p:GetPos().z } end
  end
  if IsValidPlayer(ply) and not trusted then ctx.mine = registry_list(ply:SteamID64()) end
  if IsValidPlayer(ply) then
    local tr = ply:GetEyeTrace()
    if tr.Hit and tr.HitWorld and tr.HitNormal.z > 0.7 and tr.HitPos:Distance(pos) < 2000 then
      ctx.aim = { x = tr.HitPos.x, y = tr.HitPos.y, z = tr.HitPos.z }
    end
  elseif opts.aim then
    ctx.aim = opts.aim
  end
  return ctx
end

-- ------------------------------------------------------------------------------------------------ making and unmaking
local function push_undo(ply, op)
  local key = player_key(ply)
  local st = H.undo_stacks[key] or {}
  H.undo_stacks[key] = st
  st[#st + 1] = op
  if #st > 30 then table.remove(st, 1) end
end

local function entity_count() return #H.order end

local function fade_in(ent, color, duration)
  local c = color or { 255, 255, 255 }
  ent:SetRenderMode(RENDERMODE_TRANSALPHA)
  local steps, i = 12, 0
  local name = "holodeck_fade_" .. ent:EntIndex()
  timer.Create(name, duration / steps, steps, function()
    if not IsValid(ent) then timer.Remove(name); return end
    i = i + 1
    ent:SetColor(Color(c[1], c[2], c[3], math.floor(255 * i / steps)))
    if i >= steps and not color then ent:SetRenderMode(RENDERMODE_NORMAL) end
  end)
  ent:SetColor(Color(c[1], c[2], c[3], 0))
end

--- Is a living player standing where this prop is about to be set free?
local function player_inside(ent)
  local mins, maxs = ent:WorldSpaceAABB()
  for _, e in ipairs(ents.FindInBox(mins, maxs)) do
    if IsValid(e) and e:IsPlayer() and e:Alive() and e:GetMoveType() ~= MOVETYPE_NOCLIP then return true end
  end
  return false
end

--- Set one registered thing free once nobody stands in it. Never solidifies around a person: it waits (passable) and says so,
--- and if somebody simply stays for minutes the prop is taken down instead.
local WAIT_FAST, WAIT_SLOW, WAIT_GIVE_UP = 20, 145, 20 + 145     -- polls: 20 x 0.5 s, then 145 x 2 s (about five minutes)

local remove_entry                                                       -- defined below

--- The entity for a standing thing: a `holodeck_found` box for a found kind (its variant names the class and the card's numbers),
--- a prop_physics of the library's model for everything else.
local function create_entity(entry)
  local v = entry.variant
  if v and v.entity then
    local ent = ents.Create(v.entity)
    if not IsValid(ent) then return nil end
    local f = v.found or {}
    local ok = ent.SetFound and ent:SetFound({ mins = v.mins, maxs = v.maxs, name = f.name, source = f.source, licence = f.licence,
                                                basis = f.basis, mass = cv_real_mass:GetBool() and v.mass or 0 })
    if not ok then ent:Remove(); return nil end
    return ent
  end
  local ent = ents.Create("prop_physics")
  if not IsValid(ent) then return nil end
  ent:SetModel(entry.model)
  return ent
end

local function spawn_prop(entry, fade)
  local ent = create_entity(entry)
  if not ent then return nil end
  ent:SetPos(Vector(entry.origin.x, entry.origin.y, entry.origin.z))
  ent:SetAngles(Angle(0, entry.yaw, 0))
  ent:Spawn()
  ent:Activate()
  if not IsValid(ent) then return nil end
  if H.Mesh and entry.variant and entry.variant.found and entry.variant.found.mesh then      -- a found object whose card has a real shape (meshnet.lua)
    H.Mesh.attach(ent, entry.variant.found, player_of(entry.owner))
  end
  ent.HolodeckId = entry.id
  local owner_ply = player_of(entry.owner)                               -- the owner, never whoever pressed undo
  if IsValid(owner_ply) then
    if ent.SetCreator then ent:SetCreator(owner_ply) end
    if ent.CPPISetOwner then ent:CPPISetOwner(owner_ply) end             -- prop-protection addons treat it as the player's own
  end
  ent:SetCollisionGroup(COLLISION_GROUP_PASSABLE_DOOR)                   -- nobody is trapped or flung while it materialises
  local po = ent:GetPhysicsObject()
  local mass = cv_real_mass:GetBool() and entry.variant and entry.variant.mass
  if IsValid(po) then
    if mass and mass > 0 then po:SetMass(math.Clamp(mass, 0.1, 50000)) end   -- the real-world mass the catalog documents
    po:EnableMotion(false)                                               -- held still while it materialises
  end
  if fade then fade_in(ent, entry.color, 0.5)
  elseif entry.color then ent:SetColor(Color(entry.color[1], entry.color[2], entry.color[3])) end
  local waited = 0
  local function release()
    if not IsValid(ent) then return end
    if player_inside(ent) then
      waited = waited + 1
      if waited == WAIT_FAST then
        local o = player_of(entry.owner)
        if o then H.Tell(o, "Something of yours stays see-through while someone stands in it. It turns solid when they step out.") end
      end
      if waited >= WAIT_GIVE_UP then                                     -- someone parked in it: take the prop down, never set it solid around them
        local o = player_of(entry.owner)
        if o then H.Tell(o, "I took down a thing of yours because someone kept standing in it.") end
        if H.registry[entry.id] == entry then remove_entry(entry, true) else ent:Remove() end
        return
      end
      timer.Simple(waited < WAIT_FAST and 0.5 or 2, release)
      return
    end
    ent:SetCollisionGroup(COLLISION_GROUP_NONE)
    local p2 = ent:GetPhysicsObject()
    if IsValid(p2) then p2:EnableMotion(true); p2:Wake() end
    timer.Simple(3.9, function()                                         -- a settled prop sleeps instead of jittering
      if IsValid(ent) then
        local p3 = ent:GetPhysicsObject()
        if IsValid(p3) and p3:GetVelocity():LengthSqr() < 400 then p3:Sleep() end
      end
    end)
  end
  timer.Simple(0.6, release)
  return ent
end

--- Register a placement. A restored one (undo) keeps the id, owner and place in the order it had.
local function register(pl, group, ply)
  H.seq = H.seq + 1
  local id = pl.rid
  if not id or H.registry[id] then id = H.next_id end
  if id >= H.next_id then H.next_id = id + 1 end
  -- The variant comes from the MODEL, looked up in the library as it is now: an index saved earlier (an undo snapshot) can name a
  -- different variant, or none, once the library was rebuilt. A model the library no longer has stays legacy: no index, and the
  -- variant it was built from (carried by the snapshot) for its footprint math and mass.
  local vi, variant, legacy
  if pl.model then
    vi, variant = find_variant(pl.kind, pl.model)
    if not variant then variant, legacy = pl.variant, true end
  else
    vi, variant = pl.vi, H.catalog.kinds[pl.kind] and H.catalog.kinds[pl.kind].variants[pl.vi]
  end
  local entry = {
    id = id, kind = pl.kind, vi = vi, model = pl.model, x = pl.x, y = pl.y, z = pl.z, theta = pl.theta, w = pl.w, d = pl.d, h = pl.h,
    origin = pl.origin, yaw = pl.yaw, color = pl.color, color_name = pl.color_name, group = group, seq = pl.seq or H.seq,
    owner = pl.owner or player_key(ply), variant = variant, legacy = legacy,
  }
  H.registry[id] = entry
  local at = #H.order + 1
  for i, other in ipairs(H.order) do
    if H.registry[other] and H.registry[other].seq > entry.seq then at = i; break end
  end
  table.insert(H.order, at, id)
  return entry
end

local function unregister(id)
  H.registry[id] = nil
  for i, v in ipairs(H.order) do
    if v == id then table.remove(H.order, i); break end
  end
end

hook.Add("EntityRemoved", "Holodeck.Forget", function(ent)
  local id = ent.HolodeckId
  if id and H.registry[id] and H.registry[id].ent == ent then unregister(id) end
end)

function remove_entry(entry, fade)
  local ent = entry.ent
  unregister(entry.id)
  if IsValid(ent) then
    if fade then
      ent:SetRenderMode(RENDERMODE_TRANSALPHA)
      local steps, i = 8, 0
      local c = entry.color or { 255, 255, 255 }
      timer.Create("holodeck_out_" .. ent:EntIndex(), 0.3 / steps, steps, function()
        if not IsValid(ent) then return end
        i = i + 1
        ent:SetColor(Color(c[1], c[2], c[3], 255 - math.floor(255 * i / steps)))
        if i >= steps then ent:Remove() end
      end)
    else
      ent:Remove()
    end
  end
end

--- Everything needed to put a removed thing back exactly as it was, under the same id and owner.
local function snapshot(entry)
  return { rid = entry.id, owner = entry.owner, seq = entry.seq, kind = entry.kind, vi = entry.vi, model = entry.model, x = entry.x, y = entry.y,
           z = entry.z, theta = entry.theta, w = entry.w, d = entry.d, h = entry.h, origin = entry.origin, yaw = entry.yaw, color = entry.color,
           color_name = entry.color_name, group = entry.group, variant = entry.variant, legacy = entry.legacy }
end

-- ------------------------------------------------------------------------------------------------ how many may stand
--- Who would be taken down to make room for `need` more, without taking anyone down: first the props of players who left, oldest
--- first; then the oldest props of whoever holds more than a fair share of the server's allowance. The requester's own things are
--- never taken.
local function eviction_list(ply, need)
  local cap = cv_max:GetInt()
  need = math.min(need, cap)
  local out = {}
  local free = cap - #H.order
  if free >= need then return out end
  local mine = player_key(ply)
  local connected, players = {}, player.GetAll()
  for _, p in ipairs(players) do connected[p:SteamID64()] = true end
  local gone = {}
  local function mark(e) gone[e.id] = true; out[#out + 1] = e; free = free + 1 end
  for _, id in ipairs({ unpack(H.order, 1, math.min(#H.order, 5000)) }) do
    if free >= need then return out end
    local e = H.registry[id]
    if e and e.owner ~= "console" and e.owner ~= mine and not connected[e.owner] then mark(e) end
  end
  local fair = math.max(1, math.floor(cap / math.max(1, #players)))
  local counts, seq = {}, {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e and not gone[id] and e.owner ~= "console" and e.owner ~= mine then
      if not counts[e.owner] then counts[e.owner] = 0; seq[#seq + 1] = e.owner end
      counts[e.owner] = counts[e.owner] + 1
    end
  end
  while free < need do
    local biggest, most = nil, fair
    for _, owner in ipairs(seq) do
      if counts[owner] > most then biggest, most = owner, counts[owner] end
    end
    if not biggest then break end
    for _, id in ipairs(H.order) do
      local e = H.registry[id]
      if e and e.owner == biggest and not gone[id] then mark(e); break end
    end
    counts[biggest] = counts[biggest] - 1
  end
  return out
end

local function make_room(ply, need)
  local victims = eviction_list(ply, need)
  if #victims == 0 then return end
  local freed = {}
  for _, e in ipairs(victims) do
    freed[e.owner] = (freed[e.owner] or 0) + 1
    remove_entry(e, true)
  end
  for owner, n in pairs(freed) do
    local p = player_of(owner)
    if p then H.Tell(p, "I took down " .. n .. " of your things to make room for someone else.") end
  end
end

--- How many things this player may still add under their own limit (managers and the console have none).
local function personal_left(ply)
  if not IsValidPlayer(ply) or manages_all(ply) then return math.huge end
  return cv_per_player:GetInt() - owner_count(ply:SteamID64())
end

local ENGINE_ENTS = 7900     -- the engine's entity table holds 8192; stay clear of it

--- Room the engine itself still has (when it can say), so a request is not promised props the engine will refuse.
local function engine_room()
  if ents.GetCount then
    local ok, n = pcall(ents.GetCount)
    if ok and tonumber(n) then return ENGINE_ENTS - n end
  end
  return math.huge
end

--- How many more things this player may add right now, and which limit is the tight one ("player" or "server"). `want` is the
--- number that will really be built: only for that many is anyone else's work taken down (dry = just ask, take down nothing).
local function room_for(ply, want, dry)
  local left = personal_left(ply)
  local n = want and math.min(want, left) or 0
  local free = cv_max:GetInt() - entity_count()
  if n > 0 and free < n then
    if dry then free = free + #eviction_list(ply, n) else make_room(ply, n); free = cv_max:GetInt() - entity_count() end
  end
  free = math.min(free, engine_room())
  local room, which = free, "server"
  if left <= room then room, which = left, "player" end
  return math.max(room, 0), which
end

--- The server's own rules about spawning props (jails, spawn bans, prop protection blocklists, the gamemode's own refusals and
--- sbox_maxprops) apply to the holodeck too. Asked the way the engine asks its own spawn path: registered hooks first (the first
--- answer that is not nil wins), then the gamemode's function. `hook.Run` is that (`hook.Call` with the active gamemode); with a
--- nil gamemode `hook.Call` returns before GAMEMODE:PlayerSpawnProp, which is where Sandbox keeps its prop limit and DarkRP-style
--- gamemodes keep their jail rules. Only an explicit `false` is a veto: a gamemode without such a function (TTT and the like) does
--- not forbid anything, and `gamemode.Call` would answer false for it. Nothing here may throw.
--- The answer is asked once per MODEL of a request, not once per prop: the holodeck's props are not registered with the player's
--- sandbox counts (AddCount), so the count a rule sees does not move while we build, and one refusal is one notice, not forty.
local function vetoed(ply, model)
  if not IsValidPlayer(ply) then return false end
  local ok1, a = pcall(hook.Run, "PlayerSpawnObject", ply, model, 0)
  local ok2, b = pcall(hook.Run, "PlayerSpawnProp", ply, model)
  if (ok1 and a == false) or (ok2 and b == false) then return true end
  -- A gamemode that does not define PlayerSpawnProp never reaches the sandbox limit through the call above, but the player may
  -- still have sbox_maxprops (their ply:CheckLimit, from the sandbox extension). Asked here ONLY then: where the gamemode
  -- function exists it already made that check, and a second one would count the same limit twice (and show its notice twice).
  local gm = GAMEMODE
  local has_rule = istable(gm) and gm.PlayerSpawnProp ~= nil
  if not has_rule and not (ok2 and b ~= nil) and isfunction(ply.CheckLimit) then
    local ok3, c = pcall(ply.CheckLimit, ply, "props")
    if ok3 and c == false then return true end
  end
  return false
end

--- Is this owner free of the per-player limit? The console, and players who manage everything (admins, the host).
local function owner_exempt(owner, ply, reqkey)
  if owner == "console" then return true end
  if owner == reqkey then return manages_all(ply) end
  local p = player_of(owner)
  return p ~= nil and manages_all(p)
end

--- A batch whose engine spawns failed: forget the missing ids and tell the requester what really stands.
local function batch_failed(batch)
  local made = batch.total - batch.failed
  local st = H.undo_stacks[batch.key]
  if st then
    for i = #st, 1, -1 do
      local op = st[i]
      if op.type == "add" then
        local keep = {}
        for _, id in ipairs(op.ids) do if not batch.failed_ids[id] then keep[#keep + 1] = id end end
        op.ids = keep
        if #keep == 0 then table.remove(st, i) end
      end
    end
  end
  local lg = H.last_group[batch.key]
  if lg then
    local keep = {}
    for _, id in ipairs(lg) do if not batch.failed_ids[id] then keep[#keep + 1] = id end end
    H.last_group[batch.key] = keep
  end
  if IsValidPlayer(batch.ply) then
    H.Tell(batch.ply, "Only " .. made .. " of the " .. batch.total .. " could be created: the server is out of entity slots.")
  end
end

--- Put things on the floor, one after another. The one place props are made, so every caller meets the same limits.
--- Returns the registered entries, how many placements the limits or the server's rules turned away, the placements the LIMITS
--- (not the rules) turned away (so an undo can keep them), and how many the server's rules ruled out.
local function build(ply, placements)
  local reqkey = player_key(ply)
  -- 1. the server's rules first: a vetoed prop never costs anyone anything
  local verdicts, allowed, ruled_out = {}, {}, 0
  for _, pl in ipairs(placements) do
    if verdicts[pl.model] == nil then verdicts[pl.model] = vetoed(ply, pl.model) end
    if verdicts[pl.model] then ruled_out = ruled_out + 1 else allowed[#allowed + 1] = pl end
  end
  -- 2. each OWNER's own limit (a restored thing counts against the player who had it, not whoever pressed undo)
  local left, capped, rejected = {}, {}, {}
  for _, pl in ipairs(allowed) do
    local owner = pl.owner or reqkey
    if left[owner] == nil then
      left[owner] = owner_exempt(owner, ply, reqkey) and math.huge or (cv_per_player:GetInt() - owner_count(owner))
    end
    if left[owner] > 0 then left[owner] = left[owner] - 1; capped[#capped + 1] = pl else rejected[#rejected + 1] = pl end
  end
  -- 3. the server's room: other people's things come down only for what will really be built
  local room = room_for(ply, #capped)
  local kept = {}
  for _, pl in ipairs(capped) do
    if #kept < room then kept[#kept + 1] = pl else rejected[#rejected + 1] = pl end
  end
  H.group = H.group + 1
  local entries = {}
  local delay = cv_stagger:GetFloat()
  local batch = { total = #kept, left = #kept, failed = 0, failed_ids = {}, key = reqkey, ply = ply }
  for i, pl in ipairs(kept) do
    local entry = register(pl, H.group, ply)
    entries[i] = entry
    H.pending = H.pending + 1
    local spawned = false
    local function spawn_now()
      if spawned then return end
      spawned = true
      H.pending_spawns[entry.id] = nil
      H.pending = H.pending - 1
      local failed = false
      if H.registry[entry.id] == entry then
        local ent = spawn_prop(entry, true)
        if ent then entry.ent = ent else unregister(entry.id); failed = true end
      end
      batch.left = batch.left - 1
      if failed then batch.failed = batch.failed + 1; batch.failed_ids[entry.id] = true end
      if batch.left == 0 and batch.failed > 0 then batch_failed(batch) end
    end
    -- Kept where a player's departure can find it (see PlayerDisconnected): a prop still waiting for its turn when its owner
    -- leaves is made right then, while the owner can still be looked up, so it is created as theirs like the rest.
    H.pending_spawns[entry.id] = { owner = entry.owner, run = spawn_now }
    timer.Simple((i - 1) * delay, spawn_now)
  end
  return entries, #placements - #kept, rejected, ruled_out
end

-- ------------------------------------------------------------------------------------------------ is the ground free?
local TUCK_PEN = 14          -- how far a chair may sit under a table (the planner's own tolerance)

--- Is the spot a pose names taken now (by another prop, a person, an NPC, the world)? `ignore` = ids that do not count (the
--- things being put back together). Things resting on something belong to their support and are never judged here.
local function pose_blocked(p, ignore)
  local env = make_env(p.z)
  local fz = env.floor_z(p.x, p.y)
  if fz and p.z - fz > 1.5 then return false end
  local r = Layout.rect(p.x, p.y, p.w, p.d, p.theta)
  for _, id in ipairs(H.order) do
    local e = not ignore[id] and H.registry[id]
    if e and e.x and e.w and not (e.z >= p.z + p.h - 1) and not (p.z >= e.z + e.h - 1) then
      local er = Layout.rect(e.x, e.y, e.w, e.d, e.theta or 0)
      if Layout.overlap(er, r, 0.4) and Layout.penetration(er, r) > math.min(TUCK_PEN, 0.5 * math.min(p.w, p.d, e.w, e.d)) then return true end
    end
  end
  return not env.clear(r, fz or p.z, p.h)
end

--- Move registered things. `guard` (undo): a set of ids that do not count as obstacles; a pose that is taken is not applied.
--- Returns what was moved (their old poses) and the ids that were held back by something in the way.
local function apply_moves(ply, moves, guard)
  local old, blocked = {}, {}
  for _, mv in ipairs(moves) do
    local e = H.registry[mv.id]
    -- what stands on something that moves rides with it, whoever made it; everything else must be the player's own
    if e and (mv.carried or can_touch(ply, e)) and e.variant then
      local v = e.variant
      if guard and pose_blocked({ x = mv.x, y = mv.y, z = mv.z, theta = mv.theta, w = mv.w or v.w, d = mv.d or v.d, h = mv.h or v.h }, guard) then
        blocked[#blocked + 1] = mv.id
      else
        old[#old + 1] = { id = e.id, x = e.x, y = e.y, z = e.z, theta = e.theta, w = e.w, d = e.d, h = e.h, tilted = e.tilted,
                          carried = mv.carried or nil, to = mv.carried and { x = mv.x, y = mv.y, z = mv.z, theta = mv.theta } or nil }
        local origin, yaw = Plan.entity_pose(v, mv.x, mv.y, mv.z, mv.theta)
        e.x, e.y, e.z, e.theta, e.origin, e.yaw = mv.x, mv.y, mv.z, mv.theta, origin, yaw
        e.w, e.d, e.h, e.tilted = v.w, v.d, v.h, nil
        if IsValid(e.ent) then
          local po = e.ent:GetPhysicsObject()
          e.ent:SetPos(Vector(origin.x, origin.y, origin.z))
          e.ent:SetAngles(Angle(0, yaw, 0))
          if IsValid(po) then po:SetVelocity(Vector(0, 0, 0)); po:Wake() end
        end
      end
    end
  end
  return old, blocked
end

-- ------------------------------------------------------------------------------------------------ the entry point
local function centre_of(res, ply)
  local n, cx, cy, cz = 0, 0, 0, 0
  for _, p in ipairs(res.placements or {}) do cx, cy, cz, n = cx + p.x, cy + p.y, cz + p.z, n + 1 end
  if n > 0 then return Vector(cx / n, cy / n, cz / n + 30) end
  return IsValidPlayer(ply) and ply:GetPos() or nil
end

local function fixed_words(list)
  local out = {}
  for _, w in ipairs(list or {}) do
    w = tostring(w):lower():gsub("[^%w ]", ""):sub(1, 24)
    if w ~= "" and #out < 4 then out[#out + 1] = w end
  end
  return out
end

--- Say the outcome: in Skippy's own words when the bridge has them, else the plain sentence. Fixed sentences (limits, undo,
--- nothing to do) never cost a model call, and a model's words are scoped and cleaned like any other line.
local function announce(ply, text, res)
  if not res or not res.say or res.say == "" then return end
  local pos = centre_of(res, ply)
  local opts = { speak = not res.quiet }
  local was_player = IsValidPlayer(ply)
  local function say(line)
    if was_player and not IsValidPlayer(ply) then return end             -- they left while Skippy was thinking
    H.Say(ply, line, pos, opts)
  end
  if cv_persona:GetBool() and bridge_url() ~= "" and not res.fixed and not res.quiet and bridge_allow("reply", ply) then
    local done = false
    local function plain() if not done then done = true; say(res.say) end end
    local notes = {}
    for i, n in ipairs(res.notes or {}) do if i <= 5 then notes[i] = clean_text(n, 160) end end
    local started = bridge_post("/reply", { user = clean_text(text, 160), did = clean_text(res.say, 300), notes = notes, who = player_key(ply) }, function(body)
      local obj = util.JSONToTable(body or "")
      local line = istable(obj) and isstring(obj.say) and clean_text(obj.say, 240) or ""
      if not done and line ~= "" then done = true; res.said = line; say(line) else plain() end
    end, function() plain() end, 12)
    if not started then plain() end
    timer.Simple(13, plain)
  else
    say(res.say)
  end
end

--- Tell the requester privately which of their words were not understood (never broadcast: they are the player's own words).
local function tell_unknown(ply, res)
  local words = fixed_words(res and res.unknown_words)
  if #words > 0 and IsValidPlayer(ply) then H.Tell(ply, "I did not know: " .. table.concat(words, ", ") .. ".") end
end

local function limit_text(ply, which)
  if which == "player" then
    return "You already have " .. owner_count(player_key(ply)) .. " things standing, which is your limit of " .. cv_per_player:GetInt() .. ". Clear some first."
  end
  return "That is as many props as I will keep standing (" .. cv_max:GetInt() .. "). Clear some first."
end

--- May the planner run now? It runs on the server's main thread. Each ordinary player has a share of their own (a player whose
--- share is spent is refused alone); the server as a whole has a ceiling that is always higher than the shares of the players
--- present, so players who stay inside their share can never empty it. Returns true, or false and which limit said no.
local function planner_admit(ply)
  if manages_all(ply) then return true end
  local mine = player_bucket(player_key(ply), "cpu")
  refill(mine)
  if mine.tokens < 0.02 then return false, "own" end
  local n = 0
  for _, p in ipairs(player.GetAll()) do if not manages_all(p) then n = n + 1 end end
  planner_cpu.rate = math.Clamp(1.25 * PLAYER_CPU_RATE * n, 0.1, 0.4)
  planner_cpu.cap = math.Clamp(PLAYER_CPU_CAP * n, 0.4, 0.8)
  refill(planner_cpu)
  if planner_cpu.tokens < 0.02 then return false, "global" end
  return true
end

local function charge_planner(ply, seconds)
  if not IsValidPlayer(ply) or manages_all(ply) then return end         -- the console and admins have their own, bigger budget
  seconds = math.max(seconds, 0)
  local mine = player_bucket(player_key(ply), "cpu")
  mine.tokens = math.max(mine.tokens - seconds, -mine.cap)
  planner_cpu.tokens = math.max(planner_cpu.tokens - seconds, -planner_cpu.cap)
  local key = player_key(ply)
  H.next_ok[key] = math.max(H.next_ok[key] or 0, CurTime() + 10 * seconds)    -- an expensive request earns a longer rest
end

--- Within a few HU and degrees of where a move left it? (A stale undo must not drag something its owner has moved since.)
local function still_at(e, to)
  return math.abs(e.x - to.x) <= 6 and math.abs(e.y - to.y) <= 6 and math.abs((e.z or 0) - (to.z or 0)) <= 6
         and math.abs(Layout.angle_diff(e.theta or 0, to.theta or 0)) <= 6
end

local MAX_RETRIES = 3        -- an undo that could put nothing back this many times in a row is let go

--- Undo the player's last operation. What could not be put back (the limits, something standing there) stays on the stack.
local function undo_one(ply, key)
  local st = H.undo_stacks[key]
  local op = st and table.remove(st)
  if not op then return { action = "undo", say = "There is nothing to undo.", fixed = true } end
  resync()
  local say
  if op.type == "add" then
    local n = 0
    for _, id in ipairs(op.ids) do
      local e = H.registry[id]
      if e and can_touch(ply, e) then remove_entry(e, true); n = n + 1 end
    end
    say = n > 0 and ("Undone: " .. n .. " thing" .. (n == 1 and "" or "s") .. " gone.") or "Those are already gone."
    -- "it" means what the player made BEFORE the group just undone, but only when that group was the one "it" pointed at
    local mine, lg = {}, H.last_group[key]
    for _, id in ipairs(op.ids) do mine[id] = true end
    local points_here = lg ~= nil
    for _, id in ipairs(lg or {}) do if not mine[id] then points_here = false end end
    if points_here then H.last_group[key] = op.prev_last end
  elseif op.type == "remove" then
    local ready, rest, taken, gone = {}, {}, 0, 0
    for _, snap in ipairs(op.snaps) do
      if not H.registry[snap.rid] then
        if not H.catalog.kinds[snap.kind] or not (find_variant(snap.kind, snap.model) or model_present(snap.model)) then
          gone = gone + 1                                                    -- what it was made from is not here any more: never respawned
        elseif pose_blocked(snap, {}) then rest[#rest + 1] = snap; taken = taken + 1
        else ready[#ready + 1] = snap end
      end
    end
    local entries, away, rejected = build(ply, ready)
    for _, snap in ipairs(rejected) do rest[#rest + 1] = snap end
    local n = #entries
    say = n > 0 and ("Undone: " .. n .. " thing" .. (n == 1 and "" or "s") .. " back.") or "Nothing could go back just now."
    if gone > 0 then say = say .. " " .. gone .. " could not go back: this server no longer has " .. (gone == 1 and "its model." or "their models.") end
    if taken > 0 then say = say .. " " .. taken .. " could not go back: something is standing there now." end
    if away > #rejected then say = say .. " " .. (away - #rejected) .. " would break a rule on this server." end
    if #rejected > 0 then say = say .. " " .. #rejected .. " did not fit under the limits." end
    if #rest > 0 then
      op.snaps, op.tries = rest, n > 0 and 0 or (op.tries or 0) + 1
      if op.tries < MAX_RETRIES then push_undo(ply, op); say = say .. " Say undo again to retry those." end
    end
  elseif op.type == "move" then
    local mvs, ignore, stale = {}, {}, 0
    for _, o in ipairs(op.old) do ignore[o.id] = true end
    for _, o in ipairs(op.old) do
      local e = H.registry[o.id]
      if e then
        local carried = nil
        if o.carried then
          if o.to and still_at(e, o.to) then carried = true else stale = stale + 1 end
        end
        if not o.carried or carried then
          mvs[#mvs + 1] = { id = o.id, x = o.x, y = o.y, z = o.z, theta = o.theta, w = o.w, d = o.d, h = o.h, carried = carried }
        end
      end
    end
    local done, blocked = apply_moves(ply, mvs, ignore)
    say = #done > 0 and "Undone." or ((#blocked > 0 and "Nothing could go back just now.") or "Those are gone, so there is nothing to put back.")
    local missing = #mvs - #done
    if missing > 0 then
      say = say .. " " .. missing .. " could not be put back" .. (#blocked > 0 and ": something is in the way." or ".")
    end
    if stale > 0 then say = say .. " " .. stale .. " left where their owners put them." end
    if #blocked > 0 then
      local isb, rest = {}, {}
      for _, id in ipairs(blocked) do isb[id] = true end
      for _, o in ipairs(op.old) do if isb[o.id] then rest[#rest + 1] = o end end
      op.old, op.tries = rest, #done > 0 and 0 or (op.tries or 0) + 1
      if op.tries < MAX_RETRIES then push_undo(ply, op) end
    end
  end
  return { action = "undo", say = say or "Undone.", fixed = true }
end

--- Do what a request says. Returns the result table (its `say` is what to tell the player); announces nothing.
local function run_one(ply, req, opts)
  local key = player_key(ply)
  if req.action == "undo" then
    -- "undo twice", "undo the last three", "undo everything" (req.times, at most 30): the player's OWN undo steps, one after the other
    local n = math.min(math.max(math.floor(tonumber(req.times) or 1), 1), 30)
    local res = undo_one(ply, key)
    if n == 1 or res.say == "There is nothing to undo." then return res end
    local done, says = 1, { res.say }
    while done < n and not res.say:find("Nothing could go back", 1, true) do
      local st = H.undo_stacks[key]
      if not st or #st == 0 then break end
      res = undo_one(ply, key)
      done = done + 1
      says[#says + 1] = res.say
    end
    local say = "Undone " .. done .. " step" .. (done == 1 and "" or "s") .. "."
    if done <= 3 then say = say .. " " .. table.concat(says, " ") end
    return { action = "undo", fixed = true, say = say }
  end

  if req.action == "clear" then
    local snaps, ids, all = {}, {}, manages_all(ply)
    for i = 1, #H.order do ids[i] = H.order[i] end
    for _, id in ipairs(ids) do
      local e = H.registry[id]
      if e and (all or e.owner == key) then snaps[#snaps + 1] = snapshot(e); remove_entry(e, true) end
    end
    local n = #snaps
    if n > 0 then push_undo(ply, { type = "remove", snaps = snaps }) end
    return { action = "clear", removed = n, fixed = true,
             say = n > 0 and ("Cleared " .. n .. " thing" .. (n == 1 and "" or "s") .. (all and "." or " of yours.")) or "There was nothing of yours to clear." }
  end

  if req.action == "add" then
    local room, which = room_for(ply, 1, true)                              -- only asks: nobody's things come down for a request not yet planned
    if room <= 0 then return { action = "add", say = limit_text(ply, which), placements = {}, limit = true, fixed = true, quiet = true } end
  end

  local admitted, why = planner_admit(ply)
  if not admitted then
    return { action = req.action, say = why == "own" and "You have used your share of planning for the moment. Ask again in a few seconds."
                                                        or "I am busy planning for someone else. Ask again in a moment.",
             placements = {}, fixed = true, quiet = true }
  end
  local ctx = context(ply, opts)
  local t0 = SysTime()
  local ok, res = pcall(H.planner.run, H.planner, req, ctx)
  charge_planner(ply, SysTime() - t0)
  if not ok then
    ErrorNoHalt("[holodeck] planner error: " .. tostring(res) .. "\n")
    return { action = "error", say = "I tripped over that one. The error is in the server console.", fixed = true, quiet = true }
  end
  res.request = req
  if res.timed_out then res.fixed = true end                              -- its sentence is a constant: no persona call

  if res.action == "add" and #(res.placements or {}) > 0 then
    local want = #res.placements
    local entries, away, _, ruled_out = build(ply, res.placements)
    if #entries < want then
      local why2 = away > 0 and ((#entries == 0) and "" or " The rest would break a limit or a rule on this server.") or ""
      if #entries == 0 then
        local room, which = room_for(ply)
        res.say = (ruled_out < want and room <= 0 and limit_text(ply, which)) or "None of that could be placed under this server's rules."
        res.fixed, res.quiet = true, true
      else
        res.say = "I had room for only " .. #entries .. " of the " .. want .. "." .. why2
      end
    end
    local ids = {}
    for i, e in ipairs(entries) do ids[i] = e.id end
    res.ids = ids
    if #ids > 0 then
      push_undo(ply, { type = "add", ids = ids, prev_last = H.last_group[key] })   -- so undo can hand "it" back to the group before
      H.last_group[key] = ids
      H.last_add[key] = req                                               -- what "another one" and "again" repeat, after any other request
    end
  elseif res.action == "remove" and res.removals and #res.removals > 0 then
    local snaps = {}
    for _, id in ipairs(res.removals) do
      local e = H.registry[id]
      if e and can_touch(ply, e) then snaps[#snaps + 1] = snapshot(e); remove_entry(e, true) end
    end
    if #snaps > 0 then push_undo(ply, { type = "remove", snaps = snaps }) end
  elseif (res.action == "move" or res.action == "rotate" or res.action == "rearrange") and res.moves and #res.moves > 0 then
    local old = apply_moves(ply, res.moves)
    if #old > 0 then push_undo(ply, { type = "move", old = old }) end
  end
  return res
end

local function finish(ply, text, res, opts)
  announce(ply, text, res)
  tell_unknown(ply, res)
  if opts and opts.done then opts.done(res) end
end

--- Run a parsed (or agent-made) request. Returns the planner's result table.
function H.Execute(ply, req, opts, text)
  opts = opts or {}
  if opts.was_player and not IsValidPlayer(ply) then return nil end     -- the player left while the agent was thinking: do nothing
  local ok, res = pcall(run_one, ply, req, opts)
  if not ok then
    ErrorNoHalt("[holodeck] request error: " .. tostring(res) .. "\n")
    res = { action = "error", say = "I tripped over that one. The error is in the server console.", fixed = true, quiet = true }
  end
  finish(ply, text or req.raw or "", res, opts)
  return res
end

--- Does this parsed request build or change anything the planner understands?
local function actionable(req)
  if req.action ~= "add" then return true end
  if req.scene and req.scene ~= "" then return true end
  for _, it in ipairs(req.items or {}) do
    if not tostring(it.kind):match("^UNKNOWN") and it.kind ~= "@last" then return true end
  end
  return false
end

local function nearby_kinds()
  local seen, out = {}, {}
  for _, id in ipairs(H.order) do
    local k = H.registry[id].kind
    if not seen[k] then seen[k] = true; out[#out + 1] = k end
    if #out >= 12 then break end
  end
  return out
end

--- The agent's answer, made safe: only fields of the right type, sizes clamped, and (unless allowed) only additions.
local function agent_request(obj, text)
  local req = istable(obj.request) and obj.request or nil
  if not (req and isstring(req.action) and req.action ~= "none") then return nil end
  if req.action ~= "add" and not cv_agent_destructive:GetBool() then return nil, "I can only add things when I work from your words alone." end
  local items = {}
  for _, it in ipairs(istable(req.items) and req.items or {}) do
    if istable(it) and isstring(it.kind) and #items < 12 then
      local adjs = {}
      for _, a in ipairs(istable(it.adjectives) and it.adjectives or {}) do if isstring(a) and #adjs < 4 then adjs[#adjs + 1] = a:lower():sub(1, 20) end end
      items[#items + 1] = { kind = it.kind, count = math.Clamp(math.floor(tonumber(it.count) or 1), 0, 60), adjectives = adjs,
                            word = isstring(it.word) and it.word:sub(1, 24) or it.kind, position = #items + 1 }
    end
  end
  local out = { action = req.action, items = items, raw = text, normalised = text:lower(), unknown = {}, notes = {},
                arrangement = (isstring(req.arrangement) and req.arrangement ~= "") and req.arrangement or "unspecified",
                target = isstring(req.target) and req.target or "", scene = isstring(req.scene) and req.scene or "" }
  return out
end

--- Ask the agent (through the bridge) to turn a vague wish into a request; run it. fallback() runs when the agent cannot help.
--- Returns true when the question was sent (the answer arrives later), false when the agent cannot be asked now.
local function ask_agent(ply, text, opts, fallback)
  local key = player_key(ply)
  local was_player = IsValidPlayer(ply)
  if H.agent_busy[key] or not bridge_allow("plan", ply) then return false end
  local function gone() return was_player and not IsValidPlayer(ply) end
  H.agent_busy[key] = true
  local sent = bridge_post("/plan", { text = text, nearby = nearby_kinds(), who = key }, function(body)
    H.agent_busy[key] = nil
    if gone() then print("[holodeck] the player left; dropping the agent's answer"); return end
    if GetConVar("developer"):GetInt() > 0 or H.debug then print("[holodeck] agent answered: " .. tostring(body):sub(1, 600)) end
    local obj = util.JSONToTable(body or "")
    if not istable(obj) then return fallback() end
    local line = isstring(obj.say) and clean_text(obj.say, 240) or ""
    local req, why = agent_request(obj, text)
    if not req then
      if line == "" then line = why or "" end
      if line ~= "" then
        H.Say(ply, line)
        if opts.done then opts.done({ action = "none", say = line }) end
      else
        fallback()
      end
      return
    end
    local res = H.Execute(ply, req, opts, text)
    if res then
      res.by_agent = true
      for _, n in ipairs(istable(obj.notes) and obj.notes or {}) do
        if isstring(n) then res.notes = res.notes or {}; table.insert(res.notes, clean_text(n, 160)) end
      end
    end
  end, function(reason)
    H.agent_busy[key] = nil
    if gone() then return end
    H.last_agent_error = reason
    print("[holodeck] the agent could not help: " .. tostring(reason):sub(1, 200))
    fallback()
  end, 30)
  if not sent then H.agent_busy[key] = nil end
  return sent
end

-- ------------------------------------------------------------------------------------------------ found objects
-- A noun the library lacks ("a fire hydrant") is asked of the bridge's /resolve (the SkippyVerse inventory). An answer that may be
-- placed carries a planner kind; it is validated here (the bridge is trusted to be the bridge, not to be right), added to the live
-- library as a virtual kind whose one variant is a `holodeck_found` box of the card's size and mass, and the request is then run
-- again through the normal pipeline, so layouts, overlap rules, undo, limits and ownership treat it like any other prop. Everything
-- else the resolver says ("the closest I have is ...", not found, too big, size unknown, not shareable) is only a sentence.
local cv_found = CreateConVar("holodeck_found", "1", FCVAR_ARCHIVE, "1 = ask the bridge about things the library lacks and show what it finds as placeholder boxes")
H.Wisps = include("mentat/holodeck/wisps.lua")            -- the failure wisps: a floating light carrying Skippy's reason (not a placed thing, see wisps.lua)
H.Wisps.init({ key = player_key })
local WispFx = include("mentat/fx/wisp.lua")                -- (the pure half: reason class from the resolver's status)
H.Mesh = include("mentat/holodeck/meshnet.lua")           -- the real shape of a found object, relayed to the clients (see meshnet.lua)
H.Mesh.init({ post = bridge_post, allow = bridge_allow, key = player_key,
              quiet_post = function(path, tbl, ok, fail, timeout) return bridge_post(path, tbl, ok, fail, timeout, true) end,
              say = function(ply, text) H.Say(ply, text, nil, { speak = IsValid(ply) }) end,           -- the transporter: status polls, and Skippy saying why one gave up
              wisp = function(ply, text, class, pos) return H.Wisps.drop(ply, text, class, pos) end })   -- ...and leaving the reason behind as a floating wisp
local F = {}
F.MAX_KINDS, F.MAX_NOUNS, F.MIN_HU, F.MAX_HU = 64, 3, 1, 2000      -- kinds kept per session, nouns asked per sentence, size limits (HU)
F.refused = {}                                                       -- noun -> { say =, t = } : a definite "no", remembered for a while
F.REFUSED_FOR = 300
H.resolving = H.resolving or {}                                      -- player key -> true while their nouns are being looked up

--- A noun as the bridge and the library write it: lower-case letters, digits, hyphen, apostrophe and single spaces, up to 40.
function F.noun(s)
  s = tostring(s or ""):lower():gsub("[^a-z0-9'%-]+", " "):gsub("%s+", " "):match("^%s*(.-)%s*$") or ""
  return string.sub(s, 1, 40):match("^(.-)%s*$")
end

local function finite(n) return type(n) == "number" and n == n and n > -math.huge and n < math.huge end

--- The planner kind for a bridge answer, or nil and why. Rebuilds the box from the card's size: the numbers that arrive are only
--- w, d, h and the mass; everything else about the variant is fixed here (front unknown => symmetric, no top, one entity class).
function F.kind_from(obj)
  local k = istable(obj) and obj.kind
  if not istable(k) then return nil, "no kind" end
  local id = F.noun(k.kind)
  if id == "" then return nil, "no name" end
  local v = istable(k.variants) and k.variants[1]
  if not (istable(v) and v.entity == "holodeck_found" and isstring(v.model) and v.model ~= "") then return nil, "no placeable variant" end
  local w, d, h = tonumber(v.w), tonumber(v.d), tonumber(v.h)
  if not (finite(w) and finite(d) and finite(h)) then return nil, "no size" end
  if math.max(w, d, h) > F.MAX_HU then return nil, "too big" end
  if math.min(w, d, h) < F.MIN_HU then return nil, "too small" end
  local mass = tonumber(v.mass)
  if not finite(mass) or mass <= 0 then mass = nil else mass = math.Clamp(mass, 0.1, 50000) end
  local f = istable(v.found) and v.found or {}
  local names, seen = { id }, { [id] = true }
  for _, n in ipairs(istable(k.names) and k.names or {}) do
    n = F.noun(n)
    if n ~= "" and not seen[n] and #names < 8 then seen[n] = true; names[#names + 1] = n end
  end
  return {
    kind = id, names = names, approx = {}, category = "found", seat = false, surface = false, wall = "free", found = true,
    variants = { {
      model = clean_text(v.model, 120), front = 0, front_source = "unknown", symmetric = true, top = nil,
      w = w, d = d, h = h, mins = { -d / 2, -w / 2, 0 }, maxs = { d / 2, w / 2, h }, cx = 0, cy = 0,
      mass = mass, mass_status = isstring(v.mass_status) and clean_text(v.mass_status, 20) or "guessed", material = "default", skins = 1,
      entity = "holodeck_found",
      found = { card = clean_text(f.card, 120), name = clean_text(f.name or id, 60), source = clean_text(f.source, 40),
                licence = clean_text(f.licence, 24), basis = clean_text(f.basis, 16),
                mesh = (isstring(f.mesh) and #f.mesh == 16 and f.mesh:match("^%x+$")) and f.mesh or nil },     -- the key of the real shape, if the card has one
    } },
  }
end

local function standing_kinds()
  local used = {}
  for _, e in pairs(H.registry) do used[e.kind] = true end
  return used
end

--- Put a validated kind into the live library. Returns the kind id, or nil and why.
function F.add(kind)
  local id = kind.kind
  if H.base_catalog.kinds[id] then return nil, "the server's own library already has " .. id end
  -- a name the library already understands keeps meaning what it meant; only this kind's own new names are added
  local mine = {}
  for _, n in ipairs(kind.names) do
    local old = H.parser and H.parser.noun and H.parser.noun[n]
    if n == id and old and not (H.found_kinds[id]) then return nil, "the library already knows the word " .. id end
    if not old or old.kind == id then mine[#mine + 1] = n end
  end
  kind.names = mine
  if not H.found_kinds[id] then
    if #H.found_order >= F.MAX_KINDS then                                -- the shelf is full: the oldest kind nothing stands as makes way
      local used = standing_kinds()
      local victim
      for i, old in ipairs(H.found_order) do if not used[old] then victim = i; break end end
      if not victim then return nil, "too many different found things are standing" end
      H.found_kinds[table.remove(H.found_order, victim)] = nil
    end
    H.found_order[#H.found_order + 1] = id
  end
  H.found_kinds[id] = kind
  install_library(assemble(H.base_catalog))
  return id
end

H.AddFoundKind = F.add

--- May a noun be asked about now? (the bridge is set up, found objects are on)
function F.enabled() return cv_found:GetBool() and bridge_url() ~= "" end

--- The distinct nouns worth asking the bridge about, in order (adds only; at most F.MAX_NOUNS): { noun =, strict = }.
--- STRICT: the parser knows nothing of the word (an UNKNOWN item), so whatever the resolver answers is the reply ("I don't have a
--- dragon"). Not strict: the parser guessed (an approximate name: "fire" is a wood stove, "ottoman" a stool, with "hydrant" left over);
--- the resolver is asked in case it has the thing itself, and if it has not, the parser's own stated guess stands, unchanged.
function F.unknown_nouns(reqs)
  local out, seen = {}, {}
  local function want(noun, strict)
    noun = F.noun(noun)
    if noun ~= "" and not seen[noun] and #out < F.MAX_NOUNS then seen[noun] = true; out[#out + 1] = { noun = noun, strict = strict } end
  end
  for _, r in ipairs(reqs) do
    if r.action == "add" then
      local text = " " .. tostring(r.normalised or "") .. " "
      for _, it in ipairs(r.items or {}) do
        if tostring(it.kind):match("^UNKNOWN") then
          want(it.phrase or tostring(it.kind):sub(9), true)
        elseif it.approx and isstring(it.word) then
          local noun = it.word
          for _, left in ipairs(r.leftover or {}) do                   -- "fire" + the leftover "hydrant" right after it
            if isstring(left) and text:find(" " .. it.word .. " " .. left .. " ", 1, true) then noun = it.word .. " " .. left; break end
          end
          want(noun, false)
        end
      end
    end
  end
  return out
end

--- Ask about one noun. cb(result): { kind = id } when a kind was added, { say = } for any other answer, {} when nothing could be asked.
function F.ask(ply, noun, cb)
  local old = F.refused[noun]
  if old and CurTime() - old.t < F.REFUSED_FOR then return cb({ say = old.say, class = old.class, cached = true }) end
  if not bridge_allow("resolve", ply) then return cb({}) end
  local done = false
  local function once(r) if not done then done = true; cb(r) end end
  local sent = bridge_post("/resolve", { noun = noun, audience = "game", who = player_key(ply) }, function(body)
    local obj = util.JSONToTable(body or "")
    if not (istable(obj) and obj.ok) then return once({}) end
    local say = isstring(obj.say) and clean_text(obj.say, 240) or ""
    if obj.place == true then
      local kind, why = F.kind_from(obj)
      if not kind then
        say = (why == "too big" and "That is bigger than anything I will put down here.") or (why == "too small" and "That is too small for me to place.")
              or "I found it, but the numbers I have for it do not add up, so I will not place it."
        F.refused[noun] = { say = say, t = CurTime(), class = "refusal" }
        return once({ say = say, class = "refusal" })
      end
      local id, why2 = F.add(kind)
      if not id then return once({ say = "I found " .. noun .. " but cannot add it: " .. tostring(why2) .. ".", class = "failure" }) end
      return once({ kind = id, say = say })
    end
    if say == "" then return once({}) end
    -- failure (amber) | suggest (cyan, "the closest I have") | refusal (red). A card that matched by name but was not placed ("a model I may
    -- not share") keeps the matching status from the resolver, so "matched and not placed" is the refusal.
    local tag = obj.status
    if obj.place ~= true and (tag == "exact" or tag == "alias" or tag == "synonym") then tag = "not_shareable" end
    local class = WispFx.class_for(tag)
    F.refused[noun] = { say = say, t = CurTime(), class = class }
    once({ say = say, class = class })
  end, function(reason)
    H.last_resolve_error = reason
    once({})
  end, 28)
  if not sent then return cb({}) end
  timer.Simple(31, function() once({}) end)
end

--- Look up the nouns ({ noun =, strict = }) one after another (the bridge is rationed per player), then cb(results, added).
--- A result is only to be SAID when its noun was strict.
function F.resolve_all(ply, nouns, cb)
  local results, added, i = {}, 0, 0
  local function nextone()
    i = i + 1
    local want = nouns[i]
    if not want then return cb(results, added) end
    F.ask(ply, want.noun, function(r)
      r.strict = want.strict
      results[i] = r
      if r.kind then added = added + 1 end
      nextone()
    end)
  end
  nextone()
end

--- The parsed requests of one utterance, in order: one, or several for a compound sentence.
--- `prev` is the player's previous request, so "another one" and "again" know what to repeat.
--- How many requests of one sentence are carried out: the parser's own limit (one number, so the parser's "I only did the first N"
--- is true), or 8 when the parser does not say.
local function max_clauses() return tonumber(H.parser and H.parser.MAX_REQUESTS) or tonumber(Parse.MAX_REQUESTS) or 8 end

local function parse_requests(text, prev)
  if H.parser.parse_all then
    local ok, list = pcall(H.parser.parse_all, H.parser, text, prev)
    if ok and istable(list) and #list > 0 then return list end
  end
  return { H.parser:parse(text, prev) }
end

-- The parser's notes about a sentence that was cut short are told once, privately, before anything runs (see cut_notes).
local function is_cut_note(n) return isstring(n) and (n:match("^I only did the first") ~= nil or n:match("^I only read the first") ~= nil) end

--- What the parser wants the requester to know about their words (a repaired typo, a refusal): private, plain letters only.
local function tell_notes(ply, req)
  local lines = {}
  for _, n in ipairs(req.notes or {}) do
    if isstring(n) and #lines < 3 and not is_cut_note(n) then lines[#lines + 1] = clean_text(n, 160) end
  end
  if req.unknown_verb then lines[#lines + 1] = "I can add, remove, move and turn things, but I do not know how to do that." end
  if #lines > 0 then H.Tell(ply, table.concat(lines, " ")) end
end

--- Handle one request in plain words. ply may be nil (the console, a test): then opts = {pos=, yaw=, aim=, seed=, done=fn}.
--- Returns the planner's result table when it ran at once, or nil (the words were not a request, or the agent is thinking;
--- opts.done receives the result either way).
function H.Handle(ply, text, opts)
  opts = opts or {}
  local heard = text
  text = clean_text(heard, MAX_INPUT)                 -- what the agent, the persona and the log get
  opts.was_player = IsValidPlayer(ply)
  if H.Wisps.is_clear_command(heard) then                        -- "clear the wisps": only the floating notes, never the placed things
    local n = H.Wisps.clear(IsValidPlayer(ply) and ply or nil)
    local res = { action = "wisps", removed = n, fixed = true, placements = {},
                  say = n > 0 and ("Dismissed " .. n .. " wisp" .. (n == 1 and "" or "s") .. ".") or "There are no wisps to dismiss." }
    H.Tell(ply, res.say)
    if opts.done then opts.done(res) end
    return res
  end
  if opts.was_player and not manages_all(ply) then
    -- the rest between requests is checked first, so refusing a request costs nothing either
    local key, now = player_key(ply), CurTime()
    if not opts.found_done then                                   -- (the second pass of a request that waited for the bridge is the same request)
      if now < (H.next_ok[key] or 0) then return nil end
      H.next_ok[key] = now + cv_cooldown:GetFloat()
    end
  end
  if IsValidPlayer(ply) and cv_admin:GetBool() and not ply:IsAdmin() then
    H.Tell(ply, "The holodeck is for admins on this server.")
    return nil
  end
  local key = player_key(ply)
  -- The parser reads MAX_INPUT characters and knows by itself whether there was more (req.truncated and its note), but only if it
  -- is given at least one more: so it gets MAX_INPUT + 1.
  local full = clean_text(heard, MAX_INPUT + 1)
  local prev = H.prev_req[key]
  local reqs = parse_requests(full, prev)
  -- "another one" / "again" repeat the last thing ADDED, not merely the last request: after a turn, a move, a remove or an undo
  -- in between the parser has nothing to repeat, so it is asked again with the last add as the one before.
  local last_add = H.last_add[key]
  if last_add and last_add ~= prev then
    for _, r in ipairs(reqs) do
      if r.again and not actionable(r) then reqs = parse_requests(full, last_add); break end
    end
  end
  -- A noun the library lacks is looked up first, and the request then runs again with what was found (see "found objects").
  if not opts.found_done and F.enabled() then
    local nouns = F.unknown_nouns(reqs)
    if #nouns > 0 then
      if H.resolving[key] then
        H.Tell(ply, "I am still looking up your last request.")
        return nil
      end
      H.resolving[key] = true
      local returned, out = false, nil              -- answers from the cache arrive at once: then the result is this call's own
      F.resolve_all(ply, nouns, function(results, added)
        H.resolving[key] = nil
        if opts.was_player and not IsValidPlayer(ply) then return end
        local spoke, other = false, false
        for _, r in ipairs(results) do
          if r.say and (r.strict or r.kind) then H.Say(ply, r.say, nil, { speak = false }); spoke = true end
          if r.say and r.strict and not r.kind then                  -- a definite no: the sentence is also left hanging where the thing would have stood
            H.Wisps.drop(ply, r.say, r.class or "failure", H.Wisps.spot(ply))
          end
        end
        for _, r in ipairs(reqs) do if actionable(r) then other = true end end
        if added == 0 and spoke and not other then
          if opts.done then opts.done({ action = "none", say = H.last_say or "", placements = {} }) end
          return                                                   -- asked, answered, nothing to place: the answer was the whole reply
        end
        opts.found_done = true
        local r = H.Handle(ply, heard, opts)
        if not returned then out = r end
      end)
      returned = true
      return out
    end
  end
  local req = reqs[1]
  local cut = false
  for _, r in ipairs(reqs) do if r.truncated then cut = true end end
  if cut then
    -- never remove or move things on half a sentence: the part that was dropped may hold the "except the couch"
    for _, r in ipairs(reqs) do
      if r.action == "clear" or r.action == "remove" or r.action == "move" or r.action == "rotate" or r.action == "rearrange" then
        H.Tell(ply, "That was too long for me to read to the end, and I will not remove or move things on half a sentence. Say it again, shorter.")
        return nil
      end
    end
  end
  -- what the parser cut short is told to the requester, once, before anything runs
  local told_cap = false
  for _, r in ipairs(reqs) do
    for _, n in ipairs(r.notes or {}) do
      if isstring(n) and n:match("^I only did the first") and not told_cap then
        told_cap = true
        H.Tell(ply, clean_text(n, 140) .. ". Ask again for the rest.")
      end
    end
  end
  if cut then H.Tell(ply, "I only read the first " .. MAX_INPUT .. " characters of that, so the rest was left out.") end
  if #reqs == 1 and req.action ~= "none" then H.prev_req[key] = req end      -- (several: each is remembered as it is carried out)
  if req.action == "none" and #reqs == 1 then
    tell_notes(ply, req)
    return nil
  end
  if #reqs > 1 then
    -- One sentence is one request: the planning time and the number of new things are ONE budget shared by all its parts
    local trusted = manages_all(ply)
    local time_left = opts.budget or (trusted and 0.25 or 0.05)
    local things_left = opts.max_total or (trusted and 150 or 40)
    local says, notes, fixed, last = {}, {}, true, nil
    local cap = max_clauses()
    for i, r in ipairs(reqs) do
      if i > cap then
        says[#says + 1] = "I only do the first " .. cap .. " things you ask for in one go."
        break
      end
      local plans = r.action ~= "undo" and r.action ~= "clear"
      if r.action == "none" then
        tell_notes(ply, r)
      elseif plans and (time_left < 0.004 or (r.action == "add" and things_left <= 0)) then
        says[#says + 1] = "That is as much as I will plan in one go. Ask for the rest separately."
        break
      else
        H.prev_req[key] = r                                                -- only what really ran is remembered (not the clauses cut off)
        local part = setmetatable({ budget = time_left, max_total = things_left }, { __index = opts })
        local t0 = SysTime()
        local ok, res = pcall(run_one, ply, r, part)
        time_left = time_left - (SysTime() - t0)
        if not ok then
          ErrorNoHalt("[holodeck] request error: " .. tostring(res) .. "\n")
          res = { action = "error", say = "I tripped over that one. The error is in the server console.", fixed = true }
        end
        if res then
          if res.say and res.say ~= "" then says[#says + 1] = res.say end
          if res.ids then things_left = things_left - #res.ids end
          if res.clamped and res.action == "add" and not res.fixed then says[#says + 1] = "That is as many as I will build in one go." end
          for _, n in ipairs(res.notes or {}) do notes[#notes + 1] = n end
          fixed = fixed and res.fixed
          if res.placements and #res.placements > 0 then last = res end
          tell_unknown(ply, res)
        end
      end
    end
    local combined = { action = "many", say = table.concat(says, " "), notes = notes, fixed = fixed, placements = last and last.placements or {} }
    announce(ply, text, combined)
    if opts.done then opts.done(combined) end
    return combined
  end
  local mode = cv_agent:GetInt()
  -- the words build nothing, or two or more of them carry meaning the parser could not place ("somewhere to relax")
  local unexplained = req.action == "add" and #(req.leftover or {}) >= 2
  local use_agent = bridge_url() ~= "" and mode > 0 and ((mode == 2 and req.action == "add") or not actionable(req) or unexplained)
  if use_agent then
    if H.agent_busy[player_key(ply)] then
      H.Tell(ply, "I am still thinking about your last request.")
      return nil
    end
    local function fallback() H.Execute(ply, req, opts, text) end
    if ask_agent(ply, text, opts, fallback) then return nil end
  end
  return H.Execute(ply, req, opts, text)
end

-- ------------------------------------------------------------------------------------------------ chat and console
hook.Add("PlayerSay", "Holodeck.Chat", function(ply, text)
  if cv_wake:GetBool() then
    local ok, rest = Parse.strip_wake(text)
    if not ok then return end                                         -- ordinary talk between players: leave it alone
    timer.Simple(0, function() if IsValid(ply) then H.Handle(ply, rest) end end)
    return
  end
  timer.Simple(0, function() if IsValid(ply) then H.Handle(ply, text) end end)
end)

-- what a player said aloud, after their own microphone bridge heard the wake word (see skippy_client.lua)
local last_spoken = setmetatable({}, { __mode = "k" })
net.Receive("holodeck_voice", function(_, ply)
  local text = net.ReadString()
  if #text == 0 or #text > 300 then return end
  if last_spoken[ply] and CurTime() - last_spoken[ply] < 1.0 then return end
  last_spoken[ply] = CurTime()
  local woke, rest = Parse.strip_wake(text)
  H.Handle(ply, woke and rest or text)
end)

concommand.Add("holodeck", function(ply, _, args, line)
  local text = table.concat(args, " ")
  if text == "" then
    print("holodeck <what you want>   e.g. holodeck add three couches   (also: remove the lamp, undo, clear)")
    return
  end
  H.Handle(ply, text)
end, nil, "Ask the holodeck for something in plain words")

concommand.Add("holodeck_list", function(ply)
  local counts = {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    counts[e.kind] = (counts[e.kind] or 0) + 1
  end
  local parts = {}
  for k, v in pairs(counts) do parts[#parts + 1] = v .. " " .. k end
  table.sort(parts)
  print("[holodeck] " .. #H.order .. " things: " .. table.concat(parts, ", "))
end)

concommand.Add("holodeck_status", function(ply)
  local r = H.library_report
  print(string.format("[holodeck] %d of %d models available, %d kinds, %d rooms; content=%s; %d things standing; bridge=%s agent=%d persona=%s voice=%s",
    r.kept, r.total, #H.catalog.order, table.Count(H.catalog.recipes), cv_content:GetString(), #H.order,
    bridge_url() ~= "" and "set" or "none", cv_agent:GetInt(), tostring(cv_persona:GetBool()), tostring(cv_voice:GetBool())))
  print("[holodeck] found objects: " .. #H.found_order .. " kinds this session" .. (cv_found:GetBool() and "" or " (switched off)"))
  if H.Mesh then
    local ready, waiting, failed = H.Mesh.status()
    print(string.format("[holodeck] real shapes: %d ready, %d waiting, %d failed%s%s", ready, waiting, failed, H.Mesh.cv_fetch:GetBool() and "" or " (switched off)",
      H.Mesh.last_error and ("; last error: " .. tostring(H.Mesh.last_error)) or ""))
  end
  if #r.lost_models > 0 then print("[holodeck] missing here: " .. table.concat(r.lost_models, ", ", 1, math.min(#r.lost_models, 12)) .. (#r.lost_models > 12 and " ..." or "")) end
end, nil, "What the holodeck can show on this server")

-- re-read the models available (after changing holodeck_content, or after the addon's files were updated on disk)
concommand.Add("holodeck_reload", function(ply)
  if IsValid(ply) and not ply:IsSuperAdmin() then return end
  local ok, err = pcall(H.BuildLibrary)
  print(ok and "[holodeck] library rebuilt" or ("[holodeck] reload failed: " .. tostring(err)))
end, nil, "Rebuild the holodeck's library (superadmin)")

concommand.Add("holodeck_undo", function(ply) H.Handle(ply, "undo") end)
concommand.Add("holodeck_clear", function(ply) H.Handle(ply, "clear everything") end)

-- ------------------------------------------------------------------------------------------------ players coming and going
hook.Add("PlayerDisconnected", "Holodeck.Leave", function(ply)
  local key = ply:SteamID64()
  H.absent[key] = CurTime()
  last_spoken[ply] = nil
  H.agent_busy[key] = nil
  H.prev_req[key] = nil
  H.last_add[key] = nil
  H.player_buckets[key] = nil
  H.next_ok[key] = nil
  -- Props of theirs still waiting for their turn to appear are made NOW, while the player can still be looked up: made later they
  -- would have no creator and no prop-protection owner although the registry says they are theirs.
  local mine = {}
  for _, p in pairs(H.pending_spawns) do if p.owner == key then mine[#mine + 1] = p end end
  for _, p in ipairs(mine) do
    local ok, err = pcall(p.run)
    if not ok then ErrorNoHalt("[holodeck] spawn error: " .. tostring(err) .. "\n") end
  end
  -- their work stays for a while (they may be back); the undo history and bookkeeping do not
  timer.Simple(600, function()
    if H.absent[key] then H.undo_stacks[key] = nil; H.last_group[key] = nil end
  end)
end)

hook.Add("PlayerInitialSpawn", "Holodeck.Return", function(ply) H.absent[ply:SteamID64()] = nil end)

timer.Create("Holodeck.Orphans", 60, 0, function()
  local minutes = cv_orphans:GetFloat()
  if minutes <= 0 then return end
  local connected = {}
  for _, p in ipairs(player.GetAll()) do connected[p:SteamID64()] = true end
  local ids = {}
  for i = 1, #H.order do ids[i] = H.order[i] end
  for _, id in ipairs(ids) do
    local e = H.registry[id]
    if e and e.owner ~= "console" and not connected[e.owner] and H.absent[e.owner] and CurTime() - H.absent[e.owner] > minutes * 60 then
      remove_entry(e, true)
    end
  end
end)

--- Entities still being materialised (for tests that wait on the spawner).
function H.Busy() return H.pending > 0 or H.thinking > 0 or #audio_q > 0 end

MsgC(Color(0, 255, 136), "[mentat] holodeck ready: ", Color(255, 255, 255), #H.catalog.order .. " kinds. Say \"Skippy, add three couches\" or `holodeck add three couches`.\n")
