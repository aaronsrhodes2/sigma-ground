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

--- (Re)build the parser and planner over the models this server can actually show.
function H.BuildLibrary()
  local filtered, report = Library.filter(raw_catalog, util.IsValidModel, { hl2_only = cv_content:GetString() == "hl2" })
  H.catalog, H.library_report = filtered, report
  H.parser = Parse.new(filtered)
  H.planner = Plan.new(filtered)
  -- things already standing keep their model; find it again in the (possibly different) variant list
  for _, e in pairs(H.registry or {}) do
    local k = filtered.kinds[e.kind]
    e.legacy = true
    if k then
      for vi, v in ipairs(k.variants) do
        if v.model == e.model then e.vi, e.variant, e.legacy = vi, v, nil; break end
      end
    end
  end
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
H.prev_req = H.prev_req or {}            -- player key -> their previous request (for "another one", "again")
local function IsValidPlayer(ply) return ply ~= nil and IsValid(ply) and ply:IsPlayer() end
H.BuildLibrary()
H.pending = 0              -- spawns still scheduled
H.log = H.log or {}
H.debug = true

util.AddNetworkString("holodeck_say")
util.AddNetworkString("holodeck_audio")
util.AddNetworkString("holodeck_voice")

local cv_max = CreateConVar("holodeck_max_ents", "400", FCVAR_ARCHIVE, "Most props the holodeck will keep standing at once")
local cv_admin = CreateConVar("holodeck_admin_only", "0", FCVAR_ARCHIVE, "1 = only admins may use the holodeck")
local cv_stagger = CreateConVar("holodeck_stagger", "0.05", FCVAR_ARCHIVE, "Seconds between one prop appearing and the next")
local cv_wake = CreateConVar("holodeck_need_wake", "1", FCVAR_ARCHIVE, "1 = chat and voice need the wake word (Skippy); the console never does")
local cv_bridge = CreateConVar("holodeck_bridge", "", FCVAR_ARCHIVE + FCVAR_PROTECTED, "Skippy bridge URL, e.g. http://127.0.0.1:8765 (empty = no agent and no voice)")
local cv_token = CreateConVar("holodeck_bridge_token", "", FCVAR_ARCHIVE + FCVAR_PROTECTED + FCVAR_DONTRECORD, "shared token for the Skippy bridge")
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

--- A refilling allowance: `cap` units at most, `rate` units back per second.
local function bucket(cap, rate) return { cap = cap, rate = rate, tokens = cap, t = SysTime() } end
local function take(b, n)
  local now = SysTime()
  b.tokens = math.min(b.cap, b.tokens + (now - b.t) * b.rate)
  b.t = now
  if b.tokens >= n then b.tokens = b.tokens - n; return true end
  return false
end

local global_calls, global_chars, planner_cpu = bucket(12, 0.4), bucket(1800, 30), bucket(0.4, 0.1)
H.player_buckets = H.player_buckets or {}
local function player_bucket(key, kind)
  local set = H.player_buckets[key]
  if not set then set = {}; H.player_buckets[key] = set end
  if not set[kind] then set[kind] = kind == "plan" and bucket(3, 1 / 4) or bucket(2, 1 / 8) end
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
    mine = player_bucket(player_key(ply), kind)
    if not take(mine, 1) then return false end
  end
  global_calls.cap, global_calls.rate = math.max(2, cv_calls:GetFloat() / 2), cv_calls:GetFloat() / 60
  if not take(global_calls, 1) then
    if mine then mine.tokens = mine.tokens + 1 end
    return false
  end
  if kind == "speak" then
    global_chars.cap, global_chars.rate = cv_chars:GetFloat(), cv_chars:GetFloat() / 60
    if not take(global_chars, chars or 0) then return false end
  end
  return true
end

--- POST json to the bridge. ok(body, headers) on 200, fail(reason) otherwise. Returns false if nothing was sent.
H.thinking = 0             -- bridge calls in flight
local function bridge_post(path, tbl, ok, fail, timeout)
  if bridge_url() == "" then return false end
  H.thinking = H.thinking + 1
  local finished = false
  local function settle() if not finished then finished = true; H.thinking = math.max(0, H.thinking - 1) end end
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
    net.Send(targets)
  end
  hook.Run("HolodeckSaid", ply, text)
  if opts.speak ~= false then H.Speak(text, pos or (IsValidPlayer(ply) and ply:GetPos()) or nil, ply) end
end

-- ------------------------------------------------------------------------------------------------ the world, as the planner asks about it
local MASK = MASK_SOLID_BRUSHONLY

--- What blocks a footprint besides the world: players (alive, walking), NPCs, ragdolls and other people's props. The holodeck's
--- own props are the registry's business, so they never block here (a chair may sit under its table).
local function blocks(ent)
  if ent.HolodeckId then return false end
  if ent:IsPlayer() then return ent:Alive() and ent:GetMoveType() ~= MOVETYPE_NOCLIP end
  return true
end

local function make_env(origin_z)
  local env = {}
  function env.floor_z(x, y)
    local tr = util.TraceLine({ start = Vector(x, y, origin_z + 48), endpos = Vector(x, y, origin_z - 700), mask = MASK })
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
    local tr = util.TraceLine({ start = Vector(x, y, z + 3), endpos = Vector(x, y, z + 600), mask = MASK })
    if not tr.Hit or tr.StartSolid or tr.HitSky then return nil end
    return tr.HitPos.z
  end
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

local function registry_list(owner)
  local out = {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e and (not owner or e.owner == owner) then
      out[#out + 1] = { id = e.id, kind = e.kind, vi = e.vi, x = e.x, y = e.y, z = e.z, theta = e.theta, w = e.w, d = e.d, h = e.h,
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

local function spawn_prop(entry, fade, owner_ply)
  local ent = ents.Create("prop_physics")
  if not IsValid(ent) then return nil end
  ent:SetModel(entry.model)
  ent:SetPos(Vector(entry.origin.x, entry.origin.y, entry.origin.z))
  ent:SetAngles(Angle(0, entry.yaw, 0))
  ent:Spawn()
  ent:Activate()
  if not IsValid(ent) then return nil end
  ent.HolodeckId = entry.id
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
  local tries = 0
  local function release()
    if not IsValid(ent) then return end
    if tries < 20 and player_inside(ent) then tries = tries + 1; timer.Simple(0.5, release); return end   -- wait for them to step out
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
  local entry = {
    id = id, kind = pl.kind, vi = pl.vi, model = pl.model, x = pl.x, y = pl.y, z = pl.z, theta = pl.theta, w = pl.w, d = pl.d, h = pl.h,
    origin = pl.origin, yaw = pl.yaw, color = pl.color, color_name = pl.color_name, group = group, seq = pl.seq or H.seq,
    owner = pl.owner or player_key(ply), variant = H.catalog.kinds[pl.kind] and H.catalog.kinds[pl.kind].variants[pl.vi],
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

local function remove_entry(entry, fade)
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
           color_name = entry.color_name, group = entry.group }
end

-- ------------------------------------------------------------------------------------------------ how many may stand
--- Take down what is not in use to make room for `need` more: first the props of players who left, oldest first; then the
--- oldest props of whoever holds more than a fair share of the server's allowance. The requester's own things are never taken.
local function make_room(ply, need)
  if cv_max:GetInt() - #H.order >= need then return end
  local mine = player_key(ply)
  local connected, players = {}, player.GetAll()
  for _, p in ipairs(players) do connected[p:SteamID64()] = true end
  local freed = {}
  local function take_down(e)
    freed[e.owner] = (freed[e.owner] or 0) + 1
    remove_entry(e, true)
  end
  for _, id in ipairs({ unpack(H.order, 1, math.min(#H.order, 5000)) }) do
    if cv_max:GetInt() - #H.order >= need then break end
    local e = H.registry[id]
    if e and e.owner ~= "console" and e.owner ~= mine and not connected[e.owner] then take_down(e) end
  end
  local fair = math.max(1, math.floor(cv_max:GetInt() / math.max(1, #players)))
  while cv_max:GetInt() - #H.order < need do
    local counts, biggest, most = {}, nil, fair
    for _, id in ipairs(H.order) do
      local e = H.registry[id]
      if e and e.owner ~= "console" and e.owner ~= mine then
        counts[e.owner] = (counts[e.owner] or 0) + 1
        if counts[e.owner] > most then biggest, most = e.owner, counts[e.owner] end
      end
    end
    if not biggest then break end
    for _, id in ipairs(H.order) do
      local e = H.registry[id]
      if e and e.owner == biggest then take_down(e); break end
    end
  end
  for owner, n in pairs(freed) do
    for _, p in ipairs(players) do
      if p:SteamID64() == owner then H.Tell(p, "I took down " .. n .. " of your things to make room for someone else.") end
    end
  end
end

--- How many more things this player may add right now, and which limit is the tight one ("player" or "server").
local function room_for(ply, want)
  if want then make_room(ply, want) end
  local room, which = cv_max:GetInt() - entity_count(), "server"
  if IsValidPlayer(ply) and not manages_all(ply) then
    local left = cv_per_player:GetInt() - owner_count(ply:SteamID64())
    if left < room then room, which = left, "player" end
  end
  return math.max(room, 0), which
end

--- The server's own rules about spawning props (jails, spawn bans, prop protection blocklists) apply to the holodeck too.
local function vetoed(ply, model)
  if not IsValidPlayer(ply) then return false end
  local ok1, a = pcall(hook.Call, "PlayerSpawnObject", nil, ply, model, 0)
  local ok2, b = pcall(hook.Call, "PlayerSpawnProp", nil, ply, model)
  return (ok1 and a == false) or (ok2 and b == false)
end

--- Put things on the floor, one after another. The one place props are made, so every caller meets the same limits.
--- Returns the registered entries, and how many placements the limits or the server's rules turned away.
local function build(ply, placements)
  local room = room_for(ply, #placements)
  local verdicts, kept, turned_away = {}, {}, 0
  for _, pl in ipairs(placements) do
    if #kept >= room then turned_away = turned_away + 1
    else
      if verdicts[pl.model] == nil then verdicts[pl.model] = vetoed(ply, pl.model) end
      if verdicts[pl.model] then turned_away = turned_away + 1 else kept[#kept + 1] = pl end
    end
  end
  H.group = H.group + 1
  local entries = {}
  local delay = cv_stagger:GetFloat()
  for i, pl in ipairs(kept) do
    local entry = register(pl, H.group, ply)
    entries[i] = entry
    H.pending = H.pending + 1
    timer.Simple((i - 1) * delay, function()
      H.pending = H.pending - 1
      if H.registry[entry.id] == entry then
        local ent = spawn_prop(entry, true, ply)
        if ent then entry.ent = ent else unregister(entry.id) end
      end
    end)
  end
  return entries, turned_away
end

local function apply_moves(ply, moves)
  local old = {}
  for _, mv in ipairs(moves) do
    local e = H.registry[mv.id]
    -- what stands on something that moves rides with it, whoever made it; everything else must be the player's own
    if e and (mv.carried or can_touch(ply, e)) and e.variant then
      old[#old + 1] = { id = e.id, x = e.x, y = e.y, z = e.z, theta = e.theta, w = e.w, d = e.d, h = e.h, tilted = e.tilted }
      local v = e.variant
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
  return old
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

--- May the planner run now? It runs on the server's main thread, so the server as a whole gets a share of the time.
local function planner_admit(ply)
  if manages_all(ply) then return true end
  return take(planner_cpu, 0.02)
end

local function charge_planner(ply, seconds)
  planner_cpu.tokens = planner_cpu.tokens - seconds
  if IsValidPlayer(ply) and not manages_all(ply) then
    local key = player_key(ply)
    H.next_ok[key] = math.max(H.next_ok[key] or 0, CurTime() + 10 * seconds)    -- an expensive request earns a longer rest
  end
end

--- Do what a request says. Returns the result table (its `say` is what to tell the player); announces nothing.
local function run_one(ply, req, opts)
  local key = player_key(ply)
  if req.action == "undo" then
    local st = H.undo_stacks[key]
    local op = st and table.remove(st)
    if not op then return { action = "undo", say = "There is nothing to undo.", fixed = true } end
    local say
    if op.type == "add" then
      local n = 0
      for _, id in ipairs(op.ids) do
        local e = H.registry[id]
        if e and can_touch(ply, e) then remove_entry(e, true); n = n + 1 end
      end
      say = n > 0 and ("Undone: " .. n .. " thing" .. (n == 1 and "" or "s") .. " gone.") or "Those are already gone."
    elseif op.type == "remove" then
      local placements = {}
      for _, snap in ipairs(op.snaps) do
        if not H.registry[snap.rid] and H.catalog.kinds[snap.kind] then placements[#placements + 1] = snap end
      end
      local entries, away = build(ply, placements)
      say = "Undone: " .. #entries .. " thing" .. (#entries == 1 and "" or "s") .. " back."
      if away > 0 then say = say .. " " .. away .. " did not fit under the limits." end
    elseif op.type == "move" then
      local mvs = {}
      for _, o in ipairs(op.old) do mvs[#mvs + 1] = { id = o.id, x = o.x, y = o.y, z = o.z, theta = o.theta } end
      local done = apply_moves(ply, mvs)
      say = #done > 0 and "Undone." or "Those are gone, so there is nothing to put back."
    end
    return { action = "undo", say = say or "Undone.", fixed = true }
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
    local room, which = room_for(ply, 1)
    if room <= 0 then return { action = "add", say = limit_text(ply, which), placements = {}, limit = true, fixed = true, quiet = true } end
  end

  if not planner_admit(ply) then
    return { action = req.action, say = "I am busy planning for someone else. Ask again in a moment.", placements = {}, fixed = true, quiet = true }
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
    local entries, away = build(ply, res.placements)
    if #entries < want then
      local why = away > 0 and ((#entries == 0) and "" or " The rest would break a limit or a rule on this server.") or ""
      if #entries == 0 then
        local room, which = room_for(ply)
        res.say = (room <= 0 and limit_text(ply, which)) or "None of that could be placed under this server's rules."
        res.fixed, res.quiet = true, true
      else
        res.say = "I had room for only " .. #entries .. " of the " .. want .. "." .. why
      end
    end
    local ids = {}
    for i, e in ipairs(entries) do ids[i] = e.id end
    res.ids = ids
    if #ids > 0 then
      push_undo(ply, { type = "add", ids = ids })
      H.last_group[key] = ids
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

--- The parsed requests of one utterance, in order: one, or several for a compound sentence.
--- `prev` is the player's previous request, so "another one" and "again" know what to repeat.
local function parse_requests(text, prev)
  if H.parser.parse_all then
    local ok, list = pcall(H.parser.parse_all, H.parser, text, prev)
    if ok and istable(list) and #list > 0 then return list end
  end
  return { H.parser:parse(text, prev) }
end

--- What the parser wants the requester to know about their words (a repaired typo, a refusal): private, plain letters only.
local function tell_notes(ply, req)
  local lines = {}
  for _, n in ipairs(req.notes or {}) do
    if isstring(n) and #lines < 3 then lines[#lines + 1] = clean_text(n, 160) end
  end
  if req.unknown_verb then lines[#lines + 1] = "I can add, remove, move and turn things, but I do not know how to do that." end
  if #lines > 0 then H.Tell(ply, table.concat(lines, " ")) end
end

--- Handle one request in plain words. ply may be nil (the console, a test): then opts = {pos=, yaw=, aim=, seed=, done=fn}.
--- Returns the planner's result table when it ran at once, or nil (the words were not a request, or the agent is thinking;
--- opts.done receives the result either way).
function H.Handle(ply, text, opts)
  opts = opts or {}
  text = clean_text(text, MAX_INPUT)
  opts.was_player = IsValidPlayer(ply)
  if opts.was_player and not manages_all(ply) then
    -- the rest between requests is checked first, so refusing a request costs nothing either
    local key, now = player_key(ply), CurTime()
    if now < (H.next_ok[key] or 0) then return nil end
    H.next_ok[key] = now + cv_cooldown:GetFloat()
  end
  if IsValidPlayer(ply) and cv_admin:GetBool() and not ply:IsAdmin() then
    H.Tell(ply, "The holodeck is for admins on this server.")
    return nil
  end
  local key = player_key(ply)
  local reqs = parse_requests(text, H.prev_req[key])
  local req = reqs[1]
  for _, r in ipairs(reqs) do if r.action ~= "none" then H.prev_req[key] = r end end
  if req.action == "none" and #reqs == 1 then
    tell_notes(ply, req)
    return nil
  end
  if #reqs > 1 then
    local says, notes, fixed, last = {}, {}, true, nil
    for i, r in ipairs(reqs) do
      if i > 4 then break end
      if r.action == "none" then
        tell_notes(ply, r)
      else
        local ok, res = pcall(run_one, ply, r, opts)
        if not ok then
          ErrorNoHalt("[holodeck] request error: " .. tostring(res) .. "\n")
          res = { action = "error", say = "I tripped over that one. The error is in the server console.", fixed = true }
        end
        if res then
          if res.say and res.say ~= "" then says[#says + 1] = res.say end
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
  H.player_buckets[key] = nil
  H.next_ok[key] = nil
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
