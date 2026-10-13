-- avatar/director.lua -- Skippy alive (SERVER): the behaviour director. See stage/gmod/AVATAR.md ("Skippy alive") and holodeck/README.md.
--
-- What it does, in one paragraph: it watches what the owner builds and what happens in the world (hooks: HolodeckResult, HolodeckBeam, HolodeckResolve,
-- HolodeckWakeMiss, water splashes, the owner entering water), keeps a private notebook of it (journal.lua, notes.lua), decides with director_logic.lua
-- WHEN he may speak (chattiness, cooldowns, never over a human or himself, yielding to a command in progress, quiet hours, interruptible), asks the
-- bridge's POST /quip for a line (a bank line or a model's line written from the notes; at most N NEW spoken lines a day, never a repeat: skippy_quips.py),
-- shows it in chat, sends the clip from where he floats, and replicates a very small animation state on the bot (NetworkVars) that every client turns
-- into body language (anim.lua). Everything that makes him talk or move on his own can be switched off with mentat_skippy_director 0.
--
-- Convars:  mentat_skippy_director 1        0 = no unprompted behaviour at all (the avatar behaves exactly as before)
--           mentat_skippy_chat normal       the default chattiness: quiet | normal | chatty (each player's own "shush"/"be chatty" overrides it)
--           mentat_skippy_quiet_hours ""    e.g. "23-7": no unprompted remarks between those local hours
--           mentat_skippy_new_lines_per_day 20   NEW spoken lines a day (enforced by the bridge; cached replays are free)
--           mentat_skippy_journal 1         0 = nothing is recorded and no notes are made, for anyone
--           mentat_skippy_captain_sids ""   SteamID64s (comma separated) whose default roast level is savage; the host of a listen server and single player always are
--           mentat_skippy_debug 0           1 = print every decision (event, said or not, why)
-- Console:  mentat_skippy_status, skippy_director_log
if not SERVER then return end

Mentat = Mentat or {}
local Logic = include("mentat/avatar/director_logic.lua")
local AvLogic = include("mentat/avatar/logic.lua")        -- the avatar's own arithmetic (yaw_to)
local Anim = include("mentat/avatar/anim.lua")
local Notes = include("mentat/avatar/notes.lua")
local Journal = include("mentat/avatar/journal.lua")
local Persist = include("mentat/persist.lua")

local Dr = Mentat.Director or {}
Mentat.Director = Dr
Dr.Logic, Dr.Anim, Dr.Notes, Dr.Journal = Logic, Anim, Notes, Journal
Dr.st = Dr.st or Logic.new()
Dr.ring = Dr.ring or {}              -- the last decisions: { t, ev, say, why }
Dr.last_cmd = Dr.last_cmd or {}      -- steamid64 -> CurTime() of that player's last request
Dr.talk = Dr.talk or {}              -- player -> { on = bool, t = CurTime() of the last report }
Dr.deferred = Dr.deferred or {}      -- remarks waiting for a quiet moment: { ev =, t0 = }
Dr.session = Dr.session or {}        -- steamid64 -> true once this server run has logged their session start
Dr.cue_seq = Dr.cue_seq or 0
Dr.counts = Dr.counts or { events = 0, spoken = 0, silent = 0, text_only = 0, cached = 0, new = 0, errors = 0, interrupted = 0 }
Dr.Rand = Dr.Rand or math.random      -- the dice (tests replace it)
Dr.pile = Dr.pile or {}              -- steamid64 -> highest big-pile threshold announced

util.AddNetworkString("mentat_av_talk")
util.AddNetworkString("mentat_av_hush")

local CV = {
  on = CreateConVar("mentat_skippy_director", "1", FCVAR_ARCHIVE, "1 = Skippy remarks on what happens and moves expressively; 0 = only what he is asked"),
  chat = CreateConVar("mentat_skippy_chat", "normal", FCVAR_ARCHIVE, "default chattiness: quiet | normal | chatty"),
  quiet_hours = CreateConVar("mentat_skippy_quiet_hours", "", FCVAR_ARCHIVE, "local hours with no unprompted remarks, e.g. 23-7"),
  per_day = CreateConVar("mentat_skippy_new_lines_per_day", "20", FCVAR_ARCHIVE, "most NEW spoken lines per local day (cached replays are free)"),
  journal = CreateConVar("mentat_skippy_journal", "1", FCVAR_ARCHIVE, "1 = keep the notebook of what players do; 0 = record nothing"),
  captains = CreateConVar("mentat_skippy_captain_sids", "", FCVAR_ARCHIVE, "SteamID64s whose default roast level is savage"),
  debug = CreateConVar("mentat_skippy_debug", "0", 0, "1 = print the director's decisions"),
  quips = CreateConVar("mentat_skippy_quips", "1", FCVAR_ARCHIVE, "1 = ask the bridge for lines (/quip); 0 = body language only"),
}
Dr.CV = CV

local function is_player(p) return p ~= nil and IsValid(p) and p.IsPlayer and p:IsPlayer() end
local function sid_of(p) return is_player(p) and tostring(p:SteamID64()) or nil end
local function H() return Holodeck end
local function AV() return Mentat.Avatar end
local function clock() return CurTime() end

local function log(ev, say, why)
  local row = { t = clock(), ev = ev, say = say, why = why }
  Dr.ring[#Dr.ring + 1] = row
  if #Dr.ring > 80 then table.remove(Dr.ring, 1) end
  if CV.debug:GetBool() then print(string.format("[skippy] %s -> %s (%s)", tostring(ev), say and "SAYS" or "silent", tostring(why))) end
end
Dr.Log = log

-- ------------------------------------------------------------------------------------------------ private per-player settings
-- The file is a LIST of records { sid =, roast =, chat =, journal = }, not a table keyed by SteamID64: the engine's util.JSONToTable turns a numeric-looking KEY into a number
-- (and loses its low digits), which a string VALUE never suffers. Written through persist.lua: directly and read back (the engine silently refuses temp-file extensions), the last good copy in skippy_prefs.bak.txt.
Dr.PREFS_FILE, Dr.PREFS_MAX = "mentat/skippy_prefs.json", 4000
local prefs = {}
Dr.prefs = prefs

local ROAST_OK = { gentle = true, standard = true, savage = true }
local CHAT_OK = { quiet = true, normal = true, chatty = true }

local function parse_prefs(raw)
  local obj = util.JSONToTable(raw)
  local players = type(obj) == "table" and type(obj.players) == "table" and obj.players or nil
  if not players then return nil end
  local out, n = {}, 0
  for _, p in ipairs(players) do
    local sid = type(p) == "table" and p.sid
    if Persist.sid_ok(sid) and n < Dr.PREFS_MAX then
      local clean = {}
      if ROAST_OK[p.roast] then clean.roast = p.roast end
      if CHAT_OK[p.chat] then clean.chat = p.chat end
      if p.journal == false then clean.journal = false end
      if next(clean) then out[sid] = clean; n = n + 1 end
    end
  end
  return out
end

function Dr.LoadPrefs()
  for k in pairs(prefs) do prefs[k] = nil end
  local got = Persist.load(Dr.PREFS_FILE, parse_prefs)
  if got then for k, v in pairs(got) do prefs[k] = v end end
  return prefs
end

function Dr.SavePrefs()
  local list = {}
  for sid, p in pairs(prefs) do
    if #list < Dr.PREFS_MAX then list[#list + 1] = { sid = sid, roast = p.roast, chat = p.chat, journal = p.journal } end
  end
  table.sort(list, function(a, b) return a.sid < b.sid end)
  local payload = util.TableToJSON({ version = 2, players = list })
  if not payload then return false end
  return Persist.write(Dr.PREFS_FILE, payload, function(t) return parse_prefs(t) ~= nil end)
end

local function set_pref(ply, key, value)
  local sid = sid_of(ply)
  if not sid then return end
  local p = prefs[sid] or {}
  p[key] = value
  prefs[sid] = next(p) and p or nil
  Dr.SavePrefs()
end

-- The captains' file: data/mentat/captain_sids.txt, a plain .txt with one SteamID64 per line (blank lines and # comments allowed; never committed). Read through
-- persist.lua's rules (a string value, a size cap, the .bak.txt fallback); a missing or unreadable file means no captains from the file. Re-read every 30 s.
Dr.CAPTAIN_FILE = "mentat/captain_sids.txt"
local captain_cache = { t = -1e9, set = {} }

local function parse_captains(raw)
  local set, n = {}, 0
  for line in (tostring(raw) .. "\n"):gmatch("([^\r\n]*)[\r\n]") do
    line = line:gsub("#.*$", ""):gsub("^%s+", ""):gsub("%s+$", "")
    if line ~= "" and Persist.sid_ok(line) and n < 200 then set[line] = true; n = n + 1 end
  end
  return set
end
Dr.parse_captains = parse_captains

--- The SteamID64s listed in the captains' file (a set; empty when the file is missing).
function Dr.CaptainSids()
  local t = clock()
  if t - captain_cache.t < 30 then return captain_cache.set end
  captain_cache.t = t
  captain_cache.set = Persist.load(Dr.CAPTAIN_FILE, parse_captains) or {}
  return captain_cache.set
end

--- The roast level for this player: their own choice, else gentle (savage for the host of a listen server, single player, the ids in the convar
--- mentat_skippy_captain_sids and the ids in data/mentat/captain_sids.txt).
function Dr.DefaultRoast(ply)
  if not is_player(ply) then return "gentle" end
  if (ply.IsListenServerHost and ply:IsListenServerHost()) or (game.SinglePlayer and game.SinglePlayer()) then return "savage" end
  local me = sid_of(ply)
  for sid in CV.captains:GetString():gmatch("%d+") do if sid == me then return "savage" end end
  if me and Dr.CaptainSids()[me] then return "savage" end
  return "gentle"
end

function Dr.RoastLevel(ply)
  local p = prefs[sid_of(ply) or ""]
  return (p and p.roast) or Dr.DefaultRoast(ply)
end

function Dr.ChatLevel(ply)
  local p = prefs[sid_of(ply) or ""]
  if p and p.chat then return p.chat end
  local c = CV.chat:GetString()
  return CHAT_OK[c] and c or "normal"
end

function Dr.JournalOn(ply)
  if not CV.journal:GetBool() then return false end
  local p = prefs[sid_of(ply) or ""]
  return not (p and p.journal == false)
end

-- ------------------------------------------------------------------------------------------------ the notebook
--- Record something this player did (a no-op when journaling is off for them or for the server).
function Dr.Record(ply, ev)
  local sid = sid_of(ply)
  if not sid or not Dr.JournalOn(ply) then return false end
  return Journal.log(sid, ev, os.time())
end

local function banned_names(ply)
  local out = {}
  local h = H()
  for _, p in ipairs(player.GetAll()) do
    if p.Nick then out[#out + 1] = p:Nick() end
  end
  if h and h.Wake and h.Wake.names then
    for _, p in ipairs(player.GetAll()) do
      for _, n in ipairs(h.Wake.names(p)) do out[#out + 1] = n end
    end
  end
  return out
end

--- The notes (and the small table of facts) the model may see for this player, or empty ones when journaling is off.
function Dr.NotesFor(ply)
  local sid = sid_of(ply)
  if not sid or not Dr.JournalOn(ply) then return { notes = {}, facts = {} } end
  return Notes.summarise(Journal.read(sid, os.time()), os.time(), { banned = banned_names(ply) })
end

-- ------------------------------------------------------------------------------------------------ the body language (replicated state)
--- Play a gesture on every client: sets NetworkVars on the bot, which avatar_client.lua turns into motion (anim.lua).
function Dr.Cue(name, mood)
  local bot = AV() and AV().Get and AV().Get()
  if not bot or not Anim.CUES[name] then return false end
  Dr.cue_seq = (Dr.cue_seq % 30000) + 1
  bot:SetNWInt("mentat_av_cue", Anim.CUES[name])
  bot:SetNWInt("mentat_av_cue_seq", Dr.cue_seq)
  if mood and Anim.MOODS[mood] then bot:SetNWString("mentat_av_mood", mood) end
  Dr.last_cue = { name = name, t = clock() }
  return true
end

local function publish(bot)
  if not IsValid(bot) then return end
  bot:SetNWBool("mentat_av_dir", CV.on:GetBool())
  bot:SetNWBool("mentat_av_talking", clock() < Dr.st.speaking_until)
  local look = Dr.look
  if look and clock() - look.t < 9 then
    bot:SetNWVector("mentat_av_look", look.pos)
    bot:SetNWBool("mentat_av_look_on", true)
  else
    bot:SetNWBool("mentat_av_look_on", false)
  end
end

--- Make him look at a world point for a while (the thing just placed, or what the owner aims at).
function Dr.LookAt(pos, why)
  Dr.look = { pos = pos, t = clock(), why = why }
end

--- Called by the avatar's tick: the slow wander added to his hover point (world offsets from a point in his owner's frame).
function Dr.Drift(want, yaw, t)
  if not CV.on:GetBool() then return want end
  local f, l, z = Logic.drift(t)
  local rad = math.rad(yaw)
  local cx, sx = math.cos(rad), math.sin(rad)
  return { x = want.x + f * cx - l * sx, y = want.y + f * sx + l * cx, z = want.z + z }
end

--- The yaw he should turn to while looking at something, or nil.
function Dr.LookYaw(from)
  if not CV.on:GetBool() then return nil end
  local look = Dr.look
  if not look or clock() - look.t > 7 then return nil end
  return AvLogic.yaw_to(from, look.pos)
end

-- ------------------------------------------------------------------------------------------------ who is talking
local function human_speaking()
  local t = clock()
  for ply, s in pairs(Dr.talk) do
    if not is_player(ply) then Dr.talk[ply] = nil
    elseif s.on and t - s.t < 2.5 then return true end
  end
  for _, p in ipairs(player.GetAll()) do
    if not (p.IsBot and p:IsBot()) and p.IsSpeaking then
      local ok, speaking = pcall(p.IsSpeaking, p)
      if ok and speaking then return true end
    end
  end
  return false
end
Dr.HumanSpeaking = human_speaking

--- Cut his own line short (a human started talking over it).
function Dr.Hush()
  local t = clock()
  Dr.counts.interrupted = Dr.counts.interrupted + 1
  Logic.line_cut(Dr.st, t)
  local av = AV()
  if av then av.speaking_until = t end
  net.Start("mentat_av_hush")
  net.Broadcast()
  local bot = av and av.Get and av.Get()
  if bot then bot:SetNWBool("mentat_av_talking", false) end
  log("interrupt", false, "a human started talking over him")
end

net.Receive("mentat_av_talk", function(_, ply)
  if not is_player(ply) then return end
  local t = clock()
  local s = Dr.talk[ply] or { on = false, t = -1e9, last = -1e9 }
  if t - s.last < 0.1 then return end                                  -- a client cannot flood this
  local on = net.ReadBool()
  local started = on and not s.on
  s.on, s.t, s.last = on, t, t
  Dr.talk[ply] = s
  if started and Logic.should_interrupt(Dr.st, t, true) then Dr.Hush() end
end)

-- ------------------------------------------------------------------------------------------------ deciding and speaking
local function owner_of_avatar()
  local av = AV()
  return av and av.IsActive and av.IsActive() and av.owner or nil
end

--- What the logic needs to know right now about this owner.
function Dr.Context(owner)
  local q = Logic.parse_quiet_hours(CV.quiet_hours:GetString())
  local sid = sid_of(owner)
  local last = sid and Dr.last_cmd[sid]
  return {
    chat = Dr.ChatLevel(owner),
    disabled = not CV.on:GetBool(),
    quiet_hours = q and Logic.in_quiet_hours(q, tonumber(os.date("%H")) or 12) or false,
    human_speaking = human_speaking(),
    owner_command_age = last and (clock() - last) or 1e9,
    budget_left = Dr.budget and Dr.budget.left or nil,
  }
end

local function est_duration(text) return #tostring(text) * 0.07 + 1 end

local function parse_meta(headers)
  if type(headers) ~= "table" then return nil end
  for k, v in pairs(headers) do
    if type(k) == "string" and k:lower() == "x-skippy-quip" then return util.JSONToTable(v) end
  end
  return nil
end

local function is_mp3(b)
  if type(b) ~= "string" or #b < 200 then return false end
  if b:sub(1, 3) == "ID3" then return true end
  local b1, b2 = b:byte(1, 2)
  return b1 == 0xFF and b2 ~= nil and b2 >= 0xE0
end

local function clean_line(text)
  text = tostring(text or ""):gsub("[%c\127]", " "):gsub("%s+", " ")
  text = text:gsub("^%s+", ""):gsub("%s+$", "")
  return text:sub(1, 240)
end

--- Show and speak a line the bridge returned. meta is the decoded metadata; clip the mp3 bytes or nil.
function Dr.Present(owner, ev, meta, clip, via)
  local h, av = H(), AV()
  local text = clean_line(meta.text)
  if text == "" then return false end
  local bot = av and av.Get and av.Get()
  local pos = bot and bot:GetPos() or (is_player(owner) and owner:GetPos()) or nil
  Dr.self_say = true
  if h and h.Say then h.Say(owner, text, pos, { speak = false }) end              -- chat to those within earshot of where he floats
  Dr.self_say = false
  local t = clock()
  local dur = tonumber(meta.dur) or est_duration(text)
  dur = math.max(1, math.min(dur, 30))
  if clip and h and h.SendAudio and is_mp3(clip) then
    h.SendAudio(clip, pos, owner)
    Dr.counts.spoken = Dr.counts.spoken + 1
    if meta.cached then Dr.counts.cached = Dr.counts.cached + 1 end
    if meta.new then Dr.counts.new = Dr.counts.new + 1 end
  else
    Dr.counts.text_only = Dr.counts.text_only + 1
    dur = math.min(dur, 4)
  end
  Logic.line_started(Dr.st, t, dur)
  if av then av.speaking_until = t + dur + 0.5 end
  local mood = Anim.MOODS[meta.mood] and meta.mood or "musing"
  local cue = (Logic.BEHAVIOURS[ev.name] and Logic.BEHAVIOURS[ev.name].cue) or Logic.MOOD_CUE[mood]
  if mood == "opera" then cue = "conduct" end                                       -- an aria is always conducted
  Dr.Cue(cue, mood)
  Dr.last_line = { text = text, mood = mood, cue = cue, source = meta.source, t = t, dur = dur, new = meta.new, cached = meta.cached, ev = ev.name, via = via }
  if type(meta.budget) == "table" then Dr.budget = meta.budget end
  for _, tag in ipairs(type(meta.tags) == "table" and meta.tags or {}) do
    if type(tag) == "string" and tag ~= "callback" then Dr.Record(owner, { k = "joke", tag = tag }) end
  end
  hook.Run("SkippyDirectorSaid", owner, text, meta)
  return true
end

--- Ask the bridge for a line about `situation`. cb(ok) is called with true when a line was presented, false when there was none.
function Dr.Quip(owner, ev, situation, cb, force)
  local h = H()
  cb = cb or function() end
  if not CV.quips:GetBool() or not h or not h.BridgePost or not h.BridgeAllow then return cb(false, "quips are off or there is no bridge") end
  if Dr.inflight and clock() - Dr.inflight < 35 and not force then return cb(false, "a line is already being prepared") end
  local audio = GetConVar("holodeck_voice_out") and GetConVar("holodeck_voice_out"):GetBool() or false
  if not h.BridgeAllow(audio and "speak" or "reply", owner, 120) then return cb(false, "the bridge is rationed just now") end
  local nf = Dr.NotesFor(owner)
  local req = { situation = situation, roast = Dr.RoastLevel(owner), notes = nf.notes, facts = nf.facts, daily_limit = CV.per_day:GetInt(),
                audio = audio, who = sid_of(owner) }
  Dr.inflight = clock()
  Dr.last_request = { situation = situation, notes = #nf.notes, t = clock() }
  local sent = h.BridgePost("/quip", req, function(body, headers)
    Dr.inflight = nil
    local meta, clip
    if is_mp3(body) then meta, clip = parse_meta(headers), body else meta = util.JSONToTable(body or "") end
    if type(meta) ~= "table" or meta.ok ~= true then Dr.counts.errors = Dr.counts.errors + 1; return cb(false, "the bridge's answer was not a line") end
    if type(meta.budget) == "table" then Dr.budget = meta.budget end
    if meta.silent or type(meta.text) ~= "string" then Dr.counts.silent = Dr.counts.silent + 1; return cb(false, meta.why or "nothing fresh to say") end
    if not is_player(owner) then return cb(false, "the player left") end
    cb(Dr.Present(owner, ev, meta, clip, "quip"), "presented")
  end, function(reason)
    Dr.inflight = nil
    Dr.counts.errors = Dr.counts.errors + 1
    Dr.last_error = tostring(reason):sub(1, 160)
    cb(false, "the bridge said: " .. Dr.last_error)
  end, 30)
  if not sent then Dr.inflight = nil; return cb(false, "no bridge") end
  return true
end

--- The one entry for events: decide, then speak or defer. data = { ply = who it is about, pos = where, t0 }.
function Dr.Event(name, data)
  data = data or {}
  local owner = owner_of_avatar()
  Dr.counts.events = Dr.counts.events + 1
  if not owner then log(name, false, "he is away"); return false end
  if data.ply and data.ply ~= owner then log(name, false, "not about his owner"); return false end
  local ev = { name = name, t0 = data.t0 or clock(), data = data }
  local dec = Logic.decide(Dr.st, ev, clock(), Dr.Context(owner), Dr.Rand)
  if dec.defer and clock() - ev.t0 < Logic.DEFER_MAX_S then
    Dr.deferred[#Dr.deferred + 1] = { ev = ev, due = clock() + Logic.DEFER_S }
    log(name, false, dec.why .. " (waiting)")
    return false
  end
  if not dec.say then log(name, false, dec.why); return false end
  Logic.commit(Dr.st, name, clock())
  if dec.look and Dr.placed and clock() - Dr.placed.t < 12 then
    local av = AV()
    if av and av.Point then av.Point(Dr.placed.pos, 3) end
  end
  log(name, true, "situation " .. dec.situation)
  Dr.Quip(owner, ev, dec.situation, function(ok, why) if not ok then log(name, false, why) end end)
  return true
end

local function drain_deferred()
  if #Dr.deferred == 0 then return end
  local t = clock()
  local keep, due = {}, {}
  for _, d in ipairs(Dr.deferred) do
    if d.due <= t then due[#due + 1] = d else keep[#keep + 1] = d end
  end
  Dr.deferred = keep
  for _, d in ipairs(due) do Dr.Event(d.ev.name, { ply = d.ev.data.ply, t0 = d.ev.t0, pos = d.ev.data.pos }) end
end

-- ------------------------------------------------------------------------------------------------ arrival (the avatar hands over the canned answer)
--- Skippy was called. Plays the hat tip at once; the director then speaks a fresh arrival line instead of the canned "Here I am.".
--- Returns true when it took over (the canned answer is used only if the line does not come), false when the canned answer should be spoken now.
function Dr.Arrival(ply, canned)
  Dr.Cue("hat_tip", "boast")
  local h = H()
  if not CV.on:GetBool() or not CV.quips:GetBool() or not h or not h.BridgeAllow then return false end
  if GetConVar("holodeck_bridge"):GetString() == "" then return false end
  local ev = { name = "arrive", t0 = clock(), data = { ply = ply } }
  Logic.commit(Dr.st, "arrive", clock())
  Dr.Quip(ply, ev, "arrive", function(ok, why)
    log("arrive", ok, ok and "an arrival line" or why)
    if not ok and canned then canned() end
  end, true)
  return true
end

--- He was sent away ("go away"): a short sign-off line from the bank (the same route as the arrival line: it is counted against the day's NEW lines only if it
--- needs a new synthesis, and is text only when the budget is spent). He stays until it is said (at most a few seconds), then `finish()` sends him off.
--- Returns true when the director took over (canned() runs only if no line comes), false when the caller should say the canned line and dismiss at once.
function Dr.Signoff(ply, canned, finish)
  local h = H()
  if not CV.on:GetBool() or not CV.quips:GetBool() or not h or not h.BridgeAllow then return false end
  local b = GetConVar("holodeck_bridge")
  if not b or b:GetString() == "" then return false end
  Dr.Cue("hat_tip", "musing")
  local ev = { name = "dismiss", t0 = clock(), data = { ply = ply } }
  local spoken, done = false, false
  local function fin() if done then return end done = true; if finish then finish() end end
  Dr.Quip(ply, ev, "dismiss", function(ok, why)
    log("dismiss", ok, ok and "a sign-off line" or why)
    if ok then
      spoken = true
      local dur = Dr.last_line and tonumber(Dr.last_line.dur) or 2
      timer.Simple(math.max(1, math.min(dur, 5)) + 0.4, fin)
    else
      if canned and not done then canned() end
      fin()
    end
  end, true)
  timer.Simple(9, function() if not spoken and not done and canned then canned() end; fin() end)
  return true
end

-- ------------------------------------------------------------------------------------------------ his own commands
local CANNED = {
  chat_quiet = "Hush mode on. I'll keep my magnificent thoughts to myself, unless you ask.",
  chat_normal = "Normal chatter it is. A respectable amount of brilliance.",
  chat_chatty = "Oh, you asked for it. Prepare for a flood of commentary.",
  roast_gentle = "Gentle it is. I'll hide the claws, mostly.",
  roast_standard = "Standard roast level. Cheeky, with a dash of contempt.",
  roast_savage = "Savage mode. Brace yourself, monkey. I'll be affectionate underneath.",
  roast_max = "That's already as savage as I go. Any more and the hat catches fire.",
  roast_min = "That's already as gentle as I get. Any gentler and I'd be a houseplant.",
  forget_that = "Forgotten. Wiped it from my vast memory. Not that it fit.",
  forget_today = "Today is gone from my notes. You never did any of it.",
  forget_all = "Every note is deleted. I know nothing about you. Not that I knew much.",
  journal_off = "No more notes about you, and the old ones are deleted. I'll just be magnificent in general.",
  journal_on = "Notes back on. I'll remember what you do from now on, privately.",
  journal_is_off = "I'm not keeping notes on you. Say start remembering, and I will.",
  nothing = "I have nothing in my notes yet. Build something and I'll make fun of it later.",
}
Dr.CANNED = CANNED

local function reply(ply, text)
  local h = H()
  if h and h.Say then h.Say(ply, text, nil, { speak = false }) end          -- text only: a settings confirmation is not worth a synthesis
end

--- Handle one of his own commands. Returns true when `text` was one (it is then not passed on to the holodeck).
function Dr.TryCommand(ply, text)
  if not is_player(ply) then return false end
  local sid = sid_of(ply)
  if sid then Dr.last_cmd[sid] = clock() end
  local cmd = Logic.match_command(text)
  if not cmd then return false end
  if cmd.cmd == "chat" then
    set_pref(ply, "chat", cmd.level)
    reply(ply, CANNED["chat_" .. cmd.level])
  elseif cmd.cmd == "roast" then
    local cur = Dr.RoastLevel(ply)
    local new = cmd.level or (cmd.delta and Logic.roast_step(cur, cmd.delta)) or cur
    if cmd.delta and new == cur then
      reply(ply, CANNED[cmd.delta > 0 and "roast_max" or "roast_min"])
    elseif new ~= cur then
      set_pref(ply, "roast", new)
      Dr.Record(ply, { k = "roast", level = new })
      reply(ply, CANNED["roast_" .. new])
    end
    if cmd.now then Dr.Quip(ply, { name = "roast_now", t0 = clock(), data = { ply = ply } }, "roast", function() end, true) end
  elseif cmd.cmd == "recall" then
    Dr.Recall(ply)
  elseif cmd.cmd == "forget" then
    Journal.forget(sid, cmd.scope, os.time())
    reply(ply, CANNED["forget_" .. cmd.scope])
  elseif cmd.cmd == "journal" then
    if cmd.on then set_pref(ply, "journal", nil); reply(ply, CANNED.journal_on)
    else
      Journal.delete_all(sid)
      set_pref(ply, "journal", false)
      reply(ply, CANNED.journal_off)
    end
  end
  return true
end

--- "What do you remember": answered from the notes, in character (a model line when the bridge can write one, else a template callback).
function Dr.Recall(ply)
  local sid = sid_of(ply)
  if not Dr.JournalOn(ply) then reply(ply, CANNED.journal_is_off); return end
  local nf = Dr.NotesFor(ply)
  if #nf.notes == 0 then reply(ply, CANNED.nothing); return end
  local ev = { name = "recall", t0 = clock(), data = { ply = ply } }
  Dr.Quip(ply, ev, "recall", function(done, why)
    log("recall", done, done and "answered in character" or why)
    if not done then reply(ply, Notes.plain_recall(nf)) end
  end, true)
end

-- ------------------------------------------------------------------------------------------------ what happens in the world
local function entry_pos(id)
  local h = H()
  local e = h and h.registry and h.registry[id]
  if e and e.x then return Vector(e.x, e.y, (e.z or 0) + 20) end
  return nil
end

local function source_of(kind)
  local h = H()
  local k = h and h.found_kinds and h.found_kinds[kind]
  local f = k and k.variants and k.variants[1] and k.variants[1].found
  if not f then return nil end
  local s = tostring(f.source or f.card or ""):lower()
  if s:find("poly", 1, true) then return "Poly Haven" elseif s:find("fuel", 1, true) then return "Fuel" elseif s:find("nasa", 1, true) then return "NASA" end
  return "found"
end

local PILE_STEPS = { 30, 80, 160 }

--- HolodeckResult: a request ran. Journal it; look at / point at the thing just placed.
function Dr.OnResult(ply, req, res)
  if not is_player(ply) then return end
  local sid = sid_of(ply)
  if sid then Dr.last_cmd[sid] = clock() end
  local act = res.action
  if act == "add" and res.ids and #res.ids > 0 then
    local h = H()
    local counts, order = {}, {}
    for _, id in ipairs(res.ids) do
      local e = h and h.registry and h.registry[id]
      if e and e.kind then
        if not counts[e.kind] then counts[e.kind] = 0; order[#order + 1] = e.kind end
        counts[e.kind] = counts[e.kind] + 1
      end
    end
    for _, kind in ipairs(order) do
      local k = h.catalog and h.catalog.kinds and h.catalog.kinds[kind]
      local v = k and k.variants and k.variants[1]
      local w = v and v.water
      if w then                                                                  -- a pool, a lake: a water body, with its liquid
        Dr.Record(ply, { k = "add", name = tostring(w.base or "pool"), n = counts[kind], src = "water" })
        Dr.Record(ply, { k = "water", cmd = tostring(w.base or "pool"), liquid = w.liquid })
      else
        Dr.Record(ply, { k = "add", name = (tostring(kind):gsub("_", " ")), n = counts[kind], src = source_of(kind) or "curated" })
      end
    end
    local last = res.ids[#res.ids]
    local p = entry_pos(last)
    if p then
      Dr.placed = { pos = p, t = clock() }
      if owner_of_avatar() == ply then
        Dr.LookAt(p, "what you just placed")
        if not Dr.last_cue or clock() - Dr.last_cue.t > 2.5 then Dr.Cue("nod") end
      end
    end
    -- a big pile? (once per step, again only after the pile has shrunk well below it)
    local total = h and h.order and #h.order or 0
    local reached = 0
    for _, step in ipairs(PILE_STEPS) do if total >= step then reached = step end end
    if reached > (Dr.pile[sid] or 0) then
      Dr.pile[sid] = reached
      Dr.Event("big_pile", { ply = ply })
    end
    Dr.Event("build", { ply = ply })
  elseif act == "undo" then Dr.Record(ply, { k = "undo" })
  elseif act == "remove" then Dr.Record(ply, { k = "remove", n = res.removals and #res.removals or 1 })
  elseif act == "clear" then
    Dr.Record(ply, { k = "clear", n = res.removed or 0 })
    for i = #PILE_STEPS, 1, -1 do if (H() and H().order and #H().order or 0) < PILE_STEPS[i] * 0.7 then Dr.pile[sid] = (PILE_STEPS[i - 1] or 0) end end
  end
end

--- HolodeckBeam: a found object's shape arrived ("ready") or could not ("failed").
function Dr.OnBeam(what, m, why)
  local ply = m and m.asker
  if not is_player(ply) then return end
  local name = tostring(m.card or ""):match("([^%.]+)$") or ""
  name = name:gsub("[-_]", " ")
  if what == "ready" then
    Dr.Record(ply, { k = "beam_ok", name = name, src = (tostring(m.card):find("poly", 1, true) and "Poly Haven") or "found" })
    if owner_of_avatar() == ply then
      if not Dr.last_cue or clock() - Dr.last_cue.t > 2.5 then Dr.Cue("flourish") end         -- what he asked for landed: a flourish, even with nothing to say
      Dr.Event("beam_ok", { ply = ply })
    end
  else
    Dr.Record(ply, { k = "beam_fail", name = name, why = "mesh" })
    Dr.Event("beam_fail", { ply = ply })
  end
end

--- HolodeckResolve: the bridge's /resolve answered for a noun (plumbs the bridge's request results into the notebook).
function Dr.OnResolve(ply, noun, r)
  if not is_player(ply) or type(r) ~= "table" or r.kind then return end
  if r.class == "failure" or r.class == "refusal" or (r.say and not r.kind and r.strict) then
    Dr.Record(ply, { k = "beam_fail", name = noun, why = r.class or "none" })
    Dr.Event("beam_fail", { ply = ply })
  end
end

function Dr.OnWakeMiss(ply)
  Dr.Event("wake_miss", { ply = ply })
end

-- ------------------------------------------------------------------------------------------------ installing into the other modules
local function install_wrappers()
  local h = H()
  if h and h.Wake and h.Wake.try_command and not Dr.wake_wrapped then
    local orig = h.Wake.try_command
    h.Wake.try_command = function(ply, text, ...)
      local before = h.Wake.names and table.concat(h.Wake.names(ply), "|") or ""
      local r = orig(ply, text, ...)
      if r and h.Wake.names and table.concat(h.Wake.names(ply), "|") ~= before then
        Dr.Record(ply, { k = "name_change" })                                       -- THAT it changed, never to what
        Dr.Event("name_change", { ply = ply })
      end
      return r
    end
    Dr.wake_wrapped = true
  end
  local W = Mentat.Water
  if W and W.send_splash and not Dr.splash_wrapped then
    local orig = W.send_splash
    W.send_splash = function(body, kind, x, y, mass, speed)
      local r = orig(body, kind, x, y, mass, speed)
      if kind == 1 and (tonumber(mass) or 0) >= 5 then Dr.OnSplash(x, y, mass) end
      return r
    end
    Dr.splash_wrapped = true
  end
end
Dr.Install = install_wrappers

--- A splash somewhere: counts for the owner when it is near them.
function Dr.OnSplash(x, y, mass)
  local owner = owner_of_avatar()
  if not owner then return end
  local p = owner:GetPos()
  if (p.x - x) ^ 2 + (p.y - y) ^ 2 > 1500 * 1500 then return end
  Dr.splash_n = (Dr.splash_n or 0) + 1
  if clock() - (Dr.splash_t or -1e9) > 20 then
    Dr.Record(owner, { k = "splash", n = Dr.splash_n })
    Dr.splash_n, Dr.splash_t = 0, clock()
  end
  Dr.Event("splash", { ply = owner })
end

local function in_water(ply)
  local W = Mentat.Water
  local Vol = W and W.volumes
  if not (Vol and Vol.list and Vol.holding) then return false end
  local ok, r = pcall(function()
    local o = ply:GetPos()
    return Vol.holding(Vol.list(), o.x, o.y, o.z + 36)
  end)
  return ok and r ~= nil and r ~= false
end

-- ------------------------------------------------------------------------------------------------ the tick (twice a second)
function Dr.Tick()
  install_wrappers()
  local t = clock()
  local owner = owner_of_avatar()
  local av = AV()
  local bot = av and av.Get and av.Get()
  if bot then publish(bot) end
  -- stale talk reports expire
  for ply, s in pairs(Dr.talk) do if not is_player(ply) or t - s.t > 5 then Dr.talk[ply] = nil end end
  -- the server-side view of a human speaking also interrupts
  if Dr.st.speaking_until > t and human_speaking() and not Dr.hushed_for_human then
    Dr.hushed_for_human = true
    Dr.Hush()
  elseif not human_speaking() then
    Dr.hushed_for_human = false
  end
  drain_deferred()
  -- gravity changes (the world module answers outside the holodeck's request path): attribute to whoever asked most recently
  local Wd = Mentat.World
  if Wd and Wd.state then
    local ok, st = pcall(Wd.state)
    local preset = ok and st and st.preset or nil
    if preset and Dr.preset and preset ~= Dr.preset then
      local who, when = nil, -1e9
      for _, p in ipairs(player.GetAll()) do
        local sid = sid_of(p)
        if sid and (Dr.last_cmd[sid] or -1e9) > when then who, when = p, Dr.last_cmd[sid] end
      end
      if who and t - when < 6 then Dr.Record(who, { k = "gravity", preset = tostring(preset) }) end
    end
    Dr.preset = preset or Dr.preset
  end
  for _, p in ipairs(player.GetAll()) do                                              -- session starts
    local sid = sid_of(p)
    if sid and not (p.IsBot and p:IsBot()) and not Dr.session[sid] then
      Dr.session[sid] = true
      Dr.Record(p, { k = "session_start" })
    end
  end
  if not owner then return end
  local sid = sid_of(owner)
  -- the thing under the crosshair: where he looks when nothing was just placed
  if (not Dr.look or t - Dr.look.t > 6) and owner.GetShootPos and owner.GetAimVector then
    local from = owner:GetShootPos()
    local tr = util.TraceLine({ start = from, endpos = from + owner:GetAimVector() * 2500, filter = { owner, bot } })
    if tr and tr.Hit and IsValid(tr.Entity) and not (tr.Entity.IsPlayer and tr.Entity:IsPlayer()) and tr.Entity.GetClass and tr.Entity:GetClass() ~= "worldspawn" then
      Dr.LookAt(tr.HitPos, "what you are aiming at")
      Dr.look.t = t - 3                                                              -- a gaze that fades sooner than a placed thing
    end
  end
  -- the owner idle for a while
  if Logic.owner_idle(Dr.st, t, owner:GetPos(), sid and Dr.last_cmd[sid] or nil, Dr.ChatLevel(owner)) then Dr.Event("owner_idle", { ply = owner }) end
  -- a jump into water
  local wet = in_water(owner)
  if wet and not Dr.was_wet and (Dr.prev_vz or 0) < -100 then
    Dr.Record(owner, { k = "fall_water" })
    Dr.Event("water_jump", { ply = owner })
  end
  Dr.was_wet = wet
  local v = owner.GetVelocity and owner:GetVelocity()
  Dr.prev_vz = v and v.z or 0
  -- an idle whim
  if Dr.st.next_whim == 0 then Dr.st.next_whim = Logic.next_whim(t, Dr.ChatLevel(owner), Dr.Rand()) end
  if t >= Dr.st.next_whim then
    Dr.st.next_whim = Logic.next_whim(t, Dr.ChatLevel(owner), Dr.Rand())
    Dr.Event(Logic.pick_whim(Dr.Rand()), { ply = owner })
  end
end

-- ------------------------------------------------------------------------------------------------ hooks
local function add_hooks()
  hook.Add("HolodeckResult", "Mentat.Director", function(ply, req, res) Dr.OnResult(ply, req, res) end)
  hook.Add("HolodeckBeam", "Mentat.Director", function(what, m, why) Dr.OnBeam(what, m, why) end)
  hook.Add("HolodeckResolve", "Mentat.Director", function(ply, noun, r) Dr.OnResolve(ply, noun, r) end)
  hook.Add("HolodeckWakeMiss", "Mentat.Director", function(ply) Dr.OnWakeMiss(ply) end)
  hook.Add("HolodeckSaid", "Mentat.Director", function(ply, text)
    if Dr.self_say then return end
    Logic.line_started(Dr.st, clock(), math.min(est_duration(text), 12))             -- the holodeck's own answers are lines too: he does not talk over them
  end)
  hook.Add("PlayerDisconnected", "Mentat.Director", function(ply)
    local sid = sid_of(ply)
    if sid and Dr.session[sid] then Dr.Record(ply, { k = "session_end" }); Dr.session[sid] = nil end
    Dr.talk[ply] = nil
  end)
end
add_hooks()
timer.Create("Mentat.DirectorTick", 0.5, 0, function()
  local ok, err = pcall(Dr.Tick)
  if not ok and not Dr.tick_error then Dr.tick_error = tostring(err); print("[skippy] director tick error: " .. Dr.tick_error) end
end)
Dr.LoadPrefs()

concommand.Add("mentat_skippy_status", function()
  local c = Dr.counts
  print(string.format("[skippy] director %s; events %d, spoke %d (%d new, %d cached replays), text only %d, silent %d, errors %d, interrupted %d",
    CV.on:GetBool() and "on" or "off", c.events, c.spoken, c.new, c.cached, c.text_only, c.silent, c.errors, c.interrupted))
  local b = Dr.budget
  print(b and string.format("[skippy] today (%s): %d of %d new lines used, %d left", tostring(b.day), b.used or 0, b.limit or 0, b.left or 0) or "[skippy] budget: not known yet (no line asked for)")
end, nil, "Skippy director: what he decided and how many new lines are left today")
concommand.Add("skippy_director_log", function()
  for _, r in ipairs(Dr.ring) do print(string.format("[skippy] %7.1f %-12s %s  %s", r.t, tostring(r.ev), r.say and "SAYS  " or "silent", tostring(r.why))) end
end, nil, "The director's last decisions")

return Dr
