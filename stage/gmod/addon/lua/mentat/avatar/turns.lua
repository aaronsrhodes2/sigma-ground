-- avatar/turns.lua -- Skippy's solid turns (A7b) on the SERVER: decide whether a line was for him, route it, and answer through the bridge's POST /turn.
-- The rules (weights, window, routes) are the pure holodeck/turnlogic.lua; this file gathers the signals from the world and acts on the verdict.
--
--   * Every path that ends in Holodeck.Handle passes through Turns.Intercept (avatar.lua puts it in front of the old flow): a chat line that began with
--     one of the speaker's wake names, a spoken line, push-to-talk, the console. A chat line WITHOUT his name arrives by the hook HolodeckOverheard
--     (holodeck.lua's chat hook): it is scored too, and answered only when the verdict is yes (a follow-up inside the 20 s window while the speaker looks at or
--     stands by him; never chatter between humans, never a line addressed to another player by name).
--   * Routes: parser (the old flow, untouched), plan (the old /plan agent), talk (POST /turn: a validated {say, mood, action?} from the model, a cache, or a
--     fallback line: never silence), ignore.
--   * Nothing here prints what a player said unless mentat_skippy_debug is 1; the decision log keeps only lengths, scores and reasons.
--
-- Convars: mentat_skippy_turns 1 (0 = the old flow exactly), mentat_skippy_tiebreak 0 (1 = ask the small model about a "maybe"), mentat_skippy_tiebreak_per_day 60,
--          mentat_skippy_frontier_per_day 300, mentat_skippy_turn_timeout 8.   Console: skippy_turn_log, skippy_turns_status.
if not SERVER then return end

Mentat = Mentat or {}
local Logic = include("mentat/holodeck/turnlogic.lua")
local AvLogic = include("mentat/avatar/logic.lua")
local DirLogic = include("mentat/avatar/director_logic.lua")
local WaterIntent, WaterC, WorldIntent
do                                                    -- the water and physics words are claimed before the object parser (water.lua, world.lua): so are they here
  local ok1, wi = pcall(include, "mentat/water/intent.lua")
  local ok2, wc = pcall(include, "mentat/water/constants.lua")
  local ok3, wo = pcall(include, "mentat/world/intent.lua")
  if ok1 and ok2 then WaterIntent, WaterC = wi, wc end
  if ok3 then WorldIntent = wo end
end

local Tn = Mentat.Turns or {}
Mentat.Turns = Tn
Tn.Logic = Logic
Tn.spoke = Tn.spoke or {}          -- steamid64 -> CurTime() he last spoke to that player
Tn.lines = Tn.lines or {}          -- recent human lines: { sid =, t =, x =, y =, z = }
Tn.ring = Tn.ring or {}            -- decisions: { t, src, n (length), addressed, score, final, why }
Tn.busy = Tn.busy or {}            -- steamid64 -> CurTime() a /turn is in flight
Tn.tb_last = Tn.tb_last or {}      -- steamid64 -> CurTime() of that player's last tiebreak
Tn.counts = Tn.counts or { decided = 0, answered = 0, ignored = 0, maybe = 0, talk = 0, fallback = 0, cache = 0, frontier = 0, local_ = 0, rejected = 0, tiebreaks = 0 }

local CV = {
  on = CreateConVar("mentat_skippy_turns", "1", FCVAR_ARCHIVE, "1 = Skippy decides whether a line was for him and answers chat through the model route; 0 = the old flow"),
  tiebreak = CreateConVar("mentat_skippy_tiebreak", "0", FCVAR_ARCHIVE, "1 = ask the small model about a line that is only maybe for him (budget-limited)"),
  tiebreak_cap = CreateConVar("mentat_skippy_tiebreak_per_day", "60", FCVAR_ARCHIVE, "most tiebreak calls a day"),
  frontier_cap = CreateConVar("mentat_skippy_frontier_per_day", "300", FCVAR_ARCHIVE, "most frontier-model calls a day (the bridge enforces it)"),
  timeout = CreateConVar("mentat_skippy_turn_timeout", "8", FCVAR_ARCHIVE, "seconds a model call may take before the fallback line is used"),
  debug = GetConVar("mentat_skippy_debug") or CreateConVar("mentat_skippy_debug", "0", 0, "1 = print the director's and the turns' decisions"),
}
Tn.CV = CV

local function is_player(p) return p ~= nil and IsValid(p) and p.IsPlayer and p:IsPlayer() end
-- Skippy's own bot is not a person; any other player is one (a helper bot in a test is a player).
local function is_bot(p) local av = Mentat.Avatar; return av ~= nil and av.bot ~= nil and p == av.bot end
local function sid_of(p) return is_player(p) and tostring(p:SteamID64()) or nil end
local function H() return Holodeck end
local function AV() return Mentat.Avatar end
local function now() return CurTime() end

function Tn.Enabled() return CV.on:GetBool() end

--- Can the model route be used at all (a bridge, and the server owner has not forbidden the agent with holodeck_agent 0)?
function Tn.ModelOk()
  local h = H()
  if not (h and h.BridgePost and h.BridgeAllow) then return false end
  local b = GetConVar("holodeck_bridge")
  if not b or b:GetString() == "" then return false end
  local a = GetConVar("holodeck_agent")
  if a and a:GetInt() <= 0 then return false end
  return true
end

-- ------------------------------------------------------------------------------------------------ the decisions log
local function ring_add(e)
  Tn.ring[#Tn.ring + 1] = e
  if #Tn.ring > 80 then table.remove(Tn.ring, 1) end
end

local function dprint(...)
  if CV.debug:GetBool() then print("[turns] " .. string.format(...)) end
end

-- ------------------------------------------------------------------------------------------------ signals from the world
local function dist3(a, b) local dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z; return math.sqrt(dx * dx + dy * dy + dz * dz) end

local function forward(ply)
  local a = ply:EyeAngles()
  local p, y = math.rad(a.p or 0), math.rad(a.y or 0)
  return math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)
end

--- The angle in degrees between where `ply` looks and the point `to`.
local function angle_to(ply, to)
  local e = ply:EyePos()
  local dx, dy, dz = to.x - e.x, to.y - e.y, to.z - e.z
  local len = math.sqrt(dx * dx + dy * dy + dz * dz)
  if len < 1 then return 0 end
  local fx, fy, fz = forward(ply)
  local c = (fx * dx + fy * dy + fz * dz) / len
  return math.deg(math.acos(math.max(-1, math.min(1, c))))
end
Tn.angle_to = angle_to

local function humans_other_than(ply)
  local out = {}
  for _, p in ipairs(player.GetAll()) do
    if p ~= ply and is_player(p) and not is_bot(p) then out[#out + 1] = p end
  end
  return out
end

--- Everything the rules need to know about this speaker right now.
function Tn.Context(ply, src)
  local h = H()
  local ctx = { src = src, names = (h and h.Wake and h.Wake.names and h.Wake.names(ply)) or { "skippy" }, human_names = {}, others_near = 0 }
  local others = humans_other_than(ply)
  for _, p in ipairs(others) do ctx.human_names[#ctx.human_names + 1] = tostring(p:Nick()) end
  local t = now()
  local sid = sid_of(ply)
  if sid and Tn.spoke[sid] then ctx.since_spoke = t - Tn.spoke[sid] end
  local pos = ply:GetPos()
  local best
  for _, p in ipairs(others) do
    if dist3(p:GetPos(), pos) <= Logic.CHATTER_RANGE then
      ctx.others_near = ctx.others_near + 1
      if p.IsSpeaking and p:IsSpeaking() then best = 0 end
    end
  end
  for _, l in ipairs(Tn.lines) do
    if l.sid ~= sid and dist3(l, pos) <= Logic.CHATTER_RANGE then
      local dt = t - l.t
      if dt >= 0 and (not best or dt < best) then best = dt end
    end
  end
  ctx.chatter_s = best
  local av, bot = AV(), nil
  bot = av and av.Get and av.Get() or nil
  local gaze_av
  if bot then
    local bp = bot:GetPos()
    gaze_av = angle_to(ply, bp)
    ctx.avatar = { dist = dist3(ply:EyePos(), bp), gaze = gaze_av }
  end
  for _, p in ipairs(others) do
    if dist3(p:GetPos(), pos) <= 1000 then
      local a = angle_to(ply, p:EyePos())
      if a <= Logic.LOOK_HUMAN_DEG and (not gaze_av or a < gaze_av) then ctx.looking_at_human = true; break end
    end
  end
  return ctx
end

-- ------------------------------------------------------------------------------------------------ the verdict
function Tn.Deps(ply)
  local h = H()
  local Wake = h and h.Wake and h.Wake.Wake
  local deps = { Wake = Wake, av_command = AvLogic.match_command, dir_command = DirLogic.match_command, parser = h and h.parser,
                 prev = h and h.prev_req and h.prev_req[sid_of(ply) or ""] or nil, intents = {} }
  if WaterIntent then deps.intents[#deps.intents + 1] = function(t) return WaterIntent.parse(t, WaterC) end end
  if WorldIntent then deps.intents[#deps.intents + 1] = function(t) return WorldIntent.parse(t) end end
  if Wake then
    deps.wake_intent = Wake.intent
    deps.strip = function(text) return h.Wake.strip(ply, text) end
  end
  return deps
end

--- Decide one line. -> the table of turnlogic.decide (addressed, score, reasons, route, final ...). Records the decision; prints nothing about the words.
function Tn.Decide(ply, text, src)
  local ctx = Tn.Context(ply, src)
  local d = Logic.decide(text, ctx, Tn.Deps(ply))
  d.ctx = ctx
  Tn.counts.decided = Tn.counts.decided + 1
  if d.addressed == "maybe" then Tn.counts.maybe = Tn.counts.maybe + 1 end
  if d.final == "ignore" then Tn.counts.ignored = Tn.counts.ignored + 1 else Tn.counts.answered = Tn.counts.answered + 1 end
  local why = d.reasons[1] or "no signal"
  if d.reasons[2] then why = why .. "; " .. d.reasons[2] end
  ring_add({ t = now(), src = src, n = #tostring(text), addressed = d.addressed, score = d.score, route = d.route, final = d.final, why = why, promoted = d.promoted })
  dprint("%s %d chars -> %s (%d) route %s -> %s: %s", src, #tostring(text), d.addressed, d.score, d.route, d.final, why)
  return d
end

-- ------------------------------------------------------------------------------------------------ acting
local function roast_of(ply)
  local D = Mentat.Director
  return D and D.RoastLevel and D.RoastLevel(ply) or "gentle"
end

-- The world's own readback (world.lua publishes it in m/s2), as a percent of Earth's gravity; nil when the world module is not running (then no gravity fact is given:
-- he never guesses a physical number). The engine's own sv_gravity is NOT Earth's: its default is about 17 percent stronger.
local EARTH_M_S2 = 9.80665
local function gravity_percent()
  local g = GetGlobalFloat and GetGlobalFloat("mentat_world_g_m_s2", 0) or 0
  if g and g > 0 then return math.floor(g / EARTH_M_S2 * 100 + 0.5) end
  return nil
end
Tn.gravity_percent = gravity_percent

local function sit_key()
  local pct = gravity_percent()
  if not pct then return "g=unknown" end
  return "g=" .. (pct < 50 and "low" or (pct > 150 and "high" or "norm"))
end

local function facts_now()
  local out = {}
  local pct = gravity_percent()
  if pct then out[#out + 1] = string.format("gravity here is %d percent of Earth's", pct) end
  local h = H()
  if h and h.order then out[#out + 1] = string.format("%d things are standing in the holodeck", #h.order) end
  return out
end

local ADD_OK = { add = true, undo = true }

--- The model's action, checked again here (the bridge already did; the game does not trust the bridge to be right). -> canonical text for the parser, or nil.
function Tn.ActionText(action)
  if type(action) ~= "table" then return nil end
  if not ADD_OK[action.name] then return nil end
  if action.name == "undo" then return "undo" end
  local kind, count = action.kind, action.count
  if type(kind) ~= "string" or #kind < 1 or #kind > 40 or not kind:match("^[%a][%a ]*$") then return nil end
  if type(count) ~= "number" or count ~= math.floor(count) or count < 1 or count > 12 then return nil end       -- (a number, not a string that reads as one)
  return "add " .. math.floor(count) .. " " .. kind:lower()
end

local function clean_say(s)
  s = tostring(s or ""):gsub("[%c\127]", " "):gsub("%s+", " ")
  s = s:gsub("^%s+", ""):gsub("%s+$", "")
  return s:sub(1, 240)
end

local MOOD_OK = { opera = true, boast = true, roast = true, helpful = true, musing = true, exclaim = true }

--- Show a reply: chat to those near him and aloud (the holodeck's own Say), and the matching body language.
local function present(ply, say, mood, source)
  local h, D = H(), Mentat.Director
  if not (h and h.Say) then return end
  h.Say(ply, say, nil, { speak = true })
  if D and D.Cue and D.Logic and MOOD_OK[mood] then D.Cue(D.Logic.MOOD_CUE[mood] or "emphasis", mood) end
  local sid = sid_of(ply)
  if sid then Tn.spoke[sid] = now() end
end

--- Answer a line that is for him and that the parser does not own. Calls the bridge; always ends in some line (never silence).
function Tn.Talk(ply, text, d, opts)
  local h = H()
  local sid = sid_of(ply)
  local function canned_local()
    Tn.counts.fallback = Tn.counts.fallback + 1
    present(ply, "I heard you, monkey, but I only build things. Say 'add a couch' or 'help'.", "helpful", "local")
  end
  if not Tn.ModelOk() then return false end
  if sid and Tn.busy[sid] and now() - Tn.busy[sid] < 40 then
    h.Tell(ply, "I am still thinking about your last request.")
    return true
  end
  if not h.BridgeAllow("reply", ply) then canned_local(); return true end
  if sid then Tn.busy[sid] = now() end
  Tn.counts.talk = Tn.counts.talk + 1
  local strip = {}
  for _, p in ipairs(player.GetAll()) do if is_player(p) and not is_bot(p) then strip[#strip + 1] = tostring(p:Nick()) end end
  for _, n in ipairs((d and d.ctx and d.ctx.names) or {}) do if n ~= "skippy" then strip[#strip + 1] = n end end
  local body = { kind = "talk", text = tostring(text):sub(1, 300), roast = roast_of(ply), facts = facts_now(), sit = sit_key(), strip = strip,
                 frontier_cap = CV.frontier_cap:GetInt(), timeout = CV.timeout:GetFloat(), who = sid }
  local sent = h.BridgePost("/turn", body, function(raw)
    if sid then Tn.busy[sid] = nil end
    if not is_player(ply) then return end
    local obj = util.JSONToTable(raw or "")
    local say = type(obj) == "table" and clean_say(obj.say) or ""
    if say == "" then canned_local(); return end
    local src = type(obj.source) == "string" and obj.source or "?"
    if src == "cache" then Tn.counts.cache = Tn.counts.cache + 1
    elseif src == "frontier" then Tn.counts.frontier = Tn.counts.frontier + 1
    elseif src == "local" then Tn.counts.local_ = Tn.counts.local_ + 1
    else Tn.counts.fallback = Tn.counts.fallback + 1 end
    Tn.last_turn = { t = now(), source = src, tries = obj.tries, why = obj.why, action = type(obj.action) == "table" and obj.action.name or nil }
    dprint("reply from %s after %s tries", src, tostring(obj.tries))
    local act = Tn.ActionText(obj.action)
    if act then
      if h.Handle then h.Handle(ply, act, { turn_ok = true, src = "turn" }) end        -- the holodeck's own answer reports what really happened
      local sidk = sid_of(ply)
      if sidk then Tn.spoke[sidk] = now() end
    else
      present(ply, say, MOOD_OK[obj.mood] and obj.mood or "helpful", src)
    end
  end, function(reason)
    if sid then Tn.busy[sid] = nil end
    if not is_player(ply) then return end
    Tn.last_error = tostring(reason):sub(1, 120)
    canned_local()
  end, 30)
  if not sent then
    if sid then Tn.busy[sid] = nil end
    canned_local()
  end
  return true
end

--- The line is for him and the parser (or the old planner) owns it: hand it to the old flow, marked so it is not decided twice.
local function run_old(ply, text, d, src)
  local h = H()
  local name = d.name and d.name ~= "skippy" and d.name or nil
  if h and h.Handle then h.Handle(ply, text, { turn_ok = true, src = src, woke = d.woke, wake_name = name }) end
end

--- Act on a verdict that says the line was for him.
function Tn.Act(ply, text, d, src)
  if d.final == "talk" then
    if Tn.Talk(ply, d.rest or text, d) then return true end
    run_old(ply, d.rest or text, d, src)
    return true
  end
  run_old(ply, d.rest or text, d, src)
  return true
end

--- The small-model tiebreak for a line that is only maybe for him. The answer arrives later; the default is silence.
function Tn.Tiebreak(ply, text, d, src)
  if not CV.tiebreak:GetBool() or not Tn.ModelOk() then return end
  local sid = sid_of(ply)
  if not sid or now() - (Tn.tb_last[sid] or -99) < 6 then return end
  local h = H()
  if not h.BridgeAllow("reply", ply) then return end
  Tn.tb_last[sid] = now()
  Tn.counts.tiebreaks = Tn.counts.tiebreaks + 1
  local c = d.ctx or {}
  local strip = {}
  for _, p in ipairs(player.GetAll()) do if is_player(p) and not is_bot(p) then strip[#strip + 1] = tostring(p:Nick()) end end
  local sig = { window = c.since_spoke ~= nil and c.since_spoke <= Logic.WINDOW_S, gaze = c.avatar and c.avatar.gaze and c.avatar.gaze <= 60 or false,
                near = c.avatar and c.avatar.dist and c.avatar.dist <= 400 or false, others = (c.others_near or 0) > 0 }
  h.BridgePost("/turn", { kind = "tiebreak", text = tostring(text):sub(1, 300), signals = sig, strip = strip, frontier_cap = CV.frontier_cap:GetInt(),
                          tiebreak_cap = CV.tiebreak_cap:GetInt(), who = sid }, function(raw)
    local obj = util.JSONToTable(raw or "")
    if not is_player(ply) or type(obj) ~= "table" then return end
    ring_add({ t = now(), src = src, n = #tostring(text), addressed = obj.addressed == "yes" and "yes" or "no", score = d.score, route = d.route, final = "tiebreak",
               why = "tiebreak says " .. tostring(obj.addressed) })
    if obj.addressed == "yes" and d.route ~= "ignore" then
      d.final = d.route
      d.addressed = "yes"
      Tn.Act(ply, text, d, src)
    end
  end, function() end, 10)
end

-- ------------------------------------------------------------------------------------------------ entry points
--- In front of Holodeck.Handle (avatar.lua). Returns true when the line was dealt with here (the old flow must not run), false to let it run.
function Tn.Intercept(ply, said, opts)
  if not Tn.Enabled() or not is_player(ply) then return false end
  opts = opts or {}
  if opts.turn_ok then return false end
  if opts.found_done then return false end                                   -- the second pass of a request that waited for the resolver: decided already
  local src = opts.src or "console"
  local full = type(opts.heard) == "string" and opts.heard or said
  local d = Tn.Decide(ply, full, src)
  if d.final == "ignore" then
    if d.addressed == "maybe" then Tn.Tiebreak(ply, full, d, src) end
    return true                                                               -- not for him (a topic, noise), or nothing in it: the old flow has nothing to add
  end
  if d.final == "talk" then
    return Tn.Talk(ply, d.rest or said, d, opts)
  end
  return false
end

--- A chat line that did not begin with one of the speaker's names (holodeck.lua's chat hook).
function Tn.Overheard(ply, text)
  if not Tn.Enabled() or not is_player(ply) or is_bot(ply) then return end
  local d = Tn.Decide(ply, text, "chat")
  if d.final == "ignore" then
    if d.addressed == "maybe" then Tn.Tiebreak(ply, text, d, "chat") end
    return
  end
  Tn.Act(ply, text, d, "chat")
end

-- ------------------------------------------------------------------------------------------------ hooks
hook.Add("PlayerSay", "Mentat.TurnsSeen", function(ply, text)
  if not is_player(ply) or is_bot(ply) then return end
  local sid = sid_of(ply)
  local p = ply:GetPos()
  Tn.lines[#Tn.lines + 1] = { sid = sid, t = now(), x = p.x, y = p.y, z = p.z }
  if #Tn.lines > 24 then table.remove(Tn.lines, 1) end
end)

hook.Add("HolodeckSaid", "Mentat.Turns", function(ply, text)
  local sid = sid_of(ply)
  if sid then Tn.spoke[sid] = now() end
end)

hook.Add("HolodeckOverheard", "Mentat.Turns", function(ply, text) Tn.Overheard(ply, text) end)

hook.Add("PlayerDisconnected", "Mentat.Turns", function(ply)
  local sid = sid_of(ply)
  if sid then Tn.spoke[sid], Tn.busy[sid], Tn.tb_last[sid] = nil, nil, nil end
end)

concommand.Add("skippy_turn_log", function(ply)
  if is_player(ply) and not ply:IsAdmin() then return end
  print(string.format("[turns] %d decisions (%d answered, %d ignored, %d maybe); talk %d: frontier %d, local %d, cache %d, fallback %d; tiebreaks %d",
    Tn.counts.decided, Tn.counts.answered, Tn.counts.ignored, Tn.counts.maybe, Tn.counts.talk, Tn.counts.frontier, Tn.counts.local_, Tn.counts.cache,
    Tn.counts.fallback, Tn.counts.tiebreaks))
  for i = math.max(1, #Tn.ring - 29), #Tn.ring do
    local e = Tn.ring[i]
    print(string.format("[turns] %6.1f %-7s %3d chars %-5s %4d  %-8s -> %-8s %s", e.t, e.src, e.n, e.addressed, e.score or 0, tostring(e.route), tostring(e.final), e.why or ""))
  end
end, nil, "The last decisions about who a line was for")

concommand.Add("skippy_turns_status", function(ply)
  if is_player(ply) and not ply:IsAdmin() then return end
  print("[turns] enabled=" .. tostring(CV.on:GetBool()) .. " model route " .. (Tn.ModelOk() and "available" or "unavailable (no bridge, or holodeck_agent 0)") ..
    " tiebreak=" .. tostring(CV.tiebreak:GetBool()) .. " last error: " .. tostring(Tn.last_error))
end, nil, "Skippy's solid turns: state")

return Tn
