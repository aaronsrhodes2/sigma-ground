-- turnlogic.lua -- PURE (no engine calls). Skippy's "solid turns" rules (A7b, 2026-10-13): was a line aimed at Skippy, and what happens to it.
-- Loaded by mentat/avatar/turns.lua (the game) and by holodeck/turn_eval.py (the corpus score). Both run exactly this file.
--
-- 1. ADDRESSEE. A deterministic score from signals the server can measure; documented weights, no model. Verdict yes (score >= 60), maybe (35..59), no (< 35).
--    "maybe" is NEVER answered by default; it becomes yes only through the conversation-window rule below, or through the optional small-model tiebreak (turns.lua).
--      +100  push-to-talk (+skippy / the talk key) and the console: always for him (verdict yes, nothing else is looked at)
--      +100  a sentence that STARTS with one of the speaker's own wake names (a few little words like "hey" may come first)
--       -60  ... but his name is the subject of a remark ("Skippy is silly", "Skippy keeps ..."): a topic, not an address
--       -10  his name is later in the sentence ("I think skippy broke it"): in passing
--       -70  the sentence starts with another player's name ("Bob, add a couch", "hey Bob ...") or a group call ("guys, look")
--      gaze (the angle between where the speaker looks and where he floats; only while he is in the world):
--       +30 within 12 degrees, +20 within 30, +8 within 60, -15 at 100 or more
--       -30  the speaker looks straight at another player (within 15 degrees and nearer to their line than to his)
--      distance to him: +15 within 300 HU (5.7 m), +6 within 700, -10 beyond 1500
--      conversation window: he spoke to this player within 8 s: +50; within 20 s: +40 (a follow-up needs no wake word)
--      -30   two humans are talking: another player spoke within 10 s and is within 600 HU of the speaker
--      text: a one or two word interjection with no question or order -15; 30 words or more -10; a question or "you"/order +4 / +6
--    Window rule: a "maybe" inside the 20 s window whose gaze is within 45 degrees OR whose distance is within 400 HU becomes yes
--    ("he just spoke to you and you are looking at him / standing by him"). Without that agreement it stays maybe = no answer.
--    Never promoted: topic, another player's name, looking at another player, two humans talking.
--
-- 2. ROUTE. What happens to an addressed line, deterministic first:
--      ignore  nothing in it (noise, "um", a recogniser's silence hallucination "thank you")
--      parser  the existing parser owns it: his commands (come/go away, shush, roast level, notebook), wake-name commands, any request the
--              holodeck parser understands (add/remove/move/undo/water/gravity, and an unknown noun, which the object resolver looks up)
--      plan    a vague wish ("make it cozy", "somewhere to relax"): the existing /plan agent
--      talk    everything else (chat, a question, "help"): the model route with a validated, fallback-safe reply
local T = {}

T.WEIGHTS = {
  wake_start = 100, topic = -60, passing = -10, vocative = -70,
  gaze_12 = 30, gaze_30 = 20, gaze_60 = 8, gaze_away = -15, looking_human = -30,
  near_300 = 15, near_700 = 6, far = -10,
  window_8 = 50, window_20 = 40, chatter = -30,
  short = -15, long = -10, question = 4, second = 6,
}
T.THRESH = { yes = 60, maybe = 35 }
T.WINDOW_S = 20
T.WINDOW_FAST_S = 8
T.CHATTER_S = 10
T.CHATTER_RANGE = 600
T.PROMOTE_GAZE = 45
T.PROMOTE_DIST = 400
T.LOOK_HUMAN_DEG = 15

local function set(s) local t = {}; for w in s:gmatch("%S+") do t[w] = true end; return t end

local LEAD = set("hey hi hello yo oi ok okay so yeah well um uh")
local GROUP = set("guys everyone everybody folks team anyone anybody dude bro mate yall y'all lads gang")
local STOP_NICK = set("the and you add can hey not his her she him man boy guy pro dad mom mum xx xxx bot new old big")
local TOPIC_HEAD = set("is was has had isnt wasnt hasnt seems seemed looks looked sounds keeps kept always never does did doesnt didnt got gets thinks likes loves hates needs wants broke just sure really actually probably")
local WH = set("what why how when where who which whose can could do does did is are will would should shall may")
local ORDERS = set("please add make build put place remove undo come show tell give help pour drop turn move rotate clear fill spawn bring get set let sing")
local SECOND = set("you your youre youll youve u ur")
local FILLER_NOISE = set("um uh hmm hm er ah oh mm mmm huh eh aha uhh umm hmmm ahh ohh")
local HALLUCINATIONS = { ["thank you"] = true, ["thanks for watching"] = true, ["you"] = true, ["bye"] = true, ["the end"] = true, ["thank you for watching"] = true }
local QAUX = set("is are was were does did do")
local QSUBJ = set("it that there this the")
local WISH = set("make cozy somewhere place room need want like furnish decorate vibe feel cosy comfortable setup")

--- Lower case, accents folded (wakewords' fold), punctuation to single spaces.
function T.fold(Wake, s)
  s = Wake and Wake.fold and Wake.fold(tostring(s or "")) or tostring(s or ""):lower()
  s = s:gsub("[%p]", function(c) if c == "'" then return "" end return " " end):gsub("%s+", " ")
  return (s:gsub("^ ", ""):gsub(" $", ""))
end

local function tokens(folded)
  local out = {}
  for w in folded:gmatch("%S+") do out[#out + 1] = w end
  return out
end
T.tokens = tokens

--- The words of a player's name that someone could call them by: the first word and the whole name, at least 3 letters, nothing common.
local function nick_keys(Wake, nick)
  local f = T.fold(Wake, nick)
  local toks = tokens(f)
  local keys = {}
  if toks[1] and #toks[1] >= 3 and not STOP_NICK[toks[1]] then keys[toks[1]] = true end
  local whole = f:gsub(" ", "")
  if #whole >= 3 and not STOP_NICK[whole] then keys[whole] = true end
  if #toks >= 2 and toks[#toks] and #toks[#toks] >= 4 and not STOP_NICK[toks[#toks]] then keys[toks[#toks]] = true end
  return keys
end

--- Does the sentence start (after a "hey") with another player's name or a group call? -> the matched word, "group" | nil
function T.vocative(Wake, folded, human_names)
  local toks = tokens(folded)
  local i = 1
  while toks[i] and LEAD[toks[i]] and i <= 3 do i = i + 1 end
  local w = toks[i]
  if not w then return nil end
  if GROUP[w] then return w, "group" end
  for _, nick in ipairs(human_names or {}) do
    local keys = nick_keys(Wake, nick)
    if keys[w] then return w, "human" end
    if i + 1 <= #toks and keys[w .. toks[i + 1]] then return w, "human" end
  end
  return nil
end

--- Is the name later in the sentence (not at the start)? names are stored forms; Skippy's spelling variants count.
function T.mentions(Wake, folded, names)
  local pad = " " .. folded .. " "
  local list = {}
  for _, n in ipairs(names or { "skippy" }) do list[#list + 1] = n end
  if Wake and Wake.SKIPPY_VARIANTS then for _, v in ipairs(Wake.SKIPPY_VARIANTS) do list[#list + 1] = v end end
  for _, n in ipairs(list) do
    local f = T.fold(Wake, n)
    if f ~= "" and (pad:find(" " .. f .. " ", 1, true) or pad:find(" " .. f .. "s ", 1, true)) then return true end
  end
  return false
end

--- His name was the start of the sentence and the words after it make a remark about him, not a request to him.
--- orig = the text as typed, rest = what the wake matcher left. A comma / colon / dash right after the name is an address ("Skippy, is it...").
function T.is_topic(Wake, orig, rest)
  local o = tostring(orig or "")
  local r = tostring(rest or "")
  local cut = #o - #r
  if cut > 0 and o:sub(1, cut):find("[,:;!?%-]") then return false end
  local toks = tokens(T.fold(Wake, r))
  local h = toks[1]
  if not h then return false end
  if not TOPIC_HEAD[h] then return false end
  if toks[2] and (SECOND[toks[2]]) then return false end            -- "does you ..." is not a remark
  -- "Skippy is the moon heavier or lighter than earth": spoken questions have no comma. A copula or "do" followed by it/that/there/this/the and at
  -- least five words in all reads as a question to him; "Skippy is the best" (four) stays a remark.
  if QAUX[h] and toks[2] and QSUBJ[toks[2]] and #toks >= 5 then return false end
  if tostring(orig):find("?", 1, true) then return false end
  return true
end

--- Text features that need no context.
function T.text_features(Wake, text)
  local f = T.fold(Wake, text)
  local toks = tokens(f)
  local q = tostring(text):find("?", 1, true) ~= nil or (toks[1] ~= nil and WH[toks[1]] == true)
  local second, order = false, false
  for i, w in ipairs(toks) do
    if SECOND[w] then second = true end
    if i <= 2 and ORDERS[w] then order = true end
  end
  return { words = #toks, question = q, second = second, order = order, folded = f }
end

local function angle_ok(a) return type(a) == "number" and a >= 0 end

--- The score. in_ = { src, woke, topic, passing, vocative, avatar = { dist=, gaze= } | nil, looking_at_human, since_spoke, chatter_s, others_near, tf = text_features }
--- -> { addressed = "yes"|"no"|"maybe", score, reasons = { ... }, promoted = bool }
function T.score(in_)
  local W, reasons, score = T.WEIGHTS, {}, 0
  local function add(n, why) score = score + n; reasons[#reasons + 1] = string.format("%s (%+d)", why, n) end
  if in_.src == "console" then return { addressed = "yes", score = 100, reasons = { "the console is always for him" } } end
  if in_.src == "ptt" then return { addressed = "yes", score = 100, reasons = { "push-to-talk is always for him" } } end
  if in_.woke then
    add(W.wake_start, "named at the start")
    if in_.topic then add(W.topic, "his name is the subject of a remark, not an address") end
  else
    if in_.passing then add(W.passing, "his name is in the sentence but not the start") end
    if in_.vocative then add(W.vocative, in_.vocative == "group" and "a call to a group" or "addressed to another player by name") end
  end
  -- A sentence that starts with his name needs no help from the room: gaze, distance, the window and chatter are only weighed when he was NOT named,
  -- or when his name was only the subject of a remark (then they may tip a remark he is plausibly answering).
  local context = (not in_.woke) or in_.topic
  local av = context and in_.avatar or nil
  if av then
    local g = av.gaze
    if angle_ok(g) then
      if g <= 12 then add(W.gaze_12, "looking at him")
      elseif g <= 30 then add(W.gaze_30, "looking near him")
      elseif g <= 60 then add(W.gaze_60, "looking roughly toward him")
      elseif g >= 100 then add(W.gaze_away, "looking away from him") end
    end
    local d = av.dist
    if type(d) == "number" then
      if d <= 300 then add(W.near_300, "standing next to him")
      elseif d <= 700 then add(W.near_700, "within a few steps of him")
      elseif d >= 1500 then add(W.far, "far from him") end
    end
  end
  if context and in_.looking_at_human then add(W.looking_human, "looking straight at another player") end
  local win = context and in_.since_spoke or nil
  local in_window = type(win) == "number" and win >= 0 and win <= T.WINDOW_S
  if in_window then
    if win <= T.WINDOW_FAST_S then add(W.window_8, "he spoke to you moments ago") else add(W.window_20, "he spoke to you a few seconds ago") end
  end
  local chat = context and type(in_.chatter_s) == "number" and in_.chatter_s <= T.CHATTER_S and (in_.others_near or 0) >= 1
  if chat then add(W.chatter, "two humans are talking") end
  local tf = in_.tf
  if tf and not in_.woke then
    if tf.words <= 2 and not (tf.question or tf.second or tf.order) then add(W.short, "a short interjection")
    elseif tf.words >= 30 then add(W.long, "a long ramble") end
    if tf.question then add(W.question, "it is a question") end
    if tf.second or tf.order then add(W.second, "it says you / gives an order") end
  end
  local verdict = score >= T.THRESH.yes and "yes" or (score >= T.THRESH.maybe and "maybe" or "no")
  local promoted = false
  if verdict == "maybe" and in_window and not in_.topic and not in_.vocative and not in_.looking_at_human and not chat and av then
    local g, d = av.gaze, av.dist
    if (angle_ok(g) and g <= T.PROMOTE_GAZE) or (type(d) == "number" and d <= T.PROMOTE_DIST) then
      verdict, promoted = "yes", true
      reasons[#reasons + 1] = "window rule: he just spoke to you and gaze or distance agree"
    end
  end
  return { addressed = verdict, score = score, reasons = reasons, promoted = promoted }
end

local QWORD = set("what whats why how when where who whom whose which is are am was were do does did can could will would should shall may might have has had isnt arent wasnt")
local REQUEST_VERBS = set("add make build put place remove undo show pour drop turn move rotate clear fill spawn bring get set give create summon furnish delete have grab throw toss")
local WISHWORD = set("vibe feel atmosphere ambience ambiance mood cozy cosy relax relaxing somewhere furnish decorate theme style comfortable homey")

--- Is this a question to him (rather than an order)? "can you add a lamp" / "could we get a couch" are orders in a polite form, not questions.
function T.is_question(raw, toks)
  local i = 1
  while toks[i] and LEAD[toks[i]] and i <= 3 do i = i + 1 end
  local first = toks[i]
  if not first then return false end
  if (first == "can" or first == "could" or first == "would" or first == "will") and (toks[i + 1] == "you" or toks[i + 1] == "u" or toks[i + 1] == "i"
      or toks[i + 1] == "we" or toks[i + 1] == "me") then
    for k = i + 2, math.min(i + 6, #toks) do if REQUEST_VERBS[toks[k]] then return false end end
  end
  if QWORD[first] and not (first == "do" and (toks[i + 1] == "it" or toks[i + 1] == "that" or toks[i + 1] == "this")) then return true end
  local last = tostring(raw):match("(%S)%s*$")
  return last == "?" and not REQUEST_VERBS[first]
end

--- The route of a line that IS for him. deps = { Wake, parser = <Parse instance>, prev = the player's previous request (for "another one"),
--- wake_intent, av_command, dir_command = fn(text) -> truthy, intents = { fn(text) -> truthy, ... } (the water and physics words) } (any may be nil).
--- -> route ("parser"|"plan"|"talk"|"ignore"), why
function T.route(text, src, deps)
  deps = deps or {}
  local Wake = deps.Wake
  local f = T.fold(Wake, text)
  local toks = tokens(f)
  local letters = 0
  for _ in tostring(text):gmatch("[%a\128-\255]") do letters = letters + 1 end
  if letters < 2 or #toks == 0 then return "ignore", "nothing to answer" end
  local allnoise = true
  for _, w in ipairs(toks) do if not FILLER_NOISE[w] then allnoise = false; break end end
  if allnoise then return "ignore", "only filler sounds" end
  if (src == "voice" or src == "ptt") and HALLUCINATIONS[f] then return "ignore", "a recogniser's silence hallucination" end
  if deps.wake_intent and deps.wake_intent(text) then return "parser", "a wake-name command" end
  if deps.av_command and deps.av_command(text) then return "parser", "come / go away" end
  if deps.dir_command and deps.dir_command(text) then return "parser", "one of his own commands" end
  for _, fn in ipairs(deps.intents or {}) do
    local ok, r = pcall(fn, text)
    if ok and r then return "parser", "water or physics words" end
  end
  local question = T.is_question(text, toks)
  local p = deps.parser
  if not p then return "talk", "no parser" end
  local ok, req = pcall(function() return p:parse(text, deps.prev) end)
  if not (ok and type(req) == "table") then return "talk", "the parser could not read it" end
  if req.again and not question then return "parser", "repeat the last thing" end
  if question then return "talk", "a question" end
  local act = req.action
  if act and act ~= "add" and act ~= "none" then return "parser", "the parser reads a " .. tostring(act) end
  if req.scene and req.scene ~= "" then return "parser", "a room the parser knows" end
  local real, unknown = false, false
  for _, it in ipairs(req.items or {}) do
    local k = tostring(it.kind)
    if k:match("^UNKNOWN") then unknown = true elseif k ~= "@last" and k ~= "" then real = true end
  end
  if act == "add" and real then return "parser", "the parser reads things to add" end
  local wish = false
  for i, w in ipairs(toks) do
    if WISHWORD[w] then wish = true; break end
    if (w == "place" or w == "spot" or w == "space" or w == "corner") and (toks[i + 1] == "to" or toks[i + 1] == "for") then wish = true; break end
  end
  if wish and (act == "add" or act == "none") then return "plan", "a vague wish for the planner" end
  if act == "add" and unknown then return "parser", "an object the resolver can look up" end
  return "talk", "the parser found nothing to do"
end

--- Everything in one call. ctx = { src, names, human_names, avatar, looking_at_human, since_spoke, chatter_s, others_near }
--- deps additionally: strip(text) -> woke, rest, name   (the speaker's own names)
function T.decide(text, ctx, deps)
  deps = deps or {}
  ctx = ctx or {}
  local Wake = deps.Wake
  local src = ctx.src or "console"
  local woke, rest, name = false, text, nil
  if deps.strip then woke, rest, name = deps.strip(text) end
  rest = woke and rest or text
  local inp = { src = src, woke = woke and true or false, avatar = ctx.avatar, looking_at_human = ctx.looking_at_human, since_spoke = ctx.since_spoke,
                chatter_s = ctx.chatter_s, others_near = ctx.others_near }
  inp.tf = T.text_features(Wake, rest)
  if woke then
    inp.topic = T.is_topic(Wake, text, rest)
  else
    local f = inp.tf.folded
    local w, kind = T.vocative(Wake, f, ctx.human_names)
    if w then inp.vocative = kind end
    inp.passing = T.mentions(Wake, f, ctx.names)
  end
  local s = T.score(inp)
  local route, why = T.route(rest, src, deps)
  if woke and route == "ignore" and why == "nothing to answer" and (src == "chat" or src == "voice") then
    route, why = "talk", "called by name only"                       -- "Skippy?" is a call: he answers, he is never silent for his own name
  end
  local out = { addressed = s.addressed, score = s.score, reasons = s.reasons, promoted = s.promoted, woke = inp.woke, name = name, rest = rest,
                route = route, route_why = why, src = src }
  out.answer = (s.addressed == "yes") and route ~= "ignore"
  out.final = out.answer and route or "ignore"
  return out
end

return T
