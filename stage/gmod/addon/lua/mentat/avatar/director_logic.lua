-- avatar/director_logic.lua -- WHEN Skippy speaks and moves on his own: the pure half of the behaviour director (avatar/director.lua). No engine calls,
-- so every rule is tested offline (test_gmod_director.py). The rules:
--   * a global CHATTINESS (quiet / normal / chatty): the least gap between two unprompted lines, an hourly cap, how likely a given event is to
--     get a remark, and the lowest priority allowed to speak at all
--   * per-behaviour COOLDOWNS (a splash does not get a remark every second) and a CHANCE for each event
--   * never talk over a human who is speaking, never over his own previous line; a remark that only has to wait is DEFERRED a moment instead of lost
--   * yield while the owner is mid-command (typing or speaking a request, or the agent still planning one): the answer to that command comes first
--   * quiet hours (optional "23-7" in the server's local time)
--   * INTERRUPTIBLE: when a human starts talking over one of his lines, the line is cut
--   * his own commands: "shush", "be chatty", "roast me harder", "go easy on me", "what do you remember", "forget that / today / everything", "stop remembering"
local D = {}

-- gap: least seconds between two unprompted lines; hour_cap: lines an hour; chance: multiplies each event's own chance; floor: lowest priority that may speak;
-- whim: average seconds between idle whims; idle_s: seconds of a still, silent owner before he remarks on it
D.CHAT = {
  quiet = { gap = 420, hour_cap = 3, chance = 0.25, floor = 2, whim = 1500, idle_s = 600 },
  normal = { gap = 100, hour_cap = 16, chance = 0.7, floor = 1, whim = 420, idle_s = 150 },
  chatty = { gap = 35, hour_cap = 50, chance = 1.0, floor = 0, whim = 150, idle_s = 60 },
}
D.CHAT_ORDER = { "quiet", "normal", "chatty" }
D.MID_COMMAND_S = 4                -- seconds after the owner's last request during which unprompted remarks (priority below 3) wait
D.DEFER_S = 1.0                    -- how long a deferred remark waits before trying again
D.DEFER_MAX_S = 9                  -- ... and how long it stays worth saying
D.TAIL_S = 0.5                     -- quiet seconds after a line before the next may begin

-- priority 3: answers to something the player just did (arrival, dismissal, a name change): spoken whatever the chattiness (unless a human is mid-sentence)
-- 2: reactions he should usually make; 1: comments; 0: pure whims
D.BEHAVIOURS = {
  arrive = { priority = 3, chance = 1.0, cooldown = 0, cue = "hat_tip" },
  dismiss = { priority = 3, chance = 1.0, cooldown = 0, cue = "hat_tip" },
  name_change = { priority = 3, chance = 1.0, cooldown = 20, cue = "nod" },
  recall = { priority = 3, chance = 1.0, cooldown = 0, cue = "emphasis" },
  roast_now = { priority = 3, chance = 1.0, cooldown = 0, cue = "emphasis", situation = "roast" },
  beam_ok = { priority = 2, chance = 0.5, cooldown = 40, cue = "flourish", look = true },
  build = { priority = 1, chance = 0.3, cooldown = 60, cue = "flourish", look = true },
  beam_fail = { priority = 2, chance = 0.8, cooldown = 25, cue = "shrug" },
  big_pile = { priority = 1, chance = 0.7, cooldown = 420, cue = "sway" },
  splash = { priority = 1, chance = 0.4, cooldown = 90, cue = "bounce" },
  water_jump = { priority = 1, chance = 0.7, cooldown = 120, cue = "bounce" },
  owner_idle = { priority = 1, chance = 0.8, cooldown = 300, cue = "quizzical" },
  wake_miss = { priority = 1, chance = 0.5, cooldown = 240, cue = "quizzical" },
  musing = { priority = 0, chance = 1.0, cooldown = 0, cue = "sway" },
  physics = { priority = 0, chance = 1.0, cooldown = 0, cue = "emphasis" },
  banana = { priority = 0, chance = 1.0, cooldown = 1800, cue = "shrug" },
  opera = { priority = 0, chance = 1.0, cooldown = 2400, cue = "conduct" },
  boast = { priority = 0, chance = 1.0, cooldown = 0, cue = "flourish" },
  tip = { priority = 0, chance = 1.0, cooldown = 900, cue = "emphasis" },
  roast = { priority = 0, chance = 1.0, cooldown = 0, cue = "emphasis" },
}

--- The mood a line carries (from the bridge) -> the cue to play with it. An opera is always a conductor's sweep.
D.MOOD_CUE = { opera = "conduct", boast = "flourish", roast = "emphasis", helpful = "emphasis", musing = "sway", exclaim = "bounce" }

-- ------------------------------------------------------------------------------------------------ quiet hours
--- "23-7" (from 23:00 until 07:00) -> { from, to } hours, or nil for "" / nonsense.
function D.parse_quiet_hours(spec)
  local a, b = tostring(spec or ""):match("^%s*(%d%d?)%s*%-%s*(%d%d?)%s*$")
  a, b = tonumber(a), tonumber(b)
  if not a or not b or a > 24 or b > 24 or a == b then return nil end
  return { from = a % 24, to = b % 24 }
end

function D.in_quiet_hours(q, hour)
  if not q then return false end
  if q.from < q.to then return hour >= q.from and hour < q.to end
  return hour >= q.from or hour < q.to
end

-- ------------------------------------------------------------------------------------------------ state
function D.new()
  return { cool = {}, last_spoke = -1e9, spoke = {}, speaking_until = 0, idle_since = 0, idle_fired = false, owner_pos = nil, next_whim = 0 }
end

local function recent_count(st, now)
  local n, keep = 0, {}
  for _, t in ipairs(st.spoke) do if now - t < 3600 then n = n + 1; keep[#keep + 1] = t end end
  st.spoke = keep
  return n
end

--- Should he say something about this event now? ev = { name }; ctx = { chat = "quiet|normal|chatty", disabled, quiet_hours (bool), human_speaking (bool),
--- owner_command_age (seconds since the owner's last request), budget_left (new lines left today, or nil if unknown) }.
--- Returns { say = true, situation, cue, look } or { say = false, why = "...", defer = true|nil }.
function D.decide(st, ev, now, ctx, rng)
  local b = D.BEHAVIOURS[ev.name]
  if not b then return { say = false, why = "no such behaviour" } end
  if ctx.disabled then return { say = false, why = "the director is off" } end
  local chat = D.CHAT[ctx.chat] or D.CHAT.normal
  local can_wait = b.priority >= 2
  if ctx.quiet_hours and b.priority < 3 then return { say = false, why = "quiet hours" } end
  if ctx.human_speaking then return { say = false, why = "a human is speaking", defer = can_wait or nil } end
  if now < st.speaking_until + D.TAIL_S then return { say = false, why = "he is still speaking", defer = can_wait or nil } end
  if b.priority < 3 and (ctx.owner_command_age or 1e9) < D.MID_COMMAND_S then return { say = false, why = "the owner is mid-command", defer = can_wait or nil } end
  if b.priority < chat.floor then return { say = false, why = "chattiness " .. tostring(ctx.chat or "normal") .. " is too low for this" } end
  if now < (st.cool[ev.name] or 0) then return { say = false, why = "cooldown" } end
  if b.priority < 3 then
    if now - st.last_spoke < chat.gap then return { say = false, why = "too soon after his last remark" } end
    if recent_count(st, now) >= chat.hour_cap then return { say = false, why = "the hourly cap" } end
    local p = chat.chance * b.chance
    if ctx.budget_left == 0 then p = p * 0.35 end                  -- the day's new lines are spent: only the occasional replay
    if rng() >= p then return { say = false, why = "not this time (chance)" } end
  end
  return { say = true, situation = b.situation or ev.name, cue = b.cue, look = b.look, priority = b.priority }
end

--- He is going ahead: start the cooldown and the pacing for this event.
function D.commit(st, ev_name, now)
  local b = D.BEHAVIOURS[ev_name]
  if not b then return end
  st.cool[ev_name] = now + b.cooldown
  if b.priority < 3 then
    st.last_spoke = now
    st.spoke[#st.spoke + 1] = now
  end
end

--- A line of `dur` seconds begins now.
function D.line_started(st, now, dur) st.speaking_until = now + math.max(0.3, dur or 0) end

--- A human started talking: is he in the middle of a line (then it is cut)?
function D.should_interrupt(st, now, human_started)
  return human_started == true and now < st.speaking_until
end

function D.line_cut(st, now) st.speaking_until = now end

-- ------------------------------------------------------------------------------------------------ whims and idling
local WHIMS = { { "musing", 0.50 }, { "physics", 0.15 }, { "banana", 0.07 }, { "opera", 0.06 }, { "boast", 0.08 }, { "tip", 0.04 }, { "roast", 0.10 } }
D.WHIMS = WHIMS

--- Which kind of whim. r is a uniform number in [0,1).
function D.pick_whim(r)
  local acc = 0
  for _, w in ipairs(WHIMS) do
    acc = acc + w[2]
    if r < acc then return w[1] end
  end
  return "musing"
end

--- When the next idle whim is due (seconds from `now`): the chattiness' average gap, jittered 0.6x to 1.6x. r in [0,1).
function D.next_whim(now, chat, r)
  local c = D.CHAT[chat] or D.CHAT.normal
  return now + c.whim * (0.6 + r)
end

--- Is the owner idle (still and silent for the chattiness' idle time)? Returns true once per idle spell. pos = { x, y }, last_cmd = time of their last request.
function D.owner_idle(st, now, pos, last_cmd, chat)
  local c = D.CHAT[chat] or D.CHAT.normal
  local p = st.owner_pos
  if not p or (pos.x - p.x) ^ 2 + (pos.y - p.y) ^ 2 > 40 * 40 then
    st.owner_pos = { x = pos.x, y = pos.y }
    st.idle_since, st.idle_fired = now, false
    return false
  end
  if last_cmd and last_cmd > st.idle_since then st.idle_since, st.idle_fired = last_cmd, false end
  if not st.idle_fired and now - st.idle_since >= c.idle_s then
    st.idle_fired = true
    return true
  end
  return false
end

--- A small slow wander added to his hover point so he drifts instead of being rigid: x (his forward), y (to his left), z, in HU, from the time.
function D.drift(t)
  return 9 * math.sin(t * 0.37), 12 * math.sin(t * 0.23 + 1.3), 5 * math.sin(t * 0.51 + 0.4)
end

--- What he looks at: the thing the owner just placed (if recent), else what the owner is aiming at (if there is something), else nil.
--- ctx = { placed = { pos, t }, aim = pos or nil }, now. Returns pos, why.
function D.look_target(ctx, now)
  local pl = ctx.placed
  if pl and pl.pos and now - (pl.t or -1e9) < 12 then return pl.pos, "what you just placed" end
  if ctx.aim then return ctx.aim, "what you are aiming at" end
  return nil
end

-- ------------------------------------------------------------------------------------------------ his own commands
local function squash(text)
  local s = tostring(text or ""):lower():gsub("[%p]", " "):gsub("%s+", " ")
  s = s:gsub("^%s+", ""):gsub("%s+$", "")
  local changed = true
  while changed do                                       -- polite padding
    changed = false
    for _, pre in ipairs({ "please ", "hey ", "ok ", "okay ", "so ", "could you ", "can you ", "would you ", "will you ", "skippy ", "now " }) do
      if s:sub(1, #pre) == pre then s = s:sub(#pre + 1); changed = true end
    end
    for _, suf in ipairs({ " please", " now", " thanks", " thank you", " skippy", " for me", " ok" }) do
      if #s > #suf and s:sub(-#suf) == suf then s = s:sub(1, -#suf - 1); changed = true end
    end
  end
  return s
end

local function one_of(s, list) for _, x in ipairs(list) do if s == x then return true end end return false end

--- A sentence (already without the wake word) -> a command table or nil.
---   { cmd = "chat", level = "quiet|normal|chatty" }   { cmd = "roast", delta = +1|-1 | level = "gentle|standard|savage" | now = true }
---   { cmd = "recall" }   { cmd = "forget", scope = "that|today|all" }   { cmd = "journal", on = true|false }
function D.match_command(text)
  local s = squash(text)
  if s == "" or #s > 80 then return nil end
  if one_of(s, { "shush", "shh", "shhh", "hush", "quiet", "be quiet", "be quieter", "stop talking", "stop chatting", "silence", "zip it", "pipe down",
                 "shut up", "less talking", "talk less", "talk less often", "go quiet", "be silent", "stop chattering" }) then return { cmd = "chat", level = "quiet" } end
  if one_of(s, { "be chatty", "be more chatty", "talk more", "chat more", "be talkative", "chatty mode", "you can talk more", "speak up more",
                 "talk to me more", "be chattier", "chatter more", "go chatty" }) then return { cmd = "chat", level = "chatty" } end
  if one_of(s, { "be normal", "normal chatter", "talk normally", "normal chattiness", "be less chatty", "stop being chatty", "be less quiet", "normal mode" }) then
    return { cmd = "chat", level = "normal" }
  end
  if one_of(s, { "roast me harder", "roast me more", "roast me properly", "roast me savagely", "roast harder", "be harsher", "be meaner", "be meaner to me",
                 "insult me more", "insult me harder", "be mean to me", "go harder", "harder", "more insults", "roast me like you mean it", "go savage",
                 "savage mode", "be savage", "full monkey", "roast me mercilessly" }) then
    return { cmd = "roast", delta = 1, now = true }
  end
  if one_of(s, { "go easy on me", "be nice", "be nicer", "be gentler", "be gentle", "gentle mode", "roast me gently", "roast me softer", "roast me less",
                 "tone it down", "be kind", "be kinder", "less insults", "fewer insults", "take it easy on me", "easy on the insults", "be nice to me",
                 "stop insulting me", "stop roasting me", "go gentle" }) then
    return { cmd = "roast", delta = -1 }
  end
  if one_of(s, { "standard roast", "roast me normally", "normal roast", "medium roast", "roast level standard", "standard roast level" }) then return { cmd = "roast", level = "standard" } end
  if one_of(s, { "gentle roast", "roast level gentle", "roast level low" }) then return { cmd = "roast", level = "gentle" } end
  if one_of(s, { "savage roast", "roast level savage", "roast level high", "roast level max" }) then return { cmd = "roast", level = "savage" } end
  if one_of(s, { "roast me", "insult me", "say something mean", "say something mean to me", "insult me please" }) then return { cmd = "roast", now = true } end
  if one_of(s, { "what do you remember", "what do you remember about me", "what did i do today", "what have i done", "what have i done today",
                 "what did i do", "remind me what i did", "what have i been up to", "what have we done", "what have we done today",
                 "what do you remember about today", "recap", "recap today", "what do you know about me" }) then return { cmd = "recall" } end
  if one_of(s, { "forget that", "forget it", "forget the last thing", "forget what i just did" }) then return { cmd = "forget", scope = "that" } end
  if one_of(s, { "forget today", "forget about today", "forget everything from today", "forget today s stuff", "forget what i did today" }) then return { cmd = "forget", scope = "today" } end
  if one_of(s, { "forget everything", "forget all of it", "forget all that", "forget what you remember", "forget what you know about me", "forget me",
                 "forget everything about me", "wipe your notes", "clear your notes", "delete your notes", "delete the journal", "clear the journal" }) then
    return { cmd = "forget", scope = "all" }
  end
  if one_of(s, { "stop remembering", "stop taking notes", "stop keeping notes", "dont remember", "don t remember", "stop keeping a journal",
                 "turn off your journal", "no notes", "stop remembering me", "stop writing things down", "journal off", "no more notes" }) then return { cmd = "journal", on = false } end
  if one_of(s, { "remember again", "start remembering", "start taking notes", "turn on your journal", "keep notes", "keep notes again", "journal on", "you can remember again", "start writing things down" }) then
    return { cmd = "journal", on = true }
  end
  return nil
end

--- One step up or down the roast ladder.
D.ROAST_LEVELS = { "gentle", "standard", "savage" }
function D.roast_step(level, delta)
  local i = 1
  for k, v in ipairs(D.ROAST_LEVELS) do if v == level then i = k end end
  i = math.max(1, math.min(#D.ROAST_LEVELS, i + (delta or 0)))
  return D.ROAST_LEVELS[i]
end

return D
