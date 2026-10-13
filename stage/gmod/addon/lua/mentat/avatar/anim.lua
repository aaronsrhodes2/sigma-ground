-- avatar/anim.lua -- Skippy's body language: a layered, bounded animation state machine. PURE (no engine calls), shared by the client (which
-- draws it, avatar_client.lua + avatar/body_client.lua) and the server (which only names the cues). Tested offline (test_gmod_director.py).
--
-- The server decides WHAT happens ("hat tip now", "he is in an opera mood") and replicates a very small state on the bot (NetworkVars:
-- mentat_av_cue + mentat_av_cue_seq, mentat_av_mood, mentat_av_talking, mentat_av_look). The client turns that into motion every frame:
--   * a CUE is a short keyframed gesture (hat tip, flourish, conductor's sweep, shrug, ...): a function of the time since it started
--   * the MOOD sets the continuous style while he talks (how big the sway, how hard the head chatters, how far he leans in)
--   * the VOICE LEVEL drives the head chatter (the "jaw": the head mesh has no jaw part, so the whole head nods a few degrees with each
--     syllable) and the red tongue's brightness
--   * IDLE is a slow breathing sway so he is never a statue
-- Every channel is clamped to LIMITS, every cue ends at exactly zero, and nothing here moves a vertex: only part transforms.
local A = {}

A.CUES = { none = 0, hat_tip = 1, flourish = 2, conduct = 3, shrug = 4, emphasis = 5, sway = 6, bounce = 7, quizzical = 8, nod = 9 }
A.CUE_NAMES = {}
for name, id in pairs(A.CUES) do A.CUE_NAMES[id] = name end

--- seconds each cue lasts
A.DUR = { hat_tip = 1.4, flourish = 1.8, conduct = 4.5, shrug = 1.3, emphasis = 1.6, sway = 2.4, bounce = 0.9, quizzical = 1.8, nod = 0.9 }

--- hard limits per channel (degrees unless said): the whole body never moves farther than this from rest
A.LIMITS = {
  tilt_p = 14, tilt_r = 20,            -- the body tips about his middle
  spin = 30,                           -- turn about the vertical (a flourish)
  bounce = 0.08,                       -- metres up/down (scaled to his size by the caller)
  hat_tip = 1,                         -- 0..1: the hat lifts and tips forward (0 = on his head)
  hat_p = 30, hat_r = 14,              -- extra hat lean on top of the voice lean (body_client adds them to the nod)
  head_yaw = 22, head_pitch = 9, head_roll = 12,
  hand = 1,                            -- 0..1: how far the free hand swings out for emphasis / conducting
  hand_side = 1,                       -- -1..1: which way (conducting sweeps it from side to side)
}
A.JAW_DEG = 4.5                        -- degrees of head pitch at full voice level (the "jaw" chatter)
A.CHATTER_HZ = 7                       -- syllables a second

--- the continuous style of each mood (while he talks): sway amplitude (deg), sway rate (rad/s), chatter gain, lean forward (deg)
A.MOODS = {
  opera = { sway = 12, rate = 2.2, chatter = 1.0, lean = -3, bounce = 0.02 },
  boast = { sway = 3, rate = 1.4, chatter = 0.9, lean = -4, bounce = 0.0 },     -- chest out: leans back a little
  roast = { sway = 2, rate = 1.8, chatter = 0.9, lean = 5, bounce = 0.0 },       -- leans in
  helpful = { sway = 2, rate = 1.2, chatter = 0.8, lean = 2, bounce = 0.0 },
  musing = { sway = 4, rate = 0.9, chatter = 0.7, lean = 0, bounce = 0.0 },
  exclaim = { sway = 3, rate = 3.0, chatter = 1.0, lean = 1, bounce = 0.015 },
}

local function clamp(x, a, b) return x < a and a or (x > b and b or x) end
local function smooth(x) x = clamp(x, 0, 1); return x * x * (3 - 2 * x) end                    -- 0..1 ease
local function bell(x) x = clamp(x, 0, 1); local s = math.sin(math.pi * x); return s * s end   -- 0 -> 1 -> 0, exactly 0 at both ends
local function zero() return { tilt_p = 0, tilt_r = 0, spin = 0, bounce = 0, hat_tip = 0, hat_p = 0, hat_r = 0, head_yaw = 0, head_pitch = 0, head_roll = 0, hand = 0, hand_side = 0 } end
A.zero = zero

--- The pose a cue asks for `t` seconds after it began (all zero before 0 and after its duration).
function A.cue_pose(cue, t)
  local p = zero()
  local name = type(cue) == "number" and A.CUE_NAMES[cue] or cue
  local dur = name and A.DUR[name]
  if not dur or t < 0 or t >= dur then return p end
  local u = t / dur
  if name == "hat_tip" then
    -- the hat lifts and tips forward, holds a beat, comes back; a little bow with it
    local k = u < 0.3 and smooth(u / 0.3) or (u < 0.65 and 1 or 1 - smooth((u - 0.65) / 0.35))
    p.hat_tip, p.hat_p, p.tilt_p, p.head_pitch = k, 28 * k, 6 * k, 4 * k
  elseif name == "flourish" then
    -- a sweep: the hand throws out, the body turns a little and rises, then settles
    local k = bell(u)
    p.hand, p.hand_side, p.spin, p.bounce, p.tilt_r = k, (math.sin(u * math.pi * 2) * 0.6 + 0.4) * k, 18 * math.sin(u * math.pi * 2), 0.04 * k, -8 * k
  elseif name == "conduct" then
    -- a conductor: the hand sweeps side to side for the whole aria, the body sways with it, the head follows the beat
    local env = bell(u) ^ 0.5
    local ph = u * math.pi * 2 * 3.5
    p.hand, p.hand_side, p.tilt_r, p.spin, p.head_roll, p.bounce = env, env * math.sin(ph), 15 * env * math.sin(ph), 16 * env * math.sin(ph + 0.8), 9 * env * math.sin(ph), 0.025 * env * math.abs(math.sin(ph))
  elseif name == "shrug" then
    local k = bell(u)
    p.bounce, p.head_roll, p.tilt_r, p.hand, p.hand_side, p.head_pitch = 0.025 * k, 8 * k, 4 * k, 0.5 * k, 0.3 * k, -3 * k
  elseif name == "emphasis" then
    local k = bell(u)
    p.hand, p.hand_side, p.tilt_p, p.head_pitch, p.bounce = k, 0.2 * k, 7 * k, 5 * k, 0.015 * k
  elseif name == "sway" then
    local k = bell(u)
    p.tilt_r, p.head_roll, p.spin = 9 * k * math.sin(u * math.pi * 2 * 1.5), 4 * k * math.sin(u * math.pi * 2 * 1.5 + 1), 6 * k * math.sin(u * math.pi * 2 * 1.5 + 0.5)
  elseif name == "bounce" then
    local k = math.abs(math.sin(u * math.pi * 2)) * (1 - u)
    p.bounce, p.tilt_p = 0.05 * k, -4 * k
  elseif name == "quizzical" then
    local k = bell(u)
    p.head_roll, p.head_pitch, p.tilt_r, p.hat_r = 11 * k, -3 * k, 3 * k, 6 * k
  elseif name == "nod" then
    local k = bell(u)
    p.head_pitch, p.hat_p = 8 * k, 6 * k
  end
  for ch, lim in pairs(A.LIMITS) do p[ch] = clamp(p[ch], -lim, lim) end
  return p
end

-- ------------------------------------------------------------------------------------------------ the layered state machine
--- A new state. Feed it with step() every frame.
function A.new()
  return { pose = zero(), cue = nil, cue_seq = 0, cue_t0 = 0, queued = nil, mood = "musing", level = 0, jaw = 0, tongue = 0.55, idle_t = 0, talk = 0, style = A.MOODS.musing }
end

--- inputs = { now (seconds), dt, cue (id or name, from the server), cue_seq (counts up with each new cue), mood (name), talking (bool), level (0..1) }
--- A new cue_seq starts that cue at once (a cue still playing is replaced, so a flourish is never late); the layers are summed and clamped.
function A.step(st, inp)
  local dt = clamp(inp.dt or 0.016, 0, 0.1)
  local now = inp.now or 0
  if inp.mood and A.MOODS[inp.mood] then st.mood = inp.mood end
  if inp.cue_seq and inp.cue_seq ~= st.cue_seq then
    st.cue_seq = inp.cue_seq
    local name = type(inp.cue) == "number" and A.CUE_NAMES[inp.cue] or inp.cue
    st.cue = (name and A.DUR[name]) and name or nil
    st.cue_t0 = now
  end
  if st.cue and now - st.cue_t0 >= A.DUR[st.cue] then st.cue = nil end
  -- talking eases in and out so the style does not snap
  st.talk = st.talk + ((inp.talking and 1 or 0) - st.talk) * (1 - math.exp(-5 * dt))
  st.level = st.level + (clamp(inp.level or 0, 0, 1) - st.level) * (1 - math.exp(-(inp.level and inp.level > st.level and 22 or 8) * dt))
  st.idle_t = st.idle_t + dt
  local style = A.MOODS[st.mood] or A.MOODS.musing
  local p = st.cue and A.cue_pose(st.cue, now - st.cue_t0) or zero()
  -- layer 1: idle breathing, always
  local breath = math.sin(st.idle_t * 1.1)
  p.tilt_r = p.tilt_r + 1.2 * breath
  p.tilt_p = p.tilt_p + 0.7 * math.sin(st.idle_t * 0.8 + 1)
  -- layer 2: the mood's style while he talks
  local t = st.talk
  p.tilt_r = p.tilt_r + t * style.sway * math.sin(st.idle_t * style.rate)
  p.tilt_p = p.tilt_p + t * style.lean
  p.bounce = p.bounce + t * style.bounce * math.abs(math.sin(st.idle_t * style.rate * 1.7))
  -- layer 3: the voice: head chatter (the jaw), tongue brightness
  local chatter = 0.5 + 0.5 * math.sin(st.idle_t * 2 * math.pi * A.CHATTER_HZ)
  st.jaw = st.level * style.chatter * chatter
  p.head_pitch = p.head_pitch + A.JAW_DEG * st.jaw
  st.tongue = 0.55 + 0.45 * clamp(st.level, 0, 1)
  for ch, lim in pairs(A.LIMITS) do p[ch] = clamp(p[ch], -lim, lim) end
  st.pose = p
  return p
end

return A
