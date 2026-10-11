-- fx/wisp.lua -- the PURE half of the failure wisp (no engine calls; tested offline with lupa). Shared: the server runs the state machine
-- (it sees where the owner looks), the client turns a state into a picture. See stage/gmod/holodeck/README.md ("Failure wisp").
--
-- What a wisp is: when a found-object transport FAILS, Skippy's honest reason is left behind as a small floating light at about eye height
-- at the failed spot. The owner who asked reads it by walking up and looking at it; after a dwell proportional to the text it notices
-- ("I saw you read it"), the text dissolves into sparks and the wisp goes. Never read, it fades by itself after a couple of minutes.
--
--   state machine   appear -> idle <-> reading -> dissolving -> gone        (W.new, W.step)
--   dwell           max(2.5 s, characters / 18 per second): reading time counts ONLY while it is looked at up close
--   colour          by reason class: failure amber, suggest cyan, refusal red (the project's palette), "ok" green
--   placement       W.victims: which existing wisps a new one replaces (a close one of the same owner, then the oldest over the cap)
--   text            W.cap_text (what the server stores), W.wrap (3-4 lines for the floating text)
--   picture         W.look(state, since, ...): the alphas of orb, text and sparks for a state
local W = {}

W.STATES = { "appear", "idle", "reading", "dissolving", "gone" }
W.STATE_INDEX = {}
for i, s in ipairs(W.STATES) do W.STATE_INDEX[s] = i end

W.CLASSES = { "failure", "suggest", "refusal", "ok" }
W.CLASS_INDEX = {}
for i, c in ipairs(W.CLASSES) do W.CLASS_INDEX[c] = i end
W.COLOURS = {                                   -- the project's palette (CLAUDE.md "Passthrough UI Color Doctrine")
  failure = { 255, 170, 0 },                    -- #FFAA00 amber: a real failure (not found, fetch or decode failed, timeout)
  suggest = { 0, 204, 255 },                    -- #00CCFF cyan: a neutral suggestion ("the closest I have is ...")
  refusal = { 255, 68, 68 },                    -- #FF4444 red: a refusal (too big for the room, not shareable)
  ok = { 0, 255, 136 },                         -- #00FF88 green
}

W.MAX_TEXT = 160             -- characters of a reason that are stored and sent
W.APPEAR_S = 1.2             -- the fade-in after the failure collapse
W.DISSOLVE_S = 1.0           -- the sparks after a read
W.EXPIRE_S = 2.5             -- the slow fade of a wisp nobody read
W.MIN_DWELL = 2.5            -- seconds of looking, at least
W.CPS = 18                   -- characters a reader is given per second of dwell
W.GRACE_S = 0.4              -- a glance away this short does not fold the text up (the dwell still only counts while looking)
W.LIFE_S = 120               -- never read: gone after this long (convar holodeck_wisp_life)
W.MAX_DIST = 210             -- Hammer units: 4 m, how close the owner must be to read
W.MIN_DOT = 0.88             -- cos of the half-angle (about 28 degrees) of "looking toward it"
W.JOIN_DIST = 79             -- Hammer units: 1.5 m, a new wisp this close to an old one of the same owner replaces it
W.CAP = 3                    -- wisps per owner
W.FAR = 1312                 -- Hammer units: 25 m, nothing is drawn beyond this
W.HU_M = 0.01905

local function clamp01(x)
  if x ~= x then return 0 end
  return x < 0 and 0 or (x > 1 and 1 or x)
end
local function smooth(x) x = clamp01(x); return x * x * (3 - 2 * x) end
W.clamp01, W.smooth = clamp01, smooth

-- ------------------------------------------------------------------------------------------------ colour and class
--- {r, g, b} for a class (a copy); an unknown class is a failure's amber.
function W.colour(class)
  local c = W.COLOURS[class] or W.COLOURS.failure
  return { c[1], c[2], c[3] }
end

--- The class of a reason, from the bridge's resolver status (or the glue's own tag). Real failures are amber, "the closest I have" is a
--- neutral suggestion (cyan), refusals are red.
function W.class_for(tag)
  tag = tostring(tag or ""):lower()
  if tag == "closest" or tag == "ambiguous" or tag == "category" or tag == "suggest" then return "suggest" end
  if tag == "too_big" or tag == "untrusted_size" or tag == "not_shareable" or tag == "refusal" or tag == "too_small" then return "refusal" end
  return "failure"
end

-- ------------------------------------------------------------------------------------------------ text
--- What the server stores and networks: one line, control characters gone, single spaces, at most `n` characters. A longer sentence is
--- cut at a word and ends in "..." (the full sentence was also said in chat), never inside a multi-byte character.
function W.cap_text(s, n)
  n = n or W.MAX_TEXT
  s = tostring(s or ""):gsub("[%cW]", " "):gsub("%s+", " "):match("^%s*(.-)%s*$") or ""
  if #s <= n then return s end
  local cut = s:sub(1, n - 3)
  local sp
  for i = #cut, math.max(1, #cut - 40), -1 do
    if cut:byte(i) == 32 then sp = i; break end
  end
  if sp then cut = cut:sub(1, sp - 1) end
  while #cut > 0 do                                           -- do not leave half a multi-byte character at the cut
    local i = #cut
    while i > 1 and cut:byte(i) >= 0x80 and cut:byte(i) < 0xC0 do i = i - 1 end
    local lead = cut:byte(i)
    local need = lead >= 0xF0 and 4 or (lead >= 0xE0 and 3 or (lead >= 0xC0 and 2 or 1))
    if lead < 0x80 or #cut - i + 1 >= need then break end
    cut = cut:sub(1, i - 1)
  end
  return (cut:gsub("[%s,;:%-]+$", "")) .. "..."
end

--- Seconds of looking before the wisp notices it was read: max(MIN_DWELL, characters / CPS).
function W.dwell(text)
  return math.max(W.MIN_DWELL, #tostring(text or "") / W.CPS)
end

--- Word-wrap `text` into at most `maxlines` lines (default 4), as few columns as it takes to fit in three. Returns the lines and the
--- widest line in characters. A word longer than a line is broken; text that still does not fit ends in "...".
function W.wrap(text, maxlines, maxcols)
  maxlines, maxcols = maxlines or 4, maxcols or 40
  text = tostring(text or "")
  local function try(cols, limit)
    local lines, cur = {}, ""
    for word in text:gmatch("%S+") do
      while #word > cols do                                 -- a very long word is cut at the column
        if cur ~= "" then lines[#lines + 1] = cur; cur = "" end
        lines[#lines + 1] = word:sub(1, cols)
        word = word:sub(cols + 1)
      end
      if cur == "" then cur = word
      elseif #cur + 1 + #word <= cols then cur = cur .. " " .. word
      else lines[#lines + 1] = cur; cur = word end
    end
    if cur ~= "" then lines[#lines + 1] = cur end
    return lines, #lines <= limit
  end
  local lines, ok
  for cols = 22, maxcols do
    lines, ok = try(cols, math.min(3, maxlines))
    if ok then break end
  end
  if not ok then lines, ok = try(maxcols, maxlines) end
  if not ok then
    local kept = {}
    for i = 1, maxlines do kept[i] = lines[i] end
    local last = kept[maxlines]
    kept[maxlines] = (#last > maxcols - 3 and last:sub(1, maxcols - 3) or last) .. "..."
    lines = kept
  end
  local widest = 0
  for _, l in ipairs(lines) do if #l > widest then widest = #l end end
  return lines, widest
end

-- ------------------------------------------------------------------------------------------------ the state machine
--- A fresh wisp. `text` fixes the dwell; opts.life overrides LIFE_S (seconds before a never-read wisp fades).
function W.new(text, opts)
  opts = opts or {}
  return { state = "appear", age = 0, since = 0, read = 0, last_look = -1e9, dwell = W.dwell(text), life = tonumber(opts.life) or W.LIFE_S, why = nil }
end

local function enter(st, state)
  st.state, st.since = state, 0
end

--- Advance by dt seconds. `looking`: the owner is within MAX_DIST, looking toward it, with nothing in between (W.sees). Returns the state.
function W.step(st, dt, looking)
  if st.state == "gone" then return "gone" end
  dt = math.max(0, tonumber(dt) or 0)
  st.age = st.age + dt
  st.since = st.since + dt
  if st.state == "appear" then
    if st.since >= W.APPEAR_S then enter(st, "idle") end
  end
  if st.state == "idle" or st.state == "reading" then
    if looking then
      st.last_look = st.age
      st.read = st.read + dt                                 -- the reading clock runs only while it is looked at up close
      if st.state == "idle" then enter(st, "reading") end
    elseif st.state == "reading" and st.age - st.last_look > W.GRACE_S then
      enter(st, "idle")
    end
    if st.read >= st.dwell then
      st.why = "read"
      enter(st, "dissolving")
    elseif st.age >= st.life then
      st.why = "expired"
      enter(st, "dissolving")
    end
  elseif st.state == "dissolving" then
    if st.since >= (st.why == "expired" and W.EXPIRE_S or W.DISSOLVE_S) then enter(st, "gone") end
  end
  return st.state
end

--- The owner dismisses it at once (the "clear the wisps" command): it dissolves like a read one.
function W.dismiss(st)
  if st.state == "gone" or st.state == "dissolving" then return end
  st.why = "dismissed"
  enter(st, "dissolving")
end

-- ------------------------------------------------------------------------------------------------ who can read it
--- Is the reader looking at the wisp from close enough? eye, fwd (unit), pos: {x,y,z}-like; `clear` is the trace result (nothing between
--- the eye and the wisp). Returns true/false and the distance.
function W.sees(eye, fwd, pos, clear, max_dist, min_dot)
  max_dist, min_dot = max_dist or W.MAX_DIST, min_dot or W.MIN_DOT
  local dx, dy, dz = pos.x - eye.x, pos.y - eye.y, pos.z - eye.z
  local d = math.sqrt(dx * dx + dy * dy + dz * dz)
  if d > max_dist then return false, d end
  if d < 1e-6 then return clear ~= false, d end
  local dot = (dx * fwd.x + dy * fwd.y + dz * fwd.z) / d
  if dot < min_dot then return false, d end
  return clear ~= false, d
end

-- ------------------------------------------------------------------------------------------------ replacing and capping
--- Which of `list` ({ { owner =, pos = {x,y,z}, born = } ... }) a new wisp of `owner` at `pos` removes: every wisp of the same owner
--- within JOIN_DIST (it replaces the near one), then the oldest of that owner until there is room under `cap`. Returns indices into list.
function W.victims(list, owner, pos, cap, join)
  cap, join = cap or W.CAP, join or W.JOIN_DIST
  local out, taken = {}, {}
  for i, w in ipairs(list) do
    if w.owner == owner then
      local dx, dy, dz = w.pos.x - pos.x, w.pos.y - pos.y, w.pos.z - pos.z
      if dx * dx + dy * dy + dz * dz <= join * join then out[#out + 1] = i; taken[i] = true end
    end
  end
  local mine = {}
  for i, w in ipairs(list) do if w.owner == owner and not taken[i] then mine[#mine + 1] = i end end
  table.sort(mine, function(a, b) return list[a].born < list[b].born end)
  local over = #mine + 1 - cap                               -- the new one counts
  for k = 1, math.max(0, over) do out[#out + 1] = mine[k] end
  return out
end

-- ------------------------------------------------------------------------------------------------ the picture
--- What to draw for a state. `since`: seconds in it. `unfurl`: the client's own 0..1 ease of the text (0 folded, 1 open).
--- `mine`: this viewer is the owner (the others see a dim light and no text). Returns
---   orb     overall brightness of the light 0..1
---   size    scale of the light (1 = rest)
---   text    opacity of the floating text 0..1
---   sparks  0..1: the dissolve, the text turning to sparks (read or dismissed)
---   fade    0..1 how far gone the light is (the whole thing, for the last moments)
function W.look(state, since, unfurl, mine, why)
  local orb, size, text, sparks, fade = 0, 1, 0, 0, 0
  if state == "appear" then
    orb = smooth(since / W.APPEAR_S); size = 0.4 + 0.6 * orb
  elseif state == "idle" then
    orb, size = 0.55, 1
  elseif state == "reading" then
    orb, size = 0.95, 1.12
    text = smooth(unfurl or 0)
  elseif state == "dissolving" then
    if why == "expired" then
      fade = smooth(since / W.EXPIRE_S); orb = 0.55 * (1 - fade); size = 1 - 0.3 * fade
    else
      local k = clamp01(since / W.DISSOLVE_S)
      sparks = k
      text = (mine and (unfurl or 0) or 0) * (1 - smooth(k * 1.4))
      orb = 0.95 * (1 - smooth(clamp01((k - 0.3) / 0.7))); size = 1.12 + 0.5 * k
      fade = smooth(clamp01((k - 0.4) / 0.6))
    end
  end
  if not mine then orb = orb * 0.4; text = 0 end             -- other players see it dimly and cannot read it
  return { orb = orb, size = size, text = text, sparks = sparks, fade = fade }
end

--- Where the light hangs: a slow bob and drift around its place (t seconds, phase 0..1 per wisp). Returns dx, dy, dz in Hammer units.
function W.bob(t, phase)
  local p = (phase or 0) * 6.2831853
  return math.sin(t * 0.37 + p) * 2.2, math.cos(t * 0.29 + p * 1.7) * 2.2, math.sin(t * 0.8 + p) * 2.6
end

--- How fast the pulse beats: the light breathes slowly at rest, a little faster while it is read.
function W.pulse(t, phase, reading)
  local f = reading and 1.6 or 0.9
  return 0.85 + 0.15 * math.sin(t * f * 2.0 + (phase or 0) * 6.2831853)
end

return W
