-- core.lua -- the numbers behind proximity voice. Pure Lua (no GMod globals), unit-tested offline.
--
-- A voice carries a distance that depends on how the talker is speaking (crouched = a whisper, holding the shout key = a
-- shout) and is muffled by walls. Inside the range it is heard; the last part of the range fades smoothly so a voice
-- dies away instead of cutting off.
local V = {}

V.DEFAULTS = {
  range = 1100,       -- HU a normal voice carries (about 21 m)
  whisper = 0.35,     -- crouching: a fraction of the range
  shout = 2.2,        -- the shout key: a multiple of the range
  wall = 0.55,        -- through a wall or floor: a fraction of the range
  edge = 0.6,         -- the fade starts at this fraction of the range
}

local function merged(opts)
  local o = {}
  for k, v in pairs(V.DEFAULTS) do o[k] = v end
  for k, v in pairs(opts or {}) do o[k] = v end
  return o
end

--- How far this talker's voice carries right now.
function V.effective_range(opts, state)
  local o = merged(opts)
  local r = o.range
  if state and state.shouting then r = r * o.shout
  elseif state and state.crouching then r = r * o.whisper end
  return r
end

--- Loudness 0..1 at a distance: full up to `edge` of the range, then a smooth cosine fade to silence at the range.
function V.gain(dist, range, edge)
  edge = edge or V.DEFAULTS.edge
  if range <= 0 or dist >= range then return 0 end
  local start = range * edge
  if dist <= start then return 1 end
  local t = (dist - start) / (range - start)
  return 0.5 * (1 + math.cos(math.pi * t))
end

--- Can a listener at `dist` hear a talker whose voice carries `range`? Walls shrink the range. Returns (heard, gain).
function V.decide(dist, range, occluded, opts)
  local o = merged(opts)
  local r = occluded and range * o.wall or range
  local g = V.gain(dist, r, o.edge)
  return g > 0, g
end

return V
