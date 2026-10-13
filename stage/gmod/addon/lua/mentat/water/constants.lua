-- water/constants.lua -- the construct's water as pure logic over the generated table (data.lua).  PURE Lua 5.1 / LuaJIT.
--
-- REAL-WORLD PHYSICS ONLY (Aaron, 2026-10-12): the liquids are the sigma-ground library's (density, viscosity, surface tension, each with
-- its source); a liquid the library does not have is REFUSED, not guessed.  The caps are not physics and say so.
local M = {}
local inc = include or function(p) return dofile(p) end
local data = inc("mentat/water/data.lua")
M.data = data
M.HU_M = data.hu_m
M.LIMITS = data.limits
M.DEFAULTS = data.defaults
M.LIQUIDS = data.liquids

local floor, min, max, pi, sqrt = math.floor, math.min, math.max, math.pi, math.sqrt

-- word -> library key
local word_map = {}
for key, l in pairs(data.liquids) do
  word_map[key] = key
  for _, w in ipairs(l.words) do word_map[w] = key end
end
local refused_map = {}
for _, w in ipairs(data.refused_words) do refused_map[w] = true end

--- What a word means as a liquid: key, nil  |  nil, "refused" (a liquid the library lacks)  |  nil, nil (not a liquid word).
function M.find_liquid(word)
  word = tostring(word or ""):lower():gsub("^%s+", ""):gsub("%s+$", "")
  if word_map[word] then return word_map[word], nil end
  if refused_map[word] then return nil, "refused" end
  return nil, nil
end

function M.liquid(key) return data.liquids[key] end

function M.refusal(word)
  return "I do not have " .. tostring(word) .. ": " .. data.refused_reason .. "."
end

M.WATER_KEY = "water"
M.REFUSED_REASON = data.refused_reason

--- Volume (m^3) of a body: rect = l * w * depth, round = pi/4 * l * w * depth (an ellipse or circle).
function M.volume(shape, l, w, depth)
  if shape == "round" then return pi / 4 * l * w * depth end
  return l * w * depth
end

local function clamp(x, a, b) return x < a and a or (x > b and b or x) end

--- Clamp a body's dimensions into the construct's caps.  Returns l, w, depth (m) and a list of what was changed (texts).
function M.clamp_dims(l, w, depth)
  local L = M.LIMITS
  local notes = {}
  local function c(name, v, lo, hi, what)
    local nv = clamp(v, lo, hi)
    if nv ~= v then notes[#notes + 1] = what .. " " .. string.format("%.3g", nv) .. " m (the construct's cap, not physics)" end
    return nv
  end
  l = c("l", l, L.min_footprint_m.value, L.max_footprint_m.value, "length")
  w = c("w", w, L.min_footprint_m.value, L.max_footprint_m.value, "width")
  depth = c("d", depth, L.min_depth_m.value, L.max_depth_m.value, "depth")
  return l, w, depth, notes
end

--- May a new (or resized) body of volume v stand next to the others?  bodies = list of { volume = m^3 }, `skip` = index replaced.
--- Returns true, or false and a sentence.
function M.room_for(bodies, v, skip)
  local L = M.LIMITS
  local n, total = 0, 0
  for i, b in ipairs(bodies) do
    if i ~= skip then n = n + 1; total = total + (b.volume or 0) end
  end
  if not skip and n >= L.max_bodies.value then
    return false, "There are already " .. n .. " bodies of water, which is as many as this server allows (a cap, not physics). Remove one first."
  end
  if total + v > L.max_volume_m3.value then
    return false, string.format("That would put %.0f cubic metres of water in the room; the cap is %.0f (a cap, not physics).", total + v, L.max_volume_m3.value)
  end
  return true
end

--- Sizes for a noun with the player's numbers (any nil = the default).  Returns { shape, l, w, depth, notes }.
function M.dims_for(noun, over)
  over = over or {}
  local key = noun
  if key == "ocean" then key = "sea" end
  if key == "swimming pool" then key = "pool" end
  local d = M.DEFAULTS[key] or M.DEFAULTS.pool
  local shape = d.shape
  local l, w, depth
  if shape == "round" then
    local dia = over.l or over.w or d.diameter_m
    l, w = dia, over.w and over.l and over.w or dia
    if over.l and over.w then l, w = over.l, over.w end
  else
    l = over.l or d.l_m
    w = over.w or (over.l and over.l * (d.w_m / d.l_m) or d.w_m)
  end
  depth = over.depth or d.depth_m
  if over.shape then shape = over.shape end
  local notes
  l, w, depth, notes = M.clamp_dims(l, w, depth)
  return { shape = shape, l = l, w = w, depth = depth, notes = notes, volume = M.volume(shape, l, w, depth) }
end

local function g3(x)
  if x == 0 then return "0" end
  local s = string.format("%.3g", x)
  if s:find("e", 1, true) then s = string.format("%.4f", x) end
  return s
end
M.g3 = g3

--- One sentence about a liquid, with its source.
function M.describe_liquid(key)
  local l = data.liquids[key]
  return string.format("%s: density %s kg/m3, viscosity %s Pa s, at %.0f K [%s] %s", l.label, g3(l.density_kg_m3), g3(l.viscosity_pa_s), l.T_K, l.status, l.source)
end

--- Does a body of density rho float in this liquid?  Returns floats (bool), the submerged fraction at rest (<= 1).
function M.floats(rho_body, key)
  local rho = data.liquids[key].density_kg_m3
  if rho_body < rho then return true, rho_body / rho end
  return false, 1
end

return M
