-- voice.lua -- proximity voice, native to the server: players hear each other by distance, with no external software.
--
-- The engine already carries every player's microphone to the server; this rule decides, per listener and talker, whether
-- the voice is passed on and whether it is positional (3D, fading with distance and panned in stereo). The range depends on
-- how the talker is speaking (crouch = whisper, shout key = shout), and walls between the two shrink it. A player who is not
-- speaking costs nothing (the engine only asks while someone talks). See voice/core.lua for the numbers.
--
-- The rule is installed as the gamemode's own PlayerCanHearPlayersVoice, wrapped around what the gamemode already had, NOT
-- as a competing hook: hooks win by whichever answers first, so a gag or mute addon (which returns false from its hook)
-- could be overridden by ours. Wrapped like this, every hook gets its say first, and the gamemode's own refusal (team
-- voice, sv_alltalk 0) still holds; ours only decides what remains.
--
-- Convars:  mentat_voice_mode  proximity | global | off   (default proximity; global = everyone hears everyone, flat; off = the
--                                                          game's own rules, untouched)
--           mentat_voice_range HU a normal voice carries   (default 1100)
--           mentat_voice_wall  1 = walls shrink the range  (default 1)
-- A client may hold +mentat_shout to carry farther (the server remembers it for a second at a time, so a lost key-up cannot
-- leave a player shouting for ever). A player can turn the smooth fade off for themselves with mentat_voice_fade 0.
if not SERVER then return end

local Core = include("mentat/voice/core.lua")
Mentat = Mentat or {}
Mentat.Voice = Core
Mentat.VoiceWrappers = Mentat.VoiceWrappers or setmetatable({}, { __mode = "k" })

util.AddNetworkString("mentat_voice_shout")

local cv_mode = CreateConVar("mentat_voice_mode", "proximity", FCVAR_ARCHIVE + FCVAR_NOTIFY, "proximity | global | off")
local cv_range = CreateConVar("mentat_voice_range", tostring(Core.DEFAULTS.range), FCVAR_ARCHIVE + FCVAR_NOTIFY, "HU a normal voice carries")
local cv_wall = CreateConVar("mentat_voice_wall", "1", FCVAR_ARCHIVE, "1 = walls between talker and listener shrink the range")

local SHOUT_HOLD = 1.5       -- seconds a shout lasts after the client last said so (the client repeats it while the key is down)
local shouting = {}          -- player -> CurTime() until which they count as shouting
local cache = {}             -- "listener|talker" -> { t, heard }

net.Receive("mentat_voice_shout", function(_, ply)
  shouting[ply] = net.ReadBool() and (CurTime() + SHOUT_HOLD) or nil
end)

local function opts()
  return { range = cv_range:GetFloat() }
end

local function state_of(ply)
  return { crouching = ply:Crouching(), shouting = shouting[ply] ~= nil and shouting[ply] > CurTime() }
end

--- The reach of a talker's voice right now (HU).
function Core.TalkerRange(ply) return Core.effective_range(opts(), state_of(ply)) end

local function occluded(a, b)
  local tr = util.TraceLine({ start = a:GetShootPos(), endpos = b:GetShootPos(), mask = MASK_SOLID_BRUSHONLY })
  return tr.Hit
end

--- The decision for one pair, uncached: used by the rule and by tests.
function Core.CanHear(listener, talker)
  local o = opts()
  local range = Core.effective_range(o, state_of(talker))
  local dist = listener:GetPos():Distance(talker:GetPos())
  if dist > range then return false, 0 end                            -- beyond the unmuffled reach: no trace needed
  local blocked = cv_wall:GetBool() and occluded(listener, talker)
  return Core.decide(dist, range, blocked, o)
end

--- Forget remembered decisions (the rule remembers each pair for a quarter of a second).
function Core.ClearCache() cache = {} end

local MODES = { proximity = true, global = true, off = true }
local warned_mode

--- The voice mode, normalised ONCE for the server and the clients alike: lower case, trimmed, and anything that is not one of the
--- three names (a typo, "Global " with a stray space, an empty string) is proximity. Everything below reads the mode through this,
--- and publish() sends this value, so the rule, the clients' fade and the range they are told can never disagree.
local function mode()
  local raw = tostring(cv_mode:GetString() or "")
  local m = raw:lower():match("^%s*(.-)%s*$")
  if MODES[m] then return m end
  if warned_mode ~= raw then
    warned_mode = raw
    print("[mentat] mentat_voice_mode '" .. raw:sub(1, 40) .. "' is not proximity, global or off: using proximity")
  end
  return "proximity"
end
Core.Mode = mode

--- Mentat's verdict for one pair, in the modes that have one (global, proximity).
local function verdict(listener, talker)
  if mode() == "global" then return true, false end
  if listener == talker then return false end
  local key = listener:EntIndex() .. "|" .. talker:EntIndex()
  local now = CurTime()
  local c = cache[key]
  if c and now - c.t < 0.25 then return c.heard, c.heard end
  local heard = Core.CanHear(listener, talker)
  cache[key] = { t = now, heard = heard }
  return heard, heard
end

local current_wrapper      -- the wrapper we last put on the gamemode (compared on a timer: see the end of the file)

--- Wrap the gamemode's rule (idempotent: a second call, or a Lua refresh, replaces our wrapper and keeps the original).
local function install()
  local gm = GAMEMODE
  if not gm then return false end
  local current = gm.PlayerCanHearPlayersVoice
  local base = current
  if current and Mentat.VoiceWrappers[current] then base = gm.MentatVoiceBase else gm.MentatVoiceBase = current end
  local function rule(g, listener, talker)
    local heard, positional
    if base then heard, positional = base(g, listener, talker) end
    if mode() == "off" then return heard, positional end                 -- leave the game's own rules alone
    if heard == false then return false, false end                       -- the game (or a rule it runs) says no
    return verdict(listener, talker)
  end
  Mentat.VoiceWrappers[rule] = true
  gm.PlayerCanHearPlayersVoice = rule
  current_wrapper = rule
  return true
end

hook.Add("PostGamemodeLoaded", "Mentat.ProximityVoice", install)
timer.Simple(0, install)                                              -- a Lua refresh after the gamemode loaded

-- what the clients need to fade voices smoothly: every talker's reach (0 = do not fade: global and off modes), the wall factor
local function publish()
  local m = mode()
  local proximity = m == "proximity"
  SetGlobalString("mentat_voice_mode", m)
  SetGlobalFloat("mentat_voice_wallf", cv_wall:GetBool() and Core.DEFAULTS.wall or 1)
  for _, p in ipairs(player.GetAll()) do
    p:SetNWFloat("mentat_voice_range", proximity and Core.TalkerRange(p) or 0)
  end
end
cvars.AddChangeCallback("mentat_voice_mode", publish, "Mentat.VoiceMode")
cvars.AddChangeCallback("mentat_voice_wall", publish, "Mentat.VoiceWall")

-- An addon that loads later and assigns GAMEMODE.PlayerCanHearPlayersVoice takes our wrapper off (and with it the range, walls,
-- whisper and shout), and nothing else would notice. So the timer that already runs checks, with one pointer comparison, that the
-- gamemode's rule is still ours and wraps again when it is not. Wrapping is safe to repeat: the replacement becomes the base, and
-- if it is itself a wrapper around ours the pair settles after one round.
local warned_replaced
local function keep_installed()
  local gm = GAMEMODE
  if gm and gm.PlayerCanHearPlayersVoice ~= current_wrapper then
    if install() and not warned_replaced then
      warned_replaced = true
      print("[mentat] proximity voice: another addon replaced the gamemode's voice rule; wrapped it again")
    end
  end
end

timer.Create("Mentat.VoiceRange", 0.4, 0, function()
  keep_installed()
  publish()
  for k, c in pairs(cache) do
    if CurTime() - c.t > 5 then cache[k] = nil end
  end
end)

hook.Add("PlayerDisconnected", "Mentat.VoiceForget", function(ply) shouting[ply] = nil end)
hook.Add("PlayerSpawn", "Mentat.VoiceForgetShout", function(ply) shouting[ply] = nil end)
hook.Add("PlayerDeath", "Mentat.VoiceForgetShoutDeath", function(ply) shouting[ply] = nil end)

MsgC(Color(0, 255, 136), "[mentat] proximity voice: ", Color(255, 255, 255), cv_mode:GetString() .. ", range " .. cv_range:GetInt() .. " HU\n")
