-- voice.lua -- proximity voice, native to the server: players hear each other by distance, with no external software.
--
-- The engine already carries every player's microphone to the server; this hook decides, per listener and talker, whether
-- the voice is passed on and whether it is positional (3D, fading with distance and panned in stereo). The range depends on
-- how the talker is speaking (crouch = whisper, shout key = shout), and walls between the two shrink it. A player who is not
-- speaking costs nothing (the engine only asks while someone talks). See voice/core.lua for the numbers.
--
-- Convars:  mentat_voice_mode  proximity | global | off   (default proximity)
--           mentat_voice_range HU a normal voice carries   (default 1100)
--           mentat_voice_wall  1 = walls shrink the range  (default 1)
-- A client may hold +mentat_shout to carry farther; the server keeps the state.
if not SERVER then return end

local Core = include("mentat/voice/core.lua")
Mentat = Mentat or {}
Mentat.Voice = Core

util.AddNetworkString("mentat_voice_shout")

local cv_mode = CreateConVar("mentat_voice_mode", "proximity", FCVAR_ARCHIVE + FCVAR_NOTIFY, "proximity | global | off")
local cv_range = CreateConVar("mentat_voice_range", tostring(Core.DEFAULTS.range), FCVAR_ARCHIVE + FCVAR_NOTIFY, "HU a normal voice carries")
local cv_wall = CreateConVar("mentat_voice_wall", "1", FCVAR_ARCHIVE, "1 = walls between talker and listener shrink the range")

local shouting = {}      -- player -> true while the shout key is held
local cache = {}         -- "listener|talker" -> { t, heard }

net.Receive("mentat_voice_shout", function(_, ply)
  shouting[ply] = net.ReadBool() or nil
end)

local function opts()
  return { range = cv_range:GetFloat() }
end

local function state_of(ply)
  return { crouching = ply:Crouching(), shouting = shouting[ply] }
end

--- The reach of a talker's voice right now (HU).
function Core.TalkerRange(ply) return Core.effective_range(opts(), state_of(ply)) end

local function occluded(a, b)
  local tr = util.TraceLine({ start = a:GetShootPos(), endpos = b:GetShootPos(), mask = MASK_SOLID_BRUSHONLY })
  return tr.Hit
end

--- The decision for one pair, uncached: used by the hook and by tests.
function Core.CanHear(listener, talker)
  local o = opts()
  local range = Core.effective_range(o, state_of(talker))
  local dist = listener:GetPos():Distance(talker:GetPos())
  if dist > range then return false, 0 end                            -- beyond the unmuffled reach: no trace needed
  local blocked = cv_wall:GetBool() and occluded(listener, talker)
  return Core.decide(dist, range, blocked, o)
end

--- Forget remembered decisions (the hook remembers each pair for a quarter of a second).
function Core.ClearCache() cache = {} end

hook.Add("PlayerCanHearPlayersVoice", "Mentat.ProximityVoice", function(listener, talker)
  local mode = cv_mode:GetString()
  if mode == "off" then return end                                    -- leave the game's own rules alone
  if mode == "global" then return true, false end
  if listener == talker then return false end
  local key = listener:EntIndex() .. "|" .. talker:EntIndex()
  local now = CurTime()
  local c = cache[key]
  if c and now - c.t < 0.25 then return c.heard, c.heard end
  local heard = Core.CanHear(listener, talker)
  cache[key] = { t = now, heard = heard }
  return heard, heard
end)

-- every player's reach, for the clients' smooth fade
timer.Create("Mentat.VoiceRange", 0.4, 0, function()
  for _, p in ipairs(player.GetAll()) do
    p:SetNWFloat("mentat_voice_range", Core.TalkerRange(p))
  end
  for k, c in pairs(cache) do
    if CurTime() - c.t > 5 then cache[k] = nil end
  end
end)

hook.Add("PlayerDisconnected", "Mentat.VoiceForget", function(ply) shouting[ply] = nil end)

MsgC(Color(0, 255, 136), "[mentat] proximity voice: ", Color(255, 255, 255), cv_mode:GetString() .. ", range " .. cv_range:GetInt() .. " HU\n")
