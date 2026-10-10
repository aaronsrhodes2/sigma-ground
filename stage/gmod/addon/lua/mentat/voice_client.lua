-- voice_client.lua -- the player's side of proximity voice: a smooth fade near the edge of a voice's reach, a muffle through
-- walls, and the shout key. (Skippy's microphone bridge lives in skippy_client.lua.)
if not CLIENT then return end

local Core = include("mentat/voice/core.lua")

local cv_fade = CreateClientConVar("mentat_voice_fade", "1", true, false, "1 = voices fade smoothly toward the edge of their reach")

-- the shout key:  bind <key> +mentat_shout
concommand.Add("+mentat_shout", function()
  net.Start("mentat_voice_shout"); net.WriteBool(true); net.SendToServer()
end)
concommand.Add("-mentat_shout", function()
  net.Start("mentat_voice_shout"); net.WriteBool(false); net.SendToServer()
end)

-- other players' loudness
local scaled = {}
local function set_scale(ply, s)
  if ply.SetVoiceVolumeScale then ply:SetVoiceVolumeScale(s) end
  scaled[ply] = s
end

timer.Create("Mentat.VoiceFade", 0.1, 0, function()
  local me = LocalPlayer()
  if not IsValid(me) then return end
  if not cv_fade:GetBool() then
    for p, s in pairs(scaled) do if IsValid(p) and s ~= 1 then set_scale(p, 1) end end
    return
  end
  for _, p in ipairs(player.GetAll()) do
    if p ~= me and p:IsSpeaking() then
      local range = p:GetNWFloat("mentat_voice_range", Core.DEFAULTS.range)
      local dist = me:GetPos():Distance(p:GetPos())
      local tr = util.TraceLine({ start = me:EyePos(), endpos = p:EyePos(), mask = MASK_SOLID_BRUSHONLY })
      local r = tr.Hit and range * Core.DEFAULTS.wall or range
      set_scale(p, math.max(0.0, Core.gain(dist, r, Core.DEFAULTS.edge)))
    elseif scaled[p] and scaled[p] ~= 1 then
      set_scale(p, 1)                         -- the setting survives between servers: always leave it at full
    end
  end
end)

hook.Add("ShutDown", "Mentat.VoiceReset", function()
  for p, _ in pairs(scaled) do if IsValid(p) then set_scale(p, 1) end end
end)
