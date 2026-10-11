-- voice_client.lua -- the player's side of proximity voice: a smooth fade near the edge of a voice's reach, a muffle through
-- walls, and the shout key. (Skippy's microphone bridge lives in skippy_client.lua.)
if not CLIENT then return end

local Core = include("mentat/voice/core.lua")

local cv_fade = CreateClientConVar("mentat_voice_fade", "1", true, false, "1 = voices fade smoothly toward the edge of their reach (0 = full volume for everyone the server lets you hear)")

-- the shout key:  bind <key> +mentat_shout
-- The server counts a shout for a moment at a time, so the client repeats it while the key is held.
local shout_held = false
local function send_shout(on)
  net.Start("mentat_voice_shout"); net.WriteBool(on); net.SendToServer()
end
concommand.Add("+mentat_shout", function() shout_held = true; send_shout(true) end)
concommand.Add("-mentat_shout", function() shout_held = false; send_shout(false) end)
timer.Create("Mentat.VoiceShoutRepeat", 0.5, 0, function() if shout_held then send_shout(true) end end)

-- other players' loudness
local scaled = {}
local function set_scale(ply, s)
  if ply.SetVoiceVolumeScale then ply:SetVoiceVolumeScale(s) end
  scaled[ply] = s
end

local function reset_all()
  for p, s in pairs(scaled) do
    if IsValid(p) then
      if s ~= 1 then set_scale(p, 1) end
    else
      scaled[p] = nil
    end
  end
end

timer.Create("Mentat.VoiceFade", 0.1, 0, function()
  local me = LocalPlayer()
  if not IsValid(me) then return end
  -- the server decides how voices carry: only in proximity mode do we fade them; global and off leave every voice at full
  if not cv_fade:GetBool() or GetGlobalString("mentat_voice_mode", "proximity") ~= "proximity" then reset_all(); return end
  local wall = GetGlobalFloat("mentat_voice_wallf", Core.DEFAULTS.wall)
  for _, p in ipairs(player.GetAll()) do
    if p ~= me and p:IsSpeaking() then
      local range = p:GetNWFloat("mentat_voice_range", Core.DEFAULTS.range)
      if range <= 0 then
        if scaled[p] and scaled[p] ~= 1 then set_scale(p, 1) end
      else
        local dist = me:GetPos():Distance(p:GetPos())
        local tr = util.TraceLine({ start = me:EyePos(), endpos = p:EyePos(), mask = MASK_SOLID_BRUSHONLY })
        local r = tr.Hit and range * wall or range
        set_scale(p, math.max(0.0, Core.gain(dist, r, Core.DEFAULTS.edge)))
      end
    elseif scaled[p] and scaled[p] ~= 1 then
      set_scale(p, 1)                         -- the setting survives between servers: always leave it at full
    end
  end
end)

hook.Add("ShutDown", "Mentat.VoiceReset", function()
  for p, _ in pairs(scaled) do if IsValid(p) then set_scale(p, 1) end end
end)
