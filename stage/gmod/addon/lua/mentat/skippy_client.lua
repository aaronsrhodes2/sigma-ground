-- skippy_client.lua -- Skippy in the player's ears and mouth.
--
--   * Skippy speaks: the server sends a short mp3 (made by the Skippy bridge next to the server); this plays it in 3D from where
--     Skippy built things, so everyone near hears it from that spot and nobody needs any extra software to listen.
--   * The player speaks: while the player is transmitting on voice chat, this writes data/mentat/ptt.txt. The optional
--     `skippy_ears.py` (on the player's own PC) records the microphone ONLY while that file says "transmitting", has the speech
--     recognised by the local Whisper, and hands back only sentences that start with the wake word "Skippy" as
--     data/mentat/voice_in.json. This file sends those to the server like a chat request. Everything else the player says to
--     other players is neither kept nor sent anywhere.
--   * `+skippy` (bind a key to it) is a push-to-talk for Skippy alone: it records without opening voice chat, no wake word needed.
if not CLIENT then return end

local DIR = "mentat"
file.CreateDir(DIR)

local cv_voice = CreateClientConVar("skippy_voice", "1", true, false, "1 = play Skippy's spoken answers")
local cv_volume = CreateClientConVar("skippy_volume", "1", true, false, "loudness of Skippy's voice, 0 to 1")
local cv_debug = CreateClientConVar("skippy_debug", "0", false, false, "1 = leave a note in data/mentat when Skippy's voice starts playing (for tests)")

-- ------------------------------------------------------------------------------------------------ the player speaks
local transmitting, direct = false, false

local function write_ptt()
  local on = transmitting or direct
  file.Write(DIR .. "/ptt.txt", (on and "1 " or "0 ") .. string.format("%.2f", SysTime()) .. (direct and " direct" or ""))
end
write_ptt()

-- the voice key (and open-mic activation) tell us when the player is talking
hook.Add("PlayerBindPress", "Mentat.SkippyPTT", function(ply, bind, pressed)
  if bind:find("+voicerecord", 1, true) then transmitting = pressed; write_ptt() end
  if bind:find("-voicerecord", 1, true) then transmitting = false; write_ptt() end
end)
hook.Add("PlayerStartVoice", "Mentat.SkippyPTT", function(ply)
  if ply == LocalPlayer() and not transmitting then transmitting = true; write_ptt() end
end)
hook.Add("PlayerEndVoice", "Mentat.SkippyPTT", function(ply)
  if ply == LocalPlayer() and transmitting then transmitting = false; write_ptt() end
end)
timer.Create("Mentat.SkippyPTTWatch", 0.1, 0, function()
  local me = LocalPlayer()
  if IsValid(me) and me.IsSpeaking and not direct then
    local now = me:IsSpeaking()
    if now ~= transmitting then transmitting = now; write_ptt() end
  end
end)

-- bind <key> +skippy : talk to Skippy only (no wake word, other players hear nothing)
concommand.Add("+skippy", function() direct = true; write_ptt() end)
concommand.Add("-skippy", function() direct = false; write_ptt() end)

-- what the microphone bridge heard
timer.Create("Mentat.SkippyHearing", 0.25, 0, function()
  local path = DIR .. "/voice_in.json"
  if not file.Exists(path, "DATA") then return end
  local raw = file.Read(path, "DATA")
  file.Delete(path)
  local obj = util.JSONToTable(raw or "")
  if obj and isstring(obj.text) and obj.text ~= "" then
    net.Start("holodeck_voice")
    net.WriteString(string.sub(obj.text, 1, 300))
    net.SendToServer()
    chat.AddText(Color(130, 130, 130), "[you, aloud] ", Color(230, 230, 230), obj.text)
  end
end)

-- ------------------------------------------------------------------------------------------------ Skippy speaks
local parts = {}

net.Receive("holodeck_audio", function()
  local id, idx, total = net.ReadUInt(16), net.ReadUInt(8), net.ReadUInt(8)
  local pos = net.ReadVector()
  local len = net.ReadUInt(16)
  local data = net.ReadData(len)
  if not cv_voice:GetBool() then return end
  local a = parts[id] or { chunks = {}, got = 0, pos = pos, born = CurTime() }
  parts[id] = a
  if not a.chunks[idx] then a.chunks[idx] = data; a.got = a.got + 1 end
  for k, v in pairs(parts) do if CurTime() - v.born > 30 then parts[k] = nil end end
  if a.got < total then return end
  parts[id] = nil
  local path = DIR .. "/skippy_" .. id .. ".mp3"
  file.Write(path, table.concat(a.chunks))
  sound.PlayFile("data/" .. path, "3d mono", function(chan, errid, errname)
    if not IsValid(chan) then
      print("[skippy] could not play Skippy's voice: " .. tostring(errid) .. " " .. tostring(errname))
      return
    end
    chan:SetPos(a.pos)
    chan:Set3DFadeDistance(250, 2400)
    chan:SetVolume(math.Clamp(cv_volume:GetFloat(), 0, 1))
    chan:Play()
    hook.Run("SkippySpoke", chan)
    if cv_debug:GetBool() then file.Write(DIR .. "/skippy_played.txt", string.format("%d %.2f", id, chan:GetLength())) end
    timer.Simple(45, function() file.Delete(path) end)
  end)
end)
