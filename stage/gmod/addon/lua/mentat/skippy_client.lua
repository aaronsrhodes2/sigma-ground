-- skippy_client.lua -- Skippy in the player's ears and mouth.
--
--   * Skippy speaks: the server sends a short mp3 (made by the Skippy bridge next to the server); this plays it in 3D from where
--     Skippy built things, so everyone near hears it from that spot and nobody needs any extra software to listen.
--   * The player speaks: while the player is transmitting on voice chat, this writes data/mentat/ptt.txt. The optional
--     `skippy_ears.py` (on the player's own PC) records the microphone ONLY while that file says "transmitting", has the speech
--     recognised by the local Whisper, and hands back only sentences that start with one of YOUR names for Skippy ("Skippy" unless you
--     changed it: the server sends your own list, see below) as data/mentat/voice_in.json. This file sends those to the server like a chat request. Everything else the player says to
--     other players is neither kept nor sent anywhere.
--   * `+skippy` (bind a key to it) is a push-to-talk for Skippy alone: it records without opening voice chat, no wake word needed.
if not CLIENT then return end

local DIR = "mentat"
file.CreateDir(DIR)

local cv_voice = CreateClientConVar("skippy_voice", "1", true, false, "1 = play Skippy's spoken answers")
local cv_volume = CreateClientConVar("skippy_volume", "1", true, false, "loudness of Skippy's voice, 0 to 1")
local cv_debug = CreateClientConVar("skippy_debug", "0", false, false, "1 = leave a note in data/mentat when Skippy's voice starts playing (for tests)")
-- Off by default and never saved: holding this key sends EVERYTHING the microphone hears to the server as a request (no wake word),
-- so it must be a key the player chose for it. The launcher (play.py) sets it each session; a key that already does something else
-- (a shout, a spawn menu, anything bound) is refused with a warning rather than turned into a way to transcribe what is shouted at players.
local cv_key = CreateClientConVar("skippy_key", "", false, false, "the key you hold to talk to Skippy alone (no wake word; EVERYTHING you say while it is down goes to the server); empty = none")

-- ------------------------------------------------------------------------------------------------ the player speaks
local transmitting, direct, direct_cmd, direct_key = false, false, false, false

local write_ptt
local function set_direct()
  local d = direct_cmd or direct_key
  if d ~= direct then direct = d; write_ptt() end
end

function write_ptt()
  local on = transmitting or direct
  file.Write(DIR .. "/ptt.txt", (on and "1 " or "0 ") .. string.format("%.2f", SysTime()) .. (direct and " direct" or ""))
end
write_ptt()

-- false (and one warning) when the player has given this key another job
local warned = {}
local function key_is_free(name, code)
  if not input.LookupKeyBinding then return true end
  local cmd = input.LookupKeyBinding(code)
  if not cmd or cmd == "" or cmd:find("skippy", 1, true) then return true end
  local tag = name .. ">" .. cmd
  if not warned[tag] then
    warned[tag] = true
    local msg = "[skippy] skippy_key " .. name:upper() .. " is already bound to '" .. cmd .. "', so it is ignored; pick a free key with skippy_key"
    print(msg)
    if chat and chat.AddText then chat.AddText(Color(255, 170, 0), msg) end
  end
  return false
end

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
local last_beat = 0
timer.Create("Mentat.SkippyPTTWatch", 0.1, 0, function()
  local me = LocalPlayer()
  if IsValid(me) and me.IsSpeaking and not direct then
    local now = me:IsSpeaking()
    if now ~= transmitting then transmitting = now; write_ptt() end
  end
  -- the key that talks to Skippy alone: read here rather than bound, because the server is not allowed to bind keys for a player
  local name = cv_key:GetString()
  local code = (name ~= "") and input.GetKeyCode(name) or 0
  local down = false
  if code and code > 0 and IsValid(me) and key_is_free(name, code) then
    down = input.IsKeyDown(code) and not me:IsTyping() and not vgui.CursorVisible() and not gui.IsGameUIVisible() and not gui.IsConsoleVisible()
  end
  if down ~= direct_key then direct_key = down; set_direct() end
  -- while the key is down, keep saying so: the microphone bridge treats a "1" that has not been rewritten for a few seconds as a
  -- game that closed or crashed, and stops recording
  if (transmitting or direct) and SysTime() - last_beat >= 0.5 then last_beat = SysTime(); write_ptt() end
end)

-- the key named by skippy_key (off until set), or `bind <key> +skippy`: talk to Skippy only (no wake word, other players hear nothing)
concommand.Add("+skippy", function() direct_cmd = true; set_direct() end)
concommand.Add("-skippy", function() direct_cmd = false; set_direct() end)

-- what the microphone bridge heard
timer.Create("Mentat.SkippyHearing", 0.25, 0, function()
  local path = DIR .. "/voice_in.json"
  if not file.Exists(path, "DATA") then return end
  local raw = file.Read(path, "DATA")
  file.Delete(path)
  local obj = util.JSONToTable(raw or "")
  if istable(obj) and isnumber(obj.ts) and os.time() - obj.ts > 10 then return end      -- heard long ago: not what the player means now
  if istable(obj) and isstring(obj.text) and obj.text ~= "" then
    net.Start("holodeck_voice")
    net.WriteString(string.sub(obj.text, 1, 300))
    net.SendToServer()
    chat.AddText(Color(130, 130, 130), "[you, aloud] ", Color(230, 230, 230), obj.text)
  end
end)

-- the names Skippy answers to FOR THIS PLAYER, sent by the server to this client alone (on join, on every change, and when asked):
-- written where the microphone bridge reads them. Missing or unreadable means the bridge falls back to "Skippy".
local NAMES_FILE = DIR .. "/wake_names.json"
local have_names = false
net.Receive("holodeck_wake_names", function()
  local count = math.min(net.ReadUInt(4), 8)
  local names = {}
  for _ = 1, count do
    local s = string.sub((tostring(net.ReadString() or ""):gsub("%c", " ")), 1, 100)     -- (a name is up to 24 characters: 96 bytes)
    if s:find("%S") then names[#names + 1] = s end
  end
  if #names == 0 then names[1] = "skippy" end
  have_names = true
  file.Write(NAMES_FILE, util.TableToJSON({ version = 1, names = names, ts = os.time() }))
end)
-- ask once the game has loaded (a few tries: the first can be sent before the server is listening); the server also pushes on join
hook.Add("InitPostEntity", "Mentat.SkippyAskNames", function()
  timer.Create("Mentat.SkippyAskNames", 5, 5, function()
    if have_names then timer.Remove("Mentat.SkippyAskNames"); return end
    local me = LocalPlayer()
    if IsValid(me) then net.Start("holodeck_wake_sync"); net.SendToServer() end
  end)
end)

-- ------------------------------------------------------------------------------------------------ Skippy speaks
local MAX_CLIP = 200000      -- bytes of one spoken clip we are willing to keep (the server sends far less)
local MAX_PARTS = 4          -- clips being assembled at once
local MAX_PLAYING = 6        -- clips playing at once
local parts, playing, play_seq = {}, {}, 0

-- leftovers from a crashed session
for _, f in ipairs((file.Find(DIR .. "/skippy_*.mp3", "DATA"))) do file.Delete(DIR .. "/" .. f) end

--- A playing channel must stay referenced until it ends, or the garbage collector frees it and the sound stops mid-sentence.
--- Release stops it (if asked) and removes its file after the stream is dropped.
local function release(key, stop)
  local p = playing[key]
  if not p then return end
  playing[key] = nil
  if stop and IsValid(p.chan) then p.chan:Stop() end
  file.Delete(p.path)
end

timer.Create("Mentat.SkippyPlayWatch", 0.5, 0, function()
  for key, p in pairs(playing) do
    if not IsValid(p.chan) or p.chan:GetState() == GMOD_CHANNEL_STOPPED or SysTime() > p.deadline then release(key, true) end
  end
end)

net.Receive("holodeck_audio", function()
  local id, idx, total = net.ReadUInt(16), net.ReadUInt(8), net.ReadUInt(8)
  local pos = net.ReadVector()
  local len = net.ReadUInt(16)
  local data = net.ReadData(len)
  if not cv_voice:GetBool() then return end
  if total == 0 or idx == 0 or idx > total or len == 0 then return end           -- not a message we sent
  for k, v in pairs(parts) do if CurTime() - v.born > 30 then parts[k] = nil end end
  local a = parts[id]
  if not a then
    if table.Count(parts) >= MAX_PARTS then return end
    a = { chunks = {}, got = 0, bytes = 0, total = total, pos = pos, born = CurTime() }
    parts[id] = a
  end
  if a.total ~= total then parts[id] = nil; return end
  if not a.chunks[idx] then
    a.chunks[idx] = data; a.got = a.got + 1; a.bytes = a.bytes + len
    if a.bytes > MAX_CLIP then parts[id] = nil; return end
  end
  if a.got < a.total then return end
  parts[id] = nil
  play_seq = play_seq + 1
  local key = play_seq
  local path = DIR .. "/skippy_" .. key .. ".mp3"
  file.Write(path, table.concat(a.chunks))
  sound.PlayFile("data/" .. path, "3d mono", function(chan, errid, errname)
    if not IsValid(chan) then
      print("[skippy] could not play Skippy's voice: " .. tostring(errid) .. " " .. tostring(errname))
      file.Delete(path)
      return
    end
    chan:SetPos(a.pos)
    chan:Set3DFadeDistance(250, 2400)
    chan:SetVolume(math.Clamp(cv_volume:GetFloat(), 0, 1))
    chan:Play()
    if table.Count(playing) >= MAX_PLAYING then release(next(playing), true) end   -- a spammer cannot pile channels up
    playing[key] = { chan = chan, path = path, deadline = SysTime() + math.max(chan:GetLength(), 1) + 10 }
    hook.Run("SkippySpoke", chan)
    if cv_debug:GetBool() then file.Write(DIR .. "/skippy_played.txt", string.format("%d %.2f", id, chan:GetLength())) end
  end)
end)
