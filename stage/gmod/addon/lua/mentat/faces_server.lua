-- faces_server.lua -- the server half of "a player's picture is their head" (see stage/gmod/holodeck/README.md, section Faces).
--
-- WHO WEARS WHAT (per player, stored and persisted; the active source is "set" | "steam" | "off" | "none"):
--   set    a picture the player chose (`skippy face <https link>` / a local file for the host / `skippy link <code>`). THEIR choice.
--   steam  the automatic default: their public Steam avatar, fetched through the bridge for the SERVER's own ply:SteamID64().
--   off    `skippy face off`: no automatic picture, ever (a picture they set themselves is still theirs to show).
--   none   nothing yet.
-- THE HARD RULE: the automatic default is applied ONLY to a player with no explicit choice (a picture of their own, `off`, `hide`, or an
-- admin block). It never overwrites, replaces, refreshes or re-enables anything of theirs, including when it is fetched again later or when
-- the answer arrives after they chose something. `remove` clears a picture of their own and returns them to the default unless they said off.
--
-- Wire (server -> client):
--   "mentat_face_map"  UInt8 n | n x { String steamid64, UInt8 slot, Bool optin }      the whole current list of visible heads
--   "mentat_face_img"  UInt8 slot | UInt32 crc | UInt8 chunk (1-based) | UInt8 chunks | UInt16 total bytes | UInt8 r,g,b (skin tone: the head's sides, back and neck)
--                      | UInt16 bytes in this chunk | Data      one 128x128 JPEG in 8 KB chunks, one chunk per 0.1 s, once per image per player
-- Wire (client -> server): "mentat_face_want" UInt8 slot (0 = everything the server has) -- late joiners and lost chunks; rate limited.
local SVM1 = include("mentat/holodeck/svm1.lua")        -- only for b64decode (the bridge answers in base64)

local Faces = MentatFaces or {}
MentatFaces = Faces

Faces.DIR = "mentat/faces"
Faces.FILE = "mentat/faces.json"
Faces.MAX_IMAGE = 24 * 1024               -- bytes: the bridge makes at most 20 KB; anything over this is never taken, stored or relayed
Faces.CHUNK = 8 * 1024
Faces.PUMP = 0.1
Faces.MAX_SLOTS = 64
Faces.MAX_QUEUED = 24
Faces.MAX_URL = 400
Faces.REFRESH_AFTER = 24 * 3600           -- an automatic Steam picture older than this is fetched again when its owner joins
Faces.TRY_AGAIN = 600                     -- seconds before a failed automatic lookup is repeated for the same player
Faces.WANT_PER_MIN = 12
Faces.rec = Faces.rec or {}               -- steamid64 -> { source, hash, crc, skin, hidden, off, optin, t }
Faces.blocked = Faces.blocked or {}       -- steamid64 -> true (an admin's decision)
Faces.images = Faces.images or {}         -- hash -> { data, crc, skin }
Faces.slot_hash = Faces.slot_hash or {}   -- slot -> hash
Faces.hash_slot = Faces.hash_slot or {}
Faces.sent = Faces.sent or {}             -- steamid64 of the viewer -> { hash = true }
Faces.queue = Faces.queue or {}
Faces.tried = Faces.tried or {}           -- steamid64 -> time of the last automatic lookup
Faces.fetching = Faces.fetching or {}
Faces.wants = Faces.wants or {}
Faces.last_msg = Faces.last_msg or {}
Faces.stats = Faces.stats or { chunks = 0, maps = 0, fetches = 0 }

util.AddNetworkString("mentat_face_map")
util.AddNetworkString("mentat_face_img")
util.AddNetworkString("mentat_face_want")

local cv_on = CreateConVar("mentat_faces", "1", FCVAR_ARCHIVE + FCVAR_NOTIFY + FCVAR_REPLICATED, "1 = players' pictures are worn as heads; 0 = nobody's is")
Faces.cv_on = cv_on
local cv_default = CreateConVar("mentat_faces_steam_default", "1", FCVAR_ARCHIVE, "1 = players with no choice of their own get their public Steam avatar as a head; 0 = only pictures players set themselves")

local function now() return CurTime() end
--- A player's key: their SteamID64 (a bot may have none: then a key of its own, never written as a real id).
local function sid_of(p) return p:SteamID64() or ("bot" .. p:EntIndex()) end
Faces.sid_of = sid_of
local function valid_sid(s) return type(s) == "string" and #s == 17 and s:match("^7656119%d+$") ~= nil end
local function valid_hash(h) return type(h) == "string" and h:match("^%x+$") ~= nil and #h == 24 end

local function tell(ply, text)
  local sid = IsValid(ply) and ply:SteamID64() or "console"
  Faces.last_msg[sid] = text
  if IsValid(ply) and ply.ChatPrint then ply:ChatPrint("[Skippy] " .. text) else print("[Skippy faces] " .. text) end
  return text
end
Faces.tell = tell

-- ------------------------------------------------------------------------------------------------ the record of who has what
local function rec_of(sid, create)
  local r = Faces.rec[sid]
  if not r and create then r = {}; Faces.rec[sid] = r end
  return r
end

--- "set" | "steam" | "off" | "none"
function Faces.source_of(sid)
  local r = Faces.rec[sid]
  if not r then return "none" end
  if r.source == "set" and r.hash then return "set" end
  if r.off then return "off" end
  if r.source == "steam" and r.hash then return "steam" end
  return "none"
end

--- Has this player made a choice the automatic default must not touch?
function Faces.explicit(sid)
  local r = Faces.rec[sid]
  if Faces.blocked[sid] then return true end
  return r ~= nil and ((r.source == "set" and r.hash ~= nil) or r.off == true or r.hidden == true)
end

function Faces.visible(sid)
  local r = Faces.rec[sid]
  if not cv_on:GetBool() or not r or not r.hash or r.hidden or Faces.blocked[sid] then return false end
  if r.source ~= "set" and (r.off or r.source ~= "steam") then return false end
  return Faces.images[r.hash] ~= nil
end

-- ------------------------------------------------------------------------------------------------ persistence (data/mentat/faces.json)
local function clean_rec(r)
  if type(r) ~= "table" then return nil end
  local out = { hidden = r.hidden == true, off = r.off == true, optin = r.optin == true, t = tonumber(r.t) or 0 }
  if (r.source == "set" or r.source == "steam") and valid_hash(r.hash) then
    out.source, out.hash = r.source, r.hash
    out.crc = tonumber(r.crc)
    local b = r.skin
    if type(b) == "table" and #b == 3 then out.skin = { math.Clamp(tonumber(b[1]) or 128, 0, 255), math.Clamp(tonumber(b[2]) or 128, 0, 255), math.Clamp(tonumber(b[3]) or 128, 0, 255) } end
    if not (out.crc and out.skin) then out.source, out.hash, out.crc, out.skin = nil, nil, nil, nil end
  end
  if not (out.source or out.hidden or out.off or out.optin) then return nil end
  return out
end

local function parse_state(txt)
  if type(txt) ~= "string" or txt == "" then return nil end
  local ok, obj = pcall(util.JSONToTable, txt)
  if not ok or type(obj) ~= "table" or type(obj.rec) ~= "table" then return nil end
  local rec, blocked = {}, {}
  for sid, r in pairs(obj.rec) do
    if valid_sid(sid) then
      local c = clean_rec(r)
      if c then rec[sid] = c end
    end
  end
  if type(obj.blocked) == "table" then
    for sid, v in pairs(obj.blocked) do if valid_sid(sid) and v then blocked[sid] = true end end
  end
  return rec, blocked
end

function Faces.load()
  Faces.rec, Faces.blocked, Faces.images = {}, {}, {}
  local rec, blocked
  for _, path in ipairs({ Faces.FILE, Faces.FILE .. ".bak" }) do
    rec, blocked = parse_state(file.Read(path, "DATA"))
    if rec then Faces.loaded_from = path; break end
    if file.Exists(path, "DATA") then Faces.corrupt = (Faces.corrupt or 0) + 1 end
  end
  Faces.rec, Faces.blocked = rec or {}, blocked or {}
  for sid, r in pairs(Faces.rec) do               -- the pictures themselves: gone or damaged means no picture (the default may fetch again)
    if r.hash then
      local data = file.Read(Faces.DIR .. "/" .. r.hash .. ".jpg", "DATA")
      if data and #data <= Faces.MAX_IMAGE and #data > 4 and tonumber(util.CRC(data)) == r.crc and data:sub(1, 3) == "\255\216\255" then
        Faces.images[r.hash] = { data = data, crc = r.crc, skin = r.skin }
      else
        r.source, r.hash, r.crc, r.skin = nil, nil, nil, nil
      end
    end
  end
end

function Faces.save()
  local out = { version = 1, rec = Faces.rec, blocked = Faces.blocked }
  local txt = util.TableToJSON(out)
  if not txt then return false end
  file.CreateDir("mentat")
  local old = file.Read(Faces.FILE, "DATA")
  if old and parse_state(old) then file.Write(Faces.FILE .. ".bak", old) end      -- the last good copy
  if file.Rename then                                                                -- write whole, then swap: a crash mid-write leaves the old file
    file.Write(Faces.FILE .. ".tmp", txt)
    if file.Exists(Faces.FILE, "DATA") then file.Delete(Faces.FILE) end
    file.Rename(Faces.FILE .. ".tmp", Faces.FILE)
  else
    file.Write(Faces.FILE, txt)
  end
  return true
end

local function store_image(ans)
  file.CreateDir(Faces.DIR)
  file.Write(Faces.DIR .. "/" .. ans.hash .. ".jpg", ans.data)
  Faces.images[ans.hash] = { data = ans.data, crc = ans.crc, skin = ans.skin }
end

--- Delete picture files nobody's record names any more (retention: a picture lives only as long as someone wears it).
function Faces.sweep()
  local used = {}
  for _, r in pairs(Faces.rec) do if r.hash then used[r.hash] = true end end
  for h in pairs(Faces.images) do
    if not used[h] then
      Faces.images[h] = nil
      file.Delete(Faces.DIR .. "/" .. h .. ".jpg")
      local s = Faces.hash_slot[h]
      if s then Faces.hash_slot[h], Faces.slot_hash[s] = nil, nil end
    end
  end
end

-- ------------------------------------------------------------------------------------------------ what the bridge sends back
--- Check the bridge's answer; returns { hash, data, crc, skin } or nil and why. Nothing from the bridge is believed unchecked.
function Faces.check_answer(body)
  local obj = util.JSONToTable(body or "")
  if type(obj) ~= "table" then return nil, "the bridge's answer was not understood" end
  if obj.ok ~= true then
    return nil, tostring(obj.kind or "unavailable"), tostring(obj.error or ""):sub(1, 120)
  end
  if not (valid_hash(obj.hash) and isstring(obj.b64) and #obj.b64 <= math.ceil(Faces.MAX_IMAGE * 4 / 3) + 8) then return nil, "bad_image", "the picture was not usable" end
  local b = obj.skin
  if not (type(b) == "table" and #b == 3) then return nil, "bad_image", "no skin colour" end
  local data = SVM1.b64decode(obj.b64)
  if not data or #data < 16 or #data > Faces.MAX_IMAGE then return nil, "bad_image", "the picture was not usable" end
  if data:sub(1, 3) ~= "\255\216\255" then return nil, "bad_image", "not a jpeg" end
  local crc = tonumber(util.CRC(data))
  if not crc or (obj.crc32 ~= nil and tonumber(obj.crc32) ~= crc) then return nil, "bad_image", "checksum mismatch" end
  local skin = {}
  for i = 1, 3 do skin[i] = math.floor(math.Clamp(tonumber(b[i]) or 128, 0, 255)) end
  return { hash = obj.hash, data = data, crc = crc, skin = skin, source = obj.source }
end

-- ------------------------------------------------------------------------------------------------ rationing
local function bucket(cap, rate) return { cap = cap, rate = rate, tokens = cap, t = SysTime() } end
local function take(b, n)
  local t = SysTime()
  b.tokens = math.min(b.cap, b.tokens + (t - b.t) * b.rate)
  b.t = t
  if b.tokens >= n then b.tokens = b.tokens - n; return true end
  return false
end
Faces.global_calls = Faces.global_calls or bucket(10, 20 / 60)      -- bridge calls the whole server may spend on pictures: 20 a minute
Faces.player_calls = Faces.player_calls or {}
local function allow_call(sid)
  local b = Faces.player_calls[sid]
  if not b then b = bucket(3, 1 / 10); Faces.player_calls[sid] = b end
  if not take(b, 1) then return false end
  if not take(Faces.global_calls, 1) then b.tokens = b.tokens + 1; return false end
  return true
end

-- ------------------------------------------------------------------------------------------------ slots and sending
local function slot_for(hash)
  if Faces.hash_slot[hash] then return Faces.hash_slot[hash] end
  local worn = {}
  for sid, r in pairs(Faces.rec) do if r.hash and Faces.visible(sid) then worn[r.hash] = true end end
  for s = 1, Faces.MAX_SLOTS do
    local h = Faces.slot_hash[s]
    if not h or not worn[h] then
      if h then Faces.hash_slot[h] = nil end
      Faces.slot_hash[s], Faces.hash_slot[hash] = hash, s
      for _, seen in pairs(Faces.sent) do seen[h or ""] = nil; seen[hash] = nil end     -- everyone needs the new picture for this number
      return s
    end
  end
  return nil
end

local function pump()
  local job = Faces.queue[1]
  if not job then timer.Remove("mentat_face_pump"); return end
  local alive = {}
  for _, p in ipairs(job.targets) do if IsValid(p) then alive[#alive + 1] = p end end
  job.targets = alive
  local img = Faces.images[job.hash]
  if #alive == 0 or not img or Faces.hash_slot[job.hash] ~= job.slot or job.i > job.chunks then table.remove(Faces.queue, 1); return end
  local piece = img.data:sub((job.i - 1) * Faces.CHUNK + 1, job.i * Faces.CHUNK)
  net.Start("mentat_face_img")
  net.WriteUInt(job.slot, 8)
  net.WriteUInt(img.crc, 32)
  net.WriteUInt(job.i, 8)
  net.WriteUInt(job.chunks, 8)
  net.WriteUInt(#img.data, 16)
  net.WriteUInt(img.skin[1], 8); net.WriteUInt(img.skin[2], 8); net.WriteUInt(img.skin[3], 8)
  net.WriteUInt(#piece, 16)
  net.WriteData(piece, #piece)
  net.Send(alive)
  Faces.stats.chunks = Faces.stats.chunks + 1
  job.i = job.i + 1
  if job.i > job.chunks then table.remove(Faces.queue, 1) end
end

--- Send a picture to these players, unless each has it already (`force` resends: a client asked).
function Faces.send_image(hash, targets, force)
  local img, slot = Faces.images[hash], Faces.hash_slot[hash]
  if not (img and slot) then return false end
  local list = {}
  for _, p in ipairs(targets) do
    if IsValid(p) then
      local sid = sid_of(p)
      Faces.sent[sid] = Faces.sent[sid] or {}
      if force or not Faces.sent[sid][hash] then Faces.sent[sid][hash] = true; list[#list + 1] = p end
    end
  end
  if #list == 0 then return true end
  if #Faces.queue >= Faces.MAX_QUEUED then
    for _, p in ipairs(list) do Faces.sent[sid_of(p)][hash] = nil end     -- not sent: the next sync tries again
    return false
  end
  Faces.queue[#Faces.queue + 1] = { hash = hash, slot = slot, targets = list, i = 1, chunks = math.max(1, math.ceil(#img.data / Faces.CHUNK)) }
  if not timer.Exists("mentat_face_pump") then timer.Create("mentat_face_pump", Faces.PUMP, 0, pump) end
  return true
end

--- The visible list: { {sid, slot, optin}, ... } for connected players.
function Faces.map()
  local out = {}
  for _, p in ipairs(player.GetAll()) do
    local sid = sid_of(p)
    if Faces.visible(sid) then
      local slot = slot_for(Faces.rec[sid].hash)
      if slot then out[#out + 1] = { sid = sid, slot = slot, optin = Faces.rec[sid].optin == true } end
    end
  end
  return out
end

local function send_map(targets)
  local list = Faces.map()
  net.Start("mentat_face_map")
  net.WriteUInt(#list, 8)
  for _, e in ipairs(list) do
    net.WriteString(e.sid); net.WriteUInt(e.slot, 8); net.WriteBool(e.optin)
  end
  if targets then net.Send(targets) else net.Broadcast() end
  Faces.stats.maps = Faces.stats.maps + 1
  return list
end

--- Make every client agree with the records: send the pictures anyone lacks, then the list. Removal, hiding and blocking take effect at once.
function Faces.sync(only)
  local who = only and { only } or player.GetAll()
  local list = Faces.map()
  local seen = {}
  for _, e in ipairs(list) do
    local h = Faces.rec[e.sid].hash
    if not seen[h] then seen[h] = true; Faces.send_image(h, who) end
  end
  return send_map(only)
end

local function changed(sid)
  Faces.save()
  Faces.sync()
end

-- ------------------------------------------------------------------------------------------------ asking the bridge
function Faces.bridge(body, done)
  local H = Holodeck
  if not (H and H.BridgePost) then return done(nil, "unavailable", "no bridge here") end
  local sent = H.BridgePost("/face", body, function(txt)
    local ans, kind, why = Faces.check_answer(txt)
    done(ans, kind, why)
  end, function(reason) done(nil, "unavailable", tostring(reason)) end, 25)
  if sent == false then done(nil, "unavailable", "the bridge is not set up") end
end

--- Adopt an answer for a player. source "set" is always theirs; "steam" only while they have made no choice (checked NOW, not when asked).
local function adopt(sid, ans, source)
  if source == "steam" and (Faces.explicit(sid) or not cv_default:GetBool()) then return false end
  if Faces.blocked[sid] then return false end
  local r = rec_of(sid, true)
  if source == "set" then r.hidden = false end                  -- choosing a picture means wanting it seen
  store_image(ans)
  r.source, r.hash, r.crc, r.skin, r.t = source, ans.hash, ans.crc, ans.skin, os.time()
  Faces.sweep()
  changed(sid)
  return true
end
Faces.adopt = adopt

local function steam_ok(ply)
  return IsValid(ply) and valid_sid(ply:SteamID64()) and not (ply.IsBot and ply:IsBot())
end

--- The automatic default: this player's public Steam avatar, if (and only if) they have made no choice of their own.
function Faces.default_for(ply, force)
  if not steam_ok(ply) or not cv_on:GetBool() or not cv_default:GetBool() then return false end
  local sid = ply:SteamID64()
  if Faces.explicit(sid) or Faces.fetching[sid] then return false end
  local r = Faces.rec[sid]
  if r and r.source == "steam" and r.hash and not force and os.time() - (r.t or 0) < Faces.REFRESH_AFTER then return false end   -- still fresh
  if Faces.tried[sid] and now() - Faces.tried[sid] < Faces.TRY_AGAIN and not force then return false end
  if not allow_call(sid) then return false end
  Faces.tried[sid], Faces.fetching[sid] = now(), true
  Faces.stats.fetches = Faces.stats.fetches + 1
  Faces.bridge({ action = "steam", steamid64 = sid, who = sid }, function(ans)
    Faces.fetching[sid] = nil
    if ans then adopt(sid, ans, "steam") end                 -- (silently: a missing or private profile just means the usual head)
  end)
  return true
end

-- ------------------------------------------------------------------------------------------------ the player's commands
local function can_local(ply)
  return not IsValid(ply) or ply:IsListenServerHost() or game.SinglePlayer()
end

local function clean_arg(s)
  s = tostring(s or "")
  s = s:gsub("[%c\127]", " "):match("^%s*(.-)%s*$") or ""
  return s
end

local function looks_like_url(s) return s:lower():match("^%a[%w+.-]*://") ~= nil end

function Faces.set_picture(ply, arg)
  local sid = IsValid(ply) and ply:SteamID64() or nil
  if not sid then return tell(ply, "The console has no head; run this as a player.") end
  if Faces.blocked[sid] then return tell(ply, "An admin has switched your picture off here.") end
  local body
  if looks_like_url(arg) then
    if #arg > Faces.MAX_URL or arg:find("%s") or not arg:lower():match("^https://") then
      return tell(ply, "I can only use one https link, from one of the picture sites (Google, Gravatar, Steam).")
    end
    body = { action = "url", url = arg, who = sid }
  elseif can_local(ply) and arg ~= "" and #arg <= 260 then
    body = { action = "path", path = arg, who = sid }
  else
    return tell(ply, "Give me an https link to your picture.")
  end
  if not allow_call(sid) then return tell(ply, "Slow down a little, then try again.") end
  Faces.bridge(body, function(ans, kind, why)
    if not ans then return tell(ply, "I could not use that picture (" .. tostring(why ~= "" and why or kind) .. "). You keep your usual head.") end
    if adopt(sid, ans, "set") then tell(ply, "Done. That is your head now; say `skippy face hide` to put it away.") end
  end)
  return tell(ply, "Fetching your picture...")
end

function Faces.link(ply, code)
  local sid = IsValid(ply) and ply:SteamID64() or nil
  if not sid then return tell(ply, "Run this as a player.") end
  if Faces.blocked[sid] then return tell(ply, "An admin has switched your picture off here.") end
  code = clean_arg(code)
  if not code:match("^[%w%-]+$") or #code < 4 or #code > 16 then return tell(ply, "That does not look like a link code.") end
  if not allow_call(sid) then return tell(ply, "Slow down a little, then try again.") end
  Faces.bridge({ action = "redeem", code = code, who = sid }, function(ans, kind, why)
    if not ans then
      if kind == "unavailable" then return tell(ply, "The portal link is not available right now. Try `skippy face <link>`.") end
      return tell(ply, "That code did not work (it may have expired). Get a new one in the portal.")
    end
    if adopt(sid, ans, "set") then tell(ply, "Linked. Your portal picture is your head now.") end
  end)
  return tell(ply, "Checking your code...")
end

local verbs = {}
function verbs.hide(ply, sid)
  rec_of(sid, true).hidden = true
  changed(sid)
  return tell(ply, "Your picture is hidden for everyone. `skippy face show` brings it back.")
end
function verbs.show(ply, sid)
  rec_of(sid, true).hidden = false
  changed(sid)
  if not Faces.rec[sid].hash then Faces.default_for(ply, true) end
  return tell(ply, "Your picture is shown again.")
end
function verbs.remove(ply, sid)
  local r = Faces.rec[sid]
  if not (r and r.source == "set" and r.hash) then
    return tell(ply, "You have not set a picture of your own. Use `skippy face off` to stop using your Steam picture.")
  end
  r.source, r.hash, r.crc, r.skin = nil, nil, nil, nil
  Faces.sweep()
  changed(sid)
  if not r.off then Faces.default_for(ply, true) end                        -- back to the default only if they never said off
  return tell(ply, r.off and "Removed. You said off, so you have your usual head." or "Removed. You are back to your Steam picture.")
end
function verbs.off(ply, sid)
  local r = rec_of(sid, true)
  r.off, r.optin = true, false
  if r.source == "steam" then r.source, r.hash, r.crc, r.skin = nil, nil, nil, nil end
  Faces.sweep()
  changed(sid)
  return tell(ply, "Off. I will not use your Steam picture. A picture you set yourself is still yours to show or remove.")
end
verbs["on"] = function(ply, sid)
  local r = rec_of(sid, true)
  r.off, r.hidden, r.optin = false, false, true            -- "on" is also the explicit yes that lets a custom model have its head replaced
  changed(sid)
  if not r.hash then Faces.default_for(ply, true) end
  return tell(ply, "On. Your picture is your head, whatever model you wear.")
end
function verbs.status(ply, sid)
  local r = Faces.rec[sid] or {}
  return tell(ply, ("Source: %s%s%s%s. Faces are %s on this server."):format(Faces.source_of(sid), r.hidden and ", hidden" or "",
    Faces.blocked[sid] and ", blocked by an admin" or "", r.optin and ", opted in" or "", cv_on:GetBool() and "on" or "off"))
end

--- `skippy face <rest>` / `skippy link <code>` from chat or console. ply is nil for the server console.
function Faces.run(ply, verb, rest)
  rest = clean_arg(rest)
  if verb == "link" then return Faces.link(ply, rest) end
  local sid = IsValid(ply) and ply:SteamID64() or nil
  local low = rest:lower()
  if low == "" then low = "status" end
  if verbs[low] and sid then return verbs[low](ply, sid) end
  return Faces.set_picture(ply, rest)
end

--- "skippy face ..." / "skippy link ..." -> verb, rest (or nil).
function Faces.parse_chat(text)
  local verb, rest = tostring(text or ""):match("^%s*[Hh]?[Ee]?[Yy]?,?%s*[Ss][Kk][Ii][Pp][Pp][Yy][%s,:]+(%a+)%s*(.*)$")
  if not verb then return nil end
  verb = verb:lower()
  if verb ~= "face" and verb ~= "link" then return nil end
  return verb, rest
end

local last_chat = setmetatable({}, { __mode = "k" })
--- Chat entry; returns "" (hide the message: a link code is nobody else's business) for a face command, else nil. Safe to call twice per message.
function Faces.chat(ply, text)
  local verb, rest = Faces.parse_chat(text)
  if not verb then return nil end
  local k = last_chat[ply]
  if not (k and k.text == text and k.t == CurTime()) then
    last_chat[ply] = { text = text, t = CurTime() }
    Faces.run(ply, verb, rest)
  end
  return ""
end

hook.Add("PlayerSay", "Mentat.Faces.Chat", function(ply, text) return Faces.chat(ply, text) end)

--- Holodeck's own chat hook would otherwise hear "skippy face ..." and ask the planner (and the model) about it: put ours in front of it.
function Faces.wrap_holodeck_chat()
  local t = hook.GetTable and hook.GetTable() or nil
  local old = t and t.PlayerSay and t.PlayerSay["Holodeck.Chat"]
  if not old or Faces.wrapped == old then return false end
  local wrapper = function(ply, text)
    if Faces.chat(ply, text) ~= nil then return "" end
    return old(ply, text)
  end
  hook.Add("PlayerSay", "Holodeck.Chat", wrapper)
  Faces.wrapped = wrapper
  return true
end

concommand.Add("skippy_face", function(ply, _, args)
  Faces.run(ply, "face", table.concat(args, " ", 1))
end, nil, "skippy_face hide|show|remove|off|on|status|<https link>")
concommand.Add("skippy_face_set", function(ply, _, args) Faces.run(ply, "face", table.concat(args, " ")) end, nil,
  "skippy_face_set <https link, or a file on the host>: use this picture as your head")
concommand.Add("skippy_link", function(ply, _, args) Faces.run(ply, "link", args[1] or "") end, nil, "skippy_link <code from the portal>")

-- ------------------------------------------------------------------------------------------------ admin
local function is_admin(ply) return not IsValid(ply) or ply:IsAdmin() or ply:IsSuperAdmin() end

concommand.Add("mentat_face_block", function(ply, _, args)
  if not is_admin(ply) then return end
  local sid = tostring(args[1] or "")
  if not valid_sid(sid) then return tell(ply, "Usage: mentat_face_block <steamid64>") end
  Faces.blocked[sid] = true
  changed(sid)
  tell(ply, "Blocked " .. sid .. ": their head is gone for everyone. `mentat_face_unblock` lifts it.")
end, nil, "mentat_face_block <steamid64>: no picture for this player, now and later (admins)")
concommand.Add("mentat_face_unblock", function(ply, _, args)
  if not is_admin(ply) then return end
  local sid = tostring(args[1] or "")
  if not valid_sid(sid) then return tell(ply, "Usage: mentat_face_unblock <steamid64>") end
  Faces.blocked[sid] = nil
  changed(sid)
  tell(ply, "Unblocked " .. sid .. ".")
end, nil, "mentat_face_unblock <steamid64> (admins)")

cvars.AddChangeCallback("mentat_faces", function() Faces.sync() end, "mentat_faces_sync")

-- ------------------------------------------------------------------------------------------------ clients
net.Receive("mentat_face_want", function(_, ply)
  local slot = net.ReadUInt(8)
  if not IsValid(ply) then return end
  local sid = sid_of(ply)
  local w = Faces.wants[sid]
  local t = now()
  if not w or t - w.t0 >= 60 then w = { t0 = t, n = 0 }; Faces.wants[sid] = w end
  w.n = w.n + 1
  if w.n > Faces.WANT_PER_MIN then return end
  if slot == 0 then
    Faces.sent[sid] = nil
    Faces.sync(ply)
  else
    local h = Faces.slot_hash[slot]
    if h and Faces.images[h] then Faces.send_image(h, { ply }, true) end
  end
end)

hook.Add("PlayerInitialSpawn", "Mentat.Faces.Join", function(ply)
  timer.Simple(5, function() if IsValid(ply) then Faces.default_for(ply) end end)
end)

hook.Add("PlayerDisconnected", "Mentat.Faces.Leave", function(ply)
  local sid = IsValid(ply) and sid_of(ply)
  if sid then Faces.sent[sid], Faces.wants[sid], Faces.fetching[sid] = nil, nil, nil end
  timer.Simple(0, function() Faces.sync() end)
end)

hook.Add("InitPostEntity", "Mentat.Faces.Wrap", function() Faces.wrap_holodeck_chat() end)
timer.Simple(0, function() Faces.wrap_holodeck_chat() end)

Faces.load()
return Faces
