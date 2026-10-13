-- meshnet.lua -- the server half of "the real shape of a found object" (the other halves: the bridge's POST /mesh, and the client's
-- mentat/holodeck/meshclient.lua). Loaded by mentat/holodeck.lua.
--
--   spawn of a found object whose card has a mesh -> attach(): the bridge is asked for the payload ONCE per model (not per entity);
--   the answer is checked (size cap, CRC-32, LZMA header, SVM1/SVM3 decode) and kept; it is sent to every client in 16 KB chunks, one chunk
--   per 0.1 s; every entity of that model is told the model's number (ent:SetVisual("mesh", { id = n })); a client that joins later
--   (or missed a chunk) asks with "holodeck_mesh_want" and is sent it alone. A failure of any kind leaves the box.
--
-- Wire (server -> client, net message "holodeck_mesh", one per chunk):
--   UInt8 mesh number (1..64) | UInt8 chunk number (1-based) | UInt8 chunk count | UInt20 total bytes | UInt32 CRC-32 of the whole
--   payload | UInt16 bytes in this chunk | Data. The payload is the bridge's LZMA bytes as they came (util.Decompress reads them).
-- Wire (client -> server, "holodeck_mesh_want"): UInt8 mesh number, 0 = every mesh the server has.
--
-- The client cache between sessions (README "Object caches"): a client that announces itself ("holodeck_mesh_hello", sent once on joining)
-- is OFFERED a model instead of being sent it:
--   "holodeck_mesh_offer"  server -> client: UInt8 mesh number | UInt32 CRC-32 | UInt20 total bytes | String SHA-256 (hex) of the payload
--   "holodeck_mesh_have"   client -> server: UInt8 mesh number | UInt32 CRC-32 | Bool have
-- have = true: the client holds those exact bytes (its own data folder, checksummed) and nothing is sent; have = false (or no answer within
-- Mesh.OFFER_WAIT seconds): the chunks are streamed to that client alone. A client that never said hello (an older one) is streamed to as before;
-- a hello also asks for everything the server already has, as offers. `holodeck_mesh_want` still works for any client at any time.
local SVM1 = include("mentat/holodeck/svm1.lua")
local FP = include("mentat/holodeck/foundprog.lua")        -- the transporter's progress, heartbeat and failure (server side)

local Mesh = (Holodeck and Holodeck.Mesh) or {}            -- a reload of holodeck.lua keeps what was fetched (entities keep their model numbers)
Mesh.CAP_BYTES = 256 * 1024          -- a payload over this is never taken, relayed or accepted (also the client's limit)
Mesh.MAX_RAW = 1024 * 1024           -- what the LZMA header may claim to inflate to
Mesh.CHUNK = 16 * 1024               -- bytes per net message (the engine allows about 64 KB; reliable buffers are shared with everything else)
Mesh.MAX_CHUNKS = math.ceil(Mesh.CAP_BYTES / Mesh.CHUNK)
Mesh.MAX_MESHES = 64                 -- models per session
Mesh.PUMP = 0.1                      -- seconds between chunks
Mesh.MAX_QUEUED = 8                  -- sends waiting their turn
Mesh.RETRY_AFTER = 60                -- seconds before a failed model may be asked for again, at most MAX_TRIES times
Mesh.MAX_TRIES = 3
Mesh.WANT_PER_MINUTE = 12            -- "holodeck_mesh_want" messages per player per minute
Mesh.REQUEST_TIMEOUT = 60

Mesh.by_key = Mesh.by_key or {}                     -- bridge key (16 hex) -> record
Mesh.by_id = Mesh.by_id or {}                      -- mesh number -> record
Mesh.count = Mesh.count or 0
Mesh.queue = Mesh.queue or {}
Mesh.wants = Mesh.wants or {}                      -- player key -> { t0, n }
Mesh.deps = nil
Mesh.sent_chunks = Mesh.sent_chunks or 0                 -- for the status line and the tests

local cv_fetch = CreateConVar("holodeck_mesh", "1", FCVAR_ARCHIVE, "1 = fetch the real shape of found objects from the bridge and show it in place of the box; 0 = boxes only")
Mesh.cv_fetch = cv_fetch

util.AddNetworkString("holodeck_mesh")
util.AddNetworkString("holodeck_mesh_want")
util.AddNetworkString("holodeck_mesh_hello")
util.AddNetworkString("holodeck_mesh_offer")
util.AddNetworkString("holodeck_mesh_have")

Mesh.OFFER_WAIT = 3                  -- seconds a client has to answer an offer before it is streamed to anyway
Mesh.caps = Mesh.caps or {}                          -- player -> true once it said hello (it can answer offers)
Mesh.cache_hits, Mesh.cache_wants, Mesh.offers_sent = Mesh.cache_hits or 0, Mesh.cache_wants or 0, Mesh.offers_sent or 0

--- deps = { post = bridge_post, allow = bridge_allow, key = player_key, quiet_post = <bridge_post that is not counted as a call in flight>,
---          say = fn(ply, text), wisp = fn(ply, text, class, pos) (leaves the reason hanging where the object failed) }
function Mesh.init(deps)
  Mesh.deps = deps
  FP.init({ mesh = Mesh, quiet_post = deps.quiet_post, say = deps.say, wisp = deps.wisp })
end
Mesh.FP = FP

local function now() return CurTime() end

--- Check a payload as the bridge delivered it. Returns { crc, total, tris, verts } or nil and why.
function Mesh.check(data, declared_bytes, declared_crc)
  if type(data) ~= "string" or #data == 0 then return nil, "empty payload" end
  if #data > Mesh.CAP_BYTES then return nil, "payload over the " .. Mesh.CAP_BYTES .. " byte cap" end
  if declared_bytes ~= nil and #data ~= declared_bytes then return nil, "length differs from the declared one" end
  local crc = tonumber(util.CRC(data))
  if not crc or (declared_crc ~= nil and crc ~= declared_crc) then return nil, "checksum mismatch" end
  local size = SVM1.lzma_size(data)
  if not size or size < 40 or size > Mesh.MAX_RAW then return nil, "the LZMA header declares no usable size" end
  local ok, raw = pcall(util.Decompress, data)
  if not ok or type(raw) ~= "string" or #raw ~= size then return nil, "util.Decompress did not give the declared bytes" end
  local dec, why = SVM1.decode_found(raw)
  if not dec then return nil, "not a mesh: " .. tostring(why) end
  return { crc = crc, total = #data, tris = dec.nt, verts = dec.nv }
end

-- ------------------------------------------------------------------------------------------------ sending
local function pump()
  local job = Mesh.queue[1]
  if not job then timer.Remove("holodeck_mesh_pump"); return end
  local alive = {}
  for _, p in ipairs(job.targets) do if IsValid(p) then alive[#alive + 1] = p end end
  job.targets = alive
  local m = job.m
  if #alive == 0 or not m.data or job.i > m.chunks then
    table.remove(Mesh.queue, 1)
    if job.track then FP.sent(m) end                       -- nobody to send to: whoever asks later gets it; the objects are done waiting
    return
  end
  local piece = m.data:sub((job.i - 1) * Mesh.CHUNK + 1, job.i * Mesh.CHUNK)
  net.Start("holodeck_mesh")
  net.WriteUInt(m.id, 8)
  net.WriteUInt(job.i, 8)
  net.WriteUInt(m.chunks, 8)
  net.WriteUInt(m.total, 20)
  net.WriteUInt(m.crc, 32)
  net.WriteUInt(#piece, 16)
  net.WriteData(piece, #piece)
  net.Send(alive)
  Mesh.sent_chunks = Mesh.sent_chunks + 1
  job.i = job.i + 1
  if job.track then FP.stage(m, "sending", (job.i - 1) / m.chunks) end          -- chunks sent of chunks to send
  if job.i > m.chunks then
    table.remove(Mesh.queue, 1)
    if job.track then FP.sent(m) end
  end
end

--- Queue a ready model for these players. Returns false when the queue is full or there is nobody to send to.
function Mesh.send(m, targets, track)
  if m.state ~= "ready" then return false end
  local list = {}
  for _, p in ipairs(targets) do if IsValid(p) then list[#list + 1] = p end end
  if #list == 0 then return false end
  if #list == 1 then                                    -- the same player asking again while it is queued is one send
    for _, job in ipairs(Mesh.queue) do
      if job.m == m and #job.targets == 1 and job.targets[1] == list[1] and job.i == 1 then return true end
    end
  end
  if #Mesh.queue >= Mesh.MAX_QUEUED then return false end
  Mesh.queue[#Mesh.queue + 1] = { m = m, targets = list, i = 1, track = track }
  if not timer.Exists("holodeck_mesh_pump") then timer.Create("holodeck_mesh_pump", Mesh.PUMP, 0, pump) end
  return true
end

--- Offer a ready model to players who can answer (see the header). `track` = this is the transporter's own push: when every offer has been
--- answered "have" the objects are done; a "want" streams the model to that player and that send ends the wait.
function Mesh.offer(m, list, track)
  if not (m.offers and next(m.offers)) then m.offers, m.offer_wanted = {}, 0 end
  m.offer_track = m.offer_track or track
  for _, p in ipairs(list) do
    m.offers[p] = now()
    net.Start("holodeck_mesh_offer")
    net.WriteUInt(m.id, 8)
    net.WriteUInt(m.crc, 32)
    net.WriteUInt(m.total, 20)
    net.WriteString(m.sha)
    net.Send(p)
    Mesh.offers_sent = Mesh.offers_sent + 1
  end
  if track then FP.stage(m, "sending", 0) end
  timer.Simple(Mesh.OFFER_WAIT, function() Mesh.expire(m) end)
end

--- The player's answer to an offer (or its silence). Returns false when there was no such offer.
function Mesh.resolve_offer(m, ply, have)
  if not (m.offers and m.offers[ply] ~= nil) then return false end
  m.offers[ply] = nil
  if have then
    Mesh.cache_hits = Mesh.cache_hits + 1
  else
    Mesh.cache_wants = Mesh.cache_wants + 1
    m.offer_wanted = (m.offer_wanted or 0) + 1
    if not Mesh.send(m, { ply }, m.offer_track) and m.offer_track then FP.sent(m) end
  end
  if next(m.offers) == nil and m.offer_track and (m.offer_wanted or 0) == 0 then
    m.offer_track = nil
    FP.sent(m)
  end
  return true
end

--- Offers nobody answered in time are treated as "want".
function Mesh.expire(m)
  if not m.offers then return end
  local t = now()
  for p, t0 in pairs(m.offers) do
    if not IsValid(p) then m.offers[p] = nil elseif t - t0 >= Mesh.OFFER_WAIT - 0.001 then Mesh.resolve_offer(m, p, false) end
  end
  if next(m.offers) == nil and m.offer_track and (m.offer_wanted or 0) == 0 then m.offer_track = nil; FP.sent(m) end
end

--- Get a ready model to these players: offered to those who can answer, streamed to the rest. Returns false when nobody could be served.
function Mesh.deliver(m, targets, track)
  if m.state ~= "ready" then return false end
  local direct, offered = {}, {}
  for _, p in ipairs(targets) do
    if IsValid(p) then
      if Mesh.caps[p] and m.sha then offered[#offered + 1] = p else direct[#direct + 1] = p end
    end
  end
  if #direct + #offered == 0 then return false end
  if #offered > 0 then Mesh.offer(m, offered, track and #direct == 0) end
  if #direct > 0 then return Mesh.send(m, direct, track) or #offered > 0 end
  return true
end

-- ------------------------------------------------------------------------------------------------ entities
local function tell_entities(m)
  local keep = {}
  for _, e in ipairs(m.ents) do
    if IsValid(e) then
      if m.state == "ready" and e.SetVisual then e:SetVisual("mesh", { id = m.id }) end
      keep[#keep + 1] = e
    end
  end
  m.ents = keep
end

-- ------------------------------------------------------------------------------------------------ asking the bridge
local function fail(m, why)
  m.state, m.why, m.t = "failed", tostring(why), now()
  Mesh.last_error = m.why
  pcall(hook.Run, "HolodeckBeam", "failed", m, m.why)         -- (the skippy director's notebook and reactions)
  FP.fail(m, why)                                         -- the beam collapses, the placeholder goes, Skippy says why
end
Mesh.fail = fail

Mesh.BUSY_RETRY = 2.0                -- seconds before asking again when the bridge says "busy" (one download at a time: this one waits its turn)

local ask

local function on_answer(m, body)
  local obj = util.JSONToTable(body or "")
  if not (istable(obj) and obj.ok == true and obj.key == m.key and (obj.format == "svm1-lzma" or obj.format == "svm3-lzma") and isstring(obj.b64)) then
    return fail(m, "the bridge's answer was not a mesh")
  end
  if #obj.b64 > math.ceil(Mesh.CAP_BYTES * 4 / 3) + 8 then return fail(m, "the bridge's payload is over the cap") end
  local data = SVM1.b64decode(obj.b64)
  if not data then return fail(m, "the payload was not base64") end
  local info, why = Mesh.check(data, tonumber(obj.bytes), tonumber(obj.crc32))
  if not info then return fail(m, why) end
  m.data, m.crc, m.total, m.tris, m.verts = data, info.crc, info.total, info.tris, info.verts
  m.sha = util.SHA256 and util.SHA256(data) or nil          -- the content name a client caches it under (no SHA-256: every client is streamed to)
  m.chunks = math.ceil(m.total / Mesh.CHUNK)
  m.state, m.why = "ready", nil
  pcall(hook.Run, "HolodeckBeam", "ready", m)
  FP.watch[m] = nil
  FP.stage(m, "sending", 0)
  tell_entities(m)
  if not Mesh.deliver(m, player.GetAll(), true) then FP.sent(m) end
end

--- One POST /mesh. A "busy" (HTTP 503: the bridge downloads one thing at a time) is not a failure: this model waits its turn and asks again,
--- until the transporter's own timeout gives up on it; any other error is the end of it.
function ask(m, ply)
  local d = Mesh.deps
  return d.post("/mesh", { card = m.card, key = m.key, who = d.key(ply) }, function(body) on_answer(m, body) end, function(reason)
    if m.state == "pending" and tostring(reason):find("HTTP 503", 1, true) and now() - (m.t or 0) < FP.cv_timeout:GetFloat() then
      timer.Simple(Mesh.BUSY_RETRY, function() if m.state == "pending" then ask(m, ply) end end)
      return
    end
    fail(m, reason)
  end, Mesh.REQUEST_TIMEOUT)
end

local function request(m, ply)
  local d = Mesh.deps
  if not d or not cv_fetch:GetBool() then return false end
  if m.tries >= Mesh.MAX_TRIES then return false end
  if not d.allow("mesh", ply) then return false end
  m.tries = m.tries + 1
  m.state, m.t = "pending", now()
  m.asker = ply
  local sent = ask(m, ply)
  if not sent then m.state = "wanted"; m.tries = m.tries - 1; return false end
  if m.state == "pending" then FP.begin(m) end              -- (a synchronous failure has already settled it)
  return true
end

--- A found object has been created: remember it, ask for its model's mesh if nobody has, show it when it is there.
--- `found` is the variant's card fields ({ card =, mesh = <key> }); `ply` is who asked for the object.
function Mesh.attach(ent, found, ply)
  local key = type(found) == "table" and found.mesh
  if type(key) ~= "string" or not key:match("^%x+$") or #key ~= 16 then return false end
  local card = tostring(found.card or "")
  if card == "" or #card > 120 then return false end
  local m = Mesh.by_key[key]
  if not m then
    if Mesh.count >= Mesh.MAX_MESHES then return false end
    Mesh.count = Mesh.count + 1
    m = { key = key, card = card, id = Mesh.count, state = "wanted", tries = 0, ents = {} }
    Mesh.by_key[key] = m
    Mesh.by_id[m.id] = m
  end
  m.ents[#m.ents + 1] = ent
  ent.FxOwner = ply
  if m.state == "ready" then
    if ent.SetVisual then ent:SetVisual("mesh", { id = m.id }) end
    FP.join(m, ent, ply)
  elseif m.state == "wanted" or (m.state == "failed" and now() - (m.t or 0) >= Mesh.RETRY_AFTER) then
    request(m, ply)                                          -- (a request that goes out stages every entity of the model: FP.begin)
  else
    FP.join(m, ent, ply)                                     -- in flight (the entity shows the model's state) or failed a moment ago (it is given up on)
  end
  return true
end

--- Ask now for every model that is waiting (holodeck_mesh was 0 when they were created, or the bridge was busy).
function Mesh.resume(ply)
  local n = 0
  for _, m in pairs(Mesh.by_key) do
    if m.state == "wanted" and request(m, ply) then n = n + 1 end
  end
  return n
end

function Mesh.status()
  local ready, wait, bad = 0, 0, 0
  for _, m in pairs(Mesh.by_key) do
    if m.state == "ready" then ready = ready + 1 elseif m.state == "failed" then bad = bad + 1 else wait = wait + 1 end
  end
  return ready, wait, bad
end

-- ------------------------------------------------------------------------------------------------ clients that ask
local function allowed(ply)
  local key = ply:SteamID64() or tostring(ply)
  local w = Mesh.wants[key]
  local t = now()
  if not w or t - w.t0 >= 60 then w = { t0 = t, n = 0 }; Mesh.wants[key] = w end
  w.n = w.n + 1
  return w.n <= Mesh.WANT_PER_MINUTE
end

net.Receive("holodeck_mesh_want", function(len, ply)
  if not IsValid(ply) then return end
  local id = net.ReadUInt(8)
  if not allowed(ply) then return end
  if id == 0 then
    for i = 1, Mesh.count do
      local m = Mesh.by_id[i]
      if m and m.state == "ready" then Mesh.send(m, { ply }) end
    end
  else
    local m = Mesh.by_id[id]
    if m and m.state == "ready" then Mesh.send(m, { ply }) end
  end
end)

--- A client that keeps a cache says hello once on joining: from now on it is offered models, and it is offered everything already here.
net.Receive("holodeck_mesh_hello", function(len, ply)
  if not IsValid(ply) then return end
  if not allowed(ply) then return end
  Mesh.caps[ply] = true
  for i = 1, Mesh.count do
    local m = Mesh.by_id[i]
    if m and m.state == "ready" then Mesh.deliver(m, { ply }) end
  end
end)

--- The answer to an offer. Only an offer that is outstanding for this player counts (so the message cannot be used to make the server send).
net.Receive("holodeck_mesh_have", function(len, ply)
  if not IsValid(ply) then return end
  local id = net.ReadUInt(8)
  local crc = net.ReadUInt(32)
  local have = net.ReadBool()
  local m = Mesh.by_id[id]
  if not (m and m.state == "ready" and m.crc == crc) then return end
  Mesh.resolve_offer(m, ply, have)
end)

return Mesh
