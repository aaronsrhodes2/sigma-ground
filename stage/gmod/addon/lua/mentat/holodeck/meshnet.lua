-- meshnet.lua -- the server half of "the real shape of a found object" (the other halves: the bridge's POST /mesh, and the client's
-- mentat/holodeck/meshclient.lua). Loaded by mentat/holodeck.lua.
--
--   spawn of a found object whose card has a mesh -> attach(): the bridge is asked for the payload ONCE per model (not per entity);
--   the answer is checked (size cap, CRC-32, LZMA header, SVM1 decode) and kept; it is sent to every client in 16 KB chunks, one chunk
--   per 0.1 s; every entity of that model is told the model's number (ent:SetVisual("mesh", { id = n })); a client that joins later
--   (or missed a chunk) asks with "holodeck_mesh_want" and is sent it alone. A failure of any kind leaves the box.
--
-- Wire (server -> client, net message "holodeck_mesh", one per chunk):
--   UInt8 mesh number (1..64) | UInt8 chunk number (1-based) | UInt8 chunk count | UInt20 total bytes | UInt32 CRC-32 of the whole
--   payload | UInt16 bytes in this chunk | Data. The payload is the bridge's LZMA bytes as they came (util.Decompress reads them).
-- Wire (client -> server, "holodeck_mesh_want"): UInt8 mesh number, 0 = every mesh the server has.
local SVM1 = include("mentat/holodeck/svm1.lua")

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

--- deps = { post = bridge_post, allow = bridge_allow, key = player_key }
function Mesh.init(deps) Mesh.deps = deps end

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
  local dec, why = SVM1.decode(raw)
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
  if #alive == 0 or not m.data or job.i > m.chunks then table.remove(Mesh.queue, 1); return end
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
  if job.i > m.chunks then table.remove(Mesh.queue, 1) end
end

--- Queue a ready model for these players. Returns false when the queue is full or there is nobody to send to.
function Mesh.send(m, targets)
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
  Mesh.queue[#Mesh.queue + 1] = { m = m, targets = list, i = 1 }
  if not timer.Exists("holodeck_mesh_pump") then timer.Create("holodeck_mesh_pump", Mesh.PUMP, 0, pump) end
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
end

local function request(m, ply)
  local d = Mesh.deps
  if not d or not cv_fetch:GetBool() then return false end
  if m.tries >= Mesh.MAX_TRIES then return false end
  if not d.allow("mesh", ply) then return false end
  m.tries = m.tries + 1
  m.state, m.t = "pending", now()
  local sent = d.post("/mesh", { card = m.card, key = m.key, who = d.key(ply) }, function(body)
    local obj = util.JSONToTable(body or "")
    if not (istable(obj) and obj.ok == true and obj.key == m.key and obj.format == "svm1-lzma" and isstring(obj.b64)) then
      return fail(m, "the bridge's answer was not a mesh")
    end
    if #obj.b64 > math.ceil(Mesh.CAP_BYTES * 4 / 3) + 8 then return fail(m, "the bridge's payload is over the cap") end
    local data = SVM1.b64decode(obj.b64)
    if not data then return fail(m, "the payload was not base64") end
    local info, why = Mesh.check(data, tonumber(obj.bytes), tonumber(obj.crc32))
    if not info then return fail(m, why) end
    m.data, m.crc, m.total, m.tris, m.verts = data, info.crc, info.total, info.tris, info.verts
    m.chunks = math.ceil(m.total / Mesh.CHUNK)
    m.state, m.why = "ready", nil
    tell_entities(m)
    Mesh.send(m, player.GetAll())
  end, function(reason) fail(m, reason) end, Mesh.REQUEST_TIMEOUT)
  if not sent then m.state = "wanted"; m.tries = m.tries - 1; return false end
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
  if m.state == "ready" then
    if ent.SetVisual then ent:SetVisual("mesh", { id = m.id }) end
  elseif m.state == "wanted" or (m.state == "failed" and now() - (m.t or 0) >= Mesh.RETRY_AFTER) then
    request(m, ply)
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
net.Receive("holodeck_mesh_want", function(len, ply)
  if not IsValid(ply) then return end
  local id = net.ReadUInt(8)
  local key = ply:SteamID64() or tostring(ply)
  local w = Mesh.wants[key]
  local t = now()
  if not w or t - w.t0 >= 60 then w = { t0 = t, n = 0 }; Mesh.wants[key] = w end
  w.n = w.n + 1
  if w.n > Mesh.WANT_PER_MINUTE then return end
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

return Mesh
