-- avatar/body_net.lua -- the server half of "the avatar wears the REAL Skippy": read the private payloads (manifest.json and the six
-- SVM2 parts) from a folder under garrysmod/data/, check every one, and send them to the clients once, in chunks. Server only.
--
--   Where the files are:  garrysmod/data/<mentat_avatar_body_dir>/   (convar, default "mentat/skippy" -> garrysmod/data/mentat/skippy/)
--     manifest.json  can.svm2  jacket.svm2  head.svm2  hat.svm2  hand_a.svm2  hand_b.svm2
--   stage/gmod/install_skippy_body.sh copies them there from stage/skippyverse/private/skippy/ (gitignored, never committed) for the
--   game and the dedicated server. A cloud server gets the same files copied by hand, outside git. No files, or any file bad: this
--   module reports why, the global "mentat_avatar_body" stays false, nothing is sent, and every client keeps the procedural can.
--
--   Checks per part (the same as the found-mesh relay, meshnet.lua): size cap, CRC-32, the LZMA header's declared size, util.Decompress,
--   and a full SVM2 decode. The manifest: size cap, JSON, and the pivots the animation needs.
--
-- Wire (server -> client, "mentat_avatar_body", one message per chunk, one chunk per 0.1 s):
--   UInt8 item (1 = manifest.json, 2..7 = the parts in Body.PARTS order) | UInt8 chunk (1-based) | UInt8 chunk count | UInt20 total bytes
--   | UInt32 CRC-32 of the whole item | UInt16 bytes in this chunk | Data
-- Wire (client -> server, "mentat_avatar_body_want"): UInt8 item, 0 = everything. A client that joins late, or missed a chunk, asks.
local SVM1 = include("mentat/holodeck/svm1.lua")
local Body = include("mentat/avatar/body.lua")

local N = {}
N.CAP_BYTES = 256 * 1024
N.MAX_RAW = 1024 * 1024
N.CHUNK = 16 * 1024
N.PUMP = 0.1
N.MAX_QUEUED = 24
N.WANT_PER_MINUTE = 12
N.items = nil                    -- [1..7] = { name =, data =, crc =, total =, chunks = }
N.ok, N.why, N.tried = false, "not loaded yet", false
N.queue, N.wants, N.sent_chunks = {}, {}, 0

local cv_dir = CreateConVar("mentat_avatar_body_dir", "mentat/skippy", FCVAR_ARCHIVE, "folder under garrysmod/data/ with the private Skippy body files (manifest.json and the six .svm2 parts)")
local cv_on = CreateConVar("mentat_avatar_body", "1", FCVAR_ARCHIVE, "1 = send the real Skippy body to clients when its files are there; 0 = always the procedural can")
N.cv_dir, N.cv_on = cv_dir, cv_on

util.AddNetworkString("mentat_avatar_body")
util.AddNetworkString("mentat_avatar_body_want")

local function fail(why)
  N.items, N.ok, N.why = nil, false, tostring(why)
  SetGlobalBool("mentat_avatar_body", false)
  return false, N.why
end

--- Check one LZMA SVM2 part. Returns { tris =, verts = } or nil and why.
function N.check_part(data)
  if type(data) ~= "string" or #data == 0 then return nil, "empty" end
  if #data > N.CAP_BYTES then return nil, "over the " .. N.CAP_BYTES .. " byte cap" end
  local size = SVM1.lzma_size(data)
  if not size or size < 40 or size > N.MAX_RAW then return nil, "the LZMA header declares no usable size" end
  local ok, raw = pcall(util.Decompress, data)
  if not ok or type(raw) ~= "string" or #raw ~= size then return nil, "util.Decompress did not give the declared bytes" end
  local dec, why = SVM1.decode(raw, true)
  if not dec then return nil, "not a mesh: " .. tostring(why) end
  if not dec.svm2 then return nil, "no normals (not SVM2)" end
  return { tris = dec.nt, verts = dec.nv }
end

local function safe_dir(d)
  d = tostring(d or ""):gsub("\\", "/"):gsub("^/+", ""):gsub("/+$", "")
  if d == "" or d:find("..", 1, true) or d:find("[^%w_/%-%.]") then return nil end
  return d
end

local function item(name, data)
  return { name = name, data = data, crc = tonumber(util.CRC(data)), total = #data, chunks = math.ceil(#data / N.CHUNK) }
end

--- Read and check everything. Returns true, or false and why. Sets the global flag the clients read.
function N.load()
  N.tried = true
  if not cv_on:GetBool() then return fail("switched off (mentat_avatar_body 0)") end
  local dir = safe_dir(cv_dir:GetString())
  if not dir then return fail("mentat_avatar_body_dir is not a plain folder name") end
  local mtext = file.Read(dir .. "/manifest.json", "DATA")
  if type(mtext) ~= "string" or mtext == "" then return fail("no " .. dir .. "/manifest.json in the data folder: the body files are not installed") end
  if #mtext > Body.MAX_MANIFEST then return fail("manifest.json is over the size cap") end
  local m = util.JSONToTable(mtext)
  local piv, why = Body.parse_manifest(m)
  if not piv then return fail("manifest.json: " .. tostring(why)) end
  local items = { item("manifest.json", mtext) }
  for i, name in ipairs(Body.PARTS) do
    local data = file.Read(dir .. "/" .. name .. ".svm2", "DATA")
    if type(data) ~= "string" then return fail(name .. ".svm2 is missing") end
    local info, w2 = N.check_part(data)
    if not info then return fail(name .. ".svm2: " .. tostring(w2)) end
    items[i + 1] = item(name .. ".svm2", data)
  end
  N.items, N.ok, N.why = items, true, nil
  SetGlobalBool("mentat_avatar_body", true)
  return true
end

--- Load once (the first call), return whether the body is available.
function N.ensure()
  if not N.tried then N.load() end
  return N.ok
end

-- ------------------------------------------------------------------------------------------------ sending
local function pump()
  local job = N.queue[1]
  if not job then timer.Remove("mentat_avatar_body_pump"); return end
  local alive = {}
  for _, p in ipairs(job.targets) do if IsValid(p) then alive[#alive + 1] = p end end
  job.targets = alive
  local it = job.item
  if #alive == 0 or not N.items or job.i > it.chunks then table.remove(N.queue, 1); return end
  local piece = it.data:sub((job.i - 1) * N.CHUNK + 1, job.i * N.CHUNK)
  net.Start("mentat_avatar_body")
  net.WriteUInt(job.index, 8)
  net.WriteUInt(job.i, 8)
  net.WriteUInt(it.chunks, 8)
  net.WriteUInt(it.total, 20)
  net.WriteUInt(it.crc, 32)
  net.WriteUInt(#piece, 16)
  net.WriteData(piece, #piece)
  net.Send(alive)
  N.sent_chunks = N.sent_chunks + 1
  job.i = job.i + 1
  if job.i > it.chunks then table.remove(N.queue, 1) end
end

local function humans(list)
  local out = {}
  for _, p in ipairs(list) do if IsValid(p) and not (p.IsBot and p:IsBot()) then out[#out + 1] = p end end
  return out
end

--- Queue items (index, or nil for all of them) for these players. False when there is nothing to send or nobody to send it to.
function N.send(targets, index)
  if not N.ok or not N.items then return false end
  local list = humans(targets)
  if #list == 0 then return false end
  for i, it in ipairs(N.items) do
    if index == nil or index == i then
      if #N.queue >= N.MAX_QUEUED then return false end
      N.queue[#N.queue + 1] = { item = it, index = i, targets = list, i = 1 }
    end
  end
  if not timer.Exists("mentat_avatar_body_pump") then timer.Create("mentat_avatar_body_pump", N.PUMP, 0, pump) end
  return true
end

--- (Re)load the files and, when they are good, send them to everyone.
function N.reload()
  N.queue = {}
  local ok, why = N.load()
  if ok then N.send(player.GetAll()) end
  return ok, why
end

net.Receive("mentat_avatar_body_want", function(len, ply)
  if not IsValid(ply) then return end
  local index = net.ReadUInt(8)
  local key = ply:SteamID64() or tostring(ply)
  local w = N.wants[key]
  local t = CurTime()
  if not w or t - w.t0 >= 60 then w = { t0 = t, n = 0 }; N.wants[key] = w end
  w.n = w.n + 1
  if w.n > N.WANT_PER_MINUTE then return end
  if not N.ok or not N.items then return end
  if index == 0 then N.send({ ply })
  elseif N.items[index] then N.send({ ply }, index) end
end)

concommand.Add("mentat_avatar_body_reload", function(ply)
  if IsValid(ply) and not ply:IsSuperAdmin() then return end
  local ok, why = N.reload()
  print("[avatar] body files: " .. (ok and "loaded and sent" or ("not available: " .. tostring(why))))
end, nil, "Re-read the private Skippy body files and send them to the clients")

-- loaded a moment after start so a joining client finds the flag already set
timer.Simple(1, function() if not N.tried then N.load() end end)

return N
