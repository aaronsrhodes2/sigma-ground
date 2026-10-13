-- meshcache.lua -- CLIENT. The between-sessions copy of the found-object payloads this machine has received, so a returning player downloads
-- nothing for a model they already have (README "Object caches"). Loaded by meshclient.lua.
--
--   data/mentat/meshcache/<first 32 hex of the payload's SHA-256>.dat    the payload exactly as the server sent it (the LZMA bytes)
--   data/mentat/meshcache/index.json                                      { v = 1, tick = N, entries = [ { sha, bytes, crc, u } ] }
--
-- Engine rules obeyed (stage/gmod/holodeck/README.md "Persistence rules of the engine"): only the extensions .dat and .json are written,
-- the directory is made first, a write is read back and compared, no temp or backup file of another extension, and no SteamID anywhere
-- (this folder belongs to the client's own machine; there is nothing to key by player). `u` is a use counter (larger = used more recently), not a
-- clock, so eviction order is exact and survives a clock change.
-- Every read is checked against the length, the CRC-32 and the SHA-256 the server announced; any mismatch deletes the entry and answers "not here".
local Persist = include("mentat/persist.lua")

local C = {}
C.DIR = "mentat/meshcache"
C.INDEX = C.DIR .. "/index.json"
C.CAP_BYTES = 120 * 1024 * 1024         -- this cache counts toward the player's 200 MB download budget; it never exceeds this
C.MAX_ENTRY = 256 * 1024                -- the payload cap of the protocol
C.MAX_ENTRIES = 512
C.idx = nil

local function parse_index(text)
  local t = util.JSONToTable(text or "")
  if type(t) ~= "table" or t.v ~= 1 or type(t.entries) ~= "table" then return nil end
  local out = { tick = tonumber(t.tick) or 0, entries = {} }
  for _, e in ipairs(t.entries) do
    if type(e) == "table" and type(e.sha) == "string" and e.sha:match("^%x+$") and #e.sha == 64 and tonumber(e.bytes) and tonumber(e.crc) then
      out.entries[#out.entries + 1] = { sha = e.sha, bytes = tonumber(e.bytes), crc = tonumber(e.crc), u = tonumber(e.u) or 0 }
      out.tick = math.max(out.tick, tonumber(e.u) or 0)
    end
  end
  return out
end

local function index()
  if C.idx then return C.idx end
  C.idx = Persist.load(C.INDEX, parse_index) or { tick = 0, entries = {} }
  return C.idx
end

local function save()
  local idx = index()
  local list = {}
  for i, e in ipairs(idx.entries) do list[i] = { sha = e.sha, bytes = e.bytes, crc = e.crc, u = e.u } end
  return Persist.write(C.INDEX, util.TableToJSON({ v = 1, tick = idx.tick, entries = list }), function(old) return parse_index(old) ~= nil end)
end

function C.path(sha) return C.DIR .. "/" .. sha:sub(1, 32) .. ".dat" end
local function sha_ok(sha) return type(sha) == "string" and #sha == 64 and sha:match("^%x+$") ~= nil end

local function find(sha)
  for i, e in ipairs(index().entries) do if e.sha == sha then return e, i end end
end

function C.total_bytes()
  local n = 0
  for _, e in ipairs(index().entries) do n = n + e.bytes end
  return n
end

function C.count() return #index().entries end

--- Forget an entry and delete its file.
function C.drop(sha)
  local e, i = find(sha)
  if e then table.remove(index().entries, i) end
  if sha_ok(sha) then file.Delete(C.path(sha)) end
  return e ~= nil
end

--- The payload cached under `sha` when its bytes are exactly the announced ones; nil and why otherwise (a damaged entry is dropped).
function C.lookup(sha, total, crc)
  if not sha_ok(sha) then return nil, "bad name" end
  local e = find(sha)
  if not e then return nil, "not cached" end
  local blob = file.Read(C.path(sha), "DATA")
  if type(blob) ~= "string" or #blob ~= total or e.bytes ~= total or e.crc ~= crc
     or tonumber(util.CRC(blob)) ~= crc or util.SHA256(blob) ~= sha then
    C.drop(sha)
    save()
    return nil, "damaged"
  end
  local idx = index()
  idx.tick = idx.tick + 1
  e.u = idx.tick
  save()
  return blob
end

--- Delete the least recently used entries until the cache fits; `keep` is never chosen.
function C.evict(keep)
  local idx = index()
  local n = 0
  while #idx.entries > 0 and (C.total_bytes() > C.CAP_BYTES or #idx.entries > C.MAX_ENTRIES) do
    local oldest, oi
    for i, e in ipairs(idx.entries) do
      if e.sha ~= keep and (not oldest or e.u < oldest.u) then oldest, oi = e, i end
    end
    if not oldest then break end
    C.drop(oldest.sha)
    n = n + 1
  end
  return n
end

--- Keep a payload the client has just received and verified. Returns true when it is on disk (read back) and in the index.
function C.store(sha, blob, crc)
  if not sha_ok(sha) or type(blob) ~= "string" or #blob == 0 or #blob > C.MAX_ENTRY or #blob > C.CAP_BYTES then return false end
  file.CreateDir(C.DIR)
  local path = C.path(sha)
  file.Write(path, blob)
  if file.Read(path, "DATA") ~= blob then file.Delete(path); return false end
  local idx = index()
  local e = find(sha)
  idx.tick = idx.tick + 1
  if e then e.u, e.bytes, e.crc = idx.tick, #blob, crc
  else idx.entries[#idx.entries + 1] = { sha = sha, bytes = #blob, crc = crc, u = idx.tick } end
  C.evict(sha)
  save()
  return true
end

--- Delete everything this cache holds.
function C.clear()
  local idx = index()
  local n = #idx.entries
  for i = #idx.entries, 1, -1 do file.Delete(C.path(idx.entries[i].sha)) end
  idx.entries = {}
  save()
  return n
end

--- Forget the in-memory index (the next call reads the folder again); for tests and for `holodeck_mesh_cache_reload`.
function C.reset() C.idx = nil end

return C
