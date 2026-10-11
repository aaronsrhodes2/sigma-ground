-- svm1.lua -- decode and check an SVM1 mesh payload (stage/skippyverse/meshpack.py). Shared: the server checks what the bridge sent
-- before it relays it, every client checks what the server relayed before it builds a mesh. Pure Lua (no engine calls), so it is
-- tested offline against the Python unpacker on a real model.
--
-- The payload AFTER decompression (little endian):
--   'SVM1' | u16 n_verts | u16 n_tris | f32 min[3] | f32 max[3] | u16 verts[n_verts][3] | u16 idx[n_tris][3] | u8 rgb[n_tris][3]
-- Vertices are quantised across min..max; metres, +Z up, origin on the floor under the footprint centre. The colour is flat, per triangle.
local SVM1 = {}

SVM1.MAX_TRIS = 4000          -- more than meshpack's target ever makes (1500); a client builds at most this
SVM1.MAX_TRIS2 = 8000         -- the body's parts (SVM2) are finer: the head is 4200
SVM1.MAX_VERTS = 12000        -- and never more than 3 per triangle
SVM1.MAX_EXTENT_M = 1000      -- no coordinate beyond this (a bad float is not a model)
SVM1.HEADER = 32

--- A little-endian IEEE-754 single from four bytes, or nil for NaN / infinity.
local function f32(b1, b2, b3, b4)
  local sign = b4 >= 128 and -1 or 1
  local e = (b4 % 128) * 2 + math.floor(b3 / 128)
  local m = (b3 % 128) * 65536 + b2 * 256 + b1
  if e == 255 then return nil end
  if e == 0 then return sign * m * 2 ^ -149 end
  return sign * (1 + m / 8388608) * 2 ^ (e - 127)
end
SVM1.f32 = f32

--- Decode `raw` (the decompressed payload). `allow2`: also accept SVM2 (SVM1 plus `i8 normals[nv][3]` after the colours; the avatar's body
--- uses it, a found object never does). Returns a table or nil and why:
---   { nv, nt, min = {x,y,z}, max = {x,y,z}, x = {}, y = {}, z = {}  (metres, 1-based per vertex),
---     a = {}, b = {}, c = {}  (vertex numbers, 1-based, per triangle), r = {}, g = {}, bl = {}  (0-255 per triangle) }
function SVM1.decode(raw, allow2)
  if type(raw) ~= "string" then return nil, "no data" end
  local n = #raw
  if n < SVM1.HEADER then return nil, "too short" end
  local magic = raw:sub(1, 4)
  local two = allow2 == true and magic == "SVM2"
  if magic ~= "SVM1" and not two then return nil, "not an SVM1 payload" end
  local byte = string.byte
  local nv = byte(raw, 5) + byte(raw, 6) * 256
  local nt = byte(raw, 7) + byte(raw, 8) * 256
  if nt < 1 or nt > (two and SVM1.MAX_TRIS2 or SVM1.MAX_TRIS) then return nil, "triangle count " .. nt .. " out of range" end
  if nv < 3 or nv > SVM1.MAX_VERTS or nv > 3 * nt then return nil, "vertex count " .. nv .. " out of range" end
  if n ~= SVM1.HEADER + 6 * nv + 6 * nt + 3 * nt + (two and 3 * nv or 0) then return nil, "length " .. n .. " does not match the counts" end
  local mn, mx = {}, {}
  for k = 0, 2 do
    local o = 9 + 4 * k
    mn[k + 1] = f32(byte(raw, o, o + 3))
    local p = 21 + 4 * k
    mx[k + 1] = f32(byte(raw, p, p + 3))
    if not mn[k + 1] or not mx[k + 1] then return nil, "bad bounds" end
    if mn[k + 1] > mx[k + 1] or math.abs(mn[k + 1]) > SVM1.MAX_EXTENT_M or math.abs(mx[k + 1]) > SVM1.MAX_EXTENT_M then return nil, "bad bounds" end
  end
  local X, Y, Z = {}, {}, {}
  local o = SVM1.HEADER
  local sx, sy, sz = mx[1] - mn[1], mx[2] - mn[2], mx[3] - mn[3]
  for i = 1, nv do
    local q1 = byte(raw, o + 1) + byte(raw, o + 2) * 256
    local q2 = byte(raw, o + 3) + byte(raw, o + 4) * 256
    local q3 = byte(raw, o + 5) + byte(raw, o + 6) * 256
    X[i] = mn[1] + sx * q1 / 65535
    Y[i] = mn[2] + sy * q2 / 65535
    Z[i] = mn[3] + sz * q3 / 65535
    o = o + 6
  end
  local A, B, C = {}, {}, {}
  for i = 1, nt do
    local a = byte(raw, o + 1) + byte(raw, o + 2) * 256
    local b = byte(raw, o + 3) + byte(raw, o + 4) * 256
    local c = byte(raw, o + 5) + byte(raw, o + 6) * 256
    if a >= nv or b >= nv or c >= nv then return nil, "triangle " .. i .. " names a vertex that does not exist" end
    if a == b or b == c or a == c then return nil, "triangle " .. i .. " is degenerate" end
    A[i], B[i], C[i] = a + 1, b + 1, c + 1
    o = o + 6
  end
  local R, G, BL = {}, {}, {}
  for i = 1, nt do
    R[i], G[i], BL[i] = byte(raw, o + 1), byte(raw, o + 2), byte(raw, o + 3)
    o = o + 3
  end
  local dec = { nv = nv, nt = nt, min = mn, max = mx, x = X, y = Y, z = Z, a = A, b = B, c = C, r = R, g = G, bl = BL }
  if two then
    local NX, NY, NZ = {}, {}, {}
    local function s8(v) return (v >= 128 and v - 256 or v) / 127 end
    for i = 1, nv do
      NX[i], NY[i], NZ[i] = s8(byte(raw, o + 1)), s8(byte(raw, o + 2)), s8(byte(raw, o + 3))
      o = o + 3
    end
    dec.nx, dec.ny, dec.nz, dec.svm2 = NX, NY, NZ, true
  end
  return dec
end

--- The size the LZMA stream declares for its output, or nil: Garry's Mod's util.Compress / util.Decompress use the legacy ".lzma"
--- framing (5 bytes of properties, a u64 size, then the stream), so the size is readable before anything is inflated. Used to refuse
--- a "bomb" (a tiny payload that claims a huge result) and an unknown size (which util.Decompress cannot read at all).
function SVM1.lzma_size(blob)
  if type(blob) ~= "string" or #blob < 14 then return nil end
  local b = { string.byte(blob, 6, 13) }
  for i = 5, 8 do if b[i] ~= 0 then return nil end end          -- a size over 4 GB (or -1, "unknown") is not ours
  return b[1] + b[2] * 256 + b[3] * 65536 + b[4] * 16777216
end

--- base64 -> bytes (standard alphabet, padding optional). nil for anything else.
local B64 = {}
do
  local alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  for i = 1, #alphabet do B64[alphabet:byte(i)] = i - 1 end
end
function SVM1.b64decode(s)
  if type(s) ~= "string" or #s > 4 * 1024 * 1024 then return nil end
  s = s:gsub("=+$", "")
  if s:find("[^%w%+/]") then return nil end
  local out, n = {}, 0
  local byte, char = string.byte, string.char
  local len = #s
  local i = 1
  while i + 3 <= len do
    local a, b, c, d = B64[byte(s, i)], B64[byte(s, i + 1)], B64[byte(s, i + 2)], B64[byte(s, i + 3)]
    local v = a * 262144 + b * 4096 + c * 64 + d
    n = n + 1
    out[n] = char(math.floor(v / 65536), math.floor(v / 256) % 256, v % 256)
    i = i + 4
  end
  local rest = len - i + 1
  if rest == 1 then return nil end
  if rest == 2 then
    local v = B64[byte(s, i)] * 64 + B64[byte(s, i + 1)]
    n = n + 1; out[n] = char(math.floor(v / 16))
  elseif rest == 3 then
    local v = B64[byte(s, i)] * 4096 + B64[byte(s, i + 1)] * 64 + B64[byte(s, i + 2)]
    n = n + 1; out[n] = char(math.floor(v / 1024), math.floor(v / 4) % 256)
  end
  return table.concat(out)
end

return SVM1
