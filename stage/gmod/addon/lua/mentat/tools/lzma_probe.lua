-- lzma_probe.lua -- does util.Decompress accept LZMA made by Python's lzma module, and what does util.Compress produce?
-- args: { hex = { name = "<hex of a candidate>", ... }, expect_len = N, expect_sum = N }
local function tohex(s) return (s:gsub(".", function(c) return string.format("%02x", c:byte()) end)) end
local function fromhex(h) return (h:gsub("%x%x", function(c) return string.char(tonumber(c, 16)) end)) end
local function sum(s) local t = 0; for i = 1, #s do t = (t + s:byte(i) * i) % 4294967291 end; return t end

return function(args, done)
  local out = { gmod = {}, tries = {} }
  local sample = string.rep("holodeck mesh payload 0123456789 ", 20)
  local c = util.Compress(sample)
  out.gmod.compress_hex = c and tohex(c:sub(1, 40)) or "nil"
  out.gmod.compress_len = c and #c or -1
  out.gmod.sample_len = #sample
  out.gmod.roundtrip = (c and util.Decompress(c) == sample) or false
  for name, h in pairs(args.hex or {}) do
    local ok, r = pcall(util.Decompress, fromhex(h))
    local t = { ok = ok, type = type(r) }
    if ok and type(r) == "string" then t.len = #r; t.sum = sum(r) end
    out.tries[name] = t
  end
  done(out)
end
