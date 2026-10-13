-- persist.lua -- how this addon keeps a file in data/. One place for the engine's rules, found live (2026-10-12) and listed in
-- stage/gmod/holodeck/README.md "Persistence rules of the engine":
--   1. file.Write / file.Append / file.Rename silently do NOTHING for an extension outside txt, dat, json, xml, csv, jpg, jpeg, png, vtf, vmt, mp3, wav, ogg.
--      So no ".tmp", ".bak" or ".jsonl" file, ever. A backup is "name.bak.txt".
--   2. util.JSONToTable turns a numeric-looking object KEY into a number, and a SteamID64 (17 digits, over 2^53) loses its low digits that way. So a
--      SteamID64 is only ever a string VALUE ({ sid = "7656..." } inside a list), never a key and never a number.
--   3. A write into a directory nobody made with file.CreateDir is dropped (CreateDir makes the whole path).
-- Writing is therefore direct, then read back and compared. The previous good copy goes to <name>.bak.txt first, and a load falls back to it.
local P = {}
P.MAX = 2 * 1024 * 1024               -- bytes read or written; a bigger file is not ours

--- A SteamID64 as the code may hold it: a string of at most 20 digits.
function P.sid_ok(sid) return type(sid) == "string" and #sid <= 20 and sid:match("^%d+$") ~= nil end

--- "mentat/faces.json" -> "mentat/faces.bak.txt" (an extension the engine allows).
function P.bak_name(path) return (path:gsub("%.%w+$", "")) .. ".bak.txt" end

--- Write `payload` to `path` and read it back. false when the engine did not keep it exactly.
--- `valid(text)` (optional) says whether the file that is there now is worth keeping as the backup.
function P.write(path, payload, valid)
  if type(payload) ~= "string" or #payload > P.MAX then return false end
  local dir = path:match("^(.*)/[^/]*$")
  if dir then file.CreateDir(dir) end
  local old = file.Read(path, "DATA")
  if old and old ~= "" and old ~= payload and (not valid or valid(old)) then
    file.Write(P.bak_name(path), old)
  end
  file.Write(path, payload)
  return file.Read(path, "DATA") == payload
end

--- The first of path, then its backup, that `parse(text)` accepts (parse returns nil/false for a bad file). -> value, path used.
function P.load(path, parse)
  for _, p in ipairs({ path, P.bak_name(path) }) do
    local raw = file.Read(p, "DATA")
    if raw and raw ~= "" and #raw <= P.MAX then
      local v = parse(raw)
      if v then return v, p end
    end
  end
  return nil
end

--- Entries of a legacy table that the engine keyed by a number (a rounded SteamID64) cannot be recovered; the caller counts them with this and logs once.
function P.note_dropped(what, n)
  if n and n > 0 then print("[mentat] " .. what .. ": " .. n .. " old entries had a rounded SteamID64 key and were dropped (they cannot be recovered)") end
end

return P
