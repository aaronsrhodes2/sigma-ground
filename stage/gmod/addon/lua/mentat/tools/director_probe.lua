-- director_probe.lua -- checks of the skippy director's engine-facing pieces that the offline mock cannot prove (run on the dedicated server, no window):
-- the real file API under the journal (CreateDir, Append, Size, Rename, Read), util.TableToJSON / JSONToTable on its events and on SteamID64 keys, the notes
-- from them, and what Player:IsSpeaking does on a server. args: none. Result: a table of what the engine did. (A finding from the first live pass: see AVATAR.md.)
return function(args, done, ply)
  local Dr = Mentat and Mentat.Director
  if not Dr then return done(nil, "no director") end
  local J, Notes = Dr.Journal, Dr.Notes
  local sid = "76561190000000042"
  local out = {}
  J.delete_all(sid)
  out.before = file.Exists("mentat/journal/" .. sid .. ".txt", "DATA")
  out.log1 = J.log(sid, { k = "add", name = "couch", n = 3, src = "curated" }, os.time())
  out.log2 = J.log(sid, { k = "beam_fail", name = "kettle", why = "x" }, os.time())
  out.log3 = J.log(sid, { k = "add", name = "couch", n = 1 }, os.time())
  out.exists = file.Exists("mentat/journal/" .. sid .. ".txt", "DATA")
  out.size = file.Size("mentat/journal/" .. sid .. ".txt", "DATA")
  out.raw = file.Read("mentat/journal/" .. sid .. ".txt", "DATA")
  local evs = J.read(sid, os.time())
  out.read_n = #evs
  local res = Notes.summarise(evs, os.time(), { banned = {} })
  out.notes = res.notes
  out.facts = res.facts
  local old = J.MAX_BYTES
  J.MAX_BYTES = 200
  for i = 1, 12 do J.log(sid, { k = "add", name = "chair", n = 1 }, os.time()) end
  J.MAX_BYTES = old
  out.rotated = file.Exists("mentat/journal/" .. sid .. ".1.txt", "DATA")
  out.after_rotate = #J.read(sid, os.time())
  out.forget_that = J.forget(sid, "that", os.time())
  out.forget_all = J.forget(sid, "all", os.time())
  out.gone = not file.Exists("mentat/journal/" .. sid .. ".txt", "DATA") and not file.Exists("mentat/journal/" .. sid .. ".1.txt", "DATA")
  -- the file API, step by step (what exactly fails when Append makes nothing)
  local step = {}
  file.CreateDir("mentat/journal")
  step.dir_exists = file.IsDir("mentat/journal", "DATA")
  file.Write("mentat/journal/probe_w.txt", "one\n")
  step.write_exists = file.Exists("mentat/journal/probe_w.txt", "DATA")
  file.Append("mentat/journal/probe_w.txt", "two\n")
  step.after_append = file.Read("mentat/journal/probe_w.txt", "DATA")
  file.Append("mentat/journal/probe_a.txt", "fresh\n")
  step.append_creates = file.Exists("mentat/journal/probe_a.txt", "DATA")
  file.Append("mentat/probe_a2.txt", "fresh\n")
  step.append_creates_in_mentat = file.Exists("mentat/probe_a2.txt", "DATA")
  step.size_w = file.Size("mentat/journal/probe_w.txt", "DATA")
  file.Delete("mentat/journal/probe_w.txt"); file.Delete("mentat/journal/probe_a.txt"); file.Delete("mentat/probe_a2.txt")
  out.step = step
  -- string keys that look like numbers (SteamID64s) through TableToJSON / JSONToTable
  local txt = util.TableToJSON({ players = { [sid] = { roast = "savage" } } })
  out.json_text = txt
  local t1 = util.JSONToTable(txt)
  local t2 = util.JSONToTable(txt, true)
  local function keys(t) local o = {} for k in pairs(t.players or {}) do o[#o + 1] = type(k) .. ":" .. string.format("%.0f", tonumber(k) or 0) .. ":" .. tostring(k) end return o end
  out.keys_default = keys(t1)
  out.keys_ignore_conversions = t2 and keys(t2) or "nil"
  -- which file extensions the engine lets a script write into data/ (a refused one is silently ignored)
  out.ext = {}
  for _, ext in ipairs({ "txt", "json", "jsonl", "tmp", "dat", "csv", "log" }) do
    file.Write("mentat/probe_ext." .. ext, "x")
    out.ext[ext] = file.Exists("mentat/probe_ext." .. ext, "DATA")
    file.Delete("mentat/probe_ext." .. ext)
  end
  -- a SteamID64 list as an array of records (a string VALUE is never converted)
  local arr = util.JSONToTable(util.TableToJSON({ players = { { sid = sid, roast = "savage" } } }))
  out.array_sid = arr and arr.players and arr.players[1] and arr.players[1].sid
  out.array_sid_ok = out.array_sid == sid
  local anyone = player.GetAll()[1]
  if anyone then local ok, v = pcall(function() return anyone:IsSpeaking() end); out.is_speaking_server = { ok, tostring(v) } end
  out.os_date = os.date("%Y-%m-%d %H")
  done(out)
end
