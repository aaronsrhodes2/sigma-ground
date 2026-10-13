-- persist_probe.lua -- what the REAL engine does to every file this addon persists, through the addon's own code paths (dedicated server, no window).
-- args: { phase = "write" | "read", sid = "<made-up 17 digit id>", bot_sid = "<the bot's id from the write phase>" }. The driver (holodeck/persist_probe.py) runs "write", STOPS the server, starts it again and
-- runs "read": the second run proves the data survived a restart (each module loads its own file when the server starts).
-- Phase write : wake names (holodeck/wakenames.lua), faces (faces_server.lua), skippy prefs (avatar/director.lua) and the journal (avatar/journal.lua) each store
--               a record for a made-up SteamID64 of real magnitude (past 2^53) AND for a bot's SteamID64; also probes the engine's file extension rules and
--               util.JSONToTable's number-key conversion, so the doc rests on what this engine does.
-- Phase read  : after the restart, reads the same records back through the modules (what they loaded by themselves), then feeds the OLD keyed file
--               formats to the loaders to prove they drop rounded keys with a log line instead of crashing.
-- Every check is a plain true/false in the result, with the evidence next to it. Nothing here touches a real player's record.
return function(args, done)
  local phase = args.phase or "write"
  local SID = tostring(args.sid or "76561197912345678")
  local out = { phase = phase, sid = SID, checks = {} }
  local function check(name, ok, evidence) out.checks[name] = { ok = ok == true, evidence = evidence } end
  local function head(p) local r = file.Read(p, "DATA"); return r and (#r .. " bytes: " .. r:sub(1, 160)) or "(none)" end

  local H, Dr, Faces = Holodeck, Mentat and Mentat.Director, MentatFaces
  if not (H and H.Wake and Dr and Faces) then
    return done(nil, "missing module(s): Holodeck.Wake=" .. tostring(H and H.Wake ~= nil) .. " Director=" .. tostring(Dr ~= nil) .. " Faces=" .. tostring(Faces ~= nil))
  end
  local Wake, J = H.Wake, Dr.Journal

  -- a stand-in player for the made-up SteamID64 (the real code only asks IsValid / IsPlayer / SteamID64); net.Send is muted while it is used
  local fake = { IsValid = function() return true end, IsPlayer = function() return true end, SteamID64 = function() return SID end, Nick = function() return "probe" end }
  local bot = nil
  if phase == "write" then
    local okb, b = pcall(function() return player.CreateNextBot("PersistProbe") end)
    bot = okb and IsValid(b) and b or nil
  end
  local BOT = bot and tostring(bot:SteamID64()) or (args.bot_sid ~= nil and tostring(args.bot_sid)) or nil
  out.bot_sid = BOT
  local sids = { SID }
  if BOT then sids[#sids + 1] = BOT end

  if phase == "write" then
    -- ------------------------------------------------------------------ the engine's rules, measured
    file.CreateDir("mentat/persist_probe")
    out.ext = {}
    for _, ext in ipairs({ "txt", "dat", "json", "xml", "csv", "jpg", "jpeg", "png", "vtf", "vmt", "mp3", "wav", "ogg", "bak", "tmp", "jsonl", "log", "lua", "cfg", "ini", "bin", "md", "html", "bak.txt" }) do
      local p = "mentat/persist_probe/x." .. ext
      file.Write(p, "x")
      out.ext[ext] = file.Exists(p, "DATA")
      file.Delete(p)
    end
    out.dir_rule = {}
    file.Write("mentat/persist_probe_nodir/x.txt", "x")                       -- a directory nobody created
    out.dir_rule.write_into_missing_dir = file.Exists("mentat/persist_probe_nodir/x.txt", "DATA")
    file.Write("mentat/persist_probe/rn.txt", "x")
    out.rename_to_tmp = file.Rename("mentat/persist_probe/rn.txt", "mentat/persist_probe/rn.tmp") and file.Exists("mentat/persist_probe/rn.tmp", "DATA")
    file.Delete("mentat/persist_probe/rn.txt"); file.Delete("mentat/persist_probe/rn.tmp")
    local txt = util.TableToJSON({ players = { [SID] = { 1 } }, list = { { sid = SID } }, num = tonumber(SID) })
    out.json_text = txt
    local t1, t2 = util.JSONToTable(txt), util.JSONToTable(txt, true)
    local function ktypes(t) local o = {} for k in pairs(t and t.players or {}) do o[#o + 1] = type(k) .. ":" .. string.format("%.0f", tonumber(k) or 0) end return o end
    out.keys_default, out.keys_ignore_conversions = ktypes(t1), ktypes(t2)
    out.sid_value_roundtrip = t1 and t1.list and t1.list[1] and t1.list[1].sid == SID
    out.number_roundtrip = t1 and string.format("%.0f", t1.num or 0)
    check("rule_ext_txt_json_writable", out.ext.txt == true and out.ext.json == true, "txt=" .. tostring(out.ext.txt) .. " json=" .. tostring(out.ext.json))
    check("rule_ext_tmp_bak_jsonl_refused", out.ext.tmp == false and out.ext.bak == false and out.ext.jsonl == false, "tmp=" .. tostring(out.ext.tmp) .. " bak=" .. tostring(out.ext.bak) .. " jsonl=" .. tostring(out.ext.jsonl))
    check("rule_bak_txt_writable", out.ext["bak.txt"] == true, tostring(out.ext["bak.txt"]))
    check("rule_sid_string_value_survives", out.sid_value_roundtrip == true, tostring(out.sid_value_roundtrip))
    file.Delete("mentat/persist_probe_nodir/x.txt")

    -- ------------------------------------------------------------------ the modules, through their own code
    local real = { [SID] = fake }
    if bot then real[BOT] = bot end
    local oldsend = net.Send
    net.Send = function() end
    for sid, ply in pairs(real) do
      local ok = Wake.add(ply, "Rex" .. sid:sub(-3))
      check("wake_add_" .. sid, ok == true, tostring(ok))
    end
    net.Send = oldsend
    check("wake_file_written", file.Exists(Wake.FILE, "DATA"), head(Wake.FILE))

    -- (faces only keep ids that look like a human SteamID64, 7656119...: a bot's id is refused by design, so only the made-up one is stored)
    Faces.rec[SID] = { hidden = true, off = true, optin = true, t = os.time() }
    Faces.blocked[SID] = true
    check("faces_save", Faces.save() == true, tostring(Faces.FILE))
    check("faces_file_written", file.Exists(Faces.FILE, "DATA"), head(Faces.FILE))

    for _, sid in ipairs(sids) do Dr.prefs[sid] = { roast = "savage", chat = "quiet" } end
    check("prefs_save", Dr.SavePrefs() == true, tostring(Dr.PREFS_FILE))
    check("prefs_file_written", file.Exists(Dr.PREFS_FILE, "DATA"), head(Dr.PREFS_FILE))

    for _, sid in ipairs(sids) do
      J.delete_all(sid)
      check("journal_log_" .. sid, J.log(sid, { k = "add", name = "probecouch", n = 3 }, os.time()) == true, "")
      check("journal_file_" .. sid, file.Exists("mentat/journal/" .. sid .. ".txt", "DATA"), head("mentat/journal/" .. sid .. ".txt"))
    end
    return done(out)
  end

  -- ------------------------------------------------------------------ phase read: after the restart
  local expect_names = { "skippy" }
  for _, sid in ipairs(sids) do
    local n = Wake.names(sid)
    check("wake_survived_" .. sid, #n == 2 and n[2] == ("rex" .. sid:sub(-3)):lower(), table.concat(n, ","))
  end
  do
    local r = Faces.rec[SID]
    check("faces_survived_" .. SID, r ~= nil and r.hidden == true and r.off == true and r.optin == true and Faces.blocked[SID] == true, r and "rec ok" or "no rec")
  end
  out.prefs_loaded_by_server_start = Dr.prefs[SID] ~= nil
  Dr.LoadPrefs()
  for _, sid in ipairs(sids) do
    local p = Dr.prefs[sid]
    check("prefs_survived_" .. sid, p ~= nil and p.roast == "savage" and p.chat == "quiet", p and (p.roast .. "/" .. p.chat) or "none")
  end
  for _, sid in ipairs(sids) do
    local ev = J.read(sid, os.time())
    check("journal_survived_" .. sid, #ev == 1 and ev[1].name == "probecouch" and ev[1].n == 3, "events=" .. #ev)
  end

  -- legacy files: the old formats keyed by SteamID64 must load without a crash (the rounded keys are dropped with a log line)
  local lw = util.TableToJSON({ version = 1, players = { [SID] = { "skippy", "legacy" } } })
  local lf = util.TableToJSON({ version = 1, rec = { [SID] = { hidden = true } }, blocked = { [SID] = true } })
  local keep_w, keep_f = file.Read(Wake.FILE, "DATA"), file.Read(Faces.FILE, "DATA")
  file.Write(Wake.FILE, lw); file.Write(Faces.FILE, lf)
  local okw, errw = pcall(Wake.load)
  local okf, errf = pcall(Faces.load)
  check("legacy_wake_loads_without_crash", okw, tostring(errw))
  check("legacy_faces_loads_without_crash", okf, tostring(errf))
  out.legacy_wake_names = Wake.names(SID)
  out.legacy_faces_rec = Faces.rec[SID] ~= nil
  file.Write(Wake.FILE, keep_w or ""); file.Write(Faces.FILE, keep_f or "")
  pcall(Wake.load); pcall(Faces.load)
  if bot then pcall(function() bot:Kick("probe done") end) end
  done(out)
end
