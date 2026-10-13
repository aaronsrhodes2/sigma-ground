-- director_shots.lua -- Skippy alive, in the real game: drive scripted situations through the behaviour director and photograph the body language
-- (stage/gmod/AVATAR.md "Skippy alive", live pass). args: { bridge, token, limit (new lines per day to allow; default 3) }
-- Needs the game client (a listen server: a player with a camera) and a running Skippy bridge with /quip. The owner is the camera; he is parked in front of it
-- (the follow hook is paused, as in avatar_shots.lua), so the pictures show the animation, not the follow. Writes shot_dir_*.jpg and a result with what the
-- director decided and why, the budget, the clips, and the client's measurements of the animation (mentat_avatar_dir_sample).
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local H, Av, Dr = Holodeck, Mentat and Mentat.Avatar, Mentat and Mentat.Director
  if not (IsValid(ply) and Av and H and Dr) then return done(nil, "needs the game client, the holodeck, the avatar and the director") end
  local out = { shots = {}, lines = {}, samples = {}, sample_tags = {}, notes = {}, perf = {} }
  RunConsoleCommand("holodeck_bridge", args.bridge)
  RunConsoleCommand("holodeck_bridge_token", args.token or "")
  RunConsoleCommand("holodeck_persona", "0")
  RunConsoleCommand("holodeck_agent", "0")
  RunConsoleCommand("holodeck_voice_out", "1")
  RunConsoleCommand("holodeck_bridge_per_min", "600")
  RunConsoleCommand("holodeck_voice_chars_per_min", "100000")
  RunConsoleCommand("mentat_skippy_director", "1")
  RunConsoleCommand("mentat_skippy_chat", "chatty")
  RunConsoleCommand("mentat_skippy_journal", "1")
  RunConsoleCommand("mentat_skippy_new_lines_per_day", tostring(args.limit or 3))
  RunConsoleCommand("mentat_avatar_debug", "0")
  ply:ConCommand("skippy_debug 1")
  ply:ConCommand("skippy_volume 0.15")                -- (quiet: Aaron may be at the desk; the clip is real, the level is what we need)
  ply:GodEnable()
  Dr.Rand = function() return 0.0 end                 -- the dice always say yes: the script chooses what happens
  file.CreateDir("mentat")
  file.Delete("mentat/skippy_played.txt")

  local co
  local function sleep(t)
    timer.Simple(t, function()
      local ok, e = coroutine.resume(co)
      if not ok then done(nil, "script: " .. tostring(e)) end
    end)
    coroutine.yield()
  end
  local function floor_at(x, y)
    return util.TraceLine({ start = Vector(x, y, 3000), endpos = Vector(x, y, -30000), mask = MASK_SOLID_BRUSHONLY }).HitPos.z
  end
  local function snapshot(label)
    local l = Dr.last_line
    out.lines[#out.lines + 1] = { label = label, line = l and { text = l.text, mood = l.mood, cue = l.cue, source = l.source, new = l.new, cached = l.cached, dur = l.dur, ev = l.ev } or false,
                                  budget = Dr.budget and { used = Dr.budget.used, limit = Dr.budget.limit, left = Dr.budget.left } or false, counts = table.Copy(Dr.counts) }
  end
  local function mark() return Dr.last_line and Dr.last_line.t or -1 end
  --- Wait until a line newer than `since` (a mark() taken BEFORE the trigger) has been presented, or `limit` seconds pass. Returns true when one came.
  local function await_line(limit, since)
    local before = since or mark()
    local waited = 0
    while waited < (limit or 25) do
      sleep(0.25)
      waited = waited + 0.25
      if Dr.last_line and Dr.last_line.t ~= before then return true end
    end
    return false
  end
  local function wait_quiet(limit)
    local waited = 0
    while CurTime() < Dr.st.speaking_until + 0.6 and waited < (limit or 20) do sleep(0.3); waited = waited + 0.3 end
  end
  local P
  --- The scripted situations come one after another faster than a real afternoon: forget the pacing so each one is judged on its own.
  local function pace()
    Dr.st.last_spoke, Dr.st.spoke = -1e9, {}
    for k in pairs(Dr.st.cool) do Dr.st.cool[k] = 0 end
  end
  local function camera(dist, yaw, elev, aim_z)
    local ya, el = math.rad(yaw or 0), math.rad(elev or 8)
    local dir = Vector(math.cos(ya) * math.cos(el), math.sin(ya) * math.cos(el), math.sin(el))
    local aim = P + Vector(0, 0, aim_z or 6)
    local cam = P + dir * (dist or 80)
    ply:SetPos(cam - Vector(0, 0, 64))
    ply:SetVelocity(-ply:GetVelocity())
    ply:SetEyeAngles((aim - cam):Angle())
  end
  local function shots(prefix, n, every)
    for i = 1, n do
      Mentat.Shot("dir_" .. prefix .. "_" .. i)
      out.shots[#out.shots + 1] = "dir_" .. prefix .. "_" .. i
      sleep(every)
    end
  end
  local function sample(tag, secs)
    ply:ConCommand("mentat_avatar_dir_sample " .. secs .. " " .. tag)
    out.sample_tags[#out.sample_tags + 1] = tag
  end

  co = coroutine.create(function()
    sleep(1)
    ply:SetPos(Vector(0, 0, floor_at(0, 0) + 1))
    ply:SetEyeAngles(Angle(0, 0, 0))
    sleep(1.5)
    H.Handle(ply, "skippy, come here")                        -- the same entry a spoken sentence takes (the arrival line is asked for here)
    sleep(2.5)
    local bot = Av.bot
    out.summoned = IsValid(bot) and Av.IsActive() or false
    if not out.summoned then return done(out) end
    for _ = 1, 16 do diag(ply, "Mentat.Avatar.BodyC.poll()") sleep(0.5) end        -- the real body arrives in chunks
    wait_quiet(20)
    snapshot("summon (arrival line)")
    -- park him in front of the camera, as avatar_shots.lua does
    hook.Remove("Tick", "Mentat.AvatarFollow")
    P = Vector(0, 0, floor_at(0, 0) + 100)
    Av.st.yaw = 0
    bot:SetPos(P); bot:SetEyeAngles(Angle(0, 0, 0)); bot:SetAngles(Angle(0, 0, 0))
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(70, 0)
    camera(70, 0, 6, 2)
    sleep(1.5)

    -- A. arrival: the hat tip at once, then the fresh line
    out.notes[#out.notes + 1] = "A arrival: cue at once, line after the bridge answers"
    sample("arrival", 7)
    pace()
    local m = mark()
    Dr.Arrival(ply, nil)
    shots("arrive", 5, 0.32)
    await_line(20, m)
    snapshot("A arrival")
    wait_quiet(20)

    -- B. something he asked for lands: a flourish (a beam arrived)
    pace()
    sample("flourish", 8)
    m = mark()
    hook.Run("HolodeckBeam", "ready", { card = "found.poly-haven.boombox", asker = ply })
    shots("flourish", 6, 0.45)
    await_line(10, m)
    snapshot("B beam ready")
    wait_quiet(20)

    -- C. a bad-opera aria, conducted (the opera whim, through the same decision path)
    pace()
    sample("opera", 10)
    m = mark()
    Dr.Event("opera", { ply = ply })
    sleep(2.5)                                                  -- (the bridge makes the clip first; the pictures are taken while he sings)
    shots("opera", 7, 0.6)
    local got = await_line(25, m)
    out.notes[#out.notes + 1] = "C opera line arrived: " .. tostring(got)
    snapshot("C opera")
    wait_quiet(20)

    -- D. he looks at and points to the thing just placed
    local prop = ents.Create("prop_physics")
    prop:SetModel("models/hunter/misc/sphere025x025.mdl")
    prop:SetPos(P + Vector(22, 18, 34))
    prop:SetColor(Color(255, 40, 40))
    prop:Spawn()
    local po = prop:GetPhysicsObject()
    if IsValid(po) then po:EnableMotion(false) end
    Dr.LookAt(prop:GetPos(), "what you just placed")
    Dr.placed = { pos = prop:GetPos(), t = CurTime() }
    Av.Point(prop:GetPos(), 6)
    Dr.Cue("emphasis", "helpful")
    sample("point", 5)
    pace()
    m = mark()
    Dr.Event("build", { ply = ply })
    sleep(2.0)
    shots("point", 4, 0.5)
    await_line(10, m)
    snapshot("D look/point (build)")
    wait_quiet(20)
    Av.Point(nil)
    if IsValid(prop) then prop:Remove() end

    -- E. a roast at the owner's level (the host of a listen server is savage by default)
    out.roast_level = Dr.RoastLevel(ply)
    pace()
    sample("roast", 6)
    m = mark()
    H.Handle(ply, "roast me")
    sleep(2.0)
    shots("roast", 3, 0.7)
    await_line(15, m)
    snapshot("E roast (" .. tostring(out.roast_level) .. ")")
    wait_quiet(20)

    -- F. a human starts talking over him: the line is cut and the clients are told
    pace()
    m = mark()
    Dr.Event("opera", { ply = ply })
    await_line(25, m)
    sleep(1.2)
    local guest = player.CreateNextBot("Guest")
    local before = Dr.counts.interrupted
    if IsValid(guest) then
      Dr.talk[guest] = { on = true, t = CurTime(), last = CurTime() }
      sleep(1.2)
      out.interrupted = Dr.counts.interrupted - before
      Dr.talk[guest] = nil
      guest:Kick("test over")
    end
    out.talking_after_interrupt = Dr.st.speaking_until > CurTime()
    snapshot("F interrupted")
    sleep(1.5)

    -- I. the notebook and a callback: he builds things, undoes one, and is asked what he remembers (the notes go to /quip; the template model answers)
    H.Handle(ply, "add three couches"); sleep(2.5)
    H.Handle(ply, "add a lamp"); sleep(2.5)
    H.Handle(ply, "undo"); sleep(2.5)
    Dr.Record(ply, { k = "beam_fail", name = "flux capacitor", why = "failure" })           -- (the journal entry the resolve hook would make, without a remark of its own)
    Dr.Record(ply, { k = "beam_fail", name = "flux capacitor", why = "failure" })
    sleep(5)
    out.notes_sent = Dr.NotesFor(ply)
    pace()
    m = mark()
    H.Handle(ply, "what do you remember")
    await_line(25, m)
    snapshot("I recall")
    wait_quiet(20)

    -- G. the budget: ask for more lines than the day allows (new lines are only allowed up to the limit)
    for i, sit in ipairs({ "musing", "tip", "boast", "musing", "physics" }) do
      pace()
      m = mark()
      Dr.Event(sit, { ply = ply })
      await_line(14, m)
      snapshot("G " .. sit .. " #" .. i)
      wait_quiet(20)
    end
    out.played = file.Exists("mentat/skippy_played.txt", "DATA") and file.Read("mentat/skippy_played.txt", "DATA") or nil
    out.audio = H.last_audio
    out.voice = H.last_voice

    -- H. frame cost: the director's body language running (a conducted aria, mood opera) vs the director off vs no avatar at all (uncapped frame rate)
    camera(75, 0, 4, 2)
    ply:ConCommand("fps_max 0"); ply:ConCommand("mat_vsync 0")
    sleep(1.5)
    local function perf(label, secs)
      diag(ply, 'local n, t = 0, 0 hook.Add("Think", "dirperf", function() n = n + 1 t = t + RealFrameTime() end) timer.Simple(' .. (secs - 0.2)
        .. ', function() hook.Remove("Think", "dirperf") file.Write("mentat/dirperf_' .. label .. '.json", util.TableToJSON({ frames = n, avg_ms = 1000 * t / math.max(n, 1), fps = n / math.max(t, 1e-9), draw_ms = Mentat.Avatar.cost_ms or -1 })) end)')
      sleep(secs)
      local raw = file.Read("mentat/dirperf_" .. label .. ".json", "DATA")
      out.perf[label] = raw and util.JSONToTable(raw) or false
    end
    -- keep an aria "playing" for the whole measurement: the cue is re-sent every 4 s and the talking flag held
    local keep = true
    timer.Create("dir_perf_cues", 3.5, 0, function()
      if not keep then return end
      Dr.Cue("conduct", "opera")
      Dr.st.speaking_until = CurTime() + 4
    end)
    Dr.Cue("conduct", "opera"); Dr.st.speaking_until = CurTime() + 4
    ply:ConCommand("mentat_avatar_draw 1")
    sleep(0.6); perf("director_on", 5)
    RunConsoleCommand("mentat_skippy_director", "0")
    sleep(0.6); perf("director_off", 5)
    ply:ConCommand("mentat_avatar_draw 0")
    sleep(0.6); perf("no_avatar", 5)
    ply:ConCommand("mentat_avatar_draw 1")
    RunConsoleCommand("mentat_skippy_director", "1")
    keep = false
    timer.Remove("dir_perf_cues")
    ply:ConCommand("fps_max 300")
    sleep(0.5)

    for _, tag in ipairs(out.sample_tags) do
      local raw = file.Read("mentat/dirsample_" .. tag .. ".json", "DATA")
      out.samples[tag] = raw and util.JSONToTable(raw) or false
    end
    out.ring = {}
    for i = math.max(1, #Dr.ring - 60), #Dr.ring do local r = Dr.ring[i]; out.ring[#out.ring + 1] = { ev = r.ev, say = r.say, why = r.why } end
    out.counts, out.budget, out.last_error = Dr.counts, Dr.budget, Dr.last_error
    out.journal_notes = Dr.NotesFor(ply).notes
    out.journal_file = file.Read("mentat/journal/" .. ply:SteamID64() .. ".txt", "DATA")
    out.journal_file = out.journal_file and #out.journal_file or 0
    ply:SetMoveType(MOVETYPE_WALK)
    ply:SetFOV(0, 0)
    ply:ConCommand("skippy_debug 0")
    ply:ConCommand("skippy_volume 1")
    done(out)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
