-- avatar_live.lua -- the live pass for the Skippy avatar on the DEDICATED server (stage/gmod/AVATAR.md, "Live pass"). No human is needed:
-- one bot plays the caller (the same Holodeck.Handle entry the voice path uses), another is a second listener. Everything is
-- measured with the real engine and returned as one table; the driver (holodeck/avatar_live.py) greps the console log for errors.
-- args: { }  (needs a server with maxplayers >= 4: GMOD_DS_MAXPLAYERS=4)
return function(args, done)
  local Av, H = Mentat and Mentat.Avatar, Holodeck
  if not (Av and H) then return done(nil, "avatar or holodeck not loaded") end
  local out = { steps = {}, events = {} }
  local function V3(v) return { math.floor(v.x * 10) / 10, math.floor(v.y * 10) / 10, math.floor(v.z * 10) / 10 } end

  hook.Add("SkippyAvatarArrive", "avatar_live", function(bot, from, to) out.events[#out.events + 1] = { "arrive", from and V3(from) or false, V3(to), CurTime() } end)
  hook.Add("SkippyAvatarLeave", "avatar_live", function(bot, from) out.events[#out.events + 1] = { "leave", V3(from), CurTime() } end)
  local spawns = 0
  hook.Add("PlayerSpawn", "avatar_live", function(p) if Av.bot and p == Av.bot then spawns = spawns + 1 end end)

  local co
  local function sleep(t)
    timer.Simple(t, function()
      local ok, e = coroutine.resume(co)
      if not ok then done(nil, "script: " .. tostring(e)) end
    end)
    coroutine.yield()
  end

  local function floor_at(x, y)
    local tr = util.TraceLine({ start = Vector(x, y, 3000), endpos = Vector(x, y, -30000), mask = MASK_SOLID_BRUSHONLY })
    return tr.HitPos.z
  end
  local function stand(p, x, y, yaw)
    p:SetPos(Vector(x, y, floor_at(x, y) + 1))
    p:SetEyeAngles(Angle(0, yaw or 0, 0))
    p:SetVelocity(-p:GetVelocity())
  end
  local function body(bot)
    local weps = 0
    for _ in ipairs(bot:GetWeapons()) do weps = weps + 1 end
    return { isbot = bot:IsBot(), alive = bot:Alive(), health = bot:Health(), movetype = bot:GetMoveType(), nodraw = bot:GetNoDraw(),
             solid = bot:IsSolid(), solidflags = bot:GetSolidFlags(), cgroup = bot:GetCollisionGroup(), god = bot:HasGodMode(), weapons = weps,
             onground = bot:OnGround(), notarget = bot:IsFlagSet(FL_NOTARGET) }
  end
  local function rel(bot, c)
    local d = bot:GetPos() - c:GetPos()
    local yaw = math.rad(-c:EyeAngles().y)
    return { dist = d:Length(), above_eye = bot:GetPos().z - c:GetShootPos().z,
             fwd = d.x * math.cos(yaw) - d.y * math.sin(yaw), left = d.x * math.sin(yaw) + d.y * math.cos(yaw), z = d.z }
  end

  co = coroutine.create(function()
    local R = out
    R.item1 = { status_before = { visible = Av.visible, bot_exists = IsValid(Av.bot), is_active = Av.IsActive() }, maxplayers = game.MaxPlayers() }
    sleep(1)
    local caller = player.CreateNextBot("Caller")
    local listener = player.CreateNextBot("Listener")
    if not (IsValid(caller) and IsValid(listener)) then return done(nil, "no helper bots") end
    sleep(2)
    stand(caller, 0, 0, 0)
    stand(listener, 3000, 3000, 0)
    sleep(1)
    -- 2. summon through the same entry the voice path uses
    RunConsoleCommand("mentat_avatar_debug", "1")
    local t0 = SysTime()
    local res = H.Handle(caller, "Skippy, come here")
    R.item2 = { handle_returned = tostring(res), summon_ms = (SysTime() - t0) * 1000, slots = #player.GetAll() }
    local bot = Av.bot
    R.item2.bot_valid = IsValid(bot)
    if not IsValid(bot) then return done(R) end
    R.item2.name = bot:Nick(); R.item2.bot = body(bot); R.item2.rel_instant = rel(bot, caller); R.item2.active = Av.IsActive()
    R.item2.facing_yaw = bot:EyeAngles().y; R.item2.spawns_so_far = spawns
    sleep(1.0)
    R.item2.rel_after_1s = rel(bot, caller)
    R.item2.bot_after_1s = body(bot)
    local mn, mx = { 1e9, 1e9, 1e9 }, { -1e9, -1e9, -1e9 }
    for i = 1, 20 do
      local p = bot:GetPos() - caller:GetPos()
      for k, c in ipairs({ p.x, p.y, p.z }) do mn[k] = math.min(mn[k], c); mx[k] = math.max(mx[k], c) end
      sleep(0.1)
    end
    R.item2.hover_range_min = mn; R.item2.hover_range_max = mx

    -- 3. follow: walk 600 HU in 60 steps (200 HU/s), run (800 HU/s), rise, teleport
    local lag = {}
    for i = 1, 60 do
      caller:SetPos(caller:GetPos() + Vector(10, 0, 0))
      sleep(0.05)
      lag[#lag + 1] = math.floor((bot:GetPos() - caller:GetPos()):Length())
    end
    R.item3 = { walk_lag = lag }
    local lag2 = {}
    for i = 1, 40 do
      caller:SetPos(caller:GetPos() + Vector(40, 0, 0))
      sleep(0.05)
      lag2[#lag2 + 1] = math.floor((bot:GetPos() - caller:GetPos()):Length())
    end
    R.item3.run_lag = lag2
    for i = 1, 10 do caller:SetPos(caller:GetPos() + Vector(0, 0, 8)); sleep(0.05) end
    R.item3.after_rise_dz = { math.floor(bot:GetPos().z - caller:GetPos().z) }
    sleep(0.5)
    R.item3.after_rise_dz[2] = math.floor(bot:GetPos().z - caller:GetPos().z)
    local n_before = #out.events
    caller:SetPos(Vector(-5000, 4000, floor_at(-5000, 4000) + 1))
    sleep(0.1)
    R.item3.tp_dist_100ms = (bot:GetPos() - caller:GetPos()):Length()
    R.item3.tp_events = {}
    for i = n_before + 1, #out.events do R.item3.tp_events[#R.item3.tp_events + 1] = out.events[i] end
    sleep(0.5)
    R.item3.tp_dist_600ms = (bot:GetPos() - caller:GetPos()):Length()
    -- a wall: the world's edge to the east
    local from = Vector(0, 0, floor_at(0, 0) + 60)
    local tr = util.TraceLine({ start = from, endpos = from + Vector(40000, 0, 0), mask = MASK_SOLID_BRUSHONLY })
    R.item3.wall_trace = { hit = tr.Hit, x = tr.HitPos.x }
    if tr.Hit then
      local wx = tr.HitPos.x
      stand(caller, wx - 30, 0, 90)                         -- facing north: his right (east) is the wall
      Av.Summon(caller)
      sleep(0.3)
      local b = bot:GetPos()
      R.item3.wall_right = { bot_x = b.x, wall_x = wx, inside = b.x >= wx, rel = rel(bot, caller), clear_to_wall = wx - b.x }
      R.item3.wall_right.los_blocked = util.TraceLine({ start = caller:GetShootPos(), endpos = b, mask = MASK_SOLID_BRUSHONLY }).Hit
      stand(caller, wx - 30, 0, 180)                        -- back to the wall (facing west): he follows behind the shoulder, to the east
      Av.Summon(caller)
      sleep(0.8)
      b = bot:GetPos()
      R.item3.wall_back = { bot_x = b.x, wall_x = wx, inside = b.x >= wx, rel = rel(bot, caller), clear_to_wall = wx - b.x }
    end
    stand(caller, 0, 0, 0)
    Av.Summon(caller)
    sleep(0.5)

    -- 4. idle watch (60 s), then kill, damage, the void
    local before = { spawns = spawns, health = bot:Health(), pos = V3(bot:GetPos()) }
    local trace, weps_max, far_max = {}, 0, 0
    for i = 1, 60 do
      sleep(1)
      local bd = body(bot)
      weps_max = math.max(weps_max, bd.weapons)
      far_max = math.max(far_max, (bot:GetPos() - caller:GetPos()):Length())
      if i % 10 == 0 then trace[#trace + 1] = { i, V3(bot:GetPos()), bd.health, bd.weapons, bd.alive, bd.movetype } end
    end
    R.item4 = { before = before, idle_trace = trace, max_weapons = weps_max, spawns_during = spawns - before.spawns, max_dist_from_caller = far_max,
                still_visible = Av.IsActive(), after = body(bot) }
    local sp0 = spawns
    bot:TakeDamage(100000, game.GetWorld(), game.GetWorld())
    sleep(0.3)
    R.item4.after_takedamage = body(bot)
    bot:Kill()
    sleep(0.5)
    local k1 = body(bot); k1.dist = (bot:GetPos() - caller:GetPos()):Length()
    sleep(3)
    local k2 = body(bot); k2.dist = (bot:GetPos() - caller:GetPos()):Length()
    R.item4.after_kill_0_5s = k1; R.item4.after_kill_3_5s = k2; R.item4.respawns_after_kill = spawns - sp0; R.item4.visible_after_kill = Av.IsActive()
    bot:TakeDamage(500, game.GetWorld(), game.GetWorld())
    sleep(0.2)
    R.item4.after_hurt = body(bot)
    bot:SetPos(Vector(0, 0, -100000))
    sleep(0.05)
    R.item4.void_dist_50ms = (bot:GetPos() - caller:GetPos()):Length()
    sleep(0.5)
    R.item4.after_void = body(bot); R.item4.after_void.dist = (bot:GetPos() - caller:GetPos()):Length()
    R.item4.respawns_total = spawns - sp0
    caller:Kill()
    sleep(0.5)
    R.item4.caller_dead_pos_kept = V3(bot:GetPos())
    caller:Spawn()
    sleep(0.5)
    stand(caller, 0, 0, 0)
    sleep(0.5)

    -- 6. voice: per-listener hearing, sweep, whisper
    R.item6 = { rows = {} }
    local function hear(label)
      local rows = {}
      for _, h in ipairs(Av.WhoHears()) do rows[#rows + 1] = { h.nick, math.floor(h.dist), h.range, h.blocked, math.floor(h.gain * 1000) / 1000, h.heard } end
      R.item6.rows[#R.item6.rows + 1] = { label, rows }
    end
    RunConsoleCommand("mentat_voice_mode", "proximity")
    RunConsoleCommand("mentat_voice_range", "1100")
    sleep(0.5)
    hear("owner near, listener far (3000,3000)")
    for _, d in ipairs({ 100, 400, 800, 1050, 1500 }) do
      local bp = bot:GetPos()
      stand(listener, bp.x + d, bp.y, 180)
      sleep(0.15)
      hear("listener at +" .. d .. " east of him")
    end
    caller:AddFlags(FL_DUCKING)
    sleep(1.2)
    R.item6.crouch_flag = caller:Crouching()
    hear("owner crouching (whisper)")
    caller:RemoveFlags(FL_DUCKING)
    sleep(1.2)
    R.item6.say_ok = pcall(H.Say, caller, "this is a test line of speech", nil, { speak = true })
    R.item6.speaking_until_set = (Av.speaking_until or 0) > CurTime()
    R.item6.last_heard = {}
    for _, h in ipairs(Av.last_heard or {}) do R.item6.last_heard[#R.item6.last_heard + 1] = { h.nick, math.floor(h.dist), h.range, h.blocked, h.gain } end
    sleep(0.5)

    -- 9. net.Send to the bot: spoken lines and tells; the driver reads the console log
    for i = 1, 3 do H.Say(caller, "net send probe " .. i, nil, { speak = false }); sleep(0.2) end
    H.Tell(caller, "tell probe")
    local okn, errn = pcall(function() net.Start("holodeck_say"); net.WriteString("direct to the bot"); net.WriteString(""); net.WriteUInt(0, 32); net.Send(bot) end)
    R.item9 = { sent = 4, direct_net_send_to_bot = { okn, tostring(errn) }, bot_is_bot = bot:IsBot(),
                audience_has_bot = (function() for _, p in ipairs(player.GetAll()) do if p == bot then return true end end return false end)() }

    -- 8. kick mode, then the full server
    RunConsoleCommand("mentat_avatar_kick", "1")
    sleep(0.3)
    local n0 = #player.GetAll()
    Av.Dismiss()
    sleep(1)
    R.item8 = { players_before = n0, players_after_kick = #player.GetAll(), bot_exists_after = IsValid(Av.bot), active = Av.IsActive() }
    H.Handle(caller, "skippy come here")
    sleep(1.2)
    R.item8.rejoined = IsValid(Av.bot); R.item8.players_after_recall = #player.GetAll()
    Av.Dismiss(); sleep(1)
    local fillers = {}
    while #player.GetAll() < game.MaxPlayers() do
      local f = player.CreateNextBot("Filler" .. #fillers)
      if not IsValid(f) then break end
      fillers[#fillers + 1] = f
      sleep(0.3)
    end
    R.item8.full_players = #player.GetAll()
    local ok, why = Av.Summon(caller)
    R.item8.full_summon = { ok, why }
    H.last_tell = nil
    H.Handle(caller, "skippy come here"); sleep(1.1)
    R.item8.full_tell = H.last_tell
    for _, f in ipairs(fillers) do f:Kick("done") end
    RunConsoleCommand("mentat_avatar_kick", "0")
    sleep(1)

    -- 10. other words, out and in
    local ents0 = #ents.GetAll()
    local r1 = H.Handle(caller, "Skippy, add a couch")
    local waited = 0
    while H.Busy() and waited < 20 do sleep(0.5); waited = waited + 0.5 end
    sleep(1)
    R.item10 = { couch_say_out = r1 and r1.say, ents_added_out = #ents.GetAll() - ents0, avatar_active_out = Av.IsActive() }
    H.Handle(caller, "skippy come here"); sleep(1.2)
    R.item10.active_after_call = Av.IsActive()
    local e1 = #ents.GetAll()
    local r2 = H.Handle(caller, "Skippy, add a lamp")
    waited = 0
    while H.Busy() and waited < 20 do sleep(0.5); waited = waited + 0.5 end
    sleep(1)
    R.item10.lamp_say_in = r2 and r2.say; R.item10.ents_added_in = #ents.GetAll() - e1
    H.Handle(caller, "Skippy, go away"); sleep(0.5)
    R.item10.active_after_away = Av.IsActive(); R.item10.bot_exists_after_away = IsValid(Av.bot)
    R.item10.nodraw_after_away = IsValid(Av.bot) and Av.bot:GetNoDraw() or "n/a"
    R.spawns_total = spawns
    R.events_total = #out.events
    done(R)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
