-- wisp_shots.lua -- photograph the failure wisp in the real game client (stage/gmod/holodeck/README.md, "Failure wisp").
-- args: { scenario = "lapse" | "colours" | "unread" | "fetch" | "perf", bridge =, token =, timeout =, stall =, life = }
--   lapse   : "add a gargoyle" (the resolver says no): the wisp appears in front of the player; seen from 7 m (not reading), then from 2.5 m
--             looking at it (text unfurls), through the dissolve, and gone
--   colours : three wisps side by side: amber (not found), cyan ("I read 'boombx' as 'boombox'"), red (a jumbo jet I may not share)
--   unread  : a wisp nobody reads, holodeck_wisp_life shortened: idle, fading, gone
--   fetch   : a download that breaks (hydrant) and one that goes quiet until the timeout (cannon): the wisp after each collapse
--   perf    : frame cost with three wisps (one being read) against none
-- The camera is pinned each tick by moving the player (the server's reading logic uses the player's real eye position and angles).
-- Pictures are taken by Mentat.Shot (shot_wsp_<name>.jpg) and the result lists the wisp's state at each.
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local H = Holodeck
  if not (IsValid(ply) and H and H.Wisps) then return done(nil, "needs the game client and the holodeck") end
  local scenario = args.scenario or "lapse"
  local out = { scenario = scenario, frames = {}, notes = {} }
  ply:GodEnable()
  ply:StripWeapons()
  if args.bridge then
    RunConsoleCommand("holodeck_bridge", args.bridge)
    RunConsoleCommand("holodeck_bridge_token", args.token or "")
    RunConsoleCommand("holodeck_agent", "0")
    RunConsoleCommand("holodeck_persona", "0")
    RunConsoleCommand("holodeck_voice_out", "0")
  end
  if args.timeout then RunConsoleCommand("holodeck_found_timeout", tostring(args.timeout)) end
  if args.stall then RunConsoleCommand("holodeck_found_stall", tostring(args.stall)) end
  if args.life then RunConsoleCommand("holodeck_wisp_life", tostring(args.life)) end

  local co
  local function sleep(t)
    timer.Simple(t, function()
      local ok, e = coroutine.resume(co)
      if not ok then done(nil, "script: " .. tostring(e)) end
    end)
    coroutine.yield()
  end
  local function until_(cond, limit, step)
    local waited = 0
    while not cond() and waited < limit do sleep(step or 0.1); waited = waited + (step or 0.1) end
    return cond()
  end
  local function floor_at(x, y)
    return util.TraceLine({ start = Vector(x, y, 3000), endpos = Vector(x, y, -30000), mask = MASK_SOLID_BRUSHONLY }).HitPos.z
  end
  local function wisps() return ents.FindByClass("holodeck_wisp") end
  local t0 = SysTime()
  local pin
  hook.Add("Tick", "wsp_pin", function()
    if pin and IsValid(ply) then ply:SetPos(pin.pos); ply:SetVelocity(-ply:GetVelocity()); ply:SetEyeAngles(pin.ang) end
  end)
  local function shot(name, e)
    local row = { name = name, t = SysTime() - t0 }
    if IsValid(e) then
      row.state, row.read, row.dwell, row.why = e:WispState(), e:GetRead(), e:GetDwell(), e:WispWhy()
      row.class, row.text = e:WispClass(), e:GetReason()
      row.dist = pin and (pin.pos + Vector(0, 0, 64)):Distance(e:GetPos()) or -1
    else
      row.state = "gone"
    end
    out.frames[#out.frames + 1] = row
    Mentat.Shot("wsp_" .. name)
  end
  --- Stand `dist` units from the wisp (on the side it was made from), eyes level with it, looking at it. yaw_off turns the look aside.
  local function stand(e, dist, yaw_off, from)
    local p = e:GetPos()
    local dir = (from or Vector(-1, 0, 0))
    dir = Vector(dir.x, dir.y, 0); dir:Normalize()
    local eye = p + dir * dist
    local feet = Vector(eye.x, eye.y, p.z - 64)
    local ang = (p - eye):Angle()
    ang.y = ang.y + (yaw_off or 0)
    pin = { pos = feet, ang = ang }
    ply:SetPos(feet); ply:SetEyeAngles(ang)
  end
  local function ask(text, away)
    local n = #wisps()
    H.Handle(ply, text)
    local ok = until_(function() return #wisps() > n end, 40, 0.1)
    if ok and away then ply:SetEyeAngles(Angle(0, ply:EyeAngles().y + 150, 0)) end      -- (the wisp is made right in front of the asker: looking at it reads it)
    return ok
  end

  co = coroutine.create(function()
    sleep(1)
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(80, 0)
    local z = floor_at(0, 0)
    ply:SetPos(Vector(0, 0, z + 1)); ply:SetEyeAngles(Angle(0, 0, 0))
    sleep(1.5)

    if scenario == "lapse" then
      local got = ask("add a gargoyle")
      out.say = H.last_say
      local e = wisps()[1]
      if not (got and IsValid(e)) then out.notes[#out.notes + 1] = "no wisp appeared"; return done(out) end
      out.text = e:GetReason()
      stand(e, 300, 0)
      sleep(0.25); shot("l0_appearing", e)
      sleep(0.5); shot("l1_appearing2", e)
      until_(function() return e:WispState() == "idle" end, 5, 0.1)
      sleep(1.0); shot("l2_idle_far", e)
      stand(e, 140, 0)                                              -- 2.7 m, looking at it
      sleep(0.45); shot("l3_reading_start", e)
      sleep(0.5); shot("l4_reading_half", e)
      sleep(0.6); shot("l5_reading_full", e)
      local dwell = e:GetDwell()
      until_(function() return not IsValid(e) or e:WispState() == "dissolving" end, 15, 0.05)
      sleep(0.05); shot("l6_dissolve_a", e)
      sleep(0.2); shot("l7_dissolve_b", e)
      sleep(0.25); shot("l8_dissolve_c", e)
      sleep(0.25); shot("l9_dissolve_d", e)
      sleep(0.8); shot("l10_gone", e)
      out.wisp_removed = not IsValid(e)
      out.dwell = dwell
      return done(out)
    end

    if scenario == "colours" then
      local ws = {}
      for i, text in ipairs({ "add a gargoyle", "add a boombx", "add a jumbo jet" }) do
        ply:SetEyeAngles(Angle(0, (i - 2) * 60, 0)); ply:SetPos(Vector(0, 0, z + 1)); pin = nil
        sleep(0.2)
        if not ask(text, true) then out.notes[#out.notes + 1] = "no wisp for " .. text end
        out["say" .. i] = H.last_say
        ws[#ws + 1] = wisps()[#wisps()]
        sleep(0.3)
      end
      pin = { pos = Vector(0, 0, z + 1), ang = Angle(0, 0, 0) }
      ply:SetPos(pin.pos); ply:SetEyeAngles(pin.ang)
      ply:SetPos(Vector(-170, 0, z + 1)); pin.pos = Vector(-170, 0, z + 1)
      for _, e in ipairs(ws) do if IsValid(e) then out.notes[#out.notes + 1] = e:WispClass() .. ": " .. e:GetReason() end end
      until_(function() return ws[1] and IsValid(ws[1]) and ws[1]:WispState() == "idle" end, 6, 0.1)
      sleep(1.5); shot("c0_three_idle", ws[1])
      sleep(1.0); shot("c1_three_idle_b", ws[1])
      for i, e in ipairs(ws) do                                      -- each one close up, unread (looking a little to the side)
        if IsValid(e) then
          stand(e, 120, 40)
          sleep(0.7); shot("c" .. (i + 1) .. "_close_" .. e:WispClass(), e)
        end
      end
      return done(out)
    end

    if scenario == "unread" then
      ask("add a gargoyle", true)
      local e = wisps()[1]
      if not IsValid(e) then return done(out) end
      stand(e, 300, 0)                                               -- in view but 5.7 m away: too far to read
      sleep(1.5); shot("u0_idle", e)
      until_(function() return not IsValid(e) or e:WispState() == "dissolving" end, 40, 0.1)
      shot("u1_fade_start", e)
      sleep(1.0); shot("u2_fade_mid", e)
      sleep(1.2); shot("u3_fade_late", e)
      sleep(1.5); shot("u4_gone", e)
      out.removed = not IsValid(e)
      out.why = IsValid(e) and e:WispWhy() or "gone"
      return done(out)
    end

    if scenario == "fetch" then
      H.Handle(ply, "add a fire hydrant")
      until_(function() return #ents.FindByClass("holodeck_found") > 0 end, 30, 0.05)
      local hyd = ents.FindByClass("holodeck_found")[1]
      if not IsValid(hyd) then return done(out) end
      local p = hyd:GetPos()
      pin = { pos = p + Vector(-150, 0, 1), ang = Angle(0, 0, 0) }
      ply:SetPos(pin.pos); ply:SetEyeAngles(Angle(8, 0, 0))
      until_(function() return not IsValid(hyd) or hyd:FxStageName() == "failed" end, 40, 0.05)
      out.say_hydrant = H.last_say
      sleep(1.0); shot("f0_collapsing")
      until_(function() return #wisps() > 0 end, 5, 0.05)
      sleep(0.4); shot("f1_hydrant_wisp", wisps()[1])
      out.hydrant_wisp = wisps()[1] and wisps()[1]:GetReason()
      out.hydrant_class = wisps()[1] and wisps()[1]:WispClass()
      sleep(1.0); shot("f2_hydrant_wisp_b", wisps()[1])
      -- the quiet download, ended by the hard timeout
      for _, w in ipairs(wisps()) do w:Remove() end
      H.Wisps.list = {}
      H.Handle(ply, "add a cannon")
      until_(function() return #ents.FindByClass("holodeck_found") > 0 end, 30, 0.05)
      local can = ents.FindByClass("holodeck_found")[1]
      if not IsValid(can) then return done(out) end
      p = can:GetPos()
      pin = { pos = p + Vector(-170, 0, 1), ang = Angle(0, 0, 0) }
      ply:SetPos(pin.pos); ply:SetEyeAngles(Angle(8, 0, 0))
      until_(function() return not IsValid(can) or can:FxStageName() == "failed" end, 90, 0.1)
      out.say_cannon = H.last_say
      until_(function() return #wisps() > 0 end, 8, 0.05)
      sleep(0.5); shot("f3_cannon_wisp", wisps()[1])
      out.cannon_wisp = wisps()[1] and wisps()[1]:GetReason()
      out.cannon_class = wisps()[1] and wisps()[1]:WispClass()
      -- read it from close by: the whole sentence the transport failed with
      local e = wisps()[1]
      if IsValid(e) then
        stand(e, 130, 0, Vector(-1, 0, 0))
        sleep(2.0); shot("f4_cannon_reading", e)
      end
      return done(out)
    end

    if scenario == "perf" then
      local function mk(text, yaw)
        ply:SetPos(Vector(0, 0, z + 1)); ply:SetEyeAngles(Angle(0, yaw, 0)); pin = nil
        sleep(0.2)
        ask(text, true)
      end
      mk("add a gargoyle", -60); mk("add a boombx", 0); mk("add a jumbo jet", 60)
      local ws = wisps()
      if #ws < 3 then out.notes[#out.notes + 1] = "only " .. #ws .. " wisps"; return done(out) end
      pin = { pos = Vector(-250, 0, z + 1), ang = Angle(0, 0, 0) }
      ply:SetPos(pin.pos)
      until_(function() return ws[1]:WispState() == "idle" end, 6, 0.1)
      sleep(1.0)
      ply:ConCommand("fps_max 0"); ply:ConCommand("mat_vsync 0")
      sleep(1.5)
      local function perf(label, secs)
        diag(ply, 'local n, t = 0, 0 hook.Add("Think", "wspperf", function() n = n + 1 t = t + RealFrameTime() end) timer.Simple(' .. (secs - 0.2)
          .. ', function() hook.Remove("Think", "wspperf") local W = Mentat.WispFX or {} local s = string.format("%s frames=%d avg_ms=%.3f fps=%.1f wisp_ms=%.4f sprites=%d\\n", "' .. label
          .. '", n, 1000 * t / math.max(n, 1), n / math.max(t, 1e-9), W.cost_ms or -1, W.drawn or -1) print("[wsp-perf] " .. s) file.Append("mentat/wisp_perf.txt", s) end)')
        sleep(secs)
      end
      file.Delete("mentat/wisp_perf.txt")
      pin = { pos = Vector(-250, 0, z + 1), ang = Angle(0, 0, 0) }
      for _, round in ipairs({ "A", "B" }) do
        pin.ang = Angle(0, 0, 0)                                    -- three wisps in view, none read (they would dissolve)
        for _, e in ipairs(ws) do if IsValid(e) then e:SetPos(e:GetPos()) end end
        perf("three_wisps_" .. round, 5)
        pin.ang = Angle(0, 180, 0)                                  -- looking away: none in view
        sleep(0.5); perf("none_in_view_" .. round, 5)
      end
      for _, e in ipairs(wisps()) do e:Remove() end
      H.Wisps.list = {}
      sleep(0.8); perf("no_wisps", 5)
      ply:ConCommand("fps_max 300")
      sleep(0.5)
      return done(out)
    end
    done(nil, "unknown scenario " .. tostring(scenario))
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
