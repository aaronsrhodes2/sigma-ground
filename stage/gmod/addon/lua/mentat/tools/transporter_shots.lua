-- transporter_shots.lua -- photograph the transporter effect in the real game client (stage/gmod/holodeck/README.md, "The transporter effect").
-- args: { scenario = "success" | "failure" | "avatar" | "perf", bridge =, token =, chunk =, pump =, timeout = }
--   success : "add a boombox" with the bridge delivering it slowly; a picture each time the REAL progress passes 0 / 25 / 50 % (fetching), when
--             packing starts, partway through sending, then three frames of the materialise and one of the finished object
--   failure : "add a cannon and a fire hydrant": the hydrant's download breaks half way (immediate failure), the cannon's stops delivering
--             (the beam destabilises, then the hard timeout fails it); pictures of each stage of that
--   avatar  : Skippy leaves (beam out), is called back (beam in; his body waits), pictures during both
--   perf    : frame cost with four pending beams in view, the effect on / off (the plain boxes) / nothing
-- The pictures are taken by the same Mentat.Shot every other tool uses; they are named tp_<name> and the result lists what the server
-- reported (stage, progress, time) at each one, so a picture can be checked against the number it should show.
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local H, Av = Holodeck, Mentat and Mentat.Avatar
  if not (IsValid(ply) and H) then return done(nil, "needs the game client and the holodeck") end
  local scenario = args.scenario or "success"
  local out = { scenario = scenario, frames = {}, notes = {} }
  ply:GodEnable()
  ply:StripWeapons()                                            -- (no gravity gun in the pictures)
  if args.bridge then
    RunConsoleCommand("holodeck_bridge", args.bridge)
    RunConsoleCommand("holodeck_bridge_token", args.token or "")
    RunConsoleCommand("holodeck_agent", "0")
    RunConsoleCommand("holodeck_persona", "0")
    RunConsoleCommand("holodeck_voice_out", "0")
  end
  if args.timeout then RunConsoleCommand("holodeck_found_timeout", tostring(args.timeout)) end
  if args.stall then RunConsoleCommand("holodeck_found_stall", tostring(args.stall)) end
  if args.chunk and H.Mesh then H.Mesh.CHUNK = args.chunk end       -- live-test knobs: a slow send, so progress inside "sending" can be photographed
  if args.pump and H.Mesh then H.Mesh.PUMP = args.pump end

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
  local t0 = SysTime()
  local function found() return ents.FindByClass("holodeck_found") end
  local function state(e)
    if not IsValid(e) then return "gone", 0 end
    return e:FxStageName() or "none", e:GetFxProgress()
  end
  local pin
  local function shot(name, e)
    if pin then ply:SetPos(pin.pos); ply:SetEyeAngles(pin.ang) end         -- (the picture is taken from where the camera was put, whatever moved it)
    local st, pr = state(e)
    out.frames[#out.frames + 1] = { name = name, stage = st, progress = pr, t = SysTime() - t0, stalled = IsValid(e) and (CurTime() - e:GetFxBeat()) or -1,
      cam = { ply:GetPos().x, ply:GetPos().y, ply:GetPos().z, ply:EyeAngles().p, ply:EyeAngles().y, ply:GetMoveType() } }
    Mentat.Shot("tp_" .. name)
  end
  --- Put the camera (noclip, the player) looking at a point from `dist` units, yaw/elev in degrees.
  local function look(c, dist, yaw, elev)
    local ya, el = math.rad(yaw), math.rad(elev)
    local cam = c + Vector(math.cos(ya) * math.cos(el), math.sin(ya) * math.cos(el), math.sin(el)) * dist
    ply:SetPos(cam - Vector(0, 0, 64))
    ply:SetVelocity(-ply:GetVelocity())
    ply:SetEyeAngles((c - cam):Angle())
    pin = { pos = cam - Vector(0, 0, 64), ang = (c - cam):Angle() }
  end
  local function frame_all(yaw, elev, margin)
    local list = found()
    if #list == 0 then return nil end
    local cx, cy, cz, r = 0, 0, 0, 0
    for _, e in ipairs(list) do local p = e:GetPos(); cx, cy, cz = cx + p.x, cy + p.y, cz + p.z end
    cx, cy, cz = cx / #list, cy / #list, cz / #list
    local top = 0
    for _, e in ipairs(list) do
      local p, mx = e:GetPos(), e:GetBoxMaxs()
      r = math.max(r, math.sqrt((p.x - cx) ^ 2 + (p.y - cy) ^ 2) + math.max(mx.x, mx.y))
      top = math.max(top, mx.z)
    end
    local c = Vector(cx, cy, cz + top * 0.5)
    local dist = (r + top * 0.6 + 20) / math.tan(math.rad(33)) * (margin or 1)
    look(c, dist, yaw, elev)
    return c, dist
  end
  local function until_(cond, limit, step)
    local waited = 0
    while not cond() and waited < limit do sleep(step or 0.1); waited = waited + (step or 0.1) end
    return cond()
  end

  co = coroutine.create(function()
    sleep(1)
    ply:SetPos(Vector(0, 0, floor_at(0, 0) + 1))
    ply:SetEyeAngles(Angle(0, 0, 0))
    sleep(1.5)

    if scenario == "avatar" then
      Av.Summon(ply)
      sleep(4)                                                    -- the bot is made, comes, the body files arrive
      for _ = 1, 10 do diag(ply, [[Mentat.Avatar.BodyC.poll()]]); sleep(0.5) end
      diag(ply, [[print("[tp-body] " .. tostring(Mentat.Avatar.BodyC.state) .. " " .. tostring(Mentat.Avatar.BodyC.why))]])
      local bot = Av.bot
      out.summoned = IsValid(bot) and Av.IsActive() or false
      if not out.summoned then return done(out) end
      ply:SetMoveType(MOVETYPE_NOCLIP)
      ply:SetFOV(70, 0)
      hook.Remove("Tick", "Mentat.AvatarFollow")                  -- hold still while the pictures are taken: only the beams are under test
      local P = bot:GetPos()
      local aim = P + Vector(0, 0, 4)
      look(aim, 90, 25, 6)
      sleep(0.8)
      shot("av_before_leave")
      Av.Dismiss()                                                -- SkippyAvatarLeave: the beam-out at the old place
      for _, dt in ipairs({ 0.12, 0.25, 0.25, 0.25 }) do sleep(dt); shot("av_leave_" .. string.format("%02d", math.floor((SysTime() - t0) * 10) % 100)) end
      sleep(1.0)
      hook.Add("Tick", "Mentat.AvatarFollow", function() local ok, e = pcall(Av.Tick); if not ok then print("[tp] tick error: " .. tostring(e)) end end)
      Av.Summon(ply)                                              -- SkippyAvatarArrive: the beam-in; the body waits for it
      bot = Av.bot
      hook.Remove("Tick", "Mentat.AvatarFollow")
      local P2 = bot:GetPos()
      look(P2 + Vector(0, 0, 4), 90, 25, 6)
      for _, dt in ipairs({ 0.1, 0.25, 0.25, 0.25, 0.35 }) do sleep(dt); shot("av_arrive_" .. string.format("%02d", math.floor((SysTime() - t0) * 10) % 100)) end
      sleep(0.8)
      shot("av_arrived")
      sleep(0.4)
      ply:SetMoveType(MOVETYPE_WALK); ply:SetFOV(0, 0)
      hook.Add("Tick", "Mentat.AvatarFollow", function() local ok, e = pcall(Av.Tick); if not ok then print("[tp] tick error: " .. tostring(e)) end end)
      Av.Dismiss()
      sleep(0.5)
      return done(out)
    end

    local text = scenario == "failure" and "add a fire hydrant" or (scenario == "perf" and "add four boomboxes" or "add a boombox")
    local res = H.Handle(ply, text)
    out.say = res and res.say or ""
    until_(function() return #found() > 0 end, 30, 0.05)
    sleep(0.3)
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(70, 0)
    out.found = #found()
    local list = found()
    if #list == 0 then return done(out) end

    if scenario == "success" then
      local e = list[1]
      frame_all(35, 18, 1.0)
      sleep(0.5)
      shot("s0_start", e)                                          -- progress 0.08: the request has gone out
      for _, want in ipairs({ { "s1_p25", 0.25 }, { "s2_p50", 0.47 }, { "s3_packing", 0.58 }, { "s4_sending", 0.80 } }) do
        local ok = until_(function() return not IsValid(e) or e:GetFxProgress() >= want[2] end, 60, 0.05)
        if not IsValid(e) then out.notes[#out.notes + 1] = "the object went away before " .. want[1]; return done(out) end
        sleep(0.2)                                                 -- the picture lags the number by the easing: let it catch up
        shot(want[1], e)
        out.notes[#out.notes + 1] = want[1] .. (ok and " reached" or " NOT reached")
      end
      until_(function() return not IsValid(e) or e:FxStageName() == "decoding" or e:FxStageName() == "done" end, 30, 0.02)
      for _, dt in ipairs({ 0.12, 0.2, 0.2 }) do sleep(dt); shot("s5_arrive_" .. string.format("%02d", math.floor((SysTime() - t0) * 10) % 100), e) end
      sleep(1.6)
      shot("s6_done", e)
      out.mesh = H.Mesh and select(1, H.Mesh.status()) or -1
      return done(out)
    end

    if scenario == "failure" then
      -- 1. a download that BREAKS half way: the beam collapses at once
      local hydrant = list[1]
      frame_all(30, 18, 1.1)
      sleep(2.0)
      shot("f0_hydrant_pending", hydrant)
      until_(function() return not IsValid(hydrant) or hydrant:FxStageName() == "failed" end, 40, 0.02)
      for _, dt in ipairs({ 0.1, 0.3, 0.4, 0.4 }) do sleep(dt); shot("f1_hydrant_fail_" .. string.format("%02d", math.floor((SysTime() - t0) * 10) % 100), hydrant) end
      sleep(1.5)
      out.hydrant_removed = not IsValid(hydrant)
      out.say_hydrant = H.last_say
      -- 2. a download that goes QUIET: the beam destabilises, then the hard timeout fails it
      H.Handle(ply, "add a cannon")
      until_(function() return #found() > 0 end, 40, 0.05)
      sleep(0.3)
      local cannon = found()[1]
      if not IsValid(cannon) then return done(out) end
      frame_all(30, 18, 1.1)
      sleep(1.5)
      shot("f2_cannon_pending", cannon)
      local stall_s = tonumber(args.stall) or 6
      until_(function() return not IsValid(cannon) or CurTime() - cannon:GetFxBeat() >= stall_s * 0.55 end, 40, 0.1)
      shot("f3_cannon_quiet", cannon)
      until_(function() return not IsValid(cannon) or CurTime() - cannon:GetFxBeat() >= stall_s * 1.1 end, 40, 0.1)
      shot("f4_cannon_stalled", cannon)
      until_(function() return not IsValid(cannon) or cannon:FxStageName() == "failed" end, 60, 0.02)
      for _, dt in ipairs({ 0.1, 0.3, 0.4, 0.4 }) do sleep(dt); shot("f5_cannon_fail_" .. string.format("%02d", math.floor((SysTime() - t0) * 10) % 100), cannon) end
      sleep(1.5)
      out.cannon_removed = not IsValid(cannon)
      out.say_after = H.last_say
      return done(out)
    end

    if scenario == "perf" then
      frame_all(30, 16, 0.9)
      sleep(2.0)
      shot("perf_beams", list[1])
      ply:ConCommand("fps_max 0"); ply:ConCommand("mat_vsync 0")
      sleep(1.5)
      local function perf(label, secs)
        diag(ply, 'local n, t = 0, 0 hook.Add("Think", "tpperf", function() n = n + 1 t = t + RealFrameTime() end) timer.Simple(' .. (secs - 0.2)
          .. ', function() hook.Remove("Think", "tpperf") print(string.format("[tp-perf] ' .. label
          .. ' frames=%d avg_ms=%.3f fps=%.1f fx_ms=%.4f sparkles=%d", n, 1000 * t / math.max(n, 1), n / math.max(t, 1e-9), Mentat.FX and Mentat.FX.cost_ms or -1, Mentat.FX and Mentat.FX.drawn or -1)) end)')
        sleep(secs)
      end
      for _, round in ipairs({ "A", "B" }) do                      -- twice, alternating, so a drift in the machine's load shows
        ply:ConCommand("holodeck_fx 1"); sleep(0.8); perf("beams_on_" .. round, 5)
        ply:ConCommand("holodeck_fx 0"); sleep(0.8); perf("plain_boxes_" .. round, 5)
      end
      ply:ConCommand("holodeck_fx 1")
      for _, e in ipairs(found()) do e:Remove() end
      sleep(0.8); perf("nothing", 5)
      ply:ConCommand("fps_max 300")
      sleep(0.5)
      return done(out)
    end
    done(nil, "unknown scenario " .. tostring(scenario))
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
