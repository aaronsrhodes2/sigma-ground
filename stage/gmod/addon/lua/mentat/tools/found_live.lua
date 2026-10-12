-- found_live.lua -- one REAL found-object request in the real game client: the normal path (H.Handle -> bridge /resolve -> box ->
-- /mesh -> chunks -> decode), a time-lapse of the beam and the arrival, then close views and measurements of what stands.
-- args: { request = "add a ...", bridge =, token =, tag = "x", timeout = 90 }   (stage/gmod/holodeck/found_live.py runs it)
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local H = Holodeck
  if not (IsValid(ply) and H) then return done(nil, "needs the game client and the holodeck") end
  local tag = args.tag or "fl"
  local out = { tag = tag, frames = {}, timeline = {}, notes = {} }
  ply:GodEnable()
  ply:StripWeapons()
  RunConsoleCommand("holodeck_bridge", args.bridge)
  RunConsoleCommand("holodeck_bridge_token", args.token or "")
  RunConsoleCommand("holodeck_agent", "0")
  RunConsoleCommand("holodeck_persona", "0")
  RunConsoleCommand("holodeck_voice_out", "0")
  RunConsoleCommand("holodeck_found_timeout", tostring(args.timeout or 90))
  if args.chunk and H.Mesh then H.Mesh.CHUNK = args.chunk end       -- the one live-test knob (as in transporter_shots): a slower SEND, so progress inside it can be photographed
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
  local pin
  local function state(e)
    if not IsValid(e) then return "gone", 0 end
    return e:FxStageName() or "none", e:GetFxProgress()
  end
  local function shot(name, e)
    if pin then ply:SetPos(pin.pos); ply:SetEyeAngles(pin.ang) end
    local st, pr = state(e)
    out.frames[#out.frames + 1] = { name = name, stage = st, progress = pr, t = SysTime() - t0, visual = IsValid(e) and e:GetVisualKind() or -1 }
    Mentat.Shot(tag .. "_" .. name)
  end
  local function look(c, dist, yaw, elev)
    local ya, el = math.rad(yaw), math.rad(elev)
    local cam = c + Vector(math.cos(ya) * math.cos(el), math.sin(ya) * math.cos(el), math.sin(el)) * dist
    ply:SetPos(cam - Vector(0, 0, 64))
    ply:SetVelocity(-ply:GetVelocity())
    ply:SetEyeAngles((c - cam):Angle())
    pin = { pos = cam - Vector(0, 0, 64), ang = (c - cam):Angle() }
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
    local tr = SysTime()
    local res = H.Handle(ply, args.request)
    out.say = res and res.say or ""
    local appeared = until_(function() return #found() > 0 end, 40, 0.05)
    out.t_entity = SysTime() - tr                                  -- the request until the box stands (the /resolve, including a download that sizes the card)
    sleep(0.3)
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(70, 0)
    local e = found()[1]
    if not appeared or not IsValid(e) then
      out.notes[#out.notes + 1] = "no object stood"
      sleep(4)
      out.say_after = H.last_say
      out.wisps = #ents.FindByClass("holodeck_wisp")
      local w = ents.FindByClass("holodeck_wisp")[1]
      local c = IsValid(w) and w:GetPos() or Vector(120, 0, floor_at(120, 0) + 64)
      look(c, 110, 180, 5)
      shot("nothing_wisp", nil)
      return done(out)
    end
    local mn, mx = e:GetBoxMins(), e:GetBoxMaxs()
    local ext = math.max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)
    local c = e:GetPos() + Vector(0, 0, (mx.z + mn.z) / 2)
    local dist = math.max(ext * 2.3, 90)
    out.label = e:LabelText()
    out.card_hu = { mx.x - mn.x, mx.y - mn.y, mx.z - mn.z }
    out.found_source = e:GetFoundSource()
    local function frame(yaw, elev, k) look(c, dist * (k or 1), yaw, elev) end
    frame(35, 18)
    sleep(0.4)
    shot("a_start", e)
    -- the time-lapse: a picture at each stage change and each quarter of the progress, plus a clock every 2.5 s; and a timeline of every change
    local last_stage, last_prog, last_shot_t, n = "", -1, SysTime(), 0
    local marks = { 0.2, 0.4, 0.6, 0.75, 0.85 }
    local mi = 1
    local limit = (args.timeout or 90) + 5
    local waited = 0
    local finished = false
    while waited < limit do
      if not IsValid(e) then break end
      local st, pr = state(e)
      if st ~= last_stage or math.abs(pr - last_prog) > 0.02 then
        out.timeline[#out.timeline + 1] = { t = SysTime() - tr, stage = st, progress = pr, vis = e:GetVisualKind() }
        last_stage, last_prog = st, pr
      end
      local want = false
      if mi <= #marks and pr >= marks[mi] then want = true; while mi <= #marks and pr >= marks[mi] do mi = mi + 1 end end
      if st == "decoding" or st == "done" then
        if n < 8 and SysTime() - last_shot_t > 0.12 then want = true end
      elseif SysTime() - last_shot_t > 2.5 then want = true end
      if want and n < 9 then
        n = n + 1; last_shot_t = SysTime()
        shot(string.format("b_%02d_%s", n, st), e)
      end
      if st == "done" then finished = true; break end
      if st == "failed" then                                       -- the collapse (1.5 s), then the placeholder goes: photograph it
        out.failed = true
        for k = 1, 4 do sleep(0.35); if IsValid(e) then shot(string.format("f_%02d_collapse", k), e) end end
        until_(function() return not IsValid(e) end, 5, 0.1)
        break
      end
      sleep(0.05); waited = waited + 0.05
    end
    out.finished = finished
    out.t_done = SysTime() - tr
    out.entity_alive = IsValid(e)
    if not IsValid(e) or out.failed then
      out.notes[#out.notes + 1] = "the object went away (a failure takes the placeholder down)"
      sleep(2.5)
      out.say_after = H.last_say
      local w = ents.FindByClass("holodeck_wisp")[1]
      out.wisps = #ents.FindByClass("holodeck_wisp")
      if IsValid(w) then look(w:GetPos(), 100, 180, 5) else frame(35, 18) end
      shot("z_after_failure", nil)
      return done(out)
    end
    sleep(2.2)                                                     -- the reveal and the fade finish
    shot("c_arrived", e)
    -- what stands: the pose and the size the ENGINE has (the entity's own box, which the card sets), and the floor
    local p, a = e:GetPos(), e:GetAngles()
    local fz = floor_at(p.x, p.y)
    out.entity = { pos = { p.x, p.y, p.z }, ang = { a.p, a.y, a.r }, mins = { mn.x, mn.y, mn.z }, maxs = { mx.x, mx.y, mx.z },
      bottom_above_floor = p.z + mn.z - fz, visual = e:GetVisualKind(), mesh_id = e:GetMeshId(), label = e:LabelText(), fx_stage = e:FxStageName() }
    local ph = e:GetPhysicsObject()
    if IsValid(ph) then out.entity.mass = ph:GetMass(); out.entity.asleep = ph:IsAsleep() end
    out.n_found = #found()
    for _, v in ipairs({ { "d_iso", 35, 18, 1 }, { "d_front", 0, 6, 1 }, { "d_side", 90, 6, 1 }, { "d_back", 200, 18, 1 }, { "d_top", 0, 80, 1.05 }, { "d_close", 35, 22, 0.65 } }) do
      frame(v[2], v[3], v[4]); sleep(0.5); shot(v[1], e)
    end
    -- frame cost with the mesh drawn (beam gone), against the same object drawn as the box
    frame(35, 18)
    ply:ConCommand("fps_max 0"); ply:ConCommand("mat_vsync 0")
    sleep(1.0)
    local function perf(label, secs)
      diag(ply, 'local n, t = 0, 0 hook.Add("Think", "flperf", function() n = n + 1 t = t + RealFrameTime() end) timer.Simple(' .. (secs - 0.2)
        .. ', function() hook.Remove("Think", "flperf") print(string.format("[fl-perf] ' .. label .. ' frames=%d avg_ms=%.3f fps=%.1f", n, 1000 * t / math.max(n, 1), n / math.max(t, 1e-9))) end)')
      sleep(secs)
    end
    perf("mesh_drawn_A", 4)
    if IsValid(e) and out.entity.visual == 1 then
      e:SetVisual("box", {}); sleep(0.8); perf("box_B", 4)
      e:SetVisual("mesh", { id = out.entity.mesh_id }); sleep(0.8); perf("mesh_drawn_C", 4)
    end
    ply:ConCommand("fps_max 300")
    sleep(0.3)
    return done(out)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
