-- found_look.lua -- the look pass on found objects: summon several real found objects one after another (normal path: H.Handle -> bridge
-- /resolve -> box -> /mesh -> chunks -> decode), photograph each from the same fixed cameras once the real shape stands, measure the
-- frame cost with the shape drawn, remove it and go on. args: { items = { { tag =, request = }, ... }, bridge =, token =, timeout = 90,
-- prefix = "lk", views = optional }. stage/gmod/holodeck/found_look.py runs it. Hands off: the window stays behind, no mouse.
return function(args, done, ply)
  local H = Holodeck
  if not (IsValid(ply) and H) then return done(nil, "needs the game client and the holodeck") end
  local prefix = args.prefix or "lk"
  local out = { items = {}, frames = {} }
  ply:GodEnable()
  ply:StripWeapons()
  RunConsoleCommand("holodeck_bridge", args.bridge)
  RunConsoleCommand("holodeck_bridge_token", args.token or "")
  RunConsoleCommand("holodeck_agent", "0")
  RunConsoleCommand("holodeck_persona", "0")
  RunConsoleCommand("holodeck_voice_out", "0")
  RunConsoleCommand("holodeck_found_timeout", tostring(args.timeout or 90))
  ply:ConCommand("holodeck_found_outline 0")                          -- no box around the real shape (the client config may remember a 1)
  ply:ConCommand("holodeck_found_lit " .. tostring(args.lit == false and 0 or 1))
  -- (2026-10-12) the water module (mentat/water.lua, another feature, still in progress) wraps Holodeck.Handle on top of the avatar's wrapper, and the
  -- avatar then wraps THAT again: Handle calls itself without end and the game freezes on the first request. This run is about how things look, so
  -- the water layer is taken off for its duration (this session only; nothing on disk changes).
  local W = Mentat and Mentat.Water
  if W and W.wrapped and W.inner and H.Handle == W.wrapped then H.Handle = W.inner end
  hook.Remove("Tick", "MentatWater.Tick")
  hook.Remove("Move", "MentatWater.Swim")
  if args.diag then
    print("[fl-trace] water=" .. tostring(W ~= nil) .. " wrapped=" .. tostring(W and W.wrapped ~= nil) .. " handle_is_inner=" .. tostring(W and H.Handle == W.inner))
    local depth, orig = 0, H.Handle
    H.Handle = function(...)
      depth = depth + 1
      if depth > 6 then
        print("[fl-trace] RECURSION " .. debug.traceback())
        depth = 0
        error("recursion in Handle")
      end
      local r = { pcall(orig, ...) }
      depth = math.max(0, depth - 1)
      if not r[1] then error(r[2], 0) end
      return unpack(r, 2)
    end
  end
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
  local pin
  local function trace(msg) print("[fl-trace] " .. msg) end
  local function shot(name)
    trace("shot " .. name)
    if pin then ply:SetPos(pin.pos); ply:SetEyeAngles(pin.ang) end
    Mentat.Shot(prefix .. "_" .. name)
    out.frames[#out.frames + 1] = name
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
  local function found() return ents.FindByClass("holodeck_found") end
  local views = args.views or { { "iso", 35, 18, 1 }, { "front", 0, 8, 1 }, { "side", 90, 8, 1 }, { "back", 215, 25, 1 }, { "close", 20, 14, 0.6 }, { "top", 0, 80, 0.8 } }

  co = coroutine.create(function()
    sleep(1)
    ply:SetPos(Vector(0, 0, floor_at(0, 0) + 1))
    ply:SetEyeAngles(Angle(0, 0, 0))
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(70, 0)
    sleep(1.5)
    for _, it in ipairs(args.items) do
      local rec = { tag = it.tag, request = it.request }
      out.items[#out.items + 1] = rec
      for _, e in ipairs(found()) do e:Remove() end
      sleep(0.3)
      local t0 = SysTime()
      ply:SetPos(Vector(0, 0, floor_at(0, 0) + 1))
      ply:SetEyeAngles(Angle(0, 0, 0))
      trace("handle " .. it.tag)
      local res = H.Handle(ply, it.request)
      rec.say = res and res.say or ""
      local appeared = until_(function() return #found() > 0 end, 40, 0.05)
      local e = found()[1]
      if not appeared or not IsValid(e) then
        rec.note = "no object stood"
      else
        local limit, waited = (args.timeout or 90) + 5, 0
        while IsValid(e) and waited < limit do
          local st = e:FxStageName()
          if st == "done" or st == "failed" then break end
          sleep(0.1); waited = waited + 0.1
        end
        if not IsValid(e) or e:FxStageName() == "failed" then
          rec.note = "transport failed"
        else
          trace("arrived " .. it.tag)
          sleep(2.6)                                                       -- the arrival reveal and fade finish
          rec.t_total = SysTime() - t0
          local mn, mx = e:GetBoxMins(), e:GetBoxMaxs()
          local ext = math.max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)
          local c = e:GetPos() + Vector(0, 0, (mx.z + mn.z) / 2)
          local dist = math.max(ext * 2.3, 90)
          rec.visual, rec.mesh_id, rec.label = e:GetVisualKind(), e:GetMeshId(), e:LabelText()
          rec.size_hu = { mx.x - mn.x, mx.y - mn.y, mx.z - mn.z }
          for _, v in ipairs(views) do
            look(c, dist * v[4], v[2], v[3]); sleep(0.7); shot(it.tag .. "_" .. v[1]); sleep(0.4)
          end
          look(c, dist, 35, 18)
          if not args.noperf then
            -- the CPU cost of drawing this object once (200 draws inside the translucent pass, timed with SysTime), averaged over about 1.5 s
            ply:ConCommand("holodeck_mesh_bench " .. it.tag)
            sleep(1.9)
          end
        end
      end
      if IsValid(e) then e:Remove() end
      sleep(0.5)
    end
    return done(out)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
