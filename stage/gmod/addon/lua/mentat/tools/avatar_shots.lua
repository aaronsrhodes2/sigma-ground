-- avatar_shots.lua -- summon Skippy in the real game client and photograph him (stage/gmod/AVATAR.md, live pass items 2, 3 and 5).
-- args: { views = { {name, dist, yaw, elev, aim_z}, ... } }   yaw is relative to the way HE faces (0 = looking at his face, 90 = his left)
-- Needs the game client (a player with a camera). The follow hook is paused while the pictures are taken so he holds still.
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local Av, H = Mentat and Mentat.Avatar, Holodeck
  if not (IsValid(ply) and Av and H) then return done(nil, "needs the game client, the holodeck and the avatar") end
  local out = { shots = {}, notes = {}, maxplayers = game.MaxPlayers(), players = #player.GetAll() }
  -- the REAL body (the print kit) is about 31 HU tall; the procedural can in a bicorn about 50. `dist` is from the middle of him.
  local views = args.views or {
    { name = "front", dist = 75, yaw = 0, elev = 6, aim_z = 2 },
    { name = "threequarter", dist = 75, yaw = 40, elev = 10, aim_z = 2 },
    { name = "side", dist = 75, yaw = 90, elev = 6, aim_z = 2 },
    { name = "back", dist = 75, yaw = 180, elev = 8, aim_z = 2 },
    { name = "far5m", dist = 262, yaw = 35, elev = 8 },
    { name = "point", dist = 80, yaw = 50, elev = 6, aim_z = 14, point = true },
  }
  ply:GodEnable()
  hook.Add("HolodeckSaid", "avatar_shots_trace", function(p, t) print("[avatar-trace] said '" .. tostring(t) .. "' for " .. (IsValid(p) and p:Nick() or "?") .. " " .. debug.traceback()) end)
  RunConsoleCommand("mentat_avatar_debug", "1")

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

  local prop
  co = coroutine.create(function()
    sleep(1)
    ply:SetPos(Vector(0, 0, floor_at(0, 0) + 1))
    ply:SetEyeAngles(Angle(0, 0, 0))
    sleep(1.5)
    -- summon through the same entry a spoken sentence takes
    H.Handle(ply, "skippy, come here")
    sleep(2.5)
    local bot = Av.bot
    out.summoned = IsValid(bot) and Av.IsActive() or false
    if not out.summoned then return done(out) end
    out.rel = { (bot:GetPos() - ply:GetPos()):Length(), bot:GetPos().z - ply:GetShootPos().z }
    -- the real body arrives in chunks after he is called; give it time (a client without the files keeps the procedural can)
    for _ = 1, 16 do
      diag(ply, [[Mentat.Avatar.BodyC.poll()]])
      sleep(0.5)
    end
    diag(ply, [[print("[avatar-body] " .. tostring(Mentat.Avatar.BodyC.state) .. " " .. tostring(Mentat.Avatar.BodyC.why) .. " tris=" .. tostring(Mentat.Avatar.BodyC.tris) .. " global=" .. tostring(GetGlobalBool("mentat_avatar_body", false)))]])
    sleep(0.3)
    diag(ply, [[print("[avatar-diag] draw=" .. tostring(GetConVar("mentat_avatar_draw"):GetInt()) .. " get=" .. tostring(Mentat.Avatar.Get()) .. " err=" .. tostring(Mentat.Avatar.draw_error) .. " tris=" .. tostring(Mentat.Avatar.Geo and Mentat.Avatar.Geo.triangle_count()))]])
    sleep(0.5)
    -- first person: the player's own view, looking where they look, he is beside them
    ply:SetEyeAngles(Angle(8, 0, 0))
    sleep(0.8)
    Mentat.Shot("av_firstperson")
    sleep(1.0)
    -- the player is the owner and the camera: freeze the follow and park him
    hook.Remove("Tick", "Mentat.AvatarFollow")
    local P = Vector(0, 0, floor_at(0, 0) + 100)
    Av.st.yaw = 0                                         -- (StartCommand pins his view to this)
    bot:SetPos(P); bot:SetEyeAngles(Angle(0, 0, 0)); bot:SetAngles(Angle(0, 0, 0))
    sleep(0.3)
    diag(ply, [[local g = Mentat.Avatar.Geo.build(); local f = g.parts.face; local lo, hi = Mentat.Avatar.Geo.bbox({"face"}); print("[avatar-diag3] face tris=" .. #f.tris .. " bbox x " .. lo.x .. ".." .. hi.x .. " eye yaw=" .. tostring(Mentat.Avatar.Get():EyeAngles().y))]])
    sleep(0.3)
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(70, 0)
    for _, v in ipairs(views) do
      local ya = math.rad((v.yaw or 0))
      local el = math.rad(v.elev or 8)
      local dir = Vector(math.cos(ya) * math.cos(el), math.sin(ya) * math.cos(el), math.sin(el))
      local aim = P + Vector(0, 0, v.aim_z or 6)
      local cam = P + dir * (v.dist or 90)
      ply:SetPos(cam - Vector(0, 0, 64))
      ply:SetVelocity(-ply:GetVelocity())
      ply:SetEyeAngles((aim - cam):Angle())
      ply:ConCommand("mentat_avatar_hide " .. (v.hide or ""))
      for k, val in pairs(v.cvars or {}) do ply:ConCommand(k .. " " .. tostring(val)) end   -- look tuning: a sweep of client convars, one picture each
      if v.point and not prop then
        -- a red ball hung in the air up and ahead of his raised hand; he is told to point at it through the hook the server relays to the clients
        prop = ents.Create("prop_physics")
        prop:SetModel("models/hunter/misc/sphere025x025.mdl")
        prop:SetPos(P + Vector(18, 16, 36))
        prop:SetColor(Color(255, 40, 40))
        prop:Spawn()
        local po = prop:GetPhysicsObject()
        if IsValid(po) then po:EnableMotion(false) end
        Av.Point(prop:GetPos(), 60)
      end
      sleep(0.9)
      ply:SetPos(cam - Vector(0, 0, 64))               -- once more: the first request after a mode change is sometimes ignored
      ply:SetEyeAngles((aim - cam):Angle())
      sleep(0.9)
      local pp, pa = ply:GetPos(), ply:EyeAngles()
      out.cam = out.cam or {}
      out.cam[#out.cam + 1] = { v.name, { pp.x, pp.y, pp.z }, { pa.p, pa.y }, ply:GetMoveType(), engine.TickCount(), FrameTime() }
      Mentat.Shot("av_" .. v.name)
      out.shots[#out.shots + 1] = v.name
      sleep(0.9)
    end
    ply:ConCommand("mentat_avatar_hide \"\"")
    if IsValid(prop) then prop:Remove() end
    Av.Point(nil)
    if args.perf ~= false then
      -- frame cost: uncapped frame rate, he in view at 75 HU; the real body, no avatar at all, the procedural can. avg_ms is the whole frame.
      local cam = P + Vector(75, 0, 4)
      ply:SetPos(cam - Vector(0, 0, 64)); ply:SetEyeAngles((P - cam):Angle())
      ply:ConCommand("fps_max 0"); ply:ConCommand("mat_vsync 0")
      sleep(1.5)
      local function perf(label, secs)
        diag(ply, 'local n, t = 0, 0 hook.Add("Think", "avperf", function() n = n + 1 t = t + RealFrameTime() end) timer.Simple(' .. (secs - 0.2)
          .. ', function() hook.Remove("Think", "avperf") print(string.format("[avatar-perf] ' .. label
          .. ' frames=%d avg_ms=%.3f fps=%.1f draw_ms=%.4f", n, 1000 * t / math.max(n, 1), n / math.max(t, 1e-9), Mentat.Avatar.cost_ms or -1)) end)')
        sleep(secs)
      end
      ply:ConCommand("mentat_avatar_real 1"); ply:ConCommand("mentat_avatar_draw 1"); sleep(0.6); perf("real_body", 4)
      ply:ConCommand("mentat_avatar_draw 0"); sleep(0.6); perf("no_avatar", 4)
      ply:ConCommand("mentat_avatar_draw 1"); ply:ConCommand("mentat_avatar_real 0"); sleep(0.6); perf("procedural", 4)
      ply:ConCommand("mentat_avatar_real 1"); ply:ConCommand("fps_max 300")
      sleep(0.5)
    end
    if args.tone ~= false then
      -- 3D speech: the test tone through the same hook his voice takes, heard from three distances (the console log has what the engine did)
      for _, d in ipairs({ 100, 800, 1000 }) do
        ply:SetPos(P + Vector(d, 0, -64))
        sleep(1.0)
        ply:ConCommand("mentat_avatar_tone")
        sleep(1.4)
      end
    end
    diag(ply, [[print("[avatar-diag2] err=" .. tostring(Mentat.Avatar.draw_error))]])
    sleep(0.5)
    ply:SetMoveType(MOVETYPE_WALK)
    ply:SetFOV(0, 0)
    -- put the follow back and prove he comes back to the player after a teleport
    hook.Add("Tick", "Mentat.AvatarFollow", function()
      local ok, e = pcall(Av.Tick)
      if not ok then print("[avatar-trace] tick error: " .. tostring(e)) end
    end)
    out.owner_state = { visible = Av.visible, owner_valid = IsValid(Av.owner), owner_is_ply = Av.owner == ply, alive = ply:Alive(), bot_valid = IsValid(Av.bot) }
    ply:SetPos(Vector(2000, 500, floor_at(2000, 500) + 1))
    out.tick0 = engine.TickCount()
    sleep(0.3)
    out.snap_back_dist = (bot:GetPos() - ply:GetPos()):Length()
    out.ticks_in_300ms = engine.TickCount() - out.tick0
    sleep(1.0)
    out.snap_back_dist_1_3s = (bot:GetPos() - ply:GetPos()):Length()
    Av.Dismiss()
    sleep(0.5)
    out.nodraw_after_dismiss = bot:GetNoDraw()
    done(out)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
