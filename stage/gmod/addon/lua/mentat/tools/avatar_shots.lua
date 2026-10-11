-- avatar_shots.lua -- summon Skippy in the real game client and photograph him (stage/gmod/AVATAR.md, live pass items 2, 3 and 5).
-- args: { views = { {name, dist, yaw, elev, aim_z}, ... } }   yaw is relative to the way HE faces (0 = looking at his face, 90 = his left)
-- Needs the game client (a player with a camera). The follow hook is paused while the pictures are taken so he holds still.
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local Av, H = Mentat and Mentat.Avatar, Holodeck
  if not (IsValid(ply) and Av and H) then return done(nil, "needs the game client, the holodeck and the avatar") end
  local out = { shots = {}, notes = {}, maxplayers = game.MaxPlayers(), players = #player.GetAll() }
  local views = args.views or {
    { name = "front", dist = 90, yaw = 0, elev = 8 },
    { name = "side", dist = 90, yaw = 90, elev = 8 },
    { name = "far5m", dist = 262, yaw = 35, elev = 8 },
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
