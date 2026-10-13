-- face_shots.lua -- photograph a picture head in the real game client (stage/gmod/holodeck/README.md, section Faces, live check).
-- args: { b64 = <a 128x128 jpeg, base64>, hash = <24 hex>, skin = {r,g,b}, views = { {name, dist, yaw, elev, aim_z}, ... } }
-- A bot wears the synthetic picture (set on the server directly, as if the bridge had answered); the local player is the camera. Nothing of
-- the player's own is written: the bot's record is removed again at the end. yaw is relative to the way the bot faces (0 = his face).
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local F = MentatFaces
  if not (IsValid(ply) and F and F.adopt) then return done(nil, "needs the game client and the faces addon") end
  local SVM1 = include("mentat/holodeck/svm1.lua")
  local out = { shots = {}, maxplayers = game.MaxPlayers() }
  local views = args.views or {
    { name = "front", dist = 60, yaw = 0, elev = 4, aim_z = 66 },
    { name = "threequarter", dist = 60, yaw = 40, elev = 6, aim_z = 66 },
    { name = "side", dist = 60, yaw = 90, elev = 4, aim_z = 66 },
    { name = "back", dist = 60, yaw = 180, elev = 6, aim_z = 66 },
    { name = "far", dist = 220, yaw = 25, elev = 6, aim_z = 50 },
  }
  ply:GodEnable()
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
  local function take(prefix, vs, P, base)
    for _, v in ipairs(vs) do
      local ya, el = math.rad((v.yaw or 0) + (base or 0)), math.rad(v.elev or 6)
      local dir = Vector(math.cos(ya) * math.cos(el), math.sin(ya) * math.cos(el), math.sin(el))
      local aim = P + Vector(0, 0, v.aim_z or 66)
      local cam = aim + dir * (v.dist or 60)
      ply:SetPos(cam - Vector(0, 0, 64))
      ply:SetVelocity(-ply:GetVelocity())
      ply:SetEyeAngles((aim - cam):Angle())
      sleep(0.9)
      ply:SetPos(cam - Vector(0, 0, 64))
      ply:SetEyeAngles((aim - cam):Angle())
      sleep(0.9)
      Mentat.Shot(prefix .. v.name)
      out.shots[#out.shots + 1] = prefix .. v.name
      sleep(0.8)
    end
  end

  co = coroutine.create(function()
    sleep(1)
    RunConsoleCommand("bot")
    sleep(2.5)
    local bot = player.GetBots()[1]
    if not IsValid(bot) then out.note = "no bot could be created (maxplayers " .. game.MaxPlayers() .. ")"; return done(out) end
    local P = Vector(0, 0, floor_at(0, 0) + 1)
    bot:SetPos(P); bot:SetEyeAngles(Angle(0, 0, 0)); bot:Freeze(true)
    out.bot_model = bot:GetModel()
    local sid = F.sid_of(bot)
    out.bot_sid = tostring(bot:SteamID64())
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(60, 0)
    sleep(0.5)
    local data = SVM1.b64decode(args.b64 or "")
    if not data then return done(nil, "the picture did not decode") end
    local ans = { hash = args.hash, data = data, crc = tonumber(util.CRC(data)), skin = args.skin or { 128, 128, 128 } }
    out.adopted = F.adopt(sid, ans, "set")
    out.visible_on_server = F.visible(sid)
    sleep(2.5)                                                   -- the picture crosses the (loopback) network in chunks
    diag(ply, [[local FC = MentatFacesClient; local b = nil; for _, p in ipairs(player.GetAll()) do if p:IsBot() then b = p end end
      local e = FC and b and FC.map[b:SteamID64()]
      local img = e and FC.images[e.slot]
      print("[face-diag] map_entry=" .. tostring(e ~= nil) .. " slot=" .. tostring(e and e.slot) .. " image=" .. tostring(img and img.state)
        .. " mat=" .. tostring(img and img.mat) .. " path=" .. tostring(img and img.path) .. " draw_error=" .. tostring(FC and FC.draw_error))
      if b then local bone = b:LookupBone("ValveBiped.Bip01_Head1"); print("[face-diag] model=" .. b:GetModel() .. " headbone=" .. tostring(bone)
        .. " scale=" .. tostring(bone and b:GetManipulateBoneScale(bone)) .. " mode=" .. tostring(FC and FC.mode_of(b))) end
      if img and img.mat then print("[face-diag] mat_error=" .. tostring(img.mat:IsError()) .. " shader=" .. tostring(img.mat:GetShader()) .. " name=" .. tostring(img.mat:GetName())) end]])
    sleep(0.6)
    local base = bot:EyeAngles().y                                -- the way he really faces now (a bot turns its own head)
    out.bot_yaw = base
    take("face_", views, P, base)
    diag(ply, [[print("[face-diag] after: drawn=" .. tostring(MentatFacesClient.stats.drawn) .. " builds=" .. tostring(MentatFacesClient.stats.mesh_builds) .. " draw_error=" .. tostring(MentatFacesClient.draw_error))]])
    -- hidden: the real head must come back at once
    F.rec[sid].hidden = true; F.sync()
    sleep(1.5)
    take("face_hidden_", { views[1] }, P, bot:EyeAngles().y)
    -- leave nothing behind
    F.rec[sid] = nil
    F.sweep(); F.save(); F.sync()
    ply:SetMoveType(MOVETYPE_WALK)
    ply:SetFOV(0, 0)
    RunConsoleCommand("kickid", tostring(bot:UserID()), "done")
    sleep(0.5)
    done(out)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
