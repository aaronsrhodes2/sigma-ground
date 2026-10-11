-- avatar_client.lua -- Skippy's avatar on the player's screen and in the player's ears. See stage/gmod/AVATAR.md.
--
--   * The bot's own player model is never drawn. He is drawn from avatar/geometry.lua: a floating beer can in a giant bicorn, as a
--     mesh built once and drawn with colour only (no model, no texture, no material file).
--   * His voice level drives a glow on the label, the mouth, a dynamic light and the way the hat leans toward whoever is talking.
--   * While he is in the world, his spoken clips (skippy_client.lua hands each playing channel to the SkippySpoke hook) are moved to
--     his position every frame and made as loud as proximity voice would make a talker there: Core.gain of the distance, shrunk by
--     a wall between you and him, and halved-and-more when he is whispering. With no avatar in the world nothing here touches a
--     clip, so Skippy sounds as before.
if not CLIENT then return end

local Logic = include("mentat/avatar/logic.lua")
local Geo = include("mentat/avatar/geometry.lua")
local BodyC = include("mentat/avatar/body_client.lua")        -- the real Skippy (the print kit), when its private files were sent
local Core = Logic.Core

Mentat = Mentat or {}
local Av = Mentat.Avatar or {}
Mentat.Avatar = Av
Av.Logic, Av.Geo, Av.BodyC = Logic, Geo, BodyC

local cv_draw = CreateClientConVar("mentat_avatar_draw", "1", true, false, "1 = draw Skippy's avatar when he is in the world")
local cv_light = CreateClientConVar("mentat_avatar_light", "1", true, false, "1 = his glow lights the room a little")
local cv_hide = CreateClientConVar("mentat_avatar_hide", "", false, false, "debugging: part names to leave out (can, face, label, mouth, hat), comma separated")
local cv_scale = CreateClientConVar("mentat_avatar_scale", "1", true, false, "size of the avatar (1 = a 0.5 m can; the real body is mentat_avatar_body_m tall at 1)")
local cv_real = CreateClientConVar("mentat_avatar_real", "1", true, false, "1 = draw the real Skippy when his files arrived; 0 = always the procedural can in a bicorn")
local cv_body_m = CreateClientConVar("mentat_avatar_body_m", "0.6", true, false, "total height in metres of the real body at scale 1")
local cv_point = CreateClientConVar("mentat_avatar_point", "0", true, false, "1 = his hand points at what his owner looks at; 0 = only when told (hook SkippyAvatarPoint)")

--- The bot, when the server says he is in the world.
function Av.Get()
  if not GetGlobalBool("mentat_avatar_on", false) then return nil end
  local e = GetGlobalEntity and GetGlobalEntity("mentat_avatar") or nil
  if e ~= nil and IsValid(e) then return e end
  return nil
end

-- ------------------------------------------------------------------------------------------------ 3D speech
local tracked = {}          -- channel -> { base = loudness the player asked for }
Av.tracked = tracked
Av.gain, Av.heard = 1, true

--- How well this client hears him right now: (heard, gain, dist, range, blocked), or nil when he is not in the world.
function Av.Hearing()
  local bot, me = Av.Get(), LocalPlayer()
  if not bot or not IsValid(me) then return nil end
  local at, eye = bot:GetPos(), me:EyePos()
  local dist = eye:Distance(at)
  local range = bot:GetNWFloat("mentat_avatar_range", Core.DEFAULTS.range)
  local blocked = util.TraceLine({ start = eye, endpos = at, mask = MASK_SOLID_BRUSHONLY }).Hit
  local heard, gain = Logic.hear(dist, range, blocked, GetGlobalFloat("mentat_voice_wallf", Core.DEFAULTS.wall), nil,
    GetGlobalString("mentat_voice_mode", "proximity"))
  return heard, gain, dist, range, blocked
end

local function apply(chan, entry, gain, snap)
  if not IsValid(chan) then return end
  local bot = Av.Get()
  if bot then chan:SetPos(bot:GetPos()) end
  local want = entry.base * gain
  entry.vol = snap and want or Logic.approach(entry.vol or want, want, 18, FrameTime and FrameTime() or 0.015)
  chan:SetVolume(math.Clamp(entry.vol, 0, 1))
end

--- Once per 0.1 s: the (expensive) wall trace. Per frame: follow his position and ease the volume.
local function refresh_gain()
  local heard, gain = Av.Hearing()
  if heard == nil and gain == nil then return end
  Av.heard, Av.gain = heard, gain
end

--- Bring every tracked clip up to date (position every call, gain from the last trace). `snap` jumps the volume.
function Av.UpdateVoices(snap)
  for chan, entry in pairs(tracked) do
    if not IsValid(chan) or chan:GetState() == GMOD_CHANNEL_STOPPED then tracked[chan] = nil
    else apply(chan, entry, Av.gain, snap) end
  end
end

-- a clip just started (skippy_client.lua): while he is in the world it becomes HIS voice
hook.Add("SkippySpoke", "Mentat.AvatarVoice", function(chan)
  if not Av.Get() then return end
  local cv = GetConVar("skippy_volume")
  tracked[chan] = { base = math.Clamp(cv and cv:GetFloat() or 1, 0, 1) }
  -- the engine's own distance fade is made flat: the fade is Core.gain's job, so it matches proximity voice exactly
  if chan.Set3DFadeDistance then chan:Set3DFadeDistance(100000, 200000) end
  refresh_gain()
  Av.UpdateVoices(true)
end)

timer.Create("Mentat.AvatarHear", 0.1, 0, function()
  if next(tracked) ~= nil then refresh_gain() end
end)
hook.Add("Think", "Mentat.AvatarVoiceFollow", function()
  if next(tracked) ~= nil then Av.UpdateVoices(false) end
end)

-- ------------------------------------------------------------------------------------------------ the test tone
--- A 1.5 s pulsing tone (16-bit mono WAV bytes), for hearing (and measuring) where his voice comes from without the bridge.
function Av.ToneWav(seconds, rate)
  seconds, rate = seconds or 1.5, rate or 22050
  local n = math.floor(seconds * rate)
  local out = {}
  local function u32(v) return string.char(v % 256, math.floor(v / 256) % 256, math.floor(v / 65536) % 256, math.floor(v / 16777216) % 256) end
  local function u16(v) return string.char(v % 256, math.floor(v / 256) % 256) end
  out[1] = "RIFF" .. u32(36 + n * 2) .. "WAVEfmt " .. u32(16) .. u16(1) .. u16(1) .. u32(rate) .. u32(rate * 2) .. u16(2) .. u16(16) .. "data" .. u32(n * 2)
  local chunk = {}
  for i = 0, n - 1 do
    local t = i / rate
    local v = math.floor(12000 * math.sin(2 * math.pi * 440 * t) * (0.6 + 0.4 * math.sin(2 * math.pi * 3 * t)))
    if v < 0 then v = v + 65536 end
    chunk[#chunk + 1] = u16(v)
    if #chunk >= 4096 then out[#out + 1] = table.concat(chunk); chunk = {} end
  end
  out[#out + 1] = table.concat(chunk)
  return table.concat(out)
end

--- Play the tone as if it were his voice (the same hook the real clips take) and report what the engine did with it.
function Av.PlayTone(report)
  if not Av.Get() then print("[avatar] tone: he is not in the world (say \"Skippy, come here\" first)"); return end
  file.CreateDir("mentat")
  file.Write("mentat/avatar_tone.wav", Av.ToneWav())
  sound.PlayFile("data/mentat/avatar_tone.wav", "3d mono", function(chan, errid, errname)
    if not IsValid(chan) then print("[avatar] tone: could not play: " .. tostring(errid) .. " " .. tostring(errname)); return end
    local bot = Av.Get()
    if bot then chan:SetPos(bot:GetPos()) end
    chan:Set3DFadeDistance(250, 2400)
    chan:SetVolume(1)
    chan:Play()
    hook.Run("SkippySpoke", chan)
    local info = { dist = nil }
    timer.Simple(0.5, function()
      if not IsValid(chan) then return end
      local heard, gain, dist, range, blocked = Av.Hearing()
      local l, r = 0, 0
      if chan.GetLevel then l, r = chan:GetLevel() end
      local px, py, pz = 0, 0, 0
      if chan.GetPos then local v = chan:GetPos(); px, py, pz = v.x, v.y, v.z end
      local tr = tracked[chan]
      info = { volume = chan:GetVolume(), want = tr and (tr.base * Av.gain) or nil, gain = gain, heard = heard, dist = dist, range = range, blocked = blocked,
               level_l = l, level_r = r, chan_pos = { px, py, pz }, tracked = tr ~= nil, state = chan:GetState() }
      print("[avatar] tone: " .. util.TableToJSON(info))
      if report then report(info) end
    end)
  end)
end
concommand.Add("mentat_avatar_tone", function() Av.PlayTone() end, nil, "Play a test tone from where Skippy hovers (hear the 3D position; prints what the engine did)")

-- ------------------------------------------------------------------------------------------------ his loudness
Av.level = 0
local function channel_level()
  local best
  for chan in pairs(tracked) do
    if IsValid(chan) and chan:GetState() == GMOD_CHANNEL_PLAYING then
      local l, r = 0, 0
      if chan.GetLevel then l, r = chan:GetLevel() end
      if isnumber(l) and isnumber(r) and (l > 0 or r > 0) then best = math.max(best or 0, l, r)
      else best = math.max(best or 0, Logic.speech_envelope(CurTime())) end        -- the engine would not tell: a timed envelope
    end
  end
  return best
end

-- ------------------------------------------------------------------------------------------------ drawing
local meshes, light_id

local function build_meshes()
  meshes = {}
  local geo = Geo.build()
  for _, name in ipairs(geo.order) do
    local part = geo.parts[name]
    local m = Mesh()
    mesh.Begin(m, MATERIAL_TRIANGLES, #part.tris)
    for _, t in ipairs(part.tris) do
      local r, g, b = Geo.vertex_color(t)
      for _, v in ipairs({ t.a, t.b, t.c }) do
        mesh.Position(Vector(v.x, v.y, v.z))
        mesh.Normal(Vector(t.n.x, t.n.y, t.n.z))
        mesh.Color(r, g, b, 255)
        mesh.AdvanceVertex()
      end
    end
    mesh.End()
    meshes[name] = m
  end
end

local function who_is_talking(bot)
  local best, best_d
  for _, p in ipairs(player.GetAll()) do
    if p ~= bot and p:IsSpeaking() then
      local d = p:GetPos():Distance(bot:GetPos())
      if d < 900 and (not best_d or d < best_d) then best, best_d = p, d end
    end
  end
  if best then return best end
  local owner = bot:GetNWEntity("mentat_avatar_owner")
  if IsValid(owner) then return owner end
  return nil
end

local pitch, roll, last_t = 0, 0, nil

-- ------------------------------------------------------------------------------------------------ pointing
-- Hook SkippyAvatarPoint(pos, seconds): the real body's right hand swings to aim at `pos` for `seconds` (4 by default); pos nil lowers it.
-- The server sends the order to every client (avatar.lua); mentat_avatar_point 1 makes the hand follow his owner's gaze instead.
Av.aim = nil
function Av.Point(pos, seconds)
  if pos == nil then Av.aim = nil; return end
  Av.aim = { pos = pos, t1 = CurTime() + (seconds or 4) }
end
hook.Add("SkippyAvatarPoint", "Mentat.AvatarPoint", function(pos, seconds) Av.Point(pos, seconds) end)
net.Receive("mentat_avatar_point", function()
  local on = net.ReadBool()
  local pos = on and net.ReadVector() or nil
  local secs = net.ReadUInt(16) / 10
  Av.Point(pos, secs)
end)

--- The world point his hand should aim at now, or nil (rest pose).
local function aim_point(bot)
  local a = Av.aim
  if a and CurTime() < a.t1 then return a.pos end
  if cv_point:GetBool() then
    local owner = bot:GetNWEntity("mentat_avatar_owner")
    if IsValid(owner) then
      local tr = owner:GetEyeTrace()
      if tr and tr.Hit then return tr.HitPos end
    end
  end
  return nil
end

--- The real body: the print kit's six parts, animated. Drawn with the same depth-writing rule as the procedural body.
local function draw_real(bot, dt, target, yaw)
  local lx, ly
  if target then
    local d = target:EyePos() - bot:GetPos()
    local rad = math.rad(-yaw)
    lx, ly = d.x * math.cos(rad) - d.y * math.sin(rad), d.x * math.sin(rad) + d.y * math.cos(rad)
  end
  local hide = "," .. cv_hide:GetString():gsub("%s", "") .. ","
  render.SetColorMaterial()
  render.CullMode(MATERIAL_CULLMODE_NONE)
  if render.OverrideDepthEnable then render.OverrideDepthEnable(true, true) end
  BodyC.draw(bot, { level = Av.level, lx = lx, ly = ly, aim = aim_point(bot), dt = dt, height_m = cv_body_m:GetFloat(),
    scale = math.Clamp(cv_scale:GetFloat(), 0.25, 4), hat_tilt = Logic.hat_tilt,
    hide = function(name) return hide:find("," .. name .. ",", 1, true) ~= nil end })
  if render.OverrideDepthEnable then render.OverrideDepthEnable(false, false) end
  render.CullMode(MATERIAL_CULLMODE_CCW)
end

local function draw()
  local bot = Av.Get()
  if not bot or not cv_draw:GetBool() then return end
  local real = cv_real:GetBool() and BodyC.ready
  if not real and not meshes then build_meshes() end
  local now = CurTime()
  local dt = math.min(math.max(now - (last_t or now), 0), 0.1)
  last_t = now

  -- how loud he is: the voice if there is one, else he settles to a slow breathing glow
  local lvl = channel_level()
  Av.level = Logic.approach(Av.level, lvl or 0, lvl and 20 or 6, dt)
  local glow = 0.55 + 0.45 * Av.level

  -- the hat leans toward whoever is talking
  local ang = bot:EyeAngles()
  local yaw = ang.y
  local target = who_is_talking(bot)
  local pos = bot:GetPos()
  if real then
    draw_real(bot, dt, target, yaw)
    Av.drew_real = true
  else
    Av.drew_real = false
    local tp, tr = 0, 0
    if target then
      local d = target:EyePos() - bot:GetPos()
      local rad = math.rad(-yaw)                                 -- into his frame: x forward, y to his left
      local lx, ly = d.x * math.cos(rad) - d.y * math.sin(rad), d.x * math.sin(rad) + d.y * math.cos(rad)
      tp, tr = Logic.hat_tilt(lx, ly, 0.35 + 0.65 * Av.level)
    end
    pitch, roll = Logic.approach(pitch, tp, 6, dt), Logic.approach(roll, tr, 6, dt)
  local base = Matrix()
  base:Translate(pos)
  base:Rotate(Angle(0, yaw, 0))
  base:Scale(Vector(1, 1, 1) * math.Clamp(cv_scale:GetFloat(), 0.25, 4))

  local hide = "," .. cv_hide:GetString():gsub("%s", "") .. ","
  local function shown(name) return not hide:find("," .. name .. ",", 1, true) end
  render.SetColorMaterial()
  render.CullMode(MATERIAL_CULLMODE_NONE)                       -- two-sided: the hat is a thin shell
  -- the colour material is translucent: it does not write depth, so the label painted over the eyes drawn before it (live pass).
  -- With depth writing forced on, what is in front stays in front whatever order the parts are drawn in.
  if render.OverrideDepthEnable then render.OverrideDepthEnable(true, true) end
  cam.PushModelMatrix(base)
  if shown("can") then meshes.can:Draw() end
  if shown("face") then meshes.face:Draw() end
  render.SetColorModulation(glow, glow, glow)                   -- the label brightens with his voice
  if shown("label") then meshes.label:Draw() end
  render.SetColorModulation(1, 1, 1)
  -- the mouth opens with the voice
  local mp = Geo.PIVOT_MOUTH
  local mm = Matrix()
  mm:Translate(Vector(mp.x, mp.y, mp.z))
  mm:Scale(Vector(1, 1, 0.5 + 3.5 * Av.level))
  mm:Translate(Vector(-mp.x, -mp.y, -mp.z))
  cam.PushModelMatrix(mm, true)
  if shown("mouth") then meshes.mouth:Draw() end
  cam.PopModelMatrix()
  -- the hat tips about the top of the can
  local hp = Geo.PIVOT_HAT
  local hm = Matrix()
  hm:Translate(Vector(hp.x, hp.y, hp.z))
  hm:Rotate(Angle(pitch, 0, roll))
  hm:Translate(Vector(-hp.x, -hp.y, -hp.z))
  cam.PushModelMatrix(hm, true)
  if shown("hat") then meshes.hat:Draw() end
  cam.PopModelMatrix()
  cam.PopModelMatrix()
  if render.OverrideDepthEnable then render.OverrideDepthEnable(false, false) end
  render.CullMode(MATERIAL_CULLMODE_CCW)
  end

  if cv_light:GetBool() and DynamicLight then
    light_id = light_id or (bot:EntIndex() + 4096)
    local dl = DynamicLight(light_id)
    if dl then
      dl.pos, dl.r, dl.g, dl.b = pos, 0, 255, 136
      dl.brightness, dl.size, dl.decay, dl.dietime = 1 + 2 * Av.level, 110 + 120 * Av.level, 800, now + 0.1
    end
  end
end

hook.Add("PostDrawTranslucentRenderables", "Mentat.AvatarDraw", function(_, skybox)
  if skybox then return end
  local t0 = SysTime()
  local ok, err = pcall(draw)
  Av.cost_ms = (Av.cost_ms or 0) * 0.95 + (SysTime() - t0) * 1000 * 0.05          -- CPU time of the draw call, smoothed (the GPU's share is in the frame time)
  if not ok and not Av.draw_error then
    Av.draw_error = tostring(err)
    print("[avatar] could not draw Skippy: " .. Av.draw_error)
  end
end)

-- the default model of the bot (a Half-Life player) is never shown
hook.Add("PrePlayerDraw", "Mentat.AvatarHideBody", function(ply)
  if GetGlobalEntity and ply == GetGlobalEntity("mentat_avatar") then return true end
end)
