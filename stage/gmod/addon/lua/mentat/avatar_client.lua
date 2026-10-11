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
local Core = Logic.Core

Mentat = Mentat or {}
local Av = Mentat.Avatar or {}
Mentat.Avatar = Av
Av.Logic, Av.Geo = Logic, Geo

local cv_draw = CreateClientConVar("mentat_avatar_draw", "1", true, false, "1 = draw Skippy's avatar when he is in the world")
local cv_light = CreateClientConVar("mentat_avatar_light", "1", true, false, "1 = his glow lights the room a little")
local cv_scale = CreateClientConVar("mentat_avatar_scale", "1", true, false, "size of the avatar (1 = a 0.5 m can)")

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

local function draw()
  local bot = Av.Get()
  if not bot or not cv_draw:GetBool() then return end
  if not meshes then build_meshes() end
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
  local tp, tr = 0, 0
  if target then
    local d = target:EyePos() - bot:GetPos()
    local rad = math.rad(-yaw)                                 -- into his frame: x forward, y to his left
    local lx, ly = d.x * math.cos(rad) - d.y * math.sin(rad), d.x * math.sin(rad) + d.y * math.cos(rad)
    tp, tr = Logic.hat_tilt(lx, ly, 0.35 + 0.65 * Av.level)
  end
  pitch, roll = Logic.approach(pitch, tp, 6, dt), Logic.approach(roll, tr, 6, dt)

  local pos = bot:GetPos()
  local base = Matrix()
  base:Translate(pos)
  base:Rotate(Angle(0, yaw, 0))
  base:Scale(Vector(1, 1, 1) * math.Clamp(cv_scale:GetFloat(), 0.25, 4))

  render.SetColorMaterial()
  render.CullMode(MATERIAL_CULLMODE_NONE)                       -- two-sided: the hat is a thin shell
  cam.PushModelMatrix(base)
  meshes.can:Draw()
  meshes.face:Draw()
  render.SetColorModulation(glow, glow, glow)                   -- the label brightens with his voice
  meshes.label:Draw()
  render.SetColorModulation(1, 1, 1)
  -- the mouth opens with the voice
  local mp = Geo.PIVOT_MOUTH
  local mm = Matrix()
  mm:Translate(Vector(mp.x, mp.y, mp.z))
  mm:Scale(Vector(1, 1, 0.5 + 3.5 * Av.level))
  mm:Translate(Vector(-mp.x, -mp.y, -mp.z))
  cam.PushModelMatrix(mm, true)
  meshes.mouth:Draw()
  cam.PopModelMatrix()
  -- the hat tips about the top of the can
  local hp = Geo.PIVOT_HAT
  local hm = Matrix()
  hm:Translate(Vector(hp.x, hp.y, hp.z))
  hm:Rotate(Angle(pitch, 0, roll))
  hm:Translate(Vector(-hp.x, -hp.y, -hp.z))
  cam.PushModelMatrix(hm, true)
  meshes.hat:Draw()
  cam.PopModelMatrix()
  cam.PopModelMatrix()
  render.CullMode(MATERIAL_CULLMODE_CCW)

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
  local ok, err = pcall(draw)
  if not ok and not Av.draw_error then
    Av.draw_error = tostring(err)
    print("[avatar] could not draw Skippy: " .. Av.draw_error)
  end
end)

-- the default model of the bot (a Half-Life player) is never shown
hook.Add("PrePlayerDraw", "Mentat.AvatarHideBody", function(ply)
  if GetGlobalEntity and ply == GetGlobalEntity("mentat_avatar") then return true end
end)
