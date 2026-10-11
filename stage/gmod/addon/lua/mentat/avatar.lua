-- avatar.lua -- Skippy as a presence in the world (server side). See stage/gmod/AVATAR.md.
--
-- When a player calls him ("Skippy, come here", typed or spoken, or the console command skippy_come) a bot player named Skippy
-- appears beside them (SetPos: instantly, no walking), hovers near them and follows smoothly, and his spoken answers come from
-- where he hovers (3D, with the same range, wall and whisper rules as proximity voice) instead of from inside the head.
-- "Skippy, go away" sends him off. ONE shared Skippy: he goes to whoever called last (several places at once are a later task).
--
-- The bot is only a body for the server: it holds a player slot, has no weapons, cannot be hurt, touches nothing, is NOT SOLID (so
-- nobody's aim or movement is blocked by him), and its own bot AI is switched off (StartCommand hands the engine an empty command).
-- Its player model is never drawn: the clients draw him from procedural geometry (avatar_client.lua + avatar/geometry.lua).
--
-- Hooks a later effect can use:  SkippyAvatarArrive(bot, from, to)  from = where he was (nil if he was away), to = where he is now
--                                SkippyAvatarLeave(bot, from)       where he was when he left / before he jumped
-- Convars:  mentat_avatar_enabled 1     0 = the words "come here" go to the holodeck as before and nobody can call him
--           mentat_avatar_kick 0        1 = going away kicks the bot (frees the slot; he rejoins, with a join message, when called)
--                                       0 = he is hidden and stays on the server (default: no join/leave noise, no re-creation)
--           mentat_avatar_debug 0       1 = every spoken line prints, per listener: distance, range, blocked, gain
--           mentat_avatar_ack_voice 1   1 = his "here I am" is spoken aloud, 0 = text only
-- Console:  skippy_come, skippy_go, mentat_avatar_status.   Lua: Mentat.Avatar.Summon(ply), .Dismiss(), .Get(), .IsActive(), .WhoHears(ply)
if not SERVER then return end

Mentat = Mentat or {}
local Logic = include("mentat/avatar/logic.lua")
local Core = Logic.Core
local BodyNet = include("mentat/avatar/body_net.lua")        -- the real Skippy's private files, read and sent to the clients once

local Av = Mentat.Avatar or {}
Mentat.Avatar = Av
Av.Logic = Logic
Av.Body = BodyNet
Av.visible = Av.visible or false
Av.st = Av.st or { pos = { x = 0, y = 0, z = 0 }, yaw = 0, t = 0 }
Av.wrapped = Av.wrapped or setmetatable({}, { __mode = "k" })

local cv_enabled = CreateConVar("mentat_avatar_enabled", "1", FCVAR_ARCHIVE, "1 = Skippy can be called into the world as an avatar")
local cv_kick = CreateConVar("mentat_avatar_kick", "0", FCVAR_ARCHIVE, "1 = going away kicks the bot (frees its slot); 0 = hidden but kept")
local cv_debug = CreateConVar("mentat_avatar_debug", "0", 0, "1 = print, per listener, how well each hears every spoken line")
local cv_ack = CreateConVar("mentat_avatar_ack_voice", "1", FCVAR_ARCHIVE, "1 = Skippy speaks his answer to 'come here' aloud")

local WALL_MARGIN = 24        -- HU kept between him and a wall
local NAME = "Skippy"

local function tov(t) return Vector(t.x, t.y, t.z) end
local function fromv(v) return { x = v.x, y = v.y, z = v.z } end
local function is_player(p) return p ~= nil and IsValid(p) and p:IsPlayer() end

--- The bot, when he is in the world and shown.
function Av.Get()
  if Av.visible and IsValid(Av.bot) then return Av.bot end
  return nil
end
function Av.IsActive() return Av.Get() ~= nil end

-- ------------------------------------------------------------------------------------------------ the body
local function configure(bot)
  if bot.StripWeapons then bot:StripWeapons() end
  if bot.GodEnable then bot:GodEnable() end
  if bot.SetMoveType then bot:SetMoveType(MOVETYPE_NOCLIP) end
  if bot.SetNotSolid then bot:SetNotSolid(true) end
  if bot.SetCollisionGroup then bot:SetCollisionGroup(COLLISION_GROUP_WEAPON or 11) end
  if bot.SetNoTarget then bot:SetNoTarget(true) end
end

local function publish()
  local bot = Av.Get()
  SetGlobalBool("mentat_avatar_on", bot ~= nil)
  if SetGlobalEntity then SetGlobalEntity("mentat_avatar", IsValid(Av.bot) and Av.bot or NULL) end
  if bot and is_player(Av.owner) then
    if bot.SetNWEntity then bot:SetNWEntity("mentat_avatar_owner", Av.owner) end
    local base = GetConVar("mentat_voice_range")
    base = base and base:GetFloat() or 0
    if base < 1 then base = Core.DEFAULTS.range end
    bot:SetNWFloat("mentat_avatar_range", Logic.voice_range(base, Av.owner:Crouching()))
  end
end
Av.Publish = publish

local function reset()
  Av.visible, Av.owner = false, nil
  publish()
end

--- A bot to wear: the one we made, or a new one. Returns nil and why when the server has no room.
local function ensure_bot()
  if IsValid(Av.bot) then return Av.bot end
  Av.bot = nil
  if game.MaxPlayers and #player.GetAll() >= game.MaxPlayers() then return nil, "the server has no free player slot for me" end
  local bot = player.CreateNextBot(NAME)
  if not IsValid(bot) then return nil, "the engine would not make me a body (is there a free player slot?)" end
  Av.bot = bot
  configure(bot)
  return bot
end

--- The owner's origin, eye height above it, and facing yaw, as plain tables.
local function frame(owner)
  local pos = owner:GetPos()
  return { pos = fromv(pos), eye = owner:GetShootPos().z - pos.z, yaw = owner:EyeAngles().y }
end

--- A point near `want` on the owner's side of any wall: a line from the owner's head to it must not hit the world; when it
--- does, stop WALL_MARGIN short of the wall.
local function safe_point(owner, want)
  local from = owner:GetShootPos()
  local to = tov(want)
  local tr = util.TraceLine({ start = from, endpos = to, mask = MASK_SOLID_BRUSHONLY })
  if not tr.Hit then return want, nil end
  local full = from:Distance(to)
  local d = math.max(0, full * tr.Fraction - WALL_MARGIN)
  local k = full > 0 and d / full or 0
  return { x = from.x + (to.x - from.x) * k, y = from.y + (to.y - from.y) * k, z = from.z + (to.z - from.z) * k }, full * tr.Fraction
end

--- Where to appear: beside the caller if that is clear, else the other side, else the best the room allows.
local function arrival_point(owner, fr)
  local best, best_d
  for _, want in ipairs(Logic.summon_points(fr.pos, fr.yaw, fr.eye)) do
    local p, reach = safe_point(owner, want)
    if not reach then return p end                                  -- nothing in the way
    if not best_d or reach > best_d then best, best_d = p, reach end
  end
  return best
end

local function place(bot, pos, yaw)
  bot:SetPos(tov(pos))
  local ang = Angle(0, yaw, 0)
  bot:SetEyeAngles(ang)
  if bot.SetAngles then bot:SetAngles(ang) end
end

-- ------------------------------------------------------------------------------------------------ calling and sending away
--- Bring Skippy to `ply`. Returns true, or false and a reason.
function Av.Summon(ply)
  if not cv_enabled:GetBool() then return false, "the avatar is switched off here" end
  if not is_player(ply) or not ply:Alive() then return false, "nobody to come to" end
  local bot, why = ensure_bot()
  if not bot then return false, why end
  configure(bot)
  BodyNet.ensure()                                              -- the real body's files (a failure only means the procedural can)
  local fr = frame(ply)
  local to = arrival_point(ply, fr)
  local from = Av.visible and fromv(bot:GetPos()) or nil
  if from then hook.Run("SkippyAvatarLeave", bot, tov(from)) end
  if bot.SetNoDraw then bot:SetNoDraw(false) end
  Av.owner, Av.visible = ply, true
  Av.st.pos, Av.st.last_owner, Av.st.t = to, fr.pos, 0
  Av.st.yaw = Logic.yaw_to(to, { x = fr.pos.x, y = fr.pos.y, z = fr.pos.z })
  place(bot, to, Av.st.yaw)
  publish()
  hook.Run("SkippyAvatarArrive", bot, from and tov(from) or nil, tov(to))
  return true
end

--- Send him away: hidden (default) or kicked (mentat_avatar_kick 1).
function Av.Dismiss()
  local bot = Av.Get()
  if not bot then
    reset()
    return false
  end
  hook.Run("SkippyAvatarLeave", bot, bot:GetPos())
  reset()
  if cv_kick:GetBool() then
    bot:Kick("Skippy has other places to be")
    Av.bot = nil
  elseif bot.SetNoDraw then
    bot:SetNoDraw(true)
  end
  return true
end

-- ------------------------------------------------------------------------------------------------ following
local last_tick, publish_in = nil, 0
local function tick()
  local now = CurTime()
  local dt = math.min(math.max(now - (last_tick or now), 0), 0.1)
  last_tick = now
  if not Av.visible then return end
  local bot, owner = Av.bot, Av.owner
  if not IsValid(bot) then reset(); return end
  if not is_player(owner) then Av.Dismiss(); return end
  if not owner:Alive() then return end                         -- he waits where he is until they are back
  local st = Av.st
  local fr = frame(owner)
  st.t = st.t + dt
  local want = Logic.follow_target(fr.pos, fr.yaw, fr.eye, st.t)
  local target = safe_point(owner, want)
  local before = fromv(bot:GetPos())
  local snapped = Logic.step(st, target, fr.pos, dt)
  -- face the person he is talking to; the rest of the time, the way they are looking
  local talking = (Av.speaking_until or 0) > now
  local goal = talking and Logic.yaw_to(st.pos, fr.pos) or fr.yaw
  st.yaw = Logic.turn(st.yaw, goal, Logic.DEFAULTS.turn_rate * dt)
  if snapped then hook.Run("SkippyAvatarLeave", bot, tov(before)) end
  place(bot, st.pos, st.yaw)
  -- the engine's own Spawn (after a kill) puts a walking body back; live pass: movetype was WALK after the revive
  if bot.GetMoveType and bot:GetMoveType() ~= MOVETYPE_NOCLIP then configure(bot) end
  if snapped then hook.Run("SkippyAvatarArrive", bot, tov(before), tov(st.pos)) end
  publish_in = publish_in - dt
  if publish_in <= 0 then publish_in = 0.25; publish() end
end
Av.Tick = tick
hook.Add("Tick", "Mentat.AvatarFollow", tick)

-- the engine asks every bot what it wants to do each tick: he wants nothing (his own AI would walk, jump and shoot)
hook.Add("StartCommand", "Mentat.AvatarNeutral", function(ply, cmd)
  if Av.bot ~= nil and ply == Av.bot then
    cmd:ClearButtons()
    cmd:ClearMovement()
    if cmd.SetImpulse then cmd:SetImpulse(0) end
    if cmd.SetViewAngles then cmd:SetViewAngles(Angle(0, Av.st.yaw, 0)) end
  end
end)

hook.Add("PlayerLoadout", "Mentat.AvatarNoWeapons", function(ply) if Av.bot ~= nil and ply == Av.bot then return true end end)
hook.Add("PlayerShouldTakeDamage", "Mentat.AvatarUnharmed", function(ply) if Av.bot ~= nil and ply == Av.bot then return false end end)
hook.Add("PlayerSpawn", "Mentat.AvatarRespawn", function(ply)
  if Av.bot ~= nil and ply == Av.bot then
    configure(ply)
    if Av.visible then place(ply, Av.st.pos, Av.st.yaw)
    elseif ply.SetNoDraw then ply:SetNoDraw(true) end
  end
end)
-- God mode does not stop an explicit kill (the console `kill`, a map's kill trigger, Player:Kill): the live pass found the bot then stays
-- dead forever (a bot never respawns by itself), still pinned to the owner as a corpse. He is put back on his feet at once, and
-- the default death makes no ragdoll of him.
hook.Add("DoPlayerDeath", "Mentat.AvatarNoRagdoll", function(ply)
  if Av.bot ~= nil and ply == Av.bot then return true end
end)
hook.Add("PlayerDeath", "Mentat.AvatarRevive", function(ply)
  if Av.bot ~= nil and ply == Av.bot then
    timer.Simple(0.1, function()
      if IsValid(ply) and Av.bot == ply and not ply:Alive() then ply:Spawn() end
    end)
  end
end)
hook.Add("PlayerDisconnected", "Mentat.AvatarGone", function(ply)
  if Av.bot ~= nil and ply == Av.bot then Av.bot = nil; reset()
  elseif ply == Av.owner then Av.Dismiss() end
end)

-- ------------------------------------------------------------------------------------------------ pointing
-- Hook SkippyAvatarPoint(pos, seconds): the right hand of the real body swings to aim at `pos` (a Vector) for `seconds` (default 4);
-- pos nil lowers it. The clients draw it (avatar_client.lua); this just carries the order to them. Console: skippy_point = at what I look at.
util.AddNetworkString("mentat_avatar_point")
hook.Add("SkippyAvatarPoint", "Mentat.AvatarPointNet", function(pos, seconds)
  net.Start("mentat_avatar_point")
  net.WriteBool(pos ~= nil)
  if pos ~= nil then net.WriteVector(pos) end
  net.WriteUInt(math.Clamp(math.floor((seconds or 4) * 10 + 0.5), 1, 600), 16)
  net.Broadcast()
end)
function Av.Point(pos, seconds) hook.Run("SkippyAvatarPoint", pos, seconds) end
concommand.Add("skippy_point", function(ply)
  if not is_player(ply) or ply ~= Av.owner then return end
  local tr = util.TraceLine({ start = ply:GetShootPos(), endpos = ply:GetShootPos() + ply:GetAimVector() * 4000, mask = MASK_SOLID })
  Av.Point(tr.HitPos, 4)
end, nil, "Skippy points at what you are looking at")

-- ------------------------------------------------------------------------------------------------ who hears him
--- How well one listener hears Skippy where he hovers now: the same rule as proximity voice (distance, walls, whisper).
function Av.Hearing(listener)
  local bot = Av.Get()
  if not bot then return nil end
  local at = bot:GetPos()
  local eye = listener:GetShootPos()
  local dist = eye:Distance(at)
  local range = bot:GetNWFloat("mentat_avatar_range", Core.DEFAULTS.range)
  local blocked = util.TraceLine({ start = eye, endpos = at, mask = MASK_SOLID_BRUSHONLY }).Hit
  local heard, gain = Logic.hear(dist, range, blocked, GetGlobalFloat("mentat_voice_wallf", Core.DEFAULTS.wall), nil,
    GetGlobalString("mentat_voice_mode", "proximity"))
  return { ply = listener, nick = listener:Nick(), dist = dist, range = range, blocked = blocked, gain = gain, heard = heard }
end

--- Every player's hearing of him (an array), or just `ply`'s when given. Empty when he is not in the world.
function Av.WhoHears(ply)
  local out = {}
  if not Av.IsActive() then return out end
  for _, p in ipairs(player.GetAll()) do
    if p ~= Av.bot and (ply == nil or p == ply) then out[#out + 1] = Av.Hearing(p) end
  end
  return out
end

local function debug_lines(text)
  Av.last_heard = Av.WhoHears()
  if not cv_debug:GetBool() then return end
  for _, h in ipairs(Av.last_heard) do
    print(string.format("[avatar] \"%s\" -> %s dist=%.0f range=%.0f blocked=%s gain=%.2f%s", tostring(text):sub(1, 40), h.nick, h.dist, h.range,
      tostring(h.blocked), h.gain, h.heard and "" or " (cannot hear)"))
  end
end

-- ------------------------------------------------------------------------------------------------ words
local last_cmd = setmetatable({}, { __mode = "k" })

--- Act on a recognised command from a player ("summon" | "dismiss"). Answers in his own voice.
function Av.Command(ply, cmd)
  if CurTime() - (last_cmd[ply] or -10) < 1 then return end              -- one command a second per player
  last_cmd[ply] = CurTime()
  local H = Holodeck
  local function say(text, speak)
    if H and H.Say then H.Say(ply, text, nil, { speak = speak }) end
  end
  local admin_only = GetConVar("holodeck_admin_only")
  if admin_only and admin_only:GetBool() and not ply:IsAdmin() then
    if H and H.Tell then H.Tell(ply, "The holodeck is for admins on this server.") end
    return
  end
  if cmd == "summon" then
    local ok, why = Av.Summon(ply)
    if ok then say("Here I am.", cv_ack:GetBool()) elseif H and H.Tell then H.Tell(ply, why or "I cannot come right now.") end
  elseif cmd == "dismiss" then
    if Av.IsActive() then
      say("Fine. Call me if you need me.", cv_ack:GetBool())
      Av.Dismiss()
    end
  end
end

-- The holodeck is the one place words reach Skippy (chat after the wake word, the microphone bridge, the console), and all of them
-- end in Holodeck.Handle(ply, text). This puts the avatar's two commands in front of it, and moves where his voice and his answers
-- come from while he is in the world. Nothing in holodeck.lua is changed: the functions are wrapped, and wrapping again after a
-- Lua refresh of either file does not stack.
local function wrap(H, name, make)
  local cur = H[name]
  if type(cur) ~= "function" then return end
  if not Av.wrapped[cur] then
    H.AvatarOrig = H.AvatarOrig or {}
    H.AvatarOrig[name] = cur
    local fn = make(function(...) return H.AvatarOrig[name](...) end)
    Av.wrapped[fn] = true
    H[name] = fn
  end
end

local function install()
  local H = Holodeck
  if not H then return false end
  wrap(H, "Handle", function(orig)
    return function(ply, text, opts)
      if is_player(ply) and cv_enabled:GetBool() then
        local cmd = Logic.match_command(text)
        if cmd then Av.Command(ply, cmd); return nil end
      end
      return orig(ply, text, opts)
    end
  end)
  wrap(H, "Say", function(orig)
    return function(ply, text, pos, opts)
      local bot = Av.Get()
      if bot then pos = bot:GetPos() end                 -- his answers come from where he hovers
      return orig(ply, text, pos, opts)
    end
  end)
  wrap(H, "Speak", function(orig)
    return function(text, pos, ply)
      local bot = Av.Get()
      if bot then
        pos = bot:GetPos()
        Av.speaking_until = CurTime() + #tostring(text) * 0.07 + 1
        debug_lines(text)
      end
      return orig(text, pos, ply)
    end
  end)
  return true
end
Av.Install = install
install()
timer.Simple(0, install)
timer.Create("Mentat.AvatarWrap", 2, 0, install)           -- a later Lua refresh of holodeck.lua brings back the plain functions

concommand.Add("skippy_come", function(ply) if is_player(ply) then Av.Command(ply, "summon") end end, nil, "Call Skippy to you")
concommand.Add("skippy_go", function(ply) if is_player(ply) then Av.Command(ply, "dismiss") end end, nil, "Send Skippy away")
concommand.Add("mentat_avatar_status", function()
  local bot = Av.Get()
  print("[avatar] " .. (bot and ("in the world at " .. tostring(bot:GetPos()) .. ", following " .. (is_player(Av.owner) and Av.owner:Nick() or "nobody")) or "away")
    .. "; bot " .. (IsValid(Av.bot) and "exists" or "not created") .. "; enabled=" .. tostring(cv_enabled:GetBool()))
  for _, h in ipairs(Av.WhoHears()) do
    print(string.format("[avatar]   %s dist=%.0f range=%.0f blocked=%s gain=%.2f", h.nick, h.dist, h.range, tostring(h.blocked), h.gain))
  end
end, nil, "Where Skippy is and who hears him")
