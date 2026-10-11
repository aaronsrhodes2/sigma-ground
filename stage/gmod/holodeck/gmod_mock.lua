-- gmod_mock.lua -- a small stand-in for the Garry's Mod API, enough to run the REAL server-side addon files (holodeck.lua,
-- voice.lua) and the REAL client files (skippy_client.lua, voice_client.lua, holodeck_client.lua) offline under LuaJIT.
-- Driven from sigma_ground/radiance/tests/test_gmod_glue.py. Nothing here knows the planner: the world is a flat floor at z = 0
-- (optionally enclosed by a box room), with blockers you place, and everything the engine would do over time (timers, HTTP
-- replies, network messages) is recorded or advanced by hand through the MOCK table.
--
--   MOCK.advance(dt)             run due timers (a virtual clock; CurTime() and SysTime() read it)
--   MOCK.newplayer(sid, {pos=, yaw=, admin=, name=, host=})   MOCK.disconnect(p)
--   MOCK.respond(i, code, body, headers)   answer the i-th recorded HTTP request (1-based)
--   MOCK.nets                    every net message sent: { name, writes, to = {players}, bytes, kind }
--   MOCK.room = {x0, y0, x1, y1} walls around the floor (nil = open ground);  MOCK.blockers = { {x0,y0,x1,y1, ent=} }
MOCK = { now = 0, timers = {}, nets = {}, https = {}, cmds = {}, receivers = {}, hooks = {}, printed = {}, ents = {}, players = {},
         cvars = {}, blockers = {}, files = {}, sounds = {}, hull_calls = 0 }
local M = MOCK

SERVER = SERVER == nil and true or SERVER
CLIENT = not SERVER
FCVAR_ARCHIVE, FCVAR_PROTECTED, FCVAR_DONTRECORD, FCVAR_NOTIFY = 1, 2, 4, 8
RENDERMODE_TRANSALPHA, RENDERMODE_NORMAL = 4, 0
MASK_SOLID_BRUSHONLY, MASK_SOLID = 16395, 33570827
MOVETYPE_NOCLIP, MOVETYPE_WALK = 8, 2
COLLISION_GROUP_NONE, COLLISION_GROUP_PASSABLE_DOOR = 0, 15
GMOD_CHANNEL_STOPPED, GMOD_CHANNEL_PLAYING = 0, 1
TEXT_FILTER_CHAT = 1

-- ------------------------------------------------------------------------------------------------ vectors and colours
local Vec = {}
Vec.__index = Vec
function Vector(x, y, z) return setmetatable({ x = x or 0, y = y or 0, z = z or 0 }, Vec) end
function Vec:Distance(o) return math.sqrt((self.x - o.x) ^ 2 + (self.y - o.y) ^ 2 + (self.z - o.z) ^ 2) end
function Vec:DistToSqr(o) return (self.x - o.x) ^ 2 + (self.y - o.y) ^ 2 + (self.z - o.z) ^ 2 end
function Vec:Length() return math.sqrt(self.x ^ 2 + self.y ^ 2 + self.z ^ 2) end
function Vec:LengthSqr() return self.x ^ 2 + self.y ^ 2 + self.z ^ 2 end
function Vec:Normalize() local l = self:Length(); if l > 0 then self.x, self.y, self.z = self.x / l, self.y / l, self.z / l end end
Vec.__add = function(a, b) return Vector(a.x + b.x, a.y + b.y, a.z + b.z) end
Vec.__sub = function(a, b) return Vector(a.x - b.x, a.y - b.y, a.z - b.z) end
Vec.__mul = function(a, b)
  if type(a) == "number" then a, b = b, a end
  return Vector(a.x * b, a.y * b, a.z * b)
end
local Ang = {}
Ang.__index = function(t, k)
  if k == "pitch" then return rawget(t, "p") elseif k == "yaw" then return rawget(t, "y") elseif k == "roll" then return rawget(t, "r") end
  return rawget(Ang, k)
end
function Angle(p, y, r) return setmetatable({ p = p or 0, y = y or 0, r = r or 0 }, Ang) end
function Color(r, g, b, a) return { r = r, g = g, b = b, a = a or 255 } end
vector_origin = Vector(0, 0, 0)

function math.Clamp(v, lo, hi) return math.max(lo, math.min(hi, v)) end
function math.NormalizeAngle(a) a = (a + 180) % 360 - 180; return a end
function math.AngleDifference(a, b) return math.NormalizeAngle(a - b) end

function IsValid(o)
  if o == nil or type(o) ~= "table" then return false end
  if o.IsValid then return o:IsValid() end
  return false
end
function MsgC(...) end
function ErrorNoHalt(...) M.printed[#M.printed + 1] = "ERR " .. table.concat({ ... }, "") end
function print(...)
  local t = {}
  for i = 1, select("#", ...) do t[i] = tostring((select(i, ...))) end
  M.printed[#M.printed + 1] = table.concat(t, "\t")
end
function CurTime() return M.now end
function SysTime() return M.now end
function table.Count(t) local n = 0; for _ in pairs(t) do n = n + 1 end; return n end
function table.Empty(t) for k in pairs(t) do t[k] = nil end end

function istable(v) return type(v) == "table" end
function isstring(v) return type(v) == "string" end
function isnumber(v) return type(v) == "number" end
function isfunction(v) return type(v) == "function" end
function isbool(v) return type(v) == "boolean" end

function include(path) return dofile(M.ROOT .. "/" .. path) end
function AddCSLuaFile() end

-- ------------------------------------------------------------------------------------------------ convars, hooks, timers
function CreateConVar(name, default, flags, help)
  if M.cvars[name] then return M.cvars[name] end
  local cv = { name = name, val = tostring(default), flags = flags or 0, help = help or "" }
  function cv:GetString() return self.val end
  function cv:GetInt() return math.floor(tonumber(self.val) or 0) end
  function cv:GetFloat() return tonumber(self.val) or 0 end
  function cv:GetBool() return (tonumber(self.val) or 0) ~= 0 end
  function cv:SetValue(v)
    local old = self.val
    self.val = tostring(v)
    for _, cb in ipairs(M.cvar_callbacks[name] or {}) do cb(name, old, self.val) end
  end
  M.cvars[name] = cv
  return cv
end
CreateClientConVar = CreateConVar
function GetConVar(name) return M.cvars[name] or CreateConVar(name, "0") end
M.cvar_callbacks = {}
cvars = { AddChangeCallback = function(name, fn) M.cvar_callbacks[name] = M.cvar_callbacks[name] or {}; table.insert(M.cvar_callbacks[name], fn) end }

hook = {}
function hook.Add(ev, name, fn) M.hooks[ev] = M.hooks[ev] or {}; M.hooks[ev][name] = fn end
function hook.Remove(ev, name) if M.hooks[ev] then M.hooks[ev][name] = nil end end
--- hook.Call(name, gamemode, ...): every registered hook first (the first non-nil answer wins), then the gamemode's function.
function hook.Call(ev, gm, ...)
  local names = {}
  for name in pairs(M.hooks[ev] or {}) do names[#names + 1] = name end
  table.sort(names)                                   -- the engine's order is undefined; tests choose theirs with M.hook_order
  if M.hook_order and M.hook_order[ev] then names = M.hook_order[ev] end
  for _, name in ipairs(names) do
    local fn = M.hooks[ev] and M.hooks[ev][name]
    if fn then local a, b = fn(...); if a ~= nil then return a, b end end
  end
  if gm and gm[ev] then return gm[ev](gm, ...) end
end
function hook.Run(ev, ...) return hook.Call(ev, GAMEMODE, ...) end

timer = {}
function timer.Simple(delay, fn) M.timers[#M.timers + 1] = { at = M.now + delay, fn = fn } end
function timer.Create(name, delay, reps, fn)
  timer.Remove(name)
  M.timers[#M.timers + 1] = { at = M.now + delay, fn = fn, name = name, every = delay, reps = reps }
end
function timer.Exists(name) for _, t in ipairs(M.timers) do if t.name == name then return true end end return false end
function timer.Remove(name)
  for i = #M.timers, 1, -1 do if M.timers[i].name == name then table.remove(M.timers, i) end end
end
function M.advance(dt)
  local target = M.now + dt
  while true do
    local best, bi
    for i, t in ipairs(M.timers) do if t.at <= target and (not best or t.at < best.at) then best, bi = t, i end end
    if not best then break end
    M.now = math.max(M.now, best.at)
    if best.every then
      if best.reps == 0 then best.at = best.at + best.every                         -- endless
      else
        best.reps = best.reps - 1
        if best.reps <= 0 then table.remove(M.timers, bi) else best.at = best.at + best.every end
      end
    else
      table.remove(M.timers, bi)
    end
    best.fn()
  end
  M.now = target
end

-- ------------------------------------------------------------------------------------------------ util, net, http
util = util or {}
function util.IsValidModel() return true end
function util.AddNetworkString() end
function util.TableToJSON(t) return M.json_encode(t) end
function util.JSONToTable(s) return M.json_decode(s) end
function util.FilterText(s) return M.filter and M.filter(s) or s end

function M.json_encode(v)
  local ty = type(v)
  if ty == "string" then return '"' .. v:gsub('[%c"\\]', function(c) return string.format("\\u%04x", c:byte()) end) .. '"' end
  if ty == "number" or ty == "boolean" then return tostring(v) end
  if ty == "table" then
    if #v > 0 or next(v) == nil then
      local o = {}
      for i, x in ipairs(v) do o[i] = M.json_encode(x) end
      return "[" .. table.concat(o, ",") .. "]"
    end
    local o = {}
    for k, x in pairs(v) do o[#o + 1] = M.json_encode(tostring(k)) .. ":" .. M.json_encode(x) end
    return "{" .. table.concat(o, ",") .. "}"
  end
  return "null"
end

net = {}
function net.Start(name) M.cur = { name = name, writes = {}, bytes = 0 } end
function net.WriteString(s)
  if type(s) ~= "string" then error("net.WriteString: not a string: " .. type(s)) end
  M.cur.writes[#M.cur.writes + 1] = s
end
function net.WriteUInt(v, bits) M.cur.writes[#M.cur.writes + 1] = v end
function net.WriteBool(b) M.cur.writes[#M.cur.writes + 1] = b end
function net.WriteVector(v) M.cur.vector = v end
function net.WriteData(d, n) M.cur.bytes = M.cur.bytes + n; M.cur.data = d end
function net.Broadcast() M.cur.kind = "broadcast"; M.cur.to = player.GetAll(); M.nets[#M.nets + 1] = M.cur end
function net.Send(t)
  M.cur.kind = "send"
  M.cur.to = t.IsValid and { t } or t
  M.nets[#M.nets + 1] = M.cur
end
function net.SendToServer() M.cur.kind = "server"; M.nets[#M.nets + 1] = M.cur end
function net.Receive(name, fn) M.receivers[name] = fn end
function net.ReadString() return M.rd and M.rd.string or "" end
function net.ReadUInt(bits) local v = table.remove(M.rd.uints, 1); return v end
function net.ReadVector() return M.rd.vector or Vector(0, 0, 0) end
function net.ReadData(n) return M.rd.data or "" end
function net.ReadBool() return M.rd.bool end
--- Deliver a recorded message to a client-side receiver (the client files read what the server wrote).
function M.deliver(msg)
  local w = msg.writes
  if msg.name == "holodeck_audio" then M.rd = { uints = { w[1], w[2], w[3], #(msg.data or "") }, vector = msg.vector, data = msg.data }
  elseif msg.name == "holodeck_say" then
    local strings, uints = {}, {}
    for _, x in ipairs(w) do
      if type(x) == "string" then strings[#strings + 1] = x elseif type(x) == "number" then uints[#uints + 1] = x end
    end
    M.rd = { string = strings[1], strings = strings, uints = uints }
    local n = 0
    net.ReadString = function() n = n + 1; return M.rd.strings[n] end
  end
  M.receivers[msg.name]()
end

player = {}
function player.GetAll() local o = {}; for _, p in ipairs(M.players) do if p.valid then o[#o + 1] = p end end; return o end
concommand = { Add = function(name, fn) M.cmds[name] = fn end }
game = { SinglePlayer = function() return false end }
engine = { ActiveGamemode = function() return "sandbox" end }

function HTTP(t) M.https[#M.https + 1] = t; return M.http_refuse and false or true end
function M.respond(i, code, body, headers)
  local t = M.https[i]
  if code == "fail" then t.failed(body or "timeout") else t.success(code, body or "", headers or {}) end
end

-- ------------------------------------------------------------------------------------------------ the world
local function inside_room(x, y)
  local r = M.room
  return not r or (x > r.x0 and x < r.x1 and y > r.y0 and y < r.y1)
end

--- A vertical ray against the PLATFORMS: blockers that have a `top` (and a `bottom`, default 10 below it) are slabs that stand up
--- in z. Only a trace with a solid mask (not brush-only) sees them, and only those the trace's filter accepts, as in the engine.
--- Returns the nearest hit { z =, normal_z = } or nil, and whether the ray starts inside a slab.
local function platform_hit(t, sx, sy, sz, ez)
  if t.mask ~= MASK_SOLID then return nil end
  local best
  for _, b in ipairs(M.blockers) do
    if b.top and sx > b.x0 and sx < b.x1 and sy > b.y0 and sy < b.y1 and (not t.filter or not b.ent or t.filter(b.ent)) then
      local bottom = b.bottom or (b.top - 10)
      if sz > bottom and sz < b.top then return { start_solid = true } end
      if ez < sz and sz >= b.top and ez <= b.top then
        if not best or b.top > best.z then best = { z = b.top, nz = 1 } end
      elseif ez > sz and sz <= bottom and ez >= bottom then
        if not best or bottom < best.z then best = { z = bottom, nz = -1 } end
      end
    end
  end
  return best
end

function util.TraceLine(t)
  local sx, sy, sz, ex, ey, ez = t.start.x, t.start.y, t.start.z, t.endpos.x, t.endpos.y, t.endpos.z
  if math.abs(ex - sx) < 1e-6 and math.abs(ey - sy) < 1e-6 then                       -- straight down: the floor at z = 0
    local plat = platform_hit(t, sx, sy, sz, ez)
    if plat and plat.start_solid then return { Hit = true, StartSolid = true, HitSky = false, HitNormal = Vector(0, 0, 1), HitPos = t.start, Fraction = 0 } end
    local world
    if ez > sz and M.ceiling and sz < M.ceiling and ez >= M.ceiling and inside_room(sx, sy) then       -- straight up: a ceiling, if the room has one
      world = { Hit = true, StartSolid = false, HitSky = false, HitNormal = Vector(0, 0, -1), HitPos = Vector(sx, sy, M.ceiling), Fraction = (M.ceiling - sz) / (ez - sz), HitWorld = true }
    elseif sz > 0 and ez < 0 and inside_room(sx, sy) then
      world = { Hit = true, StartSolid = false, HitSky = false, HitNormal = Vector(0, 0, 1), HitPos = Vector(sx, sy, 0), Fraction = sz / (sz - ez), HitWorld = true }
    end
    if plat and (not world or math.abs(plat.z - sz) < math.abs(world.HitPos.z - sz)) then
      return { Hit = true, StartSolid = false, HitSky = false, HitNormal = Vector(0, 0, plat.nz), HitPos = Vector(sx, sy, plat.z), Fraction = (plat.z - sz) / (ez - sz) }
    end
    if world then return world end
    return { Hit = false, StartSolid = false, HitSky = false, HitNormal = Vector(0, 0, 1), HitPos = t.endpos, Fraction = 1 }
  end
  local r = M.room
  if r and inside_room(sx, sy) then                                                   -- a horizontal ray inside a box room hits a wall
    local best, nx, ny = 2, 0, 0
    local function try(plane, from, dir, normal_x, normal_y)
      if math.abs(dir) < 1e-9 then return end
      local f = (plane - from) / dir
      if f > 0 and f < best then best, nx, ny = f, normal_x, normal_y end
    end
    try(r.x0, sx, ex - sx, 1, 0); try(r.x1, sx, ex - sx, -1, 0); try(r.y0, sy, ey - sy, 0, 1); try(r.y1, sy, ey - sy, 0, -1)
    if best <= 1 then
      return { Hit = true, StartSolid = false, HitSky = false, HitNormal = Vector(nx, ny, 0), HitPos = Vector(sx + (ex - sx) * best, sy + (ey - sy) * best, sz), Fraction = best, HitWorld = true }
    end
  end
  return { Hit = false, StartSolid = false, HitSky = false, HitNormal = Vector(0, 0, 1), HitPos = t.endpos, Fraction = 1 }
end

--- A hull hits the room's walls or any blocker box (xy overlap) whose entity passes the trace's filter.
function util.TraceHull(t)
  M.hull_calls = M.hull_calls + 1
  M.last_hull = t
  local x0, x1 = t.start.x + t.mins.x, t.start.x + t.maxs.x
  local y0, y1 = t.start.y + t.mins.y, t.start.y + t.maxs.y
  local r = M.room
  if r and (x0 <= r.x0 or x1 >= r.x1 or y0 <= r.y0 or y1 >= r.y1) then return { Hit = true, StartSolid = false } end
  local z0, z1 = t.start.z + t.mins.z, t.start.z + t.maxs.z
  for _, b in ipairs(M.blockers) do
    -- a platform (a blocker with a top) is only in the way of a hull that reaches into its slab
    local in_z = not b.top or (z1 > (b.bottom or (b.top - 10)) and z0 < b.top)
    if in_z and x0 < b.x1 and x1 > b.x0 and y0 < b.y1 and y1 > b.y0 then
      local hits = true
      if t.filter and b.ent then hits = t.filter(b.ent) and true or false end
      if t.mask == MASK_SOLID_BRUSHONLY and b.ent then hits = false end                  -- the old mask never saw entities
      if hits then return { Hit = true, StartSolid = false } end
    end
  end
  return { Hit = false, StartSolid = false }
end

-- ------------------------------------------------------------------------------------------------ entities and players
local Ent = {}
Ent.__index = Ent
local next_index = 100
function Ent:IsValid() return self.valid end
function Ent:IsPlayer() return false end
function Ent:EntIndex() return self.idx end
function Ent:GetClass() return self.class or "" end
function Ent:SetModel(m) self.model = m end
function Ent:SetPos(v) self.pos = v end
function Ent:GetPos() return self.pos end
function Ent:SetAngles(a) self.ang = a end
function Ent:GetAngles() return self.ang end
function Ent:Spawn() end
function Ent:Activate() end
function Ent:GetPhysicsObject() return self.phys end
function Ent:SetRenderMode() end
function Ent:SetColor() end
function Ent:SetCollisionGroup(g) self.group = g end
function Ent:SetCreator(p) self.creator = p end
function Ent:CPPISetOwner(p) self.cppi = p end       -- a prop-protection addon's owner record
function Ent:WorldSpaceAABB()
  local h = self.half or 20
  return Vector(self.pos.x - h, self.pos.y - h, self.pos.z), Vector(self.pos.x + h, self.pos.y + h, self.pos.z + 2 * h)
end
function Ent:Remove()
  if not self.valid then return end
  self.valid = false
  hook.Run("EntityRemoved", self)
end
ents = {}
function ents.Create(class)
  next_index = next_index + 1
  local e = setmetatable({ valid = true, idx = next_index, class = class, pos = Vector(0, 0, 0), ang = Angle(0, 0, 0), group = 0 }, Ent)
  e.phys = { mass = nil, motion = true, IsValid = function() return true end, EnableMotion = function(s, b) e.phys.motion = b end,
             Wake = function() end, Sleep = function() end, SetMass = function(s, m) e.phys.mass = m end,
             GetVelocity = function() return Vector(0, 0, 0) end, SetVelocity = function() end }
  M.ents[#M.ents + 1] = e
  return e
end
function ents.FindInBox(mins, maxs)
  local out = {}
  for _, p in ipairs(M.players) do
    local q = p.pos
    if p.valid and q.x >= mins.x and q.x <= maxs.x and q.y >= mins.y and q.y <= maxs.y then out[#out + 1] = p end
  end
  return out
end
function M.live_props()
  local n = 0
  for _, e in ipairs(M.ents) do if e.valid and e.class == "prop_physics" then n = n + 1 end end
  return n
end

local Ply = {}
Ply.__index = Ply
function Ply:IsValid() return self.valid end
function Ply:IsPlayer() return true end
function Ply:SteamID64() return self.sid end
function Ply:Nick() return self.name end
function Ply:IsAdmin() return self.admin end
function Ply:IsSuperAdmin() return self.superadmin end
function Ply:IsListenServerHost() return self.host == true end
function Ply:Alive() return self.alive ~= false end
function Ply:GetMoveType() return self.movetype or MOVETYPE_WALK end
function Ply:GetPos() return self.pos end
function Ply:EyeAngles() return Angle(0, self.yaw, 0) end
function Ply:GetEyeTrace() return { Hit = false } end
function Ply:Crouching() return self.crouch == true end
function Ply:GetShootPos() return Vector(self.pos.x, self.pos.y, self.pos.z + 64) end
function Ply:EntIndex() return self.idx end
function Ply:UserID() return self.userid or self.idx end
function Ply:IsSpeaking() return self.speaking == true end
function Ply:IsTyping() return self.typing == true end
function Ply:SetNWFloat(k, v) self.nw = self.nw or {}; self.nw[k] = v end
function Ply:GetNWFloat(k, d) return self.nw and self.nw[k] or d end
function Ply:SetVoiceVolumeScale(s) self.vscale = s end
function Ply:EyePos() return self:GetShootPos() end
function M.newplayer(sid, o)
  o = o or {}
  next_index = next_index + 1
  local p = setmetatable({ valid = true, sid = sid, name = o.name or ("P" .. sid), idx = next_index,
                           pos = o.pos or Vector(0, 0, 0), yaw = o.yaw or 0, admin = o.admin or false, host = o.host }, Ply)
  M.players[#M.players + 1] = p
  return p
end
function M.disconnect(p)
  hook.Run("PlayerDisconnected", p)
  p.valid = false
end
function M.reset_traffic() M.nets, M.https, M.printed = {}, {}, {} end

-- global state used by the voice files
local globals = {}
function SetGlobalString(k, v) globals[k] = v end
function SetGlobalFloat(k, v) globals[k] = v end
function SetGlobalBool(k, v) globals[k] = v end
function GetGlobalString(k, d) if globals[k] ~= nil then return globals[k] end return d end
function GetGlobalFloat(k, d) if globals[k] ~= nil then return globals[k] end return d end
function GetGlobalBool(k, d) if globals[k] ~= nil then return globals[k] end return d end
function LocalPlayer() return M.me end

-- client side: files, sounds
file = {}
function file.CreateDir() end
function file.Write(path, data) M.files[path] = data; M.write_count = M.write_count or {}; M.write_count[path] = (M.write_count[path] or 0) + 1 end
function file.Read(path) return M.files[path] end
function file.Exists(path) return M.files[path] ~= nil end
function file.Delete(path) M.files[path] = nil end
function file.Find(pattern)
  local dir = pattern:match("^(.*)/") or ""
  local prefix = pattern:match("^(.-)%*") or pattern
  local out = {}
  for k in pairs(M.files) do
    if k:sub(1, #prefix) == prefix then out[#out + 1] = k:sub(#dir + 2) end
  end
  return out
end
chat = { AddText = function(...) M.chat = M.chat or {}; local t = {}; for _, a in ipairs({ ... }) do if type(a) == "string" then t[#t + 1] = a end end; M.chat[#M.chat + 1] = table.concat(t, "") end }
input = { keys_down = {} }
function input.GetKeyCode(name) return ({ x = 34, v = 31 })[name] or 0 end
function input.IsKeyDown(code) return input.keys_down[code] == true end
vgui = { CursorVisible = function() return M.cursor_visible == true end }
gui = { IsGameUIVisible = function() return false end, IsConsoleVisible = function() return false end }
sound = {}
function sound.PlayFile(path, flags, cb)
  local chan = { valid = true, state = GMOD_CHANNEL_STOPPED, path = path, len = M.sound_length or 4 }
  function chan:IsValid() return self.valid end
  function chan:SetPos() end
  function chan:Set3DFadeDistance() end
  function chan:SetVolume() end
  function chan:Play() self.state = GMOD_CHANNEL_PLAYING end
  function chan:Stop() self.state = GMOD_CHANNEL_STOPPED; self.valid = false end
  function chan:GetState() return self.state end
  function chan:GetLength() return self.len end
  M.sounds[#M.sounds + 1] = chan
  cb(chan, 0, "")
end
