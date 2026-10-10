-- holodeck.lua -- the holodeck inside Garry's Mod (server side).
--
-- A player says "Skippy, add three couches" (in chat, or by voice through the bridge) or types `holodeck add three couches`
-- in the console. The parser turns the words into a request, the planner turns the request into poses that respect the
-- room's walls and floor and what already stands, and this file spawns real models (a catalog measured in the engine) with a
-- short materialising fade. Everything it makes is registered, so "remove the couch", "move it closer", "turn the chair around",
-- "undo" and "clear" work on exactly those things.
--
-- The parser and planner are pure Lua and tested offline (stage/gmod/holodeck/); this file is only the engine glue:
-- traces for floors and walls, entity spawning, chat, undo.
if not SERVER then return end

Holodeck = Holodeck or {}
local H = Holodeck

local Parse = include("mentat/holodeck/parse.lua")
local Plan = include("mentat/holodeck/plan.lua")
local Layout = include("mentat/holodeck/layout.lua")
local catalog = include("mentat/holodeck/catalog.lua")

H.catalog, H.parse_mod, H.plan_mod, H.layout = catalog, Parse, Plan, Layout
H.parser = Parse.new(catalog)
H.planner = Plan.new(catalog)
H.registry = {}            -- id -> entry
H.order = {}               -- ids in creation order
H.next_id, H.seq, H.group = 1, 0, 0
H.undo_stacks = {}         -- player key -> list of operations
H.pending = 0              -- spawns still scheduled
H.log = H.log or {}

util.AddNetworkString("holodeck_say")

local cv_max = CreateConVar("holodeck_max_ents", "400", FCVAR_ARCHIVE, "Most props the holodeck will keep standing at once")
local cv_admin = CreateConVar("holodeck_admin_only", "0", FCVAR_ARCHIVE, "1 = only admins may use the holodeck")
local cv_stagger = CreateConVar("holodeck_stagger", "0.05", FCVAR_ARCHIVE, "Seconds between one prop appearing and the next")
local cv_wake = CreateConVar("holodeck_need_wake", "1", FCVAR_ARCHIVE, "1 = chat and voice need the wake word (Skippy); the console never does")

-- ------------------------------------------------------------------------------------------------ speaking
function H.Say(ply, text)
  if not text or text == "" then return end
  H.last_say = text
  H.log[#H.log + 1] = text
  if #H.log > 200 then table.remove(H.log, 1) end
  print("[holodeck] " .. text)
  net.Start("holodeck_say")
  net.WriteString(text)
  net.Broadcast()
  hook.Run("HolodeckSaid", ply, text)
end

-- ------------------------------------------------------------------------------------------------ the world, as the planner asks about it
local MASK = MASK_SOLID_BRUSHONLY

local function make_env(origin_z)
  local env = {}
  function env.floor_z(x, y)
    local tr = util.TraceLine({ start = Vector(x, y, origin_z + 48), endpos = Vector(x, y, origin_z - 700), mask = MASK })
    if tr.StartSolid or not tr.Hit or tr.HitNormal.z < 0.7 or tr.HitSky then return nil end
    return tr.HitPos.z
  end
  function env.clear(r, z, h)
    -- axis-aligned box around the turned footprint, from just above the floor to the item's height (capped at a person's)
    local hx = r.d / 2 * math.abs(r.fx) + r.w / 2 * math.abs(r.rx)
    local hy = r.d / 2 * math.abs(r.fy) + r.w / 2 * math.abs(r.ry)
    local top = math.min(h or 40, 64)
    local tr = util.TraceHull({
      start = Vector(r.x, r.y, z + 3), endpos = Vector(r.x, r.y, z + 3),
      mins = Vector(-hx + 0.5, -hy + 0.5, 0), maxs = Vector(hx - 0.5, hy - 0.5, math.max(top - 6, 2)), mask = MASK,
    })
    return not tr.Hit and not tr.StartSolid
  end
  function env.wall_probe(x, y, dx, dy, maxd)
    local fz = env.floor_z(x, y) or origin_z
    local a = Vector(x, y, fz + 30)
    local tr = util.TraceLine({ start = a, endpos = a + Vector(dx, dy, 0) * maxd, mask = MASK })
    if not tr.Hit or tr.StartSolid or tr.HitSky then return nil end
    if math.abs(tr.HitNormal.z) > 0.5 then return nil end                 -- a slope or a ledge, not a wall
    local n = Vector(tr.HitNormal.x, tr.HitNormal.y, 0)
    n:Normalize()
    return { dist = tr.Fraction * maxd, nx = n.x, ny = n.y }
  end
  return env
end

local function registry_list()
  local out = {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e then
      out[#out + 1] = { id = e.id, kind = e.kind, vi = e.vi, x = e.x, y = e.y, z = e.z, theta = e.theta, w = e.w, d = e.d, h = e.h,
                        color_name = e.color_name, group = e.group, seq = e.seq }
    end
  end
  return out
end

local function context(ply, opts)
  opts = opts or {}
  local pos, yaw
  if IsValid(ply) then
    pos = ply:GetPos()
    yaw = ply:EyeAngles().yaw
  else
    local p = opts.pos or { x = 0, y = 0, z = 0 }
    pos, yaw = Vector(p.x, p.y, p.z), opts.yaw or 0
  end
  local ctx = {
    pos = { x = pos.x, y = pos.y, z = pos.z }, yaw = yaw, seed = opts.seed, registry = registry_list(),
    env = make_env(pos.z),
  }
  if IsValid(ply) then
    local tr = ply:GetEyeTrace()
    if tr.Hit and tr.HitWorld and tr.HitNormal.z > 0.7 and tr.HitPos:Distance(pos) < 2000 then
      ctx.aim = { x = tr.HitPos.x, y = tr.HitPos.y, z = tr.HitPos.z }
    end
  elseif opts.aim then
    ctx.aim = opts.aim
  end
  return ctx
end

-- ------------------------------------------------------------------------------------------------ making and unmaking
local function player_key(ply) return IsValid(ply) and ply:SteamID64() or "console" end

local function push_undo(ply, op)
  local key = player_key(ply)
  local st = H.undo_stacks[key] or {}
  H.undo_stacks[key] = st
  st[#st + 1] = op
  if #st > 30 then table.remove(st, 1) end
end

local function entity_count() return #H.order end

local function fade_in(ent, color, duration)
  local c = color or { 255, 255, 255 }
  ent:SetRenderMode(RENDERMODE_TRANSALPHA)
  local steps, i = 12, 0
  local name = "holodeck_fade_" .. ent:EntIndex()
  timer.Create(name, duration / steps, steps, function()
    if not IsValid(ent) then timer.Remove(name); return end
    i = i + 1
    ent:SetColor(Color(c[1], c[2], c[3], math.floor(255 * i / steps)))
    if i >= steps and not color then ent:SetRenderMode(RENDERMODE_NORMAL) end
  end)
  ent:SetColor(Color(c[1], c[2], c[3], 0))
end

local function spawn_prop(entry, fade)
  local ent = ents.Create("prop_physics")
  if not IsValid(ent) then return nil end
  ent:SetModel(entry.model)
  ent:SetPos(Vector(entry.origin.x, entry.origin.y, entry.origin.z))
  ent:SetAngles(Angle(0, entry.yaw, 0))
  ent:Spawn()
  ent:Activate()
  if not IsValid(ent) then return nil end
  ent.HolodeckId = entry.id
  local po = ent:GetPhysicsObject()
  if IsValid(po) then po:EnableMotion(false) end                      -- held still while it materialises
  if fade then fade_in(ent, entry.color, 0.5)
  elseif entry.color then ent:SetColor(Color(entry.color[1], entry.color[2], entry.color[3])) end
  timer.Simple(0.6, function()
    if IsValid(ent) then
      local p2 = ent:GetPhysicsObject()
      if IsValid(p2) then p2:EnableMotion(true); p2:Wake() end
    end
  end)
  timer.Simple(4.5, function()                                         -- a settled prop sleeps instead of jittering
    if IsValid(ent) then
      local p3 = ent:GetPhysicsObject()
      if IsValid(p3) and p3:GetVelocity():LengthSqr() < 400 then p3:Sleep() end
    end
  end)
  return ent
end

local function register(pl, group, ply)
  H.seq = H.seq + 1
  local id = H.next_id
  H.next_id = id + 1
  local entry = {
    id = id, kind = pl.kind, vi = pl.vi, model = pl.model, x = pl.x, y = pl.y, z = pl.z, theta = pl.theta, w = pl.w, d = pl.d, h = pl.h,
    origin = pl.origin, yaw = pl.yaw, color = pl.color, color_name = pl.color_name, group = group, seq = H.seq,
    owner = IsValid(ply) and ply:SteamID64() or "console",
  }
  H.registry[id] = entry
  H.order[#H.order + 1] = id
  return entry
end

local function unregister(id)
  H.registry[id] = nil
  for i, v in ipairs(H.order) do
    if v == id then table.remove(H.order, i); break end
  end
end

hook.Add("EntityRemoved", "Holodeck.Forget", function(ent)
  local id = ent.HolodeckId
  if id and H.registry[id] and H.registry[id].ent == ent and not H.registry[id].respawning then unregister(id) end
end)

local function remove_entry(entry, fade)
  local ent = entry.ent
  unregister(entry.id)
  if IsValid(ent) then
    if fade then
      ent:SetRenderMode(RENDERMODE_TRANSALPHA)
      local steps, i = 8, 0
      local c = entry.color or { 255, 255, 255 }
      timer.Create("holodeck_out_" .. ent:EntIndex(), 0.3 / steps, steps, function()
        if not IsValid(ent) then return end
        i = i + 1
        ent:SetColor(Color(c[1], c[2], c[3], 255 - math.floor(255 * i / steps)))
        if i >= steps then ent:Remove() end
      end)
    else
      ent:Remove()
    end
  end
end

local function snapshot(entry)
  return { kind = entry.kind, vi = entry.vi, model = entry.model, x = entry.x, y = entry.y, z = entry.z, theta = entry.theta, w = entry.w,
           d = entry.d, h = entry.h, origin = entry.origin, yaw = entry.yaw, color = entry.color, color_name = entry.color_name, group = entry.group }
end

--- Put things on the floor, one after another. Returns the registered entries.
local function build(ply, placements)
  H.group = H.group + 1
  local entries = {}
  local delay = cv_stagger:GetFloat()
  for i, pl in ipairs(placements) do
    local entry = register(pl, H.group, ply)
    entries[i] = entry
    H.pending = H.pending + 1
    timer.Simple((i - 1) * delay, function()
      H.pending = H.pending - 1
      if H.registry[entry.id] == entry then
        local ent = spawn_prop(entry, true)
        if ent then entry.ent = ent else unregister(entry.id) end
      end
    end)
  end
  return entries
end

local function new_pose(entry, mv)
  local v = catalog.kinds[entry.kind].variants[entry.vi]
  local origin, yaw = Plan.entity_pose(v, mv.x, mv.y, mv.z, mv.theta)
  return origin, yaw
end

local function apply_moves(moves)
  local old = {}
  for _, mv in ipairs(moves) do
    local e = H.registry[mv.id]
    if e then
      old[#old + 1] = { id = e.id, x = e.x, y = e.y, z = e.z, theta = e.theta, origin = e.origin, yaw = e.yaw }
      local origin, yaw = new_pose(e, mv)
      e.x, e.y, e.z, e.theta, e.origin, e.yaw = mv.x, mv.y, mv.z, mv.theta, origin, yaw
      if IsValid(e.ent) then
        local po = e.ent:GetPhysicsObject()
        e.ent:SetPos(Vector(origin.x, origin.y, origin.z))
        e.ent:SetAngles(Angle(0, yaw, 0))
        if IsValid(po) then po:SetVelocity(Vector(0, 0, 0)); po:Wake() end
      end
    end
  end
  return old
end

-- ------------------------------------------------------------------------------------------------ the entry point
--- Handle one request in plain words. ply may be nil (the console, a test): then opts = {pos=, yaw=, aim=, seed=}.
--- Returns the planner's result table (res.say is what was said), or nil if the words were not a request.
function H.Handle(ply, text, opts)
  opts = opts or {}
  if IsValid(ply) and cv_admin:GetBool() and not ply:IsAdmin() then
    H.Say(ply, "The holodeck is for admins on this server.")
    return nil
  end
  local req = H.parser:parse(text)
  if req.action == "none" then return nil end

  if req.action == "undo" then
    local st = H.undo_stacks[player_key(ply)]
    local op = st and table.remove(st)
    if not op then H.Say(ply, "There is nothing to undo."); return { action = "undo", say = "There is nothing to undo." } end
    if op.type == "add" then
      for _, id in ipairs(op.ids) do
        local e = H.registry[id]
        if e then remove_entry(e, true) end
      end
      H.Say(ply, "Undone: " .. #op.ids .. " things gone.")
    elseif op.type == "remove" then
      local placements = {}
      for i, snap in ipairs(op.snaps) do placements[i] = snap end
      build(ply, placements)
      H.Say(ply, "Undone: " .. #op.snaps .. " things back.")
    elseif op.type == "move" then
      local mvs = {}
      for _, o in ipairs(op.old) do mvs[#mvs + 1] = { id = o.id, x = o.x, y = o.y, z = o.z, theta = o.theta } end
      apply_moves(mvs)
      H.Say(ply, "Undone.")
    end
    return { action = "undo", say = "Undone." }
  end

  if req.action == "clear" then
    local snaps, n = {}, 0
    for _, id in ipairs({ unpack(H.order) }) do
      local e = H.registry[id]
      if e then snaps[#snaps + 1] = snapshot(e); remove_entry(e, true); n = n + 1 end
    end
    if n > 0 then push_undo(ply, { type = "remove", snaps = snaps }) end
    H.Say(ply, n > 0 and ("Cleared " .. n .. " things.") or "There was nothing to clear.")
    return { action = "clear", say = "Cleared " .. n, removed = n }
  end

  local ctx = context(ply, opts)
  if req.action == "add" and entity_count() >= cv_max:GetInt() then
    H.Say(ply, "That is as many props as I will keep standing (" .. cv_max:GetInt() .. "). Clear some first.")
    return { action = "add", say = "limit", placements = {} }
  end
  local ok, res = pcall(H.planner.run, H.planner, req, ctx)
  if not ok then
    ErrorNoHalt("[holodeck] planner error: " .. tostring(res) .. "\n")
    H.Say(ply, "I tripped over that one. The error is in the server console.")
    return { action = "error", say = tostring(res) }
  end
  res.request = req

  if res.action == "add" and #res.placements > 0 then
    local room = cv_max:GetInt() - entity_count()
    if #res.placements > room then
      for i = #res.placements, room + 1, -1 do res.placements[i] = nil end
      res.say = (res.say or "") .. " I only had room for " .. room .. " more."
    end
    local entries = build(ply, res.placements)
    local ids = {}
    for i, e in ipairs(entries) do ids[i] = e.id end
    push_undo(ply, { type = "add", ids = ids })
    res.ids = ids
  elseif res.action == "remove" and res.removals and #res.removals > 0 then
    local snaps = {}
    for _, id in ipairs(res.removals) do
      local e = H.registry[id]
      if e then snaps[#snaps + 1] = snapshot(e); remove_entry(e, true) end
    end
    push_undo(ply, { type = "remove", snaps = snaps })
  elseif (res.action == "move" or res.action == "rotate" or res.action == "rearrange") and res.moves and #res.moves > 0 then
    local old = apply_moves(res.moves)
    push_undo(ply, { type = "move", old = old })
  end
  H.Say(ply, res.say)
  return res
end

-- ------------------------------------------------------------------------------------------------ chat and console
hook.Add("PlayerSay", "Holodeck.Chat", function(ply, text)
  if cv_wake:GetBool() then
    local ok, rest = Parse.strip_wake(text)
    if not ok then return end                                         -- ordinary talk between players: leave it alone
    timer.Simple(0, function() if IsValid(ply) then H.Handle(ply, rest) end end)
    return
  end
  timer.Simple(0, function() if IsValid(ply) then H.Handle(ply, text) end end)
end)

concommand.Add("holodeck", function(ply, _, args, line)
  local text = table.concat(args, " ")
  if text == "" then
    print("holodeck <what you want>   e.g. holodeck add three couches   (also: remove the lamp, undo, clear)")
    return
  end
  H.Handle(ply, text)
end, nil, "Ask the holodeck for something in plain words")

concommand.Add("holodeck_list", function(ply)
  local counts = {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    counts[e.kind] = (counts[e.kind] or 0) + 1
  end
  local parts = {}
  for k, v in pairs(counts) do parts[#parts + 1] = v .. " " .. k end
  table.sort(parts)
  print("[holodeck] " .. #H.order .. " things: " .. table.concat(parts, ", "))
end)

concommand.Add("holodeck_undo", function(ply) H.Handle(ply, "undo") end)
concommand.Add("holodeck_clear", function(ply) H.Handle(ply, "clear everything") end)

--- Entities still being materialised (for tests that wait on the spawner).
function H.Busy() return H.pending > 0 end

MsgC(Color(0, 255, 136), "[mentat] holodeck ready: ", Color(255, 255, 255), #catalog.order .. " kinds. Say \"Skippy, add three couches\" or `holodeck add three couches`.\n")
