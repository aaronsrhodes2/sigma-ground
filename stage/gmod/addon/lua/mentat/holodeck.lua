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
local Library = include("mentat/holodeck/library.lua")
local raw_catalog = include("mentat/holodeck/catalog.lua")

local cv_content = CreateConVar("holodeck_content", "all", FCVAR_ARCHIVE, "all = every model this server has | hl2 = only Half-Life 2 models (what every player owns)")

H.parse_mod, H.plan_mod, H.layout, H.library = Parse, Plan, Layout, Library

--- (Re)build the parser and planner over the models this server can actually show.
function H.BuildLibrary()
  local filtered, report = Library.filter(raw_catalog, util.IsValidModel, { hl2_only = cv_content:GetString() == "hl2" })
  H.catalog, H.library_report = filtered, report
  H.parser = Parse.new(filtered)
  H.planner = Plan.new(filtered)
  -- things already standing keep their model; find it again in the (possibly different) variant list
  for _, e in pairs(H.registry or {}) do
    local k = filtered.kinds[e.kind]
    if k then
      for vi, v in ipairs(k.variants) do if v.model == e.model then e.vi = vi; break end end
    end
  end
  MsgC(Color(0, 255, 136), "[holodeck] ", Color(255, 255, 255), report.kept .. " of " .. report.total .. " catalog models are available here, " ..
       #filtered.order .. " kinds" .. (#report.lost_kinds > 0 and (" (no model at all for: " .. table.concat(report.lost_kinds, ", ") .. ")") or "") .. "\n")
  return report
end
H.registry = {}            -- id -> entry
H.order = {}               -- ids in creation order
H.next_id, H.seq, H.group = 1, 0, 0
H.undo_stacks = {}         -- player key -> list of operations
local function IsValidPlayer(ply) return ply ~= nil and IsValid(ply) and ply:IsPlayer() end
H.BuildLibrary()
H.pending = 0              -- spawns still scheduled
H.log = H.log or {}
H.debug = true

util.AddNetworkString("holodeck_say")
util.AddNetworkString("holodeck_audio")
util.AddNetworkString("holodeck_voice")

local cv_max = CreateConVar("holodeck_max_ents", "400", FCVAR_ARCHIVE, "Most props the holodeck will keep standing at once")
local cv_admin = CreateConVar("holodeck_admin_only", "0", FCVAR_ARCHIVE, "1 = only admins may use the holodeck")
local cv_stagger = CreateConVar("holodeck_stagger", "0.05", FCVAR_ARCHIVE, "Seconds between one prop appearing and the next")
local cv_wake = CreateConVar("holodeck_need_wake", "1", FCVAR_ARCHIVE, "1 = chat and voice need the wake word (Skippy); the console never does")
local cv_bridge = CreateConVar("holodeck_bridge", "", FCVAR_ARCHIVE + FCVAR_PROTECTED, "Skippy bridge URL, e.g. http://127.0.0.1:8765 (empty = no agent and no voice)")
local cv_token = CreateConVar("holodeck_bridge_token", "", FCVAR_ARCHIVE + FCVAR_PROTECTED + FCVAR_DONTRECORD, "shared token for the Skippy bridge")
local cv_agent = CreateConVar("holodeck_agent", "1", FCVAR_ARCHIVE, "0 = never ask the agent, 1 = ask when the words build nothing, 2 = ask for every request to add things")
local cv_persona = CreateConVar("holodeck_persona", "1", FCVAR_ARCHIVE, "1 = Skippy answers in his own words (through the bridge), else a plain sentence")
local cv_cooldown = CreateConVar("holodeck_cooldown", "1.0", FCVAR_ARCHIVE, "Seconds a player must wait between requests")
local cv_per_player = CreateConVar("holodeck_max_per_player", "150", FCVAR_ARCHIVE, "Most props one player may keep standing")
local cv_voice = CreateConVar("holodeck_voice_out", "1", FCVAR_ARCHIVE, "1 = Skippy speaks his answers aloud, from where he built things")

-- ------------------------------------------------------------------------------------------------ the bridge (agent and voice)
local function bridge_url() return cv_bridge:GetString() end

--- POST json to the bridge. ok(body, headers) on 200, fail(reason) otherwise. Returns false if no bridge is configured.
H.thinking = 0             -- bridge calls in flight
local function bridge_post(path, tbl, ok, fail, timeout)
  if bridge_url() == "" then return false end
  H.thinking = H.thinking + 1
  local finished = false
  local function settle() if not finished then finished = true; H.thinking = math.max(0, H.thinking - 1) end end
  local sent = HTTP({
    url = bridge_url() .. path, method = "POST", type = "application/json", body = util.TableToJSON(tbl),
    headers = { ["X-Skippy-Token"] = cv_token:GetString() }, timeout = timeout or 20,
    success = function(code, body, headers)
      settle()
      if code == 200 then ok(body, headers) else fail("HTTP " .. tostring(code) .. " " .. tostring(body):sub(1, 120)) end
    end,
    failed = function(reason) settle(); fail(tostring(reason)) end,
  })
  if sent == false then settle(); fail("the engine refused the request (is -allowlocalhttp set?)") end
  return true
end
H.BridgePost = bridge_post

local audio_id = 0
--- Send audio bytes to every player who could hear Skippy where he stands (and the one who asked).
local function send_audio(bytes, pos, ply)
  audio_id = (audio_id % 60000) + 1
  local size = 30000
  local total = math.max(1, math.ceil(#bytes / size))
  local targets = {}
  for _, p in ipairs(player.GetAll()) do
    if p == ply or (pos and p:GetPos():Distance(pos) < 2600) then targets[#targets + 1] = p end
  end
  H.last_audio = { id = audio_id, bytes = #bytes, chunks = total, to = #targets }
  if #targets == 0 then return end
  for i = 1, total do
    local chunk = bytes:sub((i - 1) * size + 1, i * size)
    net.Start("holodeck_audio")
    net.WriteUInt(audio_id, 16)
    net.WriteUInt(i, 8)
    net.WriteUInt(total, 8)
    net.WriteVector(pos or Vector(0, 0, 0))
    net.WriteUInt(#chunk, 16)
    net.WriteData(chunk, #chunk)
    net.Send(targets)
  end
end

--- Skippy speaks: the bridge makes the audio, the server hands it to whoever is near.
function H.Speak(text, pos, ply)
  if not cv_voice:GetBool() or bridge_url() == "" or not text or text == "" then return false end
  return bridge_post("/speak", { text = text }, function(body, headers)
    if #body > 200 then
      H.last_voice = headers and (headers["X-Skippy-Voice"] or headers["x-skippy-voice"]) or "?"
      send_audio(body, pos, ply)
    end
  end, function(reason)
    H.last_speak_error = reason
  end, 30)
end

-- ------------------------------------------------------------------------------------------------ speaking
--- Say it in chat for everyone (and aloud, if the bridge has a voice). `pos` is where the voice comes from.
function H.Say(ply, text, pos)
  if not text or text == "" then return end
  H.last_say = text
  H.log[#H.log + 1] = text
  if #H.log > 200 then table.remove(H.log, 1) end
  print("[holodeck] " .. text)
  net.Start("holodeck_say")
  net.WriteString(text)
  net.Broadcast()
  hook.Run("HolodeckSaid", ply, text)
  H.Speak(text, pos or (IsValid(ply) and ply:GetPos()) or nil, ply)
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

--- Who may change what: everyone's things belong to whoever made them; admins, the host of a listen server, a player alone in
--- single player and the console may change anything.
local function manages_all(ply)
  if not IsValidPlayer(ply) then return true end
  return ply:IsAdmin() or game.SinglePlayer() or ply:IsListenServerHost()
end

local function registry_list(owner)
  local out = {}
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    if e and (not owner or e.owner == owner) then
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
  if IsValid(ply) and not manages_all(ply) then ctx.mine = registry_list(ply:SteamID64()) end
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
  local v = H.catalog.kinds[entry.kind].variants[entry.vi]
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
local function centre_of(res, ply)
  local n, cx, cy, cz = 0, 0, 0, 0
  for _, p in ipairs(res.placements or {}) do cx, cy, cz, n = cx + p.x, cy + p.y, cz + p.z, n + 1 end
  if n > 0 then return Vector(cx / n, cy / n, cz / n + 30) end
  return IsValid(ply) and ply:GetPos() or nil
end

--- Say the outcome: in Skippy's own words when the bridge has them, else the plain sentence.
local function announce(ply, text, res)
  local pos = centre_of(res or {}, ply)
  if cv_persona:GetBool() and bridge_url() ~= "" and res and res.say and res.say ~= "" then
    local done = false
    local function plain() if not done then done = true; H.Say(ply, res.say, pos) end end
    local started = bridge_post("/reply", { user = text, did = res.say, notes = res.notes or {} }, function(body)
      local obj = util.JSONToTable(body or "") or {}
      if not done and obj.say and obj.say ~= "" then done = true; res.said = obj.say; H.Say(ply, obj.say, pos) else plain() end
    end, function() plain() end, 12)
    if not started then plain() end
    timer.Simple(13, plain)
  else
    H.Say(ply, res and res.say, pos)
  end
end

--- Run a parsed (or agent-made) request. Returns the planner's result table.
function H.Execute(ply, req, opts, text)
  opts = opts or {}
  text = text or req.raw or ""
  if req.action == "undo" then
    local st = H.undo_stacks[player_key(ply)]
    local op = st and table.remove(st)
    if not op then announce(ply, text, { say = "There is nothing to undo." }); return { action = "undo", say = "There is nothing to undo." } end
    if op.type == "add" then
      for _, id in ipairs(op.ids) do
        local e = H.registry[id]
        if e then remove_entry(e, true) end
      end
      announce(ply, text, { say = "Undone: " .. #op.ids .. " things gone." })
    elseif op.type == "remove" then
      local placements = {}
      for i, snap in ipairs(op.snaps) do placements[i] = snap end
      build(ply, placements)
      announce(ply, text, { say = "Undone: " .. #op.snaps .. " things back." })
    elseif op.type == "move" then
      local mvs = {}
      for _, o in ipairs(op.old) do mvs[#mvs + 1] = { id = o.id, x = o.x, y = o.y, z = o.z, theta = o.theta } end
      apply_moves(mvs)
      announce(ply, text, { say = "Undone." })
    end
    return { action = "undo", say = "Undone." }
  end

  if req.action == "clear" then
    local snaps, n = {}, 0
    local all = manages_all(ply)
    for _, id in ipairs({ unpack(H.order) }) do
      local e = H.registry[id]
      if e and (all or e.owner == player_key(ply)) then snaps[#snaps + 1] = snapshot(e); remove_entry(e, true); n = n + 1 end
    end
    if n > 0 then push_undo(ply, { type = "remove", snaps = snaps }) end
    announce(ply, text, { say = n > 0 and ("Cleared " .. n .. " things" .. (all and "." or " of yours.")) or "There was nothing of yours to clear." })
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
    if IsValid(ply) and not manages_all(ply) then
      local mine = #registry_list(ply:SteamID64())
      room = math.min(room, cv_per_player:GetInt() - mine)
    end
    room = math.max(room, 0)
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
  announce(ply, text, res)
  if opts.done then opts.done(res) end
  return res
end

--- Does this parsed request build or change anything the planner understands?
local function actionable(req)
  if req.action ~= "add" then return true end
  if req.scene and req.scene ~= "" then return true end
  for _, it in ipairs(req.items or {}) do
    if not tostring(it.kind):match("^UNKNOWN") and it.kind ~= "@last" then return true end
  end
  return false
end

local function nearby_kinds()
  local seen, out = {}, {}
  for _, id in ipairs(H.order) do
    local k = H.registry[id].kind
    if not seen[k] then seen[k] = true; out[#out + 1] = k end
    if #out >= 12 then break end
  end
  return out
end

--- Ask the agent (through the bridge) to turn a vague wish into a request; run it. fallback() runs when the agent cannot help.
local function ask_agent(ply, text, opts, fallback)
  return bridge_post("/plan", { text = text, nearby = nearby_kinds() }, function(body)
    if GetConVar("developer"):GetInt() > 0 or H.debug then print("[holodeck] agent answered: " .. tostring(body):sub(1, 600)) end
    local obj = util.JSONToTable(body or "")
    local req = obj and obj.request
    if not (req and req.action and req.action ~= "none") then
      if obj and obj.say and obj.say ~= "" then
        H.Say(ply, obj.say)
        if opts.done then opts.done({ action = "none", say = obj.say }) end
      else
        fallback()
      end
      return
    end
    -- the bridge validated every field against the library; give it the parser's shape
    local items = {}
    for i, it in ipairs(req.items or {}) do
      items[i] = { kind = it.kind, count = tonumber(it.count) or 1, adjectives = it.adjectives or {}, word = it.word or it.kind, position = i }
    end
    req.items, req.raw, req.normalised, req.unknown, req.notes = items, text, text:lower(), {}, {}
    req.arrangement = (req.arrangement and req.arrangement ~= "") and req.arrangement or "unspecified"
    req.target = req.target or ""
    req.scene = req.scene or ""
    local res = H.Execute(ply, req, opts, text)
    if res then
      res.by_agent = true
      for _, n in ipairs(obj.notes or {}) do res.notes = res.notes or {}; table.insert(res.notes, n) end
    end
  end, function(reason)
    H.last_agent_error = reason
    print("[holodeck] the agent could not help: " .. tostring(reason):sub(1, 200))
    fallback()
  end, 30)
end

--- Handle one request in plain words. ply may be nil (the console, a test): then opts = {pos=, yaw=, aim=, seed=, done=fn}.
--- Returns the planner's result table when it ran at once, or nil (the words were not a request, or the agent is thinking;
--- opts.done receives the result either way).
function H.Handle(ply, text, opts)
  opts = opts or {}
  if IsValid(ply) and cv_admin:GetBool() and not ply:IsAdmin() then
    H.Say(ply, "The holodeck is for admins on this server.")
    return nil
  end
  local req = H.parser:parse(text)
  if req.action == "none" then return nil end
  if IsValid(ply) and not manages_all(ply) then
    local now = CurTime()
    if H.last_request and H.last_request[ply] and now - H.last_request[ply] < cv_cooldown:GetFloat() then return nil end
    H.last_request = H.last_request or setmetatable({}, { __mode = "k" })
    H.last_request[ply] = now
  end
  local mode = cv_agent:GetInt()
  -- the words build nothing, or two or more of them carry meaning the parser could not place ("somewhere to relax")
  local unexplained = req.action == "add" and #(req.leftover or {}) >= 2
  local use_agent = bridge_url() ~= "" and mode > 0 and ((mode == 2 and req.action == "add") or not actionable(req) or unexplained)
  if use_agent then
    local function fallback()
      local res = H.Execute(ply, req, opts, text)
      if opts.done and res then opts.done(res) end
    end
    if ask_agent(ply, text, opts, fallback) then return nil end
  end
  return H.Execute(ply, req, opts, text)
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

-- what a player said aloud, after their own microphone bridge heard the wake word (see skippy_client.lua)
local last_spoken = {}
net.Receive("holodeck_voice", function(_, ply)
  local text = net.ReadString()
  if #text == 0 or #text > 300 then return end
  if last_spoken[ply] and CurTime() - last_spoken[ply] < 1.0 then return end
  last_spoken[ply] = CurTime()
  local woke, rest = Parse.strip_wake(text)
  H.Handle(ply, woke and rest or text)
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

concommand.Add("holodeck_status", function(ply)
  local r = H.library_report
  print(string.format("[holodeck] %d of %d models available, %d kinds, %d rooms; content=%s; %d things standing; bridge=%s agent=%d persona=%s voice=%s",
    r.kept, r.total, #H.catalog.order, table.Count(H.catalog.recipes), cv_content:GetString(), #H.order,
    bridge_url() ~= "" and "set" or "none", cv_agent:GetInt(), tostring(cv_persona:GetBool()), tostring(cv_voice:GetBool())))
  if #r.lost_models > 0 then print("[holodeck] missing here: " .. table.concat(r.lost_models, ", ", 1, math.min(#r.lost_models, 12)) .. (#r.lost_models > 12 and " ..." or "")) end
end, nil, "What the holodeck can show on this server")

-- re-read the models available (after changing holodeck_content, or after the addon's files were updated on disk)
concommand.Add("holodeck_reload", function(ply)
  if IsValid(ply) and not ply:IsSuperAdmin() then return end
  local ok, err = pcall(H.BuildLibrary)
  print(ok and "[holodeck] library rebuilt" or ("[holodeck] reload failed: " .. tostring(err)))
end, nil, "Rebuild the holodeck's library (superadmin)")

concommand.Add("holodeck_undo", function(ply) H.Handle(ply, "undo") end)
concommand.Add("holodeck_clear", function(ply) H.Handle(ply, "clear everything") end)

--- Entities still being materialised (for tests that wait on the spawner).
function H.Busy() return H.pending > 0 or H.thinking > 0 end

MsgC(Color(0, 255, 136), "[mentat] holodeck ready: ", Color(255, 255, 255), #H.catalog.order .. " kinds. Say \"Skippy, add three couches\" or `holodeck add three couches`.\n")
