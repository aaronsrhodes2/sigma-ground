-- world.lua -- the construct's physics, applied (server side).  REAL-WORLD PHYSICS ONLY.
--
-- What this does, and ONLY this:
--   * gravity: the real Earth value from the sigma-ground library (or the Moon, Mars, Jupiter, free fall when asked), converted with
--     the stage scale (1 HU = 0.01905 m) and set on physenv AND sv_gravity, so the players' walking and jumping follow the same number;
--   * aerodynamic drag: the real force F = 1/2 rho Cd A v^2 on every free physics object, applied each physics tick as a velocity
--     step (world/aero.lua), with Source's own air drag switched off on those objects so this is the only air drag;
--   * an optional phys_timescale (the server owner's switch; 1 = untouched).
-- Friction and restitution are the engine's surface properties and are not touched.  Players' walking speed and jump are engine
-- game feel, not physics, and are left alone.  The game models no thermodynamics, weather or fluids and this file does not pretend.
--
-- Numbers: mentat/world/constants.lua (generated from the library; each entry has units, source, status).  Words: world/intent.lua.
--
--   mentat_world_enabled      1    0 = the construct leaves the engine's own physics alone (gravity restored, engine drag back on)
--   mentat_world_preset       earth  the gravity in force: earth | moon | mars | jupiter | zero
--   mentat_world_aero         1    0 = no air drag at all (a measurement switch, labelled not real-world)
--   mentat_world_engine_drag  0    1 = leave Source's own air drag on as well (a measurement switch)
--   mentat_world_time_scale   1    phys_timescale multiplier (1 = untouched)
--   mentat_world_admin_only   1    1 = only admins (and the listen-server host, and a single-player game) change the physics
if not SERVER then return end

Mentat = Mentat or {}
MentatWorld = MentatWorld or {}
local W = MentatWorld
Mentat.World = W

local C = include("mentat/world/constants.lua")
local Intent = include("mentat/world/intent.lua")
local Aero = include("mentat/world/aero.lua")
W.constants, W.intent, W.aero = C, Intent, Aero

local cv_enabled = CreateConVar("mentat_world_enabled", "1", FCVAR_ARCHIVE, "1 = apply the construct's real-world physics (gravity, air drag); 0 = the engine's own")
local cv_preset = CreateConVar("mentat_world_preset", C.DEFAULTS.preset, FCVAR_ARCHIVE, "Gravity in force: earth | moon | mars | jupiter | zero")
local cv_aero = CreateConVar("mentat_world_aero", "1", FCVAR_ARCHIVE, "0 = no air drag (a measurement switch, not real-world)")
local cv_edrag = CreateConVar("mentat_world_engine_drag", "0", FCVAR_ARCHIVE, "1 = also leave Source's own air drag on (a measurement switch)")
local cv_time = CreateConVar("mentat_world_time_scale", "1", FCVAR_ARCHIVE, "phys_timescale multiplier (1 = untouched)")
local cv_admin = CreateConVar("mentat_world_admin_only", "1", FCVAR_ARCHIVE, "1 = only admins, the listen-server host and a single-player game change the physics")

W.log = W.log or {}
W.tracked = W.tracked or setmetatable({}, { __mode = "k" })     -- ent -> { class =, dims = {a,b,c} metres, cd = }
W.CLASSES = { "prop_physics", "prop_physics_multiplayer", "holodeck_found" }
W.MIN_SPEED = 0.05      -- m/s: below this the drag is far under the engine's own noise and the body is left alone
local SWEEP = 3         -- seconds between sweeps that catch any loose object the creation hooks missed

local function note(text)
  W.log[#W.log + 1] = text
  if #W.log > 100 then table.remove(W.log, 1) end
  print("[world] " .. text)
end

-- ------------------------------------------------------------------------------------------------ state
function W.state()
  return C.new_state({ preset = cv_preset:GetString(), engine_air_drag = cv_edrag:GetBool(), time_scale = cv_time:GetFloat() })
end

local function store(s)
  cv_preset:SetString(s.preset)
end

-- ------------------------------------------------------------------------------------------------ which objects
local function is_free_object(ent)
  if not IsValid(ent) then return false end
  if ent.IsPlayer and ent:IsPlayer() then return false end
  if ent.IsNPC and ent:IsNPC() then return false end
  if ent.IsNextBot and ent:IsNextBot() then return false end
  if ent.IsVehicle and ent:IsVehicle() then return false end
  local cls = ent:GetClass()
  for _, c in ipairs(W.CLASSES) do if cls == c then return true end end
  return ent.HolodeckId ~= nil                                  -- anything the holodeck registered
end
W.is_free_object = is_free_object

local function name_of_object(ent)
  local H = Holodeck
  if ent.HolodeckId and H and H.registry and H.registry[ent.HolodeckId] then return H.registry[ent.HolodeckId].kind or "" end
  return ent.GetModel and ent:GetModel() or ""
end

--- The object's real side lengths (metres), its shape class and Cd, measured once from its bounding box.
local function describe(ent)
  local mn, mx = ent:OBBMins(), ent:OBBMaxs()
  local dims = { (mx.x - mn.x) * C.HU_M, (mx.y - mn.y) * C.HU_M, (mx.z - mn.z) * C.HU_M }
  local class = Aero.shape_class(name_of_object(ent), dims)
  return { dims = dims, class = class, cd = C.CD[class].value }
end

--- Take an object into the construct: Source's own drag off (ours is the only one), its shape measured.
local function adopt(ent, on)
  local po = ent:GetPhysicsObject()
  if not IsValid(po) then return false end
  local rec = W.tracked[ent]
  if not rec then
    rec = describe(ent)
    W.tracked[ent] = rec
  end
  local want_off = on and not W.state().engine_air_drag
  if rec.engine_drag_off ~= want_off then
    if po.EnableDrag then po:EnableDrag(not want_off) end
    rec.engine_drag_off = want_off
  end
  return true
end

--- Track one object: adopt it now, or shortly if its physics is not ready yet.
function W.track(ent, tries)
  if not is_free_object(ent) then return false end
  if not cv_enabled:GetBool() then return false end
  if adopt(ent, true) then return true end
  tries = (tries or 0) + 1
  if tries <= 4 then timer.Simple(0.25 * tries, function() W.track(ent, tries) end) end
  return false
end

--- All free objects there are now.
function W.objects()
  local out, seen = {}, {}
  local function add(e) if not seen[e] and is_free_object(e) then seen[e] = true; out[#out + 1] = e end end
  for _, cls in ipairs(W.CLASSES) do for _, e in ipairs(ents.FindByClass(cls)) do add(e) end end
  if Holodeck and Holodeck.registry then for _, entry in pairs(Holodeck.registry) do if entry.ent then add(entry.ent) end end end
  return out
end

function W.sweep()
  local on = cv_enabled:GetBool()
  local n = 0
  for _, e in ipairs(W.objects()) do if adopt(e, on) then n = n + 1 end end
  return n
end

-- ------------------------------------------------------------------------------------------------ the drag, every tick
--- Apply one tick of real air drag to one awake, moving object. Returns the speed lost (m/s), or nil when it was left alone.
function W.drag_step(ent, rec, dt)
  local po = ent:GetPhysicsObject()
  if not IsValid(po) then return nil end
  if po.IsAsleep and po:IsAsleep() then return nil end                       -- a sleeping body is never touched or woken
  if po.IsMotionEnabled and not po:IsMotionEnabled() then return nil end
  local mass = po:GetMass()
  if not (mass and mass > 0) then return nil end
  local v = po:GetVelocity()                                                  -- HU/s, world frame
  local sp_hu = v:Length()
  local sp = sp_hu * C.HU_M
  if sp < W.MIN_SPEED then return nil end
  local n_world = v * (1 / sp_hu)
  local lp, l0 = ent:WorldToLocal(ent:GetPos() + n_world), ent:WorldToLocal(ent:GetPos())
  local n = { lp.x - l0.x, lp.y - l0.y, lp.z - l0.z }
  local area = Aero.area(rec.class, rec.dims, n)
  local k = Aero.k(C.AIR_DENSITY, rec.cd, area, mass)
  local sp2 = Aero.speed_after(sp, k, dt)
  local dv = (sp2 - sp) / C.HU_M                                              -- HU/s, along the velocity (negative)
  po:AddVelocity(n_world * dv)
  return sp - sp2
end

local function on_tick()
  if not cv_enabled:GetBool() or not cv_aero:GetBool() then return end
  local dt = engine.TickInterval()
  for ent, rec in pairs(W.tracked) do
    if IsValid(ent) then W.drag_step(ent, rec, dt) end
  end
end
W.on_tick = on_tick

-- ------------------------------------------------------------------------------------------------ applying the physics
W.applied = W.applied or {}

local function publish(s)
  local d = C.derive(s)
  SetGlobalFloat("mentat_world_g_m_s2", d.gravity_m_s2)
  SetGlobalFloat("mentat_world_g_hu_s2", d.gravity_hu_s2)
  SetGlobalFloat("mentat_world_air_density", d.air_density)
  SetGlobalFloat("mentat_world_time_scale", s.time_scale)
  SetGlobalString("mentat_world_preset", s.preset)
  SetGlobalBool("mentat_world_on", cv_enabled:GetBool())
  SetGlobalBool("mentat_world_aero", cv_aero:GetBool())
end

--- Apply the state in force. `wake` = wake the loose objects after a gravity change so they notice it.
function W.apply(wake)
  local on = cv_enabled:GetBool()
  local s = W.state()
  local d = C.derive(s)
  local g_hu = on and d.gravity_hu_s2 or C.ENGINE_GRAVITY_HU_S2
  local a = W.applied
  if a.g_hu ~= g_hu then
    physenv.SetGravity(Vector(0, 0, -g_hu))
    RunConsoleCommand("sv_gravity", string.format("%.6f", g_hu))      -- the players' walking and jumping follow the same number
    if a.g_hu ~= nil and wake then a.wake = true end
    a.g_hu = g_hu
  end
  local ts = on and s.time_scale or 1
  if (a.ts or 1) ~= ts then
    pcall(RunConsoleCommand, "phys_timescale", tostring(ts))
    a.ts = ts
  end
  W.sweep()
  if a.wake then
    a.wake = nil
    for _, e in ipairs(W.objects()) do
      local po = e:GetPhysicsObject()
      if IsValid(po) and po.Wake then po:Wake() end
    end
  end
  publish(s)
  return s
end

-- ------------------------------------------------------------------------------------------------ who may change it
local function IsPlayerEnt(p) return p ~= nil and IsValid(p) and p.IsPlayer and p:IsPlayer() end

function W.may_change(ply)
  if not IsPlayerEnt(ply) then return true end                                  -- the server console
  if not cv_admin:GetBool() then return true end
  if game.SinglePlayer() then return true end
  if ply.IsListenServerHost and ply:IsListenServerHost() then return true end
  return (ply.IsAdmin and ply:IsAdmin()) and true or false
end

-- ------------------------------------------------------------------------------------------------ talking
local function say(ply, text, speak)
  local H = Holodeck
  if H and H.Say then H.Say(ply, text, nil, { speak = speak ~= false })
  elseif IsPlayerEnt(ply) and ply.ChatPrint then ply:ChatPrint("[Skippy] " .. text) end
end
local function tell(ply, text)
  local H = Holodeck
  if H and H.Tell then H.Tell(ply, text)
  elseif IsPlayerEnt(ply) and ply.ChatPrint then ply:ChatPrint("[Skippy] " .. text) end
end
local function who(ply) return IsPlayerEnt(ply) and (ply:Nick() .. " " .. tostring(ply:SteamID64())) or "console" end

--- Read the physics back: a spoken summary and private detail lines.
function W.readback(ply)
  local lines = C.readback(W.state())
  say(ply, lines[1])
  for i = 2, #lines do tell(ply, lines[i]) end
  return lines
end

-- ------------------------------------------------------------------------------------------------ the words
W.next_ok = W.next_ok or {}

local REFUSAL = {
  gravity = "Gravity here is a real place's, not a dial. I can give you Earth's, the Moon's, Mars', Jupiter's or none at all.",
  drag = "Air drag here is the real force on each object's own shape, size and mass, so there is no dial for it.",
}

--- Try a sentence as world physics. Returns nil when it is not (the caller goes on to the object parser), else a result table.
function W.Handle(ply, text)
  local its = Intent.parse(text)
  if not its then return nil end
  local key = IsPlayerEnt(ply) and ply:SteamID64() or "console"
  local presets, queries, refusals = {}, {}, {}
  for _, it in ipairs(its) do
    if it.kind == "query" then queries[#queries + 1] = it
    elseif it.kind == "not_real" then refusals[#refusals + 1] = it
    else presets[#presets + 1] = it end
  end
  if #presets > 0 then
    if not W.may_change(ply) then
      tell(ply, "The physics of this room are for the admins on this server.")
      note(who(ply) .. " was refused: " .. tostring(text))
      return { action = "world", refused = true, say = "refused" }
    end
    if CurTime() < (W.next_ok[key] or 0) then return { action = "world", say = "", throttled = true } end
    W.next_ok[key] = CurTime() + 1.0
    local old = W.state()
    local s = old
    for _, it in ipairs(presets) do s = C.resolve(s, it) end
    if not cv_enabled:GetBool() then cv_enabled:SetBool(true) end
    store(s)
    W.apply(true)
    local d = C.derive(s)
    local msg
    if s.preset == old.preset then
      msg = "Already so: " .. (s.preset == "zero" and "zero gravity" or (C.PRESETS[s.preset].label .. "'s gravity")) .. ", " .. C.g3(d.gravity_m_s2) .. " m/s2."
    else
      msg = (s.preset == "zero" and "Free fall." or ("Gravity is now " .. (s.preset == "earth" and "Earth's" or (C.PRESETS[s.preset].label .. "'s")) .. ".")) ..
            " " .. C.g3(d.gravity_m_s2) .. " m/s2, " .. C.g3(d.gravity_hu_s2) .. " HU/s2. I woke the loose things so they notice."
    end
    note(who(ply) .. " said \"" .. tostring(text) .. "\": " .. msg)
    say(ply, msg)
    return { action = "world", say = msg, state = s, changed = s.preset ~= old.preset }
  end
  if #refusals > 0 then
    local msg = REFUSAL[refusals[1].what]
    note(who(ply) .. " asked for a knob: \"" .. tostring(text) .. "\": declined")
    say(ply, msg)
    return { action = "world", say = msg, refused_knob = true }
  end
  if #queries > 0 then
    local lines = W.readback(ply)
    return { action = "world", say = lines[1], query = true }
  end
  return nil
end

-- Consult the world words before the object parser: wrap Holodeck.Handle once (chat, voice and the console all call it).
local function install()
  local H = Holodeck
  if not (H and H.Handle) or H.Handle == W.wrapped then return false end
  local inner = H.Handle
  local function handle(ply, text, opts)
    if type(text) == "string" and not (opts and opts.found_done) then
      local ok, res = pcall(W.Handle, ply, text)
      if ok and res then
        if opts and opts.done then opts.done(res) end
        return res
      elseif not ok then
        ErrorNoHalt("[world] " .. tostring(res) .. "\n")
      end
    end
    return inner(ply, text, opts)
  end
  H.Handle = handle
  W.wrapped, W.inner = handle, inner
  return true
end
W.install = install

concommand.Add("mentat_world", function(ply, _, args)
  local text = table.concat(args, " ")
  if text == "" then
    W.readback(ply)
    return
  end
  if not W.Handle(ply, text) then tell(ply, "I did not understand that as physics. Try: set gravity to the moon, zero gravity, earth gravity, what are the world's physics.") end
end, nil, "Change or read the construct's physics in plain words")

concommand.Add("mentat_world_status", function(ply)
  for _, l in ipairs(C.readback(W.state())) do print("[world] " .. l) end
  local n = 0
  for _ in pairs(W.tracked) do n = n + 1 end
  print("[world] " .. n .. " objects tracked; applied gravity " .. tostring(W.applied.g_hu) .. " HU/s2; enabled=" .. tostring(cv_enabled:GetBool()) .. " aero=" .. tostring(cv_aero:GetBool()))
end, nil, "What the construct's physics are")

-- ------------------------------------------------------------------------------------------------ hooks
hook.Add("OnEntityCreated", "MentatWorld.Created", function(ent)
  timer.Simple(0, function() if IsValid(ent) then W.track(ent) end end)       -- the physics object does not exist yet at creation
end)
hook.Add("PlayerSpawnedProp", "MentatWorld.Spawned", function(ply, model, ent) timer.Simple(0, function() if IsValid(ent) then W.track(ent) end end) end)
hook.Add("Tick", "MentatWorld.Aero", on_tick)

for _, name in ipairs({ "mentat_world_enabled", "mentat_world_preset", "mentat_world_engine_drag", "mentat_world_time_scale" }) do
  cvars.AddChangeCallback(name, function() timer.Simple(0, function() W.apply(true) end) end, "MentatWorld.cv")
end

hook.Add("InitPostEntity", "MentatWorld.MapStart", function()
  W.applied = {}
  W.apply(false)
  note("applied at map start: " .. C.summary(W.state()))
end)

timer.Create("MentatWorld.Sweep", SWEEP, 0, function() if cv_enabled:GetBool() then W.sweep() end end)

install()
timer.Simple(0, install)                      -- once more after every addon file has loaded, in case Holodeck came later
W.applied = {}
timer.Simple(0, function() W.apply(false) end)
