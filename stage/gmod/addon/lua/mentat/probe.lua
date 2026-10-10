-- probe.lua -- engine-semantics probe: what do GMod's angular-velocity setters actually do?
--
-- The wiki does not state the units (deg/s or rad/s) or the frame (world or local) of
-- PhysObj:SetAngleVelocity / SetAngleVelocityInstantaneous / AddAngleVelocity. This probe
-- asks the engine: on a body tilted to a known orientation it applies the SAME vector
-- through each setter and records the entity's axes tick by tick. It never grades: the
-- analysis is stage/gmod/verify_probe.py.
--
-- Trigger: data/mentat/probe.txt = {"probe":"angvel","scenario":"fall_skillet"}
if not SERVER then return end

local CFG = "mentat/probe.txt"
if not file.Exists(CFG, "DATA") then return end
local cfg = util.JSONToTable(file.Read(CFG, "DATA") or "")
file.Delete(CFG)
if not (cfg and cfg.probe == "angvel") then return end

local U = Mentat.units
local TICKS = 12                     -- samples per case (0.18 s: little time to fall)
local TILT = Angle(30, 50, 20)       -- a generic orientation: local axes differ from world axes
local SPIN = Vector(0, 0, 360)       -- the vector handed to the setter-comparison cases

-- Each case applies a vector through one setter. The three Instantaneous cases cover each local
-- axis so the SIGN convention (right-hand rule about local +X/+Y/+Z) is measured, not assumed.
local CASES = {
  { name = "Instantaneous X(360,0,0)", spin = Vector(360, 0, 0), call = function(p, v) p:SetAngleVelocityInstantaneous(v) end },
  { name = "Instantaneous Y(0,360,0)", spin = Vector(0, 360, 0), call = function(p, v) p:SetAngleVelocityInstantaneous(v) end },
  { name = "Instantaneous Z(0,0,360)", spin = Vector(0, 0, 360), call = function(p, v) p:SetAngleVelocityInstantaneous(v) end },
  { name = "SetAngleVelocity Z(0,0,360)", spin = Vector(0, 0, 360), call = function(p, v) p:SetAngleVelocity(v) end },
  { name = "AddAngleVelocity Z(0,0,360)", spin = Vector(0, 0, 360), call = function(p, v) p:AddAngleVelocity(v) end },
}

local function vec(v) return { v.x, v.y, v.z } end

local function write(out)
  file.CreateDir("mentat")
  file.Write("mentat/probe_angvel.json", util.TableToJSON(out, true))
  print("[mentat-probe] wrote data/mentat/probe_angvel.json")
end

local function run(ply)
  local anchor = ply:GetPos() + Vector(300, 0, 400)       -- high up: the body only falls ~8 HU while sampled
  local out = { probe = "angvel", tick = engine.TickInterval(), spin = vec(SPIN), tilt = { TILT.p, TILT.y, TILT.r },
                cases = {}, map = game.GetMap() }
  local ok, err = Mentat.Load(cfg.scenario or "fall_skillet", ply:GetPos() + Vector(300, 0, 0))
  if not ok then out.error = err return write(out) end
  local ent = Mentat.state.ents[1]
  local phys = ent:GetPhysicsObject()
  if not IsValid(phys) then out.error = "no physics object" return write(out) end
  ent.MentatReleased = true                                 -- keep Mentat.Apply away from it
  phys:EnableDrag(false)
  phys:SetDamping(0, 0)

  local idx, step, case = 0, 0, nil
  hook.Add("Tick", "MentatProbe", function()
    if not case then
      idx = idx + 1
      if idx > #CASES then
        hook.Remove("Tick", "MentatProbe")
        return write(out)
      end
      local c = CASES[idx]
      case = { name = c.name, spin = vec(c.spin), samples = {} }
      out.cases[#out.cases + 1] = case
      phys:EnableMotion(true)
      phys:Wake()
      ent:SetPos(anchor)
      ent:SetAngles(TILT)
      phys:SetVelocity(Vector(0, 0, 0))
      phys:SetAngleVelocityInstantaneous(Vector(0, 0, 0))
      c.call(phys, c.spin)
      case.readback = vec(phys:GetAngleVelocity())
      step = 0
      return
    end
    step = step + 1
    case.samples[#case.samples + 1] = {
      ct = CurTime(), fwd = vec(ent:GetForward()), right = vec(ent:GetRight()), up = vec(ent:GetUp()),
      angvel = vec(phys:GetAngleVelocity()),
    }
    if step >= TICKS then case = nil end
  end)
end

hook.Add("PlayerInitialSpawn", "MentatProbe", function(ply)
  timer.Simple(5, function()
    if IsValid(ply) then run(ply) end
  end)
end)
print("[mentat-probe] armed (angvel)")
