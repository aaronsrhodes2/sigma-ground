-- water/swim_move.lua -- the Move hook that lets a player swim in a holodeck_water body (both realms; the client predicts with the same
-- numbers).  The law is water/swim.lua's.  Players in noclip (the Skippy avatar bot, admins flying) and in vehicles are never touched.
local M = {}
local inc = include or function(p) return dofile(p) end
local Swim = inc("mentat/water/swim.lua")
local Vol = inc("mentat/water/volumes.lua")
local C = inc("mentat/water/constants.lua")
local HU_M = C.HU_M

M.enabled = true
M.state = {}                       -- ply -> { cooldown = s }

local function gravity()
  return GetGlobalFloat("mentat_world_g_m_s2", C.data.g_earth.value)
end
M.gravity = gravity

local function is_swimmer(ply)
  if not IsValid(ply) or not ply:IsPlayer() or not ply:Alive() then return false end
  if ply:InVehicle() then return false end
  local mt = ply:GetMoveType()
  return mt == MOVETYPE_WALK
end

local function wish_of(mv, ang)
  local fwd, side = mv:GetForwardSpeed(), mv:GetSideSpeed()
  local top = mv:GetMaxClientSpeed()
  if not top or top <= 0 then top = 400 end
  local f, s = math.max(-1, math.min(1, fwd / top)), math.max(-1, math.min(1, side / top))
  local af, ar = ang:Forward(), ang:Right()
  return { af.x * f + ar.x * s, af.y * f + ar.y * s, af.z * f + ar.z * s }
end

--- A ledge to pull yourself onto in the direction of travel: returns the standing position on it, or nil.  The wall is hit one step ahead at the
--- swimmer's height, its top is found by a line down just beyond it, the top must be within Swim.reach_ok of the feet, and the hull must fit on it.
local function find_ledge(ply, origin, wish, mv)
  local dx, dy = wish[1], wish[2]
  local l = math.sqrt(dx * dx + dy * dy)
  if l < 0.05 then
    local fwd = mv:GetMoveAngles():Forward()
    dx, dy, l = fwd.x, fwd.y, math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y)
    if l < 1e-6 then return nil end
  end
  local ux, uy = dx / l, dy / l
  local mins, maxs = ply:GetHull()
  local flat = util.TraceHull({ start = origin, endpos = origin + Vector(ux * 30, uy * 30, 0), mins = mins, maxs = maxs, filter = ply, mask = MASK_PLAYERSOLID })
  if not flat.Hit then return nil end                                      -- open water ahead: nothing to climb
  local beyond = Vector(origin.x + ux * 34, origin.y + uy * 34, 0)
  local down = util.TraceLine({ start = Vector(beyond.x, beyond.y, origin.z + 110), endpos = Vector(beyond.x, beyond.y, origin.z - 10), filter = ply, mask = MASK_PLAYERSOLID })
  if not down.Hit or down.StartSolid then return nil end
  if not Swim.reach_ok(origin.z * HU_M, down.HitPos.z * HU_M) then return nil end
  local stand = Vector(beyond.x, beyond.y, down.HitPos.z + 1)
  local fit = util.TraceHull({ start = stand, endpos = stand, mins = mins, maxs = maxs, filter = ply, mask = MASK_PLAYERSOLID })
  if fit.Hit or fit.StartSolid then return nil end
  return stand, Vector(ux * 60, uy * 60, 0)
end

local function move(ply, mv)
  if not M.enabled or not is_swimmer(ply) then return end
  local list = M.list_fn and M.list_fn() or Vol.list()
  if #list == 0 then return end
  local origin = mv:GetOrigin()
  local centre_z = origin.z + 36
  local r = Vol.holding(list, origin.x, origin.y, centre_z)
  if not r then return end
  local z_feet = origin.z * HU_M
  local f = Swim.fraction(z_feet, r.surf * HU_M)
  local st = M.state[ply] or { cooldown = 0 }
  M.state[ply] = st
  if f < Swim.MIN_FRACTION then return end                    -- wading: the engine's own walking
  local dt = FrameTime()
  if not dt or dt <= 0 or dt > 0.2 then return end
  local vel = mv:GetVelocity()
  local wish = wish_of(mv, mv:GetMoveAngles())
  local want_up = mv:KeyDown(IN_JUMP)
  if want_up and f < 0.99 and st.cooldown <= 0 then                         -- head and shoulders out: an edge in reach pulls you onto it
    local stand, push = find_ledge(ply, origin, wish, mv)
    if stand then
      st.cooldown = 0.8
      mv:SetOrigin(stand)
      mv:SetVelocity(push)
      return true
    end
  end
  local S = {
    z_feet = z_feet, surface = r.surf * HU_M, vel = { vel.x * HU_M, vel.y * HU_M, vel.z * HU_M }, wish = wish,
    up = want_up, down = mv:KeyDown(IN_DUCK), g = gravity(), rho = r.L.density_kg_m3, eta = r.L.viscosity_pa_s, cooldown = st.cooldown,
    cd = M.cd,
  }
  local v2, info = Swim.step(S, dt)
  st.cooldown = info.cooldown
  local nv = Vector(v2[1] / HU_M, v2[2] / HU_M, v2[3] / HU_M)
  -- move through the world: a hull trace, then one slide along what it hit
  local mins, maxs = ply:GetHull()
  local pos = origin
  local delta = nv * dt
  for _ = 1, 2 do
    local tr = util.TraceHull({ start = pos, endpos = pos + delta, mins = mins, maxs = maxs, filter = ply, mask = MASK_PLAYERSOLID })
    if not tr.Hit then pos = pos + delta; delta = Vector(0, 0, 0); break end
    pos = tr.HitPos
    local n = tr.HitNormal
    local rest = delta * (1 - tr.Fraction)
    delta = rest - n * rest:Dot(n)
    nv = nv - n * math.min(0, nv:Dot(n))                     -- the velocity into the surface is lost
  end
  mv:SetOrigin(pos)
  mv:SetVelocity(nv)
  return true
end
M.move = move

function M.install()
  hook.Add("Move", "MentatWater.Swim", move)
end

return M
