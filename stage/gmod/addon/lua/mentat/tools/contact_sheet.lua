-- contact_sheet.lua -- one close three-quarter photograph per catalog model, each turned so that the side the catalog calls
-- its FRONT points toward +X, with the camera on that side. If the catalog is right you see the seat, the doors, the
-- screen; if it is wrong you see the back. A red cube marks the assumed front, a blue one the back.
-- Pictures land in data/mentat/shot_item_<n>.jpg (needs the game client; a camera).
-- args: {items = { { model=, front_yaw=<deg or nil>, mins={..}, maxs={..} }, ... }, view_yaw = 35, elevation = 24}
local HU = 1 / 0.01905

local function rot(x, y, deg)
  local r = math.rad(deg)
  return x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
end

local function cube(pos, color)
  local m = ents.Create("prop_physics")
  m:SetModel("models/hunter/blocks/cube025x025x025.mdl")
  m:SetPos(pos)
  m:Spawn()
  m:SetColor(color)
  local po = m:GetPhysicsObject()
  if IsValid(po) then po:EnableMotion(false) end
  return m
end

return function(args, done, ply)
  if not IsValid(ply) then return done(nil, "contact_sheet needs the game client (a player with a camera)") end
  ply:GodEnable()
  ply:SetMoveType(MOVETYPE_NOCLIP)
  ply:SetFOV(60, 0)
  local base = ply:GetPos()
  local tr = util.TraceLine({ start = base + Vector(0, 0, 64), endpos = base - Vector(0, 0, 4096), mask = MASK_SOLID_BRUSHONLY })
  local floor_z = tr.Hit and tr.HitPos.z or base.z
  local ox0, oy0 = base.x + 500, base.y
  local view_yaw, elev = math.rad(args.view_yaw or 35), math.rad(args.elevation or 24)
  local spawned, shots, i = {}, 0, 0

  local function clear()
    for _, e in ipairs(spawned) do if IsValid(e) then e:Remove() end end
    spawned = {}
  end

  local function build(it)
    local a = it.front_yaw and -it.front_yaw or 0
    local lx, ly = (it.mins[1] + it.maxs[1]) / 2, (it.mins[2] + it.maxs[2]) / 2
    local ex, ey = 0, 0
    local hx, hy = (it.maxs[1] - it.mins[1]) / 2, (it.maxs[2] - it.mins[2]) / 2
    for _, sx in ipairs({ -1, 1 }) do
      for _, sy in ipairs({ -1, 1 }) do
        local rx, ry = rot(sx * hx, sy * hy, a)
        ex, ey = math.max(ex, math.abs(rx)), math.max(ey, math.abs(ry))
      end
    end
    local e = ents.Create("prop_physics")
    e:SetModel(it.model)
    e:SetAngles(Angle(0, a, 0))
    local px, py = rot(lx, ly, a)
    e:SetPos(Vector(ox0 - px, oy0 - py, floor_z - it.mins[3] + 1))
    e:Spawn()
    local po = e:GetPhysicsObject()
    if IsValid(po) then po:EnableMotion(false) end
    spawned[#spawned + 1] = e
    if it.front_yaw ~= nil then
      spawned[#spawned + 1] = cube(Vector(ox0 + ex + 0.25 * HU, oy0, floor_z + 6), Color(255, 30, 30))
      spawned[#spawned + 1] = cube(Vector(ox0 - ex - 0.25 * HU, oy0, floor_z + 6), Color(40, 80, 255))
    end
    local height = it.maxs[3] - it.mins[3]
    local size = math.max(2 * ex, 2 * ey, height)
    local target = Vector(ox0, oy0, floor_z + height * 0.45)
    local dist = 1.35 * size + 18
    local cam = target + Vector(math.cos(view_yaw) * math.cos(elev), math.sin(view_yaw) * math.cos(elev), math.sin(elev)) * dist
    ply:SetPos(cam - Vector(0, 0, 64))
    ply:SetEyeAngles((target - cam):Angle())
  end

  local function nextitem()
    clear()
    i = i + 1
    local it = args.items[i]
    if not it then
      ply:SetMoveType(MOVETYPE_WALK)
      ply:SetFOV(0, 0)
      return done({ shots = shots })
    end
    build(it)
    timer.Simple(0.9, function()
      Mentat.Shot("item_" .. i)
      shots = shots + 1
      timer.Simple(0.7, nextitem)
    end)
  end
  nextitem()
end
