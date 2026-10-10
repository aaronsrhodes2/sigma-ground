-- holodeck_shots.lua -- ask the holodeck for things in the game client, then photograph the result from several sides.
-- args: { requests = {...}, settle = 6, views = { {yaw = 0, elev = 28, dist = 1.0}, ... } }  (needs the game client)
local function centroid_and_radius()
  local H = Holodeck
  local n, cx, cy, cz = 0, 0, 0, 0
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    cx, cy, cz, n = cx + e.x, cy + e.y, cz + e.z, n + 1
  end
  if n == 0 then return nil end
  cx, cy, cz = cx / n, cy / n, cz / n
  local r = 0
  for _, id in ipairs(H.order) do
    local e = H.registry[id]
    r = math.max(r, math.sqrt((e.x - cx) ^ 2 + (e.y - cy) ^ 2) + math.max(e.w, e.d) / 2)
  end
  return Vector(cx, cy, cz), r
end

return function(args, done, ply)
  local H = Holodeck
  if not IsValid(ply) then return done(nil, "holodeck_shots needs the game client (a player with a camera)") end
  ply:GodEnable()
  if args.pos then
    -- stand somewhere particular first (a room with walls), on the floor under that point
    local z0 = args.pos.z or ply:GetPos().z
    local tr = util.TraceLine({ start = Vector(args.pos.x, args.pos.y, z0 + 100), endpos = Vector(args.pos.x, args.pos.y, z0 - 400), mask = MASK_SOLID_BRUSHONLY })
    ply:SetPos(Vector(args.pos.x, args.pos.y, tr.Hit and tr.HitPos.z + 1 or z0))
    ply:SetEyeAngles(Angle(0, args.yaw or 0, 0))
  end
  local steps, i = {}, 0
  local views = args.views or { { yaw = 0, elev = 30, dist = 1.0 }, { yaw = 120, elev = 30, dist = 1.0 }, { yaw = 240, elev = 30, dist = 1.0 }, { yaw = 0, elev = 80, dist = 1.1 } }
  local tag = args.tag or "hd"

  local function wait_idle(cb)
    if H.Busy() then timer.Simple(0.2, function() wait_idle(cb) end) else cb() end
  end

  local function shoot()
    local c, r = centroid_and_radius()
    if not c then return done({ steps = steps, shots = 0 }) end
    ply:SetMoveType(MOVETYPE_NOCLIP)
    ply:SetFOV(70, 0)
    local k = 0
    local function nextview()
      k = k + 1
      local v = views[k]
      if not v then
        ply:SetMoveType(MOVETYPE_WALK)
        ply:SetFOV(0, 0)
        return done({ steps = steps, shots = k - 1, centre = { c.x, c.y, c.z }, radius = r })
      end
      -- far enough that the whole scene (radius r) fills the 70 degree view with a margin
      local dist = (v.dist or 1.0) * ((r + 50) / math.tan(math.rad(35)))
      local el = math.rad(v.elev or 30)
      local ya = math.rad(v.yaw or 0)
      local cam = c + Vector(math.cos(ya) * math.cos(el), math.sin(ya) * math.cos(el), math.sin(el)) * dist + Vector(0, 0, 20)
      ply:SetPos(cam - Vector(0, 0, 64))
      ply:SetVelocity(-ply:GetVelocity())
      ply:SetEyeAngles(((c + Vector(0, 0, 20)) - cam):Angle())
      timer.Simple(0.6, function()
        Mentat.Shot(tag .. "_" .. k)
        timer.Simple(0.7, nextview)
      end)
    end
    nextview()
  end

  local function nextreq()
    i = i + 1
    local text = (args.requests or {})[i]
    if not text then return timer.Simple(args.settle or 6, shoot) end
    if args.pos then ply:SetEyeAngles(Angle(0, args.yaw or 0, 0)) end
    local res = H.Handle(ply, text)
    steps[#steps + 1] = { text = text, say = res and res.say or "" }
    wait_idle(function() timer.Simple(args.gap or 1.5, nextreq) end)
  end
  nextreq()
end
