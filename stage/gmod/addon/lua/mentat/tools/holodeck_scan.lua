-- holodeck_scan.lua -- find enclosed rooms on a map: grid points whose wall rays almost all hit close by.
-- args: { x0, x1, y0, y1, step, z (floor height to keep) }
return function(args, done)
  local MASK = MASK_SOLID_BRUSHONLY
  local z0 = args.z or 0
  local out = {}
  local step = args.step or 150
  for x = args.x0 or -2000, args.x1 or 2000, step do
    for y = args.y0 or -2000, args.y1 or 2000, step do
      local tr = util.TraceLine({ start = Vector(x, y, z0 + 120), endpos = Vector(x, y, z0 - 60), mask = MASK })
      if tr.Hit and not tr.StartSolid and tr.HitNormal.z > 0.9 and math.abs(tr.HitPos.z - z0) < 6 then
        local fz = tr.HitPos.z
        local hits, near, far = 0, 1e9, 0
        local d = {}
        for a = 0, 330, 30 do
          local t = util.TraceLine({ start = Vector(x, y, fz + 30), endpos = Vector(x + math.cos(math.rad(a)) * 700, y + math.sin(math.rad(a)) * 700, fz + 30), mask = MASK })
          if t.Hit and not t.HitSky then
            hits = hits + 1
            local dist = t.Fraction * 700
            near, far = math.min(near, dist), math.max(far, dist)
            d[#d + 1] = math.floor(dist)
          end
        end
        if hits == 12 then out[#out + 1] = { x = x, y = y, z = fz, near = near, far = far } end
      end
    end
  end
  done({ rooms = out })
end
