-- holodeck_probe.lua -- what the engine says about the map around a point (debugging the engine glue).
-- args: { x, y, z? }  -> spawn points, the floor found by a long trace, and wall rays at that floor
return function(args, done)
  local MASK = MASK_SOLID_BRUSHONLY
  local out = { spawns = {} }
  for _, cls in ipairs({ "info_player_start", "info_player_teamspawn", "gmod_player_start" }) do
    for _, e in ipairs(ents.FindByClass(cls)) do out.spawns[#out.spawns + 1] = { cls, e:GetPos().x, e:GetPos().y, e:GetPos().z } end
  end
  local x, y = args.x or 0, args.y or 0
  local z0 = args.z or (out.spawns[1] and out.spawns[1][4]) or 0
  local tr = util.TraceLine({ start = Vector(x, y, z0 + 100), endpos = Vector(x, y, z0 - 20000), mask = MASK })
  out.floor = { hit = tr.Hit, startsolid = tr.StartSolid, sky = tr.HitSky, z = tr.HitPos.z, nz = tr.HitNormal.z }
  local fz = tr.HitPos.z
  local rays = {}
  for a = 0, 330, 30 do
    local dx, dy = math.cos(math.rad(a)), math.sin(math.rad(a))
    local t = util.TraceLine({ start = Vector(x, y, fz + 30), endpos = Vector(x + dx * 900, y + dy * 900, fz + 30), mask = MASK })
    rays[#rays + 1] = { a = a, hit = t.Hit, f = t.Fraction, nz = t.HitNormal.z, sky = t.HitSky }
  end
  out.rays = rays
  done(out)
end
