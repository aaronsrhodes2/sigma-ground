-- holodeck_run.lua -- drive the holodeck headlessly (dedicated server), then report what really happened to every prop:
-- where each was planned to stand, where the physics engine left it, whether it has gone to sleep.
-- args: { requests = {"add three couches", ...}, pos = {x, y, z?}, yaw = 0, seed = 1, aim = {x, y, z}?, settle = 7 }
-- the floor under (x, y), starting from a player spawn point (maps sit at wildly different heights)
local function floor_below(x, y)
  local z0 = 0
  local best
  for _, e in ipairs(ents.FindByClass("info_player_start")) do
    local d = (e:GetPos().x - x) ^ 2 + (e:GetPos().y - y) ^ 2
    if not best or d < best then best, z0 = d, e:GetPos().z end
  end
  local tr = util.TraceLine({ start = Vector(x, y, z0 + 100), endpos = Vector(x, y, z0 - 20000), mask = MASK_SOLID_BRUSHONLY })
  return tr.Hit and tr.HitPos.z or z0
end

local function vec(v) return { v.x, v.y, v.z } end

return function(args, done)
  local H = Holodeck
  if not (H and H.Handle) then return done(nil, "the holodeck is not loaded on this server") end
  local pos = args.pos or { x = 0, y = 0 }
  if pos.z == nil then pos = { x = pos.x, y = pos.y, z = floor_below(pos.x, pos.y) } end
  local steps, i = {}, 0

  local function report()
    local ents = {}
    for _, id in ipairs(H.order) do
      local e = H.registry[id]
      local row = { id = id, kind = e.kind, vi = e.vi, model = e.model, group = e.group, planned = { x = e.x, y = e.y, z = e.z, theta = e.theta, origin = e.origin, yaw = e.yaw } }
      if IsValid(e.ent) then
        local po = e.ent:GetPhysicsObject()
        row.actual = { pos = vec(e.ent:GetPos()), ang = { e.ent:GetAngles().p, e.ent:GetAngles().y, e.ent:GetAngles().r } }
        row.speed = IsValid(po) and po:GetVelocity():Length() or -1
        row.asleep = IsValid(po) and po:IsAsleep() or false
        row.mass = IsValid(po) and po:GetMass() or -1
        -- corners of the engine's own bounding box that sit inside solid world (a prop through a wall)
        local mn, mx = e.ent:OBBMins(), e.ent:OBBMaxs()
        local inside = 0
        for _, cx in ipairs({ mn.x + 1, mx.x - 1 }) do
          for _, cy in ipairs({ mn.y + 1, mx.y - 1 }) do
            for _, cz in ipairs({ mn.z + 2, mx.z - 2 }) do
              local wp = e.ent:LocalToWorld(Vector(cx, cy, cz))
              if bit.band(util.PointContents(wp), CONTENTS_SOLID) ~= 0 then inside = inside + 1 end
            end
          end
        end
        row.in_wall = inside
      else
        row.missing = true
      end
      ents[#ents + 1] = row
    end
    done({ pos = pos, steps = steps, entities = ents, count = #ents })
  end

  local function wait_idle(cb)
    if H.Busy() then timer.Simple(0.2, function() wait_idle(cb) end) else cb() end
  end

  local function nextreq()
    i = i + 1
    local text = (args.requests or {})[i]
    if not text then
      return timer.Simple(args.settle or 7, report)
    end
    local res = H.Handle(nil, text, { pos = pos, yaw = args.yaw or 0, seed = args.seed, aim = args.aim })
    local step = { text = text, say = res and res.say or "", action = res and res.action or "none",
                   placed = res and res.placements and #res.placements or 0, notes = res and res.notes or {}, ids = res and res.ids or {} }
    steps[#steps + 1] = step
    wait_idle(function() timer.Simple(args.gap or 1.2, nextreq) end)
  end
  nextreq()
end
