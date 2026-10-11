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
  if args.bridge then
    RunConsoleCommand("holodeck_bridge", args.bridge)
    RunConsoleCommand("holodeck_bridge_token", args.token or "")
    RunConsoleCommand("holodeck_agent", tostring(args.agent or 1))
    RunConsoleCommand("holodeck_persona", args.persona and "1" or "0")
    RunConsoleCommand("holodeck_voice_out", args.voice and "1" or "0")
  end

  local function report()
    local ents = {}
    for _, id in ipairs(H.order) do
      local e = H.registry[id]
      local row = { id = id, kind = e.kind, vi = e.vi, model = e.model, group = e.group, planned = { x = e.x, y = e.y, z = e.z, theta = e.theta, origin = e.origin, yaw = e.yaw } }
      local fv = e.variant
      if fv and fv.entity then                       -- a found object is not in the static catalog: its own footprint travels with it
        row.variant = { mins = fv.mins, maxs = fv.maxs, cx = fv.cx, cy = fv.cy, front = fv.front, w = fv.w, d = fv.d, h = fv.h, mass = fv.mass,
                        entity = fv.entity, card = fv.found and fv.found.card }
      end
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
        local omn, omx = e.ent:OBBMins(), e.ent:OBBMaxs()
        row.obb = { mins = vec(omn), maxs = vec(omx) }                   -- the engine's own box of it
        row.class = e.ent:GetClass()
        if e.ent.LabelText then                                           -- a holodeck_found box: what its networked variables hold
          row.found = { name = e.ent:GetFoundName(), source = e.ent:GetFoundSource(), licence = e.ent:GetFoundLicence(), basis = e.ent:GetFoundBasis(),
                        mass = e.ent:GetFoundMass(), label = e.ent:LabelText(), visual = e.ent:GetVisual(), solid = e.ent:GetSolid(), movetype = e.ent:GetMoveType() }
        end
      else
        row.missing = true
      end
      ents[#ents + 1] = row
    end
    done({ pos = pos, steps = steps, entities = ents, count = #ents, library = H.library_report and { kept = H.library_report.kept, total = H.library_report.total,
           kinds = #H.catalog.order, lost_kinds = H.library_report.lost_kinds }, last_audio = H.last_audio, last_voice = H.last_voice,
           agent_error = H.last_agent_error, speak_error = H.last_speak_error })
  end

  local function wait_idle(cb)
    if H.Busy() then timer.Simple(0.2, function() wait_idle(cb) end) else cb() end
  end

  local function fill(step, r)
    step.say, step.action = r and r.say or "", r and r.action or "none"
    step.placed = r and r.placements and #r.placements or 0
    step.notes, step.ids, step.by_agent, step.said = r and r.notes or {}, r and r.ids or {}, r and r.by_agent, r and r.said
  end

  local function nextreq()
    i = i + 1
    local text = (args.requests or {})[i]
    if not text then
      return timer.Simple(args.settle or 7, report)
    end
    local step = { text = text }
    steps[#steps + 1] = step
    local res = H.Handle(nil, text, { pos = pos, yaw = args.yaw or 0, seed = args.seed, aim = args.aim, done = function(r) fill(step, r) end })
    if res then fill(step, res) end
    -- an agent or voice answer arrives later: wait for the bridge before the next request
    local waited = 0
    local function settle_step()
      if H.Busy() and waited < 90 then waited = waited + 0.2; return timer.Simple(0.2, settle_step) end
      step.last_say = H.last_say
      timer.Simple(args.gap or 1.2, nextreq)
    end
    settle_step()
  end
  if args.content then RunConsoleCommand("holodeck_content", args.content) end
  if args.real_mass ~= nil then RunConsoleCommand("holodeck_real_mass", args.real_mass and "1" or "0") end
  -- console variables set by RunConsoleCommand take effect on the next frame: let them land before the first request
  timer.Simple((args.bridge or args.content or args.real_mass ~= nil) and 0.5 or 0, function()
    if args.content then H.BuildLibrary() end
    nextreq()
  end)
end
