-- voice_wall_test.lua -- does a wall between two players shorten how far a voice carries? Finds a spot on the map where one bot can
-- stand on each side of a real wall, then asks the proximity-voice rule (Mentat.Voice.CanHear) about that pair and about the same
-- distance in the open. args: { bases = { {x,y,z}, ... } }   (points to search from; every direction is tried)
return function(args, done)
  local V = Mentat and Mentat.Voice
  if not V then return done(nil, "proximity voice is not loaded") end
  RunConsoleCommand("mentat_voice_mode", "proximity")
  RunConsoleCommand("mentat_voice_range", "1100")
  RunConsoleCommand("mentat_voice_wall", "1")
  local a = player.CreateNextBot("wall_listener")
  local b = player.CreateNextBot("wall_talker")
  if not (IsValid(a) and IsValid(b)) then return done(nil, "could not create bots") end

  local MASK = MASK_SOLID_BRUSHONLY
  local function free(p)
    local tr = util.TraceHull({ start = p, endpos = p, mins = Vector(-16, -16, 0), maxs = Vector(16, 16, 72), mask = MASK })
    return not tr.StartSolid and not tr.Hit
  end
  local function floor_at(x, y, z0)
    local tr = util.TraceLine({ start = Vector(x, y, z0 + 60), endpos = Vector(x, y, z0 - 80), mask = MASK })
    return tr.Hit and not tr.StartSolid and tr.HitPos.z or nil
  end

  -- a pair with a wall between: standing at base, a ray hits a vertical surface; some way past it there is free floor again
  local function find_pair()
    for _, base in ipairs(args.bases or {}) do
      local fz = floor_at(base.x, base.y, base.z)
      if fz then
        local b0 = Vector(base.x, base.y, fz + 1)
        if free(b0) then
          for ang = 0, 345, 15 do
            local d = Vector(math.cos(math.rad(ang)), math.sin(math.rad(ang)), 0)
            local tr = util.TraceLine({ start = b0 + Vector(0, 0, 36), endpos = b0 + Vector(0, 0, 36) + d * 500, mask = MASK })
            if tr.Hit and not tr.StartSolid and math.abs(tr.HitNormal.z) < 0.3 then
              for _, t in ipairs({ 20, 40, 64, 96, 140, 200 }) do
                local p = tr.HitPos + d * t
                local pz = floor_at(p.x, p.y, fz)
                if pz and math.abs(pz - fz) < 4 then
                  local p0 = Vector(p.x, p.y, pz + 1)
                  if free(p0) then
                    local los = util.TraceLine({ start = b0 + Vector(0, 0, 64), endpos = p0 + Vector(0, 0, 64), mask = MASK })
                    if los.Hit then return b0, p0, d end
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  timer.Simple(1.5, function()
    local out = { rows = {} }
    local talker_pos, listener_pos, dir = find_pair()
    if not talker_pos then
      a:Kick("test over"); b:Kick("test over")
      return done({ found = false })
    end
    b:RemoveFlags(FL_DUCKING)
    local d = talker_pos:Distance(listener_pos)
    -- choose a reach so that this separation sits in the band a wall should cut (0.55 x reach < d < reach)
    RunConsoleCommand("mentat_voice_range", tostring(math.floor(d / 0.8)))
    timer.Simple(0.4, function()
      local function ask(label, lp, tp)
        V.ClearCache()
        a:SetPos(lp); b:SetPos(tp)
        local los = util.TraceLine({ start = a:GetShootPos(), endpos = b:GetShootPos(), mask = MASK })
        local heard, gain = V.CanHear(a, b)
        out.rows[#out.rows + 1] = { label = label, dist = a:GetPos():Distance(b:GetPos()), blocked = los.Hit, heard = heard, gain = gain }
      end
      ask("through a wall", listener_pos, talker_pos)
      -- the same separation with a clear line of sight: stand the listener d away on the open side of the talker
      local lo
      for _, ang in ipairs({ 0, 45, 90, 135, 180, 225, 270, 315 }) do
        local dd = Vector(math.cos(math.rad(ang)), math.sin(math.rad(ang)), 0)
        local p = talker_pos + dd * d
        local pz = floor_at(p.x, p.y, talker_pos.z)
        if pz and math.abs(pz - talker_pos.z) < 4 and free(Vector(p.x, p.y, pz + 1)) then
          local los = util.TraceLine({ start = talker_pos + Vector(0, 0, 64), endpos = Vector(p.x, p.y, pz + 65), mask = MASK })
          if not los.Hit then lo = Vector(p.x, p.y, pz + 1); break end
        end
      end
      if lo then ask("same distance, clear line", lo, talker_pos) end
      out.found = true
      out.reach = V.TalkerRange(b)
      out.wall_reach = out.reach * 0.55
      RunConsoleCommand("mentat_voice_range", "1100")
      a:Kick("test over"); b:Kick("test over")
      done(out)
    end)
  end)
end
