-- water_live.lua -- measure the construct's water in the real engine (headless, dedicated server): floating drafts, a sinking body's terminal
-- speed, splash energies, the Moon, mercury, a swimming bot, and what the water costs per physics tick.  Graded against the sigma-ground
-- library by stage/gmod/holodeck/water_live.py.
--
-- args = { depth_m = 12, densities = {600, 300}, scenarios = {"frame", "float", "sink", "splash", "moon", "mercury", "swim", "cost"} }
if not SERVER then return end

local HU_M = 0.01905

return function(args, done)
  local H, W = Holodeck, MentatWater
  if not (H and W) then return done(nil, "the holodeck and the water (mentat/water.lua) must be loaded") end
  local want = {}
  for _, s in ipairs(args.scenarios or { "frame", "float", "sink", "splash", "moon", "mercury", "swim", "cost" }) do want[s] = true end
  local out = { dt = engine.TickInterval(), notes = {}, drafts = {}, sinks = {}, splashes = {}, moon = {}, mercury = {}, swim = {}, cost = {}, frame = {} }
  local depth_m = args.depth_m or 12
  local co
  local function sleep(t)
    timer.Simple(t, function()
      local ok, e = coroutine.resume(co)
      if not ok then print("[water_live] ERROR " .. tostring(e)); done(nil, "script: " .. tostring(e)) end
    end)
    coroutine.yield()
  end
  local function floor_at(x, y)
    local z0 = 0
    for _, e in ipairs(ents.FindByClass("info_player_start")) do z0 = e:GetPos().z; break end
    local tr = util.TraceLine({ start = Vector(x, y, z0 + 200), endpos = Vector(x, y, z0 - 20000), mask = MASK_SOLID_BRUSHONLY })
    return tr.Hit and tr.HitPos.z or z0
  end
  local fz0 = floor_at(0, 0)
  local function say(text, opts) return H.Handle(nil, text, opts or { pos = { x = 0, y = 0, z = fz0 }, yaw = 0, seed = 1 }) end
  local function clear()
    say("clear everything")
    sleep(2.5)
    for _, e in ipairs(ents.FindByClass("holodeck_found")) do if IsValid(e) then e:Remove() end end
  end
  local function body_of() return W.volumes.list()[1] end

  --- a loose box of a card's kind: size in metres, mass in kg, placed with its bottom at (x, y, z) HU
  local function box(size_m, mass, x, y, z)
    local e = ents.Create("holodeck_found")
    local hx, hy, hz = size_m[1] / HU_M / 2, size_m[2] / HU_M / 2, size_m[3] / HU_M
    e:SetFound({ mins = { -hx, -hy, 0 }, maxs = { hx, hy, hz }, name = "test box", source = "water_live", licence = "", basis = "measured", mass = mass })
    e:SetPos(Vector(x, y, z))
    e:Spawn()
    e:Activate()
    local po = e:GetPhysicsObject()
    if IsValid(po) then po:SetMass(mass); po:Wake() end
    return e
  end

  local function make_pool(text)
    clear()
    say(text)
    sleep(4)
    local b = body_of()
    if not b then out.notes[#out.notes + 1] = "no water body after: " .. text; return nil end
    return b
  end

  --- sample an entity every tick for `secs`; returns rows { t, z_m, vz_m_s }
  local function record(ent, secs, stop_when)
    local rows, n = {}, 0
    local name = "WaterLiveRec" .. ent:EntIndex() .. n
    local fin, finished = nil, false
    hook.Add("Tick", name, function()
      if finished then return end
      if not IsValid(ent) then finished = true; hook.Remove("Tick", name); return end
      local po = ent:GetPhysicsObject()
      n = n + 1
      rows[#rows + 1] = { n * out.dt, ent:GetPos().z * HU_M, IsValid(po) and po:GetVelocity().z * HU_M or 0, IsValid(po) and po:IsAsleep() or false }
      if n * out.dt >= secs or (stop_when and stop_when(rows)) then finished = true; hook.Remove("Tick", name) end
    end)
    while not finished do sleep(0.2) end
    return rows
  end

  local function step(msg) print("[water_live] " .. msg) end
  step("start")
  co = coroutine.create(function()
    -- ------------------------------------------------------------------ the angular-velocity frame probe
    if want.frame then
      step('frame')
      local e = box({ 0.5, 0.5, 0.5 }, 20, 2000, 2000, fz0 + 2000)
      sleep(0.3)
      local po = e:GetPhysicsObject()
      po:EnableMotion(true)
      po:EnableGravity(false)
      e:SetAngles(Angle(0, 0, 90))                                   -- rolled a quarter turn: local and world axes differ
      po:SetAngleVelocity(Vector(0, 0, 0))
      sleep(0.1)
      local f0 = e:GetForward()
      po:AddAngleVelocity(Vector(0, 0, 100))                          -- 100 degrees per second about "z" of the frame the engine uses
      local w_read = po:GetAngleVelocity()
      sleep(0.3)
      local f1 = e:GetForward()
      -- rotation about WORLD z moves the forward vector in the xy plane (z unchanged); about the body's local z (the world -y or +y of a rolled body) it tilts forward up or down
      out.frame = { forward0 = { f0.x, f0.y, f0.z }, forward1 = { f1.x, f1.y, f1.z }, w_read = { w_read.x, w_read.y, w_read.z },
                    d_xy = math.sqrt((f1.x - f0.x) ^ 2 + (f1.y - f0.y) ^ 2), d_z = math.abs(f1.z - f0.z) }
      out.frame.kind = out.frame.d_z > out.frame.d_xy and "local" or "world"
      W.ang_frame = out.frame.kind
      e:Remove()
    end

    -- ------------------------------------------------------------------ floating drafts (Earth)
    if want.float then
      step('float')
      local b = make_pool("add a pool " .. depth_m .. " metres deep")
      if b then
        local surf = b.surf * HU_M
        out.pool = { x = b.x, y = b.y, floor_m = b.floor * HU_M, surf_m = surf, liquid = b.liquid, g = W.gravity() }
        for _, rho in ipairs(args.densities or { 600, 300 }) do
          local side = 0.5
          local e = box({ side, side, side }, rho * side ^ 3, b.x, b.y, (surf + 0.5) / HU_M)
          local rows = record(e, 40, function(r) return #r > 100 and r[#r][4] end)
          local last = rows[#rows]
          out.drafts[#out.drafts + 1] = { rho = rho, side = side, draft_m = surf - last[2], asleep = last[4], t_end = last[1], vz_end = last[3], n = #rows,
                                          z_trace = { rows[1][2], rows[math.floor(#rows / 4)][2], rows[math.floor(#rows / 2)][2], last[2] } }
          e:Remove()
          sleep(0.5)
        end
      end
    end

    -- ------------------------------------------------------------------ a sinking steel block
    if want.sink then
      step('sink')
      local b = body_of() or make_pool("add a pool " .. depth_m .. " metres deep")
      if b then
        local side = 0.2
        local e = box({ side, side, side }, 7850 * side ^ 3, b.x, b.y, b.surf + 2)
        local rows = record(e, 6, function(r) return r[#r][2] <= b.floor * HU_M + 0.01 end)
        out.sinks[#out.sinks + 1] = { name = "steel", side = side, mass = 7850 * side ^ 3, surf_m = b.surf * HU_M, floor_m = b.floor * HU_M, rows = rows }
        e:Remove()
        sleep(0.5)
      end
    end

    -- ------------------------------------------------------------------ splash energies
    if want.splash then
      step('splash')
      local b = body_of()
      if b then
        local log = {}
        local inner = W.send_splash
        W.send_splash = function(body, kind, x, y, mass, speed) log[#log + 1] = { kind = kind, mass = mass, speed = speed }; return inner(body, kind, x, y, mass, speed) end
        for _, h in ipairs({ 0.3, 2.0 }) do
          local e = box({ 0.3, 0.3, 0.3 }, 20, b.x, b.y, b.surf + h / HU_M)
          sleep(3)
          out.splashes[#out.splashes + 1] = { h_m = h, events = log }
          log = {}
          e:Remove()
          sleep(0.3)
        end
        W.send_splash = inner
      end
    end

    -- ------------------------------------------------------------------ the Moon
    if want.moon then
      step('moon')
      local b = body_of()
      if b then
        local WW = MentatWorld
        if WW then WW.Handle(nil, "set gravity to the moon") end
        sleep(1.0)
        local g = W.gravity()
        local surf = b.surf * HU_M
        local side = 0.5
        local e = box({ side, side, side }, 600 * side ^ 3, b.x, b.y, (surf + 0.5) / HU_M)
        local rows = record(e, 60, function(r) return #r > 100 and r[#r][4] end)
        local last = rows[#rows]
        out.moon.draft = { rho = 600, side = side, draft_m = surf - last[2], asleep = last[4], t_end = last[1], g = g }
        e:Remove()
        sleep(0.3)
        local s2 = 0.2
        local e2 = box({ s2, s2, s2 }, 7850 * s2 ^ 3, b.x, b.y, b.surf + 2 / HU_M)
        local rows2 = record(e2, 12, function(r) return r[#r][2] <= b.floor * HU_M + 0.01 end)
        out.moon.sink = { side = s2, mass = 7850 * s2 ^ 3, surf_m = surf, floor_m = b.floor * HU_M, g = g, rows = rows2 }
        e2:Remove()
        if WW then WW.Handle(nil, "set gravity to earth") end
        sleep(1.0)
      end
    end

    -- ------------------------------------------------------------------ mercury
    if want.mercury then
      step('mercury')
      local b = make_pool("add a pool of mercury 3 metres deep")
      if b then
        local surf = b.surf * HU_M
        local side = 0.3
        local e = box({ side, side, side }, 7850 * side ^ 3, b.x, b.y, (surf + 0.5) / HU_M)
        local rows = record(e, 60, function(r) return #r > 100 and r[#r][4] end)
        local last = rows[#rows]
        out.mercury = { rho = 7850, side = side, draft_m = surf - last[2], asleep = last[4], t_end = last[1], liquid = b.liquid }
        e:Remove()
      end
    end

    -- ------------------------------------------------------------------ cost: many bodies in the water
    if want.cost then
      step('cost')
      local b = make_pool("add a pool 6 metres deep")
      if b then
        local n = 40
        local list = {}
        for i = 1, n do
          local rho = (i % 2 == 0) and 400 or 2500
          list[i] = box({ 0.3, 0.3, 0.3 }, rho * 0.027, b.x + ((i - 1) % 8 - 4) * 40, b.y + (math.floor((i - 1) / 8) - 2) * 40, (b.surf * HU_M + 0.3 + 0.1 * (i % 5)) / HU_M)
        end
        local c0 = { ticks = W.counters.ticks, us = W.counters.us, processed = W.counters.processed }
        sleep(6)
        local c1 = { ticks = W.counters.ticks, us = W.counters.us, processed = W.counters.processed }
        out.cost = { bodies = n, ticks = c1.ticks - c0.ticks, us_per_tick = (c1.us - c0.us) / math.max(1, c1.ticks - c0.ticks),
                     processed_per_tick = (c1.processed - c0.processed) / math.max(1, c1.ticks - c0.ticks), tick_ms = out.dt * 1000 }
        sleep(8)
        local c2 = { ticks = W.counters.ticks, us = W.counters.us }
        out.cost.settled_us_per_tick = (c2.us - c1.us) / math.max(1, c2.ticks - c1.ticks)
        local asleep = 0
        for _, e in ipairs(list) do local po = IsValid(e) and e:GetPhysicsObject(); if po and IsValid(po) and po:IsAsleep() then asleep = asleep + 1 end end
        out.cost.asleep_after_14s = asleep
        for _, e in ipairs(list) do if IsValid(e) then e:Remove() end end
      end
    end

    -- ------------------------------------------------------------------ a bot swims
    if want.swim then
      step('swim')
      local b = make_pool("add a pool 3 metres deep")
      if b then
        RunConsoleCommand("bot")
        sleep(2)
        local bot
        for _, p in ipairs(player.GetAll()) do if p:IsBot() then bot = p end end
        if not bot then
          out.notes[#out.notes + 1] = "no bot could be added (maxplayers 1?)"
        else
          local keys = { forward = false, jump = false, duck = false }
          hook.Add("StartCommand", "WaterLiveBot", function(p, cmd)
            if p ~= bot then return end
            cmd:ClearMovement(); cmd:ClearButtons()
            if keys.forward then cmd:SetForwardMove(10000) end
            local btn = 0
            if keys.jump then btn = bit.bor(btn, IN_JUMP) end
            if keys.duck then btn = bit.bor(btn, IN_DUCK) end
            if keys.forward then btn = bit.bor(btn, IN_FORWARD) end
            cmd:SetButtons(btn)
            cmd:SetViewAngles(Angle(0, 0, 0))
          end)
          bot:Spawn()
          bot:SetPos(Vector(b.x - 150, b.y, b.surf + 40))
          bot:SetEyeAngles(Angle(0, 0, 0))
          local function sample(secs)
            local rows, n = {}, 0
            local fin = false
            hook.Add("Tick", "WaterLiveBotRec", function()
              n = n + 1
              local p, v = bot:GetPos(), bot:GetVelocity()
              rows[#rows + 1] = { n * out.dt, p.x * HU_M, p.z * HU_M, v.x * HU_M, v.z * HU_M }
              if n * out.dt >= secs then fin = true; hook.Remove("Tick", "WaterLiveBotRec") end
            end)
            while not fin do sleep(0.2) end
            return rows
          end
          local rest = sample(30)                                       -- dropped in, nobody pressing anything
          out.swim.rest = { feet_z_start = rest[1][3], feet_z_end = rest[#rest][3], vz_end = rest[#rest][5], surf_m = b.surf * HU_M, floor_m = b.floor * HU_M }
          keys.forward = true
          bot:SetPos(Vector(b.x - 200, b.y, b.floor + 10))
          local fwd = sample(5)
          keys.forward = false
          local tail = {}
          for i = math.floor(#fwd * 0.6), #fwd do tail[#tail + 1] = fwd[i][4] end
          local s = 0
          for _, v in ipairs(tail) do s = s + v end
          out.swim.forward = { vx_mean = s / #tail, start = { fwd[1][2], fwd[1][3] }, end_ = { fwd[#fwd][2], fwd[#fwd][3] } }
          keys.jump = true
          local up = sample(4)
          keys.jump = false
          out.swim.jump = { feet_z_start = up[1][3], feet_z_end = up[#up][3], vz_max = math.max(unpack((function() local t = {} for i, r in ipairs(up) do t[i] = r[5] end return t end)())) }
          hook.Remove("StartCommand", "WaterLiveBot")
          bot:Kick("water_live done")
        end
      end
    end

    clear()
    done(out)
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
