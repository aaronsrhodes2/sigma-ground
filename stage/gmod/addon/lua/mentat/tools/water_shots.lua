-- water_shots.lua -- photograph the construct's water in the real game client (stage/gmod/holodeck/README.md, "Water").
-- args: { scenario = "pool" | "sea" | "moon" | "perf" }
--   pool : "add a pool", floating crates of three densities, a barrel and a ball, a steel crate sinking, a splash and its rings, the camera at
--          a grazing angle, under the surface looking sideways and up, a bot swimming
--   sea  : "add a sea" and "add a pond": the big slab seen from the shore, the round pond
--   moon : the same pool under the Moon's gravity: a splash that throws its drops high and slow
--   perf : milliseconds per frame with the pool in view / out of view / no water at all
-- The camera is pinned each tick by moving the player.  Pictures are taken by Mentat.Shot (shot_wtr_<name>.jpg); the result lists what the
-- water was doing at each (and the render module's own frame cost).
local function diag(ply, code) ply:SendLua(code) end

return function(args, done, ply)
  local H, W = Holodeck, MentatWater
  if not (IsValid(ply) and H and W) then return done(nil, "needs the game client, the holodeck and the water") end
  local scenario = args.scenario or "pool"
  local out = { scenario = scenario, frames = {}, notes = {} }
  ply:GodEnable()
  ply:StripWeapons()
  RunConsoleCommand("holodeck_agent", "0")
  RunConsoleCommand("holodeck_persona", "0")
  RunConsoleCommand("holodeck_voice_out", "0")
  local HU_M = 0.01905

  local co
  local function sleep(t)
    timer.Simple(t, function()
      local ok, e = coroutine.resume(co)
      if not ok then done(nil, "script: " .. tostring(e)) end
    end)
    coroutine.yield()
  end
  local function floor_at(x, y)
    return util.TraceLine({ start = Vector(x, y, 3000), endpos = Vector(x, y, -30000), mask = MASK_SOLID_BRUSHONLY }).HitPos.z
  end
  local t0 = SysTime()
  local pin
  hook.Add("Tick", "wtr_pin", function()
    if pin and IsValid(ply) then ply:SetPos(pin.pos); ply:SetVelocity(-ply:GetVelocity()); ply:SetEyeAngles(pin.ang) end
  end)
  --- put the eye at `eye` looking at `target` (the player stands 64 HU below the eye)
  local function look(eye, target)
    pin = { pos = eye - Vector(0, 0, 64), ang = (target - eye):Angle() }
    ply:SetPos(pin.pos)
  end
  local function shot(name, info)
    local row = { name = name, t = SysTime() - t0, info = info or {} }
    row.water = { bodies = #W.entries(), splashes = W.counters.splashes, woken = W.counters.woken, slept = W.counters.slept }
    out.frames[#out.frames + 1] = row
    Mentat.Shot("wtr_" .. name)
    sleep(0.7)
  end
  local function ask(text)
    ply:SetEyeAngles(Angle(0, 0, 0))
    local r = H.Handle(ply, text)
    sleep(3.5)
    return r
  end
  local function body() return W.volumes.list()[1] end
  local function crate(rho, x, y, z, model)
    model = model or "models/props_junk/wood_crate001a.mdl"
    local e = ents.Create("prop_physics")
    e:SetModel(model)
    e:SetPos(Vector(x, y, z))
    e:Spawn(); e:Activate()
    local po = e:GetPhysicsObject()
    if not IsValid(po) then e:Remove(); return nil end
    local mn, mx = e:OBBMins(), e:OBBMaxs()
    local v = (mx.x - mn.x) * (mx.y - mn.y) * (mx.z - mn.z) * HU_M ^ 3
    if model:find("sphere") then v = v * math.pi / 6 end
    if model:find("oildrum") then v = v * math.pi / 4 end
    po:SetMass(rho * v)
    po:Wake()
    return e
  end
  local function reset()
    H.Handle(ply, "clear everything")
    sleep(2.5)
    for _, e in ipairs(ents.FindByClass("prop_physics")) do if IsValid(e) then e:Remove() end end
    for _, e in ipairs(ents.FindByClass("holodeck_found")) do if IsValid(e) then e:Remove() end end
  end

  co = coroutine.create(function()
    local fz = floor_at(0, 0)
    ply:SetPos(Vector(0, 0, fz + 1))
    sleep(1)
    reset()

    if scenario == "pool" then
      ask("add a pool in front of me")
      local b = body()
      if not b then out.notes[#out.notes + 1] = "no pool"; return done(out) end
      local sx, sy = b.x, b.y
      local surf = b.surf
      out.pool = { x = sx, y = sy, surf_m = surf * HU_M, floor_m = b.floor * HU_M, g = W.gravity() }
      -- overview from the deck
      look(Vector(sx - 330, sy - 380, surf + 330), Vector(sx, sy, surf))
      sleep(1.0); shot("p1_overview")
      -- floating things of three densities, a barrel and a ball
      local items = {}
      local spec = { { 250, -150 }, { 600, -60 }, { 900, 30 } }
      for _, s in ipairs(spec) do items[#items + 1] = crate(s[1], sx + s[2], sy - 40, surf + 60) end
      items[#items + 1] = crate(500, sx + 130, sy - 40, surf + 60, "models/props_c17/oildrum001.mdl")
      items[#items + 1] = crate(400, sx - 150, sy + 60, surf + 60, "models/hunter/misc/sphere025x025.mdl")
      sleep(2.0); shot("p2_dropping")
      sleep(14)
      look(Vector(sx - 40, sy - 330, surf + 40), Vector(sx - 40, sy, surf))
      sleep(0.8); shot("p3_floating_side")
      out.floaters = {}
      for i, e in ipairs(items) do
        if IsValid(e) then
          local po = e:GetPhysicsObject()
          out.floaters[#out.floaters + 1] = { i = i, mass = IsValid(po) and po:GetMass() or -1, bottom_m = (e:GetPos().z + e:OBBMins().z) * HU_M, surf_m = surf * HU_M, asleep = IsValid(po) and po:IsAsleep() or false }
        end
      end
      -- grazing: the mirror
      look(Vector(sx - 330, sy, surf + 8), Vector(sx + 330, sy, surf))
      sleep(0.8); shot("p4_grazing")
      look(Vector(sx - 90, sy - 70, surf + 330), Vector(sx - 90, sy - 70, surf))
      sleep(0.8); shot("p5_straight_down")
      -- a steel crate sinks
      local steel = crate(7850, sx + 60, sy + 60, surf + 40)
      look(Vector(sx - 300, sy + 60, surf + 260), Vector(sx + 60, sy + 60, surf - 50))
      sleep(0.6); shot("p6_steel_entering")
      sleep(0.5); shot("p7_steel_sinking")
      -- a big splash and its rings
      for _, e in ipairs(items) do if IsValid(e) then e:Remove() end end
      if IsValid(steel) then steel:Remove() end
      local boulder = crate(2600, sx, sy, surf + 520)
      look(Vector(sx - 420, sy - 240, surf + 160), Vector(sx, sy, surf + 60))
      sleep(1.2); shot("p8_boulder_falling")
      sleep(0.35); shot("p9_splash_a")
      sleep(0.45); shot("p10_splash_b")
      sleep(1.2); shot("p11_rings")
      if IsValid(boulder) then boulder:Remove() end
      -- under the surface
      look(Vector(sx - 150, sy, surf - 80), Vector(sx + 400, sy, surf - 60))
      sleep(0.8); shot("p12_underwater_side")
      look(Vector(sx, sy, surf - 90), Vector(sx + 120, sy + 30, surf + 200))
      sleep(0.8); shot("p13_underwater_up")
      -- a bot swims
      RunConsoleCommand("bot")
      sleep(2)
      local bot
      for _, p in ipairs(player.GetAll()) do if p:IsBot() then bot = p end end
      if bot then
        local keys = { forward = true }
        hook.Add("StartCommand", "wtr_bot", function(p, cmd)
          if p ~= bot then return end
          cmd:ClearMovement(); cmd:ClearButtons()
          if keys.forward then cmd:SetForwardMove(10000); cmd:SetButtons(IN_FORWARD) end
          cmd:SetViewAngles(Angle(0, 0, 0))
        end)
        bot:Spawn()
        bot:SetPos(Vector(sx - 200, sy, surf + 20))
        look(Vector(sx - 60, sy - 330, surf + 220), Vector(sx - 60, sy, surf))
        sleep(2.5); shot("p14_bot_swimming_a")
        sleep(2.0); shot("p15_bot_swimming_b")
        hook.Remove("StartCommand", "wtr_bot")
        out.bot = { feet_m = bot:GetPos().z * HU_M, surf_m = surf * HU_M }
        bot:Kick("water_shots done")
      else
        out.notes[#out.notes + 1] = "no bot"
      end
      pin = nil
      return done(out)
    end

    if scenario == "sea" then
      ask("add a sea in front of me")
      local b = body()
      if b then
        local surf = b.surf
        look(Vector(b.x - b.b.hx - 120, b.y, surf + 90), Vector(b.x, b.y, surf))
        sleep(1.0); shot("s1_sea_from_shore")
        look(Vector(b.x - b.b.hx - 80, b.y - 300, surf + 600), Vector(b.x, b.y, surf))
        sleep(0.8); shot("s2_sea_high")
      end
      reset()
      ask("add a pond in front of me")
      b = body()
      if b then
        local surf = b.surf
        look(Vector(b.x - 380, b.y - 200, surf + 200), Vector(b.x, b.y, surf))
        sleep(1.0); shot("s3_pond")
        crate(300, b.x, b.y, surf + 200)
        sleep(1.2); shot("s4_pond_drop")
        sleep(0.5); shot("s5_pond_ripples")
      end
      pin = nil
      return done(out)
    end

    if scenario == "moon" then
      ask("add a pool in front of me")
      local b = body()
      if not b then return done(out) end
      MentatWorld.Handle(ply, "set gravity to the moon")
      sleep(1.5)
      local surf = b.surf
      out.pool = { g = W.gravity() }
      local c = crate(600, b.x, b.y, surf + 250)
      look(Vector(b.x - 420, b.y - 240, surf + 160), Vector(b.x, b.y, surf + 60))
      sleep(1.6); shot("m1_falling_slowly")
      sleep(0.7); shot("m2_splash_a")
      sleep(0.7); shot("m3_splash_b")
      sleep(1.0); shot("m4_splash_c")
      sleep(8); shot("m5_settled")
      if IsValid(c) then out.draft_m = surf * HU_M - (c:GetPos().z * HU_M) end
      MentatWorld.Handle(ply, "set gravity to earth")
      pin = nil
      return done(out)
    end

    if scenario == "perf" then
      ask("add a pool in front of me")
      local b = body()
      if not b then return done(out) end
      local surf = b.surf
      look(Vector(b.x - 330, b.y - 300, surf + 280), Vector(b.x, b.y, surf))
      sleep(1.0)
      ply:ConCommand("fps_max 0"); ply:ConCommand("mat_vsync 0")
      sleep(1.5)
      file.Delete("mentat/water_perf.txt")
      local function perf(label, secs)
        diag(ply, 'local n, t = 0, 0 hook.Add("Think", "wtrperf", function() n = n + 1 t = t + RealFrameTime() end) timer.Simple(' .. (secs - 0.2)
          .. ', function() hook.Remove("Think", "wtrperf") local R = Mentat.WaterRender or {} local s = R.stats or {} local line = string.format("%s frames=%d avg_ms=%.3f fps=%.1f water_ms=%.4f worst_ms=%.3f tris=%d\\n", "'
          .. label .. '", n, 1000 * t / math.max(n, 1), n / math.max(t, 1e-9), s.draw_ms or -1, s.worst_ms or -1, s.tris or -1) print("[wtr-perf] " .. line) file.Append("mentat/water_perf.txt", line) end)')
        sleep(secs)
      end
      for _, round in ipairs({ "A", "B" }) do
        look(Vector(b.x - 330, b.y - 300, surf + 280), Vector(b.x, b.y, surf)); sleep(0.5)
        perf("pool_in_view_" .. round, 5)
        look(Vector(b.x - 330, b.y - 300, surf + 280), Vector(b.x - 1000, b.y - 300, surf + 280)); sleep(0.5)
        perf("pool_out_of_view_" .. round, 5)
      end
      H.Handle(ply, "remove the pool")
      sleep(3)
      perf("no_water", 5)
      ply:ConCommand("fps_max 300")
      pin = nil
      return done(out)
    end
    done(nil, "unknown scenario " .. tostring(scenario))
  end)
  local ok, e = coroutine.resume(co)
  if not ok then done(nil, "script: " .. tostring(e)) end
end
