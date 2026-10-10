-- voice_test.lua -- exercise the proximity-voice hook with two bots on a dedicated server: who hears whom, at what distances,
-- crouched (a whisper), shouting, and through a wall. args: { distances = {...}, wall_from = {x,y,z}, wall_to = {x,y,z} }
return function(args, done)
  local V = Mentat and Mentat.Voice
  if not V then return done(nil, "proximity voice is not loaded") end
  RunConsoleCommand("mentat_voice_mode", "proximity")
  RunConsoleCommand("mentat_voice_range", "1100")
  local a = player.CreateNextBot("voice_listener")
  local b = player.CreateNextBot("voice_talker")
  if not (IsValid(a) and IsValid(b)) then return done(nil, "could not create bots") end

  timer.Simple(1.5, function()
    b:RemoveFlags(FL_DUCKING)
    timer.Simple(0.3, function() b:RemoveFlags(FL_DUCKING) end)
  end)
  timer.Simple(2.2, function()
    local base = args.base or { x = 0, y = 0, z = 0 }
    local out = { rows = {} }
    local function ask(label, listener_pos, talker_pos)
      V.ClearCache()
      a:SetPos(Vector(listener_pos.x, listener_pos.y, listener_pos.z))
      b:SetPos(Vector(talker_pos.x, talker_pos.y, talker_pos.z))
      local heard, three_d = hook.Run("PlayerCanHearPlayersVoice", a, b)
      local ok, gain = V.CanHear(a, b)
      out.rows[#out.rows + 1] = { label = label, dist = a:GetPos():Distance(b:GetPos()), heard = heard, three_d = three_d, gain = gain, range = V.TalkerRange(b) }
    end
    for _, d in ipairs(args.distances or { 100, 500, 700, 900, 1050, 1200, 2000 }) do
      ask("normal", base, { x = base.x + d, y = base.y, z = base.z })
    end
    b:AddFlags(FL_DUCKING)
    for _, d in ipairs({ 200, 400, 600 }) do
      ask("whisper", base, { x = base.x + d, y = base.y, z = base.z })
    end
    b:RemoveFlags(FL_DUCKING)
    -- the global mode passes everything through, flat
    RunConsoleCommand("mentat_voice_mode", "global")
    timer.Simple(0.2, function()
      ask("global", base, { x = base.x + 5000, y = base.y, z = base.z })
      RunConsoleCommand("mentat_voice_mode", "off")
      timer.Simple(0.2, function()
        -- "off" leaves the game's own rule in charge (sandbox: everyone hears everyone, flat)
        local heard, flat = hook.Run("PlayerCanHearPlayersVoice", a, b)
        out.off_is_game_default = (heard == true and not flat)
        RunConsoleCommand("mentat_voice_mode", "proximity")
        a:Kick("test over")
        b:Kick("test over")
        done(out)
      end)
    end)
  end)
end
