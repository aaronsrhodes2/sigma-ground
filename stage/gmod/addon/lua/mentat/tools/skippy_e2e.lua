-- skippy_e2e.lua -- the player-side voice path in the real game (a listen server): the microphone bridge's file is picked up by
-- the client, sent to the server as a request, built, answered by the agent persona, spoken by the bridge, delivered to the
-- client and PLAYED through the game's own sound system. args: { bridge, token, request }
return function(args, done, ply)
  local H = Holodeck
  if not (IsValid(ply) and H) then return done(nil, "needs the game client and the holodeck") end
  RunConsoleCommand("holodeck_bridge", args.bridge)
  RunConsoleCommand("holodeck_bridge_token", args.token or "")
  RunConsoleCommand("holodeck_persona", "1")
  RunConsoleCommand("holodeck_voice_out", "1")
  ply:ConCommand("skippy_debug 1")
  ply:GodEnable()
  local out = { steps = {} }
  file.CreateDir("mentat")
  file.Delete("mentat/skippy_played.txt")
  file.Delete("mentat/voice_in.json")

  timer.Simple(1.5, function()
    local before = #H.order
    -- exactly what skippy_ears.py writes after it has recognised "Skippy, add two couches"
    file.Write("mentat/voice_in.json", util.TableToJSON({ text = args.request or "add two couches", heard = "Skippy, " .. (args.request or "add two couches"), direct = false, ts = os.time() }))
    out.wrote_voice_in = true
    local waited = 0
    local function poll()
      waited = waited + 0.5
      out.consumed = not file.Exists("mentat/voice_in.json", "DATA")
      out.built = #H.order - before
      out.said = H.last_say
      out.audio = H.last_audio
      out.voice = H.last_voice
      out.speak_error = H.last_speak_error
      out.played = file.Exists("mentat/skippy_played.txt", "DATA") and file.Read("mentat/skippy_played.txt", "DATA") or nil
      if (out.built > 0 and out.played) or waited > 40 then
        ply:ConCommand("skippy_debug 0")
        return done(out)
      end
      timer.Simple(0.5, poll)
    end
    poll()
  end)
end
