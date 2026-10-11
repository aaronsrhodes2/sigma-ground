-- skippy_play.lua -- the launcher's last step (stage/gmod/holodeck/play.py): point the running game at Skippy's bridge, turn on his
-- persona and voice, set the key that talks to Skippy alone, and tell the player how to use it. Armed through data/mentat/tool.txt
-- like every tool, so it only ever runs when whoever starts the game put the file there.
-- args: { bridge, token, key = "x", debug = false }
return function(args, done, ply)
  local H = Holodeck
  if not (IsValid(ply) and H) then return done(nil, "needs the game client and the holodeck") end
  local key = tostring(args.key or "x"):gsub("[^%w_]", ""):sub(1, 16)
  RunConsoleCommand("holodeck_bridge", args.bridge or "")
  RunConsoleCommand("holodeck_bridge_token", args.token or "")
  RunConsoleCommand("holodeck_persona", "1")
  RunConsoleCommand("holodeck_voice_out", "1")
  RunConsoleCommand("holodeck_agent", "1")
  ply:ConCommand("skippy_voice 1")
  if args.debug then ply:ConCommand("skippy_debug 1") end
  -- the key that talks to Skippy alone (no wake word, other players hear nothing); the voice key (V) still works with "Skippy, ...".
  -- The server may not bind keys for a player ("ConCommand is blocked! (bind)"), so the client reads this key itself.
  ply:ConCommand("skippy_key " .. key)
  timer.Simple(0.5, function()
    if not IsValid(ply) then return end
    H.Tell(ply, "Skippy is here. Hold " .. key:upper() .. " and say what you want (\"add three couches\"), or hold V and start with \"Skippy\".")
    done({ ok = true, bridge = args.bridge, key = key })
  end)
end
