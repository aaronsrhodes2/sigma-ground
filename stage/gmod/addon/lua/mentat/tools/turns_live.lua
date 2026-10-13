-- turns_live.lua -- the live pass for Skippy's solid turns (A7b) on the DEDICATED server: no window, no mouse, no human. Two helper bots play two players (Alice, the
-- caller, and Bob, standing close to her); every line goes through the REAL chat hook (PlayerSay) or the same Holodeck.Handle entry the voice path uses, the REAL
-- bridge answers (its model route: the frontier model when a key is reachable, else the local one), and the decisions come back as one table.
-- args: { bridge = "http://127.0.0.1:8777", token = "...", captain_file = true }   (a server with maxplayers >= 4: GMOD_DS_MAXPLAYERS=4)
return function(args, done)
  local H, Av = Holodeck, Mentat and Mentat.Avatar
  local Tn, Dr = Mentat and Mentat.Turns, Mentat and Mentat.Director
  if not (H and Av and Tn and Dr) then return done(nil, "holodeck, avatar, turns or director not loaded") end
  RunConsoleCommand("holodeck_bridge", args.bridge)
  RunConsoleCommand("holodeck_bridge_token", args.token)
  RunConsoleCommand("holodeck_agent", "1")
  RunConsoleCommand("holodeck_persona", "0")
  RunConsoleCommand("holodeck_voice_out", "1")
  RunConsoleCommand("holodeck_bridge_per_min", "600")
  RunConsoleCommand("holodeck_voice_chars_per_min", "100000")
  RunConsoleCommand("holodeck_cooldown", "0")
  RunConsoleCommand("mentat_skippy_chat", "chatty")
  RunConsoleCommand("mentat_skippy_tiebreak", args.tiebreak and "1" or "0")
  RunConsoleCommand("mentat_skippy_debug", "1")

  local out = { scenes = {}, said = {} }
  hook.Add("HolodeckSaid", "turns_live", function(ply, text) out.said[#out.said + 1] = { t = CurTime(), to = IsValid(ply) and ply:Nick() or "?", text = text } end)

  local co
  local function sleep(t)
    timer.Simple(t, function()
      local ok, e = coroutine.resume(co)
      if not ok then done(nil, "script: " .. tostring(e)) end
    end)
    coroutine.yield()
  end
  local function floor_at(x, y)
    local tr = util.TraceLine({ start = Vector(x, y, 3000), endpos = Vector(x, y, -30000), mask = MASK_SOLID_BRUSHONLY })
    return tr.HitPos.z
  end
  local function stand(p, x, y, yaw)
    p:SetPos(Vector(x, y, floor_at(x, y) + 1))
    p:SetEyeAngles(Angle(0, yaw or 0, 0))
    p:SetVelocity(-p:GetVelocity())
  end
  local function face(p, target)
    local d = target - p:GetPos()
    p:SetEyeAngles(Angle(0, math.deg(math.atan2(d.y, d.x)), 0))
  end
  local function things() return #H.order end

  --- One scripted line. how: "chat" (PlayerSay), "voice" (the voice path's entry after the wake word), "ptt" (push-to-talk, no wake word), "console".
  local function say(who, text, how, wait)
    local n0, ring0, said0, talk0 = things(), #Tn.ring, #out.said, Tn.counts.talk
    if how == "voice" then H.Handle(who, text:gsub("^%S+[,%s]+", ""), { src = "voice", woke = true, heard = text })
    elseif how == "ptt" then H.Handle(who, text, { src = "ptt", woke = false, heard = text })
    elseif how == "console" then H.Handle(who, text)
    else hook.Run("PlayerSay", who, text) end
    sleep(wait or 6)
    local ring = {}
    for i = ring0 + 1, #Tn.ring do local e = Tn.ring[i]; ring[#ring + 1] = { addressed = e.addressed, score = e.score, route = e.route, final = e.final, src = e.src, why = e.why } end
    local heard = {}
    for i = said0 + 1, #out.said do heard[#heard + 1] = out.said[i].text end
    return { by = who:Nick(), how = how or "chat", text = text, decisions = ring, built = things() - n0, skippy_said = heard, last_turn = (Tn.counts.talk > talk0 and Tn.last_turn) and { source = Tn.last_turn.source, tries = Tn.last_turn.tries, why = Tn.last_turn.why } or nil }
  end
  local function scene(name, r) r.scene = name; out.scenes[#out.scenes + 1] = r end

  co = coroutine.create(function()
    sleep(1)
    local alice, bob = player.CreateNextBot("Alice"), player.CreateNextBot("Bob")
    if not (IsValid(alice) and IsValid(bob)) then return done(nil, "no helper bots") end
    sleep(2)
    out.sids = { alice = alice:SteamID64(), bob = bob:SteamID64() }
    stand(alice, 0, 0, 0)
    stand(bob, 160, 60, 180)
    -- the captains' file: Alice is a captain, Bob is not (written before anyone's roast level is read)
    if args.captain_file then file.CreateDir("mentat"); file.Write("mentat/captain_sids.txt", "# the captains\n" .. alice:SteamID64() .. "\n") end
    out.roast = { alice = Dr.RoastLevel(alice), bob = Dr.RoastLevel(bob), file_set = Dr.CaptainSids()[alice:SteamID64()] == true }
    sleep(1)
    out.model_route = Tn.ModelOk()

    -- 1. called by name, then sign of life
    scene("summon", say(alice, "Skippy, come here", "chat", 8))
    out.avatar_active = Av.IsActive()
    local bot = Av.Get()
    if bot then face(alice, bot:GetPos()) end
    sleep(25)                                                      -- (his arrival line opened the window: let it close)
    -- 2. addressed: a command (parser), a question (the model route), a vague wish (the planner agent, if it can be reached)
    scene("named_command", say(alice, "Skippy, add a couch", "chat", 4))
    scene("named_question", say(alice, "Skippy, why do things float in water", "chat", 12))
    -- 3. a follow-up inside the window needs no name (he just answered Alice, who is looking at him)
    bot = Av.Get(); if bot then face(alice, bot:GetPos()) end
    scene("follow_up_in_window", say(alice, "and why do some things sink then", "chat", 12))
    -- 4. not for him
    sleep(22)
    scene("chatter_bob_to_alice", say(bob, "did you see that jump", "chat", 1))
    scene("chatter_alice_to_bob", say(alice, "lol nice one", "chat", 1))
    scene("order_for_bob", say(alice, "Bob, add a couch", "chat", 2))
    scene("his_name_as_topic", say(alice, "Skippy is silly", "chat", 2))
    scene("follow_up_after_window", say(alice, "and why do some things sink then", "chat", 2))
    scene("order_without_name", say(alice, "add a lamp", "chat", 2))
    -- 4b. no name, no window, standing next to him and looking at him: only maybe. Silent by default; with the tiebreak on, the small model decides
    bot = Av.Get(); if bot then face(alice, bot:GetPos()) end
    scene("maybe_then_tiebreak", say(alice, "what is that hat made of", "chat", 10))
    -- 5. push-to-talk and the console are always for him
    scene("push_to_talk", say(alice, "what can you build", "ptt", 12))
    scene("voice_with_name", say(alice, "Skippy what is two plus two", "voice", 12))
    scene("console_command", say(alice, "add a lamp", "console", 3))
    -- 6. a bare name is a call
    scene("bare_name", say(alice, "Skippy", "chat", 10))
    -- 7. the model's invalid output (the driver makes the bridge's model misbehave for this sentence) falls back to a line
    scene("invalid_model_output", say(alice, "Skippy, tell me about INVALIDOUTPUT please", "chat", 12))
    -- 8. the same question again: cached, instant, free
    scene("repeat_question_cached", say(alice, "Skippy, why do things float in water", "chat", 6))
    -- 9. the captain gets a savage barb; Bob (gentle) does not
    scene("question_as_captain", say(alice, "Skippy, how tall is a couch", "chat", 12))
    scene("question_as_stranger", say(bob, "Skippy, how tall is a lamp", "chat", 12))
    -- 10. the sign-off
    bot = Av.Get(); if bot then face(alice, bot:GetPos()) end
    scene("go_away", say(alice, "Skippy, go away", "chat", 14))
    out.avatar_active_after = Av.IsActive()
    out.counts = Tn.counts
    out.ring_size = #Tn.ring
    out.things_standing = things()
    done(out)
  end)
  coroutine.resume(co)
end
