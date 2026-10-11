-- foundprog.lua -- SERVER. The progress of a found object that is still on its way (the data the transporter beam shows), the heartbeat
-- that tells "working" from "stalled", and what happens when the pipeline gives up. Loaded by meshnet.lua; see
-- stage/gmod/holodeck/README.md ("The transporter effect").
--
--   Every pending object of one model (they share one download) is told the same thing, by REAL events only:
--     the bridge accepted the request            -> "fetching" 0
--     the bridge's /progress poll (every second)  -> "fetching" bytes / bytes expected; "packing" (a heartbeat while the worker lives)
--     the bridge's /mesh answer, checked          -> "sending" 0
--     each chunk sent                             -> "sending" chunks / chunks
--     the last chunk sent                         -> "decoding", then "done"
--   Progress only rises (ENT:FxSet). Each event stamps ENT:GetFxBeat; a client that sees no new beat for N seconds destabilises the
--   beam (convar holodeck_found_stall, replicated). A pending object that has not arrived `holodeck_found_timeout` seconds after the
--   request is given up on, as is any object whose download, check or decode failed: the beam collapses red on every client for about
--   1.5 s, then the placeholder is removed and Skippy says why to whoever asked for it.
local TM = include("mentat/fx/transporter_model.lua")

local FP = {}
FP.TIMEOUT = 75                 -- seconds from the request to the hard failure (the bridge's own deadline for /mesh is 44 s, the game's 60)
FP.POLL_EVERY = 0.9
FP.PACK_GRACE = 25              -- seconds in "packing" that still count as a living worker
FP.REMOVE_AFTER = TM.FAIL_S + 0.2
FP.DONE_AFTER = 1.2             -- from "decoding" to "done"
FP.watch = {}                   -- model record -> true while its request is in flight
FP.Mesh = nil
FP.deps = {}

local cv_timeout = CreateConVar("holodeck_found_timeout", tostring(FP.TIMEOUT), FCVAR_ARCHIVE, "seconds a found object may stay on its way before it is given up on (the beam collapses and the placeholder is removed)")
local cv_stall = CreateConVar("holodeck_found_stall", tostring(TM.STALL_S), bit.bor(FCVAR_ARCHIVE, FCVAR_REPLICATED), "seconds without a progress event before the transporter beam visibly destabilises")
FP.cv_timeout, FP.cv_stall = cv_timeout, cv_stall

local function now() return CurTime() end

--- deps = { mesh = <meshnet module>, quiet_post = fn(path, tbl, ok, fail, timeout), say = fn(ply, text) }
function FP.init(deps)
  FP.deps = deps or {}
  FP.Mesh = deps.mesh
end

-- ------------------------------------------------------------------------------------------------ telling the entities
--- Apply a real event to every live entity of a model, and remember it on the record.
function FP.stage(m, stage, frac, heartbeat)
  local any = false
  for _, e in ipairs(m.ents) do
    if IsValid(e) and e.FxSet and e:FxSet(stage, frac, heartbeat) then any = true end
  end
  m.fx_stage, m.fx_frac = stage, frac
  return any
end

--- An entity joins a model that may already be on its way: it shows the model's state, not zero.
function FP.join(m, ent, ply)
  ent.FxOwner = ply
  if not ent.FxSet then return end
  local st = m.state
  if st == "ready" then
    ent:FxSet("decoding", 0)                                  -- the clients that have the shape draw it at once, with the materialise
    timer.Simple(FP.DONE_AFTER, function() if IsValid(ent) and ent:FxStageName() == "decoding" then ent:FxSet("done", 1) end end)
  elseif st == "pending" then
    ent:FxSet(m.fx_stage or "fetching", m.fx_frac or 0)
  elseif st == "failed" then
    FP.give_up_entities({ ent }, m.why or "the shape could not be fetched")
  end
end

-- ------------------------------------------------------------------------------------------------ giving up
local function nice(why)
  why = tostring(why or ""):lower()
  if why:find("http 429", 1, true) or why:find("rationed", 1, true) then return "the bridge is rationing me just now" end
  if why:find("http 503", 1, true) then return "the download slot was busy" end
  if why:find("http 4", 1, true) or why:find("http 5", 1, true) then return "the bridge could not fetch it" end
  if why:find("timed out", 1, true) or why:find("timeout", 1, true) then return "it took too long and I lost the lock" end
  if why:find("cap", 1, true) or why:find("too big", 1, true) or why:find("over the", 1, true) then return "the model is too big for me to bring in" end
  if why:find("checksum", 1, true) or why:find("lzma", 1, true) or why:find("decompress", 1, true) or why:find("base64", 1, true) or why:find("differs", 1, true) then
    return "the model file was damaged on the way"
  end
  if why:find("not a mesh", 1, true) or why:find("answer was not", 1, true) then return "the bridge sent me something that was not a model" end
  if why:find("engine refused", 1, true) then return "the game could not reach the bridge" end
  return "it did not arrive (" .. why:sub(1, 60) .. ")"
end
FP.nice = nice

--- The sentence Skippy says when objects of one kind were given up on.
function FP.sentence(name, count, why)
  name = tostring(name or "object")
  local what = count > 1 and (count .. " of the " .. name) or ("the " .. name)
  return "I could not bring in " .. what .. ": " .. nice(why) .. ". I took the placeholder down."
end

--- Fail these entities now (the beam collapses on every client), remove them after the effect, and say why to their owners.
function FP.give_up_entities(list, why)
  local by_owner, order = {}, {}
  for _, e in ipairs(list) do
    if IsValid(e) then
      if e.FxSet then e:FxSet("failed", 0) end
      local o = e.FxOwner
      local key = o or "none"
      if not by_owner[key] then
        by_owner[key] = { ply = o, name = e.GetFoundName and e:GetFoundName() or "object", n = 0, pos = e:GetPos() }
        order[#order + 1] = key
      end
      by_owner[key].n = by_owner[key].n + 1
      timer.Simple(FP.REMOVE_AFTER, function() if IsValid(e) then e:Remove() end end)
    end
  end
  FP.last_why = tostring(why)
  for _, key in ipairs(order) do
    local g = by_owner[key]
    FP.last_say = FP.sentence(g.name, g.n, why)
    if FP.deps.say then FP.deps.say(g.ply, FP.last_say) end
    if FP.deps.wisp and g.pos then                              -- the same sentence, left hanging at the failed spot once the collapse is over
      local text, ply, pos = FP.last_say, g.ply, g.pos
      timer.Simple(FP.REMOVE_AFTER, function() FP.deps.wisp(ply, text, "failure", Vector(pos.x, pos.y, pos.z + 58)) end)
    end
  end
end

--- The model's request failed (any reason): its entities collapse, then go.
function FP.fail(m, why)
  FP.watch[m] = nil
  local list = m.ents
  m.ents = {}
  FP.give_up_entities(list, why)
end

-- ------------------------------------------------------------------------------------------------ the end of a good transfer
--- Every chunk of the model's broadcast has been sent.
function FP.sent(m)
  FP.watch[m] = nil
  FP.stage(m, "decoding", 0)
  timer.Simple(FP.DONE_AFTER, function()
    for _, e in ipairs(m.ents) do
      if IsValid(e) and e:FxStageName() == "decoding" then e:FxSet("done", 1) end
    end
  end)
end

-- ------------------------------------------------------------------------------------------------ the watch: bridge polls and the hard timeout
--- What the bridge's /progress answer means for the beam. Pure enough to test: returns stage, frac, heartbeat (or nil to say nothing).
function FP.read_progress(obj)
  if type(obj) ~= "table" or obj.ok ~= true then return nil end
  local stage = obj.stage
  local bytes, total = tonumber(obj.bytes) or 0, tonumber(obj.total) or 0
  if stage == "queued" then return "fetching", 0, false end                          -- waiting in line is not progress
  if stage == "fetching" then
    local frac
    if total > 0 then frac = math.min(1, bytes / total) elseif bytes > 0 then frac = 1 - math.exp(-bytes / 1500000) else frac = 0 end
    return "fetching", math.min(frac, 0.999), false
  end
  if stage == "packing" then
    local age = tonumber(obj.age_s) or 0
    return "packing", 0, age < FP.PACK_GRACE                                          -- the worker is alive and in this stage
  end
  return nil
end

local function poll(m)
  local d = FP.deps
  if m.polling or not d.quiet_post then return end
  m.polling = true
  d.quiet_post("/progress", { key = m.key }, function(body)
    m.polling = false
    if m.state ~= "pending" then return end
    local obj = util.JSONToTable(body or "")
    local stage, frac, hb = FP.read_progress(obj)
    if stage then FP.stage(m, stage, frac, hb) end
  end, function() m.polling = false end, 4)
end

local function tick()
  local t = now()
  local n = 0
  for m in pairs(FP.watch) do
    n = n + 1
    if m.state ~= "pending" then
      FP.watch[m] = nil
    elseif t - (m.t or t) > cv_timeout:GetFloat() then
      local why = string.format("timed out after %d s", math.floor(t - m.t))
      if FP.Mesh and FP.Mesh.fail then FP.Mesh.fail(m, why) else FP.fail(m, why) end
    elseif t - (m.last_poll or 0) >= FP.POLL_EVERY then
      m.last_poll = t
      poll(m)
    end
  end
  if n == 0 then timer.Remove("holodeck_found_watch") end
end
FP.tick = tick

--- Start watching a model whose request is in flight.
function FP.begin(m)
  FP.watch[m] = true
  m.last_poll = now()
  FP.stage(m, "fetching", 0)
  if not timer.Exists("holodeck_found_watch") then timer.Create("holodeck_found_watch", 0.5, 0, tick) end
end

return FP
