-- avatar/journal.lua -- Skippy's notebook: what each player actually did, one file per player, on the server only.
--
--   data/mentat/journal/<steamid64>.txt   one JSON event per line, newest last; local to this machine, never sent anywhere as it is. (.txt, not .jsonl: the engine
--   silently refuses to write any extension outside its short list; found in the first live pass.)
--   When a file passes MAX_BYTES it is renamed <steamid64>.1.txt (replacing the previous .1) and a fresh one starts, so a player holds at most
--   two files of at most MAX_BYTES each; events older than KEEP_DAYS are dropped when read and when rotating. Events are small (MAX_EVENT bytes).
--   What leaves the machine is NOT this file: notes.lua summarises it into a few anonymous strings (see AVATAR.md, "What leaves the machine").
--   Off switches: the convar mentat_skippy_journal 0 (nothing is recorded, nothing is read), and each player's own opt-out (director.lua keeps it).
--   Forget: Journal.forget(sid, scope) deletes the files (all), keeps only events before today (today), or drops the last event (that).
if not SERVER then return end

local J = {}
J.DIR = "mentat/journal"
J.MAX_BYTES = 96 * 1024
J.MAX_EVENT = 320
J.KEEP_DAYS = 45
J.MAX_READ_EVENTS = 1500

local function sid_ok(sid) return type(sid) == "string" and #sid <= 20 and sid:match("^%d+$") ~= nil end
local function path(sid, old) return J.DIR .. "/" .. sid .. (old and ".1" or "") .. ".txt" end

local KINDS = { add = true, remove = true, undo = true, clear = true, beam_ok = true, beam_fail = true, water = true, gravity = true, splash = true,
                fall_water = true, session_start = true, session_end = true, name_change = true, roast = true, joke = true }
J.KINDS = KINDS

local function clean_str(v, n)
  v = tostring(v or ""):gsub("[%c\"\\]", " "):gsub("%s+", " ")
  return v:sub(1, n or 40)
end

--- Keep only fields we know, with sizes bounded, so a journal line is always small and plain.
local function sanitize(ev, now)
  if type(ev) ~= "table" or not KINDS[ev.k] then return nil end
  local out = { t = math.floor(tonumber(ev.t) or now), k = ev.k }
  if ev.name then out.name = clean_str(ev.name, 28) end
  if ev.n then out.n = math.max(0, math.min(9999, math.floor(tonumber(ev.n) or 0))) end
  for _, f in ipairs({ "why", "cmd", "preset", "liquid", "tag", "src", "level" }) do
    if ev[f] then out[f] = clean_str(ev[f], 24) end
  end
  return out
end

local function read_file(p)
  local raw = file.Read(p, "DATA")
  if not raw or raw == "" then return {} end
  local out = {}
  for line in raw:gmatch("[^\n]+") do
    local t = util.JSONToTable(line)
    if type(t) == "table" and KINDS[t.k] and type(t.t) == "number" then out[#out + 1] = t end
  end
  return out
end

--- Append one event for this player. Returns true when written. `now` is epoch seconds (default os.time()).
function J.log(sid, ev, now)
  if not sid_ok(sid) then return false end
  now = now or os.time()
  local e = sanitize(ev, now)
  if not e then return false end
  local line = util.TableToJSON(e)
  if not line or #line > J.MAX_EVENT then return false end
  file.CreateDir(J.DIR)
  local p = path(sid)
  local size = file.Size and file.Size(p, "DATA") or 0
  if size and size >= J.MAX_BYTES then
    if file.Exists(path(sid, true), "DATA") then file.Delete(path(sid, true)) end
    if file.Rename then file.Rename(p, path(sid, true)) else
      local raw = file.Read(p, "DATA") or ""
      file.Write(path(sid, true), raw)
      file.Delete(p)
    end
  end
  if file.Append then file.Append(p, line .. "\n") else file.Write(p, (file.Read(p, "DATA") or "") .. line .. "\n") end
  return true
end

--- This player's events, oldest first (older than KEEP_DAYS dropped, at most MAX_READ_EVENTS newest).
function J.read(sid, now)
  if not sid_ok(sid) then return {} end
  now = now or os.time()
  local out = {}
  local cut = now - J.KEEP_DAYS * 86400
  for _, old in ipairs({ true, false }) do
    for _, e in ipairs(read_file(path(sid, old))) do if e.t >= cut then out[#out + 1] = e end end
  end
  if #out > J.MAX_READ_EVENTS then
    local trimmed = {}
    for i = #out - J.MAX_READ_EVENTS + 1, #out do trimmed[#trimmed + 1] = out[i] end
    out = trimmed
  end
  return out
end

local function rewrite(sid, events)
  file.Delete(path(sid, true))
  if #events == 0 then file.Delete(path(sid)); return end
  local lines = {}
  for _, e in ipairs(events) do lines[#lines + 1] = util.TableToJSON(e) end
  file.CreateDir(J.DIR)
  file.Write(path(sid), table.concat(lines, "\n") .. "\n")
end

--- Forget. scope: "all" (delete the files), "today" (keep what happened before local midnight), "that" (drop the last event).
--- Returns how many events were removed. `date_fn` (default os.date) decides what "today" is.
function J.forget(sid, scope, now, date_fn)
  if not sid_ok(sid) then return 0 end
  now = now or os.time()
  date_fn = date_fn or os.date
  local events = J.read(sid, now)
  local before = #events
  if scope == "all" then
    file.Delete(path(sid)); file.Delete(path(sid, true))
    return before
  end
  local keep = {}
  if scope == "today" then
    local today = date_fn("%Y-%m-%d", now)
    for _, e in ipairs(events) do if date_fn("%Y-%m-%d", e.t) ~= today then keep[#keep + 1] = e end end
  else                                                                -- "that": the last thing (a splash or a joke is not "that": the last action)
    local drop
    for i = #events, 1, -1 do
      local k = events[i].k
      if k == "add" or k == "beam_ok" or k == "beam_fail" or k == "water" or k == "gravity" or k == "undo" or k == "remove" or k == "clear" then drop = i; break end
    end
    for i, e in ipairs(events) do if i ~= drop then keep[#keep + 1] = e end end
  end
  rewrite(sid, keep)
  return before - #keep
end

--- Delete every trace (used by "forget everything" and when a player opts out of notes).
function J.delete_all(sid) return J.forget(sid, "all") end

return J
