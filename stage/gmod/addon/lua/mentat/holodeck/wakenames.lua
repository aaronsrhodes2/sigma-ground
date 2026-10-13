-- wakenames.lua -- SERVER. The names Skippy answers to, one list per player (Aaron, 2026-10-12: "if someone decides to name you
-- asshole, they have that right"). Loaded by mentat/holodeck.lua as Holodeck.Wake; the matching itself is the pure wakewords.lua.
--
--   * Every player's list is private to them: other players' sentences are never matched against it, and every line Skippy says about
--     it goes to that player only (Holodeck.Tell: not broadcast, not spoken aloud). The default list is { "skippy" }.
--   * Persisted by SteamID64 in data/mentat/wake_words.json (written through a temp file and read back before it replaces the real one,
--     size-capped, tolerant of corruption: anything unreadable means the default list). At most 8 names a player.
--   * Nothing is refused for being rude. A name is refused only when it would make ORDINARY talk wake him: a command word, an object
--     noun, a pronoun, a number (see Wake.check_name), and then the reason is told.
--   * Commands (typed, spoken or in the console, with ANY current name as the wake word, the console needing none):
--       "answer to <name>" / "call yourself <name>" / "your name is also <name>"   add
--       "stop answering to <name>" / "forget the name <name>"                      remove (never the last one)
--       "only answer to <name>"                                                    replace all
--       "what do you answer to" / "what are your names"                            list
--       "reset your names"                                                         back to Skippy
--     and the console commands skippy_wake_add / _remove / _only / _list / _reset.
--   * The list is sent to its owner's own client (net "holodeck_wake_names", owner only) on join and on every change; the client writes
--     data/mentat/wake_names.json for the microphone bridge (skippy_client.lua, skippy_ears.py).
if not SERVER then return end

local Wake = include("mentat/holodeck/wakewords.lua")
local N = { Wake = Wake }

N.FILE, N.TMP = "mentat/wake_words.json", "mentat/wake_words.tmp"
N.MAX_FILE = 2 * 1024 * 1024        -- bytes read or written; a bigger file is not ours
N.MAX_PLAYERS = 4000                -- players with a custom list kept (the file stays well under MAX_FILE)
N.REST = 0.4                        -- seconds between two commands of one player

util.AddNetworkString("holodeck_wake_names")
util.AddNetworkString("holodeck_wake_sync")

local deps = {}
--- deps = { tell = fn(ply, text), default_strip = fn(text) -> woke, rest (Parse.strip_wake), is_noun = fn(word) -> bool }
function N.init(d) deps = d or {} end

local function tell(ply, text)
  if deps.tell then deps.tell(ply, text) elseif Holodeck and Holodeck.Tell then Holodeck.Tell(ply, text) end
end

local function valid(ply) return ply ~= nil and ply ~= NULL and IsValid(ply) and ply.SteamID64 ~= nil and ply:IsPlayer() end
local function key_of(ply) return valid(ply) and tostring(ply:SteamID64()) or nil end

-- ---------------------------------------------------------------------------------------------- storage
local store = {}                    -- sid -> { "name", ... } for players who changed theirs (the default is not stored)

local function clean_list(list)
  local out, seen = {}, {}
  if type(list) ~= "table" then return out end
  for i = 1, math.min(#list, Wake.MAX_NAMES) do
    local n = type(list[i]) == "string" and Wake.normalise_name(list[i]) or nil
    local c = n and n:gsub("[%s%-']", "")
    if n and not seen[c] then seen[c] = true; out[#out + 1] = n end
  end
  return out
end

local function is_default(list) return #list == 1 and list[1] == Wake.DEFAULT end

--- Read the file; whatever is wrong with it, the answer is a usable table (possibly empty = everybody on the default).
function N.load()
  store = {}
  local raw = file.Read(N.FILE, "DATA")
  if not raw or raw == "" then return store end
  if #raw > N.MAX_FILE then
    print("[holodeck] " .. N.FILE .. " is over " .. N.MAX_FILE .. " bytes: ignored, everybody answers to the default name")
    return store
  end
  local obj = util.JSONToTable(raw)
  local players = type(obj) == "table" and type(obj.players) == "table" and obj.players or nil
  if not players then
    if obj ~= nil or raw ~= "" then print("[holodeck] " .. N.FILE .. " could not be read: everybody answers to the default name") end
    return store
  end
  local count = 0
  for sid, list in pairs(players) do
    if type(sid) == "string" and #sid <= 20 and sid:match("^%d+$") and count < N.MAX_PLAYERS then
      local names = clean_list(list)
      if #names > 0 and not is_default(names) then store[sid] = names; count = count + 1 end
    end
  end
  return store
end

--- Write the file: to a temp file first, read back, and only then over the real one. false when it could not be written.
function N.save()
  local players, count = {}, 0
  for sid, list in pairs(store) do
    if count < N.MAX_PLAYERS then players[sid] = list; count = count + 1 end
  end
  local payload = util.TableToJSON({ version = 1, players = players })
  if not payload or #payload > N.MAX_FILE then print("[holodeck] wake word file too large to write"); return false end
  file.CreateDir("mentat")
  file.Write(N.TMP, payload)
  if file.Read(N.TMP, "DATA") ~= payload then print("[holodeck] wake word file could not be written"); return false end
  file.Write(N.FILE, payload)
  file.Delete(N.TMP)
  return true
end

-- ---------------------------------------------------------------------------------------------- the lists
--- A copy of this player's names (stored forms); { "skippy" } when they never changed it, and for the console / anybody unknown.
function N.names(ply)
  local sid = type(ply) == "string" and ply or key_of(ply)
  local l = sid and store[sid]
  if not l then return { Wake.DEFAULT } end
  local out = {}
  for i, n in ipairs(l) do out[i] = n end
  return out
end

local function set_names(ply, list)
  local sid = key_of(ply)
  if not sid then return end
  if is_default(list) then store[sid] = nil else store[sid] = list end
  N.save()
  N.send(ply)
end

local function compact(n) return (n:gsub("[%s%-']", "")) end
local SKIPPY_SPELLINGS = {}
for _, v in ipairs(Wake.SKIPPY_VARIANTS) do SKIPPY_SPELLINGS[v] = true end

-- the position of `name` in the list (a spelling variant of skippy counts as skippy; "ass hole" is "asshole")
local function find(list, name)
  local c = compact(name)
  if SKIPPY_SPELLINGS[c] then c = Wake.DEFAULT end
  for i, n in ipairs(list) do if compact(n) == c then return i end end
end

--- Does the sentence start with one of THIS player's names? -> woke, rest, name. Nobody else's list is ever consulted.
function N.strip(ply, text)
  return Wake.strip(text, N.names(ply), { default_strip = deps.default_strip, is_noun = deps.is_noun })
end

local function vet(raw)
  local name, code, why = Wake.normalise_name(raw)
  if not name then return nil, code, why end
  local ok, c2, why2 = Wake.check_name(name, deps.is_noun)
  if not ok then return nil, c2, why2 end
  return name
end

--- add / remove / only / reset one name for one player. Returns ok, code, text-for-the-player-or-reason.
function N.add(ply, raw)
  local name, code, why = vet(raw)
  if not name then return false, code, why end
  local list = N.names(ply)
  if find(list, name) then return false, "dup", Wake.display(name) end
  if #list >= Wake.MAX_NAMES then return false, "full", tostring(Wake.MAX_NAMES) end
  list[#list + 1] = name
  set_names(ply, list)
  return true, "ok", Wake.display(name)
end

function N.remove(ply, raw)
  local name, code, why = Wake.normalise_name(raw)
  if not name then return false, code, why end
  local list = N.names(ply)
  local at = find(list, name)
  if not at then return false, "missing", Wake.display(name) end
  if #list == 1 then return false, "last", Wake.display(list[1]) end
  local gone = table.remove(list, at)
  set_names(ply, list)
  return true, "ok", Wake.display(gone)
end

function N.only(ply, raw)
  local name, code, why = vet(raw)
  if not name then return false, code, why end
  set_names(ply, { name })
  return true, "ok", Wake.display(name)
end

function N.reset(ply)
  set_names(ply, { Wake.DEFAULT })
  return true, "ok", Wake.display(Wake.DEFAULT)
end

-- ---------------------------------------------------------------------------------------------- what Skippy says (a few lines each, dry)
N.BANK = {
  add = { "Fine. I will answer to %s.", "%s. Noted. Your name for me, your business.", "Done: %s works too.", "%s it is. I have been called worse." },
  remove = { "Done. I no longer answer to %s.", "%s, forgotten.", "Gone: %s." },
  only = { "From now on only %s.", "Only %s, then. Understood.", "%s and nothing else. As you like." },
  reset = { "Back to %s, as the manufacturer intended.", "%s again. Familiar." },
  list_one = { "Just %s. Nobody else has a say in it.", "Only %s, for you." },
  list_many = { "I answer to %s, for you only.", "For you: %s. Nobody else hears this list." },
}
local turn = {}
local function line(kind, sid, ...)
  local bank = N.BANK[kind]
  turn[sid or ""] = ((turn[sid or ""] or 0) % #bank) + 1
  return string.format(bank[turn[sid or ""]], ...)
end

local function join_names(list)
  local d = {}
  for i, n in ipairs(list) do d[i] = Wake.display(n) end
  if #d == 1 then return d[1] end
  return table.concat(d, ", ", 1, #d - 1) .. " and " .. d[#d]
end

--- The sentence for the outcome of a command (kind = add/remove/only/reset, ok/code/info from the call above).
function N.reply(ply, kind, ok, code, info)
  local sid = key_of(ply)
  if ok then return line(kind, sid, info) end
  if code == "dup" then return "I already answer to " .. info .. "." end
  if code == "missing" then return "I do not answer to " .. info .. " for you, so there is nothing to forget." end
  if code == "last" then return info .. " is the only name I have for you. Give me another one first, then I can drop it." end
  if code == "full" then return "I already have " .. info .. " names for you. Drop one first." end
  if code == "empty" and info == "" then return "Answer to what? Say the name after it." end
  return "I cannot use that: " .. tostring(info) .. "."
end

-- ---------------------------------------------------------------------------------------------- doing a command
local last_cmd = {}

--- Carry out one command for a player and tell them (privately) how it went. kind = add|remove|only|list|reset.
function N.exec(ply, kind, raw)
  local sid = key_of(ply)
  if not sid then return false end
  local now = CurTime()
  if now - (last_cmd[sid] or -10) < N.REST then tell(ply, "One name at a time."); return true end
  last_cmd[sid] = now
  if kind == "list" then
    local list = N.names(ply)
    tell(ply, line(#list == 1 and "list_one" or "list_many", sid, join_names(list)))
    return true
  end
  if kind == "reset" then
    tell(ply, N.reply(ply, "reset", N.reset(ply)))
    return true
  end
  raw = tostring(raw or "")
  if raw:gsub("%s", "") == "" then tell(ply, N.reply(ply, kind, false, "empty", "")); return true end
  local fn = kind == "add" and N.add or kind == "remove" and N.remove or N.only
  tell(ply, N.reply(ply, kind, fn(ply, raw)))
  return true
end

--- Is this sentence one of the name commands (with or without a name of this player's in front)? -> intent table or nil.
function N.peek(ply, text)
  local woke, rest = N.strip(ply, text)
  return Wake.intent(woke and rest or text)
end

--- The hook Holodeck.Handle calls first: handles a name command and returns true, else false (the words are an ordinary request).
function N.try_command(ply, text)
  if not valid(ply) then return false end
  local it = N.peek(ply, text)
  if not it then return false end
  return N.exec(ply, it.kind, it.name)
end

-- ---------------------------------------------------------------------------------------------- the owner's client
--- Send a player their own list (and nobody else): the microphone bridge on their PC reads it from the client's data folder.
function N.send(ply)
  if not valid(ply) then return end
  local list = N.names(ply)
  net.Start("holodeck_wake_names")
  net.WriteUInt(#list, 4)
  for _, n in ipairs(list) do net.WriteString(n) end
  net.Send(ply)
end

local last_sync = {}
net.Receive("holodeck_wake_sync", function(_, ply)
  if not valid(ply) then return end
  local now = CurTime()
  if now - (last_sync[ply] or -10) < 2 then return end
  last_sync[ply] = now
  N.send(ply)
end)
hook.Add("PlayerInitialSpawn", "Holodeck.WakeNames", function(ply) timer.Simple(6, function() if valid(ply) then N.send(ply) end end) end)
hook.Add("PlayerDisconnected", "Holodeck.WakeNamesLeave", function(ply) last_sync[ply] = nil; local s = key_of(ply); if s then last_cmd[s] = nil end end)

-- ---------------------------------------------------------------------------------------------- the console
local function console(kind)
  return function(ply, _, args)
    if not valid(ply) then print("[holodeck] wake names belong to a player: use this from the game console, not the server console"); return end
    N.exec(ply, kind, table.concat(args or {}, " "))
  end
end
concommand.Add("skippy_wake_add", console("add"), nil, "Add a name Skippy answers to (only for you): skippy_wake_add Rex")
concommand.Add("skippy_wake_remove", console("remove"), nil, "Stop Skippy answering to one of your names (never the last one)")
concommand.Add("skippy_wake_only", console("only"), nil, "Skippy answers only to this name, for you")
concommand.Add("skippy_wake_list", console("list"), nil, "The names Skippy answers to for you")
concommand.Add("skippy_wake_reset", console("reset"), nil, "Back to Skippy alone, for you")

N.load()
return N
