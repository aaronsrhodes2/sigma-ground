-- tool.lua -- headless Lua jobs for the stage (catalog measuring, holodeck tests, inventories).
--
-- Trigger: data/mentat/tool.txt = {"script":"mentat/tools/NAME.lua","id":"run1","args":{...}}
-- The script returns function(args, done); it calls done(result_table) exactly once, possibly later (timers).
-- The result is written to data/mentat/tool_<NAME>.json as {id, finished, error?, result?}. The driver
-- (stage/gmod/holodeck/run_tool.py) arms the trigger, starts the server, waits for that file, stops the server.
if not SERVER then return end

local CFG = "mentat/tool.txt"
if not file.Exists(CFG, "DATA") then return end
local cfg = util.JSONToTable(file.Read(CFG, "DATA") or "")
file.Delete(CFG)
if type(cfg) ~= "table" or type(cfg.script) ~= "string" or not cfg.script:match("^mentat/tools/[%w_%-]+%.lua$") then
  print("[mentat-tool] unreadable or unsafe tool.txt; ignored")
  return
end

local NAME = cfg.script:match("([%w_%-]+)%.lua$")
local OUT = "mentat/tool_" .. NAME .. ".json"
file.CreateDir("mentat")
file.Delete(OUT)                                           -- a stale result must never pass for a fresh run

local finished = false
local function done(result, err)
  if finished then return end
  finished = true
  file.Write(OUT, util.TableToJSON({ id = cfg.id, finished = true, error = err, result = result }))
  print("[mentat-tool] " .. NAME .. " wrote data/" .. OUT .. (err and (" with error: " .. tostring(err)) or ""))
end

local function go(ply)
  local ok, fn = pcall(include, cfg.script)
  if not ok or type(fn) ~= "function" then return done(nil, "script failed to load: " .. tostring(fn)) end
  local okr, e = pcall(fn, cfg.args or {}, done, ply)
  if not okr then done(nil, "script raised: " .. tostring(e)) end
end

if game.IsDedicated() then
  hook.Add("InitPostEntity", "MentatTool", function() timer.Simple(5, function() go(nil) end) end)
else
  -- the first HUMAN to spawn runs it, once (a bot that joins later, such as Skippy's avatar, must not run it again with itself as the player)
  local started = false
  hook.Add("PlayerInitialSpawn", "MentatTool", function(ply)
    if started or ply:IsBot() then return end
    started = true
    timer.Simple(5, function() if IsValid(ply) then go(ply) end end)
  end)
end
print("[mentat-tool] armed for " .. NAME)
