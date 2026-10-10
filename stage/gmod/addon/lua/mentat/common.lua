-- common.lua -- shared (server + client): scenario loading and per-body geometry cache.
Mentat = Mentat or {}
local U, S = Mentat.units, Mentat.shapes

Mentat.VERSION = "0.2.0-engine-experiments"
Mentat.HU = U.HU_PER_M                       -- Hammer units per metre
Mentat.scenarios = Mentat.scenarios or {}
Mentat.geometry = Mentat.geometry or {}

local NAME_PATTERN = "^[%w_%-]+$"        -- scenario names double as file names: no path tricks

function Mentat.ScenarioPath(name)
  return "mentat/scenarios/" .. name .. ".lua"
end

--- Sorted scenario names available to this realm.
function Mentat.ListScenarios()
  local out = {}
  for _, f in ipairs((file.Find("mentat/scenarios/*.lua", "LUA"))) do
    out[#out + 1] = (f:gsub("%.lua$", ""))
  end
  table.sort(out)
  return out
end

--- Load (and cache) a scenario table. Returns scn, or nil + reason.
function Mentat.GetScenario(name)
  if type(name) ~= "string" or not name:match(NAME_PATTERN) then
    return nil, "bad scenario name"
  end
  local cached = Mentat.scenarios[name]
  if cached then return cached end
  local path = Mentat.ScenarioPath(name)
  if not file.Exists(path, "LUA") then return nil, "no such scenario: " .. name end
  local ok, res = pcall(include, path)
  if not ok then return nil, "scenario failed to load: " .. tostring(res) end
  if type(res) ~= "table" or res.format ~= "mentat-scenario/1" then
    return nil, "not a mentat-scenario/1 file: " .. name
  end
  Mentat.scenarios[name] = res
  return res
end

--- HU per sigma metre for a scenario (Phase 0: stage_scale is always 1).
function Mentat.Scale(scn)
  return U.HU_PER_M * (scn.stage_scale or 1)
end

--- Render + physics geometry of body `idx` (1-based) of a scenario, entity-local HU.
function Mentat.GetGeometry(name, idx)
  local key = tostring(name) .. "#" .. tostring(idx)
  local g = Mentat.geometry[key]
  if g then return g end
  local scn = Mentat.GetScenario(name)
  if not scn then return nil end
  local body = scn.bodies[idx]
  if not body then return nil end
  g = S.build_body(body, Mentat.Scale(scn))
  Mentat.geometry[key] = g
  return g
end

--- Drop caches (after regenerating a scenario file). Server console: mentat_reload.
function Mentat.ReloadScenarios()
  Mentat.scenarios = {}
  Mentat.geometry = {}
end
