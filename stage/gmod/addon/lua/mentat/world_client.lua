-- world_client.lua -- what a client knows about the construct's physics: the numbers the server published as globals, so an
-- effect (a splash, a dust puff, a fall sound) can scale with the real gravity and air density instead of assuming Earth.
-- Before the server has published (or with the construct off) the answers are the generated table's own values.
if not CLIENT then return end
Mentat = Mentat or {}
local C = include("mentat/world/constants.lua")
local R = {}
Mentat.WorldClient = R
local earth = C.PRESETS[C.DEFAULTS.preset]
function R.gravity_m_s2() return GetGlobalFloat("mentat_world_g_m_s2", earth.gravity_m_s2) end
function R.gravity_hu_s2() return GetGlobalFloat("mentat_world_g_hu_s2", earth.gravity_hu_s2) end
function R.air_density() return GetGlobalFloat("mentat_world_air_density", C.AIR_DENSITY) end
function R.preset() return GetGlobalString("mentat_world_preset", C.DEFAULTS.preset) end
function R.on() return GetGlobalBool("mentat_world_on", false) end
