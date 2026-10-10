-- client.lua -- minimal HUD for the Mentat stage: what is playing, where in time, at what rate.
-- Colours follow the Skippy passthrough palette: text on transparent, no fills.
local GREEN = Color(0, 255, 136)
local WHITE = Color(255, 255, 255)
local AMBER = Color(255, 170, 0)
local CYAN = Color(0, 204, 255)

hook.Add("HUDPaint", "MentatHUD", function()
  local name = GetGlobal2String("mentat_name", "")
  if name == "" then return end

  local state = GetGlobal2String("mentat_state", "")
  local t = GetGlobal2Float("mentat_t", 0)
  local tend = GetGlobal2Float("mentat_tend", 0)
  local rate = GetGlobal2Float("mentat_rate", 1)
  local released = GetGlobal2Int("mentat_released", 0)
  local bodies = GetGlobal2Int("mentat_bodies", 0)

  local x, y, dy = 24, 24, 20
  local font = "DermaDefaultBold"
  draw.SimpleText("MENTAT  " .. name, font, x, y, GREEN)
  y = y + dy
  draw.SimpleText(string.format("%s   t = %.3f / %.3f s   rate %.3gx", state, t, tend, rate), font, x, y, WHITE)
  y = y + dy
  if released > 0 then
    draw.SimpleText(string.format("%d / %d body(ies) RELEASED to Havok: no longer sigma-ground", released, bodies),
      font, x, y, AMBER)
    y = y + dy
  end
  draw.SimpleText("mentat_play  mentat_pause  mentat_scrub <t>  mentat_rate <x>  mentat_release  mentat_reset",
    font, x, y, CYAN)
end)
