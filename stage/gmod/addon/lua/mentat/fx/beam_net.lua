-- fx/beam_net.lua -- SERVER. Skippy's arrival and departure as a transporter beam: the avatar's own hooks (avatar.lua runs them, and
-- nothing else changes) are turned into one small message to every client, which draws the beam (fx/transporter.lua) and keeps his
-- body hidden until the arrival beam has completed. If a client has no effect (holodeck_fx 0) it simply gets the body at once.
--
--   SkippyAvatarArrive(bot, from, to)  -> "mentat_fx_beam" arrive = true,  at `to`, following `bot` while it plays (about 1.0 s)
--   SkippyAvatarLeave(bot, from)       -> "mentat_fx_beam" arrive = false, at `from` (about 0.8 s), the column dissolves
local B = {}
util.AddNetworkString("mentat_fx_beam")
B.sent = 0

local function send(arrive, pos, bot)
  if pos == nil then return end
  net.Start("mentat_fx_beam")
  net.WriteBool(arrive)
  net.WriteVector(pos)
  net.WriteEntity(bot)
  net.Broadcast()
  B.sent = B.sent + 1
end

hook.Add("SkippyAvatarArrive", "Mentat.FXBeamArrive", function(bot, from, to) send(true, to, bot) end)
hook.Add("SkippyAvatarLeave", "Mentat.FXBeamLeave", function(bot, from) send(false, from, bot) end)
B.send = send
return B
