-- holodeck_client.lua -- the holodeck's voice on the player's screen: what Skippy says appears in chat.
if not CLIENT then return end

net.Receive("holodeck_say", function()
  local text = net.ReadString()
  chat.AddText(Color(0, 255, 136), "[Skippy] ", Color(255, 255, 255), text)
  hook.Run("HolodeckSay", text)
end)
