-- holodeck_client.lua -- the holodeck's voice on the player's screen: what Skippy says appears in chat.
if not CLIENT then return end

--- Each viewer's own chat filter (Steam's) applies to what Skippy says, whoever or whatever supplied the words.
local function filtered(text)
  if util.FilterText and TEXT_FILTER_CHAT then
    local ok, out = pcall(util.FilterText, text, TEXT_FILTER_CHAT)
    if ok and isstring(out) then return out end
  end
  return text
end

net.Receive("holodeck_say", function()
  local text = filtered(net.ReadString())
  local to = net.ReadString()
  local me = LocalPlayer()
  local mine = to == "" or (IsValid(me) and string.sub(me:Nick(), 1, 24) == to)     -- others near by see whom Skippy answered
  chat.AddText(Color(0, 255, 136), mine and "[Skippy] " or ("[Skippy to " .. to .. "] "), Color(255, 255, 255), text)
  hook.Run("HolodeckSay", text)
end)
