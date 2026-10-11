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
  local uid = net.ReadUInt(32) or 0                       -- who asked (0 = nobody in particular: a private line, the console)
  local me = LocalPlayer()
  -- Others near by see whom Skippy answered. Decided by the asker's number, never by comparing names: the server sends the name
  -- after cleaning it (single spaces, cut at 24 bytes, repaired UTF-8), which is not the raw name for many players.
  local mine = to == "" or (uid ~= 0 and IsValid(me) and me:UserID() == uid)
  chat.AddText(Color(0, 255, 136), mine and "[Skippy] " or ("[Skippy to " .. to .. "] "), Color(255, 255, 255), text)
  hook.Run("HolodeckSay", text)
end)
