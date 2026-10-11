-- holodeck_wisp -- server part: no model, no physics, no collision. The state machine runs from mentat/holodeck/wisps.lua's tick
-- (ENT:Step), which also knows whether the owner is looking at it.
AddCSLuaFile("shared.lua")
AddCSLuaFile("cl_init.lua")
AddCSLuaFile("mentat/fx/wisp.lua")
include("shared.lua")
local W = include("mentat/fx/wisp.lua")

--- Before Spawn: the sentence, its class ("failure" | "suggest" | "refusal" | "ok"), the owner (a player or nil) and the lifetime of a
--- wisp nobody reads. Returns true, or false and why.
function ENT:SetWisp(text, class, owner, life)
  text = W.cap_text(text, W.MAX_TEXT)
  if text == "" then return false, "no text" end
  self:SetReason(text)
  self:SetReasonClass(W.CLASS_INDEX[class] or W.CLASS_INDEX.failure)
  self:SetOwnerPly(owner)
  self.st = W.new(text, { life = life })
  self:SetDwell(self.st.dwell)
  self:SetPhase(math.random(0, 99))
  return true
end

function ENT:Initialize()
  if not self.st then
    ErrorNoHalt("[holodeck_wisp] no text was set before Spawn\n")
    self:Remove()
    return
  end
  self:SetSolid(SOLID_NONE or 0)
  self:SetMoveType(MOVETYPE_NONE or 0)
  self:DrawShadow(false)
  self:SetWispStateIndex(W.STATE_INDEX[self.st.state])
  self:SetStateT(CurTime())
end

local WHY_INDEX = { read = 1, expired = 2, dismissed = 3 }

--- One tick of the state machine. `looking`: the owner is reading it. Removes the entity when it is gone. Returns the state name.
function ENT:Step(dt, looking)
  local st = self.st
  if not st then return "gone" end
  local before = st.state
  W.step(st, dt, looking)
  self:SetRead(st.read)
  if st.state ~= before then
    self:SetWispStateIndex(W.STATE_INDEX[st.state])
    self:SetStateT(CurTime())
    if st.why then self:SetWhy(WHY_INDEX[st.why] or 0) end
  end
  if st.state == "gone" then self:Remove() end
  return st.state
end

--- The owner dismisses it (the clear command): it dissolves like a read one.
function ENT:Dismiss()
  local st = self.st
  if not st then return end
  local before = st.state
  W.dismiss(st)
  if st.state ~= before then
    self:SetWispStateIndex(W.STATE_INDEX[st.state])
    self:SetStateT(CurTime())
    self:SetWhy(WHY_INDEX[st.why] or 3)
  end
end
