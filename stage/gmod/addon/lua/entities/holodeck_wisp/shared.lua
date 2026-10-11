-- holodeck_wisp -- shared part. The FAILURE WISP: when a found-object transport fails, Skippy's honest reason is left behind as a small
-- floating light at about eye height at the failed spot (README "Failure wisp"). The owner reads it by looking at it up close; after a
-- dwell it dissolves. It is not a prop: no model, no physics, no collision, no registry entry (undo / clear / the limits never see it).
--
-- The server owns the state machine (mentat/fx/wisp.lua: it is the one that sees where the owner looks) and publishes the result in
-- NetworkVars, so a late joiner sees the same picture; every client only draws. The reason text is the resolver's / the glue's own
-- sentence (never made up here), capped to W.MAX_TEXT characters; each viewer's chat filter is applied to it when it is drawn.
local W = include("mentat/fx/wisp.lua")

ENT.Type = "anim"
ENT.Base = "base_anim"
ENT.PrintName = "Holodeck failure wisp"
ENT.Author = "sigma-ground"
ENT.Spawnable = false
ENT.AdminOnly = true
ENT.RenderGroup = RENDERGROUP_TRANSLUCENT

ENT.WHY = { "read", "expired", "dismissed" }          -- the networked integer Why (0 = none yet)

function ENT:SetupDataTables()
  self:NetworkVar("String", 0, "Reason")             -- Skippy's sentence, at most W.MAX_TEXT characters
  self:NetworkVar("Int", 0, "ReasonClass")           -- index (from 1) into W.CLASSES: the colour
  self:NetworkVar("Int", 1, "WispStateIndex")        -- index (from 1) into W.STATES
  self:NetworkVar("Int", 2, "Why")                   -- why it is dissolving (index into ENT.WHY), 0 = not yet
  self:NetworkVar("Int", 3, "Phase")                 -- 0..99, so two wisps do not bob in step
  self:NetworkVar("Float", 0, "StateT")              -- server time the current state began
  self:NetworkVar("Float", 1, "Read")                -- seconds of looking so far
  self:NetworkVar("Float", 2, "Dwell")               -- seconds of looking it takes
  self:NetworkVar("Entity", 0, "OwnerPly")           -- the player who asked: the only one who can read (and so dismiss) it
end

--- The state name ("idle", ...).
function ENT:WispState() return W.STATES[self:GetWispStateIndex()] or "appear" end

--- The class name ("failure", ...).
function ENT:WispClass() return W.CLASSES[self:GetReasonClass()] or "failure" end

--- Why it is dissolving: "read" | "expired" | "dismissed" | nil.
function ENT:WispWhy() return self.WHY[self:GetWhy()] end
