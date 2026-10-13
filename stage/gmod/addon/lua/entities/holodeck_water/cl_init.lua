-- holodeck_water -- client part: draws the basin and the water (mentat/water/render.lua).  No model, material file or sound of the game.
include("shared.lua")
local R = include("mentat/water/render.lua")
local Vol = include("mentat/water/volumes.lua")
local C = include("mentat/water/constants.lua")
local FxState = include("mentat/water/fx.lua")
local FX = include("mentat/fx/transporter.lua")             -- the transporter effect, as for every ledger object
local TM = FX.model
local HU_M = C.HU_M

local GREEN = Color(0, 255, 136)

function ENT:Initialize()
  self.seq_seen = -1
  self:UpdateBounds()
end

--- The render bounds: the whole basin and headroom for droplets.
function ENT:UpdateBounds()
  local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
  self:SetRenderBounds(mn - Vector(40, 40, 10), mx + Vector(40, 40, 260))
  self.seq_seen = self:GetSeq()
end

local function g_now() return GetGlobalFloat("mentat_world_g_m_s2", C.data.g_earth.value) end
R.gravity = g_now

local function ease(x) x = x < 0 and 0 or (x > 1 and 1 or x); return x * x * (3 - 2 * x) end

function ENT:Think()
  if self.seq_seen ~= self:GetSeq() then self:UpdateBounds() end
  self:SetNextClientThink(CurTime() + 0.2)
  return true
end

function ENT:Draw() end                                                            -- all of it is in the translucent pass

local function alpha_of(self)
  local c = self:GetColor()
  return (c and c.a or 255) / 255
end

function ENT:DrawTranslucent()
  local me = LocalPlayer()
  if not IsValid(me) then return end
  local rec = Vol.record(self)
  if not rec then return end
  local eye = EyePos()
  local near = math.max(rec.b.hx, rec.b.hy)
  if eye:Distance(self:GetPos()) - near > R.DRAW_DISTANCE_HU then return end
  local now = CurTime()
  local age = now - self:GetBornAt()
  local g = g_now()
  local st = R.state_of(self)
  local dt = math.min(0.1, math.max(0, now - (st.t_last or now)))
  st.t_last = now
  FxState.update(st.fx, dt, g, now)
  rec.wx, rec.wy = rec.x * HU_M, rec.y * HU_M
  local a = alpha_of(self)
  -- arriving: the ghost of the volume and the beam, and the water rising to its level (the transporter's grammar)
  if age >= 0 and age < 2.0 then
    local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
    render.SetColorMaterial()
    render.DrawWireframeBox(self:GetPos(), self:GetAngles(), mn, mx, Color(GREEN.r, GREEN.g, GREEN.b, math.floor(200 * (1 - ease((age - 0.6) / 1.2)))), true)
    if FX.draw and (FX.enabled == nil or FX.enabled:GetBool()) then
      local P = TM.arriving(age)
      if P and not P.done then
        P.condensation, P.spread, P.jitter_hz, P.scan_alpha = 1, 0.02, 9, 0
        P.alpha = (P.beam_alpha or 0.6) * a
        P.band_y, P.band_alpha = P.reveal, (P.band_alpha or 0) * a
        st.beam = st.beam or FX.state()
        FX.timed(FX.draw, { pos = self:GetPos(), ang = self:GetAngles(), mins = mn, maxs = mx }, P, st.beam, now, dt, eye:Distance(self:GetPos()))
      end
    end
    rec.fill = rec.fill * ease((age - 0.3) / 1.2)
  end
  R.draw_body(self, rec, eye, now, g)
end

--- The second line of the label is not used: a water body is read by what it does.
