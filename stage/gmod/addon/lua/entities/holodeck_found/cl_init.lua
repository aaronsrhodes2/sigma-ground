-- holodeck_found -- client part: the box and its label. Colours follow the project's doctrine: the green accent (#00FF88) on black,
-- one flat colour, no gradients, text on black.
include("shared.lua")
local CM = include("mentat/holodeck/meshclient.lua")          -- receives the real shape and builds it (see meshnet.lua for the server half)
local FX = include("mentat/fx/transporter.lua")                -- the transporter effect: a beam whose coherence is the pipeline's progress
local TM = FX.model

local GREEN = Color(0, 255, 136)
local WHITE = Color(255, 255, 255)
local BLACK = Color(0, 0, 0)
local FILL_ALPHA = 60                 -- the translucent fill
local LABEL_RANGE = 525               -- Hammer units: about 10 m
local LABEL_RANGE_SQR = LABEL_RANGE * LABEL_RANGE
local cv_outline = CreateClientConVar("holodeck_found_outline", "0", true, false, "1 = keep a faint outline of the card's box around a found object's real shape")

--- The server changed the visual. The NetworkVars (VisualKind, MeshId) carry the truth to every client, including a late one; this
--- message only lets the object redraw at once, and a mesh the client does not have yet is asked for here.
net.Receive("holodeck_found_visual", function()
  local ent = Entity(net.ReadUInt(16))
  local kind = net.ReadString()
  local id = net.ReadUInt(8)
  if IsValid(ent) and ent.SetVisual then ent:SetVisual(kind, { id = id }) end
end)

function ENT:Initialize()
  local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
  local pad = math.max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)           -- the beam's cloud reaches past the footprint
  self:SetRenderBounds(mn - Vector(pad, pad, 8), mx + Vector(pad, pad, pad * 0.6), Vector(8, 8, 40))
end

--- How opaque the holodeck's materialising fade (the entity colour's alpha) makes the box right now, 0 to 1.
local function fade(self)
  local c = self:GetColor()
  return (c and c.a or 255) / 255
end

--- The real shape when this object's visual is a mesh and this client has built it; nil (draw the box) otherwise.
local function ready_mesh(self)
  if self:GetVisual() ~= "mesh" then return nil end
  return CM.get(self:GetMeshId())
end

-- ------------------------------------------------------------------------------------------------ the transporter
local function geometry(self)
  return { pos = self:GetPos(), ang = self:GetAngles(), mins = self:GetBoxMins(), maxs = self:GetBoxMaxs() }
end

local function eye_distance(self)
  local me = LocalPlayer()
  if not IsValid(me) then return 0 end
  return me:EyePos():Distance(self:GetPos())
end

--- The pipeline's progress as THIS client sees it: the server's stage and progress, and while the mesh is in flight to this client, the
--- bytes it has actually received (the per-chunk part of the sending stage). Never less than the server's.
local function target_progress(self)
  local p = self:GetFxProgress()
  if self:FxStageName() == "sending" then
    local frac = CM.recv_fraction and CM.recv_fraction(self:GetMeshId())
    if frac then p = math.max(p, TM.progress("sending", frac)) end
  end
  return p
end

local function pending_params(self, fx, now, dt)
  local target = target_progress(self)
  if fx.first then fx.st.shown, fx.first = target, false end                  -- a late joiner starts at the truth, not at zero
  fx.st.shown = TM.ease(fx.st.shown, target, dt)
  local stalled = TM.stalled_for(now, self:GetFxBeat())
  local P = TM.coherence(fx.st.shown, stalled, now, { stall_s = FX.stall_s and FX.stall_s() or nil })
  return P, stalled
end

--- Draw the beam instead of the box (pending), or the failure, or the materialise around the mesh. Returns true when it drew the object.
local function draw_transporter(self, a, real)
  local fx = self.fx
  if not fx then fx = { st = FX.state(), first = true }; self.fx = fx end
  local now = CurTime()
  local dt = math.min(0.1, math.max(0, now - (fx.last or now)))
  fx.last = now
  local stage = self:FxStageName()
  local dist = eye_distance(self)
  local g = geometry(self)
  if stage == "failed" then                                                  -- given up on: collapse red, then nothing
    local P = TM.failing(TM.stalled_for(now, self:GetFxBeat()), self:GetFxProgress())
    if P.done then return true end
    P.condensation, P.spread, P.jitter_hz, P.scan_alpha = TM.smooth(self:GetFxProgress()), 0.3 + P.scatter, P.flicker_hz, 0
    P.alpha = P.alpha * a
    FX.timed(FX.draw, g, P, fx.st, now, dt, dist)
    return true
  end
  if real then
    if fx.arrive == nil then fx.arrive = (fx.was_pending or (stage ~= nil and stage ~= "done")) and now or false end
    if fx.arrive then
      local P = TM.arriving(now - fx.arrive)
      if not P.done then
        local mn, mx = g.mins, g.maxs
        local up = self:GetUp()
        local zc = mn.z + (mx.z - mn.z) * P.reveal * 1.01
        local n = -up
        render.EnableClipping(true)
        render.PushCustomClipPlane(n, n:Dot(self:GetPos() + up * zc))         -- keeps what is below the band
        CM.draw(self, real, a)
        render.PopCustomClipPlane()
        render.EnableClipping(false)
        P.condensation, P.spread, P.jitter_hz, P.scan_alpha = 1, 0.02, 9, 0
        P.alpha = P.beam_alpha * a
        P.band_y, P.band_alpha = P.reveal, P.band_alpha * a
        FX.timed(FX.draw, g, P, fx.st, now, dt, dist)
        return true
      end
    end
    return false
  end
  if self:FxPending() or (stage == "done" and TM.stalled_for(now, self:GetFxBeat()) < 20) then       -- on its way
    fx.was_pending = true
    local P, stalled = pending_params(self, fx, now, dt)
    P.alpha = P.alpha * a
    fx.stalled = stalled
    FX.timed(FX.draw, g, P, fx.st, now, dt, dist)
    return true
  end
  return false
end

function ENT:Draw()
  local a = fade(self)
  local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
  local real = ready_mesh(self)
  if (not FX.enabled or FX.enabled:GetBool()) and (self:GetFxStage() ~= 0) and draw_transporter(self, a, real) then return end
  if real then
    CM.draw(self, real, a)
    if cv_outline:GetBool() then
      render.SetColorMaterial()
      render.DrawWireframeBox(self:GetPos(), self:GetAngles(), mn, mx, Color(GREEN.r, GREEN.g, GREEN.b, math.floor(70 * a)), true)
    end
    return
  end
  render.SetColorMaterial()
  render.DrawBox(self:GetPos(), self:GetAngles(), mn, mx, Color(GREEN.r, GREEN.g, GREEN.b, math.floor(FILL_ALPHA * a)), true)
  render.DrawWireframeBox(self:GetPos(), self:GetAngles(), mn, mx, Color(GREEN.r, GREEN.g, GREEN.b, math.floor(255 * a)), true)
end

--- The second label line while the object is on its way: "fetching 42 %" (cyan), "stalled" (amber) or "lost the beam" (red); nil otherwise.
function ENT:StatusLine()
  local stage = self:FxStageName()
  if stage == "failed" then return "transport failed", Color(255, 68, 68) end
  if not self:FxPending() then return nil end
  local stalled = TM.stalled_for(CurTime(), self:GetFxBeat())
  local pct = math.floor(100 * (self.fx and self.fx.st.shown or self:GetFxProgress()) + 0.5)
  if stalled > (FX.stall_s and FX.stall_s() or TM.STALL_S) * 0.7 then return string.format("%s %d %% - no news for %d s", stage, pct, math.floor(stalled)), Color(255, 170, 0) end
  return string.format("%s %d %%", stage, pct), Color(0, 204, 255)
end

--- The box, and the floating label for those who are near.
function ENT:DrawTranslucent()
  self:Draw()
  local me = LocalPlayer()
  if not IsValid(me) then return end
  local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
  local top = self:LocalToWorld(Vector((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mx.z + 8))
  if me:GetPos():DistToSqr(top) > LABEL_RANGE_SQR then return end
  local label = self:LabelText()
  local ang = Angle(0, me:EyeAngles().y - 90, 90)          -- upright, turned to the viewer
  cam.Start3D2D(top, ang, 0.12)
    surface.SetFont("DermaLarge")
    local w, h = surface.GetTextSize(label)
    surface.SetDrawColor(BLACK)
    surface.DrawRect(-w / 2 - 6, -h - 4, w + 12, h + 8)
    draw.SimpleText(label, "DermaLarge", 0, 0, WHITE, TEXT_ALIGN_CENTER, TEXT_ALIGN_BOTTOM)
    local note, col = self:StatusLine()
    if note then
      local w2 = surface.GetTextSize(note)
      surface.SetDrawColor(BLACK)
      surface.DrawRect(-w2 / 2 - 6, 4, w2 + 12, h + 4)
      draw.SimpleText(note, "DermaLarge", 0, 6, col, TEXT_ALIGN_CENTER, TEXT_ALIGN_TOP)
    end
  cam.End3D2D()
end
