-- holodeck_found -- client part: the box and its label. Colours follow the project's doctrine: the green accent (#00FF88) on black,
-- one flat colour, no gradients, text on black.
include("shared.lua")
local CM = include("mentat/holodeck/meshclient.lua")          -- receives the real shape and builds it (see meshnet.lua for the server half)

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
  self:SetRenderBounds(self:GetBoxMins(), self:GetBoxMaxs(), Vector(8, 8, 40))
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

function ENT:Draw()
  local a = fade(self)
  local mn, mx = self:GetBoxMins(), self:GetBoxMaxs()
  local real = ready_mesh(self)
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
  cam.End3D2D()
end
