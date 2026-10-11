-- holodeck_wisp -- client part: a soft light with a few embers, and, for the owner who looks at it up close, the reason as small floating
-- text that faces them. Procedural only: a glow texture painted once into a render target (no game material, model or sound), additive
-- sprites, and 3D2D text on a black plate. Colours are the project's palette (fx/wisp.lua). Bounded: about 20 sprites with the text up,
-- 10 without; nothing is drawn beyond 25 m or behind the viewer. The pose and the state come from the server (shared.lua).
include("shared.lua")
local W = include("mentat/fx/wisp.lua")

Mentat = Mentat or {}
local WFX = Mentat.WispFX or { cost_ms = 0, drawn = 0, built = false }      -- cost_ms: smoothed CPU time of one frame's wisps
Mentat.WispFX = WFX

local FONT = "MentatWisp"
local fonts_made = false
local function fonts()
  if fonts_made then return end
  fonts_made = true
  if surface and surface.CreateFont then
    surface.CreateFont(FONT, { font = "Tahoma", size = 44, weight = 700, antialias = true, extended = true })
  end
end

-- ------------------------------------------------------------------------------------------------ the glow texture
-- A 64x64 soft round falloff, black outside: with additive blending black adds nothing, so there is no square edge and no alpha is needed.
local glow_mat
local flat_mat
local function build_glow()
  if glow_mat or WFX.glow_failed then return end
  local ok, err = pcall(function()
    local rt = GetRenderTarget("mentat_wisp_glow_v1", 64, 64)
    render.PushRenderTarget(rt)
    render.Clear(0, 0, 0, 255, true, true)
    cam.Start2D()
    for y = 0, 63 do
      for x = 0, 63 do
        local dx, dy = (x - 31.5) / 32, (y - 31.5) / 32
        local r = math.sqrt(dx * dx + dy * dy)
        local v = 0
        if r < 1 then
          local edge = 1 - W.smooth((r - 0.55) / 0.45)                -- reaches exactly zero at the rim
          v = math.floor(255 * math.exp(-r * r * 3.2) * edge + 0.5)
        end
        surface.SetDrawColor(v, v, v, 255)
        surface.DrawRect(x, y, 1, 1)
      end
    end
    cam.End2D()
    render.PopRenderTarget()
    glow_mat = CreateMaterial("mentat_wisp_glow_v1", "UnlitGeneric", {
      ["$basetexture"] = rt:GetName(), ["$vertexcolor"] = 1, ["$vertexalpha"] = 1, ["$additive"] = 1, ["$nocull"] = 1,
    })
  end)
  if not ok then WFX.glow_failed = tostring(err); print("[wisp] glow texture failed: " .. WFX.glow_failed) end
  WFX.built = glow_mat ~= nil
end
-- (render targets are painted from a frame hook that allows it, not from inside a draw)
hook.Add("PostRender", "Mentat.WispGlow", function()
  if Mentat.WispNeeded and not glow_mat then build_glow() end
end)

local function material()
  if glow_mat then return glow_mat end
  if not flat_mat then
    flat_mat = CreateMaterial("mentat_wisp_flat_v1", "UnlitGeneric", {
      ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 1, ["$additive"] = 1, ["$nocull"] = 1,
    })
  end
  return flat_mat
end

-- ------------------------------------------------------------------------------------------------ helpers
local function hash(n)
  local s = math.sin(n * 12.9898 + 78.233) * 43758.5453
  return s - math.floor(s)
end

local C = Color(255, 255, 255, 255)
local function col(c, a, white)
  white = white or 0
  C.r = math.floor(c[1] + (255 - c[1]) * white + 0.5)
  C.g = math.floor(c[2] + (255 - c[2]) * white + 0.5)
  C.b = math.floor(c[3] + (255 - c[3]) * white + 0.5)
  C.a = math.max(0, math.min(255, math.floor(a * 255 + 0.5)))
  return C
end

--- Each viewer's own chat filter (Steam's) applies to the words, whoever supplied them (holodeck_client.lua).
local function filtered(s)
  local f = Mentat.FilterText
  if f then return f(s) end
  return s
end

function ENT:Initialize()
  self:SetRenderBounds(Vector(-40, -40, -30), Vector(40, 40, 70))
  Mentat.WispNeeded = true
  fonts()
end

function ENT:Draw() end                       -- (everything is drawn in DrawTranslucent)

local V, V2 = Vector(0, 0, 0), Vector(0, 0, 0)

local EMBERS = 7
local function draw_embers(self, pos, c, look, now, phase)
  local a = look.orb * (1 - look.fade)
  if a < 0.03 then return 0 end
  for i = 1, EMBERS do
    local life = 5 + 3 * hash(i * 1.9 + phase)
    local k = ((now + hash(i * 3.7 + phase) * life) % life) / life              -- 0 at the orb, 1 at the top of its rise
    local env = math.sin(k * math.pi)
    V.x = pos.x + (hash(i * 5.3 + phase) * 2 - 1) * (4 + 6 * k) + math.sin(now * 0.7 + i) * 1.2
    V.y = pos.y + (hash(i * 7.1 + phase) * 2 - 1) * (4 + 6 * k) + math.cos(now * 0.6 + i) * 1.2
    V.z = pos.z + 3 + k * 26
    local s = (2.2 + 2.0 * hash(i * 9.7)) * (1 - 0.4 * k) * look.size
    render.DrawSprite(V, s, s, col(c, a * env * 0.8, 0.35))
  end
  return EMBERS
end

local function draw_orb(self, pos, c, look, now, phase, reading)
  local pulse = W.pulse(now, phase, reading)
  local a = look.orb * (1 - look.fade * 0.5)
  if a < 0.02 then return 0 end
  local s = look.size * pulse
  render.DrawSprite(pos, 70 * s, 70 * s, col(c, a * 0.30))                      -- the halo
  render.DrawSprite(pos, 36 * s, 36 * s, col(c, a * 0.65))                      -- the body
  render.DrawSprite(pos, 13 * s, 13 * s, col(c, a, 0.65))                  -- the hot core
  return 3
end

--- The floating text for the owner: a black plate, white words typed out as it unfurls, a thin line in the class colour.
local function draw_text(self, pos, eye, c, look, dist, k_unfurl)
  if not self.shown_text or self.shown_src ~= self:GetReason() then
    self.shown_src = self:GetReason()
    self.shown_text = filtered(self.shown_src)
    self.lines = W.wrap(self.shown_text, 4, 40)
  end
  local lines = self.lines
  if not lines or #lines == 0 or look.text < 0.02 then return end
  local d = pos - eye
  local yaw = math.deg(math.atan2(d.y, d.x))
  local scale = 0.115 * math.max(1, math.min(1.8, dist / 110))                     -- farther reads bigger, so 2-3 m stays legible
  surface.SetFont(FONT)
  local _, lh = surface.GetTextSize("Hg")
  local total = 0
  for _, l in ipairs(lines) do total = total + #l end
  local shown = math.ceil(total * W.smooth(k_unfurl))
  local widest = 0
  for _, l in ipairs(lines) do local w = surface.GetTextSize(l); if w > widest then widest = w end end
  local pad = 12
  local bw, bh = widest + pad * 2, lh * #lines + pad * 2
  local top = pos + Vector(0, 0, 12 + bh * scale)
  cam.Start3D2D(top, Angle(0, yaw - 90, 90), scale)
    local a = look.text
    surface.SetDrawColor(0, 0, 0, math.floor(215 * a))
    surface.DrawRect(-bw / 2, 0, bw, bh)
    surface.SetDrawColor(c[1], c[2], c[3], math.floor(230 * a))
    surface.DrawRect(-bw / 2, bh - 3, bw * W.smooth(k_unfurl), 3)                -- the class colour, as a line that grows with the text
    local left = shown
    for i, l in ipairs(lines) do
      if left <= 0 then break end
      local part = left >= #l and l or l:sub(1, left)
      left = left - #l
      draw.SimpleText(part, FONT, -bw / 2 + pad, pad + (i - 1) * lh, Color(255, 255, 255, math.floor(255 * a)), TEXT_ALIGN_LEFT, TEXT_ALIGN_TOP)
    end
  cam.End3D2D()
  self.text_box = { top = top, yaw = yaw, scale = scale, bw = bw, bh = bh }
end

--- The dissolve: the text turns to sparks that rise from where it was.
local SPARKS = 26
local function draw_sparks(self, pos, c, look)
  local tb = self.text_box
  if look.sparks <= 0.01 or look.sparks >= 0.999 then return 0 end
  local k = look.sparks
  local base = tb and tb.top or (pos + Vector(0, 0, 18))
  local halfw = tb and (tb.bw * tb.scale / 2) or 18
  local h = tb and (tb.bh * tb.scale) or 8
  local yaw = tb and tb.yaw or 0
  local rx, ry = -math.sin(math.rad(yaw)), math.cos(math.rad(yaw))                -- along the text's width
  for i = 1, SPARKS do
    local u, v = hash(i * 2.3) * 2 - 1, hash(i * 4.1)
    local rise = k * (8 + 22 * hash(i * 6.7))
    V2.x = base.x + rx * u * halfw + (hash(i * 8.9) - 0.5) * 6 * k
    V2.y = base.y + ry * u * halfw + (hash(i * 10.3) - 0.5) * 6 * k
    V2.z = base.z - h * v + rise
    local s = (2.2 + 2.4 * hash(i * 12.1)) * (1 - 0.5 * k)
    render.DrawSprite(V2, s, s, col(c, (1 - k) * (0.5 + 0.5 * hash(i * 14.3)), 0.5))
  end
  return SPARKS
end

local acc_ms, acc_n, acc_frame = 0, 0, -1
function ENT:DrawTranslucent()
  local t0 = SysTime()
  local me = LocalPlayer()
  if not IsValid(me) then return end
  local eye = me:EyePos()
  local state = self:WispState()
  if state == "gone" then return end
  local now = CurTime()
  local phase = self:GetPhase() / 100
  local bx, by, bz = W.bob(now, phase)
  local o = self:GetPos()
  local pos = Vector(o.x + bx, o.y + by, o.z + bz)
  local d = pos - eye
  local dist = d:Length()
  if dist > W.FAR then return end
  if dist > 40 and d:Dot(me:EyeAngles():Forward()) < -0.2 * dist then return end   -- behind the viewer
  local mine = self:GetOwnerPly() == me
  local why = self:WispWhy()
  local since = math.max(0, now - self:GetStateT())
  -- the text eases open while the owner reads and stays open while it dissolves; the client's own ease so it is smooth between ticks
  local target = (state == "reading" or (state == "dissolving" and why ~= "expired" and (self.unfurl or 0) > 0.01)) and 1 or 0
  local dt = math.max(0, now - (self.last_t or now))
  if dt > 0.5 then dt = 0.016 end                                   -- (not drawn for a while: no jump; a slow frame still counts in full)
  self.last_t = now
  local u = self.unfurl or 0
  if u < target then u = math.min(target, u + dt / 0.9) elseif u > target then u = math.max(target, u - dt / 0.5) end
  self.unfurl = u
  local look = W.look(state, since, u, mine, why)
  local c = W.colour(self:WispClass())
  render.SetMaterial(material())
  local n = draw_orb(self, pos, c, look, now, phase, state == "reading")
  n = n + draw_embers(self, pos, c, look, now, phase)
  if mine then
    draw_text(self, pos, eye, c, look, dist, u)
    render.SetMaterial(material())
    n = n + draw_sparks(self, pos, c, look)
  end
  local fr = FrameNumber and FrameNumber() or 0
  if fr ~= acc_frame then
    if acc_frame >= 0 and acc_ms > 0 then WFX.cost_ms = WFX.cost_ms * 0.95 + acc_ms * 0.05; WFX.drawn = acc_n end
    acc_ms, acc_n, acc_frame = 0, 0, fr
  end
  acc_ms = acc_ms + (SysTime() - t0) * 1000
  acc_n = acc_n + n
end
