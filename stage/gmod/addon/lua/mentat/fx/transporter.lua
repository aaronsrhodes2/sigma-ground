-- fx/transporter.lua -- the CLIENT half of the transporter effect: a column of sparkles in an object's footprint box, drawn from the
-- parameters of fx/transporter_model.lua (the pure half, tested offline). Procedural only: no model, no material file, no sound from the
-- game. The particles are billboard squares of an additive "color/white" material (the same texture the found-mesh material uses);
-- the scan band, the outline and the silhouette fill are quads, a wireframe box and a translucent box. Bounded: at most MAXP sparkles
-- per beam (fewer with distance), nothing drawn beyond FAR.
--
--   FX.draw(g, P, st, now, dt, dist)   g = { pos, ang, mins, maxs } (the box in the entity's frame), P = parameters (see below), st = the
--                                       beam's own state table (FX.state()), dist = distance to the viewer (for the level of detail)
--   P: noise, spread, density, flicker, alpha, outline, fill, colour, scan_hz, scan_alpha, jitter_hz   (T.coherence)
--      height (0..1, default 1: the column's top; the failure collapses it), scatter (outward drift, failure), band_y/band_alpha (a scan
--      band pinned to a height: the materialise rides it up the object)
--
-- Beams for things that are not entities (Skippy's arrival and departure) are kept in FX.beams and drawn from
-- PostDrawTranslucentRenderables; FX.avatar_hidden(bot) says whether his body must wait for the beam.
if not CLIENT then return {} end

Mentat = Mentat or {}
if Mentat.FX then return Mentat.FX end                -- included by the entity and by the avatar: one module, one set of beams

local TM = include("mentat/fx/transporter_model.lua")

local FX = {}
FX.model = TM
FX.MAXP = 220               -- sparkles per beam, at most
FX.FAR = 3500               -- Hammer units: nothing is drawn beyond this
FX.NEAR_FULL = 700          -- full detail inside this; fades to a quarter of the sparkles by 2500
FX.beams = {}               -- avatar beams: { kind =, t0 =, pos =, bot =, st =, dur = }
FX.cost_ms = 0              -- smoothed CPU time of one frame's beams
FX.drawn = 0                -- sparkles drawn in the last frame (for the live measure)
FX.enabled = CreateClientConVar and CreateClientConVar("holodeck_fx", "1", true, false, "1 = draw the transporter effect; 0 = the plain box (cheapest)") or nil

-- ------------------------------------------------------------------------------------------------ the sparkles' fixed positions
local function hash(n)
  local s = math.sin(n * 12.9898 + 78.233) * 43758.5453
  return s - math.floor(s)
end
FX.hash = hash

-- per sparkle: a place in the CLOUD (unit cube), a HOME on the box surface (unit cube coordinates), a phase and a size
local CX, CY, CZ, HX, HY, HZ, PH, SZ = {}, {}, {}, {}, {}, {}, {}, {}
for i = 1, FX.MAXP do
  CX[i], CY[i], CZ[i] = hash(i * 1.7) * 2 - 1, hash(i * 2.9 + 5) * 2 - 1, hash(i * 4.1 + 11)
  PH[i], SZ[i] = hash(i * 6.3 + 17), 0.6 + 0.8 * hash(i * 8.7 + 23)
  -- home: an edge of the box for a third of them (the outline), a face for the rest (the silhouette)
  local a, b, c = hash(i * 3.3 + 29), hash(i * 5.1 + 31), hash(i * 7.7 + 37)
  if i % 3 == 0 then                                   -- an edge: two coordinates snapped to 0 or 1
    local axis = math.floor((i % 9) / 3)
    local u, v = (hash(i * 9.1) < 0.5) and 0 or 1, (hash(i * 11.3) < 0.5) and 0 or 1
    if axis == 0 then HX[i], HY[i], HZ[i] = a, u, v elseif axis == 1 then HX[i], HY[i], HZ[i] = u, a, v else HX[i], HY[i], HZ[i] = u, v, a end
  else                                                 -- a face: one coordinate snapped
    local f = i % 6
    if f == 0 then HX[i], HY[i], HZ[i] = 0, b, c elseif f == 1 then HX[i], HY[i], HZ[i] = 1, b, c
    elseif f == 2 then HX[i], HY[i], HZ[i] = a, 0, c elseif f == 3 then HX[i], HY[i], HZ[i] = a, 1, c
    elseif f == 4 then HX[i], HY[i], HZ[i] = a, b, 0 else HX[i], HY[i], HZ[i] = a, b, 1 end
  end
end

-- ------------------------------------------------------------------------------------------------ materials and scratch values
local mat_add
local function material()
  if not mat_add then
    mat_add = CreateMaterial("mentat_fx_transporter_v1", "UnlitGeneric", {
      ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 1, ["$additive"] = 1, ["$nocull"] = 1,
    })
  end
  return mat_add
end
local V = Vector(0, 0, 0)          -- scratch, mutated per sparkle (no allocation in the loop)
local C = Color(255, 255, 255, 255)
local V1, V2, V3, V4 = Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0)

--- Seconds of silence after which a beam is fully destabilised (the server's replicated convar holodeck_found_stall).
function FX.stall_s()
  local cv = GetConVar and GetConVar("holodeck_found_stall")
  local v = cv and cv:GetFloat()
  return (v and v > 0) and v or TM.STALL_S
end

function FX.state() return { shown = 0, scan = 0, last = nil } end

--- 1 inside NEAR_FULL, falling to 0.25 at 2500, nil past FAR (draw nothing).
function FX.lod(dist)
  if dist > FX.FAR then return nil end
  if dist <= FX.NEAR_FULL then return 1 end
  return math.max(0.25, 1 - 0.75 * (dist - FX.NEAR_FULL) / (2500 - FX.NEAR_FULL))
end

local function set_col(c, a)
  C.r, C.g, C.b, C.a = math.floor(c[1] + 0.5), math.floor(c[2] + 0.5), math.floor(c[3] + 0.5), math.max(0, math.min(255, math.floor(a * 255 + 0.5)))
  return C
end

-- ------------------------------------------------------------------------------------------------ the draw
--- Draw one beam. Returns the number of sparkles drawn.
function FX.draw(g, P, st, now, dt, dist)
  local lod = FX.lod(dist or 0)
  if not lod then return 0 end
  local mins, maxs = g.mins, g.maxs
  local sx, sy, sz = maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z
  local cx, cy = (mins.x + maxs.x) * 0.5, (mins.y + maxs.y) * 0.5
  local hx, hy = sx * 0.5, sy * 0.5
  local size = math.max(sx, sy, sz)
  local height = P.height or 1
  local H = sz * height                                  -- the column's current height
  local ox, oy, oz = g.pos.x, g.pos.y, g.pos.z
  local f, r, u = g.ang:Forward(), g.ang:Right(), g.ang:Up()
  local fx_, fy_, fz_, rx, ry, rz, ux, uy, uz = f.x, f.y, f.z, -r.x, -r.y, -r.z, u.x, u.y, u.z       -- local +y is the engine's -Right
  local tn = math.floor(now * P.jitter_hz)
  local c = P.condensation or 0
  local home = c * c                                      -- how much each sparkle is pulled onto the object
  local flow = now * (0.28 * (1 - 0.7 * c))               -- the cloud drifts upward
  local n = math.floor(P.density * FX.MAXP * lod)
  local sprite = math.max(0.8, math.min(3.0, size * 0.028)) * (0.7 + 0.5 * lod)
  local scatter = P.scatter or 0
  render.SetMaterial(material())
  local drawn = 0
  for i = 1, n do
    local fl = hash(i * 3.1 + tn * 0.37)
    local a = P.alpha * (1 - P.flicker * fl)
    if a > 0.03 then
      local jx, jy, jz = hash(i + tn * 1.31) * 2 - 1, hash(i * 2 + tn * 1.73) * 2 - 1, hash(i * 3 + tn * 2.11) * 2 - 1
      -- the cloud: a column wider than the footprint, flowing upward; the home: a point on the box
      local w = (CZ[i] + flow + PH[i]) % 1
      local lx = cx + CX[i] * hx * (1 + P.spread)
      local ly = cy + CY[i] * hy * (1 + P.spread)
      local lz = mins.z + w * sz * (1 + 0.5 * P.spread) * height
      local px = mins.x + HX[i] * sx
      local py = mins.y + HY[i] * sy
      local pz = mins.z + HZ[i] * H
      local x = lx + (px - lx) * home + jx * P.noise * sx * 0.5 + (scatter ~= 0 and (CX[i] * scatter * size * 0.5) or 0)
      local y = ly + (py - ly) * home + jy * P.noise * sy * 0.5 + (scatter ~= 0 and (CY[i] * scatter * size * 0.5) or 0)
      local z = lz + (pz - lz) * home + jz * P.noise * sz * 0.35 + (scatter ~= 0 and (scatter * size * 0.15 * (CZ[i] - 0.3)) or 0)
      if z > mins.z + H + size * 0.05 + (scatter * size * 0.3) then z = mins.z + H end
      V.x, V.y, V.z = ox + fx_ * x + rx * y + ux * z, oy + fy_ * x + ry * y + uy * z, oz + fz_ * x + rz * y + uz * z
      render.DrawSprite(V, sprite * SZ[i], sprite * SZ[i], set_col(P.colour, a))
      drawn = drawn + 1
    end
  end

  render.SetColorMaterial()
  -- the box's outline: shaky and faint at first, sharp when it has locked in
  if P.outline > 0.02 then
    local j = (1 - P.outline) * size * 0.06
    V.x, V.y, V.z = ox + hash(tn * 1.7) * j - j / 2, oy + hash(tn * 2.3) * j - j / 2, oz + hash(tn * 3.1) * j - j / 2
    local top = Vector(maxs.x, maxs.y, mins.z + H)
    render.DrawWireframeBox(V, g.ang, mins, top, set_col(P.colour, P.outline * (P.alpha_outline or 1)), true)
  end
  -- the silhouette of the final volume fills in
  if P.fill > 0.01 then
    render.DrawBox(g.pos, g.ang, mins, Vector(maxs.x, maxs.y, mins.z + H), set_col(P.colour, P.fill * 0.35), true)
  end
  -- the scan band
  local by, ba = P.band_y, P.band_alpha
  if not by then
    st.scan = (st.scan + (P.scan_hz or 0) * dt) % 1
    by, ba = TM.scan_y(st.scan), P.scan_alpha or 0
  end
  if ba and ba > 0.02 then
    local z = mins.z + by * H
    local function q(a, b, cc, d, zz)
      V1.x, V1.y, V1.z = ox + fx_ * a + rx * b + ux * zz, oy + fy_ * a + ry * b + uy * zz, oz + fz_ * a + rz * b + uz * zz
      V2.x, V2.y, V2.z = ox + fx_ * cc + rx * b + ux * zz, oy + fy_ * cc + ry * b + uy * zz, oz + fz_ * cc + rz * b + uz * zz
      V3.x, V3.y, V3.z = ox + fx_ * cc + rx * d + ux * zz, oy + fy_ * cc + ry * d + uy * zz, oz + fz_ * cc + rz * d + uz * zz
      V4.x, V4.y, V4.z = ox + fx_ * a + rx * d + ux * zz, oy + fy_ * a + ry * d + uy * zz, oz + fz_ * a + rz * d + uz * zz
    end
    local pad = 1 + 0.12 * P.spread
    q(cx - hx * pad, cy - hy * pad, cx + hx * pad, cy + hy * pad, z)
    local band = TM.mix(P.colour, TM.WHITE, 0.6)
    render.DrawQuad(V1, V2, V3, V4, set_col(band, ba * 0.8))
    render.DrawQuad(V4, V3, V2, V1, set_col(band, ba * 0.8))
    local th = math.max(1.2, sz * 0.03)                  -- a thin glowing slab above and below the sheet
    q(cx - hx * pad, cy - hy * pad, cx + hx * pad, cy + hy * pad, z + th)
    render.DrawQuad(V1, V2, V3, V4, set_col(P.colour, ba * 0.25))
    q(cx - hx * pad, cy - hy * pad, cx + hx * pad, cy + hy * pad, z - th)
    render.DrawQuad(V4, V3, V2, V1, set_col(P.colour, ba * 0.25))
  end
  return drawn
end

--- Run `fn` and fold its CPU time into FX.cost_ms: the smoothed TOTAL per frame of every beam drawn (entities and avatar beams alike),
--- and FX.drawn: the sparkles drawn in the last whole frame.
local acc_ms, acc_n, acc_frame = 0, 0, -1
function FX.timed(fn, ...)
  local t0 = SysTime()
  local n = fn(...)
  local fr = FrameNumber and FrameNumber() or 0
  if fr ~= acc_frame then                                        -- a new frame: fold the one that just ended
    if acc_frame >= 0 and acc_ms > 0 then FX.cost_ms = FX.cost_ms * 0.95 + acc_ms * 0.05; FX.drawn = acc_n end
    acc_ms, acc_n, acc_frame = 0, 0, fr
  end
  acc_ms = acc_ms + (SysTime() - t0) * 1000
  acc_n = acc_n + (n or 0)
  return n
end

-- ------------------------------------------------------------------------------------------------ beams that are not entities (Skippy)
--- Start a beam at a world position. kind "arrive" | "leave". `bot` (arrive): the beam follows him while it plays.
function FX.add_beam(kind, pos, bot, box)
  local dur = kind == "leave" and 0.8 or 1.0
  local mn, mx = Vector(-16, -16, -24), Vector(16, 16, 24)
  if box then mn, mx = box.mins, box.maxs end
  FX.beams[#FX.beams + 1] = { kind = kind, t0 = CurTime(), pos = pos, bot = bot, dur = dur, st = FX.state(), mins = mn, maxs = mx }
  if #FX.beams > 8 then table.remove(FX.beams, 1) end
end

--- True while his arrival beam has not completed: his body waits for it.
function FX.avatar_hidden()
  if FX.enabled and not FX.enabled:GetBool() then return false end
  local now = CurTime()
  for _, b in ipairs(FX.beams) do
    if b.kind == "arrive" and now - b.t0 < b.dur then return true end
  end
  return false
end

net.Receive("mentat_fx_beam", function()
  local kind = net.ReadBool() and "arrive" or "leave"
  local pos = net.ReadVector()
  local bot = net.ReadEntity()
  FX.add_beam(kind, pos, kind == "arrive" and IsValid(bot) and bot or nil)
end)

local function draw_beams()
  if not FX.beams[1] then return end
  if FX.enabled and not FX.enabled:GetBool() then FX.beams = {}; return end
  local now = CurTime()
  local me = LocalPlayer()
  local eye = IsValid(me) and me:EyePos() or Vector(0, 0, 0)
  local keep = {}
  for _, b in ipairs(FX.beams) do
    local s = now - b.t0
    if s < b.dur then
      keep[#keep + 1] = b
      local pos = (b.bot and IsValid(b.bot)) and b.bot:GetPos() or b.pos
      local P = TM.beam(b.kind, s, b.dur)
      local dt = math.min(0.1, now - (b.last or now))
      b.last = now
      FX.draw({ pos = pos, ang = Angle(0, 0, 0), mins = b.mins, maxs = b.maxs }, P, b.st, now, dt, eye:Distance(pos))
    end
  end
  FX.beams = keep
end

hook.Add("PostDrawTranslucentRenderables", "Mentat.FXBeams", function(_, skybox)
  if skybox then return end
  FX.timed(function() local ok, err = pcall(draw_beams); if not ok and not FX.error then FX.error = tostring(err); print("[fx] beam error: " .. FX.error) end end)
end)

Mentat.FX = FX
return FX
