-- avatar/body.lua -- the REAL Skippy (the print kit's six parts) as the avatar's body: the pure half. Shared by the server (what a
-- manifest must say before it is relayed) and the client (shading, vertex emission, the pointing hand, the head, the hat). No engine
-- calls, so all of it is tested offline. The private payloads themselves are NEVER in the repository (stage/skippyverse/private/,
-- gitignored); see stage/skippyverse/private/skippy/INTEGRATION.md and stage/gmod/AVATAR.md.
--
-- Frame: metres, +Z up, +X the way he faces, one origin on the floor under the figure's footprint centre for ALL parts. The client
-- draws in Hammer units (HU_PER_M) and turns the parts about the manifest's pivots.
local B = {}

B.PARTS = { "can", "jacket", "head", "hat", "hand_a", "hand_b" }
B.HU_PER_M = 1 / 0.01905
B.MAX_SWING = 35        -- degrees: the pointing hand never swings farther from the kit's pose (the arm is part of the coat)
B.MAX_NOD = 12          -- degrees: the hat's lean with his voice
B.MAX_HEAD_YAW = 25     -- degrees: the head turns toward the one he talks to
B.MAX_MANIFEST = 64 * 1024
B.DIMGREEN = { 0, 51, 34 }      -- #003322: the kit's black mouth is transparent on the glasses; tinted instead of left pure black

local function finite(v) return type(v) == "number" and v == v and v > -1e9 and v < 1e9 end
local function vec3(t)
  if type(t) ~= "table" then return nil end
  local x, y, z = t[1], t[2], t[3]
  if not (finite(x) and finite(y) and finite(z)) or math.abs(x) > 10 or math.abs(y) > 10 or math.abs(z) > 10 then return nil end
  return { x = x, y = y, z = z }
end

--- The numbers the animation needs from manifest.json (an already-decoded table). Returns a table or nil and why.
---   { height (m), head = {x,y,z}, hat = {x,y,z}, hat_top = {x,y,z}, wrist = {x,y,z}, tip = {x,y,z} }   (metres)
function B.parse_manifest(m)
  if type(m) ~= "table" then return nil, "manifest is not a table" end
  local p = m.pivots
  if type(p) ~= "table" then return nil, "manifest has no pivots" end
  local h = m.figure_height_m
  if not finite(h) or h < 0.05 or h > 5 then return nil, "bad figure height" end
  local out = { height = h }
  local function need(field, t, key)
    local v = vec3(t)
    if not v then return nil, "bad pivot " .. field end
    out[key] = v
    return true
  end
  local ok, why = need("head", p.head and p.head.pivot, "head")
  if not ok then return nil, why end
  ok, why = need("hat", p.hat and p.hat.pivot, "hat")
  if not ok then return nil, why end
  ok, why = need("hat.top", p.hat and p.hat.top, "hat_top")
  if not ok then return nil, why end
  ok, why = need("hand wrist", p.hand_point and p.hand_point.wrist, "wrist")
  if not ok then return nil, why end
  ok, why = need("hand tip", p.hand_point and p.hand_point.tip, "tip")
  if not ok then return nil, why end
  local dx, dy, dz = out.tip.x - out.wrist.x, out.tip.y - out.wrist.y, out.tip.z - out.wrist.z
  if dx * dx + dy * dy + dz * dz < 1e-6 then return nil, "the fingertip is at the wrist" end
  return out
end

-- ------------------------------------------------------------------------------------------------ colour and light
local KX, KY, KZ = 0.50, 0.30, 0.81                       -- the key light: above, a little to the front-left (his frame)
local FX, FY, FZ = -0.62, -0.50, 0.30                     -- a weak fill from behind-right, so his back is not black
local KL = math.sqrt(KX * KX + KY * KY + KZ * KZ)
local FL = math.sqrt(FX * FX + FY * FY + FZ * FZ)
B.AMBIENT = 0.50

--- Brightness 0..1 of a surface with this (unit) normal: ambient + a key + a fill, baked into the vertex colour (the material is unlit).
function B.shade(nx, ny, nz)
  local k = math.max(0, (nx * KX + ny * KY + nz * KZ) / KL)
  local f = math.max(0, (nx * FX + ny * FY + nz * FZ) / FL)
  return math.min(1, B.AMBIENT + 0.45 * k + 0.2 * f)
end

--- The kit's red (the tongue) is the one colour that glows with his voice.
function B.is_glow(r, g, b) return r >= 200 and g <= 60 and b <= 60 end

--- The colour a kit triangle is drawn in before shading: pure black (the mouth) becomes DimGreen.
function B.base_colour(r, g, b)
  if r == 0 and g == 0 and b == 0 then return B.DIMGREEN[1], B.DIMGREEN[2], B.DIMGREEN[3] end
  return r, g, b
end

--- Walk a decoded SVM2 part and hand every vertex of every triangle to `fn(glow, x, y, z, nx, ny, nz, r, g, b)` in HU, in the order
--- the engine wants (the kit's counter-clockwise faces are turned clockwise, Source's front). Returns the plain and glow triangle counts.
function B.emit(dec, fn)
  local k = B.HU_PER_M
  local plain, glow = 0, 0
  for t = 1, dec.nt do
    local r0, g0, b0 = dec.r[t], dec.g[t], dec.bl[t]
    local is_glow = B.is_glow(r0, g0, b0)
    local r, g, b = B.base_colour(r0, g0, b0)
    if is_glow then glow = glow + 1 else plain = plain + 1 end
    local order = { dec.a[t], dec.c[t], dec.b[t] }            -- (a, c, b): flipped
    for i = 1, 3 do
      local v = order[i]
      local nx, ny, nz = dec.nx[v], dec.ny[v], dec.nz[v]
      local s = is_glow and 1 or B.shade(nx, ny, nz)
      fn(is_glow, dec.x[v] * k, dec.y[v] * k, dec.z[v] * k, nx, ny, nz,
        math.min(255, math.floor(r * s + 0.5)), math.min(255, math.floor(g * s + 0.5)), math.min(255, math.floor(b * s + 0.5)))
    end
  end
  return plain, glow
end

-- ------------------------------------------------------------------------------------------------ rotations
local function cross(a, b) return { x = a.y * b.z - a.z * b.y, y = a.z * b.x - a.x * b.z, z = a.x * b.y - a.y * b.x } end
local function dot(a, b) return a.x * b.x + a.y * b.y + a.z * b.z end
local function unit(a)
  local l = math.sqrt(dot(a, a))
  if l < 1e-12 then return { x = 0, y = 0, z = 1 }, 0 end
  return { x = a.x / l, y = a.y / l, z = a.z / l }, l
end
B.unit, B.dot, B.cross = unit, dot, cross

--- The 3x3 rotation (row-major, 9 numbers) about the unit axis `ax` by `deg` degrees (Rodrigues).
function B.rodrigues(ax, deg)
  local a = math.rad(deg)
  local c, s = math.cos(a), math.sin(a)
  local x, y, z = ax.x, ax.y, ax.z
  local t = 1 - c
  return { t * x * x + c, t * x * y - s * z, t * x * z + s * y,
           t * x * y + s * z, t * y * y + c, t * y * z - s * x,
           t * x * z - s * y, t * y * z + s * x, t * z * z + c }
end

function B.apply(m, v)
  return { x = m[1] * v.x + m[2] * v.y + m[3] * v.z, y = m[4] * v.x + m[5] * v.y + m[6] * v.z, z = m[7] * v.x + m[8] * v.y + m[9] * v.z }
end

--- The turn that carries direction `rest` toward direction `want`, by the smallest angle, no more than `max_deg` (default MAX_SWING).
--- Returns the 3x3 matrix, the angle used (degrees) and the angle asked for.
function B.point_rotation(rest, want, max_deg)
  max_deg = max_deg or B.MAX_SWING
  local r, rl = unit(rest)
  local w, wl = unit(want)
  local ident = { x = 0, y = 0, z = 1 }
  if rl == 0 or wl == 0 then return B.rodrigues(ident, 0), 0, 0 end
  local d = math.max(-1, math.min(1, dot(r, w)))
  local ang = math.deg(math.acos(d))
  if ang < 1e-6 then return B.rodrigues(ident, 0), 0, 0 end
  local ax = cross(r, w)
  if dot(ax, ax) < 1e-12 then                              -- straight opposite: any axis across the rest direction will do
    ax = cross(r, math.abs(r.z) < 0.9 and { x = 0, y = 0, z = 1 } or { x = 1, y = 0, z = 0 })
  end
  ax = unit(ax)
  local used = math.min(ang, max_deg)
  return B.rodrigues(ax, used), used, ang
end

--- The 4x4 (row-major, 16 numbers) of "rotate by `m` about the point `p`": translate(p) * R * translate(-p).
function B.about(m, p)
  local rp = B.apply(m, p)
  return { m[1], m[2], m[3], p.x - rp.x, m[4], m[5], m[6], p.y - rp.y, m[7], m[8], m[9], p.z - rp.z, 0, 0, 0, 1 }
end

--- Degrees the head turns (about the vertical) to face (lx, ly) in his own frame (x forward, y left), at most MAX_HEAD_YAW, eased by `share`.
function B.head_yaw(lx, ly, share)
  if lx * lx + ly * ly < 1e-6 then return 0 end
  local a = math.deg(math.atan2(ly, lx))
  return math.max(-B.MAX_HEAD_YAW, math.min(B.MAX_HEAD_YAW, a * (share or 0.6)))
end

return B
