-- faces_head.lua -- the pure half of "a player's picture as their head" (shared by the server's tests and the client's drawing):
--   * the head's geometry: a rounded egg whose FRONT cap carries a planar projection of the picture (UVs in 0..1), whose back and
--     underside are drawn flat in the picture's border colour, plus a short neck so it sits on the shoulders. A few hundred triangles,
--     built once and shared by every head (the back is cached per tint).
--   * the policy for WHICH models get their head replaced (conservative: see Head.mode).
-- No engine calls here: it runs under plain LuaJIT in the offline tests. Frame: +x forward, +y left, +z up, origin at the head's centre,
-- Hammer units. Triangles are returned as { {x,y,z, nx,ny,nz, u,v} x 3 } with the winding Source calls front-facing (clockwise from
-- outside) -- and the drawing material is also two-sided, so a wrong guess about winding costs nothing but lighting.
local Head = {}

Head.DEPTH, Head.WIDTH, Head.HEIGHT = 5.2, 4.6, 6.4      -- ellipsoid radii (HU): a head about 13 HU tall, 9 wide
Head.SEGS, Head.FRONT_RINGS, Head.BACK_RINGS = 16, 6, 6
Head.CAP_DEG = 78                                        -- the front cap's half-angle: the picture is projected inside this circle
Head.NECK_RADIUS, Head.NECK_TOP, Head.NECK_BOTTOM = 0.42, -0.75, -2.0   -- fractions of the width / of the height
Head.LIGHT = { 0.45, 0.30, 0.84 }                        -- the key light in the head's own frame (above, in front, a little to the left)
Head.AMBIENT = 0.55
Head.NEUTRAL = { 205, 170, 150 }                         -- the plain head's colour when there is no picture
Head.MAX_TRIS = 600

local sin, cos, sqrt, rad, floor = math.sin, math.cos, math.sqrt, math.rad, math.floor

--- The brightness (AMBIENT..1) of a surface with this unit normal under the key light.
function Head.light(nx, ny, nz)
  local L = Head.LIGHT
  local ll = sqrt(L[1] * L[1] + L[2] * L[2] + L[3] * L[3])
  local d = (nx * L[1] + ny * L[2] + nz * L[3]) / ll
  return Head.AMBIENT + (1 - Head.AMBIENT) * (d > 0 and d or 0)
end

local function point(theta, phi, o)
  local st = sin(theta)
  local x, y, z = o.depth * cos(theta), o.width * st * cos(phi), o.height * st * sin(phi)
  local nx, ny, nz = cos(theta) / o.depth, st * cos(phi) / o.width, st * sin(phi) / o.height
  local l = sqrt(nx * nx + ny * ny + nz * nz)
  if l == 0 then nx, ny, nz, l = 1, 0, 0, 1 end
  return x, y, z, nx / l, ny / l, nz / l
end

-- Make the triangle (a, b, c) clockwise as seen from outside (cross product pointing inward) whatever order it was built in.
local function wind(a, b, c)
  local ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
  local vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
  local cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
  local mx, my, mz = (a[4] + b[4] + c[4]), (a[5] + b[5] + c[5]), (a[6] + b[6] + c[6])
  if cx * mx + cy * my + cz * mz > 0 then return { a, c, b } end
  return { a, b, c }
end

local function band(o, theta0, theta1, rings, with_uv)
  local tris = {}
  local segs = o.segs
  local sc = sin(rad(o.cap_deg))
  local function vert(i, j)
    local theta = theta0 + (theta1 - theta0) * i / rings
    local phi = 2 * math.pi * (j % segs) / segs
    local x, y, z, nx, ny, nz = point(theta, phi, o)
    local u, v = 0, 0
    if with_uv then
      u, v = 0.5 + 0.5 * (sin(theta) * cos(phi)) / sc, 0.5 - 0.5 * (sin(theta) * sin(phi)) / sc
      if u < 0 then u = 0 elseif u > 1 then u = 1 end
      if v < 0 then v = 0 elseif v > 1 then v = 1 end
    end
    return { x, y, z, nx, ny, nz, u, v }
  end
  for i = 0, rings - 1 do
    for j = 0, segs - 1 do
      local a, b, c, d = vert(i, j), vert(i + 1, j), vert(i + 1, j + 1), vert(i, j + 1)
      local degenerate_top = (theta0 + (theta1 - theta0) * i / rings) < 1e-9
      local degenerate_bottom = math.abs((theta0 + (theta1 - theta0) * (i + 1) / rings) - math.pi) < 1e-9
      if degenerate_top then
        tris[#tris + 1] = wind(a, b, c)
      elseif degenerate_bottom then
        tris[#tris + 1] = wind(a, b, d)
      else
        tris[#tris + 1] = wind(a, b, c)
        tris[#tris + 1] = wind(a, c, d)
      end
    end
  end
  return tris
end

local function neck(o)
  local tris = {}
  local r = o.neck_radius * o.width
  local zt, zb = o.neck_top * o.height, o.neck_bottom * o.height
  local function v(j, z)
    local phi = 2 * math.pi * (j % o.segs) / o.segs
    return { r * cos(phi), r * sin(phi), z, cos(phi), sin(phi), 0, 0, 0 }
  end
  for j = 0, o.segs - 1 do
    local a, b, c, d = v(j, zt), v(j, zb), v(j + 1, zb), v(j + 1, zt)
    tris[#tris + 1] = wind(a, b, c)
    tris[#tris + 1] = wind(a, c, d)
  end
  return tris
end

--- The head's three triangle lists: front (carries the picture, UVs 0..1), back (rear of the egg and the underside) and neck.
function Head.geometry(opts)
  local o = opts or {}
  local g = {
    depth = o.depth or Head.DEPTH, width = o.width or Head.WIDTH, height = o.height or Head.HEIGHT,
    segs = o.segs or Head.SEGS, cap_deg = o.cap_deg or Head.CAP_DEG,
    neck_radius = o.neck_radius or Head.NECK_RADIUS, neck_top = o.neck_top or Head.NECK_TOP, neck_bottom = o.neck_bottom or Head.NECK_BOTTOM,
  }
  local cap = rad(g.cap_deg)
  local front = band(g, 0, cap, o.front_rings or Head.FRONT_RINGS, true)
  local back = band(g, cap, math.pi, o.back_rings or Head.BACK_RINGS, false)
  return { front = front, back = back, neck = neck(g), opts = g }
end

--- Count and bounding box of triangle lists (for the tests and the status line).
function Head.stats(lists)
  local n, lo, hi = 0, { 1e9, 1e9, 1e9 }, { -1e9, -1e9, -1e9 }
  for _, tris in ipairs(lists) do
    for _, t in ipairs(tris) do
      n = n + 1
      for _, p in ipairs(t) do
        for k = 1, 3 do
          if p[k] < lo[k] then lo[k] = p[k] end
          if p[k] > hi[k] then hi[k] = p[k] end
        end
      end
    end
  end
  return n, lo, hi
end

--- A vertex's colour: white (front: it multiplies the picture) or the tint, times the key/ambient light. r, g, b in 0..255.
function Head.vertex_colour(p, tint)
  local k = Head.light(p[4], p[5], p[6])
  local t = tint or { 255, 255, 255 }
  return floor(t[1] * k + 0.5), floor(t[2] * k + 0.5), floor(t[3] * k + 0.5)
end

-- ------------------------------------------------------------------------------------------------ which models get a replaced head
-- The stock player models of the game. Anything else (an addon model, a custom playermodel, a model another addon has rigged with its own
-- head bone behaviour) is left completely alone unless the player says `skippy face on`.
Head.STANDARD = {}
for _, n in ipairs({ "alyx", "barney", "breen", "charple", "combine_soldier", "combine_soldier_prisonguard", "combine_super_soldier", "corpse1",
                     "eli", "gman_high", "kleiner", "magnusson", "monk", "mossman", "mossman_arctic", "odessa", "police", "police_fem",
                     "skeleton", "soldier_stripped", "zombie_classic", "zombiefast", "zombie_soldier", "arctic", "gasmask", "guerilla",
                     "leet", "phoenix", "riot", "swat", "urban" }) do
  Head.STANDARD["models/player/" .. n .. ".mdl"] = true
end
for _, g in ipairs({ "group01", "group02", "group03", "group03m" }) do
  for i = 1, 9 do
    Head.STANDARD[("models/player/%s/male_0%d.mdl"):format(g, i)] = true
    if i <= 7 then Head.STANDARD[("models/player/%s/female_0%d.mdl"):format(g, i)] = true end
  end
end
Head.STANDARD["models/player.mdl"] = true

function Head.standard_model(path)
  return Head.STANDARD[string.lower(tostring(path or ""))] == true
end

--- Decide how (whether) to draw a head for one player. info = { entry (the server listed a picture for them), optin, first_person,
--- has_bone, standard (stock model), bone_touched (another addon has scaled/moved the head bone), model_valid }.
--- Returns "bone" (hide the real head, draw ours at the bone), "float" (no head bone to hide: ours floats at the eyes) or nil (leave the
--- player exactly as they are).
function Head.mode(info)
  if not info.entry or info.first_person then return nil end
  if info.has_bone then
    if info.optin or (info.standard and not info.bone_touched) then return "bone" end
    return nil                                            -- a custom model, or someone else is already animating the head: hands off
  end
  if info.optin or info.model_valid == false then return "float" end   -- explicit yes, or no player model at all (a server without the game's content)
  return nil
end

return Head
