-- water/volumes.lua -- where the water is, as the world sees it (both realms).  Reads the holodeck_water entities and turns each into a
-- record in world coordinates: its footprint rotated by the entity's yaw, its floor and its surface height.
local M = {}
local inc = include or function(p) return dofile(p) end
local Basin = inc("mentat/water/basin.lua")
local C = inc("mentat/water/constants.lua")
local cos, sin, rad = math.cos, math.sin, math.rad

M.CLASS = "holodeck_water"

--- The record of one water entity: { ent, b (basin), x, y, z (origin, HU), c, s (cos/sin yaw), floor, surf (world z, HU), fill, liquid (key), L (the liquid's row) }.
function M.record(ent)
  if not (ent and IsValid(ent)) or not ent.GetBasin then return nil end
  local b = ent:GetBasin()
  if not b then return nil end
  local pos, ang = ent:GetPos(), ent:GetAngles()
  local yaw = rad(ang.y)
  local fill = ent:GetFill()
  local key = ent:GetLiquid()
  local L = C.LIQUIDS[key] or C.LIQUIDS.water
  return { ent = ent, b = b, x = pos.x, y = pos.y, z = pos.z, c = cos(yaw), s = sin(yaw), floor = pos.z + b.floor_z,
           surf = pos.z + Basin.surface_z(b, fill), fill = fill, liquid = C.LIQUIDS[key] and key or "water", L = L }
end

function M.list()
  local out = {}
  for _, e in ipairs(ents.FindByClass(M.CLASS)) do
    local r = M.record(e)
    if r then out[#out + 1] = r end
  end
  return out
end

--- The local (lx, ly) of a world point in a record's frame.
function M.to_local(r, x, y)
  local dx, dy = x - r.x, y - r.y
  return dx * r.c + dy * r.s, -dx * r.s + dy * r.c
end

--- Is the world point (x, y) over this body's water (inside its footprint, `margin` HU extra all round)?
function M.inside_xy(r, x, y, margin)
  margin = margin or 0
  local lx, ly = M.to_local(r, x, y)
  local b = r.b
  if b.shape == "rect" then return lx >= -b.hx - margin and lx <= b.hx + margin and ly >= -b.hy - margin and ly <= b.hy + margin end
  local u, v = lx / (b.hx + margin), ly / (b.hy + margin)
  return u * u + v * v <= 1
end

--- The first record whose water holds the point (x, y, z), or nil.  Between the floor and the surface (water only: a drained body has none).
function M.holding(list, x, y, z)
  for _, r in ipairs(list) do
    if r.fill > 0 and z >= r.floor and z <= r.surf and M.inside_xy(r, x, y, 0) then return r end
  end
  return nil
end

return M
