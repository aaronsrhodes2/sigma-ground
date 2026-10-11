-- testenv.lua -- synthetic worlds for the offline holodeck tests: the same three questions the engine answers
-- (where is the floor, is this footprint clear, how far to the wall along a ray), answered from plain geometry.
local M = {}

function M.open()
  return {
    floor_z = function() return 0 end,
    clear = function() return true end,
    wall_probe = function() return nil end,
  }
end

--- A rectangular room x0..x1, y0..y1 with optional box obstacles { {x0,y0,x1,y1}, ... } and an optional ceiling height (HU
--- above the floor, answered by env.ceiling_z). Outside is void (no floor).
function M.room(x0, y0, x1, y1, obstacles, ceiling)
  obstacles = obstacles or {}
  local env = {}
  local function inside(x, y, tol)
    tol = tol or 0
    return x >= x0 - tol and x <= x1 + tol and y >= y0 - tol and y <= y1 + tol
  end
  function env.floor_z(x, y)
    if inside(x, y, 0.01) then return 0 end
    return nil
  end
  function env.clear(r, z, h)
    local cs = {}
    local hw, hd = r.w / 2, r.d / 2
    for _, sd in ipairs({ { 1, 1 }, { 1, -1 }, { -1, -1 }, { -1, 1 } }) do
      local x = r.x + sd[1] * hd * r.fx + sd[2] * hw * r.rx
      local y = r.y + sd[1] * hd * r.fy + sd[2] * hw * r.ry
      if not inside(x, y, 0.6) then return false end
      for _, o in ipairs(obstacles) do
        if x > o[1] and x < o[3] and y > o[2] and y < o[4] then return false end
      end
    end
    for _, o in ipairs(obstacles) do
      local cx, cy = (o[1] + o[3]) / 2, (o[2] + o[4]) / 2
      if math.abs(r.x - cx) < (o[3] - o[1]) / 2 and math.abs(r.y - cy) < (o[4] - o[2]) / 2 then return false end
    end
    return true
  end
  function env.wall_probe(x, y, dx, dy, maxd)
    local best
    local function try(t, nx, ny)
      if t and t > 0.001 and t <= maxd and (not best or t < best.dist) then best = { dist = t, nx = nx, ny = ny } end
    end
    if dx > 1e-9 then try((x1 - x) / dx, -1, 0) elseif dx < -1e-9 then try((x0 - x) / dx, 1, 0) end
    if dy > 1e-9 then try((y1 - y) / dy, 0, -1) elseif dy < -1e-9 then try((y0 - y) / dy, 0, 1) end
    for _, o in ipairs(obstacles) do
      -- slab test against the obstacle box
      local tmin, tmax = -1e9, 1e9
      local nx, ny = 0, 0
      local ok = true
      for axis = 1, 2 do
        local p0 = axis == 1 and x or y
        local d = axis == 1 and dx or dy
        local lo = axis == 1 and o[1] or o[2]
        local hi = axis == 1 and o[3] or o[4]
        if math.abs(d) < 1e-9 then
          if p0 < lo or p0 > hi then ok = false end
        else
          local t1, t2 = (lo - p0) / d, (hi - p0) / d
          local n1 = -1
          if t1 > t2 then t1, t2 = t2, t1; n1 = 1 end
          if t1 > tmin then tmin = t1; nx, ny = (axis == 1) and n1 or 0, (axis == 2) and n1 or 0 end
          if t2 < tmax then tmax = t2 end
        end
      end
      if ok and tmin <= tmax and tmin > 0 then try(tmin, nx, ny) end
    end
    return best
  end
  if ceiling then
    function env.ceiling_z(x, y, z) return ceiling end
  end
  env.bounds = { x0, y0, x1, y1 }
  return env
end

return M
