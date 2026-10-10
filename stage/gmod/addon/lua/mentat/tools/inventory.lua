-- inventory.lua -- list the .mdl files the game can load under some roots (VPKs included).
-- args: {roots = {"models/props_c17/", ...}, depth = 4}
return function(args, done)
  local out, counts = {}, {}
  local function walk(dir, depth)
    local files, dirs = file.Find(dir .. "*", "GAME")
    for _, f in ipairs(files) do
      if f:sub(-4) == ".mdl" then out[#out + 1] = dir .. f; counts[dir] = (counts[dir] or 0) + 1 end
    end
    if depth > 0 then for _, d in ipairs(dirs) do walk(dir .. d .. "/", depth - 1) end end
  end
  for _, root in ipairs(args.roots or {}) do walk(root, args.depth or 3) end
  table.sort(out)
  done({ count = #out, models = out })
end
