-- library.lua -- which of the catalog's models THIS server can actually show. Pure Lua (the engine's `util.IsValidModel` is passed
-- in), so it is tested offline.
--
-- The catalog was measured on a machine with Half-Life 2 and Counter-Strike: Source content. A server or player without one of
-- them would see error models, so the holodeck lists only what exists, and (with `holodeck_content hl2`) only what every player
-- has: the HL2 models (models/props_*), leaving out the CS:S ones (models/props/...).
local Library = {}

--- Is this model one of Counter-Strike: Source's (rather than Half-Life 2's)?
function Library.is_css(model)
  return model:match("^models/props/") ~= nil
end

--- A copy of `catalog` with the unusable variants, and then the unusable kinds and recipe lines, taken out.
--- Returns (filtered, report) where report = { kept = n, total = n, lost_kinds = {...}, lost_models = {...}, trimmed_recipes = {...} }.
function Library.filter(catalog, is_valid, opts)
  opts = opts or {}
  local out = { version = catalog.version, hu_m = catalog.hu_m, order = {}, kinds = {}, recipes = {} }
  local report = { kept = 0, total = 0, lost_kinds = {}, lost_models = {}, trimmed_recipes = {} }
  for _, kid in ipairs(catalog.order) do
    local k = catalog.kinds[kid]
    local variants = {}
    for _, v in ipairs(k.variants) do
      report.total = report.total + 1
      local ok = (not is_valid) or is_valid(v.model)
      if ok and opts.hl2_only and Library.is_css(v.model) then ok = false end
      if ok then variants[#variants + 1] = v; report.kept = report.kept + 1
      else report.lost_models[#report.lost_models + 1] = v.model end
    end
    if #variants > 0 then
      local copy = {}
      for key, val in pairs(k) do copy[key] = val end
      copy.variants = variants
      out.kinds[kid] = copy
      out.order[#out.order + 1] = kid
    else
      report.lost_kinds[#report.lost_kinds + 1] = kid
    end
  end
  for name, lines in pairs(catalog.recipes or {}) do
    local kept = {}
    for _, line in ipairs(lines) do
      if out.kinds[line[2]] then kept[#kept + 1] = line end
    end
    -- a room is worth offering only if at least half of its furniture survived
    if #kept >= math.ceil(#lines * 0.5) and #kept > 0 then
      out.recipes[name] = kept
      if #kept < #lines then report.trimmed_recipes[#report.trimmed_recipes + 1] = name end
    else
      report.trimmed_recipes[#report.trimmed_recipes + 1] = name .. " (dropped)"
    end
  end
  return out, report
end

return Library
