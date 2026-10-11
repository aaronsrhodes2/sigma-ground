-- wisps.lua -- SERVER. The failure wisps (entity holodeck_wisp, state machine in mentat/fx/wisp.lua): when a found-object transport fails,
-- Skippy's reason is left behind as a small floating light where the object would have stood. Loaded by mentat/holodeck.lua.
--
--   Wisps.drop(ply, text, class, pos)   leave a wisp for `ply` (nil = the console); `text` is the sentence Skippy said, never made up here
--   Wisps.clear(ply)                    dismiss this player's wisps (they dissolve); returns how many
--   Wisps.spot(ply)                     a place about an arm's length in front of the player's eyes, kept out of walls (for an answer that has
--                                       no object to stand at: "I don't have a dragon on file")
--   Wisps.is_clear_command(text)        "clean up the notes" / "clear the wisps" / "dismiss the wisps"
--
-- A wisp is NOT a placed thing: it is not in the registry, so undo / clear / the per-player and server limits never count it, and a
-- "clear everything" leaves it alone. It has its own cap (convar holodeck_wisp_max per player, WISP_TOTAL on the server): the oldest
-- goes first, and a new wisp within 1.5 m of an old one of the same owner replaces it. Only the OWNER reads it (the dwell is accrued only
-- while the owner is within 4 m, looking toward it, with no wall between); nobody else can dismiss it early.
local W = include("mentat/fx/wisp.lua")

local Wisps = (Holodeck and Holodeck.Wisps) or {}
Wisps.WISP_TOTAL = 24                         -- wisps on the whole server, at most
Wisps.TICK = 0.1                              -- seconds between state-machine steps
Wisps.list = Wisps.list or {}                 -- { ent =, key =, pos = {x,y,z}, born =, dying = bool }
Wisps.made = Wisps.made or 0
Wisps.deps = Wisps.deps or {}

local cv_on = CreateConVar("holodeck_wisps", "1", FCVAR_ARCHIVE, "1 = a failed found-object transport leaves a floating wisp carrying Skippy's reason")
local cv_life = CreateConVar("holodeck_wisp_life", tostring(W.LIFE_S), FCVAR_ARCHIVE, "seconds before a wisp nobody read fades by itself")
local cv_max = CreateConVar("holodeck_wisp_max", tostring(W.CAP), FCVAR_ARCHIVE, "wisps one player may have at once (the oldest goes first)")
Wisps.cv_on, Wisps.cv_life, Wisps.cv_max = cv_on, cv_life, cv_max

--- deps = { key = fn(ply) -> the registry owner key }
function Wisps.init(deps) Wisps.deps = deps or {} end

local function key_of(ply)
  if Wisps.deps.key then return Wisps.deps.key(ply) end
  return (IsValid(ply) and ply.SteamID64 and ply:SteamID64()) or "console"
end

local function alive(w) return w.ent and IsValid(w.ent) and not w.dying end

local function prune()
  local keep = {}
  for _, w in ipairs(Wisps.list) do if IsValid(w.ent) then keep[#keep + 1] = w end end
  Wisps.list = keep
end

--- How many live (not dissolving) wisps this owner key has.
function Wisps.count(key)
  local n = 0
  for _, w in ipairs(Wisps.list) do if alive(w) and (key == nil or w.key == key) then n = n + 1 end end
  return n
end

-- ------------------------------------------------------------------------------------------------ the tick
local function looking_at(ent)
  local owner = ent:GetOwnerPly()
  if not (IsValid(owner) and owner.GetShootPos and owner:Alive()) then return false end
  local eye = owner:GetShootPos()
  local fwd = owner:EyeAngles():Forward()
  local pos = ent:GetPos()
  if not W.sees(eye, fwd, pos, true) then return false end                -- too far, or not looking toward it: no trace needed
  local tr = util.TraceLine({ start = eye, endpos = pos, mask = MASK_SOLID_BRUSHONLY })
  return not (tr and tr.Hit)                                              -- a wall between them hides it
end

function Wisps.tick()
  local now = CurTime()
  local dt = math.min(0.5, math.max(0, now - (Wisps.last or now)))
  Wisps.last = now
  for _, w in ipairs(Wisps.list) do
    if IsValid(w.ent) then
      local looking = (not w.dying) and looking_at(w.ent) or false
      w.ent:Step(dt, looking)
    end
  end
  prune()
  if #Wisps.list == 0 then timer.Remove("holodeck_wisps"); Wisps.last = nil end
end

local function ensure_timer()
  if not timer.Exists("holodeck_wisps") then
    Wisps.last = CurTime()
    timer.Create("holodeck_wisps", Wisps.TICK, 0, Wisps.tick)
  end
end

-- ------------------------------------------------------------------------------------------------ making and removing
local function retire(w)
  if not IsValid(w.ent) then return end
  w.dying = true
  w.ent:Dismiss()
end

--- Leave a wisp. class: "failure" | "suggest" | "refusal". Returns the entity, or nil and why.
function Wisps.drop(ply, text, class, pos)
  if not cv_on:GetBool() then return nil, "wisps are switched off" end
  if pos == nil then return nil, "no place" end
  text = W.cap_text(text, W.MAX_TEXT)
  if text == "" then return nil, "no text" end
  prune()
  local key = key_of(ply)
  local list = {}
  for i, w in ipairs(Wisps.list) do list[i] = { owner = alive(w) and w.key or "dying", pos = w.pos, born = w.born } end
  local cap = math.max(1, cv_max:GetInt())
  for _, i in ipairs(W.victims(list, key, { x = pos.x, y = pos.y, z = pos.z }, cap)) do retire(Wisps.list[i]) end
  if Wisps.count() >= Wisps.WISP_TOTAL then                                -- the server-wide bound: the oldest of anyone goes
    local oldest
    for _, w in ipairs(Wisps.list) do if alive(w) and (not oldest or w.born < oldest.born) then oldest = w end end
    if oldest then retire(oldest) end
  end
  local ent = ents.Create("holodeck_wisp")
  if not IsValid(ent) then return nil, "the wisp entity is not available" end
  local ok, why = ent:SetWisp(text, class, IsValid(ply) and ply or nil, cv_life:GetFloat() > 0 and cv_life:GetFloat() or W.LIFE_S)
  if not ok then ent:Remove(); return nil, why end
  ent:SetPos(pos)
  ent:Spawn()
  if not IsValid(ent) then return nil, "the wisp could not be made" end
  Wisps.made = Wisps.made + 1
  Wisps.list[#Wisps.list + 1] = { ent = ent, key = key, pos = { x = pos.x, y = pos.y, z = pos.z }, born = CurTime() }
  Wisps.last_text, Wisps.last_class = text, class
  ensure_timer()
  return ent
end

--- Dismiss every wisp this player owns (they dissolve). ply == nil: all of them (the console). Returns how many.
function Wisps.clear(ply)
  local key = ply ~= nil and key_of(ply) or nil
  local n = 0
  for _, w in ipairs(Wisps.list) do
    if alive(w) and (key == nil or w.key == key) then retire(w); n = n + 1 end
  end
  return n
end

--- A place in front of the player at about eye height, kept out of walls. nil when there is no player.
function Wisps.spot(ply)
  if not (IsValid(ply) and ply.GetShootPos) then return nil end
  local eye = ply:GetShootPos()
  local f = ply:EyeAngles():Forward()
  local len = math.sqrt(f.x * f.x + f.y * f.y)
  if len < 1e-3 then f = Vector(1, 0, 0) else f = Vector(f.x / len, f.y / len, 0) end
  local reach = 110
  local pos = Vector(eye.x + f.x * reach, eye.y + f.y * reach, eye.z - 6)
  local tr = util.TraceLine({ start = eye, endpos = pos, mask = MASK_SOLID_BRUSHONLY })
  if tr and tr.Hit then
    local back = math.max(30, reach * tr.Fraction - 24)
    pos = Vector(eye.x + f.x * back, eye.y + f.y * back, eye.z - 6)
  end
  return pos
end

--- Where a wisp for an object that stood on the floor hangs: eye height above its origin.
function Wisps.above(pos)
  return Vector(pos.x, pos.y, pos.z + 58)
end

-- ------------------------------------------------------------------------------------------------ the command
--- "clear the wisps", "clean up the notes", "dismiss all wisps", "get rid of the whisps": a request about the wisps and nothing else.
function Wisps.is_clear_command(text)
  local s = " " .. tostring(text or ""):lower():gsub("[%p]+", " "):gsub("%s+", " ") .. " "
  local verb = s:find(" clear ", 1, true) or s:find(" clean up ", 1, true) or s:find(" clean ", 1, true) or s:find(" dismiss ", 1, true)
               or s:find(" remove ", 1, true) or s:find(" get rid of ", 1, true) or s:find(" delete ", 1, true)
  local noun = s:find(" wisps? ") or s:find(" whisps? ") or s:find(" notes? ") or s:find(" will o the wisps? ")
  if not (verb and noun) then return false end
  -- anything else it names (a couch, a lamp) makes it a request for the holodeck, not for the wisps
  local rest = s
  for _, w in ipairs({ "please", "skippy", "hey", "can", "you", "could", "would", "the", "my", "all", "of", "those", "these", "every", "everything", "up", "clear", "clean", "dismiss",
                       "remove", "get", "rid", "delete", "wisps", "wisp", "whisps", "whisp", "notes", "note", "floating", "little", "small", "now", "for", "me", "will", "o" }) do
    rest = rest:gsub(" " .. w .. " ", " "):gsub(" " .. w .. " ", " ")
  end
  return rest:match("^%s*$") ~= nil
end

concommand.Add("mentat_wisps_clear", function(ply)             -- the same as saying "clear the wisps" (no ply: the server console clears them all)
  local n = Wisps.clear(IsValid(ply) and ply or nil)
  print("[holodeck] dismissed " .. n .. " wisp" .. (n == 1 and "" or "s"))
end)

hook.Add("PlayerDisconnected", "Holodeck.WispsLeave", function(ply)
  local key = key_of(ply)
  for _, w in ipairs(Wisps.list) do
    if w.key == key and IsValid(w.ent) then w.ent:Remove() end
  end
  prune()
end)

return Wisps
