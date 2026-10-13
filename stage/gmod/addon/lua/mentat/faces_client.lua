-- faces_client.lua -- the client half of "a player's picture is their head" (server half: faces_server.lua; the geometry and the model
-- policy: faces_head.lua). Receives the server's list of who wears which picture, receives the pictures (8 KB chunks, CRC checked), turns
-- each into a material, and draws a small runtime-mesh head where the player's head is.
--
--   * Real heads are replaced ONLY on a stock player model that has the head bone and whose head bone nobody else is animating
--     (Head.mode); a custom model is left completely alone unless its player said `skippy face on`. The real head is hidden by scaling
--     the head bone to zero and restored the moment the face goes (removed, hidden, blocked, switched off, model changed).
--   * With no player model at all (a server without the game's content) the head floats at the player's eyes.
--   * Never drawn in first person; the local player sees their own head in third person and in mirrors.
--   * Cost: two cached IMeshes shared by all heads (+ one back mesh per border tint), a matrix push and 2-3 mesh draws per visible head.
--   * THE UNVERIFIED PART is Material() from a file written to data/ (FC.load_material is the one small function that does it); when
--     the material is not valid the head is a plain neutral one.
if not CLIENT then return end

local Head = include("mentat/faces_head.lua")

local FC = MentatFacesClient or {}
MentatFacesClient = FC
FC.Head = Head
FC.DIR = "mentat_faces"
FC.CAP_BYTES = 24 * 1024
FC.PARTIAL_TIMEOUT = 20
FC.WANT_EVERY = 8
FC.HEAD_BONE = "ValveBiped.Bip01_Head1"
FC.map = FC.map or {}              -- steamid64 -> { slot, optin }
FC.images = FC.images or {}        -- slot -> { crc, border, mat (or nil), state = "ready" | "plain" }
FC.partial = FC.partial or {}
FC.asked = FC.asked or {}
FC.hidden = FC.hidden or setmetatable({}, { __mode = "k" })     -- player -> { bone, model }: heads we scaled to zero
FC.meshes = FC.meshes or {}
FC.stats = FC.stats or { drawn = 0, chunks = 0, rejected = 0, mesh_builds = 0 }

FC.cv_up = CreateClientConVar and CreateClientConVar("mentat_face_up", "3.5", true, false, "where the picture head's centre sits above the head bone (HU)")
FC.cv_fwd = CreateClientConVar and CreateClientConVar("mentat_face_fwd", "0.8", true, false, "...and in front of it (HU)")
FC.cv_scale = CreateClientConVar and CreateClientConVar("mentat_face_scale", "1.15", true, false, "size of the picture head (1 = a life-size head)")

local function now() return SysTime() end

-- ------------------------------------------------------------------------------------------------ the one function that makes a material
--- A material for a JPEG already written under data/ (path relative to data/). Returns an IMaterial or nil (then the head is plain).
--- UNVERIFIED in the engine until the live check: Material("data/<file>") for png/jpg, then the texture moved into an unlit,
--- vertex-coloured, two-sided material so the baked lighting multiplies the picture.
function FC.load_material(path, crc)
  if not (Material and CreateMaterial) then return nil end
  local ok, base = pcall(Material, "data/" .. path, "smooth")
  if not ok or not base or (base.IsError and base:IsError()) then return nil end
  local ok2, mat = pcall(CreateMaterial, "mentat_face_" .. tostring(crc), "UnlitGeneric",
    { ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 0, ["$nocull"] = 1 })
  if ok2 and mat and base.GetTexture then
    local tex = base:GetTexture("$basetexture")
    if tex then
      mat:SetTexture("$basetexture", tex)
      return mat
    end
  end
  return base
end

local plain_mat
--- The material for the plain parts (no picture): a white unlit vertex-coloured one that cares about nothing but the vertex colours.
function FC.plain_material()
  if plain_mat ~= nil then return plain_mat or nil end
  plain_mat = false
  if CreateMaterial then
    local ok, m = pcall(CreateMaterial, "mentat_face_plain", "UnlitGeneric", { ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 0, ["$nocull"] = 1 })
    if ok and m then plain_mat = m end
  end
  return plain_mat or nil
end

-- ------------------------------------------------------------------------------------------------ pictures arriving
local function jpeg(data) return type(data) == "string" and #data > 4 and data:sub(1, 3) == "\255\216\255" end

local function finish(slot, p)
  local data = table.concat(p.parts)
  FC.partial[slot] = nil
  if #data ~= p.total or #data > FC.CAP_BYTES or not jpeg(data) or tonumber(util.CRC(data)) ~= p.crc then
    FC.stats.rejected = FC.stats.rejected + 1
    return false
  end
  local old = FC.images[slot]
  if old and old.crc == p.crc and old.state == "ready" then return true end
  local path = FC.DIR .. "/" .. string.format("%08x", p.crc) .. ".jpg"
  file.CreateDir(FC.DIR)
  file.Write(path, data)
  local mat = FC.load_material(path, p.crc)
  FC.images[slot] = { crc = p.crc, border = p.border, mat = mat, state = mat and "ready" or "plain", path = path }
  return true
end

function FC.on_chunk(slot, crc, i, count, total, border, piece)
  if slot < 1 or slot > 64 or count < 1 or i < 1 or i > count or total < 1 or total > FC.CAP_BYTES or #piece == 0 then
    FC.stats.rejected = FC.stats.rejected + 1
    return false
  end
  FC.stats.chunks = FC.stats.chunks + 1
  local p = FC.partial[slot]
  if not p or p.crc ~= crc or p.count ~= count or p.total ~= total or now() - p.t > FC.PARTIAL_TIMEOUT then
    p = { crc = crc, count = count, total = total, border = border, parts = {}, got = 0, t = now() }
    FC.partial[slot] = p
  end
  if i ~= p.got + 1 then FC.partial[slot] = nil; return false end          -- chunks come in order on a reliable channel; anything else starts over
  p.parts[i], p.got, p.t = piece, i, now()
  if p.got == p.count then return finish(slot, p) end
  return true
end

net.Receive("mentat_face_img", function()
  local slot, crc, i, count, total = net.ReadUInt(8), net.ReadUInt(32), net.ReadUInt(8), net.ReadUInt(8), net.ReadUInt(16)
  local border = { net.ReadUInt(8), net.ReadUInt(8), net.ReadUInt(8) }
  local n = net.ReadUInt(16)
  if n < 1 or n > 16384 then return end
  FC.on_chunk(slot, crc, i, count, total, border, net.ReadData(n))
end)

function FC.want(slot)
  local t = now()
  if FC.asked[slot] and t - FC.asked[slot] < FC.WANT_EVERY then return end
  FC.asked[slot] = t
  net.Start("mentat_face_want")
  net.WriteUInt(slot, 8)
  net.SendToServer()
end

net.Receive("mentat_face_map", function()
  local n = net.ReadUInt(8)
  local map = {}
  for _ = 1, math.min(n, 128) do
    local sid, slot, optin = net.ReadString(), net.ReadUInt(8), net.ReadBool()
    if #sid == 17 then map[sid] = { slot = slot, optin = optin } end
  end
  FC.map = map
  FC.refresh()
end)

-- ------------------------------------------------------------------------------------------------ who, how
local function head_bone(ply)
  local b = ply.LookupBone and ply:LookupBone(FC.HEAD_BONE)
  return b and b >= 0 and b or nil
end

local function touched(ply, bone)
  local s = ply:GetManipulateBoneScale(bone)
  local a = ply:GetManipulateBoneAngles(bone)
  local p = ply:GetManipulateBonePosition(bone)
  return s.x ~= 1 or s.y ~= 1 or s.z ~= 1 or a.p ~= 0 or a.y ~= 0 or a.r ~= 0 or p.x ~= 0 or p.y ~= 0 or p.z ~= 0
end

local function enabled()
  local cv = GetConVar and GetConVar("mentat_faces")
  return not cv or cv:GetBool()
end

--- "bone" | "float" | nil for this player right now (see Head.mode).
function FC.mode_of(ply)
  if not IsValid(ply) or not enabled() then return nil end
  local entry = FC.map[ply:SteamID64()]
  local me = LocalPlayer()
  local bone = head_bone(ply)
  local hid = FC.hidden[ply]
  local model = ply:GetModel() or ""
  return Head.mode({
    entry = entry ~= nil, optin = entry and entry.optin, bone = bone,
    first_person = ply == me and not ply:ShouldDrawLocalPlayer(),
    has_bone = bone ~= nil, standard = Head.standard_model(model),
    bone_touched = bone ~= nil and not (hid and hid.bone == bone and hid.model == model) and touched(ply, bone),
    model_valid = model ~= "" and (not util.IsValidModel or util.IsValidModel(model)),
  })
end

--- Hide the real head on players who wear a picture and put it back on those who no longer do. Cheap: called on map changes and twice a second.
function FC.refresh()
  for _, ply in ipairs(player.GetAll()) do
    local mode = FC.mode_of(ply)
    local hid = FC.hidden[ply]
    local want = mode == "bone" and head_bone(ply) or nil
    if hid and (not want or hid.bone ~= want or hid.model ~= ply:GetModel()) then
      if IsValid(ply) then pcall(ply.ManipulateBoneScale, ply, hid.bone, Vector(1, 1, 1)) end
      FC.hidden[ply] = nil
      hid = nil
    end
    if want and not hid then
      ply:ManipulateBoneScale(want, Vector(0, 0, 0))
      FC.hidden[ply] = { bone = want, model = ply:GetModel() }
    end
  end
end

-- ------------------------------------------------------------------------------------------------ meshes (built once, cached)
local function build(tris, tint)
  if not (Mesh and mesh and mesh.Begin) then return nil end
  local ok, m = pcall(Mesh)
  if not ok or not m then return nil end
  local built = pcall(function()
    mesh.Begin(m, MATERIAL_TRIANGLES, #tris)
    for _, t in ipairs(tris) do
      for _, p in ipairs(t) do
        local r, g, b = Head.vertex_colour(p, tint)
        mesh.Position(Vector(p[1], p[2], p[3]))
        mesh.Normal(Vector(p[4], p[5], p[6]))
        mesh.Color(r, g, b, 255)
        mesh.TexCoord(0, p[7], p[8])
        mesh.AdvanceVertex()
      end
    end
    mesh.End()
  end)
  if not built then return nil end
  FC.stats.mesh_builds = FC.stats.mesh_builds + 1
  return m
end

local function cat(...)
  local out = {}
  for _, l in ipairs({ ... }) do for _, t in ipairs(l) do out[#out + 1] = t end end
  return out
end

--- The cached meshes: .front (white, carries the picture), and for a tint .back (rear + neck) or .full (the plain head).
function FC.get_meshes(tint)
  if not FC.geo then FC.geo = Head.geometry() end
  local M = FC.meshes
  if not M.front then M.front = build(FC.geo.front, nil) end
  local key = tint and (tint[1] .. "," .. tint[2] .. "," .. tint[3]) or "neutral"
  if not M[key] then
    if FC.count_tints and FC.count_tints >= 64 then
      for k, v in pairs(M) do if k ~= "front" and v and v.Destroy then v:Destroy(); M[k] = nil end end
      FC.count_tints = 0
    end
    FC.count_tints = (FC.count_tints or 0) + 1
    M[key] = { back = build(cat(FC.geo.back, FC.geo.neck), tint or Head.NEUTRAL), full = build(cat(FC.geo.front, FC.geo.back, FC.geo.neck), Head.NEUTRAL) }
  end
  return M.front, M[key]
end

-- ------------------------------------------------------------------------------------------------ drawing
function FC.head_matrix(ply, mode)
  local ea = ply:EyeAngles()
  local ang = Angle(math.Clamp(ea.p * 0.7, -45, 45), ea.y, 0)
  local centre
  if mode == "bone" then
    local bone = head_bone(ply)
    local bp = bone and ply:GetBonePosition(bone) or ply:EyePos()
    centre = bp + ang:Up() * FC.cv_up:GetFloat() + ang:Forward() * FC.cv_fwd:GetFloat()
  else
    centre = ply:EyePos() - ang:Up() * 1.5
  end
  local m = Matrix()
  m:SetTranslation(centre)
  m:SetAngles(ang)
  local s = FC.cv_scale:GetFloat()
  m:SetScale(Vector(s, s, s))
  return m
end

function FC.draw(ply, mode)
  local entry = FC.map[ply:SteamID64()]
  if not entry then return false end
  local img = FC.images[entry.slot]
  if not img then FC.want(entry.slot) end
  local plain = FC.plain_material()
  local textured = img and img.state == "ready" and img.mat
  local front, set = FC.get_meshes(textured and img.border or nil)
  if not set then return false end
  cam.PushModelMatrix(FC.head_matrix(ply, mode))
  if textured and front and plain then
    render.SetMaterial(img.mat); front:Draw()
    render.SetMaterial(plain); set.back:Draw()
  elseif plain and set.full then
    render.SetMaterial(plain); set.full:Draw()                  -- no picture (yet, or the material was not valid): the neutral head
  end
  cam.PopModelMatrix()
  FC.stats.drawn = FC.stats.drawn + 1
  return true
end

--- A drawing error must never be a spam of console errors: keep the first one (FC.draw_error) and stop drawing after 20.
local function guarded(fn, ...)
  if (FC.errors or 0) >= 20 then return end
  local ok, err = pcall(fn, ...)
  if not ok then
    FC.errors, FC.draw_error = (FC.errors or 0) + 1, FC.draw_error or tostring(err)
  end
end

hook.Add("PostPlayerDraw", "Mentat.Faces.Draw", function(ply)
  guarded(function() if FC.mode_of(ply) == "bone" then FC.draw(ply, "bone") end end)
end)

-- players whose model has no head bone to hide are not always "drawn" by PostPlayerDraw: float theirs from here
hook.Add("PostDrawOpaqueRenderables", "Mentat.Faces.Float", function()
  guarded(function()
    for _, ply in ipairs(player.GetAll()) do
      if FC.map[ply:SteamID64()] and FC.mode_of(ply) == "float" then FC.draw(ply, "float") end
    end
  end)
end)

timer.Create("Mentat.Faces.Refresh", 0.5, 0, function() guarded(FC.refresh) end)

-- the first thing a joining client does is ask for everything
hook.Add("InitPostEntity", "Mentat.Faces.Hello", function() FC.want(0) end)

hook.Add("PlayerDisconnected", "Mentat.Faces.Gone", function(ply) FC.hidden[ply] = nil end)

return FC
