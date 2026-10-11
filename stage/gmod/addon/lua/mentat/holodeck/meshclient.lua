-- meshclient.lua -- the client half of "the real shape of a found object": take the chunks the server sends ("holodeck_mesh"),
-- reassemble and verify them, decode the SVM1 payload (mentat/holodeck/svm1.lua) and build an IMesh the holodeck_found entity draws
-- in place of its box. Loaded by entities/holodeck_found/cl_init.lua. Nothing here trusts the network: a message from anyone but the
-- server cannot reach net.Receive on a client; sizes, counts and order are checked anyway; any failure leaves the box.
local SVM1 = include("mentat/holodeck/svm1.lua")

local CM = {}
CM.CAP_BYTES = 256 * 1024            -- the same cap as the server's
CM.MAX_RAW = 1024 * 1024
CM.MAX_MESHES = 64
CM.PARTIAL_TIMEOUT = 30              -- seconds a half-received mesh is kept
CM.WANT_EVERY = 8                    -- seconds between asks for one mesh
CM.HU_PER_M = 1 / 0.01905            -- Hammer units per metre
CM.FLIP = true                       -- Source draws clockwise triangles as front faces; glTF's outward faces are counter-clockwise
CM.LIGHT = { -0.30, -0.45, 0.84 }    -- a fixed light in the object's frame, baked into the vertex colours (the material is unlit)
CM.AMBIENT = 0.55

CM.store = {}                        -- mesh number -> { state = "ready" | "failed", imesh =, tris =, why = }
CM.partial = {}                      -- mesh number -> { n =, got =, count =, total =, crc =, t = }
CM.asked = {}                        -- mesh number -> time of the last ask
CM.stats = { chunks = 0, rejected = 0 }

local function now() return SysTime() end

--- The vertices of a decoded mesh as drawable triangles in the ENTITY's frame, Hammer units.
--- The payload's frame is metres, +Z up, x = the card's width, y = its depth, origin on the floor under the footprint centre; the
--- entity's box has x along the depth and y along the width, so the mesh is turned a quarter circle ((x, y) -> (-y, x)): the front
--- the model was authored with (glTF +Z, here -y) comes to +x, where the planner puts a thing's front. It is then moved to the box's
--- footprint centre and floor (`box` = { mins =, maxs = } entity-local).
function CM.triangles(dec, box)
  local k = CM.HU_PER_M
  local cx = box and (box.mins.x + box.maxs.x) / 2 or 0
  local cy = box and (box.mins.y + box.maxs.y) / 2 or 0
  local z0 = box and box.mins.z or 0
  local px, py, pz = {}, {}, {}
  for i = 1, dec.nv do
    px[i] = -dec.y[i] * k + cx
    py[i] = dec.x[i] * k + cy
    pz[i] = dec.z[i] * k + z0
  end
  local L = CM.LIGHT
  local ll = math.sqrt(L[1] * L[1] + L[2] * L[2] + L[3] * L[3])
  local tris = {}
  for t = 1, dec.nt do
    local a, b, c = dec.a[t], dec.b[t], dec.c[t]
    if CM.FLIP then b, c = c, b end
    local ux, uy, uz = px[b] - px[a], py[b] - py[a], pz[b] - pz[a]
    local vx, vy, vz = px[c] - px[a], py[c] - py[a], pz[c] - pz[a]
    -- (a, b, c) as authored is counter-clockwise seen from outside, so its cross product points out; when the order was flipped
    -- for the engine, the same normal is the cross product the other way round
    local sign = CM.FLIP and -1 or 1
    local nx, ny, nz = sign * (uy * vz - uz * vy), sign * (uz * vx - ux * vz), sign * (ux * vy - uy * vx)
    local len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if len > 0 then nx, ny, nz = nx / len, ny / len, nz / len else nx, ny, nz = 0, 0, 1 end
    local lit = math.max(0, (nx * L[1] + ny * L[2] + nz * L[3]) / ll)
    local shade = CM.AMBIENT + (1 - CM.AMBIENT) * lit
    tris[t] = {
      a = a, b = b, c = c, nx = nx, ny = ny, nz = nz,
      r = math.min(255, math.floor(dec.r[t] * shade + 0.5)), g = math.min(255, math.floor(dec.g[t] * shade + 0.5)),
      bl = math.min(255, math.floor(dec.bl[t] * shade + 0.5)),
    }
  end
  return tris, px, py, pz
end

--- Build the engine mesh. Returns the IMesh or nil and why.
function CM.build(dec, box)
  if not (Mesh and mesh and mesh.Begin) then return nil, "no mesh library" end
  local tris, px, py, pz = CM.triangles(dec, box)
  local ok, imesh = pcall(Mesh)
  if not ok or not imesh then return nil, "Mesh() failed" end
  local built, err = pcall(function()
    mesh.Begin(imesh, MATERIAL_TRIANGLES, #tris)
    for _, t in ipairs(tris) do
      for _, v in ipairs({ t.a, t.b, t.c }) do
        mesh.Position(Vector(px[v], py[v], pz[v]))
        mesh.Normal(Vector(t.nx, t.ny, t.nz))
        mesh.Color(t.r, t.g, t.bl, 255)
        mesh.TexCoord(0, 0, 0)
        mesh.AdvanceVertex()
      end
    end
    mesh.End()
  end)
  if not built then return nil, tostring(err) end
  return imesh, nil, #tris
end

--- Decode and build a verified payload, and keep the result under its number.
local function accept(id, blob, box)
  local size = SVM1.lzma_size(blob)
  if not size or size < 40 or size > CM.MAX_RAW then CM.store[id] = { state = "failed", why = "bad LZMA header" }; return false end
  local ok, raw = pcall(util.Decompress, blob)
  if not ok or type(raw) ~= "string" or #raw ~= size then CM.store[id] = { state = "failed", why = "util.Decompress failed" }; return false end
  local dec, why = SVM1.decode(raw)
  if not dec then CM.store[id] = { state = "failed", why = why }; return false end
  local imesh, why2, ntris = CM.build(dec, box)
  if not imesh then CM.store[id] = { state = "failed", why = why2 }; return false end
  CM.store[id] = { state = "ready", imesh = imesh, tris = ntris, bbox = { dec.min, dec.max } }
  return true
end
CM.accept = accept

--- The box the pose is relative to: the first holodeck_found entity of this mesh number that has one (a model's boxes are all alike).
CM.box_for = function(id)
  for _, e in ipairs(ents.FindByClass("holodeck_found")) do
    if IsValid(e) and e.GetMeshId and e:GetMeshId() == id then return { mins = e:GetBoxMins(), maxs = e:GetBoxMaxs() } end
  end
  return nil
end

--- A chunk arrived (see the wire description in meshnet.lua).
function CM.receive()
  local id = net.ReadUInt(8)
  local i = net.ReadUInt(8)
  local count = net.ReadUInt(8)
  local total = net.ReadUInt(20)
  local crc = net.ReadUInt(32)
  local n = net.ReadUInt(16)
  local function bad() CM.stats.rejected = CM.stats.rejected + 1 end
  if id < 1 or id > CM.MAX_MESHES or count < 1 or count > math.ceil(CM.CAP_BYTES / (16 * 1024)) or i < 1 or i > count then return bad() end
  if total < 1 or total > CM.CAP_BYTES or n < 1 or n > 16 * 1024 then return bad() end
  local data = net.ReadData(n)
  if type(data) ~= "string" or #data ~= n then return bad() end
  if CM.store[id] and CM.store[id].state == "ready" then return end          -- already have it (a second ask answered)
  local p = CM.partial[id]
  if p and (p.count ~= count or p.total ~= total or p.crc ~= crc or now() - p.t > CM.PARTIAL_TIMEOUT) then p = nil end
  if not p then p = { count = count, total = total, crc = crc, got = {}, n = 0, bytes = 0, t = now() }; CM.partial[id] = p end
  CM.stats.chunks = CM.stats.chunks + 1
  if p.got[i] then return end
  p.got[i] = data
  p.n = p.n + 1
  p.bytes = p.bytes + #data
  if p.bytes > CM.CAP_BYTES then CM.partial[id] = nil; return bad() end
  if p.n < p.count then return end
  CM.partial[id] = nil
  local blob = table.concat(p.got)
  if #blob ~= total or tonumber(util.CRC(blob)) ~= crc then
    CM.store[id] = { state = "failed", why = "length or checksum mismatch" }
    return bad()
  end
  accept(id, blob, CM.box_for(id))
end

--- Ask the server for a mesh (0 = all), at most once every few seconds per number.
function CM.want(id)
  local t = now()
  if CM.asked[id] and t - CM.asked[id] < CM.WANT_EVERY then return false end
  CM.asked[id] = t
  net.Start("holodeck_mesh_want")
  net.WriteUInt(id, 8)
  net.SendToServer()
  return true
end

--- How much of mesh `id` this client has received so far (chunks held / chunks expected, 0..1), or nil when none is arriving. This is
--- the per-chunk part of the transporter's "sending" progress.
function CM.recv_fraction(id)
  local p = CM.partial[id]
  if p and p.count > 0 then return p.n / p.count end
  return nil
end

--- The ready mesh for a number, or nil (the box is drawn meanwhile; a missing one is asked for).
function CM.get(id)
  if not id or id < 1 then return nil end
  local s = CM.store[id]
  if s and s.state == "ready" then return s end
  if not s then CM.want(id) end
  return nil
end

CM.material = nil
local function material()
  if not CM.material then
    CM.material = CreateMaterial("holodeck_found_mesh_v1", "UnlitGeneric", {
      ["$basetexture"] = "color/white", ["$vertexcolor"] = 1, ["$vertexalpha"] = 1, ["$translucent"] = 1,
    })
  end
  return CM.material
end

--- Draw a ready mesh at the entity's pose; `alpha` 0..1 is the holodeck's materialising fade.
function CM.draw(ent, s, alpha)
  local m = Matrix()                                       -- (Translate then Rotate: world = move * turn; SetTranslation + SetAngles misplaced it live)
  m:Translate(ent:GetPos())
  m:Rotate(ent:GetAngles())
  render.SetBlend(alpha or 1)
  render.SetMaterial(material())
  cam.PushModelMatrix(m)
  s.imesh:Draw()
  cam.PopModelMatrix()
  render.SetBlend(1)
end

net.Receive("holodeck_mesh", CM.receive)
Holodeck = Holodeck or {}
Holodeck.MeshClient = CM                                  -- (so a console command or a test can look at what this client holds)

--- `holodeck_mesh_info` prints what this client holds (the live proof reads it from console.log).
concommand.Add("holodeck_mesh_info", function()
  local n = 0
  for id, st in pairs(CM.store) do
    n = n + 1
    print(string.format("[holodeck] mesh %d: %s%s%s", id, st.state, st.tris and (", " .. st.tris .. " triangles") or "", st.why and (", " .. tostring(st.why)) or ""))
  end
  print(string.format("[holodeck] meshes held: %d; chunks received %d, rejected %d; FLIP=%s", n, CM.stats.chunks, CM.stats.rejected, tostring(CM.FLIP)))
end)

--- A client that has just loaded asks for everything the server already has.
hook.Add("InitPostEntity", "holodeck_mesh_hello", function() CM.want(0) end)

return CM
