-- meshclient.lua -- the client half of "the real shape of a found object": take the chunks the server sends ("holodeck_mesh"),
-- reassemble and verify them, decode the SVM1 payload (mentat/holodeck/svm1.lua) and build an IMesh the holodeck_found entity draws
-- in place of its box. Loaded by entities/holodeck_found/cl_init.lua. Nothing here trusts the network: a message from anyone but the
-- server cannot reach net.Receive on a client; sizes, counts and order are checked anyway; any failure leaves the box.
local SVM1 = include("mentat/holodeck/svm1.lua")
local Lit = include("mentat/lit.lua")                          -- the lighting and materials Skippy's body uses too

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
CM.cv_lit = CreateClientConVar and CreateClientConVar("holodeck_found_lit", "1", true, false, "1 = a found object's real shape is drawn lit (key, rim, fill, gloss, its own colours and materials); 0 = the old flat shading") or nil
local function lit_enabled() return Lit.LIT and (CM.cv_lit == nil or CM.cv_lit:GetBool()) end

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

--- SVM3 (colour regions, normals): one IMesh per palette entry, in the ENTITY's frame, Hammer units (see CM.triangles for the frame).
--- Lit drawing ignores vertex colours, so each entry is its own mesh drawn with its own material; the unlit fallback bakes colour and a fixed
--- light (Lit.shade from the vertex normal) into one vertex colour per vertex instead. Returns { parts = {...}, tris =, footprint = {ex, ey} } or nil, why.
function CM.build3(dec, box)
  if not (Mesh and mesh and mesh.Begin) then return nil, "no mesh library" end
  local k = CM.HU_PER_M
  local cx = box and (box.mins.x + box.maxs.x) / 2 or 0
  local cy = box and (box.mins.y + box.maxs.y) / 2 or 0
  local z0 = box and box.mins.z or 0
  local px, py, pz, qx, qy, qz = {}, {}, {}, {}, {}, {}
  for i = 1, dec.nv do
    px[i], py[i], pz[i] = -dec.y[i] * k + cx, dec.x[i] * k + cy, dec.z[i] * k + z0
    qx[i], qy[i], qz[i] = -dec.ny[i], dec.nx[i], dec.nz[i]                 -- the same quarter turn as the positions
  end
  local groups, order = {}, {}
  for t = 1, dec.nt do
    local g = groups[dec.pi[t]]
    if not g then g = {}; groups[dec.pi[t]] = g; order[#order + 1] = dec.pi[t] end
    g[#g + 1] = t
  end
  table.sort(order, function(a, b) return a < b end)
  local lit = lit_enabled()
  local parts, built = {}, {}
  local ok, err = pcall(function()
    for _, pidx in ipairs(order) do
      local e = dec.pal[pidx]
      local glow = e[6] > 20
      local list = groups[pidx]
      local imesh = Mesh()
      built[#built + 1] = imesh
      mesh.Begin(imesh, MATERIAL_TRIANGLES, #list)
      for _, t in ipairs(list) do
        local a, b, c = dec.a[t], dec.b[t], dec.c[t]
        if CM.FLIP then b, c = c, b end
        for _, v in ipairs({ a, b, c }) do
          mesh.Position(Vector(px[v], py[v], pz[v]))
          mesh.Normal(Vector(qx[v], qy[v], qz[v]))
          if lit or glow then
            mesh.Color(255, 255, 255, 255)
          else
            local sh = Lit.shade(qx[v], qy[v], qz[v])
            mesh.Color(math.min(255, math.floor(e[1] * sh + 0.5)), math.min(255, math.floor(e[2] * sh + 0.5)), math.min(255, math.floor(e[3] * sh + 0.5)), e[7])
          end
          mesh.TexCoord(0, 0, 0)
          mesh.AdvanceVertex()
        end
      end
      mesh.End()
      parts[#parts + 1] = { imesh = imesh, r = e[1], g = e[2], b = e[3], spec = e[4] / 255, metal = e[5] / 255, glow = glow, alpha = e[7], tris = #list }
    end
  end)
  if not ok then
    for _, m in ipairs(built) do if m.Destroy then pcall(m.Destroy, m) end end
    return nil, tostring(err)
  end
  return { parts = parts, tris = dec.nt, lit = lit, footprint = { (dec.max[2] - dec.min[2]) / 2 * k, (dec.max[1] - dec.min[1]) / 2 * k } }
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
  local dec, why = SVM1.decode_found(raw)
  if not dec then CM.store[id] = { state = "failed", why = why }; return false end
  if dec.svm3 then
    local b3, why3 = CM.build3(dec, box)
    if not b3 then CM.store[id] = { state = "failed", why = why3 }; return false end
    CM.store[id] = { state = "ready", imesh = b3.parts[1].imesh, parts = b3.parts, tris = b3.tris, lit = b3.lit, footprint = b3.footprint, bbox = { dec.min, dec.max } }
    return true
  end
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

--- A lit SVM3 object: a contact shadow, the engine's lights (set like Skippy's), one draw per palette entry with its own material.
--- Opaque entries first, then the see-through ones, then the glowing ones (flat, self-lit).
local function draw3(ent, s, alpha, m)
  local fadeing = (alpha or 1) < 0.99
  local pos = ent:GetPos()
  local lit = s.lit and Lit.LIT
  if lit then
    local mn, mx = ent:GetBoxMins(), ent:GetBoxMaxs()
    local foot = ent:LocalToWorld(Vector((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
    render.OverrideDepthEnable(true, false)                                  -- the shadow tests depth but never writes it
    Lit.draw_shadow(foot + Vector(0, 0, 6), math.max(s.footprint[1] * 1.8, s.footprint[1] + 6), math.max(s.footprint[2] * 1.8, s.footprint[2] + 6), ent:GetAngles().y, alpha or 1)
    render.OverrideDepthEnable(false, false)
  end
  local lights = lit and Lit.setup(pos + ent:GetUp() * ((ent:GetBoxMaxs().z + ent:GetBoxMins().z) / 2), ent:EntIndex())
  cam.PushModelMatrix(m)
  render.SetBlend(alpha or 1)
  for pass = 1, 2 do
    for _, p in ipairs(s.parts) do
      local see = p.alpha < 250
      if not p.glow and ((pass == 1 and not see) or (pass == 2 and see)) then
        if lights then
          render.SetMaterial(Lit.material(p.r, p.g, p.b, { spec = p.spec, metal = p.metal, alpha = p.alpha }, fadeing).mat)
        else
          render.SetMaterial(material())
        end
        p.imesh:Draw()
      end
    end
  end
  for _, p in ipairs(s.parts) do
    if p.glow then
      render.SetMaterial(Lit.glow_material(p.r, p.g, p.b))
      p.imesh:Draw()
    end
  end
  cam.PopModelMatrix()
  render.SetBlend(1)
  if lights then Lit.unlight() end
end

--- Draw a ready mesh at the entity's pose; `alpha` 0..1 is the holodeck's materialising fade.
function CM.draw(ent, s, alpha)
  local m = Matrix()                                       -- (Translate then Rotate: world = move * turn; SetTranslation + SetAngles misplaced it live)
  m:Translate(ent:GetPos())
  m:Rotate(ent:GetAngles())
  if s.parts then return draw3(ent, s, alpha, m) end
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
    print(string.format("[holodeck] mesh %d: %s%s%s%s", id, st.state, st.tris and (", " .. st.tris .. " triangles") or "", st.parts and (", " .. #st.parts .. " colour regions, " .. (st.lit and "lit" or "flat")) or "", st.why and (", " .. tostring(st.why)) or ""))
  end
  print(string.format("[holodeck] meshes held: %d; chunks received %d, rejected %d; FLIP=%s", n, CM.stats.chunks, CM.stats.rejected, tostring(CM.FLIP)))
end)

--- `holodeck_mesh_bench [tag]`: how long drawing the first found object takes the CPU (200 draws per frame inside the translucent pass,
--- timed with SysTime, averaged over about 1.5 s); prints "[fl-perf] tag draw_ms=... parts=... tris=...". The look pass reads it from console.log.
concommand.Add("holodeck_mesh_bench", function(_, _, args)
  local acc, n = 0, 0
  local tag = args and args[1] or "mesh"
  hook.Add("PostDrawTranslucentRenderables", "holodeck_mesh_bench", function(depth, sky)
    if sky or depth then return end
    local e = ents.FindByClass("holodeck_found")[1]
    local s = IsValid(e) and CM.get(e:GetMeshId())
    if not s then return end
    local t0 = SysTime()
    for _ = 1, 200 do CM.draw(e, s, 1) end
    acc = acc + (SysTime() - t0) / 200 * 1000
    n = n + 1
  end)
  timer.Simple(1.6, function()
    hook.Remove("PostDrawTranslucentRenderables", "holodeck_mesh_bench")
    local e = ents.FindByClass("holodeck_found")[1]
    local s = IsValid(e) and CM.store[e:GetMeshId()]
    print(string.format("[fl-perf] %s draw_ms=%.4f samples=%d parts=%d tris=%s lit=%s", tag, acc / math.max(n, 1), n, s and s.parts and #s.parts or 1, tostring(s and s.tris), tostring(s and s.lit)))
  end)
end, nil, "Time the CPU cost of drawing the first found object")

--- A client that has just loaded asks for everything the server already has.
hook.Add("InitPostEntity", "holodeck_mesh_hello", function() CM.want(0) end)

return CM
