-- fx/transporter_model.lua -- the PURE half of the transporter effect (no engine calls; tested offline with lupa). Shared: the server
-- uses the stage table and progress mapping (so every client and the server agree on what "40 % there" means), the client uses
-- everything. See stage/gmod/holodeck/README.md ("The transporter effect").
--
-- What the effect is for: a found object takes a while to arrive (resolve, download, pack, send, decode). While it is pending the
-- player sees a column of sparkles whose COHERENCE is the pipeline's real progress: sparse, wide and noisy at 0, condensing onto the
-- object's footprint as stages complete. A beam that has not heard from the pipeline for a while destabilises (jitter, dimming, amber);
-- one that is given up on collapses red. Nothing here animates by itself: every number is a function of (progress, stalled, time).
local T = {}

-- The stages a pending object goes through, in order. The index is the networked integer (0 = no effect: a plain box that was never
-- waiting for anything).
T.STAGES = { "resolving", "fetching", "measuring", "packing", "sending", "decoding", "done", "failed" }
T.STAGE_INDEX = {}
for i, s in ipairs(T.STAGES) do T.STAGE_INDEX[s] = i end

-- Where each stage sits on the 0..1 progress scale: [lo, hi]. The pipeline reports a fraction inside the stage (bytes downloaded of
-- bytes expected, chunks sent of chunks to send); the stage's band keeps the whole thing monotonic.
T.BANDS = {
  resolving = { 0.00, 0.08 }, fetching = { 0.08, 0.50 }, measuring = { 0.50, 0.58 }, packing = { 0.58, 0.74 },
  sending = { 0.74, 0.92 }, decoding = { 0.92, 1.00 }, done = { 1, 1 },
}

T.STALL_S = 6                -- seconds without a progress event before the beam is fully destabilised (convar holodeck_found_stall)
T.FAIL_S = 1.5               -- how long the failure effect lasts
T.ARRIVE_S = 0.8             -- how long the materialise (mesh reveal + beam fade) lasts
T.REVEAL_FRAC = 0.7          -- the mesh is fully revealed this far into the arrival; the beam fades for the rest
T.GREEN, T.AMBER, T.RED, T.CYAN, T.WHITE = { 0, 255, 136 }, { 255, 170, 0 }, { 255, 68, 68 }, { 0, 204, 255 }, { 255, 255, 255 }

local function clamp01(x)
  if x ~= x then return 0 end
  return x < 0 and 0 or (x > 1 and 1 or x)
end
local function smooth(x) x = clamp01(x); return x * x * (3 - 2 * x) end
local function lerp(a, b, k) return a + (b - a) * k end
local function mix(c1, c2, k) return { lerp(c1[1], c2[1], k), lerp(c1[2], c2[2], k), lerp(c1[3], c2[3], k) } end
T.clamp01, T.smooth, T.lerp, T.mix = clamp01, smooth, lerp, mix

--- The overall progress 0..1 for a stage and the fraction (0..1) done inside it. Unknown stages give nil.
function T.progress(stage, frac)
  local b = T.BANDS[stage]
  if not b then return nil end
  return lerp(b[1], b[2], clamp01(frac or 0))
end

--- The beam's parameters for a PENDING object. progress 0..1; stalled = seconds since the last progress event; t = seconds (kept for
--- callers; every amplitude here is already a function of progress and stalled alone). opts.stall_s overrides STALL_S.
---   noise     position jitter and cloud radius, as a fraction of the box's size   (falls as progress rises)
---   spread    how far past the footprint the cloud reaches, in box sizes            (falls)
---   density   fraction of the particle budget drawn                                 (rises: sparse at 0)
---   flicker   alpha jitter, 0 steady .. 1 wild                                      (falls with progress, rises with a stall)
---   alpha     overall brightness
---   outline   crispness of the box's outline, 0 faint/shaky .. 1 sharp               (rises)
---   fill      the silhouette of the final volume filling in                          (rises)
---   scan_hz   sweeps per second of the scan band; scan_alpha its brightness         (appears and quickens with progress)
---   jitter_hz how fast the jitter re-rolls                                          (faster when destabilised)
---   stall     0 healthy .. 1 fully destabilised
---   colour    {r,g,b}: green, drifting to amber as the stall grows
function T.coherence(progress, stalled, t, opts)
  opts = opts or {}
  local p = clamp01(progress)
  local stall_s = math.max(0.5, tonumber(opts.stall_s) or T.STALL_S)
  local k = smooth(((stalled or 0) - 0.35 * stall_s) / (0.65 * stall_s))          -- nothing for the first third of the window
  local c = smooth(p)                                                              -- condensation
  local inv = 1 - c
  return {
    progress = p, condensation = c, stall = k,
    noise = 0.04 + 0.62 * inv * math.sqrt(inv) + 0.30 * k,
    spread = 0.03 + 1.10 * inv,
    density = (0.16 + 0.84 * c) * (1 - 0.4 * k),
    flicker = math.min(1, 0.06 + 0.60 * inv + 0.60 * k),
    alpha = (0.45 + 0.55 * c) * (1 - 0.3 * k),
    outline = c * c * (1 - 0.7 * k),
    fill = c * c * c * 0.5 * (1 - 0.5 * k),
    scan_hz = (0.30 + 1.10 * c) * (1 - 0.75 * k),
    scan_alpha = clamp01((c - 0.08) / 0.92) * (1 - 0.6 * k),
    jitter_hz = 7 + 5 * inv + 18 * k,
    colour = mix(T.GREEN, T.AMBER, k),
  }
end

--- Where the scan band is (0 bottom .. 1 top) at a phase (the renderer integrates scan_hz * dt, so a changing rate never jumps it).
function T.scan_y(phase)
  local x = (phase or 0) % 1
  return x < 0.5 and x * 2 or (1 - x) * 2          -- up, then down
end

--- A pending beam's displayed progress eases toward the real one and never goes back (a stage can only add).
function T.ease(shown, target, dt)
  target = clamp01(target)
  if target <= shown then return shown end
  return math.min(target, shown + (target - shown) * math.min(1, 3.2 * math.max(0, dt)) + 0.002)
end

--- Seconds without news: the server stamps the time of the last progress event; the clients read it. Never negative.
function T.stalled_for(now, beat)
  if not beat or beat <= 0 then return 0 end
  return math.max(0, now - beat)
end

--- The failure effect, `s` seconds after the object was given up on. height 1 -> 0 (collapses from the top), flicker wild, colour amber
--- to red, particles scatter outward, alpha holds then fades; done = true once it is over. `from_progress` = how far it had got.
function T.failing(s, from_progress)
  local u = clamp01((s or 0) / T.FAIL_S)
  local p = clamp01(from_progress or 0.5)
  return {
    phase = "fail", u = u, done = u >= 1,
    height = 1 - smooth(clamp01((u - 0.08) / 0.82)),
    flicker = 0.85 - 0.25 * u,
    flicker_hz = 22 + 14 * u,
    colour = mix(T.AMBER, T.RED, smooth(u * 2.2)),
    scatter = 0.1 + 2.2 * u * u,
    alpha = (1 - smooth(clamp01((u - 0.55) / 0.45))) * (0.55 + 0.45 * (1 - u)),
    outline = (1 - smooth(u / 0.7)) * (0.35 + 0.65 * p),
    fill = 0.18 * (1 - u),
    density = 0.7 + 0.3 * (1 - u),
    noise = 0.08 + 0.5 * u,
  }
end

--- The materialise, `s` seconds after the mesh arrived. reveal 0..1 is how much of the mesh (from the floor up) is drawn; the scan
--- band rides the reveal; the shimmer collapses onto the object and the beam fades out. done = true once the object is plain.
function T.arriving(s)
  local u = clamp01((s or 0) / T.ARRIVE_S)
  local r = clamp01(u / T.REVEAL_FRAC)
  local fade = 1 - smooth(clamp01((u - T.REVEAL_FRAC) / (1 - T.REVEAL_FRAC)))
  return {
    phase = "arrive", u = u, done = u >= 1,
    reveal = smooth(r),
    band_alpha = r < 1 and 1 or fade,
    beam_alpha = fade,
    noise = 0.03 * (1 - r),
    density = 0.6 * fade,
    outline = 0.8 * fade,
    fill = 0,
    colour = mix(T.GREEN, T.WHITE, 0.35 * (1 - u)),
    flicker = 0.1 * fade,
  }
end

--- A short beam for the avatar: kind "arrive" (the column condenses, the figure appears when it is over) or "leave" (the column
--- dissolves). `s` seconds in, over `dur` (default 1.0 for arriving, 0.8 for leaving). Coherence runs 0.1 -> 1 on arrival and 1 -> 0.1
--- on leaving, so it is the same picture as a pending object played quickly (forwards or backwards). Returns coherence's table plus
--- kind, u, done, and `band` (the scan band's strength, swelling in the middle).
function T.beam(kind, s, dur)
  dur = dur or (kind == "leave" and 0.8 or 1.0)
  local u = clamp01((s or 0) / dur)
  local prog = kind == "leave" and (1 - smooth(u)) or smooth(u)
  local p = T.coherence(0.1 + 0.9 * prog, 0, s or 0)
  p.kind, p.u, p.done = kind, u, u >= 1
  local tail = 1 - smooth(clamp01((u - 0.85) / 0.15))           -- gone by the end
  p.alpha = p.alpha * (kind == "leave" and (1 - 0.5 * smooth(u)) or 1) * tail
  p.band = math.sin(math.pi * u) * tail
  return p
end

return T
