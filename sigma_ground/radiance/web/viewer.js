/* Radiance web viewer — WebGL2 SDF raymarcher, zero dependencies.
 *
 * Loads a SceneSpec (Deckard's CSG tree, baked colors) and a Trajectory
 * (Materia's poses), generates a GLSL raymarch shader from the tree, and plays
 * it with a free orbit camera and a time-rate knob. Per-material colors are
 * BAKED into the shader as constants (no array-uniform pitfalls). A Python
 * ground-truth self-check confirms the browser rebuilds the exact same SDF.
 */
"use strict";
(() => {
const canvas = document.getElementById("gl");
const $ = (id) => document.getElementById(id);
const setErr = (m) => { $("check").innerHTML = `<span style="color:#FF4444">ERROR: ${m}</span>`; };
const setOk  = (m) => { $("check").innerHTML = m; };

// ── probe mode (?probe=1): the browser gauntlet's measurement hook ──────
// Collects console errors + the in-page SDF self-check result and POSTs them
// to /probe once the first frame has drawn — headless verification, additive.
const PROBE = new URLSearchParams(location.search).get("probe") === "1";
const probeErrors = [];
let probeSent = false;
if (PROBE) {
  const _ce = console.error.bind(console);
  console.error = (...a) => { probeErrors.push(a.map(String).join(" ").slice(0, 300)); _ce(...a); };
  window.addEventListener("error", e => probeErrors.push(String(e.message || e).slice(0, 300)));
  fetch("/probe", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({slug: "__boot_" + (new URLSearchParams(location.search).get("scene") || "x")})}).catch(() => {});
}

// ── capture mode (?probe=1&capture=<id>): R0 pixel-regression hook ──────
// Renders one PINNED, deterministic exposure and POSTs the raw framebuffer
// to /capture (tools/render_regression.py drives this; serve.py is the sink).
// EVERY pin below is gated behind CAP being non-null, so plain ?probe=1 (the
// gauntlet) and normal viewing stay byte-identical — no render-behavior
// change outside capture mode. The nondeterminism sources and their pins:
// (a) uFrame seed → accN, the accumulation index; (b) ptTime wall
// clock → virtual accN/60 shutter clock; (c) backing-store size → exact
// px=WxH early-return in resize(); (d) camera/simTime → explicit floats
// applied AFTER load() resolves (so applyObj's camera reset can't undo
// them), with the legacy scene.camera.follow write guarded in loop();
// (e) device input → input.js polling frozen in capture mode (measured:
// a resting DualSense's stick drift walked az by +1.8 rad mid-capture);
// (f) chat.js's ?scene= deep-link auto-load is suppressed on capture tabs
// AND applyObj re-applies the pins when armed — a second load's
// preserve=false camera/playhead reset must never race the pins. Finished
// capture tabs also release their GL context (capReleaseGL) so a runner's
// tab pile can't starve the next tab's context/GPU memory.
const _capQ = new URLSearchParams(location.search);
const CAP = (PROBE && _capQ.get("capture")) ? {
  id: _capQ.get("capture"),
  size: (_capQ.get("px") || "").match(/^(\d+)x(\d+)$/),      // exact backing-store pixels
  frames: parseInt(_capQ.get("frames") || "64", 10),         // accumulation length (water: <=24, the shutter plateau)
  cam: (_capQ.get("cam") || "").split(",").map(parseFloat),  // az,el,radius,tx,ty,tz — EXPLICIT floats, never locateBody
  t: _capQ.get("t"),                                         // absolute sim seconds (trajectory scenes)
} : null;
let capSent = false, capArmed = false;
function applyCapturePins(){
  // Runs after the dispatcher's load() resolves — load is async, and
  // applyObj resets the camera from scene.camera on a fresh load, so the
  // pins must land after that reset to win.
  if(CAP.cam.length === 6 && CAP.cam.every(isFinite)){
    cam.az = CAP.cam[0]; cam.el = CAP.cam[1]; cam.radius = CAP.cam[2];
    cam.target = CAP.cam.slice(3, 6);
  }
  if(CAP.t != null){ simTime = parseFloat(CAP.t); playing = false; bodyPoses = posesAt(simTime); }
  sceneId++;                 // restart accumulation under the pinned exposure key
  capArmed = true;
}
function captureAndPost(w, h){
  // MUST run in the same rAF tick as draw(): preserveDrawingBuffer stays
  // false, so the default framebuffer is only readable before compositing.
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);  // display-pass output: post-gamma 8-bit — exactly what a human sees
  let b64 = "";
  for(let i = 0; i < px.length; i += 8190){                  // 8190 = 3*2730 bytes → every btoa chunk is padding-free except the last
    b64 += btoa(String.fromCharCode.apply(null, px.subarray(i, Math.min(i + 8190, px.length))));
  }
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  fetch("/capture", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      id: CAP.id, slug: _scene, w, h, frames: CAP.frames,
      renderer: dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER)),
      ua: navigator.userAgent,
      cam: {az: cam.az, el: cam.el, radius: cam.radius, target: cam.target.slice()}, t: simTime,
      selfcheck_pass: window.__probeSelfcheck, console_errors: probeErrors.slice(0, 10),
      rgba_b64: b64})}).catch(() => {});
}
function capReleaseGL(){
  // A finished capture tab must not HOLD a WebGL context: the browser caps
  // live contexts per origin (~16 in Chromium) and a runner's pile of done
  // tabs would evict/starve the NEXT capture tab's context. The payload is
  // already a JS string by the time this runs, so losing the context is
  // side-effect-free for the POST. Capture-mode only.
  try{ const lc = gl.getExtension("WEBGL_lose_context"); if(lc) lc.loseContext(); }catch(_e){}
}

const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
if (!gl) { setErr("WebGL2 not available in this browser"); return; }
const VAO0 = gl.createVertexArray();             // some drivers require a bound VAO
gl.bindVertexArray(VAO0);                        // (R3 keeps a ref: the splat pass binds its own VAO, then restores this one)
gl.clearColor(0, 0, 0, 1);
const extF = gl.getExtension("EXT_color_buffer_float");   // float render targets → progressive PT accumulation
// R3 capability log (informational — the splat target is RGBA16F, whose
// blending is core WITH extF; EXT_float_blend would only matter for 32F
// blending, which R3 deliberately avoids touching the accumulation buffer).
const extFB = gl.getExtension("EXT_float_blend");
const _ptRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
console.log(`R3 caps: EXT_float_blend=${!!extFB} ALIASED_POINT_SIZE_RANGE=[${_ptRange[0]},${_ptRange[1]}]`);
if (_ptRange[1] < 3) console.warn(
  `R3: max point size ${_ptRange[1]} < 3 — splat sprites capped; near the 3 px seam a capped sprite `
  + `under-delivers up to (3/max)^2 of its flux (the flux clamp binds). Disclosed, not silent.`);

// ── vec3 helpers ──────────────────────────────────────────────────────
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const len=(a)=>Math.hypot(a[0],a[1],a[2]);
const norm=(a)=>{const L=len(a)||1;return [a[0]/L,a[1]/L,a[2]/L];};
function glf(x){ let s=(+x).toFixed(8); if(s.indexOf(".")<0) s+=".0"; return s; }
// quaternion (x,y,z,w): inverse-rotate a vector (Rᵀv), and shortest-path nlerp
function qrotInvJS(q,v){ const u=[-q[0],-q[1],-q[2]], t=mul(cross(u,v),2);
  return add(add(v,mul(t,q[3])),cross(u,t)); }
function qrotJS(q,v){ const u=[q[0],q[1],q[2]], t=mul(cross(u,v),2);   // forward rotate (emitter world pos)
  return add(add(v,mul(t,q[3])),cross(u,t)); }
function nlerp(a,b,u){ const d=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3], s=d<0?-1:1;
  const q=[a[0]+(b[0]*s-a[0])*u,a[1]+(b[1]*s-a[1])*u,a[2]+(b[2]*s-a[2])*u,a[3]+(b[3]*s-a[3])*u];
  const L=Math.hypot(q[0],q[1],q[2],q[3])||1; return [q[0]/L,q[1]/L,q[2]/L,q[3]/L]; }

// ── primitive SDFs (GLSL + JS, both match shapes.py exactly) ───────────
const PRIMS_GLSL = `
float sdSphere(vec3 p, vec3 c, float r){ return length(p-c)-r; }
float sdCyl(vec3 p, vec3 c, float r, float h){ vec3 q=p-c;
  float dr=length(q.xy)-r, da=abs(q.z)-h*0.5;
  return min(max(dr,da),0.0)+length(max(vec2(dr,da),0.0)); }
float sdBox(vec3 p, vec3 c, vec3 b){ vec3 d=abs(p-c)-b*0.5;
  return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0)); }
// Ellipsoid (semi-axes r): IQ's 2nd-order bound — accurate even for a thin vane,
// and never overshoots, so sphere-tracing won't skip it. Surface (k0==1) is the
// exact (x/rx)²+(y/ry)²+(z/rz)²=1 the kernel integrates.
float sdEllip(vec3 p, vec3 c, vec3 r){ vec3 q=p-c;
  float k0=length(q/r); float k1=length(q/(r*r)); return k0*(k0-1.0)/max(k1,1e-9); }
// Right circular cone about +z, matching shapes.py: center is the centre of mass
// (h/4 above the base), so base sits at local z=0, apex at z=h; r(z)=R(1−z/h).
float sdConeZ(vec3 p, vec3 c, float R, float H){ vec3 q=p-c;
  float lz=q.z+H*0.25; float rho=length(q.xy);
  if(lz<0.0){ float dr=max(rho-R,0.0); return length(vec2(dr,-lz)); }
  if(lz>H){ return length(vec2(rho,lz-H)); }
  float rAtZ=R*(1.0-lz/H); float slant=sqrt(R*R+H*H);
  float dLat=(rho-rAtZ)*H/slant;
  return (rho<=rAtZ) ? max(dLat,-lz) : dLat; }
vec3 qrotInv(vec4 q, vec3 v){ vec3 u=-q.xyz; vec3 t=2.0*cross(u,v); return v+q.w*t+cross(u,t); }
float sdTorus(vec3 p, vec3 c, float R, float r){ vec3 q=p-c; return length(vec2(length(q.xy)-R, q.z))-r; }
// Quilez polynomial smooth-min union, blend radius k (k<=0 -> exact min). Mirrors
// kernel/csg.py's sdf_smooth_union exactly (used by InvoluteGear's root fillet).
float sdSmoothUnion(float a, float b, float k){
  if(k<=0.0) return min(a,b);
  float h=max(k-abs(a-b),0.0)/k;
  return min(a,b)-h*h*k*0.25; }
// Involute spur gear (axis +z), extruded to face width. Mirrors kernel/gear.py's
// InvoluteGear._gear_sdf_2d + _local_surface_distance EXACTLY, including both
// CRITICAL clamps in the nearest-involute-t calc (rho<=r_b -> t*=0; the max(0,.)
// floor on t*). r_b/r_a/r_f/beta/thetaRb/filletR/halfW are DERIVED constants
// baked in by the JS caller (glslLeafCall) via the same _gearParams() used by
// jsGearSDF, so the two mirrors can never disagree on the derivation step.
float sdInvoluteGear(vec3 p, vec3 c, float r_b, float r_a, float r_f, float beta,
                      float thetaRb, float filletR, float halfW){
  vec3 q = p - c;
  float rho = length(q.xy);
  float phi = atan(q.y, q.x);
  float twoBeta = 2.0*beta;
  float phiFoldedSigned = mod(phi+beta, twoBeta) - beta;
  float phiFolded = abs(phiFoldedSigned);
  // gear frame -> involute frame is a REFLECTION (flank curls CW, involute
  // CCW): phiInv = thetaRb - phiFolded, and the query point is rebuilt with
  // the SAME reflected angle (isometry) — see kernel/gear.py _gear_sdf_2d
  float phiInv = thetaRb - phiFolded;
  float tStar;
  if(rho <= r_b){
    tStar = 0.0;
  } else {
    float arc = acos(clamp(r_b/rho, -1.0, 1.0));
    tStar = max(0.0, phiInv + arc);              // CRITICAL clamp: floor at 0
  }
  float cosT = cos(tStar), sinT = sin(tStar);
  float cxT = r_b*(cosT + tStar*sinT);
  float cyT = r_b*(sinT - tStar*cosT);
  vec2 qFlank = vec2(rho*cos(phiInv), rho*sin(phiInv));
  float dFlankMag = length(qFlank - vec2(cxT, cyT));
  float ratio = rho > 1e-15 ? (r_b/rho) : 1.0;
  float alphaRho = acos(clamp(ratio, -1.0, 1.0));
  float thetaRho = thetaRb - (tan(alphaRho) - alphaRho);   // theta(rho) = thetaRb - inv(alpha_rho)
  float dFlank = (phiFolded < thetaRho) ? -dFlankMag : dFlankMag;
  float dAddendum = rho - r_a;
  float dRootDisk = rho - r_f;
  float dToothWedge = max(dFlank, dAddendum);
  float dGear2d = sdSmoothUnion(dToothWedge, dRootDisk, filletR);
  float dExtrude = abs(q.z) - halfW;
  if(dGear2d<=0.0 && dExtrude<=0.0) return max(dGear2d, dExtrude);
  if(dGear2d>0.0 && dExtrude>0.0) return length(vec2(dGear2d, dExtrude));
  return max(dGear2d, dExtrude); }
vec3 leafRot(vec4 rot, vec3 p, vec3 c){ return qrotInv(rot, p-c)+c; }
// Exact SDF for a capped cylinder between two arbitrary world points a→b
// (Quilez's closed form) — used only by the ghost-tether Plug-disclosure
// marker (see buildPathShader), which needs an axis that is NOT
// necessarily x/y/z-aligned: a RevoluteJoint's motor axis is an arbitrary
// world direction, unlike every other leaf's local-frame shape.
float sdCapsuleCyl(vec3 p, vec3 a, vec3 b, float r){
  vec3 ba=b-a, pa=p-a; float baba=dot(ba,ba), paba=dot(pa,ba);
  float x=length(pa*baba-ba*paba)-r*baba, y=abs(paba-baba*0.5)-baba*0.5;
  float x2=x*x, y2=y*y*baba;
  float dd=(max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
  return sign(dd)*sqrt(abs(dd))/baba; }
`;
// ── the ONE incandescence chunk (R3) ─────────────────────────────────────
// Planck × Kirchhoff blackbody glow, extracted from the two shader builders
// so the PT, the fast shim AND the R3 splat vertex shader consume the SAME
// text — physics claims derive from one shared derivation, never parallel
// math that can drift. Consumers must #define EMISSION_SCALE before it.
const INCANDESCENCE_GLSL = `
vec3 incandescence(float T, vec3 emis){   // Planck's law × Kirchhoff emissivity — nature's glow, no colour table
  if(T < 700.0) return vec3(0.0);          // Draper point: below ~700 K emission is entirely IR
  vec3 lam=vec3(650e-9,550e-9,450e-9);
  vec3 x=1.4388e-2/(lam*T);                // c2/(lambda T), c2 = hc/k = 1.4388e-2 m*K
  vec3 L=emis/(pow(lam/650e-9,vec3(5.0))*(exp(x)-vec3(1.0)));   // ε(λ)·B(λ,T), spectral radiance
  vec3 e=L*EMISSION_SCALE;
  // NON-DERIVED (audit): the Planck×Kirchhoff spectrum above IS physics; this
  // Reinhard tone-map + 1.7 gain (and EMISSION_SCALE) is a camera/exposure
  // choice — the error term between the real glow and the pixel, not the glow.
  return e/(1.0+max(e.r,max(e.g,e.b)))*1.7;   // tone-map: compress brightness, preserve the Planck hue
}`;
// Euclidean modulo (always same sign as n, for n>0) — matches Python's `%`
// on floats and GLSL's `mod()` builtin, which JS's native `%` does NOT.
function emod(a,n){ return a-n*Math.floor(a/n); }
// Quilez smooth-min union, JS mirror of csg.py's sdf_smooth_union / GLSL's sdSmoothUnion.
function sdfSmoothUnionJS(a,b,k){
  if(k<=0.0) return Math.min(a,b);
  const h=Math.max(k-Math.abs(a-b),0.0)/k;
  return Math.min(a,b)-h*h*k*0.25;
}
// Derives InvoluteGear's radii/angles from the serialized raw params (module,
// teeth, pressure_angle, addendum_coeff, dedendum_coeff, fillet_coeff,
// face_width) EXACTLY as kernel/gear.py's __init__ does. Single source used by
// both jsGearSDF (JS self-check) and glslLeafCall (GLSL constant baking) so the
// two mirrors can never diverge on the derivation step.
function _gearParams(s){
  const m=s.module, N=s.teeth, alpha=s.pressure_angle;
  const addC=s.addendum_coeff!==undefined?s.addendum_coeff:1.0;
  const dedC=s.dedendum_coeff!==undefined?s.dedendum_coeff:1.25;
  const filC=s.fillet_coeff!==undefined?s.fillet_coeff:0.35;
  const rP=m*N/2.0, rB=rP*Math.cos(alpha);
  const rA=rP+addC*m, rF=rP-dedC*m;
  const psi=Math.PI/(2.0*N), beta=Math.PI/N;
  const inv=(x)=>Math.tan(x)-x;
  const thetaRb=psi+inv(alpha);
  const filletR=filC*m;
  const halfW=s.face_width*0.5;
  return {rB,rA,rF,beta,thetaRb,filletR,halfW};
}
// JS mirror of InvoluteGear._gear_sdf_2d + _local_surface_distance (gear.py),
// including both CRITICAL clamps on the nearest-involute-t (see that file's
// docstring) — statement-for-statement with sdInvoluteGear in PRIMS_GLSL.
function jsGearSDF(s,q){
  const c=s.center, P=_gearParams(s);
  const dx=q[0]-c[0], dy=q[1]-c[1], dz=q[2]-c[2];
  const rho=Math.hypot(dx,dy);
  const phi=Math.atan2(dy,dx);
  const twoBeta=2.0*P.beta;
  const phiFoldedSigned=emod(phi+P.beta, twoBeta)-P.beta;
  const phiFolded=Math.abs(phiFoldedSigned);
  // gear frame -> involute frame is a REFLECTION: phiInv = thetaRb - phiFolded,
  // query point rebuilt with the same reflected angle (see kernel/gear.py)
  const phiInv=P.thetaRb-phiFolded;
  let tStar;
  if(rho<=P.rB){
    tStar=0.0;
  } else {
    const arc=Math.acos(_clamp1(P.rB/rho));
    tStar=Math.max(0.0, phiInv+arc);              // CRITICAL clamp: floor at 0
  }
  const cosT=Math.cos(tStar), sinT=Math.sin(tStar);
  const cxT=P.rB*(cosT+tStar*sinT), cyT=P.rB*(sinT-tStar*cosT);
  const qx=rho*Math.cos(phiInv), qy=rho*Math.sin(phiInv);
  const dFlankMag=Math.hypot(qx-cxT, qy-cyT);
  const ratio = rho>1e-15 ? (P.rB/rho) : 1.0;
  const alphaRho=Math.acos(_clamp1(ratio));
  const thetaRho=P.thetaRb-(Math.tan(alphaRho)-alphaRho);
  const dFlank = phiFolded<thetaRho ? -dFlankMag : dFlankMag;
  const dAddendum=rho-P.rA, dRootDisk=rho-P.rF;
  const dToothWedge=Math.max(dFlank,dAddendum);
  const dGear2d=sdfSmoothUnionJS(dToothWedge,dRootDisk,P.filletR);
  const dExtrude=Math.abs(dz)-P.halfW;
  if(dGear2d<=0.0 && dExtrude<=0.0) return Math.max(dGear2d,dExtrude);
  if(dGear2d>0.0 && dExtrude>0.0) return Math.hypot(dGear2d,dExtrude);
  return Math.max(dGear2d,dExtrude);
}
function _clamp1(x){ return x<-1.0?-1.0:(x>1.0?1.0:x); }
function jsLeafSDF(leaf, q){
  const s=leaf.shape, c=s.center;
  if(s.type==="Gear") return jsGearSDF(s, q);
  if(s.type==="Sphere"||s.type==="HollowSphere") return len(sub(q,c))-s.radius;
  if(s.type==="Cylinder"){ const d=sub(q,c), dr=Math.hypot(d[0],d[1])-s.radius, da=Math.abs(d[2])-s.height*0.5;
    return Math.min(Math.max(dr,da),0)+Math.hypot(Math.max(dr,0),Math.max(da,0)); }
  if(s.type==="Cone"){ const d=sub(q,c), lz=d[2]+s.height*0.25, rho=Math.hypot(d[0],d[1]);
    if(lz<0) { const dr=Math.max(rho-s.radius,0); return Math.hypot(dr,lz); }
    if(lz>s.height) return Math.hypot(rho,lz-s.height);
    const rAtZ=s.radius*(1-lz/s.height), slant=Math.hypot(s.radius,s.height), dLat=(rho-rAtZ)*s.height/slant;
    return rho<=rAtZ ? Math.max(dLat,-lz) : dLat; }
  if(s.type==="Ellipsoid"){ const d=sub(q,c), r=[s.rx,s.ry,s.rz];
    const k0=Math.hypot(d[0]/r[0],d[1]/r[1],d[2]/r[2]), k1=Math.hypot(d[0]/(r[0]*r[0]),d[1]/(r[1]*r[1]),d[2]/(r[2]*r[2]));
    return k0*(k0-1)/Math.max(k1,1e-9); }
  if(s.type==="Box"){ const d=[Math.abs(q[0]-c[0])-s.x*0.5,Math.abs(q[1]-c[1])-s.y*0.5,Math.abs(q[2]-c[2])-s.z*0.5];
    return Math.min(Math.max(d[0],Math.max(d[1],d[2])),0)+Math.hypot(Math.max(d[0],0),Math.max(d[1],0),Math.max(d[2],0)); }
  if(s.type==="Water"){ const lvl=s.level!==undefined?s.level:c[1];   // flat approx (waves are GPU-side)
    const top=q[1]-lvl, bottom=(lvl-s.depth)-q[1], dx=Math.abs(q[0]-c[0])-s.x, dz=Math.abs(q[2]-c[2])-s.z;
    return Math.max(Math.max(top,bottom),Math.max(dx,dz)); }
  return 1e9;
}
function glslLeafCall(leaf, qvar){
  const s=leaf.shape, c=s.center, ctr=`vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])})`;
  if(s.type==="Gear"){ const P=_gearParams(s);
    return `sdInvoluteGear(${qvar},${ctr},${glf(P.rB)},${glf(P.rA)},${glf(P.rF)},`
         + `${glf(P.beta)},${glf(P.thetaRb)},${glf(P.filletR)},${glf(P.halfW)})`; }
  if(s.type==="Sphere"||s.type==="HollowSphere") return `sdSphere(${qvar},${ctr},${glf(s.radius)})`;
  if(s.type==="Cylinder") return `sdCyl(${qvar},${ctr},${glf(s.radius)},${glf(s.height)})`;
  if(s.type==="Cone") return `sdConeZ(${qvar},${ctr},${glf(s.radius)},${glf(s.height)})`;
  if(s.type==="Ellipsoid") return `sdEllip(${qvar},${ctr},vec3(${glf(s.rx)},${glf(s.ry)},${glf(s.rz)}))`;
  if(s.type==="Box") return `sdBox(${qvar},${ctr},vec3(${glf(s.x)},${glf(s.y)},${glf(s.z)}))`;
  if(s.type==="Water") return `sdWater(${qvar},${ctr},${glf(s.x)},${glf(s.z)},${glf(s.depth)},${glf(s.level!==undefined?s.level:s.center[1])})`;
  return "1e9";
}
// ── wrapper shapes + organic outlines (Deckard's researched geometry) ────
// JS + GLSL stay statement-for-statement with kernel/outline.py + deckard's
// _Rotated/_Clipped/_Subtracted, so the in-page sdf_samples self-check holds.
function segDist2(px,py,ax,ay,bx,by){ const dx=bx-ax,dy=by-ay,L2=dx*dx+dy*dy;
  if(L2<=1e-18) return Math.hypot(px-ax,py-ay);
  let t=((px-ax)*dx+(py-ay)*dy)/L2; t=t<0?0:(t>1?1:t);
  return Math.hypot(px-(ax+t*dx),py-(ay+t*dy)); }
function polySD(px,py,poly){ let inside=false,minD=Infinity; const n=poly.length;
  for(let i=0;i<n;i++){ const a=poly[i],b=poly[(i+1)%n];
    const d=segDist2(px,py,a[0],a[1],b[0],b[1]); if(d<minD)minD=d;
    if((a[1]>py)!==(b[1]>py)){ const xint=a[0]+(py-a[1])/(b[1]-a[1])*(b[0]-a[0]);
      if(px<xint) inside=!inside; } }
  return inside?-minD:minD; }
function sectionSD(z,rho,sec){ let inside=false,minD=Infinity; const n=sec.length;
  for(let i=0;i<n;i++){ const a=sec[i],b=sec[(i+1)%n];
    if((a[1]>rho)!==(b[1]>rho)){ const zint=a[0]+(rho-a[1])/(b[1]-a[1])*(b[0]-a[0]);
      if(z<zint) inside=!inside; }
    if(Math.abs(a[1])<1e-12&&Math.abs(b[1])<1e-12) continue;   // axis edge: not a surface
    const d=segDist2(z,rho,a[0],a[1],b[0],b[1]); if(d<minD)minD=d; }
  return inside?-minD:minD; }
function jsOutlineSDF(s,q){ const c=s.center;
  if(s.mode==="revolve"){
    if(!s._section){ const zs=s.profile.map(p=>p[0]);
      s._section=s.profile.concat([[Math.max(...zs),0],[Math.min(...zs),0]]); }
    const rho=Math.hypot(q[0]-c[0],q[1]-c[1]);
    return sectionSD(q[2]-c[2],rho,s._section); }
  const d2=polySD(q[0]-c[0],q[1]-c[1],s.profile);
  const dz=Math.abs(q[2]-c[2])-0.5*(s.thickness||0.001);
  if(d2<=0&&dz<=0) return Math.max(d2,dz);
  if(d2>0&&dz>0) return Math.hypot(d2,dz);
  return Math.max(d2,dz); }
function jsVoxelSDF(s,q){          // trilinear sample of the decoded int8 grid (self-check mirror)
  const g=s._grid; if(!g) return 1e9;
  const c=s.center, n=s.dims, vs=s.voxel_size, sc=s.band/127.0;
  const nx=n[0],ny=n[1],nz=n[2];
  let fi=(q[0]-c[0])/vs+(nx-1)*0.5, fj=(q[1]-c[1])/vs+(ny-1)*0.5, fk=(q[2]-c[2])/vs+(nz-1)*0.5;
  const ci=Math.min(Math.max(fi,0),nx-1), cj=Math.min(Math.max(fj,0),ny-1), ck=Math.min(Math.max(fk,0),nz-1);
  const i0=Math.floor(ci),j0=Math.floor(cj),k0=Math.floor(ck);
  const i1=Math.min(i0+1,nx-1),j1=Math.min(j0+1,ny-1),k1=Math.min(k0+1,nz-1);
  const di=ci-i0,dj=cj-j0,dk=ck-k0;
  const at=(i,j,k)=>g[(k*ny+j)*nx+i]*sc;          // x-fastest pack (matches _shape_to_dict)
  const c00=at(i0,j0,k0)*(1-di)+at(i1,j0,k0)*di, c01=at(i0,j0,k1)*(1-di)+at(i1,j0,k1)*di;
  const c10=at(i0,j1,k0)*(1-di)+at(i1,j1,k0)*di, c11=at(i0,j1,k1)*(1-di)+at(i1,j1,k1)*di;
  const c0=c00*(1-dj)+c10*dj, c1=c01*(1-dj)+c11*dj;
  const val=c0*(1-dk)+c1*dk;
  if(ci===fi&&cj===fj&&ck===fk) return val;        // inside the box → the interpolated distance
  const bx=c[0]+(ci-(nx-1)*0.5)*vs, by=c[1]+(cj-(ny-1)*0.5)*vs, bz=c[2]+(ck-(nz-1)*0.5)*vs;
  return Math.max(val,0)+Math.hypot(q[0]-bx,q[1]-by,q[2]-bz);
}
function jsShapeSDF(s,q){
  if(!s) return 1e9;
  if(s.type==="Torus"){ const d=sub(q,s.center);
    return Math.hypot(Math.hypot(d[0],d[1])-s.major_radius,d[2])-s.minor_radius; }
  if(s.type==="Outline") return jsOutlineSDF(s,q);
  if(s.type==="Voxel") return jsVoxelSDF(s,q);
  if(s.type==="Rotated"){ const c=s.center;
    return jsShapeSDF(s.shape, add(qrotInvJS(s.rot, sub(q,c)), c)); }
  if(s.type==="Clipped") return Math.max(jsShapeSDF(s.shape,q), s.sign*(q[s.axis]-s.level));
  if(s.type==="Subtracted") return Math.max(jsShapeSDF(s.shape,q), -jsShapeSDF(s.cut,q));
  return jsLeafSDF({shape:s}, q);
}
// per-build registry of generated GLSL outline functions (profiles baked as consts)
let EXTRA_GLSL="", outlineN=0, outlineKeys=new Map(), voxelKeys=new Set();
// PERF: bvhCull gates each leaf's mapD() term behind its (center,radius)
// bounding sphere: outside it, the leaf substitutes bd_i=|q-c|-r (a safe
// lower bound on its true distance — never overshoots) instead of its exact
// SDF. A bare "bd_i>0.0" gate is NOT enough: "never overshoots" only means
// mapD's output never claims a leaf is FARTHER than it truly is — it does
// NOT mean a small bd_i implies we're near that leaf's real surface. bd_i is
// smoothly near-zero anywhere close to the PADDED SPHERE's own boundary,
// which for a large/thin/elongated leaf (a room wall: true half-thickness
// ~0.02m, bounding_radius ~2m+) sits deep in otherwise-empty space; if that
// near-zero bd_i wins mapD()'s min/max chain where no other leaf is close,
// march()/sunShadow()/visRay() terminate on the invisible bounding sphere
// instead of real geometry. Measured on the shipped 48-leaf doughnut/plate/
// table/room scene (replaying mapD()+march() in Python for a full camera-ray
// sweep) with a bare ">0.0" gate: 99.9% of rays diverged from exact-eval by
// >10cm, worst case ~3.3m — a real, visible defect, not sub-pixel noise.
// Fix (below, GATE_BUFFER in the decls-loop comment): only substitute bd_i
// once it exceeds a buffer comfortably above every hit epsilon mapD's
// consumers use, so a substituted term can never itself register as a hit —
// any composed d that DOES drop below hit-epsilon must be won by a genuine
// exact evaluation. Re-verified after that fix with the same camera-ray
// sweep: 0 hit/miss or material divergences, >99.9% of rays within 1mm of
// the exact-eval hit point (the tiny remainder traced to sphere-tracing's
// own pre-existing grazing-angle sensitivity on thin geometry, independent
// of this gate — reproduces with bvhCull off vs a jittered ray too).
let bvhCull=true;    // debug/A-B toggle: per-leaf bounding-sphere early-rejection in mapD()
                     // (buildPathShader only). false = always exact-evaluate every leaf,
                     // i.e. byte-identical to pre-optimization codegen — for pixel-diff
                     // verification, or if a future scene's geometry needs re-checking
                     // with tools/ 's camera-ray sweep before trusting bvhCull=true on it.
// Render-only, client-side opacity OVERRIDE — {material_label: alpha in [0,1]}.
// Consulted only at shader-BUILD time (buildPathShader),
// mutated ONLY through window.Radiance.setMaterialOpacity (never poked directly by
// UI code), which rebuilds+relinks the program. A label absent here renders at its
// scene-baked `materials[label].opacity` (default 1.0 — every scene not touching
// this feature is unaffected). Never round-tripped into the scene JSON; a fresh
// load()/reset starts clean. Purely a VIEW toggle — the physics (mass/density/
// inertia) this scene renders is computed entirely upstream in Python and never
// reads this map.
let materialOpacity={};
// ── R4 SVO state (svo.js — octree-of-bricks; svoEnabled default false until
// R4.4 flips the voxel scenes). The compile gate is svoActive = svoEnabled
// AND svo.js loaded AND the scene has ELIGIBLE Voxel leaves (top-level Voxel
// shape, body-bound, op:add — subtract/intersect Voxel leaves refuse the SVO
// path and stay per-leaf codegen, the R3 band-gate honesty rule). ?svo=0/1
// presets the toggle at boot (the A/B capture surface); setSVO() flips live.
const svoCtl = (typeof window!=="undefined" && window.SGSVO) ? window.SGSVO : null;
let svoEnabled = new URLSearchParams(location.search).get("svo")==="1";
let svoGL = null;          // GL residency {nodeTex,atlasTex,lutTex,bodyTex,packed,rowBuf}
let svoActiveNow = false;  // set per buildProgram; read by draw()/loop()'s capture arm
let svoReady = false;      // uploads done — capture pins wait on this (blueprint Sec 5)
function svoEligible(scene){
  const ls=scene.csg_leaves||[], bodies=scene.bodies||[], out=[];
  ls.forEach((l,i)=>{ const s=l.shape;
    if(s && s.type==="Voxel" && l.body!=null && (i===0 || !l.op || l.op==="add"))
      out.push({leaf:i, body:l.body, shape:s, material:l.material,
                pivot:(bodies[l.body]&&bodies[l.body].pivot)||[0,0,0],
                bounds:l.bounds||deriveBounds(s)||null});
  });
  return out;
}
function resetExtraGLSL(){ EXTRA_GLSL=""; outlineN=0; outlineKeys=new Map(); voxelKeys=new Set(); }
// A real voxel solid: sample its signed-distance grid (uploaded as a 3-D texture)
// trilinearly in GLSL — the kernel Voxel's _trilinear, in hardware. uvw =
// (p-center)/extent+0.5 maps p to the exact fractional grid index (validated by
// the Python round-trip). Outside the box → distance-to-box (a safe sphere-trace
// underestimate). The leaf's `_vid` (assigned in prepareVoxels) names its sampler.
function glslVoxel(s,qvar){
  const vid=s._vid;
  if(!voxelKeys.has(vid)){
    voxelKeys.add(vid);
    const c=s.center, ex=[s.dims[0]*s.voxel_size, s.dims[1]*s.voxel_size, s.dims[2]*s.voxel_size];
    const lo=[c[0]-ex[0]*0.5, c[1]-ex[1]*0.5, c[2]-ex[2]*0.5];
    EXTRA_GLSL += `
uniform highp sampler3D uVox${vid};
float sdVox${vid}(vec3 p){
  vec3 lo=vec3(${glf(lo[0])},${glf(lo[1])},${glf(lo[2])});
  vec3 ex=vec3(${glf(ex[0])},${glf(ex[1])},${glf(ex[2])});
  vec3 uvw=(p-lo)/ex;
  vec3 cl=clamp(uvw,vec3(0.0),vec3(1.0));
  float val=(texture(uVox${vid},cl).r*255.0-128.0)*${glf(s.band/127.0)};
  if(all(equal(cl,uvw))) return val;                 // inside the box → interpolated distance
  vec3 d=max(lo-p, p-(lo+ex));                        // outside → boundary sample (≥0 void margin)
  return max(val,0.0)+length(max(d,vec3(0.0)));       // + distance to the box (kernel's formula)
}`;
  }
  return `sdVox${vid}(${qvar})`;
}
// Per-cell field samplers → EXTRA_GLSL (one fn per fid) + the per-leaf switch
// body for fieldTempK. Called by BOTH shader builders after their leaf loop.
// The grid is LEAF-LOCAL; a body-bound leaf inverse-transforms the world point
// first (same expression as mapD's bdy decls), so the hot rind rides the body.
function glslFieldFns(scene, worldOffset){
  // worldOffset: "" (buildFragmentShader — p IS world-space) or "+uEyeWorld"
  // (buildPathShader — p is camera-relative after the R1-lite rebase; a
  // non-body leaf's field grid is baked from ABSOLUTE world coordinates,
  // so it must reconstruct world-space before sampling). Body-bound leaves
  // never need this — their own p-uBodyPos[k] subtraction is already
  // eye-consistent (see draw()'s bindScene rebase comment).
  worldOffset = worldOffset || "";
  let sel="";
  for(const f of (scene._fields||[])){
    const g=f.spec.grid, c=g.center;
    const ex=[g.dims[0]*g.voxel_size, g.dims[1]*g.voxel_size, g.dims[2]*g.voxel_size];
    const lo=[c[0]-ex[0]*0.5, c[1]-ex[1]*0.5, c[2]-ex[2]*0.5];
    EXTRA_GLSL += `
uniform highp sampler3D uFldA${f.fid}, uFldB${f.fid}; uniform float uFldU${f.fid};
float fldT${f.fid}(vec3 p){                          // leaf-LOCAL point -> kelvin
  vec3 lo=vec3(${glf(lo[0])},${glf(lo[1])},${glf(lo[2])});
  vec3 ex=vec3(${glf(ex[0])},${glf(ex[1])},${glf(ex[2])});
  vec3 cl=clamp((p-lo)/ex, vec3(0.0), vec3(1.0));    // CLAMP_TO_EDGE == the sim's ambient rind
  // NON-DERIVED (audit): the keyframe mix is playback reconstruction of frozen
  // sim states (u from the scrub time), never the heat equation
  return ${glf(f.spec.t_min)} + mix(texture(uFldA${f.fid},cl).r,
                                    texture(uFldB${f.fid},cl).r,
                                    uFldU${f.fid})*${glf(f.spec.t_max-f.spec.t_min)};
}`;
    const leaf=scene.csg_leaves[f.leaf];
    if(leaf.body!=null){
      const piv=((scene.bodies||[])[leaf.body]||{}).pivot||[0,0,0];
      sel += `  if(m==${f.leaf}) return fldT${f.fid}(qrotInv(uBodyQuat[${leaf.body}], `
          +  `p-uBodyPos[${leaf.body}])+vec3(${glf(piv[0])},${glf(piv[1])},${glf(piv[2])}));\n`;
    } else {
      sel += `  if(m==${f.leaf}) return fldT${f.fid}(p${worldOffset});\n`;
    }
  }
  return sel;
}
function glslOutline(s,qvar){
  const key=JSON.stringify([s.profile,s.mode,s.thickness||0,s.center]);
  if(!outlineKeys.has(key)){
    const k=outlineN++; outlineKeys.set(key,k); const c=s.center;
    if(s.mode==="revolve"){
      const zs=s.profile.map(p=>p[0]);
      const sec=s.profile.concat([[Math.max(...zs),0],[Math.min(...zs),0]]);
      const N=sec.length, arr=sec.map(p=>`vec2(${glf(p[0])},${glf(p[1])})`).join(",");
      EXTRA_GLSL+=`
const vec2 OSEC${k}[${N}] = vec2[${N}](${arr});
float sdOutline${k}(vec3 p){
  float z=p.z-(${glf(c[2])}); float rho=length(vec2(p.x-(${glf(c[0])}),p.y-(${glf(c[1])})));
  bool inside=false; float minD=1e9;
  for(int i=0;i<${N};i++){ vec2 a=OSEC${k}[i]; vec2 b=OSEC${k}[i+1==${N}?0:i+1];
    if((a.y>rho)!=(b.y>rho)){ float zint=a.x+(rho-a.y)/(b.y-a.y)*(b.x-a.x); if(z<zint) inside=!inside; }
    if(abs(a.y)<1e-12&&abs(b.y)<1e-12) continue;
    vec2 e=b-a; float L2=dot(e,e); float t=L2<=1e-18?0.0:clamp(dot(vec2(z,rho)-a,e)/L2,0.0,1.0);
    minD=min(minD,length(vec2(z,rho)-(a+e*t))); }
  return inside?-minD:minD; }
`;
    } else {
      const N=s.profile.length, th=s.thickness||0.001;
      const arr=s.profile.map(p=>`vec2(${glf(p[0])},${glf(p[1])})`).join(",");
      EXTRA_GLSL+=`
const vec2 OPR${k}[${N}] = vec2[${N}](${arr});
float sdOutline${k}(vec3 p){
  vec2 q2=vec2(p.x-(${glf(c[0])}), p.y-(${glf(c[1])}));
  bool inside=false; float minD=1e9;
  for(int i=0;i<${N};i++){ vec2 a=OPR${k}[i]; vec2 b=OPR${k}[i+1==${N}?0:i+1];
    vec2 e=b-a; float L2=dot(e,e); float t=L2<=1e-18?0.0:clamp(dot(q2-a,e)/L2,0.0,1.0);
    minD=min(minD,length(q2-(a+e*t)));
    if((a.y>q2.y)!=(b.y>q2.y)){ float xint=a.x+(q2.y-a.y)/(b.y-a.y)*(b.x-a.x); if(q2.x<xint) inside=!inside; } }
  float d2=inside?-minD:minD;
  float dz=abs(p.z-(${glf(c[2])}))-${glf(0.5*th)};
  if(d2<=0.0&&dz<=0.0) return max(d2,dz);
  if(d2>0.0&&dz>0.0) return length(vec2(d2,dz));
  return max(d2,dz); }
`;
    }
  }
  return `sdOutline${outlineKeys.get(key)}(${qvar})`;
}
// ── bounding sphere, derived client-side when the scene didn't ship one ──
// Only leaves compiled through construct_to_scene() carry a Python-baked
// `bounds`. Trajectory recorders hand-build their csg_leaves, so the clock,
// gear, bell and motor scenes — the heaviest ones — shipped without any, and
// silently lost the bvhCull early-rejection entirely. A bounding sphere is a
// pure function of the shape dict, so the viewer can derive its own and every
// already-saved scene gets the speedup with no re-render.
//
// Mirrors scene_export.py exactly: the same radius formulas as
// kernel/shapes.py's bounding_radius(), the same 1 cm margin, and the same
// "skip leaves smaller than 4x the margin" rule (below that the fixed margin
// dominates the true size and the padded sphere can register a phantom hit).
// Anything not confidently computable returns null -> no bounds -> the leaf is
// always evaluated exactly, which is the pre-existing behaviour.
const BOUNDS_MARGIN_M = 0.01, BOUNDS_MIN_RADIUS_M = 0.04;
function shapeBoundRadius(s){
  if(!s) return null;
  switch(s.type){
    case "Sphere": case "HollowSphere": return s.radius;
    case "Cylinder": return Math.sqrt(s.radius*s.radius + (s.height/2)*(s.height/2));
    case "Box": return 0.5*Math.sqrt(s.x*s.x + s.y*s.y + s.z*s.z);
    case "Ellipsoid": return Math.max(s.rx,s.ry,s.rz);
    case "Torus": return s.major_radius + s.minor_radius;
    case "Cone": {                       // measured about the CoM (h/4 above base)
      const apex = 3*s.height/4;
      const baseEdge = Math.sqrt(s.radius*s.radius + (s.height/4)*(s.height/4));
      return Math.max(apex, baseEdge); }
    case "Voxel": {                      // half-diagonal of the grid's own box
      const d=s.dims, v=s.voxel_size;
      if(!d || !(v>0)) return null;
      return 0.5*Math.sqrt(d.reduce((a,n)=>a+(n*v)*(n*v),0)); }
    case "Gear": {                       // hypot(tip radius, half face width)
      const m=s.module, N=s.teeth;       // r_a = m*N/2 + addendum_coeff*m
      if(!(m>0) || !(N>0)) return null;
      const ac = s.addendum_coeff!==undefined ? s.addendum_coeff : 1.0;
      return Math.hypot(m*N/2 + ac*m, (s.face_width||0)/2); }
    // wrappers delegate, exactly as the Python CSG wrappers forward to _base
    case "Rotated": case "Clipped": case "Subtracted": return shapeBoundRadius(s.shape);
    default: return null;                // Outline/Gear/Water — not gated
  }
}
// the innermost primitive's OWN centre: _Clipped/_Subtracted override .center to
// a measured centroid, which is NOT the point the radius was measured about.
function shapeBoundCenter(s){
  while(s && (s.type==="Rotated"||s.type==="Clipped"||s.type==="Subtracted")) s=s.shape;
  return (s && s.center) || null;
}
function deriveBounds(s){
  const r = shapeBoundRadius(s); if(r===null || !(r>0)) return null;
  if(r < BOUNDS_MIN_RADIUS_M) return null;          // too small to gate safely
  const c = shapeBoundCenter(s); if(!c) return null;
  return {center:c, radius:r + BOUNDS_MARGIN_M};
}

function glslShapeCall(s,qvar){
  if(!s) return "1e9";
  if(s.type==="Torus"){ const c=s.center;
    return `sdTorus(${qvar},vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}),${glf(s.major_radius)},${glf(s.minor_radius)})`; }
  if(s.type==="Outline") return glslOutline(s,qvar);
  if(s.type==="Voxel") return glslVoxel(s,qvar);
  if(s.type==="Rotated"){ const c=s.center,r=s.rot;
    return glslShapeCall(s.shape,
      `leafRot(vec4(${glf(r[0])},${glf(r[1])},${glf(r[2])},${glf(r[3])}),${qvar},vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}))`); }
  if(s.type==="Clipped")
    return `max(${glslShapeCall(s.shape,qvar)},${glf(s.sign)}*((${qvar}).${"xyz"[s.axis]}-${glf(s.level)}))`;
  if(s.type==="Subtracted")
    return `max(${glslShapeCall(s.shape,qvar)},-${glslShapeCall(s.cut,qvar)})`;
  return glslLeafCall({shape:s},qvar);
}

function jsEvalSDF(scene, p, poses){
  const ls=scene.csg_leaves, bodies=scene.bodies||[]; let d=0;
  for(let i=0;i<ls.length;i++){
    let q=p; const bi=ls[i].body;
    if(bi!=null && poses && poses[bi]){             // world → body-local (rest)
      const piv=(bodies[bi]&&bodies[bi].pivot)||[0,0,0];
      q=add(qrotInvJS(poses[bi].quat, sub(p,poses[bi].pos)), piv);
    }
    const si = jsShapeSDF(ls[i].shape, q);
    if(i===0){ d=si; continue; }
    const op=ls[i].op;
    if(op==="subtract") d=Math.max(d,-si);
    else if(op==="intersect") d=Math.max(d,si);
    else d=Math.min(d,si);
  }
  return d;
}
// Which leaf's material owns world point p — the JS twin of Python's
// scene_spec_to_sdf().material_at(): walk leaves in REVERSE (later leaves win,
// same convention mapD's `matsel` and Python's `material_at` both use) and
// return the first one whose own local shape actually contains p. Used only
// by click-to-pick (a VIEW affordance); never consulted by geometry/physics.
function jsMaterialAt(scene, p, poses){
  const ls=scene.csg_leaves||[], bodies=scene.bodies||[];
  for(let i=ls.length-1;i>=0;i--){
    let q=p; const bi=ls[i].body;
    if(bi!=null && poses && poses[bi]){
      const piv=(bodies[bi]&&bodies[bi].pivot)||[0,0,0];
      q=add(qrotInvJS(poses[bi].quat, sub(p,poses[bi].pos)), piv);
    }
    if(jsShapeSDF(ls[i].shape, q) < 0.0) return ls[i].material;
  }
  return null;
}
// Cheap CPU sphere-trace of a single camera ray — click-to-pick only (the GPU
// path tracer never runs this; it's not on the render hot path at all).
function jsMarchPick(scene, ro, rd, poses, maxDist){
  let t=0;
  for(let i=0;i<300;i++){
    const p=add(ro, mul(rd,t));
    const d=jsEvalSDF(scene, p, poses);
    if(Math.abs(d) < 0.002) return p;
    t += Math.max(Math.abs(d), 0.0008);
    if(t > maxDist) return null;
  }
  return null;
}
// Finite-difference normal of the COMPOSED sdf at p — the JS twin of calcN()
// in both shaders, used only to nudge a marched hit point safely INSIDE the
// surface before asking jsMaterialAt "whose material is this": jsMarchPick
// stops within an epsilon SHELL of the surface (it can land on either side),
// exactly the ambiguity mapD(p-n*0.0008,mat) resolves on the GPU.
function jsCalcN(scene, p, poses){
  const h=1e-4;
  const d=(o)=>jsEvalSDF(scene, add(p,o), poses);
  return norm([d([h,0,0])-d([-h,0,0]), d([0,h,0])-d([0,-h,0]), d([0,0,h])-d([0,0,-h])]);
}

// ── build the fragment shader (colors BAKED as constants) ──────────────
// NOTE: this is the no-float-target compatibility shim only (single-hit
// raymarch, no bounce/transmission of any kind — see the module-level comment
// at the top of the file). It does NOT read materials[label].opacity or
// materialOpacity: a GPU too old to path-trace shows every material at its
// normal opacity. The render-only transparency toggle lives entirely in
// buildPathShader()/trace() below, the one real renderer.
function buildFragmentShader(scene){
  resetExtraGLSL();
  const ls=scene.csg_leaves, bodies=scene.bodies||[];
  // R4: the no-float shim keeps DENSE voxels for every scene it can compile
  // (legacy glslVoxel survives here permanently) — but past the measured
  // ~48-leaf compile wall (yard-class scenes) it must route SVO-eligible
  // leaves OUT or it re-hits the 101-leaf hang the SVO path exists to kill.
  // Disclosed limitation: on a no-float GPU such a scene shows no voxel
  // matter (the PT path is required); captures already refuse there.
  const shimSet = new Set((scene._svoShimRouted && scene._svoEligible)
                          ? scene._svoEligible.map(e=>e.leaf) : []);
  if(shimSet.size)
    console.warn(`R4: no-float shim omits ${shimSet.size} voxel bodies (leaf-count wall) — PT path required for this scene's voxel matter`);
  const legacyIdx = ls.map((_,i)=>i).filter(i=>!shimSet.has(i));
  // Per-body rigid transform (world→rest): inverse-rotate about the pose, then
  // re-add the body's pivot. A leaf with no `body` is static world geometry.
  const used=[...new Set(legacyIdx.map(i=>ls[i].body).filter(b=>b!=null))].sort((a,b)=>a-b);
  let bodyDecls="";
  used.forEach(k=>{ const piv=(bodies[k]&&bodies[k].pivot)||[0,0,0];
    bodyDecls += `  vec3 bdy${k}=qrotInv(uBodyQuat[${k}], p-uBodyPos[${k}])`
              +  `+vec3(${glf(piv[0])},${glf(piv[1])},${glf(piv[2])});\n`; });
  let decls="", compose=legacyIdx.length?`  float d=s${legacyIdx[0]};\n`:"  float d=1e9;\n", matsel="  mat=-1;\n", colorfn="", metalfn="", reflfn="", emisfn="", tempfn="", roughfn="", bodyTempFn="";
  // the theater's ambient datum (physics_env, from the STP/IRT/ISM presets) is
  // the temperature a leaf falls back to — deep_space stops pretending 293 K
  const envT=(scene.physics_env&&scene.physics_env.temperature_k!==undefined)?scene.physics_env.temperature_k:293.15;
  for(let i=0;i<ls.length;i++){
    if(shimSet.has(i)) continue;                   // routed past the leaf wall
    const q = ls[i].body!=null ? `bdy${ls[i].body}` : "(p)";
    decls += `  float s${i}=${glslShapeCall(ls[i].shape, q)};\n`;
    const mat=scene.materials[ls[i].material]||{};
    const c=mat.color_rgb||[0.72,0.72,0.72];
    colorfn += `  if(m==${i}) return vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])});\n`;
    const isMetal = (mat.metal!==undefined ? mat.metal : mat.emergent);  // band-gap nonmetals shade matte
    metalfn += `  if(m==${i}) return ${glf(isMetal?1.0:0.0)};\n`;
    // reflection R0: only CLEAR DIELECTRICS (water/glass) cast an environment
    // reflection ray (baked Fresnel R0). Metals keep their Drude colour + specular
    // — reflecting a generic sky would wash out their emergent hue. -1 = no ray.
    const r0 = (mat.reflect_r0!==undefined) ? mat.reflect_r0 : -1.0;
    reflfn += `  if(m==${i}) return ${glf(r0)};\n`;
    // Kirchhoff emissivity for incandescence: measured ε(λ)=1−R if baked, else 1−colour
    const em = mat.emissivity_rgb;
    emisfn += em ? `  if(m==${i}) return vec3(${glf(em[0])},${glf(em[1])},${glf(em[2])});\n`
                 : `  if(m==${i}) return clamp(vec3(1.0)-vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}),0.0,1.0);\n`;
    // Surface roughness, now a real material property rather than a guess from
    // metalness. Beckmann alpha (field/interface/texture.py) -> Blinn-Phong
    // exponent by the standard n = 2/alpha^2 - 2, clamped to [8,256]. Polished
    // metals land near alpha 0.11 (n~160, tight highlight); amorphous solids at
    // texture.py's documented 0.4 class constant (n~10, broad sheen). Absent =>
    // fall back to the old metal-keyed guess so pre-enrichment scenes are
    // unchanged.
    const rg = mat.roughness;
    const specExp = (rg!==undefined && rg>0)
      ? Math.min(256, Math.max(8, 2/(rg*rg) - 2))
      : (isMetal ? 64.0 : 24.0);
    roughfn += `  if(m==${i}) return ${glf(specExp)};
`;
    // per-object initial temperature from the sim layer (default = the theater's ambient)
    tempfn += `  if(m==${i}) return ${glf(ls[i].temperature_k!==undefined?ls[i].temperature_k:envT)};\n`;
    // a body-bound leaf can carry a PER-FRAME temperature (uBodyTemp; -1 = none)
    bodyTempFn += `  if(m==${i}) return ${ls[i].body!=null?`uBodyTemp[${ls[i].body}]`:"-1.0"};\n`;
  }
  for(let n=1;n<legacyIdx.length;n++){
    const i=legacyIdx[n]; const op=ls[i].op;
    compose += (op==="subtract") ? `  d=max(d,-s${i});\n`
             : (op==="intersect") ? `  d=max(d,s${i});\n`
             : `  d=min(d,s${i});\n`;
  }
  for(let n=legacyIdx.length-1;n>=0;n--){ const i=legacyIdx[n];
    matsel += `  ${n===legacyIdx.length-1?"if":"else if"}(s${i}<0.0) mat=${i};\n`; }
  const fieldSel=glslFieldFns(scene);               // per-cell field samplers → EXTRA_GLSL + switch
  // contact shadows + AO scale with the scene, so a 0.2 m cup and a 3 m chair both read right
  const bb=scene.bbox||[[-1,1],[-1,1],[-1,1]];
  const diag=len(sub([bb[0][1],bb[1][1],bb[2][1]],[bb[0][0],bb[1][0],bb[2][0]]))||1;
  const SHEPS=glf(Math.max(0.0015,diag*0.0025)), SHMIN=glf(Math.max(0.004,diag*0.004)),
        SHMAX=glf(Math.max(0.05,diag*0.06)), SMAXT=glf(Math.max(0.5,diag*1.2)),
        AOH=glf(Math.max(0.01,diag*0.05));
  const amb=scene.ambient||{sky:[0.10,0.12,0.16],ground:[0.06,0.05,0.045],up:[0,0,1]};
  let lightBlk="";
  (scene.lights||[]).forEach(L=>{ const d=L.dir,c=L.color,I=L.intensity;
    lightBlk += `  { vec3 Ld=normalize(vec3(${glf(d[0])},${glf(d[1])},${glf(d[2])}));\n`
             +  `    vec3 Lc=vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])})*${glf(I)};\n`
             +  `    vec3 Li=-Ld; float ndl=max(0.0,dot(n,Li));\n`
             +  `    float sh = ndl>0.0 ? softShadow(p+n*SHEPS, Li) : 1.0;   // cast toward the light\n`
             +  `    diff+=Lc*ndl*sh;\n`
             +  `    vec3 hh=normalize(Li+v); spec+=Lc*pow(max(0.0,dot(n,hh)),specExp)*ndl*sh; }\n`; });
  const sky=amb.sky,gr=amb.ground,au=amb.up||[0,0,1];
  // ── water: a rippling SDF height field from real gravity-capillary waves ──
  const hasWater = ls.some(l=>l.shape && l.shape.type==="Water");
  let waterGLSL = "";
  if(hasWater){
    const comps=(scene.water&&scene.water.components)||[];
    let hsum="", gsum="";
    comps.forEach(c=>{ const ph=`(${glf(c.kx)}*xz.x+${glf(c.kz)}*xz.y-${glf(c.omega)}*uTime+${glf(c.phase)})`;
      hsum += `  h+=${glf(c.amp)}*cos${ph};\n`;
      gsum += `  { float s=-${glf(c.amp)}*sin${ph}; g+=s*vec2(${glf(c.kx)},${glf(c.kz)}); }\n`; });
    waterGLSL = `
float waveH(vec2 xz){ float h=0.0;\n${hsum}  return h; }
vec2 waveGrad(vec2 xz){ vec2 g=vec2(0.0);\n${gsum}  return g; }
float sdWater(vec3 p, vec3 c, float hx, float hz, float depth, float level){
  float top = p.y-(level+waveH(p.xz));          // below the wavy surface
  float bottom = (level-depth)-p.y;             // above the basin floor
  vec2 dxz = abs(p.xz-c.xz)-vec2(hx,hz);        // within the footprint
  return max(max(top,bottom), max(dxz.x,dxz.y));
}`;
  }
  // light glints (the reflection of the source) for escaped reflection rays
  let glintBlk="";
  (scene.lights||[]).forEach(L=>{ const d=L.dir,c=L.color,I=L.intensity;
    glintBlk += `  col += vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])})*${glf(I)}`
             +  `*pow(max(0.0,dot(rd,-normalize(vec3(${glf(d[0])},${glf(d[1])},${glf(d[2])})))),1200.0)*9.0;\n`; });
  // cheap diffuse-only lighting for the single reflection bounce (no AO/shadow)
  let lightBlkCheap="";
  (scene.lights||[]).forEach(L=>{ const d=L.dir,c=L.color,I=L.intensity;
    lightBlkCheap += `  diff += vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])})*${glf(I)}`
                  +  `*max(0.0,dot(n,-normalize(vec3(${glf(d[0])},${glf(d[1])},${glf(d[2])}))));\n`; });
  const envExpr = `mix(vec3(${glf(gr[0])},${glf(gr[1])},${glf(gr[2])}),vec3(${glf(sky[0])},${glf(sky[1])},${glf(sky[2])}),u)`;
  return `#version 300 es
precision highp float; precision highp int;
#define MAXB ${Math.max(1, bodies.length)}
#define SHEPS ${SHEPS}
#define SHMIN ${SHMIN}
#define SHMAX ${SHMAX}
#define SMAXT ${SMAXT}
#define SHK 14.0
#define AOH ${AOH}
#define EMISSION_SCALE 2400000000.0
out vec4 frag;
uniform vec2 uRes; uniform vec3 uEye,uFwd,uRight,uUp;
uniform float uTanHalf,uAspect,uMaxDist,uTime,uTemp;
uniform vec3 uBodyPos[MAXB]; uniform vec4 uBodyQuat[MAXB];
uniform float uBodyTemp[MAXB];   // per-frame body temperature (K); -1 = no frame value
${PRIMS_GLSL}${waterGLSL}${EXTRA_GLSL}
float mapD(vec3 p, out int mat){
${bodyDecls}${decls}${compose}${matsel}  return d;
}
float mapOnly(vec3 p){ int m; return mapD(p,m); }
vec3 matColor(int m){
${colorfn}  return vec3(0.72);
}
float metalness(int m){
${metalfn}  return 0.0;
}
float reflR0(int m){         // Schlick R0: clear dielectric from Fresnel, metals strong, else -1
${reflfn}  return -1.0;
}
float specExpOf(int m){      // Blinn-Phong exponent from the material's real roughness
${roughfn}  return 32.0;
}
vec3 emissivityOf(int m){    // Kirchhoff ε(λ): measured (1−R from n+k) where available
${emisfn}  return vec3(0.5);
}
float matTempK(int m){       // per-object initial temperature (sim layer; default = theater ambient)
${tempfn}  return ${glf(envT)};
}
float bodyTempK(int m){      // the trajectory frame's body temperature, lerped on the CPU; -1 = absent
${bodyTempFn}  return -1.0;
}
float fieldTempK(int m, vec3 p){   // per-cell field, sampled leaf-local; -1 = no field on this leaf
${fieldSel}  return -1.0;
}
float effTempK(int m, vec3 p){     // precedence: per-cell field > frame body T > leaf static T
  float ft = fieldTempK(m, p);     //   (> theater ambient baked above); the heat slider (uTemp)
  if(ft >= 0.0) return ft + uTemp; //   adds a user-probe delta ON TOP of whichever won —
  float bt = bodyTempK(m);         //   replacement not addition: same quantity, different times
  return (bt >= 0.0 ? bt : matTempK(m)) + uTemp;
}
float softShadow(vec3 ro, vec3 rd){          // sphere-traced soft shadow toward a light
  float res=1.0, t=SHEPS;
  for(int i=0;i<40;i++){ float h=mapOnly(ro+rd*t);
    if(h<0.0008) return 0.0;                 // fully blocked
    res=min(res, SHK*h/t);                   // penumbra: nearer occluders → softer edge
    t+=clamp(h,SHMIN,SHMAX); if(t>SMAXT) break; }
  return clamp(res,0.0,1.0);
}
// NON-DERIVED (audit): ambient occlusion is a darkening heuristic, NOT light
// transport — a fast-preview crutch; the path tracer computes real occlusion.
// The 1.6 gain and AOH radius are hand-tuned. Retires with the fast path.
float calcAO(vec3 p, vec3 n){                 // 5-tap SDF ambient occlusion (creases self-darken)
  float occ=0.0, sca=1.0;
  for(int i=0;i<5;i++){ float hr=AOH*(0.2+0.2*float(i));
    occ += (hr-mapOnly(p+n*hr))*sca; sca*=0.7; }
  return clamp(1.0-1.6*occ,0.0,1.0);
}
vec3 calcN(vec3 p){ float h=2e-4; vec2 e=vec2(1.0,-1.0);
  return normalize( e.xyy*mapOnly(p+e.xyy*h)+e.yyx*mapOnly(p+e.yyx*h)
                  + e.yxy*mapOnly(p+e.yxy*h)+e.xxx*mapOnly(p+e.xxx*h) ); }
vec3 shade(vec3 albedo, vec3 p, vec3 n, vec3 v, float metal, float specExp){
  float ao=calcAO(p,n);
  vec3 AUP=normalize(vec3(${glf(au[0])},${glf(au[1])},${glf(au[2])}));
  float hemi=0.5+0.5*dot(n,AUP);
  vec3 ambient=mix(vec3(${glf(gr[0])},${glf(gr[1])},${glf(gr[2])}),vec3(${glf(sky[0])},${glf(sky[1])},${glf(sky[2])}),hemi)*ao;
  vec3 diff=vec3(0.0), spec=vec3(0.0);
${lightBlk}  diff += vec3(0.35)*max(0.0,dot(n,v))*(0.5+0.5*ao);   // soft camera headlight (lightly AO'd)
  vec3 F0=mix(vec3(0.04),albedo,metal);
  return albedo*ambient + albedo*diff*(1.0-0.85*metal) + F0*spec*(0.5+0.5*metal);
}
vec3 envColor(vec3 rd){            // bright sky for reflections + the glint of each light source
  vec3 AUP=normalize(vec3(${glf(au[0])},${glf(au[1])},${glf(au[2])}));
  float u=clamp(0.5+0.5*dot(rd,AUP),0.0,1.0);
  vec3 col=mix(vec3(0.42,0.50,0.58),vec3(0.27,0.42,0.70),u);   // horizon→zenith (the env, not the dim fill)
${glintBlk}  return col;
}
vec3 shadeReflected(vec3 albedo, vec3 n, vec3 v, float metal){   // cheap single-bounce shade
  vec3 AUP=normalize(vec3(${glf(au[0])},${glf(au[1])},${glf(au[2])}));
  float u=0.5+0.5*dot(n,AUP);
  vec3 ambient=mix(vec3(${glf(gr[0])},${glf(gr[1])},${glf(gr[2])}),vec3(${glf(sky[0])},${glf(sky[1])},${glf(sky[2])}),u);
  vec3 diff=vec3(0.0);
${lightBlkCheap}  diff += vec3(0.28)*max(0.0,dot(n,v));
  return albedo*(ambient*0.8+diff*(1.0-0.6*metal));
}
vec3 traceReflect(vec3 ro, vec3 rd){   // one reflection bounce: hit geometry, or escape to sky+glint
  float t=SHMIN;
  for(int i=0;i<80;i++){ vec3 q=ro+rd*t; float d=mapOnly(q);
    if(d<0.001){ vec3 n2=calcN(q); int m2; mapD(q-n2*0.0008,m2);
      return shadeReflected(matColor(m2), n2, -rd, metalness(m2)); }
    t+=clamp(d,SHMIN,SHMAX); if(t>SMAXT) break; }
  return envColor(rd);
}
${INCANDESCENCE_GLSL}
void main(){
  vec2 uv=(gl_FragCoord.xy/uRes)*2.0-1.0;
  vec3 dir=normalize(uFwd + uRight*(uv.x*uAspect*uTanHalf) + uUp*(uv.y*uTanHalf));
  float t=0.0; bool hitF=false; vec3 p=uEye;
  for(int i=0;i<256;i++){ p=uEye+dir*t; float d=mapOnly(p);
    if(d<0.0006){ hitF=true; break; } t+=d; if(t>uMaxDist) break; }
  vec3 col=vec3(0.0);                              // black = no matter (the doctrine)
  if(hitF){
    vec3 n=calcN(p);
    int mat; mapD(p-n*0.0008, mat);               // material just INSIDE the surface
    float metal=metalness(mat);
    col=shade(matColor(mat), p, n, -dir, metal, specExpOf(mat));
    float r0=reflR0(mat);
    if(r0>=0.0){                                  // reflective surface → cast a real reflection ray
      float cosi=max(0.0,dot(n,-dir));
      float fres=clamp(r0+(1.0-r0)*pow(1.0-cosi,5.0),0.0,1.0);   // Fresnel-Schlick: grazing → mirror
      vec3 rd=reflect(dir,n);
      vec3 rcol=traceReflect(p+n*max(SHMIN,0.002), rd);
      vec3 tint=metal>0.5?matColor(mat):vec3(1.0);              // metals tint their reflection
      col=mix(col, rcol*tint, fres);
    }
    col += incandescence(effTempK(mat, p), emissivityOf(mat));   // each POINT at ITS temperature — per-cell field, else per-frame body T (Planck × Kirchhoff)
  }
  frag=vec4(pow(clamp(col,0.0,1.0),vec3(0.4545)),1.0);  // sRGB gamma: black stays black (doctrine), dim hues read
}`;
}

// ── R3 anti-drift keystone: ONE generator emits BOTH NEE blocks ──────────
// (i) the surface neeBlk (visMode = visRay) — whose output is asserted
// byte-identical to the pre-R3 shipped template below, so the near-field
// emitter baselines (iron cube, solar_earth surfaces) can never drift; and
// (ii) the impostor twin (visMode = analytic spheres) — the SAME
// cone-sampling estimator, same Le/G/neeExp, same uEmitterPos/uEmitterTemp
// uniforms, only the hit-context variables and the visibility test differ.
// visRay is REJECTED for the impostor twin: its 64 × uSHMAX(0.06·R) step
// budget cannot span AU-scale body→emitter distances and silently returns
// lit — the analytic ray-vs-sphere test is exact at any distance.
function emitNEEBlock(e, k, neeExp, o){
  // o = {guard, hitVar, normalVar, albVar, visExpr}
  return `
    if(${o.guard}){ vec3 C=uEmitterPos[${k}]; float R=${glf(e.radius)}; vec3 wc=C-${o.hitVar}; float dc=length(wc);
      if(dc>R+uEPS){ vec3 wcn=wc/dc; float sinT=R/dc; float cosTm=sqrt(max(0.0,1.0-sinT*sinT));
        float cT=1.0-rnd(seed)*(1.0-cosTm); float sTs=sqrt(max(0.0,1.0-cT*cT)); float ph=6.2831853*rnd(seed);
        vec3 tt=normalize(cross(wcn, abs(wcn.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0))); vec3 bb=cross(wcn,tt);
        vec3 wl=normalize(tt*(sTs*cos(ph))+bb*(sTs*sin(ph))+wcn*cT); float ndl2=max(0.0,dot(${o.normalVar},wl));
        if(ndl2>0.0){ float vis=${o.visExpr};
          vec3 Le=incandescence(uEmitterTemp[${k}]+uTemp, vec3(${glf(e.emis[0])},${glf(e.emis[1])},${glf(e.emis[2])}));
          L += thr*${o.albVar}*Le*ndl2*2.0*(1.0-cosTm)*vis*${neeExp}; } } }`;
}
function neeSurfaceOpts(e, svoRow){
  // svoRow (R4): when the emitter's own leaf is SVO-routed its hits carry
  // mat = SVO_MAT_BASE+(row<<8)+gid where row is the STABLE body-table row
  // (index in packed.bodies — rows are never compacted, see the draw()
  // packing note), so `mat!=leaf` alone no longer skips NEE on the emitter's
  // own surface — extend the guard by the row id. Omitted (undefined)
  // everywhere else, so the frozen template assert below stays byte-exact.
  return {guard:`mat!=${e.leaf}`+(svoRow!==undefined?`&&(mat>>8)!=${1+svoRow}`:""),
          hitVar:"p", normalVar:"n", albVar:"alb",
          visExpr:"visRay(p+n*uEPS, wl, dc-R-2.0*uEPS)"};
}
function neeImpostorOpts(e){
  return {guard:`bi!=${e.body}`, hitVar:"q", normalVar:"nI", albVar:"aI",
          visExpr:`impVis(q+nI*uEPS, wl, dc-R-2.0*uEPS, bi, ${e.body})`};
}
// build-time assertion: the surface variant must reproduce the pre-R3
// shipped template CHARACTER-FOR-CHARACTER (the frozen reference below is
// that template verbatim, values substituted) — throw loudly on any drift.
(function _assertNEETemplate(){
  const e={leaf:7, radius:2.5, emis:[0.5,0.25,0.125]};
  const got=emitNEEBlock(e, 3, "6000.00000000", neeSurfaceOpts(e));
  const want=`
    if(mat!=7){ vec3 C=uEmitterPos[3]; float R=${glf(2.5)}; vec3 wc=C-p; float dc=length(wc);
      if(dc>R+uEPS){ vec3 wcn=wc/dc; float sinT=R/dc; float cosTm=sqrt(max(0.0,1.0-sinT*sinT));
        float cT=1.0-rnd(seed)*(1.0-cosTm); float sTs=sqrt(max(0.0,1.0-cT*cT)); float ph=6.2831853*rnd(seed);
        vec3 tt=normalize(cross(wcn, abs(wcn.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0))); vec3 bb=cross(wcn,tt);
        vec3 wl=normalize(tt*(sTs*cos(ph))+bb*(sTs*sin(ph))+wcn*cT); float ndl2=max(0.0,dot(n,wl));
        if(ndl2>0.0){ float vis=visRay(p+n*uEPS, wl, dc-R-2.0*uEPS);
          vec3 Le=incandescence(uEmitterTemp[3]+uTemp, vec3(${glf(0.5)},${glf(0.25)},${glf(0.125)}));
          L += thr*alb*Le*ndl2*2.0*(1.0-cosTm)*vis*6000.00000000; } } }`;
  if(got!==want) throw new Error("R3: emitNEEBlock drifted from the shipped surface NEE template");
})();

// ── path-traced renderer (toggle): real light transport — no headlight/ambient/AO.
// Rays bounce off the derived BRDFs; the sky is an area light; and every object
// EMITS its own incandescence, so a hot object lights its cold neighbours. Noisy
// at low sample count (Monte Carlo) — accumulation/denoise is the next rung.
function buildPathShader(scene){
  resetExtraGLSL();
  const ls=scene.csg_leaves, bodies=scene.bodies||[];
  // R3 compile gate: banding exists ONLY for scenes whose bodies carry a
  // real radius_m (verified census: only the solar theater today) AND when
  // splats.js loaded. Every other scene compiles byte-identical pre-R3
  // shader text and uploads nothing new — those baselines hold ARITHMETICALLY.
  const banding = !!splatsCtl && bodies.some(b=>b && b.radius_m>0);
  // ── R4 compile gate (R3 banding-flag precedent): eligible Voxel leaves are
  // routed OUT of the per-leaf codegen entirely — the shader becomes leaf-
  // count-independent for voxel matter; a zero-Voxel scene (or svoEnabled
  // false) compiles BYTE-IDENTICAL text (string-asserted at module load).
  const svoLeaves = svoCtl ? svoEligible(scene) : [];
  const svoActive = svoEnabled && svoLeaves.length>0;
  const svoSet = new Set(svoActive ? svoLeaves.map(e=>e.leaf) : []);
  const legacyIdx = ls.map((_,i)=>i).filter(i=>!svoSet.has(i));
  const used=[...new Set(legacyIdx.map(i=>ls[i].body).filter(b=>b!=null))].sort((a,b)=>a-b);
  let bodyDecls="";
  used.forEach(k=>{ const piv=(bodies[k]&&bodies[k].pivot)||[0,0,0];
    bodyDecls += `  vec3 bdy${k}=qrotInv(uBodyQuat[${k}], p-uBodyPos[${k}])+vec3(${glf(piv[0])},${glf(piv[1])},${glf(piv[2])});\n`; });
  let decls="", compose=legacyIdx.length?`  float d=s${legacyIdx[0]};\n`:"  float d=1e9;\n", matsel="  mat=-1;\n", colorfn="", metalfn="", reflfn="", emisfn="", tempfn="", iorfn="", alphafn="", bodyTempFn="";
  const envT=(scene.physics_env&&scene.physics_env.temperature_k!==undefined)?scene.physics_env.temperature_k:293.15;
  for(let i=0;i<ls.length;i++){
    if(svoSet.has(i)) continue;                    // SVO-routed: zero shader text
    // non-body leaves (floor/walls/lights — static theater dressing) are
    // baked as ABSOLUTE world-space literals; p is camera-relative here
    // (R1-lite), so reconstruct world-space via uEyeWorld before querying.
    // Body-bound leaves need no such fix — see bindScene's rebase comment.
    const q = ls[i].body!=null ? `bdy${ls[i].body}` : "(p+uEyeWorld)";
    const bnd = ls[i].bounds || deriveBounds(ls[i].shape);   // baked, else derived
    if(bvhCull && bnd){
      const bc=bnd.center, br=glf(bnd.radius);
      // bd{i} (signed distance to the padded bounding sphere) is the correct far-leaf
      // substitute for EVERY op, subtract included -- there is no separate "1e9 no-op"
      // case. Correctness (re-derived, not just asserted): this branch only fires when
      // bd{i}>GATE_BUFFER>0, and the bounding-sphere property gives 0<bd{i}<=s{i}_exact
      // (the leaf's true SDF) whenever that holds, so -bd{i}>=-s{i}_exact. compose's
      // subtract branch does d=max(d,-s{i}); by monotonicity of max() in its second
      // argument, max(d_prev,-bd{i}) >= max(d_prev,-s{i}_exact) = d_exact, for ANY
      // d_prev. Two cases: if d_exact>=0 (point already outside via prior leaves), the
      // -s{i}_exact term never wins that max (it's <=0), so the substitute changes
      // nothing and d_approx=d_exact=d_prev exactly -- no error. If d_exact<0 (point
      // still inside after this leaf -- e.g. deep in a large solid, near a buried
      // subtract cavity whose bounding sphere is gated far), both d_exact and d_approx
      // are <=0, so d_exact<=d_approx<=0 means |d_approx|<=|d_exact|: the substituted
      // step can only be SMALLER than the true distance, never larger -- exactly the
      // non-overshoot property sphere tracing needs. (A prior version of this code used
      // a 1e9 sentinel here instead, reasoning that compose's -s{i} consumption flips a
      // lower bound on s{i} into an unsafe upper bound on -s{i} -- true in isolation,
      // but the 1e9 "no-op" is *worse*: max(d_prev,-1e9)=d_prev unclamped, and d_prev
      // can have far larger magnitude than the true -s{i}_exact clip, so the ray steps
      // straight past a nearby-but-gated-far cavity wall. Reproduced: 0.55m position
      // divergence marching toward a buried r=0.05 subtract cavity 0.5m inside a r=2.0
      // solid sphere, fixed by this change -- see tools/verify_bvh_cull_subtract.py.)
      // i===0 needs no special case either way: leaf 0 is compose's unconditional base
      // `d=s0`, never routed through the subtract branch regardless of its own op
      // field, and bd0 is what it would want anyway.
      //
      // GATE_BUFFER (not just ">0.0"): bd{i} alone is a safe lower bound (never
      // overshoots), but "safe" only means mapD's OUTPUT never claims a leaf is
      // farther than it truly is — it does NOT mean a small bd{i} implies we're
      // near that leaf's REAL surface. bd{i} is the signed distance to the
      // PADDED SPHERE's own boundary, which for a leaf whose true shape is
      // large/thin/elongated relative to its bounding radius (a room wall: real
      // half-thickness ~0.02m, bounding_radius ~2m+) sits deep in otherwise-empty
      // space. march()/sunShadow()/visRay() terminate whenever the COMPOSED d
      // drops below their hit epsilon (0.0008/0.001) -- if a substituted bd{i}
      // is what's small, that termination is a phantom hit on the invisible
      // bounding sphere, not real geometry (confirmed empirically: replaying
      // mapD()+march() for a full camera-ray sweep against this codegen with a
      // bare ">0.0" gate showed 99.9% of rays landing >10cm from the exact-eval
      // hit point, worst case ~3.3m). Fix: only ever substitute bd{i} when it is
      // > GATE_BUFFER, a constant comfortably larger than every hit epsilon
      // those three functions use. That guarantees a SUBSTITUTED term can never
      // itself be < hit-epsilon, so a composed d that DOES drop below hit-epsilon
      // can only be won by a genuine exact evaluation -- the property the design
      // needed but a bare ">0.0" gate didn't provide. Re-verified after this fix:
      // the same camera-ray sweep drops to 0 hit/miss or material divergences
      // and >99.9% of rays within 1mm of the exact-eval hit point (the tiny
      // remainder traced to sphere-tracing's own pre-existing grazing-angle
      // sensitivity on thin geometry, not to this gate).
      // R3 band gate: a body-bound UNION leaf whose body has left band 0 is
      // removed from the SDF entirely — s{i}=1e9 is EXACT under op:add
      // (min(d,1e9)=d). REFUSED for subtract/intersect leaves (the
      // verify_bvh_cull_subtract lesson: a substituted -s is an unsafe upper
      // bound) — none exist in any banding scene today (all 30 theater
      // leaves are op:add, verified); a future one warns and stays ungated.
      const gateK = (banding && ls[i].body!=null && bodies[ls[i].body]
                     && bodies[ls[i].body].radius_m>0
                     && (i===0 || !ls[i].op || ls[i].op==="add")) ? ls[i].body : null;
      if(banding && ls[i].body!=null && bodies[ls[i].body] && bodies[ls[i].body].radius_m>0
         && gateK===null)
        console.warn(`R3: leaf ${i} (op=${ls[i].op}) is body-bound but NOT band-gated — subtract/intersect gating is unsafe`);
      if(gateK!=null){
        decls += `  float s${i}; if(uBand[${gateK}]!=0){ s${i}=1e9; } else { `
              +  `float bd${i}=length(${q}-vec3(${glf(bc[0])},${glf(bc[1])},${glf(bc[2])}))-${br}; `
              +  `if(bd${i}>uGateBuffer){ s${i}=bd${i}; } else { s${i}=${glslShapeCall(ls[i].shape, q)}; } }\n`;
      } else {
        decls += `  float bd${i}=length(${q}-vec3(${glf(bc[0])},${glf(bc[1])},${glf(bc[2])}))-${br};\n`;
        decls += `  float s${i}; if(bd${i}>uGateBuffer){ s${i}=bd${i}; } else { s${i}=${glslShapeCall(ls[i].shape, q)}; }\n`;
      }
    } else {
      const gateK = (banding && ls[i].body!=null && bodies[ls[i].body]
                     && bodies[ls[i].body].radius_m>0
                     && (i===0 || !ls[i].op || ls[i].op==="add")) ? ls[i].body : null;
      if(gateK!=null)
        decls += `  float s${i}; if(uBand[${gateK}]!=0){ s${i}=1e9; } else { s${i}=${glslShapeCall(ls[i].shape, q)}; }\n`;
      else
        decls += `  float s${i}=${glslShapeCall(ls[i].shape, q)};\n`;
    }
    const mat=scene.materials[ls[i].material]||{}, c=mat.color_rgb||[0.72,0.72,0.72];
    colorfn += `  if(m==${i}) return vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])});\n`;
    const isMetal=(mat.metal!==undefined?mat.metal:mat.emergent);
    metalfn += `  if(m==${i}) return ${glf(isMetal?1.0:0.0)};\n`;
    reflfn += `  if(m==${i}) return ${glf((mat.reflect_r0!==undefined)?mat.reflect_r0:-1.0)};\n`;
    const em=mat.emissivity_rgb;
    emisfn += em ? `  if(m==${i}) return vec3(${glf(em[0])},${glf(em[1])},${glf(em[2])});\n`
                 : `  if(m==${i}) return clamp(vec3(1.0)-vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}),0.0,1.0);\n`;
    tempfn += `  if(m==${i}) return ${glf(ls[i].temperature_k!==undefined?ls[i].temperature_k:envT)};\n`;
    iorfn += `  if(m==${i}) return ${glf(mat.refractive_index!==undefined?mat.refractive_index:-1.0)};\n`;
    // Render-only opacity: the scene's baked default (mat.opacity, 1.0 if the
    // export never set it), overridden per-label by the client-side toggle map
    // if one is active. Independent of ior/reflect_r0 — see opacityOf()/trace().
    { const baseOp=(mat.opacity!==undefined)?mat.opacity:1.0;
      const ov=materialOpacity[ls[i].material];
      const op=(ov!==undefined)?ov:baseOp;
      alphafn += `  if(m==${i}) return ${glf(op)};\n`; }
    bodyTempFn += `  if(m==${i}) return ${ls[i].body!=null?`uBodyTemp[${ls[i].body}]`:"-1.0"};\n`;
  }
  for(let n=1;n<legacyIdx.length;n++){ const i=legacyIdx[n]; const op=ls[i].op;
    compose += (op==="subtract")?`  d=max(d,-s${i});\n`:(op==="intersect")?`  d=max(d,s${i});\n`:`  d=min(d,s${i});\n`; }
  for(let n=legacyIdx.length-1;n>=0;n--){ const i=legacyIdx[n];
    matsel += `  ${n===legacyIdx.length-1?"if":"else if"}(s${i}<0.0) mat=${i};\n`; }
  const fieldSel=glslFieldFns(scene,"+uEyeWorld");  // per-cell field samplers → EXTRA_GLSL + switch (PT: p is camera-relative)
  const bb=scene.bbox||[[-1,1],[-1,1],[-1,1]];
  const diag=len(sub([bb[0][1],bb[1][1],bb[2][1]],[bb[0][0],bb[1][0],bb[2][0]]))||1;
  // ── ghost tethers: violet, translucent markers at each DRIVE plug's own
  // joint anchor (Plug.at/.axis, dynamics/mechanisms/plug.py) — the
  // artificial motor/actuator the scene's HUD panel already discloses in
  // TEXT (violet #818CF8, same color, see the "plugs" HUD block below),
  // made visible IN the render too. Appended as synthetic leaves AFTER
  // every real one (indices ls.length..) so they never touch
  // scene.csg_leaves/jsEvalSDF — the Python ground-truth self-check stays
  // exact, unaware ghosts exist. Real translucency (opacityOf/trace's
  // stochastic alpha gate) only lives in THIS shader — the old-GPU
  // compatibility shim (buildFragmentShader) skips ghosts entirely and
  // renders the scene exactly as it always has.
  const ghosts=(scene.plugs||[]).filter(pg=>pg.kind==="drive"&&Array.isArray(pg.at));
  const GHOST_RGB=[0.5059,0.5490,0.9725];             // #818CF8 — the HUD's own Plug violet
  const gR=Math.max(0.004,diag*0.006), gHalf=Math.max(0.02,diag*0.045);
  ghosts.forEach((pg,gi)=>{
    const at=pg.at, axRaw=Array.isArray(pg.axis)?pg.axis:null;
    const ax=(axRaw&&len(axRaw)>1e-9)?norm(axRaw):[0,1,0];   // fall back to world-up if no axis shipped
    const A=sub(at,mul(ax,gHalf)), B=add(at,mul(ax,gHalf));
    const gidx=ls.length+gi;
    decls += `  float s${gidx}=sdCapsuleCyl(p+uEyeWorld,vec3(${glf(A[0])},${glf(A[1])},${glf(A[2])}),`
           + `vec3(${glf(B[0])},${glf(B[1])},${glf(B[2])}),${glf(gR)});\n`;
    compose += `  d=min(d,s${gidx});\n`;
    matsel += `  ${(gi===0&&legacyIdx.length===0)?"if":"else if"}(s${gidx}<0.0) mat=${gidx};\n`;
    colorfn += `  if(m==${gidx}) return vec3(${glf(GHOST_RGB[0])},${glf(GHOST_RGB[1])},${glf(GHOST_RGB[2])});\n`;
    metalfn += `  if(m==${gidx}) return 0.0;\n`;
    reflfn  += `  if(m==${gidx}) return -1.0;\n`;
    // dim self-glow (not a real blackbody -- a HUD-like indicator that
    // reads clearly regardless of which way the scene's lights point)
    emisfn  += `  if(m==${gidx}) return vec3(${glf(GHOST_RGB[0]*0.5)},${glf(GHOST_RGB[1]*0.5)},${glf(GHOST_RGB[2]*0.5)});\n`;
    tempfn  += `  if(m==${gidx}) return ${glf(envT)};\n`;
    iorfn   += `  if(m==${gidx}) return -1.0;\n`;
    alphafn += `  if(m==${gidx}) return 0.35;\n`;      // low-opacity: an artificial marker, not real matter
    bodyTempFn += `  if(m==${gidx}) return -1.0;\n`;
  });
  // ── R4: the SVO traversal chunk + material-LUT heads (svoActive only —
  // every string below is "" otherwise, keeping zero-Voxel scenes byte-
  // identical). Material property functions gain ONE guarded head line each;
  // analytic leaves keep their codegen chains untouched (hybrid composition).
  let svoGlsl="", svoMapTerm="";
  if(svoActive){
    svoGlsl = svoCtl.glslSVO();
    svoMapTerm = svoCtl.glslSVOMapTerm();
    colorfn    = `  if(m>=SVO_MAT_BASE) return texelFetch(uMatLUT, ivec2(m & 255, 0), 0).rgb;\n` + colorfn;
    metalfn    = `  if(m>=SVO_MAT_BASE) return texelFetch(uMatLUT, ivec2(m & 255, 0), 0).a;\n` + metalfn;
    emisfn     = `  if(m>=SVO_MAT_BASE) return texelFetch(uMatLUT, ivec2(m & 255, 1), 0).rgb;\n` + emisfn;
    iorfn      = `  if(m>=SVO_MAT_BASE) return texelFetch(uMatLUT, ivec2(m & 255, 1), 0).a;\n` + iorfn;
    alphafn    = `  if(m>=SVO_MAT_BASE) return texelFetch(uMatLUT, ivec2(m & 255, 2), 0).x;\n` + alphafn;
    tempfn     = `  if(m>=SVO_MAT_BASE) return texelFetch(uMatLUT, ivec2(m & 255, 2), 0).w;\n` + tempfn;
    bodyTempFn = `  if(m>=SVO_MAT_BASE) return BT((m >> 8) - 1, 3).z;\n` + bodyTempFn;
  }
  // SHMIN/SHMAX/SMAXT/EPS/GATE_BUFFER/calcN's h used to be baked from the
  // scene's whole-scene bbox diagonal ONCE at shader-build time — fine for a
  // tabletop scene, but wrong for a scene spanning solar-system distances:
  // zooming into one planet within a huge bbox still used shadow/hit
  // epsilons sized for the WHOLE system. They are now real uniform floats
  // (uEPS/uSHMIN/uSHMAX/uSMAXT/uGateBuffer/uCalcNH, declared below),
  // recomputed every frame in JS from the CURRENT camera-to-target distance
  // (cam.radius) — see draw()'s bindScene. This is rendering-only LOD (the
  // physics never changes), and it is deliberately the CURRENT-VIEW scale,
  // not a per-march-step pixel-footprint term — a coarser first cut than the
  // full per-sample scaling the render-at-scale plan calls for, noted as
  // such rather than presented as the finished version.
  const amb=scene.ambient||{up:[0,0,1]}, au=amb.up||[0,0,1];
  const hasWater=ls.some(l=>l.shape&&l.shape.type==="Water");
  let waterGLSL=""; if(hasWater){ const comps=(scene.water&&scene.water.components)||[]; let hsum="",gsum="";
    comps.forEach(c=>{ const ph=`(${glf(c.kx)}*xz.x+${glf(c.kz)}*xz.y-${glf(c.omega)}*uTime+${glf(c.phase)})`;
      hsum+=`  h+=${glf(c.amp)}*cos${ph};\n`; gsum+=`  { float s=-${glf(c.amp)}*sin${ph}; g+=s*vec2(${glf(c.kx)},${glf(c.kz)}); }\n`; });
    waterGLSL=`\nfloat waveH(vec2 xz){ float h=0.0;\n${hsum}  return h; }\nfloat sdWater(vec3 p, vec3 c, float hx, float hz, float depth, float level){ float top=p.y-(level+waveH(p.xz)); float bottom=(level-depth)-p.y; vec2 dxz=abs(p.xz-c.xz)-vec2(hx,hz); return max(max(top,bottom),max(dxz.x,dxz.y)); }`; }
  const L0=(scene.lights&&scene.lights[0])||{dir:[0.45,0.4,-0.85],color:[1,0.96,0.9],intensity:1.05};
  const Sd=`vec3(${glf(L0.dir[0])},${glf(L0.dir[1])},${glf(L0.dir[2])})`,
        Sc=`vec3(${glf(L0.color[0])},${glf(L0.color[1])},${glf(L0.color[2])})`, Si=glf(L0.intensity||1.0);
  // ── emitters for next-event estimation: sphere leaves hot enough to glow.
  // Their WORLD centre + TEMPERATURE are uploaded each frame (uEmitterPos /
  // uEmitterTemp) so a body's motion carries the light and a drag-heated fall
  // brightens as it accelerates. Registration is compile-time (shaders build
  // once per scene), so a body that crosses the 700 K Draper point at ANY
  // trajectory frame must be an emitter for the program's lifetime —
  // scene._bodyTempMax (applyObj scans the incoming trajectory) covers that.
  const emitters=[];
  ls.forEach((l,i)=>{ const Ts=l.temperature_k;
    const Tdyn=(l.body!=null&&scene._bodyTempMax)?(scene._bodyTempMax[l.body]??-1):-1;
    const Tmax=Math.max(Ts!==undefined?Ts:-1, Tdyn);
    // NEE needs a sphere to sample a solid angle toward — Sphere leaves give
    // this exactly; a Voxel leaf (e.g. a body rendered as a real voxel grid,
    // not an analytic primitive) has no closed-form radius, so its baked
    // bvhCull `bounds` (a real, if slightly conservative, bounding sphere —
    // see scene_export.py's _bounding_sphere) stands in. Anything else
    // (Box, Torus, a CSG composite with no bounds) still can't be an
    // emitter — an honest scope limit, not silently wrong geometry.
    const emitCenter = l.shape&&l.shape.type==="Sphere" ? l.shape.center
                      : (l.bounds ? l.bounds.center : null);
    const emitRadius = l.shape&&l.shape.type==="Sphere" ? l.shape.radius
                      : (l.bounds ? l.bounds.radius : null);
    if(emitCenter&&emitRadius&&Tmax>700){
      const m=scene.materials[l.material]||{}, e=m.emissivity_rgb||[0.5,0.5,0.5];
      emitters.push({leaf:i, center:emitCenter.slice(), radius:emitRadius,
                     body:(l.body!=null?l.body:-1),
                     temp:(Ts!==undefined?Ts:envT), emis:e}); }});
  ptEmitters=emitters; const NE=emitters.length;
  // NEE exposure: `2.0*(1.0-cosTm)` is the real geometric solid-angle
  // falloff (correct: a sun-sized emitter at AU-scale distance subtends
  // ~1e-5 of the solid angle a bell-scale demo emitter does a few metres
  // away, which is what EMISSION_SCALE was tuned against) — the physics is
  // right, the EXPOSURE isn't, at this distance regime. `nee_exposure`
  // (scene-level, default 1.0 — every existing scene's calibration is
  // untouched) is a disclosed render-only Choice, exactly the same kind of
  // NON-DERIVED audit-flagged knob as incandescence()'s own tone-map gain.
  const neeExp=glf(scene.nee_exposure!==undefined?scene.nee_exposure:1.0);
  // surface NEE blocks (visRay) + R3 impostor twins (analytic spheres) from
  // the ONE generator — see emitNEEBlock's header; the surface variant is
  // byte-asserted against the pre-R3 template at module load.
  let neeBlk="", neeTwin="";
  emitters.forEach((e,k)=>{
    // svoLeaves order == packScene's bodies order, so the packed (stable) row
    // of an SVO-routed emitter is its index in svoLeaves
    const svoRow = svoSet.has(e.leaf)
      ? svoLeaves.findIndex(x=>x.leaf===e.leaf) : undefined;
    neeBlk  += emitNEEBlock(e, k, neeExp, neeSurfaceOpts(e, svoRow>=0?svoRow:undefined));
    if(banding) neeTwin += emitNEEBlock(e, k, neeExp, neeImpostorOpts(e));
  });
  // ── R3 module state (house precedent: ptEmitters is set the same way) ──
  bandingEnabled = banding;
  _r3EmitterBodies = new Set(emitters.filter(e=>e.body>=0).map(e=>e.body));
  // per-body render-param table: the JOIN of scene.bodies × csg_leaves ×
  // materials (bodies carry only {label,pivot,radius_m}; albedo/temp come
  // from the leaf's material and temperature_k — one param definition
  // consumed by BOTH the impostor const arrays and the splat static buffer).
  let bandHdr="";
  if(banding){
    const NBod=bodies.length;
    const rad=[], albA=[], emisA=[], tmpA=[];
    const stat=new Float32Array(NBod*8);
    const radJS=new Array(NBod);
    for(let k=0;k<NBod;k++){
      const leaf=ls.find(l=>l.body===k);
      const m=leaf?(scene.materials[leaf.material]||{}):{};
      const c=m.color_rgb||[0.72,0.72,0.72];
      const em=m.emissivity_rgb
        || [Math.min(Math.max(1-c[0],0),1),Math.min(Math.max(1-c[1],0),1),Math.min(Math.max(1-c[2],0),1)];  // the emisfn rule
      const rM=(bodies[k]&&bodies[k].radius_m>0)?bodies[k].radius_m:0;
      const T=(leaf&&leaf.temperature_k!==undefined)?leaf.temperature_k:envT;
      rad.push(glf(rM)); albA.push(`vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])})`);
      emisA.push(`vec3(${glf(em[0])},${glf(em[1])},${glf(em[2])})`); tmpA.push(glf(T));
      radJS[k]=rM;
      stat[k*8]=rM; stat[k*8+1]=c[0]; stat[k*8+2]=c[1]; stat[k*8+3]=c[2];
      stat[k*8+4]=T; stat[k*8+5]=em[0]; stat[k*8+6]=em[1]; stat[k*8+7]=em[2];
    }
    _r3Params={radius:radJS, staticArr:stat};
    // UNROLLED, CONSTANT-INDEX codegen (the house style: bdy0..bdyN, the
    // matsel/colorfn if-chains — the whole shader avoids dynamic indexing
    // except uBodyPos[CONST]). A first cut used a `for(k<MAXB)` loop with
    // dynamically-indexed tables here and it MISCOMPILED on ANGLE/D3D11:
    // its mere presence in trace()'s miss branch blacked out every
    // RAYMARCHED surface at solar_earth (measured this session — with
    // uBand all zero, i.e. the loop dormant, and in both const-array and
    // uniform-array forms). Constant-index unrolling matches what this
    // driver stack demonstrably compiles correctly.
    bandHdr=`uniform int uBand[MAXB];   // R3: 0 raymarch · 1 impostor · 2 splat (splats.js classifier; default 0 = pre-R3)
// analytic emitter visibility for impostor hits — exact at ANY distance
// (visRay's 64×uSHMAX step budget cannot span AU and silently returns lit);
// skips the impostor body itself (skipA) and the emitter's own body (skipB).
float impVis(vec3 ro, vec3 rd, float maxt, int skipA, int skipB){
  float v=1.0;
${bodies.map((b,k)=>{ const r=radJS[k]; if(!(r>0)) return "";
  return `  if(${k}!=skipA && ${k}!=skipB){ vec3 oc=uBodyPos[${k}]-ro; float tca=dot(oc,rd);
    if(tca>0.0 && tca<maxt && dot(oc,oc)-tca*tca < ${glf(r)}*${glf(r)}) v=0.0; }\n`; }).join("")}  return v;
}
`;
  } else { _r3Params=null; }
  // trace()'s miss branch: the R3 impostor band composites HERE (3–24 px
  // analytic spheres, terminal), splats (<2.4–3 px) render in a separate
  // GL_POINTS pass — the plan's split. Non-banding scenes keep the pre-R3
  // line verbatim. Per-body radii/albedo/emissivity/temps are inline
  // LITERALS in each unrolled block (baked like colorfn — zero traffic);
  // only NON-emitter bodies get a block (the emitter rule: emitters never
  // enter band 1), so no NEE double-count is possible here by construction.
  const missDefault=`vec3 p; if(!march(ro,rd,p)){ L+=thr*skyEmit(rd); break; }`;
  let impBlocks="";
  if(banding){
    const emBodies=new Set(emitters.filter(e=>e.body>=0).map(e=>e.body));
    bodies.forEach((b,k)=>{
      const r=(b&&b.radius_m>0)?b.radius_m:0;
      if(!(r>0) || emBodies.has(k)) return;
      const leaf=ls.find(l=>l.body===k);
      const m=leaf?(scene.materials[leaf.material]||{}):{};
      const c=m.color_rgb||[0.72,0.72,0.72];
      const em=m.emissivity_rgb
        || [Math.min(Math.max(1-c[0],0),1),Math.min(Math.max(1-c[1],0),1),Math.min(Math.max(1-c[2],0),1)];
      const T=(leaf&&leaf.temperature_k!==undefined)?leaf.temperature_k:envT;
      impBlocks += `
      if(uBand[${k}]==1){ vec3 oc=ro-uBodyPos[${k}]; float b2=dot(oc,rd); float c2=dot(oc,oc)-${glf(r)}*${glf(r)};
        float h2=b2*b2-c2; if(h2>=0.0){ float tI=-b2-sqrt(h2);
          if(tI>uEPS && tI<bT){ bT=tI; bi=${k}; bC=uBodyPos[${k}]; bR=${glf(r)};
            bAl=vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}); bEm=vec3(${glf(em[0])},${glf(em[1])},${glf(em[2])});
            bTs=${glf(T)}; bBt=uBodyTemp[${k}]; } } }`;
    });
  }
  const missBlk = !banding ? missDefault : `vec3 p; if(!march(ro,rd,p)){
      // R3 impostors, ALL bounces (bounce rays keep seeing distant bodies).
      // float32 note: at rebased AU-scale |oc|~1e11 the b2*b2-c2
      // cancellation is ~1e-4 relative on the disc radius — sub-pixel at
      // 3–24 px. Unrolled constant-index blocks — see the bandHdr note.
      float bT=1e30; int bi=-1; vec3 bC=vec3(0.0); float bR=1.0;
      vec3 bAl=vec3(0.0); vec3 bEm=vec3(0.0); float bTs=0.0; float bBt=-1.0;${impBlocks}
      if(bi>=0){ vec3 q=ro+rd*bT; vec3 nI=(q-bC)/bR; vec3 aI=bAl;
        // self-glow: provably vec3(0) today — every >700 K body-bound leaf
        // auto-registers as an NEE emitter (it has bounds) and emitters never
        // enter band 1, so bTs/bBt < 700 K (Draper) for every possible
        // impostor. Kept for structural identity with surface shading.
        L += thr*incandescence(bBt>=0.0?bBt:bTs, bEm);
        // direct-sun/sky L0 terms omitted: the one banding scene ships
        // PT_ENV=0.0 (all its light is NEE) — a disclosed residual for any
        // future banding scene with PT_ENV>0.${neeTwin}
        break; }
      L+=thr*skyEmit(rd); break; }`;
  return `#version 300 es
precision highp float; precision highp int;
#define MAXB ${Math.max(1,bodies.length)}
#define EMISSION_SCALE 2400000000.0
#define PT_ENV ${glf(scene.pt_env!==undefined?scene.pt_env:1.0)}
out vec4 frag;
uniform vec2 uRes; uniform vec3 uEye,uFwd,uRight,uUp;
uniform float uTanHalf,uAspect,uMaxDist,uTime,uTemp,uFrame,uReset;
// uEye is always vec3(0) post-rebase (the camera IS the origin in this
// shader's coordinate space — see bindScene). uEyeWorld carries the real
// absolute camera position, needed only to reconstruct world-space for
// geometry baked as absolute literals (non-body leaves, field grids,
// ghost tethers) that a per-frame body transform never touches.
uniform vec3 uEyeWorld;
// view-scale-adaptive epsilons (JS: draw()'s bindScene, from cam.radius) —
// bvhCull's per-leaf gate buffer (uGateBuffer): a bd{i} bounding-sphere
// substitute is only ever used when it exceeds this, comfortably above
// every hit epsilon mapD's consumers use below (march() ~0.4*uEPS,
// sunShadow()/visRay() ~0.5*uEPS) — see the decls-loop comment in
// buildPathShader for why this specific property (a substituted term can
// never itself register as a hit) is load-bearing.
uniform float uEPS,uSHMIN,uSHMAX,uSMAXT,uGateBuffer,uCalcNH;
uniform int uSPP;             // samples THIS frame — the photon budget for this single exposure
uniform float uPrevScale;     // fraction of the PRIOR exposure to keep: 1=integrate (long exposure), (N-1)/N=bounded shutter hold
uniform vec3 uBodyPos[MAXB]; uniform vec4 uBodyQuat[MAXB];
uniform float uBodyTemp[MAXB];                           // per-frame body temperature (K); -1 = none
uniform sampler2D uPrev;                                 // previous accumulation (ping-pong)
uniform vec3 uEmitterPos[${Math.max(1,NE)}];             // world centres of glowing-sphere emitters (NEE)
uniform float uEmitterTemp[${Math.max(1,NE)}];           // their live temperatures — a heating fall brightens
${bandHdr}${PRIMS_GLSL}${waterGLSL}${EXTRA_GLSL}${svoGlsl}
float mapD(vec3 p, out int mat){ ${bodyDecls}${decls}${compose}${matsel}${svoMapTerm}  return d; }
float mapOnly(vec3 p){ int m; return mapD(p,m); }
vec3 matColor(int m){ ${colorfn}  return vec3(0.72); }
float metalness(int m){ ${metalfn}  return 0.0; }
vec3 emissivityOf(int m){ ${emisfn}  return vec3(0.5); }
float matTempK(int m){ ${tempfn}  return ${glf(envT)}; }
float bodyTempK(int m){ ${bodyTempFn}  return -1.0; }
float fieldTempK(int m, vec3 p){ ${fieldSel}  return -1.0; }   // per-cell field, leaf-local; -1 = none
float effTempK(int m, vec3 p){ float ft=fieldTempK(m,p); if(ft>=0.0) return ft+uTemp;   // field > frame body T > leaf T > ambient
  float bt=bodyTempK(m); return (bt>=0.0?bt:matTempK(m))+uTemp; }
float iorOf(int m){ ${iorfn}  return -1.0; }              // refractive index (water/glass), else -1
float opacityOf(int m){ ${alphafn}  return 1.0; }         // render-only alpha (1=opaque); NOT ior — see trace()
vec3 calcN(vec3 p){ float h=uCalcNH; vec2 e=vec2(1.0,-1.0);
  return normalize( e.xyy*mapOnly(p+e.xyy*h)+e.yyx*mapOnly(p+e.yyx*h)+e.yxy*mapOnly(p+e.yxy*h)+e.xxx*mapOnly(p+e.xxx*h) ); }
${INCANDESCENCE_GLSL}
uint hashu(uint x){ x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x; }
float rnd(inout uint s){ s=hashu(s); return float(s)*(1.0/4294967296.0); }
vec3 cosHemi(vec3 n, float u1, float u2){ float r=sqrt(u1), phi=6.2831853*u2;
  vec3 t=normalize(cross(n, abs(n.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0))); vec3 b=cross(n,t);
  return normalize(t*(r*cos(phi))+b*(r*sin(phi))+n*sqrt(max(0.0,1.0-u1))); }
bool march(vec3 ro, vec3 rd, out vec3 p){ float t=uEPS;   // |d| sphere-trace → works inside a medium too (refraction)
  for(int i=0;i<220;i++){ p=ro+rd*t; float d=mapOnly(p); if(abs(d)<uEPS*0.4) return true; t+=max(abs(d),uEPS*0.25); if(t>uMaxDist) return false; } return false; }
float sunShadow(vec3 ro, vec3 rd){ float t=uEPS;
  for(int i=0;i<64;i++){ float d=mapOnly(ro+rd*t); if(d<uEPS*0.5) return 0.0; t+=clamp(d,uSHMIN,uSHMAX); if(t>uSMAXT) break; } return 1.0; }
float visRay(vec3 ro, vec3 rd, float maxt){ float t=uEPS;   // unoccluded toward an emitter up to maxt?
  for(int i=0;i<64;i++){ if(t>maxt) return 1.0; float d=mapOnly(ro+rd*t); if(d<uEPS*0.5) return 0.0; t+=clamp(d,uSHMIN,uSHMAX); } return 1.0; }
vec3 skyEmit(vec3 rd){ vec3 AUP=normalize(vec3(${glf(au[0])},${glf(au[1])},${glf(au[2])}));
  float u=clamp(0.5+0.5*dot(rd,AUP),0.0,1.0);
  // NON-DERIVED (audit): these sky RGBs are hand-picked, not Rayleigh/Mie from a
  // real atmosphere — the largest remaining physics-to-screen gap in the PT path.
  vec3 c=PT_ENV*mix(vec3(0.50,0.57,0.65),vec3(0.32,0.47,0.75),u);           // sky area light (dimmable per scene)
  c += PT_ENV*${Sc}*${Si}*9.0*pow(max(0.0,dot(rd,-normalize(${Sd}))),3000.0);  // the sun disc
  return c; }
vec3 trace(vec3 ro, vec3 rd, inout uint seed){
  vec3 L=vec3(0.0), thr=vec3(1.0); bool spec=true;       // primary ray counts as a specular arrival
  for(int b=0;b<8;b++){                                  // bounce depth: nested dielectrics (a glass of water) need >3 to cross every interface and reach the bright far side instead of dying black inside
    ${missBlk}
    vec3 n=calcN(p); int mat; mapD(p-n*0.0008,mat);
    // RENDER-ONLY opacity gate — an artificial "make X see-through" toggle, not
    // real transmission physics: independent of ior/Fresnel (the dielectric
    // branch below is untouched and still does the real glass/water optics).
    // Stochastic alpha test: with probability (1-op) this sample treats the
    // surface as ABSENT — no shading, no glow, no albedo tint, ray continues
    // straight through unbent — so across many accumulated samples the pixel
    // converges to an honest opacity BLEND of "material" and "whatever is
    // behind it", the alpha-compositing result, never a fake-refracted ghost.
    float op=opacityOf(mat);
    if(op<1.0 && rnd(seed)>=op){
      vec3 nf=n; if(dot(rd,n)>=0.0) nf=-n; ro=p-nf*uEPS*2.0; continue;   // pass through, direction unchanged
    }
    if(spec) L += thr*incandescence(effTempK(mat, p), emissivityOf(mat)); // own glow at the POINT's temperature — only on specular arrival (NEE counts it otherwise; NEE cast light stays bulk-T — flagged: the gradient shows on the emitter's own surface, not yet in its cast light)
    vec3 alb=matColor(mat); float metal=metalness(mat);
    float ior=iorOf(mat); bool diffuse=(metal<0.5 && ior<=0.0);
    if(diffuse){                                                           // direct lighting — diffuse surfaces only
      vec3 Li=-normalize(${Sd}); float ndl=max(0.0,dot(n,Li));             // direct sun (next-event)
      if(ndl>0.0){ float sh=sunShadow(p+n*uEPS, Li); L += thr*alb*${Sc}*${Si}*PT_ENV*ndl*sh; }
      ${neeBlk}                                                            // direct light from glowing objects (NEE)
    }
    if(ior>0.0){                                                           // DIELECTRIC: Fresnel reflect / Snell refract
      vec3 nf=n; float eta; if(dot(rd,n)<0.0){ eta=1.0/ior; } else { eta=ior; nf=-n; }
      float ci=clamp(-dot(rd,nf),0.0,1.0); float r0=pow((ior-1.0)/(ior+1.0),2.0);
      float fres=r0+(1.0-r0)*pow(1.0-ci,5.0); vec3 rt=refract(rd,nf,eta);
      if(dot(rt,rt)<1e-6) fres=1.0;                                        // total internal reflection
      if(rnd(seed)<fres){ rd=reflect(rd,nf); ro=p+nf*uEPS*2.0; }            // reflect (same side)
      else { rd=rt; ro=p-nf*uEPS*2.0; }                                     // transmit (cross the interface)
      spec=true;
    } else if(metal>0.5){ rd=reflect(rd,n); thr*=alb; ro=p+n*uEPS; spec=true; }   // metal: tinted specular
    else { rd=cosHemi(n, rnd(seed), rnd(seed)); thr*=alb; ro=p+n*uEPS; spec=false; }   // diffuse: cosine bounce
    if(max(thr.r,max(thr.g,thr.b))<0.02) break;
  }
  return L;
}
void main(){
  vec2 fc=gl_FragCoord.xy;
  uint seed=hashu(uint(fc.x)+uint(fc.y)*9277u+uint(uFrame)*26699u);
  vec3 col=vec3(0.0); const int MAXSPP=64;
  for(int s=0;s<MAXSPP;s++){ if(s>=uSPP) break;            // dynamic photon budget (uSPP), hard-capped at MAXSPP
    vec2 j=vec2(rnd(seed),rnd(seed))-0.5;
    vec2 uv=((fc+j)/uRes)*2.0-1.0;
    vec3 dir=normalize(uFwd + uRight*(uv.x*uAspect*uTanHalf) + uUp*(uv.y*uTanHalf));
    col += trace(uEye, dir, seed);
  }
  vec3 s=min(col/float(uSPP), vec3(16.0));                // this frame's radiance (firefly-clamped, linear HDR)
  vec3 prev = uReset>0.5 ? vec3(0.0) : texelFetch(uPrev, ivec2(fc), 0).rgb * uPrevScale;
  frag=vec4(prev+s, 1.0);                                 // ACCUMULATE — the display pass averages + gammas
}`;
}

// ── GL program plumbing ───────────────────────────────────────────────
const VS=`#version 300 es
void main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
  gl_Position=vec4(p*2.0-1.0,0.0,1.0); }`;
function compile(type,src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){ const log=gl.getShaderInfoLog(s); console.error(log,"\n",src); throw new Error(log); } return s; }
let prog=null, progPT=null, U={}, Upt={};
const UNIFORMS=["uRes","uEye","uEyeWorld","uFwd","uRight","uUp","uTanHalf","uAspect","uMaxDist","uTime","uTemp","uBodyPos","uBodyQuat","uBodyTemp","uFrame","uPrev","uReset","uEmitterPos","uEmitterTemp","uSPP","uPrevScale","uEPS","uSHMIN","uSHMAX","uSMAXT","uGateBuffer","uCalcNH","uBand",
  "uSVONodes","uSdfAtlas","uMatLUT","uSVOBody","uNumSvoBodies","uAtlasDim","uAtlasBpa"];   // R4 (null on non-SVO programs — every bind is guarded)
function linkProg(fragSrc){
  const p=gl.createProgram();
  gl.attachShader(p,compile(gl.VERTEX_SHADER,VS));
  gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fragSrc));
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const fetchU=(p)=>{ const u={}; for(const n of UNIFORMS) u[n]=gl.getUniformLocation(p,n); return u; };

// ── voxel solids → WebGL2 3-D textures (the signed-distance grid in hardware) ──
function b64ToI8(b64){ const bin=atob(b64), a=new Int8Array(bin.length);
  for(let i=0;i<bin.length;i++) a[i]=(bin.charCodeAt(i)<<24)>>24; return a; }   // byte → signed
function makeVoxTex(grid, dims){
  // Upload as R8 UNORM (i8 → u8 by +128): universally LINEAR-filterable on the
  // ANGLE/D3D backend, where R8_SNORM 3-D textures often are NOT (→ read as 0).
  // GLSL decodes (r·255−128)·band/127 back to the signed distance in metres.
  const u8=new Uint8Array(grid.length);
  for(let i=0;i<grid.length;i++) u8[i]=grid[i]+128;
  const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D,tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);                                        // rows are nx bytes — no padding
  gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,dims[0],dims[1],dims[2],0,gl.RED,gl.UNSIGNED_BYTE,u8);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);              // trilinear in hardware
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_3D,null); return tex;
}
function prepareVoxels(scene){     // decode + upload each Voxel leaf once; tag it with a stable sampler id
  const list=[]; let vid=0;
  const leaves=scene.csg_leaves||[];
  for(let i=0;i<leaves.length;i++){ const l=leaves[i]; const s=l.shape;
    if(s && s.type==="Voxel"){
      // R4: leaves the shim routed out (yard-class scenes) upload no dense
      // texture — the SVO atlas is their sole residency; every other scene
      // keeps the dense upload (shim + A/B legacy path) unchanged.
      if(scene._svoSkipVox && scene._svoSkipVox.has(i)){
        if(s._grid===undefined) s._grid=b64ToI8(s.sdf_b64);   // self-check still needs bytes
        continue;
      }
      if(s._tex===undefined){ s._vid=vid; s._grid=b64ToI8(s.sdf_b64); s._tex=makeVoxTex(s._grid,s.dims); }
      list.push({vid:s._vid, tex:s._tex}); vid++;
    } }
  scene._voxels=list;
}
function fetchVoxU(p, scene){ const m={};
  for(const v of (scene._voxels||[])) m[v.vid]=gl.getUniformLocation(p,"uVox"+v.vid); return m; }

// ── per-cell FIELDS (leaf `fields` registry — temperature first) ───────────
// A field is FROZEN sim output: uint8 keyframe grids over one shared
// [t_min,t_max] (u8-xfast — the sdf_b64 pack), uploaded once as R8/LINEAR
// 3-D textures. Playback binds the two bracketing keyframes and mixes by
// uFldU — display reconstruction between frozen sim states, NEVER the heat
// equation. Grids live in the LEAF-LOCAL frame, so the body transform carries
// a hot rind on a moving body for free.
function b64ToU8(b64){ const bin=atob(b64), a=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) a[i]=bin.charCodeAt(i); return a; }
function makeFieldTex(u8, dims){                    // same proven params as makeVoxTex, raw u8
  const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D,tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
  gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,dims[0],dims[1],dims[2],0,gl.RED,gl.UNSIGNED_BYTE,u8);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_3D,null); return tex;
}
function prepareFields(scene){    // decode + upload each leaf's temperature field once
  const list=[]; let fid=0;
  const V=(scene._voxels||[]).length;
  for(let i=0;i<(scene.csg_leaves||[]).length;i++){
    const l=scene.csg_leaves[i], f=(l.fields||{}).temperature_k;
    if(!f || f.encoding!=="u8-xfast") continue;
    // texture-unit budget: 0 = PT accumulation, 1..V = voxel SDFs, 2 per field
    if(1+V+2*(list.length+1) > 16){
      console.error(`field on leaf ${i} dropped — texture-unit budget exceeded `
        +`(1 accum + ${V} voxels + ${2*(list.length+1)} field units > 16); bulk T still shows`);
      continue;
    }
    if(f._texs===undefined){
      f._grids=f.keyframes.map(k=>b64ToU8(k.values_b64));
      f._texs=f._grids.map(g=>makeFieldTex(g,f.grid.dims));
    }
    f._fid=fid; l._fid=fid;
    list.push({fid, leaf:i, times:f.keyframes.map(k=>k.t_sim), texs:f._texs, spec:f});
    fid++;
  }
  scene._fields=list;
}
function fetchFieldU(p, scene){ const m={};
  for(const f of (scene._fields||[]))
    m[f.fid]={a:gl.getUniformLocation(p,"uFldA"+f.fid),
              b:gl.getUniformLocation(p,"uFldB"+f.fid),
              u:gl.getUniformLocation(p,"uFldU"+f.fid)};
  return m; }
// JS twin of the GPU field read (and of Python's field_trilinear) — the field
// self-check mirror. rec = {leaf, k, p, T}; p is LEAF-LOCAL (pose-independent).
function jsFieldSampleAt(scene, rec){
  const l=(scene.csg_leaves||[])[rec.leaf]; if(!l) return NaN;
  const f=(l.fields||{})[rec.quantity||"temperature_k"]; if(!f||!f._grids) return NaN;
  const g=f._grids[rec.k]; if(!g) return NaN;
  const n=f.grid.dims, vs=f.grid.voxel_size, c=f.grid.center;
  const nx=n[0],ny=n[1],nz=n[2], p=rec.p;
  let fi=(p[0]-c[0])/vs+(nx-1)*0.5, fj=(p[1]-c[1])/vs+(ny-1)*0.5, fk=(p[2]-c[2])/vs+(nz-1)*0.5;
  const ci=Math.min(Math.max(fi,0),nx-1), cj=Math.min(Math.max(fj,0),ny-1), ck=Math.min(Math.max(fk,0),nz-1);
  const i0=Math.floor(ci),j0=Math.floor(cj),k0=Math.floor(ck);
  const i1=Math.min(i0+1,nx-1),j1=Math.min(j0+1,ny-1),k1=Math.min(k0+1,nz-1);
  const di=ci-i0,dj=cj-j0,dk=ck-k0;
  const at=(i,j,k)=>g[(k*ny+j)*nx+i];               // x-fastest, the shared pack
  const c00=at(i0,j0,k0)*(1-di)+at(i1,j0,k0)*di, c01=at(i0,j0,k1)*(1-di)+at(i1,j0,k1)*di;
  const c10=at(i0,j1,k0)*(1-di)+at(i1,j1,k0)*di, c11=at(i0,j1,k1)*(1-di)+at(i1,j1,k1)*di;
  const q=(c00*(1-dj)+c10*dj)*(1-dk)+(c01*(1-dj)+c11*dj)*dk;
  return f.t_min + q*(f.t_max-f.t_min)/255.0;
}
// ── R4: SVO GL residency (one node pool + ONE brick atlas + LUT + body table)
function releaseSVO(){
  if(!svoGL) return;
  for(const t of [svoGL.nodeTex,svoGL.atlasTex,svoGL.lutTex,svoGL.bodyTex])
    try{ gl.deleteTexture(t); }catch(_e){}
  svoGL=null;
}
// svodry tracing: stage beacons dispatch off-thread even when the NEXT call
// wedges the main thread — the bisect instrument's breadcrumb trail.
const _svoTrace = ["1"].includes(new URLSearchParams(location.search).get("svodry")) || new URLSearchParams(location.search).get("svotrace")==="1";
function _svoBeacon(stage){
  if(!_svoTrace) return;
  try{  // SYNCHRONOUS XHR on purpose: strictly ordered single-flight POSTs —
        // concurrent sendBeacons race /probe's read-modify-write (measured:
        // corrupted misc/BROWSER_REPORT.json twice this session)
    const x=new XMLHttpRequest(); x.open("POST","/probe",false);
    x.send(JSON.stringify({slug:"svotrace_"+stage, t: performance.now()|0}));
  }catch(_e){}
}
function prepareSVO(scene){
  _svoBeacon("prepareSVO-enter");
  if(!scene._svoPacked){
    const t0=performance.now();
    scene._svoPacked = svoCtl.packScene(scene._svoEligible);
    const st=scene._svoPacked&&scene._svoPacked.stats;
    if(st) console.log(`R4 svo build ${(performance.now()-t0).toFixed(0)} ms: `
      +`${st.trees} trees, ${st.nodes} nodes, ${st.bricks} bricks, atlas ${st.atlas_dim}^3 `
      +`(${(st.atlas_bytes/1048576).toFixed(1)} MB) vs dense ${(st.dense_bytes/1048576).toFixed(2)} MB`);
  }
  _svoBeacon("packScene-done");
  const P=scene._svoPacked;
  if(!P){ svoActiveNow=false; return; }
  releaseSVO();
  const H=Math.max(1,Math.ceil(P.pool.length/4/4096));
  const poolPadded=new Uint32Array(4096*H*4); poolPadded.set(P.pool);
  const nt=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,nt);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32UI,4096,H,0,gl.RGBA_INTEGER,gl.UNSIGNED_INT,poolPadded);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  _svoBeacon("pool-uploaded");
  const at=gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D,at);   // makeVoxTex params (R8 UNORM LINEAR — the ANGLE law)
  gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
  gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,P.atlasDim,P.atlasDim,P.atlasDim,0,gl.RED,gl.UNSIGNED_BYTE,P.atlas);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_3D,null);
  _svoBeacon("atlas-uploaded");
  // material LUT (256x4 RGBA32F) — rebuilt every build so materialOpacity rides row 2
  const envT=(scene.physics_env&&scene.physics_env.temperature_k!==undefined)?scene.physics_env.temperature_k:293.15;
  const lut=new Float32Array(256*4*4);
  P.palette.forEach((label,id)=>{
    const m=scene.materials[label]||{};
    const c=m.color_rgb||[0.72,0.72,0.72];
    const isMetal=(m.metal!==undefined?m.metal:m.emergent)?1:0;
    const em=m.emissivity_rgb
      || [Math.min(Math.max(1-c[0],0),1),Math.min(Math.max(1-c[1],0),1),Math.min(Math.max(1-c[2],0),1)];
    const ior=(m.refractive_index!==undefined)?m.refractive_index:-1;
    const baseOp=(m.opacity!==undefined)?m.opacity:1;
    const ov=materialOpacity[label];
    const op=(ov!==undefined)?ov:baseOp;
    const r0=(m.reflect_r0!==undefined)?m.reflect_r0:-1;
    const rg=m.roughness;
    const specExp=(rg!==undefined&&rg>0)?Math.min(256,Math.max(8,2/(rg*rg)-2)):(isMetal?64:24);
    const lf=scene._svoEligible.find(e=>e.material===label);
    const leaf=lf?scene.csg_leaves[lf.leaf]:null;
    const T=(leaf&&leaf.temperature_k!==undefined)?leaf.temperature_k:envT;
    lut[(0*256+id)*4]=c[0]; lut[(0*256+id)*4+1]=c[1]; lut[(0*256+id)*4+2]=c[2]; lut[(0*256+id)*4+3]=isMetal;
    lut[(1*256+id)*4]=em[0]; lut[(1*256+id)*4+1]=em[1]; lut[(1*256+id)*4+2]=em[2]; lut[(1*256+id)*4+3]=ior;
    lut[(2*256+id)*4]=op; lut[(2*256+id)*4+1]=r0; lut[(2*256+id)*4+2]=specExp; lut[(2*256+id)*4+3]=T;
  });
  const lt=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,lt);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,256,4,0,gl.RGBA,gl.FLOAT,lut);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  const bt=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,bt);   // body table, per-frame texSubImage2D
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,4,svoCtl.MAX_SVO_BODIES,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D,null);
  svoGL={nodeTex:nt,atlasTex:at,lutTex:lt,bodyTex:bt,packed:P,
         rowBuf:new Float32Array(16*P.bodies.length)};
  svoReady=true;
  _svoBeacon("prepareSVO-done");
}
function buildProgram(scene){
  // R4: eligibility + residency decided BEFORE the shaders are generated
  scene._svoEligible = svoCtl ? svoEligible(scene) : [];
  const _svoOn = svoEnabled && scene._svoEligible.length>0;
  scene._svoShimRouted = _svoOn && scene._svoEligible.length>48;   // the measured-good analytic bound
  scene._svoSkipVox = new Set(scene._svoShimRouted ? scene._svoEligible.map(e=>e.leaf) : []);
  svoActiveNow = _svoOn; svoReady = false;
  if(_svoOn) prepareSVO(scene); else releaseSVO();
  // debug-only DRY mode (?svodry=1): stash the generated shader texts and
  // SKIP compilation — the compile-bisect instrument (a wedged ANGLE compile
  // blocks the main thread, so variants get compiled in a Worker instead).
  if(new URLSearchParams(location.search).get("svodry")==="1"){
    _svoBeacon("dry-prevoxels");
    prepareVoxels(scene); prepareFields(scene);
    _svoBeacon("dry-gentext");
    window.__shaders = {pt: buildPathShader(scene), fast: buildFragmentShader(scene)};
    _svoBeacon("dry-stashed");
    console.log("svodry: shader texts stashed on window.__shaders — nothing compiled");
    return;
  }
  prepareVoxels(scene);                              // decode/upload voxel grids before the shaders reference them
  prepareFields(scene);                              // …and the per-cell field keyframes
  _svoBeacon("shim-link-start");
  const t=linkProg(buildFragmentShader(scene));      // fast renderer (always)
  _svoBeacon("shim-linked");
  let pt=null; try{ _svoBeacon("pt-link-start"); pt=linkProg(buildPathShader(scene)); }   // path tracer (graceful if it fails)
  catch(e){ console.error("path-trace shader failed:",e); }
  _svoBeacon("pt-linked");
  if(prog) gl.deleteProgram(prog); if(progPT) gl.deleteProgram(progPT);
  prog=t; progPT=pt; U=fetchU(t); Upt=pt?fetchU(pt):{};
  U.vox=fetchVoxU(t,scene); Upt.vox=pt?fetchVoxU(pt,scene):{};
  U.fld=fetchFieldU(t,scene); Upt.fld=pt?fetchFieldU(pt,scene):{};
  buildSplatPipeline(scene);                         // R3: GL_POINTS far-field (no-op unless bandingEnabled)
  _svoBeacon("splat-built");
}

// ── R3 splat pass: GL_POINTS far field into an RGBA16F offscreen target ──
// CHOICE separate RGBA16F target, NOT the RGBA32F accumulation buffer:
// 32F blending needs EXT_float_blend (unqueried at R0-baseline time,
// unverified on the capture machine); 16F blending is core with the
// already-required EXT_color_buffer_float. Splat radiance is closed-form
// deterministic per frame, so accumulating it buys nothing — and R0's most
// sacred buffer stays untouched. 16F precision: the faintest relevant splat
// (~3e-8 linear) sits below the half-float subnormal floor and renders 0 —
// post-gamma error << 1/255, honestly invisible, matching the physics.
const SPLAT_FS=`#version 300 es
precision highp float;
in vec3 vCol; out vec4 frag;
void main(){ frag=vec4(vCol,1.0); }`;
function buildSplatPipeline(scene){
  if(splatProg){ gl.deleteProgram(splatProg); splatProg=null; }
  if(!bandingEnabled || !_r3Params) return;
  const NBod=(scene.bodies||[]).length;
  const NE=Math.max(1,ptEmitters.length);
  const emRad=ptEmitters.length?ptEmitters.map(e=>glf(e.radius)).join(","):glf(0);
  const emEmis=ptEmitters.length
    ? ptEmitters.map(e=>`vec3(${glf(e.emis[0])},${glf(e.emis[1])},${glf(e.emis[2])})`).join(",")
    : "vec3(0.0,0.0,0.0)";
  const neeExp=glf(scene.nee_exposure!==undefined?scene.nee_exposure:1.0);
  // The splat vertex shader computes the body's closed-form radiance:
  //   L_refl = Σ_k alb ⊙ Le_k · G_k · nee_exposure · f(α_k) · vis_k
  // — the closed-form disc-average EXPECTATION of the path tracer's own NEE
  // cone estimator (same Le = incandescence(emitter T, registration emis),
  // same G = 2(1−cosTm), same nee_exposure, same uEmitterTemp source), with
  // f(α) = (2/(3π))(sin α + (π−α)cos α), the Lambert-sphere phase function
  // (verified against numerical integration in test_splats.node.js §4).
  //   L_th = incandescence(T_body, emisSelf) — the SAME shared chunk the PT
  // uses (INCANDESCENCE_GLSL), so a receding hot body keeps its exact hue.
  // Flux conservation (deliverable-B honesty): total sprite flux =
  // disc radiance × (π/4)P² spread over the n×n sprite, clamped at the PT's
  // own 16.0 firefly clamp — dimming is exact inverse-square, no band pop.
  // Derivation-chain precedent: field/interface/detection.py (flux at
  // observer falls as 1/d²); the NEE comment at the emitter-registration
  // block (viewer.js buildPathShader).
  const vs=`#version 300 es
precision highp float; precision highp int;
#define EMISSION_SCALE 2400000000.0
#define MAXB ${Math.max(1,NBod)}
#define NE ${NE}
in vec3 aPos; in float aRadius; in vec3 aAlb; in float aTemp; in vec3 aEmis; in float aVis;
uniform vec3 uFwd,uRight,uUp;
uniform float uTanHalf,uAspect,uPxPerRad,uTemp,uPointCap;
uniform vec3 uEmPos[NE];            // rebased emitter centres — the SAME values the NEE upload builds
uniform float uEmTemp[NE];          // live emitter temperatures — same source as uEmitterTemp
uniform float uSplTemp[MAXB];       // per-frame body T override (-1 = none) — mirrors effTempK's frame-T precedence
const float EMRAD[NE]=float[NE](${emRad});     // NEE registration radii (bounds.radius — never radius_m)
const vec3 EMEMIS[NE]=vec3[NE](${emEmis});     // NEE registration emissivities (the [0.5,0.5,0.5] rule)
out vec3 vCol;
${INCANDESCENCE_GLSL}
void main(){
  int flags=int(aVis+0.5);          // bit0 draw · bit(k+1) emitter k visible (splats.js pack loop)
  float dist=length(aPos);
  float zf=dot(aPos,uFwd);
  if((flags&1)==0 || dist<=0.0 || zf<=0.0){    // not splat-band / eye-occluded / behind camera
    gl_Position=vec4(0.0,0.0,2.0,1.0); gl_PointSize=0.0; vCol=vec3(0.0); return; }
  float P=(2.0*aRadius/max(dist,1e-9))*uPxPerRad;      // apparent pixel diameter (same formula as splats.js)
  float n=min(uPointCap,max(1.0,ceil(P)));     // integer sprite (deterministic under R0's 3-render proof); ceil ⇒ sprite ≥ disc so the clamp never truncates in-band
  vec3 Lr=vec3(0.0);
  vec3 dirCam=-aPos/dist;
  for(int k=0;k<NE;k++){
    vec3 wc=uEmPos[k]-aPos; float dbe=length(wc);
    if(dbe<=EMRAD[k]*1.001) continue;          // self: an emitter's own splat skips its own L_refl (no sunlight reflecting off the sun) and keeps L_th
    float sinT=EMRAD[k]/dbe; float cosTm=sqrt(max(0.0,1.0-sinT*sinT));
    float G=2.0*(1.0-cosTm);                   // the NEE block's own solid-angle factor
    vec3 Le=incandescence(uEmTemp[k]+uTemp, EMEMIS[k]);
    float ca=clamp(dot(wc/dbe,dirCam),-1.0,1.0);
    float alpha=acos(ca);                      // phase angle: dir-to-emitter vs dir-to-camera
    float f=(2.0/(3.0*3.14159265358979))*(sin(alpha)+(3.14159265358979-alpha)*cos(alpha));
    float visE=float((flags>>(k+1))&1);        // eclipse test (CPU pack loop, analytic spheres)
    Lr+=aAlb*Le*G*${neeExp}*f*visE;
  }
  float Tk=uSplTemp[gl_VertexID]>=0.0?uSplTemp[gl_VertexID]:aTemp;
  vec3 Lth=incandescence(Tk+uTemp, aEmis);
  vCol=min((Lr+Lth)*min(1.0,0.78539816339745*P*P/(n*n)), vec3(16.0));   // flux-conserving spread + the PT firefly clamp
  vec2 ndc=vec2(dot(aPos,uRight)/(zf*uTanHalf*uAspect), dot(aPos,uUp)/(zf*uTanHalf));
  gl_Position=vec4(ndc,0.0,1.0);
  gl_PointSize=n;
}`;
  try{
    const p=gl.createProgram();
    gl.attachShader(p,compile(gl.VERTEX_SHADER,vs));
    gl.attachShader(p,compile(gl.FRAGMENT_SHADER,SPLAT_FS));
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    splatProg=p; splatU={};
    for(const nm of ["uFwd","uRight","uUp","uTanHalf","uAspect","uPxPerRad","uTemp","uPointCap","uEmPos","uEmTemp","uSplTemp"])
      splatU[nm]=gl.getUniformLocation(p,nm);
    // attribute state on a dedicated VAO (the shared VAO0 carries none):
    // STATIC per-scene buffer [radius, alb.rgb, temp, emis.rgb] = 32 B/body
    // uploaded ONCE; DYNAMIC per-frame position buffer 12 B/body (the
    // envelope's 72 MB/s @100k×60Hz projection) + 1 B/body visibility flags.
    if(!splatVAO) splatVAO=gl.createVertexArray();
    gl.bindVertexArray(splatVAO);
    const loc=(nm)=>gl.getAttribLocation(p,nm);
    if(splatBufStatic) gl.deleteBuffer(splatBufStatic);
    if(splatBufDyn) gl.deleteBuffer(splatBufDyn);
    if(splatBufVis) gl.deleteBuffer(splatBufVis);
    splatBufStatic=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,splatBufStatic);
    gl.bufferData(gl.ARRAY_BUFFER,_r3Params.staticArr,gl.STATIC_DRAW);
    const lR=loc("aRadius"), lA=loc("aAlb"), lT=loc("aTemp"), lE=loc("aEmis");
    if(lR>=0){ gl.enableVertexAttribArray(lR); gl.vertexAttribPointer(lR,1,gl.FLOAT,false,32,0); }
    if(lA>=0){ gl.enableVertexAttribArray(lA); gl.vertexAttribPointer(lA,3,gl.FLOAT,false,32,4); }
    if(lT>=0){ gl.enableVertexAttribArray(lT); gl.vertexAttribPointer(lT,1,gl.FLOAT,false,32,16); }
    if(lE>=0){ gl.enableVertexAttribArray(lE); gl.vertexAttribPointer(lE,3,gl.FLOAT,false,32,20); }
    splatBufDyn=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,splatBufDyn);
    gl.bufferData(gl.ARRAY_BUFFER,NBod*12,gl.DYNAMIC_DRAW);
    const lP=loc("aPos");
    if(lP>=0){ gl.enableVertexAttribArray(lP); gl.vertexAttribPointer(lP,3,gl.FLOAT,false,12,0); }
    splatBufVis=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,splatBufVis);
    gl.bufferData(gl.ARRAY_BUFFER,NBod,gl.DYNAMIC_DRAW);
    const lV=loc("aVis");
    if(lV>=0){ gl.enableVertexAttribArray(lV); gl.vertexAttribPointer(lV,1,gl.UNSIGNED_BYTE,false,1,0); }
    gl.bindVertexArray(VAO0);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);
  }catch(e){ console.error("R3 splat shader failed:",e); splatProg=null; }
}
function ensureSplatTarget(w,h){
  if(splatTarget && splatTW===w && splatTH===h) return;
  const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,w,h,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  const fbo=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  splatTarget={tex,fbo}; splatTW=w; splatTH=h;
}

// ── progressive-accumulation buffers (ping-pong float targets) ─────────
// Each PT frame ADDS its samples to a float buffer; the display pass divides by
// the sample count. Stand still and the image gathers light and converges — the
// physical analogue of a long exposure filling in (samples ≈ photons, var ∝ 1/N).
// R3: the display pass composites the splat far-field target ADDITIVELY,
// pre-clamp/pre-gamma, so splat radiance passes through the SAME tone
// mapping as PT radiance — structurally, and decoupled from uInvN/frames
// (splat radiance is closed-form per frame, not Monte-Carlo accumulated).
// Non-banding scenes bind a 1×1 black texture: c + 0.0 == c exactly for
// every c >= 0 (IEEE), so the nine non-solar baselines hold arithmetically.
const DISPLAY_FS=`#version 300 es
precision highp float;
uniform sampler2D uAccum; uniform sampler2D uSplat; uniform float uInvN;
out vec4 frag;
void main(){ vec3 c=texelFetch(uAccum, ivec2(gl_FragCoord.xy),0).rgb*uInvN;
  c += texelFetch(uSplat, ivec2(gl_FragCoord.xy),0).rgb;
  frag=vec4(pow(clamp(c,0.0,1.0),vec3(0.4545)),1.0); }`;
let dispProg=null, dispU={};
let accumA=null, accumB=null, pingA=true, accN=0, accW=0, accH=0, ptKey="", ptTime=0;
let splatBlackTex=null;                        // 1×1 black — the exact-zero splat source for non-banding scenes
// R3 band/splat state (all null/false until a banding scene loads — every
// other scene takes the byte-identical pre-R3 code paths):
let bandingEnabled=false;                      // scene.bodies carries a real radius_m (today: solar theater only)
let splatTarget=null, splatTW=0, splatTH=0;    // RGBA16F offscreen target for the GL_POINTS far field
let splatProg=null, splatU={}, splatVAO=null;  // the GL_POINTS program + its attribute state
let splatBufDyn=null, splatBufVis=null, splatBufStatic=null;
const splatsCtl = (typeof window!=="undefined" && window.SGSplats)
  ? new window.SGSplats.SGSplats() : null;     // band classifier/pack (splats.js); 404 → banding stays off
let _r3Params=null;                            // per-body render-param table {radius, staticArr} (built per banding scene)
let _r3EmitterBodies=new Set();                // NEE-registered body indices (the emitter rule)
let _r3SceneId=-1;                             // detects sceneId bumps → memoryless band reset
function initDisplay(){ if(dispProg||!extF) return;
  try{ dispProg=linkProg(DISPLAY_FS);
    dispU={uAccum:gl.getUniformLocation(dispProg,"uAccum"), uInvN:gl.getUniformLocation(dispProg,"uInvN"),
           uSplat:gl.getUniformLocation(dispProg,"uSplat")};
    splatBlackTex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,splatBlackTex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([0,0,0,0]));
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D,null); }
  catch(e){ console.error("display shader:",e); } }
function makeTarget(w,h){ const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,w,h,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  const fbo=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
  gl.bindFramebuffer(gl.FRAMEBUFFER,null); return {tex,fbo}; }
function ensureAccum(w,h){ if(accW===w&&accH===h&&accumA) return;
  accumA=makeTarget(w,h); accumB=makeTarget(w,h); accW=w; accH=h; accN=0; }

// ── state ─────────────────────────────────────────────────────────────
const cam={az:0.8,el:0.45,radius:0.3,target:[0,0,0],up:[0,0,1]};
const T0=performance.now();                        // wall-clock origin for live ripples
const STP_K=293.15;                                 // standard ambient — the sim layer's default temperature
let ptMode=!!extF, frameCount=0, sceneId=0;          // path tracing is THE renderer (whenever float targets exist)
let liveSPP=2;                                       // photons/frame for a LIVE exposure (cheap — quality comes from INTEGRATING frames, not brute force)
let liveScale=1.0;                                   // resolution of a live exposure — full res (a capable GPU handles it; lower ONLY if a weak GPU can't keep up)
const animateRipples=true;                           // water ripples always animate. Was a UI toggle; the
// still-pool alternative only ever affected the one water showcase, so the default is now the behaviour.
let ptEmitters=[];                                   // glowing-sphere emitters for next-event estimation
let scene=null, traj=null, sceneDiag=1.0;
let bodyPoses=[], simTime=0, tEnd=0, playing=false, rate=1.0, drewOnce=false;
// scene-level dynamics readout (trajectory.py's _energy_fields, one triple PER
// FRAME — not per body): kept as module state alongside bodyPoses rather than
// folded into posesAt()'s return array, since every existing bodyPoses[k] call
// site expects a per-body list. posesAt() still does the pack/interpolate/
// expose job for these — same lerp-when-both/hold-when-one pattern as
// temperature_k — it just writes the result here instead of returning it.
let frameEnergyJ=0, frameMomentum=[0,0,0], frameAngularMomentum=[0,0,0];

// ── R4 build-time assertion (the _assertNEETemplate discipline): on a scene
// with ZERO Voxel leaves, svoEnabled must not change ONE BYTE of either
// shader — the 8 zero-Voxel R0 baselines hold ARITHMETICALLY because of
// exactly this property, so it is enforced loudly at every page load.
(function _assertSVOGate(){
  const mini={csg_leaves:[{shape:{type:"Sphere",center:[0,0,0],radius:1},
                           material:"m",op:"add"}],
              materials:{m:{}}, bodies:[]};
  const was=svoEnabled;
  svoEnabled=false;
  const a1=buildPathShader(mini), a2=buildFragmentShader(mini);
  svoEnabled=true;
  const b1=buildPathShader(mini), b2=buildFragmentShader(mini);
  svoEnabled=was;
  if(a1!==b1 || a2!==b2)
    throw new Error("R4: svo compile gate leaks into a zero-Voxel scene's shader text");
})();

function frameBasis(up){
  const U2=norm(up), H=Math.abs(U2[0])<0.9?[1,0,0]:[0,0,1];
  const A=norm(cross(U2,H)), B=cross(U2,A); return [A,B,U2];
}
function eyePos(){
  const [A,B,Up]=frameBasis(cam.up), ce=Math.cos(cam.el), se=Math.sin(cam.el);
  const planar=add(mul(A,Math.cos(cam.az)*ce), mul(B,Math.sin(cam.az)*ce));
  return add(cam.target, mul(add(planar, mul(Up,se)), cam.radius));
}
// ── locate: jump the camera to a named body, framed to its own size, from
// the sun-facing side (so the lit hemisphere — where the color is — faces
// the camera, not the terminator or the dark side). "Real scale, easy
// navigation" per the Captain (2026-08-03): distances stay real and
// un-distorted; what changes is where you're LOOKING, not how big
// anything pretends to be.
function _bodyIndexByLabel(name){
  const bs=(scene&&scene.bodies)||[];
  for(let i=0;i<bs.length;i++) if(bs[i].label===name) return i;
  return -1;
}
// body radius fallback chain (radius_m || csg-leaf bounds radius || 1.0) —
// extracted from locateBody so the camera controller's clearance scan
// (camera.js, R2) shares the ONE definition.
function _bodyRadius(i){
  return (scene.bodies[i]&&scene.bodies[i].radius_m)
       || (bnd=>bnd?bnd.radius:null)((scene.csg_leaves.find(l=>l.body===i)||{}).bounds)
       || 1.0;
}
function locateBody(name){
  const i=_bodyIndexByLabel(name); if(i<0){ setErr(`locate: no body named "${name}"`); return false; }
  const poses=bodyPoses.length?bodyPoses:posesAt(simTime);
  const tgt=(poses[i]&&poses[i].pos)||[0,0,0];
  const sunI=_bodyIndexByLabel("Sun");
  const sunPos=(sunI>=0 && poses[sunI]) ? poses[sunI].pos : null;
  cam.target=tgt.slice();
  const rM=_bodyRadius(i);
  cam.radius=Math.max(rM*3.2, 1e-6);
  if(sunPos && i!==sunI){
    // aim the eye on the target-to-sun side: solve az/el so eyePos()'s own
    // orbit basis reproduces this direction (frameBasis/eyePos, unchanged)
    const [A,B,Up]=frameBasis(cam.up);
    const dirToSun=norm(sub(sunPos,tgt));
    const se=Math.max(-1,Math.min(1,dot(dirToSun,Up)));
    cam.el=Math.asin(se);
    const ce=Math.cos(cam.el);
    const az=Math.atan2(dot(dirToSun,B), dot(dirToSun,A));
    cam.az=(Math.abs(ce)>1e-6)?az:cam.az;   // near a pole: keep the existing az, el alone frames it
  }
  sceneId++;                     // forces the PT accumulation key to reset (see draw()'s `key`)
  // locate AUTO-ENTERS follow (R2, disclosed): locate IS the one sanctioned
  // camera jump — the controller ADOPTS the framing it just set (no
  // re-solve). Static scenes stay byte-equivalent to today (tick re-writes
  // the same target values); animated scenes strictly better: jump-then-track.
  if(camctl) camctl.followAdopt(name);
  return true;
}
window.locateBody=locateBody;    // console/UI hook
window._cam=cam;                 // debug hooks (module-scoped vars are otherwise invisible to console eval)
window._dbg=()=>({cam:{az:cam.az,el:cam.el,radius:cam.radius,target:cam.target.slice()},
                  bodyPoses:bodyPoses.map(p=>p.pos), eye:eyePos()});
// ── R2 camera controller (camera.js) — orbit/follow/fly modes over the ONE
// cam object above. DOM handlers stay HERE and call controller methods
// (input.js's attach pattern); env closures hand it live viewer state
// lazily. A 404'd camera.js leaves camctl null and every call site below
// degrades to the E3 behavior — belt-and-suspenders, disclosed.
const camctl = window.SGCamera ? new SGCamera(cam, {
  poses: () => bodyPoses.length ? bodyPoses : posesAt(simTime),
  bodies: () => (scene && scene.bodies) || [],
  bodyIndexByLabel: _bodyIndexByLabel,
  bodyRadius: _bodyRadius,
  sceneDiag: () => sceneDiag,
}) : null;
window._camctl = camctl;         // debug surface (window._cam/_dbg precedent)
// ── R3 debug/measurement surface (window._cam/_dbg precedent): live band
// stats (written by draw() on banding scenes) + the deliverable-C CPU-loop
// microbench — a FRESH SGSplats instance, so bench never clobbers the live
// hysteresis state.
window._r3 = {
  stats: {packMs: 0, counts: [0, 0, 0], transitions: 0, transitionsByBody: []},
  bench: (n, reps) => window.SGSplats
    ? new window.SGSplats.SGSplats().bench(n, reps) : null,
};
// lerp-when-both-endpoints-have-it, hold-when-only-one-does — the same
// missing-data policy posesAt uses for temperature_k, applied to the three
// scene-level scalars/vectors (absent on a frame → this trajectory predates
// the energy/momentum fields, or the frame is one of record_object_fall's
// pre-bounce/rest-beat entries that legitimately carries zeros already).
function _setFrameDynamics(f,u,f2){
  const A=f.energy_j, B=f2?f2.energy_j:undefined;
  frameEnergyJ = (A!==undefined&&B!==undefined) ? A+(B-A)*u : (A!==undefined?A:(B!==undefined?B:0));
  const Am=f.momentum, Bm=f2?f2.momentum:undefined;
  frameMomentum = (Am&&Bm) ? [Am[0]+(Bm[0]-Am[0])*u, Am[1]+(Bm[1]-Am[1])*u, Am[2]+(Bm[2]-Am[2])*u]
                : (Am||Bm||[0,0,0]).slice();
  const Al=f.angular_momentum, Bl=f2?f2.angular_momentum:undefined;
  frameAngularMomentum = (Al&&Bl) ? [Al[0]+(Bl[0]-Al[0])*u, Al[1]+(Bl[1]-Al[1])*u, Al[2]+(Bl[2]-Al[2])*u]
                       : (Al||Bl||[0,0,0]).slice();
}
function posesAt(t){     // → [{pos,quat}] for every body, interpolated
  if(!traj||!traj.frames.length){ frameEnergyJ=0; frameMomentum=[0,0,0]; frameAngularMomentum=[0,0,0]; return []; }
  const F=traj.frames, pack=(arr)=>arr.map(b=>({pos:b.pos.slice(),quat:(b.quat||[0,0,0,1]).slice(),
    tempK:(b.temperature_k!==undefined?b.temperature_k:undefined)}));
  if(t<=F[0].t_sim){ _setFrameDynamics(F[0],0); return pack(F[0].bodies); }
  if(t>=F[F.length-1].t_sim){ _setFrameDynamics(F[F.length-1],0); return pack(F[F.length-1].bodies); }
  let lo=0,hi=F.length-1;
  while(hi-lo>1){ const m=(lo+hi)>>1; if(F[m].t_sim<=t) lo=m; else hi=m; }
  const a=F[lo],b=F[hi], u=(t-a.t_sim)/Math.max(1e-9,b.t_sim-a.t_sim);
  _setFrameDynamics(a,u,b);
  return a.bodies.map((ba,k)=>{ const bb=b.bodies[k]||ba, pa=ba.pos, pb=bb.pos;
    const Ta=ba.temperature_k, Tb=bb.temperature_k;
    return {pos:[pa[0]+(pb[0]-pa[0])*u, pa[1]+(pb[1]-pa[1])*u, pa[2]+(pb[2]-pa[2])*u],
            quat:nlerp(ba.quat||[0,0,0,1], bb.quat||[0,0,0,1], u),
            // NON-DERIVED (audit): inter-frame temperature lerp is playback
            // reconstruction of frozen sim states, never the heat equation
            tempK:(Ta!==undefined&&Tb!==undefined)?Ta+(Tb-Ta)*u
                 :(Ta!==undefined?Ta:Tb)};
  });
}

// ── render ────────────────────────────────────────────────────────────
function resize(){
  // R0 capture: EXACT backing-store pixels — immune to window size, DPR and
  // liveScale, so the accumulation key's WxH term is a committed constant.
  if(CAP && CAP.size){ const W=+CAP.size[1], H=+CAP.size[2];
    if(canvas.width!==W||canvas.height!==H){ canvas.width=W; canvas.height=H; } return; }
  const dpr=Math.min(2,window.devicePixelRatio||1);
  // An ANIMATED scene can't converge a still frame — it lives on a fresh shutter
  // exposure EVERY frame, so at full resolution the path tracer stalls. Render it at
  // a reduced backing-store resolution (fewer pixels, full physics per pixel — an
  // honest exposure trade, not a filter) and let the canvas upscale; a STILL scene
  // keeps full resolution because it has all the frames it needs to converge.
  const anim=animateRipples && (scene.csg_leaves||[]).some(l=>l.shape&&l.shape.type==="Water");
  const sc = anim ? liveScale : 1;
  let w=Math.floor((canvas.clientWidth||window.innerWidth)*dpr*sc);
  let h=Math.floor((canvas.clientHeight||(window.innerHeight-60))*dpr*sc);
  w=Math.max(2,w); h=Math.max(2,h);
  if(canvas.width!==w||canvas.height!==h){ canvas.width=w; canvas.height=h; } }
function draw(){
  resize(); const w=canvas.width, h=canvas.height;
  let up=cam.up; const eye=eyePos(); const fwd=norm(sub(cam.target,eye));
  if(Math.abs(dot(fwd,norm(up)))>0.999) up=[0,1,0];          // anti-gimbal
  const right=norm(cross(fwd,up)); const camUp=cross(right,fwd);
  const md=len(sub(eye,cam.target))+sceneDiag*1.5+0.1;       // dynamic — survives zoom
  const tanHalf=Math.tan((scene.camera.fov_deg||40)*Math.PI/360);
  const NB=(scene.bodies||[]).length;
  // ── camera-anchored rebase (R1-lite, 2026-08-03) ──────────────────────
  // Every position handed to the GPU below is expressed relative to `eye`,
  // computed here in JS float64 BEFORE the Float32Array cast. Precision is
  // governed by DISTANCE FROM THE CAMERA, not by how far a coordinate is
  // from an arbitrary world origin — so uEye becomes exactly [0,0,0] (a
  // subtraction of a value from itself is exact) and mapD's own
  // `p-uBodyPos[k]` (viewer.js's per-body transform) stays a small-minus-
  // small operation even when the body is genuinely 1e12 m from the solar
  // system's barycenter. This is what makes a real DE440-scale scene
  // renderable at all: without it, float32 can't represent a body's own
  // few-km surface detail once its coordinate exceeds roughly 1e11 m.
  // Bodies far from the CURRENT camera position still lose precision (this
  // rebases relative to the eye, not per-object) — expected and harmless,
  // since a body that far is visually tiny regardless; the real fix for
  // that regime is the splat far-field (R3), not this.
  let fp=null,fq=null,ft=null;
  if(NB){ fp=new Float32Array(NB*3); fq=new Float32Array(NB*4); ft=new Float32Array(NB);
    for(let k=0;k<NB;k++){ const Pp=bodyPoses[k]||{pos:[0,0,0],quat:[0,0,0,1]};
      fp[k*3]=Pp.pos[0]-eye[0]; fp[k*3+1]=Pp.pos[1]-eye[1]; fp[k*3+2]=Pp.pos[2]-eye[2];
      fq[k*4]=Pp.quat[0]; fq[k*4+1]=Pp.quat[1]; fq[k*4+2]=Pp.quat[2]; fq[k*4+3]=Pp.quat[3];
      ft[k]=(Pp.tempK!==undefined)?Pp.tempK:-1.0; } }         // -1 = no frame temperature
  // ── view-scale-adaptive epsilons (R1-lite) — current cam.radius, not the
  // whole scene's bbox diagonal, so zooming into one planet inside a
  // solar-system-scale scene gets fine steps, and zooming out to see the
  // whole system gets coarse ones instead of an unresolvable fixed size.
  // NOT_PHYSICS numerical tuning (same doctrine as stepper.py's CFL
  // coefficient) — ratios carried over from the previous diag-derived
  // constants; only the driving quantity (radius vs. diagonal) changed.
  const R=Math.max(cam.radius,1e-9);
  const vEPS=Math.max(5e-7,R*0.0015), vSHMIN=Math.max(1e-6,R*0.004),
        vSHMAX=Math.max(1e-5,R*0.06), vSMAXT=Math.max(1e-4,R*1.2),
        vCalcNH=Math.max(2e-7,R*1e-4);
  // ── R3: emitter world centres + live temps + rebased upload, hoisted so
  // the NEE upload and the splat/band pass share ONE computation (same
  // floats, same order — non-banding scenes get byte-identical uploads).
  let emWC=null, emT=null, emEP=null;
  if(ptEmitters.length){
    emWC=new Array(ptEmitters.length); emT=new Float32Array(ptEmitters.length);
    emEP=new Float32Array(ptEmitters.length*3);
    ptEmitters.forEach((e,k)=>{ let c=e.center;
      if(e.body>=0 && bodyPoses[e.body]){ const Po=bodyPoses[e.body];
        const piv=((scene.bodies[e.body]||{}).pivot)||[0,0,0];
        c=add(Po.pos, qrotJS(Po.quat, sub(e.center, piv))); }
      emWC[k]=c;
      emEP[k*3]=c[0]-eye[0]; emEP[k*3+1]=c[1]-eye[1]; emEP[k*3+2]=c[2]-eye[2];   // same eye-anchored rebase as uBodyPos
      const Tf=(e.body>=0 && bodyPoses[e.body] && bodyPoses[e.body].tempK!==undefined)
        ? bodyPoses[e.body].tempK : undefined;
      emT[k]=(Tf!==undefined)?Tf:e.temp; });     // frame T > the leaf's static bake
  }
  // ── R3: per-frame band classification + splat pack (splats.js — pure,
  // no RNG, no wall clock in anything that feeds rendering; state resets
  // memorylessly on EVERY sceneId bump, so R0 capture pins classify from
  // the pinned camera with no history).
  let bandOut=null, bandI32=null;
  if(bandingEnabled && splatsCtl && _r3Params && NB){
    if(_r3SceneId!==sceneId){ splatsCtl.reset(); _r3SceneId=sceneId; }
    const posArr=new Array(NB);
    for(let k=0;k<NB;k++) posArr[k]=(bodyPoses[k]||{pos:[0,0,0]}).pos;
    bandOut=splatsCtl.frame({
      eye, poses:posArr, radius:_r3Params.radius,
      emitterBody:_r3EmitterBodies,
      emitters:ptEmitters.map((e,k)=>({body:e.body, center:emWC[k], radius:e.radius})),
      tanHalf, h});
    bandI32=Int32Array.from(bandOut.bands);
    if(window._r3) window._r3.stats={packMs:bandOut.ms, counts:bandOut.counts.slice(),
      transitions:splatsCtl.transitions,
      transitionsByBody:Array.from(splatsCtl.transitionsByBody||[])};
  }
  // ── R4: pack the per-frame body-table rows (band-0 SVO bodies ONLY — an
  // impostor/splat body costs zero traversal; uBand never enters the SVO
  // path). Same R1-lite float64-subtract-then-f32 rebase as fp above.
  let svoRowCount=0;
  if(svoActiveNow && svoGL){
    const P=svoGL.packed, buf=svoGL.rowBuf;
    // STABLE rows: body k always rides row = its index in P.bodies, so the
    // shader-visible identity mat>>8 (= 1 + row) names the same BODY on every
    // frame. A banded body (R3 splat/impostor territory) keeps its row and
    // gets a POISONED gate radius instead of being compacted out — the old
    // compaction shifted every later row, so the compile-time NEE self-guard
    // (and any mat-keyed identity) pointed at the WRONG body the moment one
    // body crossed a band edge (adversarial R4 refutation finding).
    for(let row=0; row<P.bodies.length; row++){
      const r=P.bodies[row];
      const k=r.body;
      const banded=!!(bandI32 && bandI32[k]!==0);
      const Pp=bodyPoses[k]||{pos:[0,0,0],quat:[0,0,0,1]};
      const o=row*16;
      buf[o]=Pp.pos[0]-eye[0]; buf[o+1]=Pp.pos[1]-eye[1]; buf[o+2]=Pp.pos[2]-eye[2];
      buf[o+3]=banded?-1e30:r.gateRadius;        // poisoned gate: never traversed
      buf[o+4]=Pp.quat[0]; buf[o+5]=Pp.quat[1]; buf[o+6]=Pp.quat[2]; buf[o+7]=Pp.quat[3];
      buf[o+8]=r.lo[0]; buf[o+9]=r.lo[1]; buf[o+10]=r.lo[2]; buf[o+11]=r.rootSpan;
      buf[o+12]=r.rootNode;
      buf[o+13]=r.L + 256*r.matId + 65536*r.L;   // lodCap | matId<<8 | L<<16 (cap==L until R4.4 pins)
      buf[o+14]=(Pp.tempK!==undefined)?Pp.tempK:-1.0; buf[o+15]=r.band0;
    }
    svoRowCount=P.bodies.length;
    if(frameCount<3) _svoBeacon("f"+frameCount+"-rows-"+svoRowCount);
    gl.bindTexture(gl.TEXTURE_2D,svoGL.bodyTex);
    if(svoRowCount)
      gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,4,svoRowCount,gl.RGBA,gl.FLOAT,
                       svoGL.rowBuf.subarray(0,svoRowCount*16));
    gl.bindTexture(gl.TEXTURE_2D,null);
    if(frameCount<3) _svoBeacon("f"+frameCount+"-rows-up");
  }
  const bindScene=(UU,timeVal,frameSeed)=>{
    gl.uniform2f(UU.uRes,w,h);
    gl.uniform3fv(UU.uEye,[0,0,0]); gl.uniform3fv(UU.uFwd,fwd); gl.uniform3fv(UU.uRight,right); gl.uniform3fv(UU.uUp,camUp);
    if(UU.uEyeWorld!=null) gl.uniform3fv(UU.uEyeWorld,eye);
    gl.uniform1f(UU.uTanHalf,tanHalf); gl.uniform1f(UU.uAspect,w/h); gl.uniform1f(UU.uMaxDist,md);
    gl.uniform1f(UU.uTime,timeVal); gl.uniform1f(UU.uTemp,0.0);   // no manual heat override; objects use their own temperature_k
    if(UU.uFrame!=null) gl.uniform1f(UU.uFrame,frameSeed);
    if(UU.uEPS!=null) gl.uniform1f(UU.uEPS,vEPS);
    if(UU.uSHMIN!=null) gl.uniform1f(UU.uSHMIN,vSHMIN);
    if(UU.uSHMAX!=null) gl.uniform1f(UU.uSHMAX,vSHMAX);
    if(UU.uSMAXT!=null) gl.uniform1f(UU.uSMAXT,vSMAXT);
    if(UU.uGateBuffer!=null) gl.uniform1f(UU.uGateBuffer,vEPS*5.0);
    if(UU.uCalcNH!=null) gl.uniform1f(UU.uCalcNH,vCalcNH);
    if(NB){ gl.uniform3fv(UU.uBodyPos,fp); gl.uniform4fv(UU.uBodyQuat,fq);
      if(UU.uBodyTemp!=null) gl.uniform1fv(UU.uBodyTemp,ft); }
    if(UU.uBand!=null && bandI32) gl.uniform1iv(UU.uBand,bandI32);   // R3 band ids (banding scenes only)
    // voxel 3-D textures on units ≥1 (TEXTURE0 is the PT accumulation buffer)
    (scene._voxels||[]).forEach((vx,u)=>{ gl.activeTexture(gl.TEXTURE0+1+u);
      gl.bindTexture(gl.TEXTURE_3D, vx.tex);
      if(UU.vox && UU.vox[vx.vid]!=null) gl.uniform1i(UU.vox[vx.vid], 1+u); });
    // per-cell fields: bind the two keyframes bracketing simTime + the mix
    // factor (units after the voxels; the exposure key already carries simTime,
    // so a scrub restarts accumulation — no stale-glow ghosting)
    const vbase=1+((scene._voxels||[]).length);
    (scene._fields||[]).forEach((f,fi)=>{
      const loc=UU.fld && UU.fld[f.fid]; if(!loc) return;
      const ts=f.times; let lo=0;
      if(simTime>=ts[ts.length-1]) lo=ts.length-1;
      else while(lo+1<ts.length && ts[lo+1]<=simTime) lo++;
      const hi=Math.min(lo+1, ts.length-1);
      const u=(hi===lo)?0.0:Math.min(Math.max((simTime-ts[lo])/Math.max(1e-9,ts[hi]-ts[lo]),0.0),1.0);
      gl.activeTexture(gl.TEXTURE0+vbase+fi*2);   gl.bindTexture(gl.TEXTURE_3D, f.texs[lo]);
      if(loc.a!=null) gl.uniform1i(loc.a, vbase+fi*2);
      gl.activeTexture(gl.TEXTURE0+vbase+fi*2+1); gl.bindTexture(gl.TEXTURE_3D, f.texs[hi]);
      if(loc.b!=null) gl.uniform1i(loc.b, vbase+fi*2+1);
      if(loc.u!=null) gl.uniform1f(loc.u, u);
    });
    // R4: the four SVO textures ride FIXED units 11–14 (legacy voxels/fields
    // top out at unit 10 in every shipped scene; guarded loudly if not)
    if(svoActiveNow && svoGL && UU.uNumSvoBodies!=null){
      if(vbase+((scene._fields||[]).length)*2 > 11)
        console.error("R4: texture-unit collision — voxel/field units reached the SVO block (11-14)");
      gl.activeTexture(gl.TEXTURE0+11); gl.bindTexture(gl.TEXTURE_2D,svoGL.nodeTex);
      if(UU.uSVONodes!=null) gl.uniform1i(UU.uSVONodes,11);
      gl.activeTexture(gl.TEXTURE0+12); gl.bindTexture(gl.TEXTURE_3D,svoGL.atlasTex);
      if(UU.uSdfAtlas!=null) gl.uniform1i(UU.uSdfAtlas,12);
      gl.activeTexture(gl.TEXTURE0+13); gl.bindTexture(gl.TEXTURE_2D,svoGL.lutTex);
      if(UU.uMatLUT!=null) gl.uniform1i(UU.uMatLUT,13);
      gl.activeTexture(gl.TEXTURE0+14); gl.bindTexture(gl.TEXTURE_2D,svoGL.bodyTex);
      if(UU.uSVOBody!=null) gl.uniform1i(UU.uSVOBody,14);
      gl.uniform1i(UU.uNumSvoBodies, svoRowCount);
      if(UU.uAtlasDim!=null) gl.uniform1f(UU.uAtlasDim, svoGL.packed.atlasDim);
      if(UU.uAtlasBpa!=null) gl.uniform1i(UU.uAtlasBpa, svoGL.packed.bpa);
    } };
  if(ptMode && extF && !dispProg) initDisplay();
  // R3 belt-and-braces: bands can only change when eye/simTime change, which
  // already change the key — the |b: term defends future refactors of that
  // coupling. Non-banding scenes keep the key string byte-identical.
  const key=`${eye.map(x=>x.toFixed(3))}|${cam.target.map(x=>x.toFixed(3))}|${simTime.toFixed(3)}|${w}x${h}|${sceneId}`
    +(bandOut?`|b:${Array.from(bandOut.bands).join("")}`:"");
  const moving = key!==ptKey; ptKey=key;                     // is the view changing this frame?
  if(moving) accN=0;                                         // view changed → restart accumulation
  // ── Radiance has ONE renderer: the path tracer. Appearance is light transport,
  // always — never a headlight/ambient "fast" trick. Quality comes not from brute
  // force but from a real camera EXPOSURE that INTEGRATES photons over a shutter:
  //   • STILL + static  → integrate UNBOUNDED → a long exposure → pristine.
  //   • ANIMATED, still camera (water) → integrate over a BOUNDED shutter (NMAX
  //     frames) while the ripples ADVANCE: photons pile up (clean) and the motion
  //     becomes physically-real motion blur — capped so it stays live and can't
  //     smear away. This is the camera model, NOT a denoiser.
  //   • CAMERA MOVING → the view changes too fast to integrate; one cheap exposure
  //     (honest shot noise while you drag; it clears the instant you stop).
  // Each frame is cheap (≈2 spp); the picture cleans up by integrating, not by
  // firing more rays per frame. (The fast shader survives ONLY as a no-float shim.)
  const animated = animateRipples && (scene.csg_leaves||[]).some(l=>l.shape && l.shape.type==="Water");
  const NMAX = 24;                                          // shutter length (frames) for a live exposure — bounds the motion blur
  const ptOn = ptMode && extF && dispProg && progPT;        // the real renderer (float targets carry the exposure buffer)
  if(ptOn){                                                  // ── path tracing — the ONLY renderer ──
    ensureAccum(w,h);
    let reset, prevScale, spp, frames, advanceTime;
    if(moving){                                              // view changing fast → can't integrate; one cheap exposure
      reset=true; prevScale=0.0; spp=Math.min(liveSPP,4); advanceTime=true; accN=0; frames=1;
    } else if(animated){                                    // still camera, live ripples → integrate over a bounded shutter
      advanceTime=true;                                     // ripples ADVANCE during the exposure → real motion blur
      if(accN<NMAX){ reset=(accN===0); prevScale=1.0; spp=liveSPP; accN++; frames=accN; }
      else { reset=false; prevScale=(NMAX-1)/NMAX; spp=liveSPP; frames=NMAX; }   // steady-state hold — no pulse, bounded blur
    } else {                                                // still + static → converge to pristine (freeze the ripples)
      reset=(accN===0); prevScale=1.0; spp=3; advanceTime=reset; accN++; frames=accN;
    }
    // R0 capture: virtual shutter clock + accumulation-index seed — every RNG
    // draw becomes a pure function of committed inputs (accN), not of page-load
    // timing. Water keeps REAL motion blur: the clock advances per accumulated
    // frame (accN/60 ≈ one 60 Hz frame per exposure step) instead of freezing
    // a wave phase. Outside capture mode both lines are byte-equivalent to E4.
    if(advanceTime) ptTime = CAP ? (accN/60) : (performance.now()-T0)*0.001;
    const src=pingA?accumA:accumB, dst=pingA?accumB:accumA;
    gl.bindFramebuffer(gl.FRAMEBUFFER,dst.fbo); gl.viewport(0,0,w,h);
    gl.useProgram(progPT); bindScene(Upt, ptTime, CAP ? accN : frameCount);   // THE seed pin
    gl.uniform1i(Upt.uSPP, spp); gl.uniform1f(Upt.uPrevScale, prevScale);
    if(ptEmitters.length && Upt.uEmitterPos){      // emitter world centres + live temps (hoisted above — the R3 splat pass shares them)
      gl.uniform3fv(Upt.uEmitterPos, emEP);
      if(Upt.uEmitterTemp!=null) gl.uniform1fv(Upt.uEmitterTemp, emT); }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,src.tex);
    gl.uniform1i(Upt.uPrev,0); gl.uniform1f(Upt.uReset, reset?1.0:0.0);
    if(svoActiveNow&&frameCount<3) _svoBeacon("f"+frameCount+"-pt-draw-start");
    gl.drawArrays(gl.TRIANGLES,0,3);
    if(svoActiveNow&&frameCount<3) _svoBeacon("f"+frameCount+"-pt-draw-queued");
    // ── R3 splat pass: sub-3px far field as GL_POINTS into the RGBA16F
    // target (cleared each frame — splat radiance is closed-form per frame,
    // never accumulated), additive blend, composited by the display pass.
    if(bandOut && splatProg){
      ensureSplatTarget(w,h);
      gl.bindFramebuffer(gl.FRAMEBUFFER,splatTarget.fbo); gl.viewport(0,0,w,h);
      gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT); gl.clearColor(0,0,0,1);
      gl.useProgram(splatProg);
      gl.uniform3fv(splatU.uFwd,fwd); gl.uniform3fv(splatU.uRight,right); gl.uniform3fv(splatU.uUp,camUp);
      gl.uniform1f(splatU.uTanHalf,tanHalf); gl.uniform1f(splatU.uAspect,w/h);
      gl.uniform1f(splatU.uPxPerRad,h/(2*tanHalf));
      gl.uniform1f(splatU.uTemp,0.0);                       // same pin as bindScene's uTemp
      gl.uniform1f(splatU.uPointCap,Math.min(3,_ptRange[1]));
      if(emEP && splatU.uEmPos!=null){ gl.uniform3fv(splatU.uEmPos,emEP);
        if(splatU.uEmTemp!=null) gl.uniform1fv(splatU.uEmTemp,emT); }
      if(splatU.uSplTemp!=null && ft) gl.uniform1fv(splatU.uSplTemp,ft);   // frame body T override (-1 = none)
      gl.bindBuffer(gl.ARRAY_BUFFER,splatBufDyn);
      gl.bufferSubData(gl.ARRAY_BUFFER,0,bandOut.dyn);      // 12 B/body/frame — the envelope's traffic row
      gl.bindBuffer(gl.ARRAY_BUFFER,splatBufVis);
      gl.bufferSubData(gl.ARRAY_BUFFER,0,bandOut.vis);      // +1 B/body/frame visibility flags
      gl.bindBuffer(gl.ARRAY_BUFFER,null);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);     // 16F additive blending — core with EXT_color_buffer_float
      gl.bindVertexArray(splatVAO);
      gl.drawArrays(gl.POINTS,0,NB);
      gl.bindVertexArray(VAO0);
      gl.disable(gl.BLEND);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,w,h);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,w,h);
    }
    gl.useProgram(dispProg);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,dst.tex);
    // unit 1: the splat far-field target (R3) — or the 1×1 black texture,
    // an exact +0.0 in the composite, for every non-banding scene.
    gl.activeTexture(gl.TEXTURE0+1);
    gl.bindTexture(gl.TEXTURE_2D, (bandingEnabled&&splatTarget)?splatTarget.tex:splatBlackTex);
    if(dispU.uSplat!=null) gl.uniform1i(dispU.uSplat,1);
    gl.uniform1i(dispU.uAccum,0); gl.uniform1f(dispU.uInvN,1.0/frames);
    gl.drawArrays(gl.TRIANGLES,0,3); pingA=!pingA;
    if(svoActiveNow&&frameCount<3) _svoBeacon("f"+frameCount+"-display-done");
    // R2: hudSuffix() is "" in orbit mode, so today's strings stay
    // byte-identical until a new camera mode is entered (a dbg tweak,
    // not an E4 HUD panel).
    const camSfx=camctl?camctl.hudSuffix():"";
    // R3 band-status dbg (banding scenes only — every other scene's string
    // stays byte-identical): raymarch/impostor/splat counts + pack-loop ms.
    const bandSfx=bandOut?` · bands ${bandOut.counts[0]}/${bandOut.counts[1]}/${bandOut.counts[2]} · pack ${bandOut.ms.toFixed(2)}ms`:"";
    const dd=$("dbg"); if(dd) dd.textContent =
        moving   ? `path-traced · live · ${spp} spp/frame · orbiting (clears when you stop) · ${w}x${h}${camSfx}${bandSfx}`
      : animated ? `path-traced · ${Math.min(accN,NMAX)}-frame exposure · ${frames*spp} photons/px · ripples live · ${w}x${h}${camSfx}${bandSfx}`
      :            `path-traced · ${frames*spp} samples/px · converging… · ${w}x${h}${camSfx}${bandSfx}`;
  } else {                                                   // ── compatibility shim: GPU lacks float render targets ──
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,w,h); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog); bindScene(U,(performance.now()-T0)*0.001, frameCount);
    gl.drawArrays(gl.TRIANGLES,0,3);
    const dd=$("dbg"); if(dd) dd.textContent=`compatibility view — this GPU can't path-trace (no float render targets) · ${w}x${h}${camctl?camctl.hudSuffix():""}`;
  }
  drewOnce=true;
}

// ── self-check + load ─────────────────────────────────────────────────
function fieldSelfCheck(){          // → [pass, maxΔK] | [null, 0] when no samples shipped
  const fs=scene.field_samples;
  if(!fs||!fs.length) return [null, 0];
  let md=0;
  for(const smp of fs) md=Math.max(md, Math.abs(jsFieldSampleAt(scene,smp)-smp.T));
  return [md<1e-3, md];             // decoded-payload ground truth → double-precision match
}
function selfCheck(){
  const s=scene.sdf_samples;
  let geo="geometry self-check: <span class='lbl'>(no samples)</span>";
  if(s&&s.length){
    let md=0; for(const smp of s) md=Math.max(md,Math.abs(jsEvalSDF(scene,smp.p,null)-smp.d));
    const ok=md<1e-6;
    geo=`geometry self-check: <span class="${ok?'ok':'bad'}">${ok?'PASS':'FAIL'}</span> `
       +`<span class="lbl">(max Δ ${md.toExponential(1)} vs Python)</span>`;
  }
  const [fok,fmd]=fieldSelfCheck();
  if(fok!==null)
    geo+=` &nbsp;·&nbsp; field: <span class="${fok?'ok':'bad'}">${fok?'PASS':'FAIL'}</span> `
        +`<span class="lbl">(max Δ ${fmd.toExponential(1)} K)</span>`;
  setOk(geo);
}
let loadedObj=null;                                  // the pristine JSON Radiance was handed (editor + reset)

// Apply a parsed scene/trajectory object. Compiles FIRST, so a broken edit can
// never corrupt the live render; commits the globals only after a clean compile.
// `preserve` keeps the current camera + playhead — the live editor wants that;
// a fresh scene load resets them.
function applyObj(obj, title, preserve){
  const newScene = obj.kind==="trajectory" ? obj.scene : obj;
  const newTraj  = obj.kind==="trajectory" ? obj.trajectory : (obj.trajectory||null);
  if(!newScene||!newScene.csg_leaves||!newScene.csg_leaves.length) throw new Error("scene has no csg_leaves");
  // A genuinely NEW scene (not a live re-render of the one already showing)
  // starts with a clean opacity slate — a material label from the PREVIOUS
  // scene shouldn't silently haunt an unrelated one that happens to reuse it.
  // Editing the current scene's JSON (preserve=true) keeps overrides, same as
  // it keeps the camera/playhead.
  if(!preserve) materialOpacity={};
  // Per-body MAX temperature over the whole trajectory, computed BEFORE the
  // compile: NEE emitter registration is compile-time, so a body that crosses
  // the 700 K Draper point at any frame must be an emitter for the program's
  // lifetime (its magnitude then rides the uEmitterTemp uniform per frame).
  newScene._bodyTempMax = null;
  if(newTraj && newTraj.frames){
    const mx=[];
    for(const f of newTraj.frames) (f.bodies||[]).forEach((b,k)=>{
      if(b.temperature_k!==undefined) mx[k]=Math.max(mx[k]??-1, b.temperature_k); });
    if(mx.length) newScene._bodyTempMax=mx;
  }
  buildProgram(newScene);                            // throws on a bad shader → live render untouched
  scene=newScene; traj=newTraj;                      // commit only after the compile succeeds
  if(!preserve){ const cm=scene.camera||{};
    cam.target=(cm.target||[0,0,0]).slice(); cam.up=(cm.up||[0,0,1]).slice();
    cam.radius=cm.orbit_radius||0.3;                 // a scene may suggest its opening angle (flat grids want face-on)
    cam.az=(cm.az0!==undefined?cm.az0:0.8); cam.el=(cm.el0!==undefined?cm.el0:0.45);
    if(camctl) camctl.resetToOrbit(); }              // fresh scene = fresh nav (R2; preserve=true keeps mode+follow)
  const bb=scene.bbox||[[-1,1],[-1,1],[-1,1]];
  sceneDiag=len(sub([bb[0][1],bb[1][1],bb[2][1]],[bb[0][0],bb[1][0],bb[2][0]]));
  sceneId++;                                         // new scene → PT accumulation resets
  selfCheck();
  $("title").textContent=scene.name||title||""; $("src").textContent=scene.source||"";
  // force provenance: show every declared plug (artificial drives/supports)
  // right in the HUD — a scene with none shows nothing, never a fake "clean"
  const plugsEl=$("plugs");
  if(plugsEl){ const ps=scene.plugs||[];
    plugsEl.textContent=ps.map(p=>{
      const tag=(p.kind==="support")?"⚓ plugged support":"⚡ plugged drive";
      const vs=Object.entries(p.variables||{}).map(([k,v])=>`${k}=${v}`).join(", ");
      return `${tag}: ${p.description}${vs?` (${vs})`:""}`; }).join("  ·  "); }
  showVariants(scene);
  showApply(obj);
  showLocate(scene);
  buildMaterialPanel(scene);
  const hasTraj=!!(traj&&traj.frames&&traj.frames.length>1);
  tEnd=hasTraj?traj.t_end_s:0;
  if(!preserve){ simTime=0; playing=false; $("play").textContent="▶ play";
    if(hasTraj){ rate=traj.suggested_rate||1; $("rate").value=Math.log10(rate); updRate(); } }
  else if(simTime>tEnd){ simTime=tEnd>0?tEnd:0; }
  $("playgrp").style.opacity=$("rategrp").style.opacity=hasTraj?"1":"0.35";
  bodyPoses=posesAt(simTime); drewOnce=false;
  // R0: once armed, the pins survive ANY later load — chat.js's deep-link
  // auto-load re-loads the same slug on capture tabs (suppressed there now,
  // but belt-and-suspenders: any future re-loader would silently reset the
  // camera/playhead the same way, and a capture must never race a loader).
  if(CAP && capArmed) applyCapturePins();
  return true;
}
// ── variants: a scene's declared parameter sweep (frozen recorded runs) ──
// Adjustable-by-default for cited artificial/explicit variables: the slider
// swaps between SIBLING BUNDLES (each a real recorded sim) — playback
// reconstruction, never live physics in the renderer.
let _variants=null;
function showVariants(sc){
  const grp=$("vargrp"), sep=$("varsep");
  if(!grp) return;
  const vs=sc.variants||[];
  _variants=vs.length>1?vs:null;
  grp.style.display=sep.style.display=_variants?"":"none";
  if(!_variants) return;
  const slider=$("variant");
  slider.min=0; slider.max=vs.length-1; slider.step=1;
  // the scene declares which variant IT IS (variant_current), no guessing
  const cur=Math.max(0, vs.findIndex(v=>v.slug===sc.variant_current));
  slider.value=String(cur);
  $("varlbl").textContent="variant";
  $("varval").textContent=vs[cur].label;
}
(function(){ const s=$("variant"); if(!s) return;
  s.addEventListener("input",()=>{ if(!_variants) return;
    const v=_variants[parseInt(s.value,10)||0]; if(!v) return;
    $("varval").textContent=v.label;
    load(`data/${v.slug}.json`, v.label).catch(e=>setErr("variant: "+e)); });
})();

// ── apply: an artificial force from the CLOSED taxonomy (Golden Rule 7) ──
// The drawer is data-driven from _APPLY_KINDS — mirroring the server's
// dynamics.mechanisms.applied.KINDS. Adding a kind here is UI only; the
// doctrine (and its solver) lives server-side. Kinds whose solver hasn't
// landed render disabled, honestly labeled — never silently absent.
// Every run POSTs /apply: the REAL solver re-simulates this scene's matter
// server-side and the viewer loads the recording. No viewer physics.
const _APPLY_KINDS=[
  ["impulse","impulse — push / shove / knock",true],
  ["force","force — sustained press / drag (live only)",false],
  ["drive","drive — a motor at a joint (solver pending)",false],
  ["hold","hold — an artificial pin (live only)",false],
  ["source","source — heat into a body (solver pending)",false],
  ["tilt","tilt — re-orient gravity (live only)",false]];
// per-kind meaning of the one magnitude input (label, default, unit key)
const _APPLY_MAG={
  impulse:["push speed (m/s)",2.2,"push_speed_m_s"],
  force:["force (N) — applied for 1 s",5.0,"force_n"],
  tilt:["tilt angle (deg) — 2 s ramp, whole scene",10.0,"angle_deg"],
  hold:["hold: 1 = pin, 0 = release",1.0,"on"]};
let _sim=null;                       // the loaded bundle's sim provenance
// ── god-hand live mode: the /live websocket streams the REAL solver ──
let liveMode=false, liveWS=null, _liveFrame=null, _liveKinds=null;
// debug/automation handle (script-scoped lets are invisible to console eval)
window._live={get mode(){return liveMode;}, get frame(){return _liveFrame;},
              get ws(){return liveWS&&liveWS.readyState;}, nframes:0};
function showApply(obj){
  const grp=$("applygrp"); if(!grp) return;
  _sim=(obj&&obj.sim&&obj.sim.builder==="push_off"&&Array.isArray(obj.sim.items))?obj.sim:null;
  grp.style.display=_sim?"":"none";
  if(!_sim){ $("applypanel").classList.remove("open"); $("b-apply").classList.remove("on"); return; }
  const kindSel=$("applykind");
  if(!kindSel.options.length)
    for(const [v,label,on] of _APPLY_KINDS){
      const o=document.createElement("option");
      o.value=v; o.textContent=label; o.disabled=!on; kindSel.appendChild(o); }
  kindSel.value="impulse";
  const tgt=$("applytarget"); tgt.innerHTML="";
  _sim.items.forEach((name,i)=>{
    const o=document.createElement("option");
    o.value=name; o.textContent=i===0?`${name} (the base — not pushable)`:name;
    o.disabled=(i===0); tgt.appendChild(o); });
  tgt.value=_sim.pushed&&_sim.items.indexOf(_sim.pushed)>0?_sim.pushed:_sim.items[1];
  $("applymsg").textContent="";
}
// ── locate: populate the body dropdown (only shown when the scene names
// enough bodies for it to be worth having — a 2-body doughnut/plate scene
// doesn't need this, a 30-body ephemeris does).
function showLocate(sc){
  const grp=$("locategrp"), sel=$("locate"); if(!grp||!sel) return;
  const bs=sc.bodies||[];
  grp.style.display=(bs.length>3)?"":"none";
  sel.innerHTML='<option value="">— body —</option>';
  bs.forEach(b=>{ const o=document.createElement("option");
    o.value=b.label; o.textContent=b.label; sel.appendChild(o); });
}
(function(){ const sel=$("locate"); if(!sel) return;
  sel.addEventListener("change",()=>{ if(sel.value) locateBody(sel.value); });
})();
$("b-apply").addEventListener("click",()=>{
  const open=$("applypanel").classList.toggle("open"); $("b-apply").classList.toggle("on",open);
  if(open){ for(const [p,b] of [["editor","b-edit"],["library","b-lib"],["matpanel","b-mat"]]){
    $(p).classList.remove("open"); $(b).classList.remove("on"); } } });
$("applykind").addEventListener("change",()=>{
  const m=_APPLY_MAG[$("applykind").value];
  if(m){ $("applyspeedrow").querySelector("label").textContent=m[0];
    $("applyspeed").value=String(m[1]); }
});
function _applyForce(){
  const kind=$("applykind").value, v=parseFloat($("applyspeed").value);
  const m=_APPLY_MAG[kind]||_APPLY_MAG.impulse;
  const vars={}; vars[m[2]]=(m[2]==="on")?(v>=0.5):v;
  return {kind, target:$("applytarget").value, variables:vars};
}
function _setLive(on){
  liveMode=on; _liveFrame=null;
  $("applylive").textContent=on?"⏹ stop live":"🔴 go live";
  $("applylive").classList.toggle("on",on);
  // live kinds come from the server's ready message; recorded mode is
  // impulse-only (the /apply endpoint re-records)
  for(const o of $("applykind").options)
    o.disabled = on ? !( _liveKinds||[] ).includes(o.value)
                    : (o.value!=="impulse");
  if(!on&&liveWS){ try{ liveWS.close(); }catch(_e){} liveWS=null;
    if(tEnd>0){ bodyPoses=posesAt(simTime); } }
}
$("applylive").addEventListener("click",()=>{
  if(liveMode){ _setLive(false); return; }
  if(worldMode) _setWorld(false);   // mutual exclusion (E4) — one streamer
  if(!_sim) return;
  const msg=$("applymsg");
  msg.textContent="connecting…";
  const ws=new WebSocket(`ws://${location.host}/live`);
  liveWS=ws;
  ws.onopen=()=>ws.send(JSON.stringify({type:"start", sim:_sim}));
  ws.onmessage=(ev)=>{
    const m=JSON.parse(ev.data);
    if(m.type==="building"){ msg.textContent=m.text; }
    else if(m.type==="ready"){ _liveKinds=m.live_kinds; _setLive(true);
      msg.textContent="LIVE — the real solver is stepping server-side; "
        +"forces apply mid-run"; playing=false; }
    else if(m.type==="frame"){ _liveFrame=m; window._live.nframes++;
      bodyPoses=m.bodies.map(b=>({pos:b.pos, quat:b.quat})); }
    else if(m.type==="refusal"){ msg.textContent=m.text; }
    else if(m.type==="applied"){ msg.textContent=
      `applied ${m.kind} on ${m.target||"the scene"}`; }
  };
  ws.onclose=()=>{ if(liveMode) _setLive(false); msg.textContent="live ended"; };
  ws.onerror=()=>{ msg.textContent="live: connection failed"; };
});
$("applyrun").addEventListener("click",async ()=>{
  if(!_sim) return;
  // E4 world mode: the drawer routes through /world. Two channels, both
  // live: mass-needing impulse goes as a shove INTENT (server-side
  // reduction — the client cannot honestly build J = m·v), everything
  // else as a RAW apply with the client's own values; kinds with no
  // solver go through raw apply and come back as an HONEST server
  // refusal into the tickets panel.
  if(worldMode && worldNet && worldNet.live){
    const kind=$("applykind").value, v=parseFloat($("applyspeed").value),
          tgt=$("applytarget").value;
    if(kind==="impulse")
      worldNet.intent("shove", {speed_m_s:Math.abs(v),
                                direction:v<0?"-x":"+x"}, tgt);
    else if(kind==="force")
      worldNet.apply(tgt, {kind:"force", target:tgt,
                           vector:[v,0,0], duration:1.0});
    else if(kind==="tilt")
      worldNet.apply(null, {kind:"tilt", target:"world",
                            variables:{angle_deg:v}, duration:2.0});
    else if(kind==="hold")
      worldNet.apply(tgt, {kind:"hold", target:tgt,
                           variables:{on:v>=0.5}});
    else
      worldNet.apply(tgt, {kind, target:tgt, variables:{}});
    return;
  }
  const msg=$("applymsg");
  const force=_applyForce();
  if(liveMode&&liveWS&&liveWS.readyState===1){
    liveWS.send(JSON.stringify({type:"apply", force}));   // mid-run, no reload
    return;
  }
  msg.textContent="running the real solver…";
  try{
    const r=await fetch("/apply",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({sim:_sim, force})});
    const env=await r.json();
    if(env.saved&&env.saved.slug){
      msg.textContent="";
      $("applypanel").classList.remove("open"); $("b-apply").classList.remove("on");
      await load(`data/${env.saved.slug}.json`, env.saved.title);
    } else msg.textContent=env.text||env.error||"no result";
  }catch(e){ msg.textContent="apply: "+e; }
});

// ── E4 world mode: the persistent /world protocol (net.js) ────────────────
// ALONGSIDE the old /live (which stays byte-untouched and working — it
// retires only after this replacement proves parity; proving parity is
// E4, retiring is not). Mutual exclusion BOTH directions with liveMode:
// poses must never double-write. Honest failure on the pure-static
// server: net.js closes pre-ready and the reason names the cockpit.
let worldMode=false, worldNet=null, _worldFrame=null;
const _worldEvents=[];                       // ring, cap 32
window._world={get mode(){return worldMode;}, get frame(){return _worldFrame;},
               nframes:0, events:_worldEvents};
function _defaultWorldName(){
  // deterministic from the scene's items — reload REJOINS the persistent
  // world by default (mismatch refusal can't fire on the default path);
  // the #worldname input is the join/fresh override surface.
  if(!_sim||!Array.isArray(_sim.items)) return "w-world";
  return "w-"+_sim.items.map(s=>String(s).toLowerCase()
    .replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"")).join("-");
}
function _worldBag(){ return window._hudState=(window._hudState||{}); }
function _worldActions(){
  return {
    "world.pausetoggle": ()=>{ if(!worldNet) return;
      (_worldFrame&&_worldFrame.paused)?worldNet.resume():worldNet.pause(); },
    "world.pace": (v)=>{ if(worldNet) worldNet.pace(v); },
    "world.save": ()=>_worldSave(),
    "audio.mute": ()=>{ if(window.SGAudio) SGAudio.setMuted(!SGAudio.state().muted); },
    "audio.volume": (v)=>{ if(window.SGAudio) SGAudio.setVolume(v); },
  };
}
function _worldSave(){
  if(!worldNet) return;
  worldNet.save().then((data)=>{
    const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
    const a=document.createElement("a");
    a.href=URL.createObjectURL(blob);
    a.download=`${worldNet.world}-step${data.step_count}.save.json`;
    a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href),5000);
    $("applymsg").textContent=`saved ${a.download} (the log IS the state)`;
  }).catch((e)=>{ $("applymsg").textContent="save: "+e.message; });
}
function _setWorld(on){
  if(!on){
    worldMode=false; _worldFrame=null;
    const net=worldNet; worldNet=null;
    if(net){ try{ net.detach(); }catch(_e){} }     // the world PERSISTS
    const btn=$("worldbtn");
    if(btn){ btn.textContent="🌐 world"; btn.classList.remove("on"); }
    const ctl=$("worldctl"); if(ctl) ctl.style.display="none";
    if(window.SGHud&&SGHud._mounted) SGHud.unmount();
    const bag=window._hudState;
    if(bag&&bag.world&&bag.world.mode==="live") bag.world.mode="off";
    // the _setLive restore idiom: recorded playback poses come back
    if(tEnd>0) bodyPoses=posesAt(simTime);
    // drawer kind options back to recorded-mode defaults
    for(const o of $("applykind").options){
      if(o.dataset.baselabel) o.textContent=o.dataset.baselabel;
      o.disabled=(o.value!=="impulse");
    }
    return;
  }
  // entering: connect (worldMode flips true only on the server's ready)
  if(liveMode) _setLive(false);                    // mutual exclusion
  if(!_sim||!window.SGWorldNet){
    $("applymsg").textContent="world mode needs a push-route scene (sim provenance)";
    return;
  }
  const wn=$("worldname");
  if(wn&&!wn.value) wn.value=_defaultWorldName();
  const name=(wn&&wn.value.trim())||_defaultWorldName();
  $("applymsg").textContent="connecting to /world…";
  const bag=_worldBag();
  bag.world={mode:"connecting", name, err:null};
  bag.tickets=bag.tickets||[]; bag.events=bag.events||[];
  const net=new SGWorldNet({
    world:name, items:_sim.items, pushedIndex:_sim.pushed_index,
    onReady:(ready)=>{
      worldMode=true; playing=false; $("play").textContent="▶ play";
      bag.world={mode:"live", name:ready.world, err:null};
      if(ready.audio) bag.audio_server=ready.audio;   // server disclosure
      // target dropdown from the SERVER's labels (index 0 = the base)
      const tgt=$("applytarget"); tgt.innerHTML="";
      ready.labels.forEach((label,i)=>{
        const o=document.createElement("option");
        o.value=label; o.textContent=i===0?`${label} (the base — not pushable)`:label;
        o.disabled=(i===0); tgt.appendChild(o); });
      tgt.value=ready.labels[1]||ready.labels[0];
      // DESIGN CALL (disclosed): in world mode ALL kinds are enabled —
      // the server's honest refusal into the tickets panel IS the honesty
      // surface (this is what lets a source-kind show a real refusal);
      // recorded and /live modes keep the disabled-option idiom untouched.
      for(const o of $("applykind").options){
        if(!o.dataset.baselabel) o.dataset.baselabel=o.textContent;
        const isLive=(ready.live_kinds||[]).includes(o.value);
        o.disabled=false;
        o.textContent=isLive?o.dataset.baselabel
                            :o.dataset.baselabel+" — server will refuse (no solver)";
      }
      const btn=$("worldbtn");
      if(btn){ btn.textContent="⏹ leave world"; btn.classList.add("on"); }
      const ctl=$("worldctl"); if(ctl) ctl.style.display="";
      $("applymsg").textContent=
        `WORLD "${ready.world}" — persistent server-side world (step ${ready.step}); `
        +`keyboard verbs and the drawer both apply live`;
      if(window.SGHud) SGHud.mount($("hudroot"), SGHud.PANELS,
                                   {actions:_worldActions()});
      if(window.SGAudio&&SGAudio.musicState)
        SGAudio.musicState({event:"attach", world:ready.world});
      if(window._hud) window._hud.touch();
    },
    onFrame:(m)=>{
      _worldFrame=m; window._world.nframes++;
      // the same pathway liveMode uses: streamed poses ARE the poses
      bodyPoses=m.bodies.map(b=>({pos:b.pos, quat:b.quat}));
    },
    onEvent:(ev)=>{
      const v=window.SGAudio?SGAudio.handleEvent(ev, eyePos(), scene)
                            :{play:false, reason:"audio.js not loaded"};
      const entry=Object.assign({}, ev,
        {heard:!!v.play, silenced_reason:v.play?null:v.reason});
      _worldEvents.push(entry);
      if(_worldEvents.length>32) _worldEvents.shift();
      bag.events.push(entry);
      if(bag.events.length>32) bag.events.shift();
      if(window._hud) window._hud.touch();
    },
    onTicket:(t)=>{
      bag.tickets.push(t);
      if(bag.tickets.length>50) bag.tickets.shift();
      const msg=$("applymsg");
      if(msg) msg.textContent=(t.ok?"✓ ":"✗ ")+(t.text||"");
      if(window._hud) window._hud.touch();
    },
    onProto:(_m)=>{ if(window._hud) window._hud.touch(); },
    onClose:(info)=>{
      if(bag.world) { bag.world.mode=info.everReady?"off":"failed";
                      bag.world.err=info.reason; }
      $("applymsg").textContent=info.everReady?"world detached — the world "
        +"persists server-side; 🌐 to rejoin":info.reason;
      if(worldMode||worldNet) _setWorld(false);
    },
  });
  worldNet=net; window._worldnet=net;
  net.connect();
}
(function(){
  const btn=$("worldbtn"); if(!btn) return;
  btn.addEventListener("click",()=>{
    if(worldMode||worldNet){ _setWorld(false); return; }
    _setWorld(true);
  });
  // prefill the deterministic default name whenever the drawer opens
  $("b-apply").addEventListener("click",()=>{
    const wn=$("worldname");
    if(wn&&!wn.value&&_sim) wn.value=_defaultWorldName();
  });
  const wp=$("worldpause");
  if(wp) wp.addEventListener("click",()=>{
    if(!worldNet) return;
    (_worldFrame&&_worldFrame.paused)?worldNet.resume():worldNet.pause();
  });
  const ps=$("worldpace");
  if(ps) ps.addEventListener("input",()=>{
    // log slider 0.1–10 sends the BARE pace op; the display is updated
    // from FRAMES (server echo authoritative, never slider position)
    if(worldNet) worldNet.pace(Math.pow(10,parseFloat(ps.value)));
  });
  const sv=$("worldsave");
  if(sv) sv.addEventListener("click",_worldSave);
})();

// ── render-only material opacity: named-material list + click-to-pick ─────
// The scene's baked opacity (materials[label].opacity, 1.0 if never set) with
// the live client-side override applied on top — the SAME precedence
// buildPathShader's per-leaf alphafn bakes into the shader.
function effectiveOpacity(label){
  const m=(scene&&scene.materials&&scene.materials[label])||{};
  const base=(m.opacity!==undefined)?m.opacity:1.0;
  const ov=materialOpacity[label];
  return (ov!==undefined)?ov:base;
}
// Rebuild the "◐ opacity" drawer from the CURRENT scene's material list — one
// row per label as it appears in spec["materials"], so a Captain can dial any
// part by NAME without ever clicking geometry (build_demo/deckard-authored
// labels like "engine_block" show up here verbatim).
function buildMaterialPanel(sc){
  const list=$("matlist"); if(!list) return;
  list.innerHTML="";
  const labels=Object.keys(sc.materials||{});
  if(!labels.length){ list.innerHTML=`<div class="lbl" style="padding:8px">no materials</div>`; return; }
  for(const label of labels){
    const row=document.createElement("div"); row.className="matrow"; row.dataset.label=label;
    const head=document.createElement("div"); head.className="t"; head.textContent=label;
    const ctl=document.createElement("div"); ctl.className="matctl";
    const slider=document.createElement("input");
    slider.type="range"; slider.min="0"; slider.max="100"; slider.step="1";
    slider.value=String(Math.round(effectiveOpacity(label)*100));
    const val=document.createElement("span"); val.className="lbl matval";
    val.textContent=slider.value+"%";
    slider.addEventListener("input",()=>{
      val.textContent=slider.value+"%";
      window.Radiance.setMaterialOpacity(label, parseInt(slider.value,10)/100);
    });
    ctl.append(slider,val);
    row.append(head,ctl);
    list.appendChild(row);
  }
}
// Reflect the CURRENT effective opacity of `label` into its drawer row (if the
// drawer has been built) — called by window.Radiance.setMaterialOpacity so the
// slider stays truthful no matter which path changed the value (drawer drag,
// click-to-pick toggle, or a chat/API call).
function syncMatRow(label){
  const list=$("matlist"); if(!list) return;
  const row=list.querySelector(`.matrow[data-label="${CSS.escape(label)}"]`); if(!row) return;
  const slider=row.querySelector("input[type=range]"), val=row.querySelector(".matval");
  const pct=Math.round(effectiveOpacity(label)*100);
  if(slider) slider.value=String(pct);
  if(val) val.textContent=pct+"%";
}
// Click-to-pick: cast the clicked pixel's camera ray (CPU, one-shot — not on
// the render hot path) through the SAME SDF the shader draws, find which
// leaf's material owns the hit point, and TOGGLE it opaque <-> mostly
// see-through. A quick click on the engine block makes it see-through; click
// it again to put it back. Ignored if the mouse actually dragged (orbiting).
function pickMaterialAt(clientX, clientY){
  if(!scene) return;
  const rect=canvas.getBoundingClientRect();
  if(rect.width<=0||rect.height<=0) return;
  const nx=((clientX-rect.left)/rect.width)*2-1;
  const ny=-(((clientY-rect.top)/rect.height)*2-1);
  let up=cam.up; const eye=eyePos(); const fwd=norm(sub(cam.target,eye));
  if(Math.abs(dot(fwd,norm(up)))>0.999) up=[0,1,0];
  const right=norm(cross(fwd,up)); const camUp=cross(right,fwd);
  const aspect=rect.width/rect.height;
  const tanHalf=Math.tan(((scene.camera&&scene.camera.fov_deg)||40)*Math.PI/360);
  const dir=norm(add(add(fwd, mul(right, nx*aspect*tanHalf)), mul(camUp, ny*tanHalf)));
  const maxDist=len(sub(eye,cam.target))+sceneDiag*3+0.5;
  const hitP=jsMarchPick(scene, eye, dir, bodyPoses, maxDist);
  if(!hitP) return;
  // nudge inward along the surface normal — jsMarchPick's stopping epsilon can
  // land the point on EITHER side of the surface; jsMaterialAt needs to be
  // confidently inside (mirrors the shader's mapD(p-n*0.0008,mat) convention).
  const pn=jsCalcN(scene, hitP, bodyPoses);
  const inP=sub(hitP, mul(pn, Math.max(0.003, sceneDiag*0.004)));
  const label=jsMaterialAt(scene, inP, bodyPoses);
  if(!label) return;
  const cur=effectiveOpacity(label);
  const next = cur>0.5 ? 0.15 : 1.0;              // toggle: mostly-transparent <-> opaque
  window.Radiance.setMaterialOpacity(label, next);
  const dd=$("dbg"); if(dd) dd.textContent = `${label}: opacity → ${Math.round(next*100)}%`;
}
async function load(url, title){
  try{
    const resp=await fetch(url); if(!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const obj=await resp.json();
    applyObj(obj, title, false);
    loadedObj=obj;                                   // pristine copy for the editor + reset
    syncEditor();                                    // (scene buttons retired — the toggle states persist)
    return true;
  }catch(e){ setErr((""+e).slice(0,180)); console.error(e); throw e; }
}

// ── "behind the simulation": the JSON Radiance renders from, live-editable ──
// The picture is a pure function of this JSON, so editing it IS editing the
// world. Nothing is validated — type an impossible seed and you see the
// impossible thing it implies. A bad parse / bad shader keeps the last render.
function editMsg(m,bad){ const el=$("editmsg"); if(el){ el.textContent=m||""; el.style.color=bad?"#FF4444":"#00FF88"; } }
function syncEditor(){ const ta=$("json"); if(ta&&loadedObj){ ta.value=JSON.stringify(loadedObj,null,2); } editMsg(""); }
function reRender(){ const ta=$("json"); if(!ta) return;
  let parsed; try{ parsed=JSON.parse(ta.value); }
  catch(e){ editMsg("JSON: "+e.message, true); return; }            // bad JSON → keep last render
  try{ applyObj(parsed, $("title").textContent, true);              // keep camera + playhead
       editMsg("rendered — physics is whatever you typed", false); }
  catch(e){ editMsg((""+e).slice(0,120), true); }                   // bad scene → keep last render
}

// ── time-rate knob + scrub ────────────────────────────────────────────
function updRate(){ rate=Math.pow(10,parseFloat($("rate").value));
  $("rateval").textContent = rate>=1 ? `${rate.toFixed(rate<10?2:0)} sim-s / wall-s`
    : `1 wall-s = ${rate.toPrecision(2)} sim-s (slow-mo)`; }
$("rate").addEventListener("input",updRate);
$("scrub").addEventListener("input",()=>{ simTime=parseFloat($("scrub").value)/1000*tEnd; playing=false; $("play").textContent="▶ play"; });
$("play").addEventListener("click",()=>{ if(tEnd<=0) return; playing=!playing;
  if(playing&&simTime>=tEnd) simTime=0; $("play").textContent=playing?"⏸ pause":"▶ play"; });

// ── orbit controls ────────────────────────────────────────────────────
let drag=false,lx=0,ly=0,dragDist=0;
canvas.addEventListener("mousedown",e=>{drag=true;lx=e.clientX;ly=e.clientY;dragDist=0;});
window.addEventListener("mouseup",()=>drag=false);
window.addEventListener("mousemove",e=>{ if(!drag) return;
  dragDist+=Math.hypot(e.clientX-lx,e.clientY-ly);
  // R2: the controller owns the drag/wheel MEANING per camera mode
  // (orbit/follow: legacy formulas verbatim inside camera.js; fly: look /
  // throttle). camctl-null falls back to the shipped E3 lines, verbatim.
  if(camctl) camctl.drag(e.clientX-lx, e.clientY-ly);
  else { cam.az-=(e.clientX-lx)*0.01; cam.el=Math.max(-1.5,Math.min(1.5,cam.el+(e.clientY-ly)*0.01)); }
  lx=e.clientX; ly=e.clientY; });
canvas.addEventListener("wheel",e=>{ e.preventDefault();
  if(camctl) camctl.wheel(e.deltaY);
  else cam.radius*=Math.exp(e.deltaY*0.0012); },{passive:false});
// A real click (not the tail of an orbit drag) picks the material under the
// cursor and toggles its render-only opacity — "click the engine block to
// make it see-through". See pickMaterialAt() for the raymarch + toggle.
canvas.addEventListener("click",e=>{ if(dragDist>4) return; pickMaterialAt(e.clientX,e.clientY); });
// Scene selection lives in the ⊞ Library gallery now (built from data/scenes.json) — no hardcoded buttons.
// "behind the simulation" flip + the live JSON editor
$("b-edit").addEventListener("click",()=>{ const open=$("editor").classList.toggle("open");
  $("b-edit").classList.toggle("on",open); if(open && !$("json").value) syncEditor();
  if(open){ $("library").classList.remove("open"); $("b-lib").classList.remove("on");
    $("matpanel").classList.remove("open"); $("b-mat").classList.remove("on"); } });
$("rerender").addEventListener("click",reRender);
// "◐ opacity" drawer — dial any material's render-only opacity BY NAME (no
// need to click geometry). Same one-drawer-at-a-time coordination as editor/library.
$("b-mat").addEventListener("click",()=>{
  const open=$("matpanel").classList.toggle("open"); $("b-mat").classList.toggle("on",open);
  if(open){ $("editor").classList.remove("open"); $("b-edit").classList.remove("on");
    $("library").classList.remove("open"); $("b-lib").classList.remove("on"); } });
$("editreset").addEventListener("click",()=>{ syncEditor(); reRender(); });
$("json").addEventListener("keydown",e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){ e.preventDefault(); reRender(); return; }
  if(e.key==="Tab"){ e.preventDefault(); const t=e.target,a=t.selectionStart,b=t.selectionEnd;
    t.value=t.value.slice(0,a)+"  "+t.value.slice(b); t.selectionStart=t.selectionEnd=a+2; } });
let _editTmr=0; $("json").addEventListener("input",()=>{ clearTimeout(_editTmr); _editTmr=setTimeout(reRender,650); });
// Radiance is ALWAYS the path tracer — there is no "drop to fast" anymore. A clean
// image is built by INTEGRATING many frames. Without float render targets this GPU
// cannot accumulate at all and falls back to the single-hit shim; the HUD's own
// render-mode line (see loop()) already says which path is live, so there is no
// separate indicator to keep in sync here.

// ── simulation library — a gallery built from data/scenes.json (build_library.py) ──
(async function buildLibrary(){
  const listEl=$("liblist"); if(!listEl) return;
  try{
    const r=await fetch("data/scenes.json"); if(!r.ok) return;
    const scenes=await r.json();
    listEl.innerHTML="";
    let lastGroup=null;
    const GROUP={showcase:"Showcases — emergent physics", drop:"Drops — Materia's questions",
                 gauntlet:"Gauntlet — 20 minimal prompts, fully local",
                 composition:"Compositions — multi-object scenes",
                 voxel_sdf:"Voxel (SDF-sampled) — direct from the analytic SDF",
                 voxel:"Voxel (real mesh) — ShapeNet/PartNet shapes",
                 mechanism:"Mechanisms — joints, gears, escapements",
                 thermal:"Thermal — conduction and glow",
                 object:"Objects — Deckard-grounded shapes",
                 theater:"Theaters — staging backdrops", demo:"Demos"};
    for(const s of scenes){
      const g=s.group||"drop";
      if(g!==lastGroup){ lastGroup=g;
        const hdr=document.createElement("div"); hdr.className="libgroup"; hdr.textContent=GROUP[g]||g;
        listEl.appendChild(hdr); }
      const card=document.createElement("div"); card.className="libcard";
      const t=document.createElement("div"); t.className="t"; t.textContent=s.title||s.slug;
      const q=document.createElement("div"); q.className="q";
      const meta=(s.verb&&s.verb!=="showcase"?"  ·  "+s.verb:"")+(s.frames?"  ·  "+s.frames+"f":"")
        +(s.leaves?"  ·  "+s.leaves+" leaves":"");
      q.textContent=(s.question||"")+meta;
      card.append(t,q);
      // baked-with badge, BELOW the title/meta: which of the current
      // renderer's features this file actually carries. A scene saved before
      // a feature shipped still loads and renders correctly (every consumer
      // degrades gracefully) — the badge says "re-render me to pick this up",
      // it does not mean "broken".
      if(!(s.has_bounds&&s.has_opacity)){
        const b=document.createElement("span"); b.className="libstale";
        const missing=[!s.has_bounds&&"bounds",!s.has_opacity&&"opacity",
                       s.frames&&!s.has_energy&&"energy"].filter(Boolean);
        b.textContent="stale: "+missing.join(", ");
        b.title="Saved before these renderer features shipped — still renders, "
               +"but re-render to pick them up.";
        card.appendChild(b);
      }
      card.addEventListener("click",()=>{
        load("data/"+s.slug+".json", s.title||s.slug).catch(e=>setErr("library: "+e));
        $("library").classList.remove("open"); $("b-lib").classList.remove("on"); });
      listEl.appendChild(card);
    }
    if(scenes.length) $("b-lib").title=scenes.length+" simulations — click to browse";
  }catch(e){ console.error("library:",e); }
})();
$("b-lib").addEventListener("click",()=>{
  const open=$("library").classList.toggle("open"); $("b-lib").classList.toggle("on",open);
  if(open){ $("editor").classList.remove("open"); $("b-edit").classList.remove("on");
    $("matpanel").classList.remove("open"); $("b-mat").classList.remove("on"); } });  // one drawer at a time

// ── main loop ─────────────────────────────────────────────────────────
let last=performance.now(), fpsT=last, fpsN=0;
function loop(now){
  const dt=(now-last)/1000; last=now;
  // R0 capture: device input is FROZEN — a resting gamepad's stick drift
  // (measured: a connected DualSense walked az by +1.8 rad between two
  // otherwise-pinned captures) emits cam_orbit intents that break the pinned
  // camera. The fifth nondeterminism source, found empirically; outside
  // capture mode polling is untouched.
  if(window._input&&window._input.poll&&!CAP) window._input.poll(dt);   // E3 device layer (input.js)
  if(!liveMode&&playing&&tEnd>0){ simTime+=dt*rate; if(simTime>=tEnd){ simTime=tEnd; playing=false; $("play").textContent="▶ play"; } }
  if(liveMode){
    // poses arrive from the /live websocket — the REAL solver stepping in
    // real time server-side; nothing here recomputes physics
    if(_liveFrame){ $("tval").textContent=`t=${_liveFrame.t_sim.toFixed(2)}s  LIVE`;
      const eEl=$("energy");
      if(eEl){ const p=_liveFrame.momentum||[0,0,0];
        eEl.textContent=`KE: ${(_liveFrame.energy_j||0).toFixed(2)} J   |p|: ${Math.hypot(p[0],p[1],p[2]).toFixed(2)} kg·m/s`; } }
  } else if(worldMode){
    // E4: streamed /world poses — this branch MUST sit between liveMode
    // and the tEnd playback arm, or posesAt(simTime) clobbers streamed
    // poses every rAF (the exact bug liveMode's short-circuit prevents).
    if(_worldFrame){ const m=_worldFrame;
      $("tval").textContent=`t=${(m.t||0).toFixed(2)}s  step ${m.step}  WORLD${m.paused?" ⏸":""}`;
      const eEl=$("energy");
      if(eEl){ const p=m.momentum||[0,0,0];
        eEl.textContent=`KE: ${(m.energy_j||0).toFixed(2)} J   |p|: ${Math.hypot(p[0],p[1],p[2]).toFixed(2)} kg·m/s`; }
      const pv=$("worldpaceval");
      if(pv) pv.textContent=`${(m.pace||1).toFixed(2)}×${m.paused?" ⏸":""}`;   // server echo, never slider position
      const bag=window._hudState;
      if(bag){
        bag.frame={step:m.step, t:m.t, pace:m.pace, paused:m.paused,
                   settled:m.settled, energy_j:m.energy_j,
                   momentum:m.momentum, angular_momentum:m.angular_momentum};
        bag.camera=camctl?camctl.state():null;   // STRUCTURED fields — never hudSuffix parsing
        bag.playback={simTime, tEnd, playing};
      }
    }
    if(window._hud) window._hud.touch();
  } else if(tEnd>0){ bodyPoses=posesAt(simTime); $("scrub").value=String(simTime/tEnd*1000);
    const p0=(bodyPoses[0]&&bodyPoses[0].pos)||[0,0,0];
    // follow camera: long falls opt in via scene.camera.follow (the body would
    // leave the frame for most of the drop otherwise)
    if(scene&&scene.camera&&scene.camera.follow&&bodyPoses[0]
       && !(CAP&&CAP.cam.length===6)) cam.target=p0.slice();   // R0: explicit capture pins beat the legacy follow write
    $("tval").textContent=`t=${simTime.toFixed(2)}s  y=${p0[1].toFixed(2)}m`;
    // live energy/momentum readout — frameEnergyJ/frameMomentum were just set
    // by the posesAt() call above (the same interpolated-at-simTime values,
    // not merely the nearest frame's)
    const eEl=$("energy");
    if(eEl){
      const pMag=Math.hypot(frameMomentum[0],frameMomentum[1],frameMomentum[2]);
      const fmt=(x)=>{ const ax=Math.abs(x);
        return ax>=100?x.toFixed(0):ax>=1?x.toFixed(2):x.toExponential(2); };
      eEl.textContent=`KE: ${fmt(frameEnergyJ)} J   |p|: ${fmt(pMag)} kg·m/s`;
    }
  }
  // R2 camera tick — ORDERING MATTERS: after bodyPoses=posesAt(simTime)
  // above (follow tracks THIS frame's pose) and after the legacy
  // scene.camera.follow write (in orbit mode the controller never touches
  // cam.target, so opted-in legacy scenes are pixel-identical; in
  // follow/fly the controller's later write wins).
  if(camctl) camctl.tick(dt);
  frameCount++;
  if(prog && !(CAP && capSent)){ try{ draw(); }catch(e){ setErr("draw: "+e); } }   // R0: a finished capture tab idles (its GL context is released)
  if(PROBE && !probeSent && scene && prog){
    probeSent=true;
    let pass=null, maxd=0, fpass=null, fmaxd=0;
    try{ const ss=scene.sdf_samples||[];
      for(const s of ss){ const d=jsEvalSDF(scene,s.p,null); maxd=Math.max(maxd,Math.abs(d-s.d)); }
      pass = ss.length ? maxd<1e-6 : null;        // null = no samples shipped
    }catch(e){ probeErrors.push("selfcheck: "+e); pass=false; }
    try{ [fpass,fmaxd]=fieldSelfCheck(); }        // the per-cell field twin
    catch(e){ probeErrors.push("fieldcheck: "+e); fpass=false; }
    window.__probeSelfcheck = pass;               // R0: the capture payload re-reports this (probe-mode-only stash)
    fetch("/probe",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({slug:(_scene||"?"), selfcheck_pass:pass,
        max_delta:maxd, shader_ok:!!prog,
        field_selfcheck_pass:fpass, field_max_delta_k:fmaxd,
        console_errors:probeErrors.slice(0,10), kind:scene.kind||"static",
        animated:tEnd>0})}).catch(()=>{});
  }
  // R0 capture trigger — after draw(), SAME rAF tick (readback validity with
  // preserveDrawingBuffer:false). capArmed means the pins have been applied,
  // so a slow async load can never capture an unpinned exposure.
  if(CAP && !capSent && capArmed && scene){
    if(!ptMode || !extF || !progPT){
      capSent = true;   // refuse honestly — a shim capture would baseline the wrong renderer
      fetch("/capture",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({id:CAP.id, slug:_scene,
          error:"no float render targets — PT path unavailable"})}).catch(()=>{});
      capReleaseGL();
    } else if(accN >= CAP.frames && (svoReady || !svoActiveNow)){
      // R4 capture readiness: a pinned exposure never races SVO residency
      capSent = true; captureAndPost(canvas.width, canvas.height);
      capReleaseGL();   // free this tab's context slot — see capReleaseGL()
    }
  }
  fpsN++; if(now-fpsT>500){ $("fps").textContent=`${Math.round(fpsN*1000/(now-fpsT))} fps`; fpsT=now; fpsN=0; }
  requestAnimationFrame(loop);
}
// A SAVED simulation opens directly by slug:  .../?scene=falling_feather
// This is the front-door dispatcher's hand-off — the URL it returns just works.
// The slug is sanitised (slugify's own [a-z0-9_] charset) so it can't escape data/.
const _scene=new URLSearchParams(location.search).get("scene");
if(_scene && /^[a-z0-9_]+$/i.test(_scene)){
  load(`data/${_scene}.json`,_scene.replace(/_/g," "))
    .then(()=>{ if(CAP) applyCapturePins(); })    // R0: pins land AFTER the async load's camera reset
    .catch((e)=>{
      if(PROBE && !probeSent){ probeSent=true;      // report the LOAD FAILURE itself —
        fetch("/probe",{method:"POST",headers:{"Content-Type":"application/json"},  // a silent no-report is undebuggable
          body:JSON.stringify({slug:_scene, kind:"load-failed", shader_ok:false,
            selfcheck_pass:false, console_errors:[String(e).slice(0,300)]
              .concat(probeErrors.slice(0,9))})}).catch(()=>{}); }
      return load("data/water.json","water").catch(()=>{});
    });
}else{
  load("data/water.json","water").catch(()=>{});           // open on the water showpiece
}

// ── external hook: let the Mentat chat drive this viewer without an iframe reload ──
// MUST live inside the IIFE to capture the module-scoped load()/scene(). chat.js
// awaits `radiance-ready`, then window.Radiance.load("data/<slug>.json", title)
// swaps the scene on the SHARED canvas (PT accumulation preserved) — no flicker.
window.Radiance = {
  load, applyObj, scene: () => scene, ready: true,
  // ── render-display options ────────────────────────────────────────────────
  // The app's UI drives these THROUGH the API, so it never reaches into renderer
  // internals. Each applies the change and resets the exposure. The data contract
  // is the SceneSpec (the JSON load() takes); these toggles are how it's shown.
  setBvhCull:    (on) => { bvhCull = !!on; if (scene) { buildProgram(scene); sceneId++; } },
  // R4 A/B toggle (bvhCull precedent): flip the SVO path live; rebuilds the
  // program (compile gate re-evaluates) and restarts the exposure.
  setSVO:        (on) => { svoEnabled = !!on; if (scene) { buildProgram(scene); sceneId++; } },
  svoStats:      () => (svoGL && svoGL.packed) ? svoGL.packed.stats : null,
  // ── render-only material opacity — a pure VIEW toggle, byte-isolated from
  // physics (the scene's mass/com/inertia were computed in Python, upstream of
  // and untouched by anything here). Pick a material by the exact label it
  // carries in scene.materials (e.g. "engine_block") — window.Radiance.materials()
  // lists them. alpha=1 (or null/undefined) clears any override, reverting to
  // the scene's baked default. This is the ONE place materialOpacity is
  // mutated — the UI (drawer slider, click-to-pick) and any external caller
  // (Mentat chat) both go through here, same seam as setEnvChecker.
  setMaterialOpacity: (label, alpha) => {
    if (alpha === null || alpha === undefined) delete materialOpacity[label];
    else materialOpacity[label] = Math.max(0.0, Math.min(1.0, +alpha));
    if (scene) { buildProgram(scene); sceneId++; }
    syncMatRow(label);
  },
  getMaterialOpacity: (label) => effectiveOpacity(label),
  materials:          () => scene ? Object.keys(scene.materials || {}) : [],
  // Programmatic click-to-pick (the same raymarch canvas clicks use) — lets an
  // external driver (Mentat chat, a test harness) say "toggle whatever's at
  // this pixel" without synthesizing DOM mouse events.
  pickMaterialAt: (clientX, clientY) => pickMaterialAt(clientX, clientY),
  options:       () => ({ bvhCull, svoEnabled, materialOpacity: { ...materialOpacity } }),
};
window.dispatchEvent(new Event("radiance-ready"));

// ── E3 input layer: device chords → game verbs (input.js) ─────────────
// ENGINE channel only in E3 — orbit/zoom keys drive the SAME cam object the
// mouse drag does (no second camera path). WORLD intents are PARKED on the
// instance (inp.lastWorldIntent): the /world socket wiring is E4/R2, and
// faking an apply here would be a silent drop wearing a mask. The reduction
// to AppliedForce lives server-side (engine/intents.py) — input.js never
// constructs force dicts.
if (window.SGInput) {
  SGInput.load().then(({bindings, verbs}) => {
    const inp = new SGInput({bindings, verbs,
      // target = the apply drawer's select when a live sim has built it
      getTarget: () => { const el = $("applytarget"); return (el && el.value) || null; },
      onWorldIntent: (i) => {
        inp.lastWorldIntent = i;          // debug surface stays truthful
        // E4: UN-PARKED — a keyboard shove genuinely pushes a body in
        // the persistent world. Reduction stays SERVER-side (intents.py).
        if (worldNet && worldNet.live) worldNet.intent(i.verb, i.params, i.target);
        // else: parked (honest — the static server has nothing to push against)
      },
      onEngineIntent: (i) => {
        // R2: the controller owns every camera verb (cam_orbit/cam_zoom/
        // cam_move/cam_mode/follow, mode-aware). The legacy inline
        // branches survive as the camctl-null fallback — a 404'd
        // camera.js degrades to E3 behavior instead of dead controls.
        if (camctl && camctl.handles(i.op)) {
          camctl.handleIntent(i);
        } else if (i.op === "cam_orbit") {
          cam.az += (i.args.daz || 0);
          cam.el = Math.max(-1.5, Math.min(1.5, cam.el + (i.args.delv || 0)));  // same clamp as the mousemove handler
        } else if (i.op === "cam_zoom") {
          cam.radius *= (i.args.factor || 1);
        } else if ((i.op === "pace" || i.op === "pause" || i.op === "resume")
                   && worldNet && worldNet.live) {
          // E4: host protocol ops ride the INTENT form so KeyP/bracket
          // chords get ticketed id echoes in the HUD log
          worldNet.intent(i.verb, i.args, null);
        } else {
          inp.lastEngineIntent = i;     // cam_pan (and no-host pace/pause/
          //                              resume) parked — honest: the
          //                              static viewer has no host
        }
      }});
    inp.attach(canvas);
    window._input = inp;                // debug surface: window._input.state()
  }).catch(e => console.warn("input:", e));
}

requestAnimationFrame(loop);
})();
