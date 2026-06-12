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

const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
if (!gl) { setErr("WebGL2 not available in this browser"); return; }
gl.bindVertexArray(gl.createVertexArray());      // some drivers require a bound VAO
gl.clearColor(0, 0, 0, 1);
const extF = gl.getExtension("EXT_color_buffer_float");   // float render targets → progressive PT accumulation

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
vec3 leafRot(vec4 rot, vec3 p, vec3 c){ return qrotInv(rot, p-c)+c; }
`;
function jsLeafSDF(leaf, q){
  const s=leaf.shape, c=s.center;
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
function jsShapeSDF(s,q){
  if(!s) return 1e9;
  if(s.type==="Torus"){ const d=sub(q,s.center);
    return Math.hypot(Math.hypot(d[0],d[1])-s.major_radius,d[2])-s.minor_radius; }
  if(s.type==="Outline") return jsOutlineSDF(s,q);
  if(s.type==="Rotated"){ const c=s.center;
    return jsShapeSDF(s.shape, add(qrotInvJS(s.rot, sub(q,c)), c)); }
  if(s.type==="Clipped") return Math.max(jsShapeSDF(s.shape,q), s.sign*(q[s.axis]-s.level));
  if(s.type==="Subtracted") return Math.max(jsShapeSDF(s.shape,q), -jsShapeSDF(s.cut,q));
  return jsLeafSDF({shape:s}, q);
}
// per-build registry of generated GLSL outline functions (profiles baked as consts)
let EXTRA_GLSL="", outlineN=0, outlineKeys=new Map();
function resetExtraGLSL(){ EXTRA_GLSL=""; outlineN=0; outlineKeys=new Map(); }
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
function glslShapeCall(s,qvar){
  if(!s) return "1e9";
  if(s.type==="Torus"){ const c=s.center;
    return `sdTorus(${qvar},vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}),${glf(s.major_radius)},${glf(s.minor_radius)})`; }
  if(s.type==="Outline") return glslOutline(s,qvar);
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

// ── build the fragment shader (colors BAKED as constants) ──────────────
function buildFragmentShader(scene){
  resetExtraGLSL();
  const ls=scene.csg_leaves, bodies=scene.bodies||[];
  // Per-body rigid transform (world→rest): inverse-rotate about the pose, then
  // re-add the body's pivot. A leaf with no `body` is static world geometry.
  const used=[...new Set(ls.map(l=>l.body).filter(b=>b!=null))].sort((a,b)=>a-b);
  let bodyDecls="";
  used.forEach(k=>{ const piv=(bodies[k]&&bodies[k].pivot)||[0,0,0];
    bodyDecls += `  vec3 bdy${k}=qrotInv(uBodyQuat[${k}], p-uBodyPos[${k}])`
              +  `+vec3(${glf(piv[0])},${glf(piv[1])},${glf(piv[2])});\n`; });
  let decls="", compose="  float d=s0;\n", matsel="  mat=-1;\n", colorfn="", metalfn="", reflfn="", emisfn="", tempfn="";
  for(let i=0;i<ls.length;i++){
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
    // per-object initial temperature from the sim layer (default STP = 293.15 K)
    tempfn += `  if(m==${i}) return ${glf(ls[i].temperature_k!==undefined?ls[i].temperature_k:293.15)};\n`;
  }
  for(let i=1;i<ls.length;i++){
    const op=ls[i].op;
    compose += (op==="subtract") ? `  d=max(d,-s${i});\n`
             : (op==="intersect") ? `  d=max(d,s${i});\n`
             : `  d=min(d,s${i});\n`;
  }
  for(let i=ls.length-1;i>=0;i--)
    matsel += `  ${i===ls.length-1?"if":"else if"}(s${i}<0.0) mat=${i};\n`;
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
             +  `    vec3 hh=normalize(Li+v); spec+=Lc*pow(max(0.0,dot(n,hh)),mix(24.0,64.0,metal))*ndl*sh; }\n`; });
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
vec3 emissivityOf(int m){    // Kirchhoff ε(λ): measured (1−R from n+k) where available
${emisfn}  return vec3(0.5);
}
float matTempK(int m){       // per-object initial temperature (sim layer; default STP)
${tempfn}  return 293.15;
}
float softShadow(vec3 ro, vec3 rd){          // sphere-traced soft shadow toward a light
  float res=1.0, t=SHEPS;
  for(int i=0;i<40;i++){ float h=mapOnly(ro+rd*t);
    if(h<0.0008) return 0.0;                 // fully blocked
    res=min(res, SHK*h/t);                   // penumbra: nearer occluders → softer edge
    t+=clamp(h,SHMIN,SHMAX); if(t>SMAXT) break; }
  return clamp(res,0.0,1.0);
}
float calcAO(vec3 p, vec3 n){                 // 5-tap SDF ambient occlusion (creases self-darken)
  float occ=0.0, sca=1.0;
  for(int i=0;i<5;i++){ float hr=AOH*(0.2+0.2*float(i));
    occ += (hr-mapOnly(p+n*hr))*sca; sca*=0.7; }
  return clamp(1.0-1.6*occ,0.0,1.0);
}
vec3 calcN(vec3 p){ float h=2e-4; vec2 e=vec2(1.0,-1.0);
  return normalize( e.xyy*mapOnly(p+e.xyy*h)+e.yyx*mapOnly(p+e.yyx*h)
                  + e.yxy*mapOnly(p+e.yxy*h)+e.xxx*mapOnly(p+e.xxx*h) ); }
vec3 shade(vec3 albedo, vec3 p, vec3 n, vec3 v, float metal){
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
vec3 incandescence(float T, vec3 emis){   // Planck's law × Kirchhoff emissivity — nature's glow, no colour table
  if(T < 700.0) return vec3(0.0);          // Draper point: below ~700 K emission is entirely IR
  vec3 lam=vec3(650e-9,550e-9,450e-9);
  vec3 x=1.4388e-2/(lam*T);                // c2/(lambda T), c2 = hc/k = 1.4388e-2 m*K
  vec3 L=emis/(pow(lam/650e-9,vec3(5.0))*(exp(x)-vec3(1.0)));   // ε(λ)·B(λ,T), spectral radiance
  vec3 e=L*EMISSION_SCALE;
  return e/(1.0+max(e.r,max(e.g,e.b)))*1.7;   // tone-map: compress brightness, preserve the Planck hue
}
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
    col=shade(matColor(mat), p, n, -dir, metal);
    float r0=reflR0(mat);
    if(r0>=0.0){                                  // reflective surface → cast a real reflection ray
      float cosi=max(0.0,dot(n,-dir));
      float fres=clamp(r0+(1.0-r0)*pow(1.0-cosi,5.0),0.0,1.0);   // Fresnel-Schlick: grazing → mirror
      vec3 rd=reflect(dir,n);
      vec3 rcol=traceReflect(p+n*max(SHMIN,0.002), rd);
      vec3 tint=metal>0.5?matColor(mat):vec3(1.0);              // metals tint their reflection
      col=mix(col, rcol*tint, fres);
    }
    col += incandescence(matTempK(mat)+uTemp, emissivityOf(mat));   // each object at ITS temperature + global heat (Planck × Kirchhoff)
  }
  frag=vec4(pow(clamp(col,0.0,1.0),vec3(0.4545)),1.0);  // sRGB gamma: black stays black (doctrine), dim hues read
}`;
}

// ── path-traced renderer (toggle): real light transport — no headlight/ambient/AO.
// Rays bounce off the derived BRDFs; the sky is an area light; and every object
// EMITS its own incandescence, so a hot object lights its cold neighbours. Noisy
// at low sample count (Monte Carlo) — accumulation/denoise is the next rung.
function buildPathShader(scene){
  resetExtraGLSL();
  const ls=scene.csg_leaves, bodies=scene.bodies||[];
  const used=[...new Set(ls.map(l=>l.body).filter(b=>b!=null))].sort((a,b)=>a-b);
  let bodyDecls="";
  used.forEach(k=>{ const piv=(bodies[k]&&bodies[k].pivot)||[0,0,0];
    bodyDecls += `  vec3 bdy${k}=qrotInv(uBodyQuat[${k}], p-uBodyPos[${k}])+vec3(${glf(piv[0])},${glf(piv[1])},${glf(piv[2])});\n`; });
  let decls="", compose="  float d=s0;\n", matsel="  mat=-1;\n", colorfn="", metalfn="", reflfn="", emisfn="", tempfn="", iorfn="";
  for(let i=0;i<ls.length;i++){
    const q = ls[i].body!=null ? `bdy${ls[i].body}` : "(p)";
    decls += `  float s${i}=${glslShapeCall(ls[i].shape, q)};\n`;
    const mat=scene.materials[ls[i].material]||{}, c=mat.color_rgb||[0.72,0.72,0.72];
    colorfn += `  if(m==${i}) return vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])});\n`;
    const isMetal=(mat.metal!==undefined?mat.metal:mat.emergent);
    metalfn += `  if(m==${i}) return ${glf(isMetal?1.0:0.0)};\n`;
    reflfn += `  if(m==${i}) return ${glf((mat.reflect_r0!==undefined)?mat.reflect_r0:-1.0)};\n`;
    const em=mat.emissivity_rgb;
    emisfn += em ? `  if(m==${i}) return vec3(${glf(em[0])},${glf(em[1])},${glf(em[2])});\n`
                 : `  if(m==${i}) return clamp(vec3(1.0)-vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])}),0.0,1.0);\n`;
    tempfn += `  if(m==${i}) return ${glf(ls[i].temperature_k!==undefined?ls[i].temperature_k:293.15)};\n`;
    iorfn += `  if(m==${i}) return ${glf(mat.refractive_index!==undefined?mat.refractive_index:-1.0)};\n`;
  }
  for(let i=1;i<ls.length;i++){ const op=ls[i].op;
    compose += (op==="subtract")?`  d=max(d,-s${i});\n`:(op==="intersect")?`  d=max(d,s${i});\n`:`  d=min(d,s${i});\n`; }
  for(let i=ls.length-1;i>=0;i--) matsel += `  ${i===ls.length-1?"if":"else if"}(s${i}<0.0) mat=${i};\n`;
  const bb=scene.bbox||[[-1,1],[-1,1],[-1,1]];
  const diag=len(sub([bb[0][1],bb[1][1],bb[2][1]],[bb[0][0],bb[1][0],bb[2][0]]))||1;
  const SHMIN=glf(Math.max(0.004,diag*0.004)), SHMAX=glf(Math.max(0.05,diag*0.06)),
        SMAXT=glf(Math.max(0.5,diag*1.2)), EPS=glf(Math.max(0.002,diag*0.0015));
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
  // Their WORLD centre is uploaded each frame (uEmitterPos) so a body's motion
  // carries the light; radius/temperature/emissivity are baked. NEE samples a
  // point on each and shadow-rays to it, so a hot object lights its neighbours
  // directly instead of waiting for a random bounce to land on it.
  const emitters=[];
  ls.forEach((l,i)=>{ const T=l.temperature_k;
    if(l.shape&&l.shape.type==="Sphere"&&T!==undefined&&T>700){
      const m=scene.materials[l.material]||{}, e=m.emissivity_rgb||[0.5,0.5,0.5];
      emitters.push({leaf:i, center:l.shape.center.slice(), radius:l.shape.radius,
                     body:(l.body!=null?l.body:-1), temp:T, emis:e}); }});
  ptEmitters=emitters; const NE=emitters.length;
  let neeBlk="";
  emitters.forEach((e,k)=>{ neeBlk += `
    if(mat!=${e.leaf}){ vec3 C=uEmitterPos[${k}]; float R=${glf(e.radius)}; vec3 wc=C-p; float dc=length(wc);
      if(dc>R+EPS){ vec3 wcn=wc/dc; float sinT=R/dc; float cosTm=sqrt(max(0.0,1.0-sinT*sinT));
        float cT=1.0-rnd(seed)*(1.0-cosTm); float sTs=sqrt(max(0.0,1.0-cT*cT)); float ph=6.2831853*rnd(seed);
        vec3 tt=normalize(cross(wcn, abs(wcn.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0))); vec3 bb=cross(wcn,tt);
        vec3 wl=normalize(tt*(sTs*cos(ph))+bb*(sTs*sin(ph))+wcn*cT); float ndl2=max(0.0,dot(n,wl));
        if(ndl2>0.0){ float vis=visRay(p+n*EPS, wl, dc-R-2.0*EPS);
          vec3 Le=incandescence(${glf(e.temp)}+uTemp, vec3(${glf(e.emis[0])},${glf(e.emis[1])},${glf(e.emis[2])}));
          L += thr*alb*Le*ndl2*2.0*(1.0-cosTm)*vis; } } }`; });
  return `#version 300 es
precision highp float; precision highp int;
#define MAXB ${Math.max(1,bodies.length)}
#define EMISSION_SCALE 2400000000.0
#define EPS ${EPS}
#define PT_ENV ${glf(scene.pt_env!==undefined?scene.pt_env:1.0)}
out vec4 frag;
uniform vec2 uRes; uniform vec3 uEye,uFwd,uRight,uUp;
uniform float uTanHalf,uAspect,uMaxDist,uTime,uTemp,uFrame,uReset;
uniform int uSPP;             // samples THIS frame — the photon budget for this single exposure
uniform float uPrevScale;     // fraction of the PRIOR exposure to keep: 1=integrate (long exposure), (N-1)/N=bounded shutter hold
uniform vec3 uBodyPos[MAXB]; uniform vec4 uBodyQuat[MAXB];
uniform sampler2D uPrev;                                 // previous accumulation (ping-pong)
uniform vec3 uEmitterPos[${Math.max(1,NE)}];             // world centres of glowing-sphere emitters (NEE)
${PRIMS_GLSL}${waterGLSL}${EXTRA_GLSL}
float mapD(vec3 p, out int mat){ ${bodyDecls}${decls}${compose}${matsel}  return d; }
float mapOnly(vec3 p){ int m; return mapD(p,m); }
vec3 matColor(int m){ ${colorfn}  return vec3(0.72); }
float metalness(int m){ ${metalfn}  return 0.0; }
vec3 emissivityOf(int m){ ${emisfn}  return vec3(0.5); }
float matTempK(int m){ ${tempfn}  return 293.15; }
float iorOf(int m){ ${iorfn}  return -1.0; }              // refractive index (water/glass), else -1
vec3 calcN(vec3 p){ float h=2e-4; vec2 e=vec2(1.0,-1.0);
  return normalize( e.xyy*mapOnly(p+e.xyy*h)+e.yyx*mapOnly(p+e.yyx*h)+e.yxy*mapOnly(p+e.yxy*h)+e.xxx*mapOnly(p+e.xxx*h) ); }
vec3 incandescence(float T, vec3 emis){ if(T<700.0) return vec3(0.0);
  vec3 lam=vec3(650e-9,550e-9,450e-9); vec3 x=1.4388e-2/(lam*T);
  vec3 L=emis/(pow(lam/650e-9,vec3(5.0))*(exp(x)-vec3(1.0))); vec3 e=L*EMISSION_SCALE;
  return e/(1.0+max(e.r,max(e.g,e.b)))*1.7; }
uint hashu(uint x){ x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x; }
float rnd(inout uint s){ s=hashu(s); return float(s)*(1.0/4294967296.0); }
vec3 cosHemi(vec3 n, float u1, float u2){ float r=sqrt(u1), phi=6.2831853*u2;
  vec3 t=normalize(cross(n, abs(n.y)<0.9?vec3(0.0,1.0,0.0):vec3(1.0,0.0,0.0))); vec3 b=cross(n,t);
  return normalize(t*(r*cos(phi))+b*(r*sin(phi))+n*sqrt(max(0.0,1.0-u1))); }
bool march(vec3 ro, vec3 rd, out vec3 p){ float t=EPS;   // |d| sphere-trace → works inside a medium too (refraction)
  for(int i=0;i<220;i++){ p=ro+rd*t; float d=mapOnly(p); if(abs(d)<0.0008) return true; t+=max(abs(d),0.0005); if(t>uMaxDist) return false; } return false; }
float sunShadow(vec3 ro, vec3 rd){ float t=EPS;
  for(int i=0;i<64;i++){ float d=mapOnly(ro+rd*t); if(d<0.001) return 0.0; t+=clamp(d,${SHMIN},${SHMAX}); if(t>${SMAXT}) break; } return 1.0; }
float visRay(vec3 ro, vec3 rd, float maxt){ float t=EPS;   // unoccluded toward an emitter up to maxt?
  for(int i=0;i<64;i++){ if(t>maxt) return 1.0; float d=mapOnly(ro+rd*t); if(d<0.001) return 0.0; t+=clamp(d,${SHMIN},${SHMAX}); } return 1.0; }
vec3 skyEmit(vec3 rd){ vec3 AUP=normalize(vec3(${glf(au[0])},${glf(au[1])},${glf(au[2])}));
  float u=clamp(0.5+0.5*dot(rd,AUP),0.0,1.0);
  vec3 c=PT_ENV*mix(vec3(0.50,0.57,0.65),vec3(0.32,0.47,0.75),u);           // sky area light (dimmable per scene)
  c += PT_ENV*${Sc}*${Si}*9.0*pow(max(0.0,dot(rd,-normalize(${Sd}))),3000.0);  // the sun disc
  return c; }
vec3 trace(vec3 ro, vec3 rd, inout uint seed){
  vec3 L=vec3(0.0), thr=vec3(1.0); bool spec=true;       // primary ray counts as a specular arrival
  for(int b=0;b<3;b++){
    vec3 p; if(!march(ro,rd,p)){ L+=thr*skyEmit(rd); break; }
    vec3 n=calcN(p); int mat; mapD(p-n*0.0008,mat);
    if(spec) L += thr*incandescence(matTempK(mat)+uTemp, emissivityOf(mat)); // own glow — only on specular arrival (NEE counts it otherwise)
    vec3 alb=matColor(mat); float metal=metalness(mat);
    float ior=iorOf(mat); bool diffuse=(metal<0.5 && ior<=0.0);
    if(diffuse){                                                           // direct lighting — diffuse surfaces only
      vec3 Li=-normalize(${Sd}); float ndl=max(0.0,dot(n,Li));             // direct sun (next-event)
      if(ndl>0.0){ float sh=sunShadow(p+n*EPS, Li); L += thr*alb*${Sc}*${Si}*PT_ENV*ndl*sh; }
      ${neeBlk}                                                            // direct light from glowing objects (NEE)
    }
    if(ior>0.0){                                                           // DIELECTRIC: Fresnel reflect / Snell refract
      vec3 nf=n; float eta; if(dot(rd,n)<0.0){ eta=1.0/ior; } else { eta=ior; nf=-n; }
      float ci=clamp(-dot(rd,nf),0.0,1.0); float r0=pow((ior-1.0)/(ior+1.0),2.0);
      float fres=r0+(1.0-r0)*pow(1.0-ci,5.0); vec3 rt=refract(rd,nf,eta);
      if(dot(rt,rt)<1e-6) fres=1.0;                                        // total internal reflection
      if(rnd(seed)<fres){ rd=reflect(rd,nf); ro=p+nf*EPS*2.0; }            // reflect (same side)
      else { rd=rt; ro=p-nf*EPS*2.0; }                                     // transmit (cross the interface)
      spec=true;
    } else if(metal>0.5){ rd=reflect(rd,n); thr*=alb; ro=p+n*EPS; spec=true; }   // metal: tinted specular
    else { rd=cosHemi(n, rnd(seed), rnd(seed)); thr*=alb; ro=p+n*EPS; spec=false; }   // diffuse: cosine bounce
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
const UNIFORMS=["uRes","uEye","uFwd","uRight","uUp","uTanHalf","uAspect","uMaxDist","uTime","uTemp","uBodyPos","uBodyQuat","uFrame","uPrev","uReset","uEmitterPos","uSPP","uPrevScale"];
function linkProg(fragSrc){
  const p=gl.createProgram();
  gl.attachShader(p,compile(gl.VERTEX_SHADER,VS));
  gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fragSrc));
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const fetchU=(p)=>{ const u={}; for(const n of UNIFORMS) u[n]=gl.getUniformLocation(p,n); return u; };
function buildProgram(scene){
  const t=linkProg(buildFragmentShader(scene));      // fast renderer (always)
  let pt=null; try{ pt=linkProg(buildPathShader(scene)); }   // path tracer (graceful if it fails)
  catch(e){ console.error("path-trace shader failed:",e); }
  if(prog) gl.deleteProgram(prog); if(progPT) gl.deleteProgram(progPT);
  prog=t; progPT=pt; U=fetchU(t); Upt=pt?fetchU(pt):{};
}

// ── progressive-accumulation buffers (ping-pong float targets) ─────────
// Each PT frame ADDS its samples to a float buffer; the display pass divides by
// the sample count. Stand still and the image gathers light and converges — the
// physical analogue of a long exposure filling in (samples ≈ photons, var ∝ 1/N).
const DISPLAY_FS=`#version 300 es
precision highp float;
uniform sampler2D uAccum; uniform float uInvN;
out vec4 frag;
void main(){ vec3 c=texelFetch(uAccum, ivec2(gl_FragCoord.xy),0).rgb*uInvN;
  frag=vec4(pow(clamp(c,0.0,1.0),vec3(0.4545)),1.0); }`;
let dispProg=null, dispU={};
let accumA=null, accumB=null, pingA=true, accN=0, accW=0, accH=0, ptKey="", ptTime=0;
function initDisplay(){ if(dispProg||!extF) return;
  try{ dispProg=linkProg(DISPLAY_FS);
    dispU={uAccum:gl.getUniformLocation(dispProg,"uAccum"), uInvN:gl.getUniformLocation(dispProg,"uInvN")}; }
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
let heatDeltaK=0;                                    // global heat added on top of each object's own temperature
let ptMode=!!extF, frameCount=0, sceneId=0;          // path tracing is THE renderer (whenever float targets exist)
let liveSPP=2;                                       // photons/frame for a LIVE exposure (cheap — quality comes from INTEGRATING frames, not brute force)
let liveScale=1.0;                                   // resolution of a live exposure — full res (a capable GPU handles it; lower ONLY if a weak GPU can't keep up)
let animateRipples=true;                             // live wind ripples by default — transparent AND animated. The bounded-shutter exposure stays clean at whatever frame rate the GPU delivers; toggle to a frozen still for a fully-converged showpiece.
let ptEmitters=[];                                   // glowing-sphere emitters for next-event estimation
let scene=null, traj=null, sceneDiag=1.0;
let bodyPoses=[], simTime=0, tEnd=0, playing=false, rate=1.0, drewOnce=false;

function frameBasis(up){
  const U2=norm(up), H=Math.abs(U2[0])<0.9?[1,0,0]:[0,0,1];
  const A=norm(cross(U2,H)), B=cross(U2,A); return [A,B,U2];
}
function eyePos(){
  const [A,B,Up]=frameBasis(cam.up), ce=Math.cos(cam.el), se=Math.sin(cam.el);
  const planar=add(mul(A,Math.cos(cam.az)*ce), mul(B,Math.sin(cam.az)*ce));
  return add(cam.target, mul(add(planar, mul(Up,se)), cam.radius));
}
function posesAt(t){     // → [{pos,quat}] for every body, interpolated
  if(!traj||!traj.frames.length) return [];
  const F=traj.frames, pack=(arr)=>arr.map(b=>({pos:b.pos.slice(),quat:(b.quat||[0,0,0,1]).slice()}));
  if(t<=F[0].t_sim) return pack(F[0].bodies);
  if(t>=F[F.length-1].t_sim) return pack(F[F.length-1].bodies);
  let lo=0,hi=F.length-1;
  while(hi-lo>1){ const m=(lo+hi)>>1; if(F[m].t_sim<=t) lo=m; else hi=m; }
  const a=F[lo],b=F[hi], u=(t-a.t_sim)/Math.max(1e-9,b.t_sim-a.t_sim);
  return a.bodies.map((ba,k)=>{ const bb=b.bodies[k]||ba, pa=ba.pos, pb=bb.pos;
    return {pos:[pa[0]+(pb[0]-pa[0])*u, pa[1]+(pb[1]-pa[1])*u, pa[2]+(pb[2]-pa[2])*u],
            quat:nlerp(ba.quat||[0,0,0,1], bb.quat||[0,0,0,1], u)}; });
}

// ── render ────────────────────────────────────────────────────────────
function resize(){ const dpr=Math.min(2,window.devicePixelRatio||1);
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
  let fp=null,fq=null;
  if(NB){ fp=new Float32Array(NB*3); fq=new Float32Array(NB*4);
    for(let k=0;k<NB;k++){ const Pp=bodyPoses[k]||{pos:[0,0,0],quat:[0,0,0,1]};
      fp[k*3]=Pp.pos[0]; fp[k*3+1]=Pp.pos[1]; fp[k*3+2]=Pp.pos[2];
      fq[k*4]=Pp.quat[0]; fq[k*4+1]=Pp.quat[1]; fq[k*4+2]=Pp.quat[2]; fq[k*4+3]=Pp.quat[3]; } }
  const bindScene=(UU,timeVal,frameSeed)=>{
    gl.uniform2f(UU.uRes,w,h);
    gl.uniform3fv(UU.uEye,eye); gl.uniform3fv(UU.uFwd,fwd); gl.uniform3fv(UU.uRight,right); gl.uniform3fv(UU.uUp,camUp);
    gl.uniform1f(UU.uTanHalf,tanHalf); gl.uniform1f(UU.uAspect,w/h); gl.uniform1f(UU.uMaxDist,md);
    gl.uniform1f(UU.uTime,timeVal); gl.uniform1f(UU.uTemp,heatDeltaK);
    if(UU.uFrame!=null) gl.uniform1f(UU.uFrame,frameSeed);
    if(NB){ gl.uniform3fv(UU.uBodyPos,fp); gl.uniform4fv(UU.uBodyQuat,fq); } };
  if(ptMode && extF && !dispProg) initDisplay();
  const key=`${eye.map(x=>x.toFixed(3))}|${cam.target.map(x=>x.toFixed(3))}|${simTime.toFixed(3)}|${heatDeltaK}|${w}x${h}|${sceneId}`;
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
    if(advanceTime) ptTime=(performance.now()-T0)*0.001;
    const src=pingA?accumA:accumB, dst=pingA?accumB:accumA;
    gl.bindFramebuffer(gl.FRAMEBUFFER,dst.fbo); gl.viewport(0,0,w,h);
    gl.useProgram(progPT); bindScene(Upt, ptTime, frameCount);
    gl.uniform1i(Upt.uSPP, spp); gl.uniform1f(Upt.uPrevScale, prevScale);
    if(ptEmitters.length && Upt.uEmitterPos){      // emitter world centres (body motion carries the light)
      const ep=new Float32Array(ptEmitters.length*3);
      ptEmitters.forEach((e,k)=>{ let c=e.center;
        if(e.body>=0 && bodyPoses[e.body]){ const Po=bodyPoses[e.body];
          const piv=((scene.bodies[e.body]||{}).pivot)||[0,0,0];
          c=add(Po.pos, qrotJS(Po.quat, sub(e.center, piv))); }
        ep[k*3]=c[0]; ep[k*3+1]=c[1]; ep[k*3+2]=c[2]; });
      gl.uniform3fv(Upt.uEmitterPos, ep); }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,src.tex);
    gl.uniform1i(Upt.uPrev,0); gl.uniform1f(Upt.uReset, reset?1.0:0.0);
    gl.drawArrays(gl.TRIANGLES,0,3);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,w,h);
    gl.useProgram(dispProg);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,dst.tex);
    gl.uniform1i(dispU.uAccum,0); gl.uniform1f(dispU.uInvN,1.0/frames);
    gl.drawArrays(gl.TRIANGLES,0,3); pingA=!pingA;
    const dd=$("dbg"); if(dd) dd.textContent =
        moving   ? `path-traced · live · ${spp} spp/frame · orbiting (clears when you stop) · ${w}x${h}`
      : animated ? `path-traced · ${Math.min(accN,NMAX)}-frame exposure · ${frames*spp} photons/px · ripples live · ${w}x${h}`
      :            `path-traced · ${frames*spp} samples/px · converging… · ${w}x${h}`;
  } else {                                                   // ── compatibility shim: GPU lacks float render targets ──
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,w,h); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog); bindScene(U,(performance.now()-T0)*0.001, frameCount);
    gl.drawArrays(gl.TRIANGLES,0,3);
    const dd=$("dbg"); if(dd) dd.textContent=`compatibility view — this GPU can't path-trace (no float render targets) · ${w}x${h}`;
  }
  drewOnce=true;
}

// ── self-check + load ─────────────────────────────────────────────────
function selfCheck(){
  const s=scene.sdf_samples;
  if(!s||!s.length){ setOk("geometry self-check: <span class='lbl'>(no samples)</span>"); return; }
  let md=0; for(const smp of s) md=Math.max(md,Math.abs(jsEvalSDF(scene,smp.p,null)-smp.d));
  const ok=md<1e-6;
  setOk(`geometry self-check: <span class="${ok?'ok':'bad'}">${ok?'PASS':'FAIL'}</span> `
       +`<span class="lbl">(max Δ ${md.toExponential(1)} vs Python)</span>`);
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
  buildProgram(newScene);                            // throws on a bad shader → live render untouched
  scene=newScene; traj=newTraj;                      // commit only after the compile succeeds
  if(!preserve){ const cm=scene.camera||{};
    cam.target=(cm.target||[0,0,0]).slice(); cam.up=(cm.up||[0,0,1]).slice();
    cam.radius=cm.orbit_radius||0.3;                 // a scene may suggest its opening angle (flat grids want face-on)
    cam.az=(cm.az0!==undefined?cm.az0:0.8); cam.el=(cm.el0!==undefined?cm.el0:0.45); }
  const bb=scene.bbox||[[-1,1],[-1,1],[-1,1]];
  sceneDiag=len(sub([bb[0][1],bb[1][1],bb[2][1]],[bb[0][0],bb[1][0],bb[2][0]]));
  sceneId++;                                         // new scene → PT accumulation resets
  selfCheck();
  $("title").textContent=scene.name||title||""; $("src").textContent=scene.source||"";
  const hasTraj=!!(traj&&traj.frames&&traj.frames.length>1);
  tEnd=hasTraj?traj.t_end_s:0;
  if(!preserve){ simTime=0; playing=false; $("play").textContent="▶ play";
    if(hasTraj){ rate=traj.suggested_rate||1; $("rate").value=Math.log10(rate); updRate(); } }
  else if(simTime>tEnd){ simTime=tEnd>0?tEnd:0; }
  $("playgrp").style.opacity=$("rategrp").style.opacity=hasTraj?"1":"0.35";
  bodyPoses=posesAt(simTime); drewOnce=false;
  return true;
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
$("temp").addEventListener("input",()=>{ heatDeltaK=parseFloat($("temp").value);
  const eff=STP_K+heatDeltaK;                            // effective T for a room-temperature object
  const name=eff<700?"cold":eff<1000?"dull red":eff<1300?"red-orange":eff<1700?"orange":eff<2200?"yellow":eff<2800?"yellow-white":"white-hot";
  $("tempval").textContent=`${eff|0} K (${name})`; });
$("scrub").addEventListener("input",()=>{ simTime=parseFloat($("scrub").value)/1000*tEnd; playing=false; $("play").textContent="▶ play"; });
$("play").addEventListener("click",()=>{ if(tEnd<=0) return; playing=!playing;
  if(playing&&simTime>=tEnd) simTime=0; $("play").textContent=playing?"⏸ pause":"▶ play"; });

// ── orbit controls ────────────────────────────────────────────────────
let drag=false,lx=0,ly=0;
canvas.addEventListener("mousedown",e=>{drag=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener("mouseup",()=>drag=false);
window.addEventListener("mousemove",e=>{ if(!drag) return;
  cam.az-=(e.clientX-lx)*0.01; cam.el=Math.max(-1.5,Math.min(1.5,cam.el+(e.clientY-ly)*0.01));
  lx=e.clientX; ly=e.clientY; });
canvas.addEventListener("wheel",e=>{ e.preventDefault(); cam.radius*=Math.exp(e.deltaY*0.0012); },{passive:false});
// Scene selection lives in the ⊞ Library gallery now (built from data/scenes.json) — no hardcoded buttons.
// "behind the simulation" flip + the live JSON editor
$("b-edit").addEventListener("click",()=>{ const open=$("editor").classList.toggle("open");
  $("b-edit").classList.toggle("on",open); if(open && !$("json").value) syncEditor();
  if(open){ $("library").classList.remove("open"); $("b-lib").classList.remove("on"); } });
$("rerender").addEventListener("click",reRender);
$("editreset").addEventListener("click",()=>{ syncEditor(); reRender(); });
$("json").addEventListener("keydown",e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){ e.preventDefault(); reRender(); return; }
  if(e.key==="Tab"){ e.preventDefault(); const t=e.target,a=t.selectionStart,b=t.selectionEnd;
    t.value=t.value.slice(0,a)+"  "+t.value.slice(b); t.selectionStart=t.selectionEnd=a+2; } });
let _editTmr=0; $("json").addEventListener("input",()=>{ clearTimeout(_editTmr); _editTmr=setTimeout(reRender,650); });
// Radiance is ALWAYS the path tracer — there is no "drop to fast" anymore. A clean
// image is built by INTEGRATING many frames, so a STILL pool converges to pristine;
// LIVE ripples deny the tracer those frames → it can only manage a rough, low-res
// exposure. This button lets the Captain pick which physics tradeoff to watch.
$("b-pt").addEventListener("click",()=>{
  if(!extF){ $("b-pt").textContent="≈ no float targets"; return; }      // this GPU can't path-trace at all
  animateRipples=!animateRipples; accN=0;                              // restart the exposure on the mode switch
  $("b-pt").classList.toggle("on",animateRipples);
  $("b-pt").textContent = animateRipples ? "≈ Ripples: live" : "≈ Ripples: still"; });
if(extF){ $("b-pt").textContent=animateRipples?"≈ Ripples: live":"≈ Ripples: still"; $("b-pt").classList.toggle("on",animateRipples); }
else { $("b-pt").textContent="≈ no float targets"; }

// ── simulation library — a gallery built from data/scenes.json (build_library.py) ──
(async function buildLibrary(){
  const listEl=$("liblist"); if(!listEl) return;
  try{
    const r=await fetch("data/scenes.json"); if(!r.ok) return;
    const scenes=await r.json();
    listEl.innerHTML="";
    let lastGroup=null;
    const GROUP={showcase:"Showcases — emergent physics", drop:"Drops — Materia's questions",
                 gauntlet:"Gauntlet — 20 minimal prompts, fully local"};
    for(const s of scenes){
      const g=s.group||"drop";
      if(g!==lastGroup){ lastGroup=g;
        const hdr=document.createElement("div"); hdr.className="libgroup"; hdr.textContent=GROUP[g]||g;
        listEl.appendChild(hdr); }
      const card=document.createElement("div"); card.className="libcard";
      const t=document.createElement("div"); t.className="t"; t.textContent=s.title||s.slug;
      const q=document.createElement("div"); q.className="q";
      const meta=(s.verb&&s.verb!=="showcase"?"  ·  "+s.verb:"")+(s.frames?"  ·  "+s.frames+"f":"");
      q.textContent=(s.question||"")+meta;
      card.append(t,q);
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
  if(open){ $("editor").classList.remove("open"); $("b-edit").classList.remove("on"); } });  // one drawer at a time

// ── main loop ─────────────────────────────────────────────────────────
let last=performance.now(), fpsT=last, fpsN=0;
function loop(now){
  const dt=(now-last)/1000; last=now;
  if(playing&&tEnd>0){ simTime+=dt*rate; if(simTime>=tEnd){ simTime=tEnd; playing=false; $("play").textContent="▶ play"; } }
  if(tEnd>0){ bodyPoses=posesAt(simTime); $("scrub").value=String(simTime/tEnd*1000);
    const p0=(bodyPoses[0]&&bodyPoses[0].pos)||[0,0,0];
    // follow camera: long falls opt in via scene.camera.follow (the body would
    // leave the frame for most of the drop otherwise)
    if(scene&&scene.camera&&scene.camera.follow&&bodyPoses[0]) cam.target=p0.slice();
    $("tval").textContent=`t=${simTime.toFixed(2)}s  y=${p0[1].toFixed(2)}m`; }
  frameCount++;
  if(prog){ try{ draw(); }catch(e){ setErr("draw: "+e); } }
  if(PROBE && !probeSent && scene && prog){
    probeSent=true;
    let pass=null, maxd=0;
    try{ const ss=scene.sdf_samples||[];
      for(const s of ss){ const d=jsEvalSDF(scene,s.p,null); maxd=Math.max(maxd,Math.abs(d-s.d)); }
      pass = ss.length ? maxd<1e-6 : null;        // null = no samples shipped
    }catch(e){ probeErrors.push("selfcheck: "+e); pass=false; }
    fetch("/probe",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({slug:(_scene||"?"), selfcheck_pass:pass,
        max_delta:maxd, shader_ok:!!prog,
        console_errors:probeErrors.slice(0,10), kind:scene.kind||"static",
        animated:tEnd>0})}).catch(()=>{});
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
    .catch(()=>load("data/water.json","water").catch(()=>{}));
}else{
  load("data/water.json","water").catch(()=>{});           // open on the water showpiece
}

// ── external hook: let the Mentat chat drive this viewer without an iframe reload ──
// MUST live inside the IIFE to capture the module-scoped load()/scene(). chat.js
// awaits `radiance-ready`, then window.Radiance.load("data/<slug>.json", title)
// swaps the scene on the SHARED canvas (PT accumulation preserved) — no flicker.
window.Radiance = { load, applyObj, scene: () => scene, ready: true };
window.dispatchEvent(new Event("radiance-ready"));

requestAnimationFrame(loop);
})();
