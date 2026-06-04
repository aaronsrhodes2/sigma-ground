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

const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
if (!gl) { setErr("WebGL2 not available in this browser"); return; }
gl.bindVertexArray(gl.createVertexArray());      // some drivers require a bound VAO
gl.clearColor(0, 0, 0, 1);

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
vec3 qrotInv(vec4 q, vec3 v){ vec3 u=-q.xyz; vec3 t=2.0*cross(u,v); return v+q.w*t+cross(u,t); }
`;
function jsLeafSDF(leaf, q){
  const s=leaf.shape, c=s.center;
  if(s.type==="Sphere"||s.type==="HollowSphere") return len(sub(q,c))-s.radius;
  if(s.type==="Cylinder"){ const d=sub(q,c), dr=Math.hypot(d[0],d[1])-s.radius, da=Math.abs(d[2])-s.height*0.5;
    return Math.min(Math.max(dr,da),0)+Math.hypot(Math.max(dr,0),Math.max(da,0)); }
  if(s.type==="Box"){ const d=[Math.abs(q[0]-c[0])-s.x*0.5,Math.abs(q[1]-c[1])-s.y*0.5,Math.abs(q[2]-c[2])-s.z*0.5];
    return Math.min(Math.max(d[0],Math.max(d[1],d[2])),0)+Math.hypot(Math.max(d[0],0),Math.max(d[1],0),Math.max(d[2],0)); }
  return 1e9;
}
function glslLeafCall(leaf, qvar){
  const s=leaf.shape, c=s.center, ctr=`vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])})`;
  if(s.type==="Sphere"||s.type==="HollowSphere") return `sdSphere(${qvar},${ctr},${glf(s.radius)})`;
  if(s.type==="Cylinder") return `sdCyl(${qvar},${ctr},${glf(s.radius)},${glf(s.height)})`;
  if(s.type==="Box") return `sdBox(${qvar},${ctr},vec3(${glf(s.x)},${glf(s.y)},${glf(s.z)}))`;
  return "1e9";
}
function jsEvalSDF(scene, p, poses){
  const ls=scene.csg_leaves, bodies=scene.bodies||[]; let d=0;
  for(let i=0;i<ls.length;i++){
    let q=p; const bi=ls[i].body;
    if(bi!=null && poses && poses[bi]){             // world → body-local (rest)
      const piv=(bodies[bi]&&bodies[bi].pivot)||[0,0,0];
      q=add(qrotInvJS(poses[bi].quat, sub(p,poses[bi].pos)), piv);
    }
    const si = jsLeafSDF(ls[i], q);
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
  const ls=scene.csg_leaves, bodies=scene.bodies||[];
  // Per-body rigid transform (world→rest): inverse-rotate about the pose, then
  // re-add the body's pivot. A leaf with no `body` is static world geometry.
  const used=[...new Set(ls.map(l=>l.body).filter(b=>b!=null))].sort((a,b)=>a-b);
  let bodyDecls="";
  used.forEach(k=>{ const piv=(bodies[k]&&bodies[k].pivot)||[0,0,0];
    bodyDecls += `  vec3 bdy${k}=qrotInv(uBodyQuat[${k}], p-uBodyPos[${k}])`
              +  `+vec3(${glf(piv[0])},${glf(piv[1])},${glf(piv[2])});\n`; });
  let decls="", compose="  float d=s0;\n", matsel="  mat=-1;\n", colorfn="", metalfn="";
  for(let i=0;i<ls.length;i++){
    const q = ls[i].body!=null ? `bdy${ls[i].body}` : "(p)";
    decls += `  float s${i}=${glslLeafCall(ls[i], q)};\n`;
    const mat=scene.materials[ls[i].material]||{};
    const c=mat.color_rgb||[0.72,0.72,0.72];
    colorfn += `  if(m==${i}) return vec3(${glf(c[0])},${glf(c[1])},${glf(c[2])});\n`;
    const isMetal = (mat.metal!==undefined ? mat.metal : mat.emergent);  // band-gap nonmetals shade matte
    metalfn += `  if(m==${i}) return ${glf(isMetal?1.0:0.0)};\n`;
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
  return `#version 300 es
precision highp float; precision highp int;
#define MAXB ${Math.max(1, bodies.length)}
#define SHEPS ${SHEPS}
#define SHMIN ${SHMIN}
#define SHMAX ${SHMAX}
#define SMAXT ${SMAXT}
#define SHK 14.0
#define AOH ${AOH}
out vec4 frag;
uniform vec2 uRes; uniform vec3 uEye,uFwd,uRight,uUp;
uniform float uTanHalf,uAspect,uMaxDist;
uniform vec3 uBodyPos[MAXB]; uniform vec4 uBodyQuat[MAXB];
${PRIMS_GLSL}
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
    col=shade(matColor(mat), p, n, -dir, metalness(mat));
  }
  frag=vec4(pow(clamp(col,0.0,1.0),vec3(0.4545)),1.0);  // sRGB gamma: black stays black (doctrine), dim hues read
}`;
}

// ── GL program plumbing ───────────────────────────────────────────────
const VS=`#version 300 es
void main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
  gl_Position=vec4(p*2.0-1.0,0.0,1.0); }`;
function compile(type,src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){ const log=gl.getShaderInfoLog(s); console.error(log,"\n",src); throw new Error(log); } return s; }
let prog=null, U={};
function buildProgram(scene){
  const p=gl.createProgram();
  gl.attachShader(p,compile(gl.VERTEX_SHADER,VS));
  gl.attachShader(p,compile(gl.FRAGMENT_SHADER,buildFragmentShader(scene)));
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  if(prog) gl.deleteProgram(prog);
  prog=p; gl.useProgram(p);
  U={}; for(const u of ["uRes","uEye","uFwd","uRight","uUp","uTanHalf","uAspect","uMaxDist","uBodyPos","uBodyQuat"])
    U[u]=gl.getUniformLocation(p,u);
}

// ── state ─────────────────────────────────────────────────────────────
const cam={az:0.8,el:0.45,radius:0.3,target:[0,0,0],up:[0,0,1]};
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
  let w=Math.floor((canvas.clientWidth||window.innerWidth)*dpr);
  let h=Math.floor((canvas.clientHeight||(window.innerHeight-60))*dpr);
  w=Math.max(2,w); h=Math.max(2,h);
  if(canvas.width!==w||canvas.height!==h){ canvas.width=w; canvas.height=h; } }
function draw(){
  resize(); gl.viewport(0,0,canvas.width,canvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT);
  let up=cam.up; const eye=eyePos(); const fwd=norm(sub(cam.target,eye));
  if(Math.abs(dot(fwd,norm(up)))>0.999) up=[0,1,0];          // anti-gimbal
  const right=norm(cross(fwd,up)); const camUp=cross(right,fwd);
  gl.useProgram(prog);
  gl.uniform2f(U.uRes,canvas.width,canvas.height);
  gl.uniform3fv(U.uEye,eye); gl.uniform3fv(U.uFwd,fwd);
  gl.uniform3fv(U.uRight,right); gl.uniform3fv(U.uUp,camUp);
  gl.uniform1f(U.uTanHalf,Math.tan((scene.camera.fov_deg||40)*Math.PI/360));
  gl.uniform1f(U.uAspect,canvas.width/canvas.height);
  const md=len(sub(eye,cam.target))+sceneDiag*1.5+0.1;     // dynamic — survives zoom
  gl.uniform1f(U.uMaxDist,md);
  const NB=(scene.bodies||[]).length;
  if(NB){ const fp=new Float32Array(NB*3), fq=new Float32Array(NB*4);
    for(let k=0;k<NB;k++){ const P=bodyPoses[k]||{pos:[0,0,0],quat:[0,0,0,1]};
      fp[k*3]=P.pos[0]; fp[k*3+1]=P.pos[1]; fp[k*3+2]=P.pos[2];
      fq[k*4]=P.quat[0]; fq[k*4+1]=P.quat[1]; fq[k*4+2]=P.quat[2]; fq[k*4+3]=P.quat[3]; }
    gl.uniform3fv(U.uBodyPos,fp); gl.uniform4fv(U.uBodyQuat,fq); }
  gl.drawArrays(gl.TRIANGLES,0,3);
  const dEye=jsEvalSDF(scene,eye,bodyPoses);
  const dd=$("dbg"); if(dd) dd.textContent=`eye(${eye.map(x=>+x.toFixed(3))})  d(eye)=${dEye.toFixed(3)}  maxDist=${md.toFixed(2)}  ${canvas.width}x${canvas.height}`;
  if(!drewOnce){ drewOnce=true; console.log("Radiance: first draw OK",
    {eye, fwd, target:cam.target, dEye, maxDist:md, canvas:[canvas.width,canvas.height], leaves:scene.csg_leaves.length}); }
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
async function load(url, title){
  try{
    const resp=await fetch(url); if(!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const obj=await resp.json();
    scene = obj.kind==="trajectory" ? obj.scene : obj;
    traj  = obj.kind==="trajectory" ? obj.trajectory : (obj.trajectory||null);
    if(!scene||!scene.csg_leaves||!scene.csg_leaves.length) throw new Error("scene has no csg_leaves");
    const cm=scene.camera||{};
    cam.target=(cm.target||[0,0,0]).slice(); cam.up=(cm.up||[0,0,1]).slice();
    cam.radius=cm.orbit_radius||0.3;                 // a scene may suggest its opening angle (flat grids want face-on)
    cam.az=(cm.az0!==undefined?cm.az0:0.8); cam.el=(cm.el0!==undefined?cm.el0:0.45);
    const bb=scene.bbox||[[-1,1],[-1,1],[-1,1]];
    sceneDiag=len(sub([bb[0][1],bb[1][1],bb[2][1]],[bb[0][0],bb[1][0],bb[2][0]]));
    buildProgram(scene);
    selfCheck();
    $("title").textContent=scene.name||title; $("src").textContent=scene.source||"";
    const hasTraj=!!(traj&&traj.frames&&traj.frames.length>1);
    tEnd=hasTraj?traj.t_end_s:0; simTime=0; playing=false; bodyPoses=posesAt(0); drewOnce=false;
    $("playgrp").style.opacity=$("rategrp").style.opacity=hasTraj?"1":"0.35";
    $("play").textContent="▶ play";
    if(hasTraj){ rate=traj.suggested_rate||1; $("rate").value=Math.log10(rate); updRate(); }
    document.querySelectorAll("#bar button").forEach(b=>b.classList.remove("on"));
    return true;
  }catch(e){ setErr((""+e).slice(0,180)); console.error(e); throw e; }
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
let drag=false,lx=0,ly=0;
canvas.addEventListener("mousedown",e=>{drag=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener("mouseup",()=>drag=false);
window.addEventListener("mousemove",e=>{ if(!drag) return;
  cam.az-=(e.clientX-lx)*0.01; cam.el=Math.max(-1.5,Math.min(1.5,cam.el+(e.clientY-ly)*0.01));
  lx=e.clientX; ly=e.clientY; });
canvas.addEventListener("wheel",e=>{ e.preventDefault(); cam.radius*=Math.exp(e.deltaY*0.0012); },{passive:false});
$("b-cup").addEventListener("click",()=>load("data/cup.json","coffee cup").then(()=>$("b-cup").classList.add("on")).catch(()=>{}));
$("b-drop").addEventListener("click",()=>load("data/drop.json","dropped sphere").then(()=>$("b-drop").classList.add("on")).catch(()=>{}));
$("b-mat").addEventListener("click",()=>load("data/materials.json","materials").then(()=>$("b-mat").classList.add("on")).catch(()=>{}));
$("b-tip").addEventListener("click",()=>load("data/tip.json","chair tip").then(()=>$("b-tip").classList.add("on")).catch(()=>{}));

// ── main loop ─────────────────────────────────────────────────────────
let last=performance.now(), fpsT=last, fpsN=0;
function loop(now){
  const dt=(now-last)/1000; last=now;
  if(playing&&tEnd>0){ simTime+=dt*rate; if(simTime>=tEnd){ simTime=tEnd; playing=false; $("play").textContent="▶ play"; } }
  if(tEnd>0){ bodyPoses=posesAt(simTime); $("scrub").value=String(simTime/tEnd*1000);
    const p0=(bodyPoses[0]&&bodyPoses[0].pos)||[0,0,0];
    $("tval").textContent=`t=${simTime.toFixed(2)}s  y=${p0[1].toFixed(2)}m`; }
  if(prog){ try{ draw(); }catch(e){ setErr("draw: "+e); } }
  fpsN++; if(now-fpsT>500){ $("fps").textContent=`${Math.round(fpsN*1000/(now-fpsT))} fps`; fpsT=now; fpsN=0; }
  requestAnimationFrame(loop);
}
load("data/cup.json","coffee cup").then(()=>$("b-cup").classList.add("on")).catch(()=>{});
requestAnimationFrame(loop);
})();
