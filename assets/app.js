const canvas = document.querySelector('#field');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: true });
if (!gl) document.querySelector('#detail').textContent = 'WebGL2 is required for the splat field.';

const vertex = `#version 300 es
precision highp float;
in vec3 aPosition;
in vec3 aScale;
in vec4 aColor;
in float aRadiance;
uniform mat4 uViewProjection;
uniform float uPointScale;
out vec4 vColor;
out vec2 vEllipse;
out float vGlow;
void main() {
  vec4 clip = uViewProjection * vec4(aPosition, 1.0);
  gl_Position = clip;
  float perspective = uPointScale / max(1.0, clip.w);
  gl_PointSize = clamp(max(aScale.x, max(aScale.y, aScale.z)) * perspective, 3.0, 86.0);
  vEllipse = vec2(max(.15, aScale.y / max(.001, aScale.x)), max(.15, aScale.z / max(.001, aScale.x)));
  vColor = aColor / 255.0;
  vGlow = clamp(aRadiance / (aRadiance + 1.0), .15, 1.0);
}`;
const fragment = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vEllipse;
in float vGlow;
out vec4 outColor;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  p /= vEllipse;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float gaussian = exp(-3.2 * r2);
  outColor = vec4(vColor.rgb * (1.0 + .8 * vGlow), vColor.a * gaussian * (.45 + .55 * vGlow));
}`;

function shader(type, source) {
  const value = gl.createShader(type); gl.shaderSource(value, source); gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
  return value;
}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vertex));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragment));
gl.linkProgram(program); gl.useProgram(program);
gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE); gl.enable(gl.DEPTH_TEST);

let splats = [], basinList = [], level = 'basins', yaw = .55, pitch = -.24, zoom = 25;
let dragging = false, moved = false, last = [0, 0], viewProjection = new Float32Array(16);
const attributes = ['aPosition', 'aScale', 'aColor', 'aRadiance'].map(name => gl.getAttribLocation(program, name));
const buffers = attributes.map(() => gl.createBuffer());

function upload() {
  const fields = [
    new Float32Array(splats.flatMap(s => s.position)),
    new Float32Array(splats.flatMap(s => s.scale)),
    new Float32Array(splats.flatMap(s => s.color)),
    new Float32Array(splats.map(s => s.radiance)),
  ];
  const widths = [3, 3, 4, 1];
  fields.forEach((field, index) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[index]); gl.bufferData(gl.ARRAY_BUFFER, field, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(attributes[index]); gl.vertexAttribPointer(attributes[index], widths[index], gl.FLOAT, false, 0, 0);
  });
}
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) for (let k=0;k<4;k++) o[c*4+r] += a[k*4+r]*b[c*4+k];
  return o;
}
function view() {
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const eye=[zoom*sy*cp, zoom*sp, zoom*cy*cp], len=Math.hypot(...eye);
  const z=eye.map(v=>v/len), x=[z[2],0,-z[0]], xl=Math.hypot(...x); x.forEach((_,i)=>x[i]/=xl);
  const y=[z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]];
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, 0,0,-len,1]);
}
function draw() {
  const dpr=Math.min(2,devicePixelRatio), width=Math.floor(canvas.clientWidth*dpr), height=Math.floor(canvas.clientHeight*dpr);
  if (canvas.width!==width||canvas.height!==height) { canvas.width=width; canvas.height=height; gl.viewport(0,0,width,height); }
  gl.clearColor(.01,.025,.055,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  viewProjection=multiply(perspective(.9,width/height,.1,500),view());
  gl.uniformMatrix4fv(gl.getUniformLocation(program,'uViewProjection'),false,viewProjection);
  gl.uniform1f(gl.getUniformLocation(program,'uPointScale'), height*.34);
  gl.drawArrays(gl.POINTS,0,splats.length);
  requestAnimationFrame(draw);
}
async function loadSplats(nextLevel=level, basinId='') {
  level=nextLevel;
  const query=new URLSearchParams({level,limit:'50000'}); if (basinId) query.set('basin_id',basinId);
  const page=await fetch('/api/splats?'+query).then(r=>r.json()); splats=page.splats; upload();
}
async function loadBasins() {
  basinList=await fetch('/api/basins').then(r=>r.json());
  document.querySelector('#basins').innerHTML=basinList.map(b=>`<article class="basin" data-id="${b.id}">
    <h3>${escapeHtml(b.label)}</h3><p>${escapeHtml(b.summary||b.path)}</p>
    <p class="meta">${b.member_ids.length.toLocaleString()} memories · stability ${b.stability.toFixed(2)}</p></article>`).join('');
  document.querySelectorAll('.basin').forEach(node=>node.onclick=()=>{ loadSplats('memories',node.dataset.id); showBasin(basinList.find(b=>b.id===node.dataset.id)); });
}
function showBasin(b) {
  if (!b) return;
  document.querySelector('#detail').innerHTML=`<h2>${escapeHtml(b.label)}</h2><p>${escapeHtml(b.summary||'Label pending')}</p><p class="meta">${escapeHtml(b.path)} · ${b.member_ids.length} memories</p>`;
}
async function showMemory(id) {
  const response=await fetch('/api/memories/'+encodeURIComponent(id));
  if (!response.ok) return;
  const memory=await response.json();
  document.querySelector('#detail').innerHTML=`<h2>${escapeHtml(memory.model||memory.speaker||memory.source)}</h2>
    <p>${escapeHtml(memory.text)}</p><p class="meta">${escapeHtml(memory.domain)} · ${escapeHtml(memory.timestamp||memory.ingested_at)} · ${escapeHtml(memory.source_key)}</p>`;
}
function projected(position) {
  const [x,y,z]=position, m=viewProjection;
  const w=m[3]*x+m[7]*y+m[11]*z+m[15];
  if (w<=0) return null;
  const nx=(m[0]*x+m[4]*y+m[8]*z+m[12])/w;
  const ny=(m[1]*x+m[5]*y+m[9]*z+m[13])/w;
  const nz=(m[2]*x+m[6]*y+m[10]*z+m[14])/w;
  if (Math.abs(nx)>1.1||Math.abs(ny)>1.1||Math.abs(nz)>1.1) return null;
  return [(nx*.5+.5)*canvas.clientWidth,(.5-ny*.5)*canvas.clientHeight,nz];
}
function inspectNearest(event) {
  let nearest=null, nearestDistance=28;
  const rect=canvas.getBoundingClientRect(), cursorX=event.clientX-rect.left, cursorY=event.clientY-rect.top;
  for (const splat of splats) {
    const point=projected(splat.position); if(!point)continue;
    const distance=Math.hypot(point[0]-cursorX,point[1]-cursorY);
    if(distance<nearestDistance){nearest=splat;nearestDistance=distance}
  }
  if(!nearest)return;
  if(level==='basins')showBasin(basinList.find(b=>b.id===nearest.id));
  else showMemory(nearest.id);
}
document.querySelector('#search-form').onsubmit=async event=>{
  event.preventDefault(); const q=document.querySelector('#search').value.trim(); if(!q)return;
  const hits=await fetch('/api/search?'+new URLSearchParams({q,limit:'10'})).then(r=>r.json());
  document.querySelector('#detail').innerHTML=`<h2>Recall</h2>`+hits.map(h=>`<div class="hit"><h3>${escapeHtml(h.basin_label||h.memory.source)}</h3>
    <p>${escapeHtml(h.memory.text.slice(0,900))}</p><p class="meta">score ${h.scores.final_score.toFixed(3)} · ${escapeHtml(h.memory.model||h.memory.speaker||'memory')}</p></div>`).join('');
};
document.querySelector('#basin-view').onclick=()=>{ setButtons(true); loadSplats('basins'); };
document.querySelector('#memory-view').onclick=()=>{ setButtons(false); loadSplats('memories'); };
function setButtons(basins){document.querySelector('#basin-view').classList.toggle('active',basins);document.querySelector('#memory-view').classList.toggle('active',!basins);}
canvas.onpointerdown=e=>{dragging=true;moved=false;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!dragging)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(Math.hypot(dx,dy)>2)moved=true;yaw+=dx*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+dy*.008));last=[e.clientX,e.clientY]};
canvas.onpointerup=e=>{dragging=false;if(!moved)inspectNearest(e)};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(4,Math.min(120,zoom*Math.exp(e.deltaY*.001)))};
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
Promise.all([loadSplats(),loadBasins(),fetch('/api/status').then(r=>r.json())]).then(([, ,s])=>{
  document.querySelector('#status').textContent=`${s.cold_records.toLocaleString()} memories · ${s.basins} basins · dream ${s.dream_cycle}`;
}).catch(error=>document.querySelector('#status').textContent=error.message);
draw();
