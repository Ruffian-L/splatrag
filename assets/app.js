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

let splats = [], basinList = [], level = 'memories', yaw = .55, pitch = -.24, zoom = 25;
let dragging = false, moved = false, last = [0, 0], viewProjection = new Float32Array(16);
let liveStep = 0, liveKe = 0, liveDreaming = false;
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
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[index]); gl.bufferData(gl.ARRAY_BUFFER, field, gl.DYNAMIC_DRAW);
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
function setStatus(extra='') {
  const el = document.querySelector('#status');
  const dream = liveDreaming ? ` · DREAM step ${liveStep} · KE ${liveKe.toFixed(2)}` : '';
  el.textContent = (el.dataset.base || 'field') + dream + (extra ? ' · ' + extra : '');
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
const ARM_COLOR = {
  both: [120, 220, 255, 240],
  bm25_only: [255, 200, 80, 240],
  dense_only: [180, 120, 255, 240],
  fusion_only: [255, 90, 120, 240],
  gold_hit: [80, 255, 140, 255],
  gold_miss: [255, 60, 60, 255],
};

function highlightFlip(report) {
  // Paint hybrid hits + gold geometry on the field when memory view is live.
  if (!report || !report.hybrid) return;
  const byId = new Map(splats.map(s => [s.id, s]));
  for (const h of report.hybrid) {
    const s = byId.get(h.memory_id);
    const col = ARM_COLOR[h.arm] || ARM_COLOR.both;
    if (s) {
      s.color = col.slice();
      s.radiance = 6;
      s.scale = [2.2, 1.2, 1.2];
    } else if (h.geometry && h.geometry.position) {
      splats.push({
        id: h.memory_id,
        position: h.geometry.position,
        scale: [2.2, 1.2, 1.2],
        color: col.slice(),
        radiance: 6,
      });
    }
  }
  for (const g of report.golds || []) {
    if (!g.geometry || !g.geometry.position) continue;
    const mid = g.geometry.memory_id;
    let s = byId.get(mid) || splats.find(x => x.id === mid);
    const col = g.class === 'hit' ? ARM_COLOR.gold_hit : ARM_COLOR.gold_miss;
    if (s) {
      s.color = col.slice();
      s.radiance = 8;
      s.scale = [2.8, 1.4, 1.4];
    } else {
      splats.push({
        id: mid,
        position: g.geometry.position,
        scale: [2.8, 1.4, 1.4],
        color: col.slice(),
        radiance: 8,
      });
    }
  }
  upload();
}

function renderFlip(report) {
  const goldBlock = (report.golds || []).length
    ? `<h3>Gold</h3>${report.golds.map(g =>
        `<div class="hit"><p class="meta">${escapeHtml(g.class)} · doc ${escapeHtml(g.doc_id)} · bm25=${g.in_bm25} dense=${g.in_dense} rank=${g.hybrid_rank ?? '—'}</p></div>`
      ).join('')}`
    : '';
  document.querySelector('#detail').innerHTML = `<h2>Flip / arms</h2>
    <p class="meta">bm25 pool ${report.bm25_pool} · dense pool ${report.dense_pool} · k=${report.k} cand=${report.candidate_limit}</p>
    <p class="meta"><span class="swatch both"></span>both <span class="swatch bm25"></span>bm25 <span class="swatch dense"></span>dense <span class="swatch fusion"></span>fusion</p>
    ${report.hybrid.map(h => `<div class="hit">
      <h3>#${h.rank} <span class="arm ${escapeHtml(h.arm)}">${escapeHtml(h.arm)}</span></h3>
      <p>${escapeHtml(h.text_preview)}</p>
      <p class="meta">final ${h.final_score.toFixed(3)} · bm25 ${h.bm25.toFixed(3)} · cos ${h.cosine.toFixed(3)} · ${escapeHtml(h.source_record_id||h.memory_id)}</p>
    </div>`).join('')}
    ${goldBlock}`;
  highlightFlip(report);
}

function renderJlens(report) {
  const bridgeEl = document.querySelector('#jlens-bridge');
  const wordsEl = document.querySelector('#jlens-words');
  const metaEl = document.querySelector('#jlens-meta');
  if (!bridgeEl || !wordsEl) return;
  if (!report || report.ok === false) {
    bridgeEl.className = 'jlens-bridge';
    bridgeEl.innerHTML = `<span class="jlens-error">${escapeHtml(report?.error || 'jlens unavailable')}</span>`;
    wordsEl.innerHTML = '';
    metaEl.textContent = report?.raw_stderr_tail ? String(report.raw_stderr_tail).slice(0, 280) : '';
    return;
  }
  const tokens = report.top_tokens || [];
  bridgeEl.className = 'jlens-bridge';
  bridgeEl.textContent = report.text_bridge || tokens.map(t => t.text).join(' ') || '(empty bridge)';
  wordsEl.innerHTML = tokens.map((t, i) =>
    `<span class="jlens-chip${i===0?' first':''}" title="id ${t.id}">${escapeHtml(t.text)}<span class="logit">${Number(t.logit).toFixed(2)}</span></span>`
  ).join('');
  const bits = [
    report.layer != null ? `L${report.layer}` : null,
    report.position != null ? `pos ${report.position}` : null,
    report.text_bridge_hash != null ? `hash ${report.text_bridge_hash}` : null,
    report.model_path ? report.model_path.split('/').pop() : null,
  ].filter(Boolean);
  metaEl.textContent = bits.join(' · ') + (report.note ? ` · ${report.note}` : '');
}

async function fetchJlens(q) {
  document.querySelector('#jlens-bridge').className = 'jlens-bridge quiet';
  document.querySelector('#jlens-bridge').textContent = 'first thought: loading model + residual → unembed…';
  document.querySelector('#jlens-words').innerHTML = '';
  document.querySelector('#jlens-meta').textContent = 'CPU GGUF load — often 30–90s first time. Not SplatLens hybrid.';
  try {
    const report = await fetch('/api/jlens?' + new URLSearchParams({ q, top_k: '8' })).then(async r => {
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      return j;
    });
    renderJlens(report);
    return report;
  } catch (e) {
    renderJlens({ ok: false, error: String(e) });
    return null;
  }
}

document.querySelector('#search-form').onsubmit=async event=>{
  event.preventDefault(); const q=document.querySelector('#search').value.trim(); if(!q)return;
  // SplatLens hybrid only (field paint). jlens is the orange panel button — first thought, not recall.
  setStatus('hybrid recall…');
  try {
    const flip = await fetch('/api/flip?'+new URLSearchParams({q,k:'10',cand:'80'})).then(r=>r.json());
    if (flip.error) {
      document.querySelector('#detail').innerHTML=`<p class="meta">${escapeHtml(flip.error)}</p>`;
      setStatus('flip failed');
      return;
    }
    renderFlip(flip);
    setStatus(`hybrid top ${flip.hybrid?.length||0}`);
  } catch (e) {
    document.querySelector('#detail').innerHTML=`<p class="meta">${escapeHtml(String(e))}</p>`;
    setStatus('flip error');
  }
};

document.querySelector('#jlens-btn')?.addEventListener('click', async () => {
  const q = document.querySelector('#search').value.trim();
  if (!q) {
    renderJlens({ ok: false, error: 'type something in the search box first (text to *read*)' });
    return;
  }
  setStatus('jlens first thought…');
  const jlens = await fetchJlens(q);
  if (jlens && jlens.ok) {
    const first = (jlens.top_tokens && jlens.top_tokens[0] && jlens.top_tokens[0].text) || jlens.text_bridge;
    setStatus(`first thought: ${first}`);
  } else {
    setStatus('jlens failed — see orange panel');
  }
});
document.querySelector('#basin-view').onclick=()=>{ setButtons(true); loadSplats('basins'); };
document.querySelector('#memory-view').onclick=()=>{ setButtons(false); loadSplats('memories'); };
function setButtons(basins){document.querySelector('#basin-view').classList.toggle('active',basins);document.querySelector('#memory-view').classList.toggle('active',!basins);}
canvas.onpointerdown=e=>{dragging=true;moved=false;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!dragging)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(Math.hypot(dx,dy)>2)moved=true;yaw+=dx*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+dy*.008));last=[e.clientX,e.clientY]};
canvas.onpointerup=e=>{dragging=false;if(!moved)inspectNearest(e)};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(4,Math.min(120,zoom*Math.exp(e.deltaY*.001)))};
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}

/** Apply a live frame (positions + RGBA) without a full HTTP splat reload. */
function applyLiveFrame(frame) {
  liveStep = frame.step || 0;
  liveKe = frame.kinetic_energy || 0;
  liveDreaming = !!frame.dreaming && !frame.done;
  const n = frame.n || 0;
  if (!n || !frame.positions || frame.positions.length < n * 3) {
    setStatus(frame.done ? 'dream done' : '');
    if (frame.done) loadBasins().catch(()=>{});
    return;
  }
  // Rebuild splat list if count changed or first live frame
  if (splats.length !== n || level !== 'memories') {
    level = 'memories';
    setButtons(false);
    splats = [];
    for (let i = 0; i < n; i++) {
      splats.push({
        id: String(i),
        position: [0,0,0],
        scale: [1.4, 0.9, 0.9],
        color: [180, 180, 255, 220],
        radiance: 2.0,
      });
    }
  }
  for (let i = 0; i < n; i++) {
    const p = i * 3, c = i * 4;
    splats[i].position = [frame.positions[p], frame.positions[p+1], frame.positions[p+2]];
    if (frame.colors && frame.colors.length >= c + 4) {
      splats[i].color = [frame.colors[c], frame.colors[c+1], frame.colors[c+2], frame.colors[c+3]];
    }
  }
  upload();
  setStatus(frame.done ? `settled · ${frame.basins} basins` : `${n} stars`);
  if (frame.done) loadBasins().catch(()=>{});
}

function connectLive() {
  // relative URL so 127.0.0.1 and localhost both work — avoid opening http://0.0.0.0:8768
  const es = new EventSource('/api/live');
  let fails = 0;
  es.addEventListener('frame', (ev) => {
    fails = 0;
    try { applyLiveFrame(JSON.parse(ev.data)); } catch (e) { console.warn(e); }
  });
  es.onerror = () => {
    fails += 1;
    // don't spam status forever; field still works without live dream stream
    if (fails === 1 || fails % 10 === 0) {
      const base = document.querySelector('#status')?.dataset?.base || '';
      if (document.querySelector('#status')) {
        document.querySelector('#status').textContent =
          (base ? base + ' · ' : '') + 'live SSE idle (ok unless dreaming)';
      }
    }
  };
  return es;
}

async function startDream() {
  const res = await fetch('/api/dream', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({ emit_every: 3 }),
  }).then(r => r.json());
  if (!res.ok) {
    setStatus(res.error || 'dream refused');
    return;
  }
  liveDreaming = true;
  setStatus('dream started');
  setButtons(false);
  level = 'memories';
}

// legend + dream + miss-classify
(function injectControls() {
  const controls = document.querySelector('.controls');
  if (!controls) return;
  const legend = document.createElement('span');
  legend.className = 'legend';
  legend.innerHTML = `<span class="swatch scifact"></span>SciFact <span class="swatch urban"></span>Urban <span class="swatch hot"></span>moving`;
  const dreamBtn = document.createElement('button');
  dreamBtn.id = 'dream-btn';
  dreamBtn.textContent = 'Dream live';
  dreamBtn.onclick = () => startDream().catch(e => setStatus(String(e)));
  const missBtn = document.createElement('button');
  missBtn.id = 'miss-btn';
  missBtn.textContent = 'Miss class';
  missBtn.title = 'SciFact BM25/dense/rank miss taxonomy (summary)';
  missBtn.onclick = () => runMissClassify().catch(e => setStatus(String(e)));
  controls.appendChild(legend);
  controls.appendChild(dreamBtn);
  controls.appendChild(missBtn);
})();

async function runMissClassify() {
  setStatus('miss-classify…');
  document.querySelector('#detail').innerHTML =
    `<h2>Miss classify</h2><p class="meta">running (viz uses limit=40 for a quick sample; full 300 is CLI). Needs embedder on :8081.</p>`;
  const body = {
    dataset: '/media/ruffianl/ghost_team/splatrag-runtime/datasets/scifact',
    k: 10,
    cand: 80,
    limit: 40,
    full: false,
  };
  try {
    const res = await fetch('/api/miss-classify', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(body),
    }).then(async r => {
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      return j;
    });
    if (res.error) {
      document.querySelector('#detail').innerHTML = `<h2>Miss classify</h2><p class="meta jlens-error">${escapeHtml(res.error)}</p>`;
      setStatus('miss-classify failed');
      return;
    }
    const gc = res.gold_counts || {};
    const pc = res.query_primary_counts || {};
    document.querySelector('#detail').innerHTML = `<h2>Miss classify</h2>
      <p class="meta">queries ${res.queries} · failing ${res.failing_queries} · k=${res.k} cand=${res.candidate_limit} · sample limit 40</p>
      <h3>Gold docs</h3>
      <p class="meta">hit ${gc.hit_golds||0} · bm25_miss ${gc.bm25_miss||0} · dense_miss ${gc.dense_miss||0} · both_miss ${gc.both_miss||0} · rank_miss ${gc.rank_miss||0}</p>
      <p class="meta">bm25_only_pool ${gc.bm25_only_pool||0} · dense_only_pool ${gc.dense_only_pool||0} · both_pool_rank_miss ${gc.both_pool_rank_miss||0}</p>
      <h3>Query primary</h3>
      <p class="meta">${Object.entries(pc).map(([k,v])=>`${k}:${v}`).join(' · ') || '(none)'}</p>
      <h3>Sample fails</h3>
      ${(res.sample_failing||[]).length ? (res.sample_failing||[]).map(r => `<div class="hit">
        <h3>${escapeHtml(String(r.primary_class))} · R@10 ${Number(r.recall_at_k).toFixed(2)}</h3>
        <p>${escapeHtml(r.query_text||'')}</p>
        <p class="meta">${(r.golds||[]).map(g=>`${g.doc_id}:${g.class}`).join(' · ')}</p>
      </div>`).join('') : '<p class="meta">no failing queries in this sample</p>'}`;
    setStatus(`miss-class failing ${res.failing_queries}/${res.queries}`);
  } catch (e) {
    document.querySelector('#detail').innerHTML =
      `<h2>Miss classify</h2><p class="meta jlens-error">${escapeHtml(String(e))}</p>
       <p class="meta">If embedder is down, start llama-server on 8081. Full run: CLI miss-classify --out …</p>`;
    setStatus('miss-class error');
  }
}

Promise.all([loadSplats('memories'), loadBasins(), fetch('/api/status').then(r=>r.json())]).then(([, ,s])=>{
  const base = `${s.cold_records.toLocaleString()} memories · ${s.basins} basins · dream ${s.dream_cycle}`;
  document.querySelector('#status').dataset.base = base;
  document.querySelector('#status').textContent = base;
  connectLive();
  // Dream only on button — auto-dream locked the field and broke SSE feel.
}).catch(error=>document.querySelector('#status').textContent=error.message);
draw();
