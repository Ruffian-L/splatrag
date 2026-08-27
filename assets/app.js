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

let splats = [], basinList = [], level = 'memories', yaw = .55, pitch = -.24, zoom = 32;
let dragging = false, moved = false, last = [0, 0], viewProjection = new Float32Array(16);
let liveStep = 0, liveKe = 0, liveDreaming = false;
/** Selfish highlight: clicked splat + recall hits light up so you can see *which* star. */
let selectedId = null;
let hitIds = new Set();
let hitRank = new Map();
const attributes = ['aPosition', 'aScale', 'aColor', 'aRadiance'].map(name => gl.getAttribLocation(program, name));
const buffers = attributes.map(() => gl.createBuffer());

function stampBases(list) {
  for (const s of list) {
    if (!s.color) s.color = [180, 180, 220, 200];
    if (!s.scale) s.scale = [1.2, 0.7, 0.7];
    if (s.radiance == null) s.radiance = 1.5;
    s.baseColor = s.color.slice();
    s.baseScale = s.scale.slice();
    s.baseRadiance = s.radiance;
  }
}

function restoreBases() {
  for (const s of splats) {
    if (!s.baseColor) continue;
    s.color = s.baseColor.slice();
    s.scale = s.baseScale.slice();
    s.radiance = s.baseRadiance;
  }
}

function applyHighlight() {
  // Never let a bad splat poison the whole recall path — stamp missing bases first.
  for (const s of splats) {
    if (!s.baseColor || !s.baseScale) stampBases([s]);
  }
  restoreBases();
  for (const s of splats) {
    const baseScale = s.baseScale || [1.2, 0.7, 0.7];
    if (hitIds.has(s.id)) {
      const rank = hitRank.get(s.id) ?? 9;
      const t = Math.max(0, 1 - rank * 0.09);
      // Warm gold → soft amber by rank. Bright enough to punch through basin soup.
      s.color = [
        Math.round(255),
        Math.round(210 + 40 * t),
        Math.round(70 + 50 * (1 - t)),
        255,
      ];
      const grow = 2.4 - Math.min(rank, 8) * 0.12;
      s.scale = [
        Math.max(baseScale[0] * grow, 2.0),
        Math.max(baseScale[1] * 1.7, 1.1),
        Math.max(baseScale[2] * 1.7, 1.1),
      ];
      s.radiance = 9 + (10 - Math.min(rank, 9));
    }
    if (s.id === selectedId) {
      // Primary pick: white core, biggest, hottest — "that's the one."
      const cur = s.scale || baseScale;
      s.color = [255, 255, 255, 255];
      s.scale = [
        Math.max(cur[0], baseScale[0] * 3.1, 3.0),
        Math.max(cur[1], baseScale[1] * 2.0, 1.6),
        Math.max(cur[2], baseScale[2] * 2.0, 1.6),
      ];
      s.radiance = 16;
    }
  }
  upload();
}

function focusOnId(id) {
  const s = splats.find((x) => x.id === id);
  if (!s || !s.position) return;
  const [x, y, z] = s.position;
  const d = Math.hypot(x, y, z) || 1;
  zoom = Math.max(6, Math.min(48, d * 2.4 + 7));
  yaw = Math.atan2(x, z);
  pitch = Math.asin(Math.max(-0.85, Math.min(0.85, y / d)));
}

function setSelection(id, { focus = false } = {}) {
  selectedId = id || null;
  try {
    applyHighlight();
  } catch (e) {
    console.warn('highlight failed', e);
  }
  if (focus && id) {
    try { focusOnId(id); } catch (e) { console.warn('focus failed', e); }
  }
}

function setHits(ids, { selectTop = true, focus = true } = {}) {
  const list = (ids || []).filter(Boolean);
  hitIds = new Set(list);
  hitRank = new Map(list.map((id, i) => [id, i]));
  if (selectTop && list.length) selectedId = list[0];
  try {
    applyHighlight();
  } catch (e) {
    console.warn('highlight failed', e);
    setStatus(`hits ready · field paint failed: ${e.message || e}`);
  }
  if (focus && selectedId) {
    try { focusOnId(selectedId); } catch (e) { console.warn('focus failed', e); }
  }
}

function clearHits() {
  hitIds = new Set();
  hitRank = new Map();
  try { applyHighlight(); } catch (e) { console.warn('highlight failed', e); }
}

function upload() {
  const n = splats.length;
  const pos = new Float32Array(n * 3);
  const scl = new Float32Array(n * 3);
  const col = new Float32Array(n * 4);
  const rad = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = splats[i];
    const p = s.position || [0, 0, 0];
    const sc = s.scale || [1, 1, 1];
    const c = s.color || [180, 180, 220, 200];
    pos[i * 3] = +p[0] || 0; pos[i * 3 + 1] = +p[1] || 0; pos[i * 3 + 2] = +p[2] || 0;
    scl[i * 3] = +sc[0] || 1; scl[i * 3 + 1] = +sc[1] || 1; scl[i * 3 + 2] = +sc[2] || 1;
    col[i * 4] = +c[0] || 0; col[i * 4 + 1] = +c[1] || 0; col[i * 4 + 2] = +c[2] || 0; col[i * 4 + 3] = +c[3] || 200;
    rad[i] = Number.isFinite(+s.radiance) ? +s.radiance : 1.5;
  }
  const fields = [pos, scl, col, rad];
  const widths = [3, 3, 4, 1];
  fields.forEach((field, index) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[index]); gl.bufferData(gl.ARRAY_BUFFER, field, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(attributes[index]); gl.vertexAttribPointer(attributes[index], widths[index], gl.FLOAT, false, 0, 0);
  });
}

/** Live dream frames only move/recolor — skip scale/radiance rebuild so 100k+ stays fluid. */
function uploadLive() {
  const n = splats.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const s = splats[i];
    const p = s.position || [0, 0, 0];
    const c = s.color || [180, 180, 220, 200];
    pos[i * 3] = +p[0] || 0; pos[i * 3 + 1] = +p[1] || 0; pos[i * 3 + 2] = +p[2] || 0;
    col[i * 4] = +c[0] || 0; col[i * 4 + 1] = +c[1] || 0; col[i * 4 + 2] = +c[2] || 0; col[i * 4 + 3] = +c[3] || 200;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers[0]);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers[2]);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
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
  level = nextLevel || level;
  // Personal store can be 60k+; old 50k cap silently dropped the tail and broke id↔live alignment.
  const query = new URLSearchParams({ level, limit: '200000' });
  if (basinId) query.set('basin_id', basinId);
  setStatus(`loading ${level}…`);
  const page = await fetch('/api/splats?' + query).then(r => r.json());
  if (page.error) throw new Error(page.error);
  splats = page.splats || [];
  stampBases(splats);
  // Keep selection/hits if those ids still exist on this page.
  if (selectedId && !splats.some((s) => s.id === selectedId)) selectedId = null;
  hitIds = new Set([...hitIds].filter((id) => splats.some((s) => s.id === id)));
  applyHighlight();
  const total = page.total != null ? page.total : splats.length;
  const trunc = total > splats.length ? ` · showing ${splats.length.toLocaleString()}/${total.toLocaleString()}` : '';
  setStatus(`${splats.length.toLocaleString()} points on field (${level})${trunc}`);
  return page;
}
async function loadBasins() {
  basinList=await fetch('/api/basins').then(r=>r.json());
  document.querySelector('#basins').innerHTML=basinList.map(b=>`<article class="basin" data-id="${b.id}">
    <h3>${escapeHtml(b.label)}</h3><p>${escapeHtml(b.summary||b.path)}</p>
    <p class="meta">${b.member_ids.length.toLocaleString()} memories · stability ${b.stability.toFixed(2)}</p></article>`).join('');
  document.querySelectorAll('.basin').forEach(node=>node.onclick=()=>{
    clearHits();
    setSelection(node.dataset.id, { focus: true });
    loadSplats('memories', node.dataset.id);
    showBasin(basinList.find(b=>b.id===node.dataset.id));
  });
}
function showBasin(b) {
  if (!b) return;
  document.querySelector('#detail').innerHTML=`<h2>${escapeHtml(b.label)}</h2><p>${escapeHtml(b.summary||'Label pending')}</p><p class="meta">${escapeHtml(b.path)} · ${b.member_ids.length} memories</p>`;
}
function isRealMemoryId(id) {
  // Live-frame bug used to invent "0","1",… — those are not fetchable UUIDs.
  return typeof id === 'string' && id.length >= 32 && id.includes('-');
}

async function showMemory(id, { fromHit = false } = {}) {
  if (!id) return;
  if (!isRealMemoryId(id)) {
    document.querySelector('#detail').innerHTML =
      `<h2>Can't open splat</h2>
       <p class="meta jlens-error">This point has no real memory id (got <code>${escapeHtml(String(id))}</code>).
       The field list is out of sync — click <strong>Memory splats</strong> or wait for dream to finish reloading.</p>`;
    return;
  }
  const response = await fetch('/api/memories/' + encodeURIComponent(id));
  if (!response.ok) {
    let err = `HTTP ${response.status}`;
    try { const j = await response.json(); if (j && j.error) err = j.error; } catch (_) {}
    document.querySelector('#detail').innerHTML =
      `<h2>Memory not found</h2>
       <p class="meta jlens-error">${escapeHtml(err)}</p>
       <p class="meta">id ${escapeHtml(id)}</p>`;
    return;
  }
  const memory = await response.json();
  const onField = splats.some((s) => s.id === id);
  const basin = memory.basin_id
    ? (basinList.find((b) => b.id === memory.basin_id) || null)
    : (splats.find((s) => s.id === id)?.basin_id
        ? basinList.find((b) => b.id === splats.find((s) => s.id === id).basin_id)
        : null);
  const fieldSplat = splats.find((s) => s.id === id);
  const basinLabel = fieldSplat?.label || basin?.label || '';
  document.querySelector('#detail').innerHTML = `<h2>${escapeHtml(memory.model || memory.speaker || memory.source || 'memory')}</h2>
    <p>${escapeHtml(memory.text)}</p>
    <p class="meta">${escapeHtml(memory.source || '')} · ${escapeHtml(memory.timestamp || memory.ingested_at || '')} · ${escapeHtml(memory.source_key || '')}</p>
    <p class="meta">${onField ? '● lit on field' : '○ not in current field page'}
      ${basinLabel ? ' · basin ' + escapeHtml(basinLabel) : ''}
      · domain ${escapeHtml(memory.domain || '—')}</p>`;
  if (!fromHit) setSelection(id, { focus: true });
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
  // Dense 60k fields: prefer nearest in screen space, slightly larger hit radius.
  let nearest = null, nearestDistance = 36;
  const rect = canvas.getBoundingClientRect();
  const cursorX = event.clientX - rect.left, cursorY = event.clientY - rect.top;
  for (const splat of splats) {
    const point = projected(splat.position); if (!point) continue;
    const distance = Math.hypot(point[0] - cursorX, point[1] - cursorY);
    if (distance < nearestDistance) { nearest = splat; nearestDistance = distance; }
  }
  if (!nearest) {
    setStatus('no splat under cursor — zoom in or click a denser clump');
    return;
  }
  setSelection(nearest.id, { focus: true });
  if (level === 'basins') {
    showBasin(basinList.find((b) => b.id === nearest.id));
  } else {
    showMemory(nearest.id).catch((e) => console.warn(e));
  }
}

function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function renderRecall(query, hits) {
  const list = Array.isArray(hits) ? hits : [];
  const ids = list.map((h) => h && h.memory && h.memory.id).filter(Boolean);
  const onField = ids.filter((id) => splats.some((s) => s.id === id)).length;
  // Panel first — always. Field paint is optional and must not kill recall.
  const rows = list.map((h, i) => {
    const m = (h && h.memory) || {};
    const id = m.id || '';
    const sc = (h && h.scores) || {};
    const lit = id && splats.some((s) => s.id === id);
    const text = String(m.text || '');
    const title = m.model || m.speaker || m.source || 'memory';
    const basin = h && h.basin_label ? ` · ${escapeHtml(String(h.basin_label))}` : '';
    return `<div class="hit recall-hit${i === 0 ? ' top' : ''}" data-id="${escapeHtml(id)}" role="button" tabindex="0">
        <h3>#${i + 1} ${escapeHtml(String(title))}
          ${lit ? '<span class="arm both">on field</span>' : '<span class="arm">off page</span>'}</h3>
        <p>${escapeHtml(text.slice(0, 420))}${text.length > 420 ? '…' : ''}</p>
        <p class="meta">score ${num(sc.final_score).toFixed(3)} · bm25 ${num(sc.bm25).toFixed(2)} · cos ${num(sc.cosine).toFixed(2)}${basin}</p>
      </div>`;
  }).join('');
  document.querySelector('#detail').innerHTML = `<h2>Recall</h2>
    <p class="meta">${escapeHtml(String(query))} · ${list.length} hits · ${onField} lit on field
      <span class="swatch hit"></span>hits <span class="swatch selected"></span>selected</p>
    ${list.length ? rows : '<p class="meta">No hits — try a shorter phrase or a name you used.</p>'}`;
  document.querySelectorAll('.recall-hit').forEach((node) => {
    const go = () => {
      const id = node.dataset.id;
      if (!id) return;
      setSelection(id, { focus: true });
      showMemory(id, { fromHit: true }).catch((e) => console.warn(e));
    };
    node.onclick = go;
    node.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
  });
  const paint = () => {
    try {
      setHits(ids, { selectTop: true, focus: true });
    } catch (e) {
      console.warn('setHits failed', e);
      setStatus(`recall · ${list.length} hits · field paint failed`);
    }
  };
  // Memory points (not basin orbs) so hits can light.
  if (level === 'basins') {
    loadSplats('memories').then(paint).catch((e) => {
      console.warn('loadSplats after recall failed', e);
      setStatus(`recall · ${list.length} hits · field reload failed`);
    });
  } else {
    paint();
  }
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
  document.querySelector('#jlens-meta').textContent = 'CPU GGUF load — often 30–90s first time. Not hybrid recall.';
  try {
    const report = await fetch('/api/jlens?' + new URLSearchParams({ q, top_k: '8' })).then(async r => {
      const j = await r.json();
      // jlens returns ok:false with 200 when binary/model missing — treat as failure.
      if (!r.ok || j.ok === false) {
        throw new Error(j.error || j.note || r.statusText || 'jlens failed');
      }
      return j;
    });
    renderJlens(report);
    return report;
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    renderJlens({
      ok: false,
      error: msg,
      raw_stderr_tail: /jlens binary missing|model missing/i.test(msg)
        ? 'JLens is a separate binary + GGUF. Not wired = no first-thought words. Recall still works.'
        : '',
    });
    return { ok: false, error: msg };
  }
}

document.querySelector('#search-form').onsubmit=async event=>{
  event.preventDefault();
  const q = document.querySelector('#search').value.trim();
  if (!q) return;
  // Personal hybrid recall — paint hits on the field. jlens stays its own orange button.
  setStatus('recalling…');
  try {
    const res = await fetch('/api/search?' + new URLSearchParams({ q, limit: '12' }));
    const text = await res.text();
    let hits;
    try {
      hits = JSON.parse(text);
    } catch (parseErr) {
      throw new Error(`bad JSON (${res.status}): ${text.slice(0, 180)}`);
    }
    if (!res.ok) {
      throw new Error((hits && hits.error) || res.statusText || `HTTP ${res.status}`);
    }
    if (hits && hits.error) {
      document.querySelector('#detail').innerHTML = `<p class="meta">${escapeHtml(hits.error)}</p>`;
      setStatus('recall failed');
      return;
    }
    if (!Array.isArray(hits)) {
      throw new Error('search returned non-array: ' + typeof hits);
    }
    renderRecall(q, hits);
    setStatus(`recall · ${hits.length} hits · top lit white`);
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    document.querySelector('#detail').innerHTML =
      `<h2>Recall</h2><p class="meta jlens-error">${escapeHtml(msg)}</p>
       <p class="meta">Store search itself is fine for “miyu” from the API — if you see this, hard-refresh (Ctrl+Shift+R) or tell me this exact message.</p>`;
    setStatus('recall error: ' + msg);
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

/** True when the field list has real memory UUIDs (not invent-on-live index ids). */
function fieldIdsOk(n) {
  if (!splats.length || (n != null && splats.length !== n)) return false;
  // Spot-check ends — full scan on 60k is fine but wasteful every frame.
  const a = splats[0]?.id, b = splats[splats.length - 1]?.id;
  return isRealMemoryId(a) && isRealMemoryId(b);
}

function setDreamUi(dreaming, label) {
  const btn = document.querySelector('#dream-btn');
  if (!btn) return;
  btn.disabled = !!dreaming;
  btn.textContent = dreaming
    ? (label || `Dreaming… step ${liveStep}`)
    : 'Dream live';
  btn.title = dreaming
    ? 'One dream at a time. Each dream continues from current positions, re-discovers basins, increments dream_cycle.'
    : 'Run another physics pass. Second+ dreams keep settled geometry and refine; refuse if already running.';
}

let liveReloadPromise = null;
let livePendingFrame = null;

/** Apply a live frame (positions + RGBA) without destroying memory UUIDs. */
function applyLiveFrame(frame) {
  liveStep = frame.step || 0;
  liveKe = frame.kinetic_energy || 0;
  const n = frame.n || 0;
  const dreaming = !!frame.dreaming && !frame.done;
  liveDreaming = dreaming;

  // Idle SSE seed on connect used to rebuild the whole field with id="0","1",…
  // and break click → /api/memories. Ignore non-dream frames that don't match.
  if (!dreaming && !frame.done) {
    if (frame.dream_cycle != null || frame.kinetic_energy != null) {
      // Status-only tick — keep real UUIDs on the canvas.
      return;
    }
  }

  if (frame.done) {
    liveDreaming = false;
    setDreamUi(false);
    // Always re-fetch real ids + basin tints after a dream settles.
    Promise.all([loadSplats('memories'), loadBasins()])
      .then(() => {
        const cyc = frame.dream_cycle != null ? ` · cycle ${frame.dream_cycle}` : '';
        setStatus(`settled · ${frame.basins ?? '—'} basins${cyc}`);
        const base = document.querySelector('#status');
        if (base && base.dataset) {
          fetch('/api/status').then((r) => r.json()).then((s) => {
            const nn = splats.length;
            base.dataset.base =
              `${(s.cold_records ?? nn).toLocaleString()} memories · ${nn.toLocaleString()} on field · ${s.basins} basins · dream ${s.dream_cycle}`;
          }).catch(() => {});
        }
      })
      .catch((e) => setStatus('post-dream reload failed: ' + (e.message || e)));
    return;
  }

  if (!n || !frame.positions || frame.positions.length < n * 3) {
    if (dreaming) setDreamUi(true, `Dreaming… step ${liveStep}`);
    return;
  }

  // Never invent index ids. If count/ids are wrong, reload the real list once, then re-apply.
  if (!fieldIdsOk(n) || level !== 'memories') {
    livePendingFrame = frame;
    if (!liveReloadPromise) {
      setStatus(`syncing field for dream (${n} stars)…`);
      liveReloadPromise = loadSplats('memories')
        .then(() => {
          level = 'memories';
          setButtons(false);
        })
        .catch((e) => console.warn('live reload failed', e))
        .finally(() => {
          liveReloadPromise = null;
          const pending = livePendingFrame;
          livePendingFrame = null;
          if (pending) applyLiveFrame(pending);
        });
    }
    return;
  }

  for (let i = 0; i < n; i++) {
    const p = i * 3, c = i * 4;
    splats[i].position = [frame.positions[p], frame.positions[p + 1], frame.positions[p + 2]];
    if (frame.colors && frame.colors.length >= c + 4) {
      const col = [frame.colors[c], frame.colors[c + 1], frame.colors[c + 2], frame.colors[c + 3]];
      splats[i].color = col;
      splats[i].baseColor = col.slice();
    }
  }
  // Live frames rewrite color/pos — re-apply selection so your pick stays visible mid-dream.
  if (selectedId || hitIds.size) applyHighlight();
  else uploadLive();
  setDreamUi(true, `Dreaming… step ${liveStep}`);
  setStatus(`${n.toLocaleString()} stars · KE ${liveKe.toExponential(2)}`);
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
      if (document.querySelector('#status') && !liveDreaming) {
        document.querySelector('#status').textContent =
          (base ? base + ' · ' : '') + 'live SSE idle (ok unless dreaming)';
      }
    }
  };
  return es;
}

async function startDream() {
  if (liveDreaming) {
    setStatus('dream already running — wait for settle');
    return;
  }
  // Ensure full UUID-backed field before physics frames stream in.
  if (!fieldIdsOk(null) || level !== 'memories') {
    setStatus('loading full field before dream…');
    await loadSplats('memories');
  }
  // Full-field JSON frames are ~few MB at 100k+. Every other step still reads as live
  // motion without saturating SSE. Small stores keep every step.
  const emitEvery = splats.length > 20000 ? 2 : 1;
  const res = await fetch('/api/dream', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ emit_every: emitEvery }),
  }).then((r) => r.json());
  if (res.error || res.ok === false) {
    setStatus(res.error || 'dream refused');
    setDreamUi(false);
    return;
  }
  liveDreaming = true;
  setButtons(false);
  level = 'memories';
  setDreamUi(true, 'Dreaming… starting');
  const cyc = res.dream_cycle != null ? ` · will be cycle ${res.dream_cycle}` : '';
  setStatus(`dream live · ${splats.length.toLocaleString()} stars · watch the field${cyc}`);
}

// Personal controls only — no SciFact / Urban / miss-class eval chrome.
(function injectControls() {
  const controls = document.querySelector('.controls');
  if (!controls) return;
  const legend = document.createElement('span');
  legend.className = 'legend';
  legend.innerHTML =
    `<span class="swatch basin"></span>basin color ` +
    `<span class="swatch hit"></span>recall hits ` +
    `<span class="swatch selected"></span>selected`;
  const dreamBtn = document.createElement('button');
  dreamBtn.id = 'dream-btn';
  dreamBtn.textContent = 'Dream live';
  dreamBtn.title = 'Physics pass from current positions. Click again after settle for another cycle.';
  dreamBtn.onclick = () => startDream().catch((e) => setStatus(String(e)));
  controls.appendChild(legend);
  controls.appendChild(dreamBtn);
})();

// Boot: personal store → all memory splats in 3D (PCA/projection positions), not basin orbs.
setButtons(false);
Promise.all([
  loadSplats('memories'),
  loadBasins(),
  fetch('/api/status').then(r => r.json()),
]).then(([, , s]) => {
  const eye = document.querySelector('.eyebrow');
  if (eye && s && s.scope_key) eye.textContent = s.scope_key;
  const n = splats.length;
  const bags = (s.basins || 0) > 0 && n / Math.max(1, s.basins) > 4000
    ? ' · many basins look like 5k bags (axis splits, not themes)'
    : '';
  const base = `${(s.cold_records ?? n).toLocaleString()} memories · ${n.toLocaleString()} on field · ${s.basins} basins · dream ${s.dream_cycle}${bags}`;
  document.querySelector('#status').dataset.base =
    `${(s.cold_records ?? n).toLocaleString()} memories · ${n.toLocaleString()} on field · ${s.basins} basins · dream ${s.dream_cycle}`;
  document.querySelector('#status').textContent = base;
  connectLive();
  // Probe jlens once so the orange panel isn't a silent dead button.
  fetch('/api/jlens?' + new URLSearchParams({ q: 'ping', top_k: '1' }))
    .then((r) => r.json())
    .then((j) => {
      if (j && j.ok === false) {
        renderJlens({
          ok: false,
          error: j.error || 'jlens not available',
          raw_stderr_tail: 'Optional. Recall is the real path; this is first-thought vocab only.',
        });
      } else if (j && j.ok) {
        const bridge = document.querySelector('#jlens-bridge');
        if (bridge) {
          bridge.className = 'jlens-bridge quiet';
          bridge.textContent = 'jlens ready — type text, hit First thought';
        }
      }
    })
    .catch(() => {});
  // Dream only on "Dream live" — auto-dream on 50k froze the whole server (full O(N²)).
}).catch(error => {
  document.querySelector('#status').textContent = String(error.message || error);
});
draw();
