/* AI Retention Radar — Inference-only client
 * Loads TF.js model + preprocess.json, parses CSV, transforms per spec,
 * predicts probabilities, plots ROC, and exports CSVs.
 * Works from nested GitHub Pages paths.
 */

// --------- Constants & Status ----------
const MODEL_URL = new URL('web_model/model.json?v=2', document.baseURI).toString();
const SPEC_URL  = new URL('web_model/preprocess.json?v=2', document.baseURI).toString();

const el = (id)=>document.getElementById(id);
const state = {
  spec: null,
  model: null,
  rows: [],
  header: [],
  idCol: null,
  targetCol: null,
  probs: [],
  yTrue: null,     // 0/1 or null
  rocChart: null
};

// --------- Utilities ----------
const toNum = (v) => {
  if (v === null || v === undefined || v === '') return null;
  const x = +v;
  return Number.isFinite(x) ? x : null;
};
const toStr = (v) => (v === null || v === undefined) ? '' : String(v);

// Safe CSV download
function downloadCsv(name, header, rows) {
  const esc = (s) => {
    const v = s == null ? '' : String(s);
    return /[",\n]/.test(v) ? `"${v.replace(/"/g,'""')}"` : v;
  };
  const csv = [header.map(esc).join(',')]
    .concat(rows.map(r => r.map(esc).join(',')))
    .join('\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

function setBadge(okEl, ok) {
  okEl.textContent = ok ? 'OK' : 'failed to load';
  okEl.className = 'badge ' + (ok ? '' : 'err');
}

// --------- Load Spec & Model ----------
async function loadSpec() {
  try {
    const res = await fetch(SPEC_URL, {cache:'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const spec = await res.json();
    // Minimal validation
    if (!spec.input_order || !Array.isArray(spec.input_order)) throw new Error('invalid spec: input_order');
    state.spec = spec;
    state.idCol = spec.id_col || null;
    state.targetCol = spec.target_col || null;
    el('kTarget').textContent = state.targetCol || '—';
    el('kDefThr').textContent = (spec.threshold_default ?? 0.5).toFixed(2);
    setBadge(el('specStatus'), true);
  } catch (e) {
    console.error('Spec load error:', SPEC_URL, e);
    setBadge(el('specStatus'), false);
  }
}

async function loadModel() {
  try {
    const model = await tf.loadLayersModel(MODEL_URL);
    state.model = model;
    setBadge(el('modelStatus'), true);
  } catch (e) {
    console.error('Model load error:', MODEL_URL, e);
    setBadge(el('modelStatus'), false);
  }
}

// --------- CSV Parse & Preview ----------
function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: false,     // keep raw; we’ll coerce later
      skipEmptyLines: 'greedy',
      delimiter: ',',           // Kaggle default
      quoteChar: '"',
      complete: (res) => resolve(res.data),
      error: reject
    });
  });
}

function renderPreview(rows) {
  const table = el('preview');
  table.innerHTML = '';
  if (!rows.length) return;

  const cols = Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>`;
  const bodyRows = rows.slice(0, 10).map(r =>
    `<tr>${cols.map(c => `<td>${toStr(r[c])}</td>`).join('')}</tr>`
  ).join('');
  table.innerHTML = thead + `<tbody>${bodyRows}</tbody>`;

  el('shape').textContent = `Rows: ${rows.length} | Cols: ${cols.length}`;
}

el('csvFile').addEventListener('change', async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;

  try {
    const rows = await parseCsvFile(f);
    // Strip Papa's potential trailing empty row
    const cleaned = rows.filter(r => r && Object.keys(r).length);
    state.rows = cleaned;
    state.header = cleaned.length ? Object.keys(cleaned[0]) : [];
    renderPreview(cleaned);
    el('btnPredict').disabled = !(state.model && state.spec && state.rows.length);
    el('scoredRows').textContent = 'Scored Rows: —';
    el('hasLabels').textContent = 'Has Labels: —';
    el('aucPill').textContent = 'AUC: —';
    el('btnDownloadSubmit').disabled = true;
    el('btnDownloadProbs').disabled = true;
    el('rankTable').innerHTML = '';
    clearRoc();
  } catch (err) {
    console.error('CSV parse error:', err);
    alert('Failed to parse CSV. Ensure it is a comma-separated file with quotes for text fields.');
  }
});

// --------- Transform According to Spec ----------
function transformRows(rows, spec) {
  // Collect numeric & categorical metadata
  const nMeans = spec.numeric_mean || {};
  const nScales = spec.numeric_scale || {};
  const inputOrder = spec.input_order || [];

  const X = new Array(rows.length);
  for (let i=0;i<rows.length;i++) {
    const r = rows[i];
    const vec = new Array(inputOrder.length);
    for (let j=0;j<inputOrder.length;j++) {
      const token = inputOrder[j];
      if (token.startsWith('num:')) {
        const col = token.slice(4);
        const raw = toNum(r[col]);
        const mean = nMeans[col] ?? 0;
        const scale = (nScales[col] ?? 1) || 1;
        const z = (raw == null) ? 0 : (raw - mean) / scale;
        vec[j] = Number.isFinite(z) ? z : 0;
      } else if (token.startsWith('cat:')) {
        // token looks like: cat:country=USA
        const [left, value] = token.split('=');
        const col = left.slice(4);
        const s = toStr(r[col]);
        vec[j] = (s === value) ? 1 : 0;
      } else {
        vec[j] = 0;
      }
    }
    X[i] = vec;
  }
  return X;
}

// Try to binarize labels for evaluation
function binarizeLabelVector(rows, targetCol) {
  if (!targetCol) return null;
  if (!rows.length || !(targetCol in rows[0])) return null;

  const pos = new Set(['churn','churned','cancelled','canceled','inactive','terminated','closed','ended','lost','paused','1','yes','true']);
  const neg = new Set(['active','current','subscribed','renewed','retained','open','live','0','no','false']);

  const y = new Array(rows.length).fill(null);
  let hasAny = false;
  for (let i=0;i<rows.length;i++) {
    const v = rows[i][targetCol];
    if (v === undefined) { y[i] = null; continue; }
    const s = String(v).trim().toLowerCase();
    if (pos.has(s)) { y[i] = 1; hasAny = true; }
    else if (neg.has(s)) { y[i] = 0; hasAny = true; }
    else if (s === '') { y[i] = null; }
    else if (!isNaN(+s)) { y[i] = (+s >= 0.5) ? 1 : 0; hasAny = true; }
    else { y[i] = null; }
  }
  if (!hasAny) return null;
  return y;
}

// --------- Prediction / Metrics ----------
function predictProba(X) {
  const t = tf.tensor2d(X);
  const probs = state.model.predict(t).dataSync();
  t.dispose();
  return Array.from(probs);
}

function computeRocAuc(y, p) {
  const pairs = [];
  for (let i=0;i<y.length;i++) if (y[i]===0 || y[i]===1) pairs.push([p[i], y[i]]);
  if (!pairs.length) return {auc:null, fpr:[], tpr:[]};

  pairs.sort((a,b)=>b[0]-a[0]); // desc by score
  let P = 0, N = 0;
  for (const [,yi] of pairs) (yi===1) ? P++ : N++;
  if (P===0 || N===0) return {auc:null, fpr:[], tpr:[]};

  let tp=0, fp=0, prevScore=Infinity;
  const roc = [[0,0]];
  let auc = 0, prevFpr = 0, prevTpr = 0;

  for (const [score, yi] of pairs) {
    if (score !== prevScore) {
      const fpr = fp/N, tpr = tp/P;
      auc += (fpr - prevFpr) * (tpr + prevTpr) / 2; // trapezoid
      prevFpr = fpr; prevTpr = tpr;
      roc.push([fpr, tpr]);
      prevScore = score;
    }
    if (yi===1) tp++; else fp++;
  }
  // add (1,1)
  const fpr = fp/N, tpr = tp/P;
  auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
  roc.push([fpr, tpr]);

  const fprs = roc.map(p=>p[0]), tprs = roc.map(p=>p[1]);
  return {auc, fpr:fprs, tpr:tprs};
}

function confusionAtThreshold(y, p, thr) {
  let TP=0, FP=0, TN=0, FN=0;
  for (let i=0;i<y.length;i++) {
    if (y[i]!==0 && y[i]!==1) continue;
    const pred = p[i] >= thr ? 1 : 0;
    if (pred===1 && y[i]===1) TP++;
    else if (pred===1 && y[i]===0) FP++;
    else if (pred===0 && y[i]===1) FN++;
    else TN++;
  }
  const precision = (TP+FP)? TP/(TP+FP) : 0;
  const recall = (TP+FN)? TP/(TP+FN) : 0;
  const f1 = (precision+recall)? 2*precision*recall/(precision+recall) : 0;
  return {TP,FP,TN,FN,precision,recall,f1};
}

function renderConfusion(metrics) {
  el('mTP').textContent = metrics.TP;
  el('mFP').textContent = metrics.FP;
  el('mTN').textContent = metrics.TN;
  el('mFN').textContent = metrics.FN;
  el('mP').textContent  = metrics.precision.toFixed(3);
  el('mR').textContent  = metrics.recall.toFixed(3);
  el('mF1').textContent = metrics.f1.toFixed(3);
}

// --------- ROC Chart ----------
function renderRoc(roc) {
  const ctx = document.getElementById('rocChart').getContext('2d');
  if (state.rocChart) { state.rocChart.destroy(); state.rocChart=null; }
  if (!roc || !roc.fpr.length) return;

  state.rocChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: roc.fpr,
      datasets: [
        { label:'ROC', data: roc.tpr, fill: false, borderWidth: 2 },
        { label:'Chance', data: roc.fpr, fill: false, borderWidth: 1, borderDash: [6,6] }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type:'linear', min:0, max:1, title:{display:true,text:'FPR'} },
        y: { type:'linear', min:0, max:1, title:{display:true,text:'TPR'} }
      },
      plugins: { legend: {display:true, position:'bottom'} }
    }
  });
}
function clearRoc(){ if(state.rocChart){ state.rocChart.destroy(); state.rocChart=null; } }

// --------- Ranking Table ----------
function renderRanking(rows, probs, idCol, extras=['age','purchase_frequency','unit_price','country','preferred_category']) {
  const T = el('rankTable');
  if (!rows.length) { T.innerHTML=''; return; }
  const items = rows.map((r,i)=>({i, id: idCol? r[idCol] : (i+1), p: probs[i], row:r}));
  items.sort((a,b)=>b.p - a.p);

  const cols = ['#','id','prob'].concat(extras.filter(c=>c in rows[0]));
  const thead = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>`;
  const body = items.slice(0,100).map((it,rank)=>{
    const cells = [
      `<td>${rank+1}</td>`,
      `<td>${toStr(it.id)}</td>`,
      `<td>${it.p.toFixed(4)}</td>`
    ];
    for (const c of extras) if (c in it.row) cells.push(`<td>${toStr(it.row[c])}</td>`);
    return `<tr>${cells.join('')}</tr>`;
  }).join('');
  T.innerHTML = thead + `<tbody>${body}</tbody>`;
}

// --------- Main Predict Handler ----------
async function runPredict() {
  if (!state.model || !state.spec || !state.rows.length) {
    alert('Load model/spec and CSV first.');
    return;
  }
  // Transform and predict
  const X = transformRows(state.rows, state.spec);
  const probs = predictProba(X);
  state.probs = probs;

  // Labels (optional)
  const y = binarizeLabelVector(state.rows, state.targetCol);
  state.yTrue = y;

  // KPIs
  el('scoredRows').textContent = `Scored Rows: ${probs.length}`;
  el('hasLabels').textContent = `Has Labels: ${y ? 'Yes' : 'No'}`;

  // ROC + AUC if labels exist
  clearRoc();
  if (y) {
    const roc = computeRocAuc(y, probs);
    el('aucPill').textContent = `AUC: ${roc.auc == null ? '—' : roc.auc.toFixed(4)}`;
    renderRoc(roc);
  } else {
    el('aucPill').textContent = 'AUC: —';
  }

  // Threshold default
  const defThr = +(state.spec.threshold_default ?? 0.5);
  el('thr').value = String(defThr);
  el('thrVal').textContent = defThr.toFixed(2);

  // Confusion (if labels)
  if (y) {
    renderConfusion(confusionAtThreshold(y, probs, defThr));
  } else {
    renderConfusion({TP:0,FP:0,TN:0,FN:0,precision:0,recall:0,f1:0});
  }

  // Ranking
  renderRanking(state.rows, probs, state.idCol);

  // Enable downloads
  el('btnDownloadSubmit').disabled = false;
  el('btnDownloadProbs').disabled = false;
}

el('btnPredict').addEventListener('click', runPredict);

// Update threshold live
el('thr').addEventListener('input', (e)=>{
  const thr = +e.target.value;
  el('thrVal').textContent = thr.toFixed(2);
  if (!state.yTrue || !state.yTrue.length) return;
  renderConfusion(confusionAtThreshold(state.yTrue, state.probs, thr));
});

// --------- Downloads ----------
el('btnDownloadSubmit').addEventListener('click', ()=>{
  if (!state.rows.length || !state.probs.length) return;
  const thr = +el('thr').value;
  const idCol = state.idCol || 'id';
  const header = [idCol, 'Churn'];
  const out = state.rows.map((r,i)=>{
    const id = r[idCol] ?? (i+1);
    const yhat = state.probs[i] >= thr ? 1 : 0;
    return [id, yhat];
  });
  downloadCsv('submission.csv', header, out);
});

el('btnDownloadProbs').addEventListener('click', ()=>{
  if (!state.rows.length || !state.probs.length) return;
  const idCol = state.idCol || 'id';
  const header = [idCol, 'probability'];
  const out = state.rows.map((r,i)=> [ r[idCol] ?? (i+1), state.probs[i] ]);
  downloadCsv('probabilities.csv', header, out);
});

// --------- Boot ----------
(async function(){
  await loadSpec();
  await loadModel();
  // If CSV already chosen before model/spec finished, enable Predict now
  el('btnPredict').disabled = !(state.model && state.spec && state.rows.length);
})();
