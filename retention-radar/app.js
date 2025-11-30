/* AI Retention Radar — robust loader + scorer
   Works from any subfolder on GitHub Pages.
   Expects:
     web_model/model.json
     web_model/group1-shard1of1.bin
     web_model/preprocess.json
*/

const UI = {
  bModel: document.getElementById('bModel'),
  bSpec:  document.getElementById('bSpec'),
  csv:    document.getElementById('csvFile'),
  preview:document.getElementById('preview'),
  shape:  document.getElementById('shape'),
  run:    document.getElementById('btnRun'),
  scored: document.getElementById('scored'),
  hasY:   document.getElementById('hasY'),
  aucTxt: document.getElementById('aucTxt'),
  roc:    document.getElementById('roc'),
  thr:    document.getElementById('thr'),
  thrVal: document.getElementById('thrVal'),
  cmBody: document.getElementById('cmBody'),
  btnSub: document.getElementById('btnSub'),
  btnProbs:document.getElementById('btnProbs'),
  rank:   document.getElementById('rank'),
  defaultThr: document.getElementById('defaultThr'),
  specTarget: document.getElementById('specTarget'),
};

const STATE = {
  spec: null,
  model: null,
  rows: [],
  header: [],
  probs: null,
  yTrue: null,
  idCol: null,
};

function okBadge(el, text){ el.classList.remove('fail'); el.classList.add('ok'); el.textContent = text; }
function failBadge(el, text){ el.classList.remove('ok'); el.classList.add('fail'); el.textContent = text; }
function pct(x){ return Math.round(x*1000)/10; }
function clamp01(x){ return Math.max(0, Math.min(1, x)); }

// ---------- Resilient path candidates ----------
const ROOT   = window.location.origin + window.location.pathname.replace(/[^/]*$/, '');
const CANDS  = (rel) => ([
  rel,                       // "web_model/model.json"
  './'+rel,                  // "./web_model/model.json"
  ROOT + rel,                // ".../retention-radar/web_model/model.json"
  window.location.origin + '/NeuralNetwork/retention-radar/' + rel // fallback for your repo
]);

async function tryFetchJSON(cands){
  let lastErr = null;
  for(const u of cands){
    try{
      const r = await fetch(u, {cache:'no-store'});
      if(!r.ok) { lastErr = new Error(`HTTP ${r.status} @ ${u}`); continue; }
      return await r.json();
    }catch(e){ lastErr = e; }
  }
  throw lastErr || new Error('All JSON fetch attempts failed.');
}

async function tryLoadModel(cands){
  let lastErr = null;
  for(const u of cands){
    try{
      const m = await tf.loadLayersModel(u);
      return m;
    }catch(e){ lastErr = e; }
  }
  throw lastErr || new Error('All model load attempts failed.');
}

// ---------- CSV helpers ----------
function parseCSV(file){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, {
      header:true, dynamicTyping:true, skipEmptyLines:'greedy',
      quoteChar:'"', delimiter:',',
      complete:r=>resolve(r), error:reject
    });
  });
}
const normalize = (row)=> {
  const out = {};
  for(const [k,v] of Object.entries(row)){
    if(v===undefined || v===null || v==='') out[k]=null;
    else out[k]=v;
  }
  return out;
};

// ---------- Preprocess based on spec ----------
function z(x, mean, scale){ if(x==null) return 0; return (Number(x)-mean)/ (scale||1); }

function buildMatrix(rows, spec){
  const N = rows.length;
  const numeric = spec.numeric || [];
  const cats    = spec.categorical || {}; // {col: [levels]}
  const numMeans = spec.numeric_mean || {};
  const numScales= spec.numeric_scale || {};

  const catOrder = [];
  const inputOrder = [];
  // numeric first
  for(const c of numeric){ inputOrder.push(`num:${c}`); }
  // categorical by (col, each level)
  for(const col of Object.keys(cats)){
    for(const lvl of cats[col]){ catOrder.push([col, String(lvl)]); inputOrder.push(`cat:${col}=${lvl}`); }
  }

  const X = new Array(N);
  for(let i=0;i<N;i++){
    const r = rows[i];
    const v = [];
    // numeric
    for(const c of numeric){
      v.push(z(r[c], numMeans[c]??0, numScales[c]??1));
    }
    // categorical one-hot
    for(const [col,lvl] of catOrder){
      const val = r[col]==null ? '' : String(r[col]);
      v.push(val===String(lvl) ? 1 : 0);
    }
    X[i] = v;
  }
  return tf.tensor2d(X, [N, inputOrder.length]);
}

// ---------- Metrics ----------
function rocAuc(yTrue, yProb){
  // trapezoidal ROC AUC
  const pairs = yProb.map((p,i)=>[p, yTrue[i]]);
  pairs.sort((a,b)=>b[0]-a[0]);
  let tp=0, fp=0; const P = yTrue.reduce((s,y)=>s+(y===1),0), N = yTrue.length-P;
  let prevP=-1, auc=0, lastTPR=0, lastFPR=0;
  for(const [p,y] of pairs){
    if(p!==prevP){
      auc += ( (lastTPR + tp/P) * ( (fp/N) - lastFPR ) )/2;
      lastTPR = tp/P; lastFPR = fp/N; prevP = p;
    }
    if(y===1) tp++; else fp++;
  }
  auc += ( (lastTPR + tp/P) * ( (fp/N) - lastFPR ) )/2;
  return {auc: isFinite(auc)?auc:NaN, P, N};
}

function confMat(yTrue, yProb, thr){
  let TP=0, TN=0, FP=0, FN=0;
  for(let i=0;i<yTrue.length;i++){
    const y = yTrue[i]; const pred = yProb[i]>=thr ? 1:0;
    if(y===1 && pred===1) TP++;
    else if(y===0 && pred===0) TN++;
    else if(y===0 && pred===1) FP++;
    else if(y===1 && pred===0) FN++;
  }
  const prec = (TP+FP)? TP/(TP+FP) : 0;
  const rec  = (TP+FN)? TP/(TP+FN) : 0;
  const f1   = (prec+rec)? 2*prec*rec/(prec+rec) : 0;
  return {TN,FP,FN,TP,prec,rec,f1};
}

// ---------- UI helpers ----------
function setCM(cm){
  const rows = UI.cmBody.querySelectorAll('tr td:last-child');
  const vals = [cm.TN,cm.FP,cm.FN,cm.TP, cm.prec.toFixed(3), cm.rec.toFixed(3), cm.f1.toFixed(3)];
  rows.forEach((td,i)=> td.textContent = String(vals[i]));
}

function showPreview(res){
  const rows = res.data.slice(0,12);
  const head = res.meta.fields || Object.keys(rows[0]||{});
  const lines = [head.join('\t')].concat(rows.map(r=> head.map(c=> r[c]).join('\t')));
  UI.preview.textContent = lines.join('\n');
  UI.shape.textContent = `Rows: ${res.data.length} | Cols: ${head.length}`;
  STATE.rows = res.data.map(normalize);
  STATE.header = head;
}

// ---------- Downloads ----------
function download(name, text){
  const blob = new Blob([text], {type:'text/csv;charset=utf-8;'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- Main flow ----------
async function boot(){
  try{
    // load SPEC first (preprocess.json)
    STATE.spec = await tryFetchJSON(CANDS('web_model/preprocess.json'));
    okBadge(UI.bSpec, 'Spec: loaded');
    UI.defaultThr.textContent = `Default Threshold ${STATE.spec.threshold_default ?? 0.5}`;
    UI.specTarget.textContent = `Spec Target ${STATE.spec.target_col ?? '—'}`;
    STATE.idCol = STATE.spec.id_col || null;
  }catch(e){
    failBadge(UI.bSpec, 'Spec: failed to load');
    console.error('Spec load error:', e);
  }

  try{
    // then MODEL
    STATE.model = await tryLoadModel(CANDS('web_model/model.json'));
    okBadge(UI.bModel, 'Model: loaded');
  }catch(e){
    failBadge(UI.bModel, 'Model: failed to load');
    console.error('Model load error:', e);
  }
}

UI.csv.addEventListener('change', async (ev)=>{
  const f = ev.target.files?.[0];
  if(!f){ UI.preview.textContent='Preview'; UI.shape.textContent='Rows: — | Cols: —'; return; }
  try{
    const res = await parseCSV(f);
    showPreview(res);
    UI.hasY.textContent = `Has Labels: ${STATE.header.includes(STATE.spec?.target_col||'subscription_status')?'yes':'no'}`;
  }catch(e){
    UI.preview.textContent = 'Failed to parse CSV. Make sure it is comma + double-quote CSV.';
    console.error(e);
  }
});

UI.run.addEventListener('click', async ()=>{
  if(!STATE.model || !STATE.spec){ alert('Model/spec not loaded yet.'); return; }
  if(!STATE.rows.length){ alert('Upload a CSV first.'); return; }

  // Build X
  const X = buildMatrix(STATE.rows, STATE.spec);
  const probT = STATE.model.predict(X);
  const probs = Array.from((await probT.data())); probT.dispose(); X.dispose();
  STATE.probs = probs;

  // yTrue if present
  const yCol = (STATE.spec.target_col || 'subscription_status');
  STATE.yTrue = STATE.header.includes(yCol)
      ? STATE.rows.map(r => (String(r[yCol]??'').toLowerCase().match(/^(1|yes|true|churn|churned|cancelled|canceled|inactive|terminated|closed|ended|lost|paused)$/)?1:0))
      : null;

  UI.scored.textContent = `Scored Rows: ${probs.length}`;
  UI.hasY.textContent = `Has Labels: ${STATE.yTrue? 'yes':'no'}`;

  // AUC
  if(STATE.yTrue){
    const {auc} = rocAuc(STATE.yTrue, probs);
    UI.aucTxt.textContent = `AUC: ${isFinite(auc)?auc.toFixed(4):'—'}`;
    UI.roc.textContent = `ROC AUC: ${isFinite(auc)?auc.toFixed(4):'—'} (simple)`;
  }else{
    UI.roc.textContent = 'Upload a CSV with the label column to display ROC/AUC.';
    UI.aucTxt.textContent = 'AUC: —';
  }

  // threshold block refresh
  const thr = Number(UI.thr.value);
  UI.thrVal.textContent = thr.toFixed(2);
  if(STATE.yTrue){
    setCM(confMat(STATE.yTrue, probs, thr));
  }

  // ranking preview text
  const idC = STATE.idCol || (STATE.header.includes('customer_id') ? 'customer_id' : STATE.header[0]);
  const top = probs
    .map((p,i)=>({id:STATE.rows[i][idC], p, age:STATE.rows[i].age, spend:STATE.rows[i].monetary||STATE.rows[i].unit_price}))
    .sort((a,b)=>b.p-a.p)
    .slice(0,10);
  UI.rank.textContent = top.map(r=>`${idC}=${r.id}  prob=${r.p.toFixed(4)}`).join('\n');
});

UI.thr.addEventListener('input', ()=>{
  const thr = Number(UI.thr.value);
  UI.thrVal.textContent = thr.toFixed(2);
  if(STATE.yTrue && STATE.probs){
    setCM(confMat(STATE.yTrue, STATE.probs, thr));
  }
});

UI.btnSub.addEventListener('click', ()=>{
  if(!STATE.probs){ alert('Run predictions first.'); return; }
  const idC = STATE.idCol || (STATE.header.includes('customer_id') ? 'customer_id' : STATE.header[0]);
  const thr = Number(UI.thr.value);
  const lines = ['customer_id,Churn'];
  for(let i=0;i<STATE.rows.length;i++){
    const id = STATE.rows[i][idC];
    const lab = STATE.probs[i] >= thr ? 1 : 0;
    lines.push(`${id},${lab}`);
  }
  download('submission.csv', lines.join('\n'));
});

UI.btnProbs.addEventListener('click', ()=>{
  if(!STATE.probs){ alert('Run predictions first.'); return; }
  const idC = STATE.idCol || (STATE.header.includes('customer_id') ? 'customer_id' : STATE.header[0]);
  const lines = ['customer_id,prob'];
  for(let i=0;i<STATE.rows.length;i++){
    const id = STATE.rows[i][idC];
    lines.push(`${id},${STATE.probs[i]}`);
  }
  download('probabilities.csv', lines.join('\n'));
});

boot();
