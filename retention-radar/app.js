/* AI Retention Radar — In-browser churn scoring with a TF.js model and a frozen preprocessing spec. */

// ------------ Config (paths are relative to index.html)
const MODEL_URL = './web_model/model.json';
const SPEC_URL  = './web_model/preprocess.json';

// ------------ Globals
let MODEL = null;
let SPEC  = null;
let RAW_ROWS = [];       // CSV as array of objects
let PROBS = [];          // predicted probabilities
let YTRUE = null;        // optional labels if CSV has target
let IDS = [];            // customer_id for export
let ROC_CHART = null;

// Label mapping (must match Colab binarization)
const POS = new Set(['churn','churned','cancelled','canceled','inactive','terminated','closed','ended','lost','paused','1','yes','true']);
const NEG = new Set(['active','current','subscribed','renewed','retained','open','live','0','no','false']);
function labelTo01(v){
  const s = String(v ?? '').trim().toLowerCase();
  if (POS.has(s)) return 1;
  if (NEG.has(s)) return 0;
  return null; // unknown → ignore for metrics
}

// ------------ Small helpers
const $ = (id)=> document.getElementById(id);
const fmtPct = x => isFinite(x) ? (Math.round(x*1000)/10).toFixed(1)+'%' : '—';
const download = (filename, text)=>{
  const blob = new Blob([text], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
};

// ROC + AUC from scores
function rocCurve(y, p){
  // y: 0/1, p: prob
  const pairs = y.map((yy,i)=>({y:yy, p:p[i]})).filter(d=>d.y!=null && isFinite(d.p));
  pairs.sort((a,b)=> b.p - a.p);
  const P = pairs.reduce((s,d)=> s + (d.y===1?1:0), 0);
  const N = pairs.length - P;
  let tp=0, fp=0, prev=-1, pts=[[0,0]];
  for(const d of pairs){
    if(d.p!==prev){ pts.push([fp/N, tp/P]); prev=d.p; }
    if(d.y===1) tp++; else fp++;
  }
  pts.push([1,1]);
  // trapezoid AUC
  let auc=0;
  for(let i=1;i<pts.length;i++){
    const [x1,y1]=pts[i-1], [x2,y2]=pts[i];
    auc += (x2-x1)*(y1+y2)/2;
  }
  return {pts, auc};
}

function confusion(yTrue, probs, th){
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<yTrue.length;i++){
    const y = yTrue[i]; if(y==null) continue;
    const yhat = probs[i] >= th ? 1 : 0;
    if(y===1 && yhat===1) TP++;
    else if(y===0 && yhat===1) FP++;
    else if(y===0 && yhat===0) TN++;
    else if(y===1 && yhat===0) FN++;
  }
  const prec = (TP+FP)? TP/(TP+FP) : 0;
  const rec  = (TP+FN)? TP/(TP+FN) : 0;
  const f1   = (prec+rec)? 2*prec*rec/(prec+rec) : 0;
  return {TN,FP,FN,TP,prec,rec,f1};
}

// ------------ Preprocess using SPEC (frozen means/scales + OHE)
function buildMatrix(rows){
  const numCols = SPEC.numeric || [];
  const catSpec = SPEC.categorical || {};
  const order   = SPEC.input_order || [];
  const means   = SPEC.numeric_mean || {};
  const scales  = SPEC.numeric_scale || {};

  const M = rows.length;
  const D = order.length;
  const X = new Float32Array(M*D);

  const catIndex = {}; // map "cat:col=val" -> column position
  order.forEach((key, j)=>{ catIndex[key]=j; });

  for(let i=0;i<M;i++){
    const r = rows[i];
    // Fill numeric first: "num:col"
    for(const col of numCols){
      const j = catIndex[`num:${col}`];
      let v = r[col];
      let x = (v===undefined || v===null || v==='') ? means[col] : parseFloat(v);
      if (!isFinite(x)) x = means[col];
      const z = (x - (means[col] ?? 0)) / (scales[col] || 1);
      X[i*D + j] = isFinite(z) ? z : 0;
    }
    // Fill categorical one-hots: "cat:col=value"
    for(const col in catSpec){
      const val = String(r[col] ?? '').trim();
      const allowed = catSpec[col] || [];
      for(const v of allowed){
        const j = catIndex[`cat:${col}=${v}`];
        X[i*D + j] = (val===String(v)) ? 1 : 0;
      }
      // Unknown categories ⇒ all zeros (already default)
    }
  }
  return {X, D};
}

// ------------ UI updates
function setStatus(id, ok, msg){
  const el = $(id);
  el.innerHTML = (ok ? '✅ ' : '⚠️ ') + msg;
  el.className = 'pill small ' + (ok?'ok':'bad');
}

function renderPreview(rows){
  if(!rows.length){ $('preview').textContent='Empty CSV'; return; }
  const cols = Object.keys(rows[0]);
  const head = cols.slice(0,8).map(c=>`<th>${c}</th>`).join('');
  const body = rows.slice(0,5).map(r=> `<tr>${cols.slice(0,8).map(c=>`<td>${r[c]??''}</td>`).join('')}</tr>` ).join('');
  $('preview').innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  $('shapeNote').textContent = `Rows: ${rows.length} | Cols: ${cols.length}`;
}

function renderRanking(rows, probs, idCol, yTrue=null){
  const cols = ['rank', idCol, SPEC.target_col, 'prob', 'age','purchase_frequency','unit_price','quantity','cancellations_count','country','preferred_category','category','gender'];
  const thead = cols.map(c=>`<th>${c}</th>`).join('');
  $('rankHead').innerHTML = thead;

  const items = rows.map((r,i)=>({i, id:r[idCol], prob:probs[i], y:yTrue?yTrue[i]:null}));
  items.sort((a,b)=> b.prob - a.prob);

  const tr = items.slice(0,200).map((o,rank)=> {
    const r = rows[o.i];
    return `<tr>
      <td>${rank+1}</td>
      <td>${o.id??''}</td>
      <td>${o.y==null?'':o.y}</td>
      <td>${o.prob.toFixed(4)}</td>
      <td>${r.age??''}</td>
      <td>${r.purchase_frequency??''}</td>
      <td>${r.unit_price??''}</td>
      <td>${r.quantity??''}</td>
      <td>${r.cancellations_count??''}</td>
      <td>${r.country??''}</td>
      <td>${r.preferred_category??''}</td>
      <td>${r.category??''}</td>
      <td>${r.gender??''}</td>
    </tr>`;
  }).join('');
  $('rankBody').innerHTML = tr || `<tr><td colspan="13" class="muted">No rows</td></tr>`;
}

// ------------ Main actions
async function loadModelAndSpec(){
  try{
    MODEL = await tf.loadLayersModel(MODEL_URL);
    setStatus('modelStatus', true, 'loaded');
  }catch(e){
    console.error(e);
    setStatus('modelStatus', false, 'failed to load');
  }
  try{
    const res = await fetch(SPEC_URL); SPEC = await res.json();
    setStatus('specStatus', true, 'loaded');
    // Set default threshold from spec
    const t = SPEC.threshold_default ?? 0.5;
    $('thSlider').value = t; $('thVal').textContent = (+t).toFixed(2);
  }catch(e){
    console.error(e);
    setStatus('specStatus', false, 'failed to load');
  }
}

async function parseCsvFile(file){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, {
      header:true, dynamicTyping:false, skipEmptyLines:'greedy',
      delimiter: ',', quoteChar: '"',
      complete: r => resolve(r.data),
      error: reject
    });
  });
}

function prepareLabels(rows){
  const y = [];
  let hasAny=false;
  for(const r of rows){
    const v = r[SPEC.target_col];
    const m = labelTo01(v);
    y.push(m);
    if(m!==null) hasAny=true;
  }
  return hasAny ? y : null;
}

async function runPredict(){
  if(!MODEL || !SPEC){ alert('Model/spec not loaded yet.'); return; }
  if(!RAW_ROWS.length){ alert('Please load a CSV first.'); return; }

  $('prepNote').textContent = 'Building features…';
  const {X, D} = buildMatrix(RAW_ROWS);
  if (D !== (MODEL.inputs[0].shape[1] || SPEC.input_order.length)){
    alert(`Feature length mismatch. Got ${D}, expected ${MODEL.inputs[0].shape[1]}.`);
    return;
  }

  IDS = RAW_ROWS.map(r=> r[SPEC.id_col] ?? r['customer_id'] ?? r['id'] ?? '');
  YTRUE = prepareLabels(RAW_ROWS);

  $('prepNote').textContent = 'Running model…';
  const tensor = tf.tensor2d(X, [RAW_ROWS.length, D]);
  const out = MODEL.predict(tensor);
  const probs = Array.from(await out.data());
  tensor.dispose(); out.dispose();

  PROBS = probs;
  $('kRows').textContent = String(PROBS.length);
  $('kHasY').textContent = YTRUE ? 'Yes' : 'No';

  // Metrics if labels
  if(YTRUE){
    const cleanIdx = YTRUE.map((v,i)=> v!=null ? i : -1).filter(i=>i>=0);
    const yC = cleanIdx.map(i=> YTRUE[i]);
    const pC = cleanIdx.map(i=> PROBS[i]);
    const {pts, auc} = rocCurve(yC, pC);
    $('kAUC').textContent = (Math.round(auc*1000)/1000).toFixed(3);
    drawROC(pts);
    updateConfusion();
    $('rocNote').textContent = '';
  }else{
    $('kAUC').textContent = '—';
    drawROC([]); // clear
    clearConfusion();
    $('rocNote').textContent = 'No label column in this CSV.';
  }

  renderRanking(RAW_ROWS, PROBS, SPEC.id_col || 'customer_id', YTRUE);
  $('prepNote').textContent = 'Done.';
}

function drawROC(points){
  const ctx = $('rocChart').getContext('2d');
  if(ROC_CHART){ ROC_CHART.destroy(); ROC_CHART=null; }
  ROC_CHART = new Chart(ctx, {
    type:'line',
    data:{
      labels: points.map(p=>p[0]),
      datasets: [{
        label:'ROC (TPR vs FPR)',
        data: points.map(p=>({x:p[0], y:p[1]})),
        fill:false, pointRadius:0, borderWidth:2
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      scales:{
        x:{type:'linear', min:0, max:1, title:{display:true, text:'FPR'}},
        y:{type:'linear', min:0, max:1, title:{display:true, text:'TPR'}}
      },
      plugins:{legend:{display:false}}
    }
  });
}

function clearConfusion(){
  const rows = $('cmTable').querySelectorAll('tr td:nth-child(2)');
  rows.forEach(td=> td.textContent='—');
}

function updateConfusion(){
  if(!YTRUE){ clearConfusion(); return; }
  const th = +$('thSlider').value; $('thVal').textContent = th.toFixed(2);
  const cleanIdx = YTRUE.map((v,i)=> v!=null ? i : -1).filter(i=>i>=0);
  const yC = cleanIdx.map(i=> YTRUE[i]);
  const pC = cleanIdx.map(i=> PROBS[i]);
  const {TN,FP,FN,TP,prec,rec,f1} = confusion(yC,pC,th);
  const vals = [TN,FP,FN,TP,prec,rec,f1];
  const tds = $('cmTable').querySelectorAll('tr td:nth-child(2)');
  tds[0].textContent = TN; tds[1].textContent = FP; tds[2].textContent = FN; tds[3].textContent = TP;
  tds[4].textContent = (Math.round(prec*1000)/1000).toFixed(3);
  tds[5].textContent = (Math.round(rec*1000)/1000).toFixed(3);
  tds[6].textContent = (Math.round(f1*1000)/1000).toFixed(3);
}

// ------------ Downloads
function downloadSubmission(){
  if(!PROBS.length){ alert('No predictions yet.'); return; }
  const th = +$('thSlider').value;
  const header = ['customer_id','Survived'].join(','); // “Survived” name is conventional; you can rename to Label
  const lines = [header].concat(IDS.map((id,i)=> `${id},${PROBS[i]>=th?1:0}` ));
  download('submission.csv', lines.join('\n'));
}
function downloadProbabilities(){
  if(!PROBS.length){ alert('No predictions yet.'); return; }
  const header = ['customer_id','probability'].join(',');
  const lines = [header].concat(IDS.map((id,i)=> `${id},${PROBS[i].toFixed(6)}` ));
  download('probabilities.csv', lines.join('\n'));
}

// ------------ Wire up
window.addEventListener('DOMContentLoaded', async ()=>{
  await loadModelAndSpec();

  $('btnLoad').addEventListener('click', async ()=>{
    const f = $('csvFile').files?.[0];
    if(!f){ alert('Choose a CSV first.'); return; }
    RAW_ROWS = await parseCsvFile(f);
    renderPreview(RAW_ROWS);
    $('prepNote').textContent = `Ready. Spec expects ${SPEC.input_order?.length||'—'} features.`;
  });

  $('btnPredict').addEventListener('click', runPredict);
  $('thSlider').addEventListener('input', updateConfusion);
  $('btnDownloadPred').addEventListener('click', downloadSubmission);
  $('btnDownloadProb').addEventListener('click', downloadProbabilities);
});
