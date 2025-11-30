/* Final — robust loader that works on GitHub Pages subpaths.
   - Auto-detects absolute URLs for model.json and preprocess.json
   - Shows the exact URL it used (or failed) in the UI
   - Feature vector follows preprocess.json exactly
*/

const $ = (id)=>document.getElementById(id);
const state = {
  spec:null, model:null,
  rows:[], header:[],
  idCol:null, targetCol:null,
  probs:[], yTrue:null,
  rocChart:null
};

// ---------- URL Resolver (fixes your 404) ----------
async function pickFirstReachable(urls){
  for (const u of urls){
    try{
      const res = await fetch(u, {method:'HEAD', cache:'no-store'});
      if (res.ok) return u;
    }catch(_){/* ignore */}
  }
  return null;
}
function candidates(rel){
  // Build several absolute candidates that work for /user/repo/subdir/
  const baseNoIndex = location.href.replace(/index\.html?$/i,'');
  const dir = location.origin + location.pathname.replace(/[^/]*$/,''); // directory containing index
  return [
    new URL(rel, document.baseURI).toString(),
    new URL('./'+rel, document.baseURI).toString(),
    new URL(rel, baseNoIndex).toString(),
    new URL(rel, dir).toString(),
    // last resort: absolute path keeping repo root segment, e.g. /NeuralNetwork/retention-radar/web_model/...
    location.origin + location.pathname.replace(/[^/]*$/,'') + rel
  ];
}

// ---------- Badges ----------
function setBadge(el, ok, url, err=''){
  el.textContent = ok ? 'OK' : 'failed to load';
  el.className = 'badge ' + (ok?'':'err');
  const small = (el.id==='modelStatus') ? $('modelUrlShown') : $('specUrlShown');
  if (url){
    small.innerHTML = ok
      ? `&nbsp;<a href="${url}" target="_blank" class="mut tiny">${url}</a>`
      : `&nbsp;<a href="${url}" target="_blank" class="mut tiny">${url}</a> <span class="mut tiny">(${err||'HTTP error'})</span>`;
  }
}

// ---------- Load spec & model ----------
async function loadSpec(){
  const rel = 'web_model/preprocess.json?v=4';
  const url = await pickFirstReachable(candidates(rel));
  if(!url){ setBadge($('specStatus'), false, candidates(rel)[0], 'not found'); return; }
  try{
    const res = await fetch(url, {cache:'no-store'});
    const spec = await res.json();
    if(!spec || !Array.isArray(spec.input_order)) throw new Error('invalid spec');
    state.spec = spec;
    state.idCol = spec.id_col || null;
    state.targetCol = spec.target_col || null;
    $('kTarget').textContent = state.targetCol || '—';
    $('kDefThr').textContent = (spec.threshold_default ?? 0.5).toFixed(2);
    setBadge($('specStatus'), true, url);
  }catch(e){
    console.error('spec load', e);
    setBadge($('specStatus'), false, url, e.message);
  }
}

async function loadModel(){
  const rel = 'web_model/model.json?v=4';
  const url = await pickFirstReachable(candidates(rel));
  if(!url){ setBadge($('modelStatus'), false, candidates(rel)[0], 'not found'); return; }
  try{
    const model = await tf.loadLayersModel(url);
    state.model = model;
    setBadge($('modelStatus'), true, url);
  }catch(e){
    console.error('model load', e);
    setBadge($('modelStatus'), false, url, e.message);
  }
}

// ---------- CSV ----------
function parseCsvFile(file){
  return new Promise((resolve,reject)=>{
    Papa.parse(file,{
      header:true, dynamicTyping:false, skipEmptyLines:'greedy',
      delimiter:',', quoteChar:'"', // Kaggle-style
      complete:r=>resolve(r.data.filter(x=>x&&Object.keys(x).length)),
      error:reject
    });
  });
}
function toStr(v){ return (v==null? '' : String(v)); }
function toNum(v){ const x = +v; return Number.isFinite(x)? x : null; }

function renderPreview(rows){
  const T = $('preview'); T.innerHTML='';
  if(!rows.length){ $('shape').textContent='Rows: — | Cols: —'; return; }
  const cols = Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>`;
  const body = rows.slice(0,10).map(r=>`<tr>${cols.map(c=>`<td>${toStr(r[c])}</td>`).join('')}</tr>`).join('');
  T.innerHTML = thead + `<tbody>${body}</tbody>`;
  $('shape').textContent = `Rows: ${rows.length} | Cols: ${cols.length}`;
}

$('csvFile').addEventListener('change', async (e)=>{
  const f = e.target.files?.[0]; if(!f) return;
  try{
    const rows = await parseCsvFile(f);
    state.rows = rows; state.header = rows.length? Object.keys(rows[0]) : [];
    renderPreview(rows);
    $('btnPredict').disabled = !(state.model && state.spec && state.rows.length);
    $('scoredRows').textContent = 'Scored Rows: —';
    $('hasLabels').textContent = 'Has Labels: —';
    $('aucPill').textContent = 'AUC: —';
    $('btnDownloadSubmit').disabled = true;
    $('btnDownloadProbs').disabled = true;
    $('rankTable').innerHTML=''; clearRoc();
  }catch(err){
    console.error('CSV error', err); alert('Failed to parse CSV.');
  }
});

// ---------- Transform exactly per spec ----------
function transformRows(rows, spec){
  const means = spec.numeric_mean || {};
  const scales = spec.numeric_scale || {};
  const order = spec.input_order || [];
  const X = new Array(rows.length);

  for (let i=0;i<rows.length;i++){
    const r = rows[i];
    const v = new Array(order.length);
    for (let j=0;j<order.length;j++){
      const tok = order[j];
      if (tok.startsWith('num:')){
        const col = tok.slice(4);
        const raw = toNum(r[col]);
        const mean = means[col] ?? 0;
        const scale = (scales[col] ?? 1) || 1;
        const z = (raw==null) ? 0 : (raw - mean) / scale;
        v[j] = Number.isFinite(z)? z : 0;
      }else if (tok.startsWith('cat:')){
        const [left, val] = tok.split('=');
        const col = left.slice(4);
        v[j] = (toStr(r[col]) === val) ? 1 : 0;
      }else{
        v[j] = 0;
      }
    }
    X[i]=v;
  }
  return X;
}

// ---------- Labels, ROC, metrics ----------
function binLabels(rows, col){
  if(!col || !rows.length || !(col in rows[0])) return null;
  const pos = new Set(['churn','churned','cancelled','canceled','inactive','terminated','closed','ended','lost','paused','1','yes','true']);
  const neg = new Set(['active','current','subscribed','renewed','retained','open','live','0','no','false']);
  const y = new Array(rows.length).fill(null); let any=false;
  for(let i=0;i<rows.length;i++){
    const s = String(rows[i][col] ?? '').trim().toLowerCase();
    if (pos.has(s)) {y[i]=1; any=true;}
    else if (neg.has(s)) {y[i]=0; any=true;}
    else if (s!=='' && !isNaN(+s)) {y[i]=(+s>=0.5?1:0); any=true;}
  }
  return any? y : null;
}

function predictProba(X){
  const t = tf.tensor2d(X);
  const p = state.model.predict(t).dataSync();
  t.dispose();
  return Array.from(p);
}

function computeRocAuc(y, p){
  const pairs=[]; for(let i=0;i<y.length;i++) if(y[i]===0||y[i]===1) pairs.push([p[i], y[i]]);
  if(!pairs.length) return {auc:null,fpr:[],tpr:[]};
  pairs.sort((a,b)=>b[0]-a[0]);
  let P=0,N=0; for(const [,yi] of pairs){ (yi===1)?P++:N++; }
  if(!P||!N) return {auc:null,fpr:[],tpr:[]};
  let tp=0,fp=0,prev=Infinity,prevF=0,prevT=0,auc=0;
  const roc=[[0,0]];
  for(const [s,ytrue] of pairs){
    if(s!==prev){
      const f=fp/N,t=tp/P;
      auc += (f-prevF)*(t+prevT)/2;
      prevF=f; prevT=t; roc.push([f,t]); prev=s;
    }
    if(ytrue===1) tp++; else fp++;
  }
  const f=fp/N,t=tp/P; auc += (f-prevF)*(t+prevT)/2; roc.push([f,t]);
  return {auc,fpr:roc.map(z=>z[0]),tpr:roc.map(z=>z[1])};
}

function confusionAt(y,p,thr){
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<y.length;i++){
    if(y[i]!==0 && y[i]!==1) continue;
    const yh = p[i]>=thr?1:0;
    if(yh===1 && y[i]===1) TP++;
    else if(yh===1 && y[i]===0) FP++;
    else if(yh===0 && y[i]===1) FN++;
    else TN++;
  }
  const precision=(TP+FP)?TP/(TP+FP):0;
  const recall=(TP+FN)?TP/(TP+FN):0;
  const f1=(precision+recall)?2*precision*recall/(precision+recall):0;
  return {TP,FP,TN,FN,precision,recall,f1};
}

function renderConf(m){
  $('mTP').textContent=m.TP; $('mFP').textContent=m.FP;
  $('mTN').textContent=m.TN; $('mFN').textContent=m.FN;
  $('mP').textContent=m.precision.toFixed(3);
  $('mR').textContent=m.recall.toFixed(3);
  $('mF1').textContent=m.f1.toFixed(3);
}

// ---------- ROC chart ----------
function clearRoc(){ if(state.rocChart){ state.rocChart.destroy(); state.rocChart=null; } }
function renderRoc(roc){
  const ctx = document.getElementById('rocChart').getContext('2d');
  if(state.rocChart) state.rocChart.destroy();
  if(!roc || !roc.fpr.length) return;
  state.rocChart = new Chart(ctx,{
    type:'line',
    data:{
      labels:roc.fpr,
      datasets:[
        {label:'ROC', data:roc.tpr, borderWidth:2, fill:false},
        {label:'Chance', data:roc.fpr, borderWidth:1, borderDash:[6,6], fill:false}
      ]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      scales:{
        x:{type:'linear',min:0,max:1,title:{display:true,text:'FPR'}},
        y:{type:'linear',min:0,max:1,title:{display:true,text:'TPR'}}
      },
      plugins:{legend:{display:true,position:'bottom'}}
    }
  });
}

// ---------- Ranking + Downloads ----------
function renderRanking(rows, probs, idCol, extras=['age','purchase_frequency','unit_price','country','preferred_category']){
  const T=$('rankTable'); if(!rows.length){T.innerHTML='';return;}
  const items = rows.map((r,i)=>({i, id: (idCol && r[idCol]!=null)? r[idCol] : (i+1), p: probs[i], row:r}))
                    .sort((a,b)=>b.p-a.p).slice(0,100);
  const cols = ['#','id','prob'].concat(extras.filter(c=>c in rows[0]));
  const thead = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>`;
  const body = items.map((it,rank)=>{
    const cells=[ `<td>${rank+1}</td>`,`<td>${String(it.id)}</td>`,`<td>${it.p.toFixed(4)}</td>` ];
    for(const c of extras) if(c in it.row) cells.push(`<td>${String(it.row[c])}</td>`);
    return `<tr>${cells.join('')}</tr>`;
  }).join('');
  T.innerHTML = thead + `<tbody>${body}</tbody>`;
}

function downloadCsv(name, header, rows){
  const esc=v=>{const s=v==null?'':String(v); return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
  const csv=[header.map(esc).join(',')].concat(rows.map(r=>r.map(esc).join(','))).join('\n');
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
  const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}

// ---------- Predict flow ----------
async function runPredict(){
  if(!state.model || !state.spec || !state.rows.length){ alert('Load model/spec and CSV first.'); return; }
  const X = transformRows(state.rows, state.spec);
  state.probs = predictProba(X);
  state.yTrue = binLabels(state.rows, state.targetCol);

  $('scoredRows').textContent = `Scored Rows: ${state.probs.length}`;
  $('hasLabels').textContent  = `Has Labels: ${state.yTrue? 'Yes':'No'}`;

  clearRoc();
  if(state.yTrue){
    const roc = computeRocAuc(state.yTrue, state.probs);
    $('aucPill').textContent = `AUC: ${roc.auc==null?'—':roc.auc.toFixed(4)}`;
    renderRoc(roc);
  }else{
    $('aucPill').textContent = 'AUC: —';
  }

  const thr = +(state.spec.threshold_default ?? 0.5);
  $('thr').value = String(thr); $('thrVal').textContent = thr.toFixed(2);
  if(state.yTrue) renderConf(confusionAt(state.yTrue, state.probs, thr));
  else renderConf({TP:0,FP:0,TN:0,FN:0,precision:0,recall:0,f1:0});

  renderRanking(state.rows, state.probs, state.idCol);
  $('btnDownloadSubmit').disabled=false; $('btnDownloadProbs').disabled=false;
}
$('btnPredict').addEventListener('click', runPredict);

$('thr').addEventListener('input', e=>{
  const thr=+e.target.value; $('thrVal').textContent=thr.toFixed(2);
  if(state.yTrue) renderConf(confusionAt(state.yTrue, state.probs, thr));
});

$('btnDownloadSubmit').addEventListener('click', ()=>{
  if(!state.rows.length || !state.probs.length) return;
  const thr=+$('thr').value, idCol=state.idCol || 'id';
  const header=[idCol,'Churn'];
  const out = state.rows.map((r,i)=>[ r[idCol] ?? (i+1), (state.probs[i]>=thr?1:0) ]);
  downloadCsv('submission.csv', header, out);
});
$('btnDownloadProbs').addEventListener('click', ()=>{
  if(!state.rows.length || !state.probs.length) return;
  const idCol=state.idCol || 'id';
  const header=[idCol,'probability'];
  const out = state.rows.map((r,i)=>[ r[idCol] ?? (i+1), state.probs[i] ]);
  downloadCsv('probabilities.csv', header, out);
});

// ---------- Boot ----------
(async function(){
  await loadSpec();
  await loadModel();
  $('btnPredict').disabled = !(state.model && state.spec && state.rows.length);
})();
