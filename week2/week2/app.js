/*
  Titanic Binary Classifier — TensorFlow.js (Week 2)
  FINAL SAFE build (after "Preprocessing failed" screenshot)

  What this file does:
   • Robust CSV parsing (auto-detect delimiter+quote; uses optional #csvDelimiter/#csvQuote if present)
   • Row normalization ('' -> null, trims strings)
   • Bullet‑proof preprocessing (JS medians/modes, JS mean/std, clamps non‑finite, fixed feature length)
   • Shallow NN: Dense(16,'relu') -> Dense(1,'sigmoid'); adam + binaryCrossentropy + accuracy
   • Training with tfjs‑vis charts, early stopping
   • ROC/AUC, threshold slider -> confusion matrix + Precision/Recall/F1
   • Predict on test.csv; download submission.csv, probabilities.csv; save model to disk
*/

// ----------------------------- Global State -----------------------------
const state = {
  rawTrain: [],
  rawTest: [],
  pre: null,
  xsTrain: null, ysTrain: null,
  xsVal: null,   ysVal: null,
  model: null,
  valProbs: null,
  testProbs: null,
  thresh: 0.5,
};

// Schema (swap here if reusing for another dataset)
const SCHEMA = {
  target: 'Survived',
  id: 'PassengerId',
  cols: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],
};

// Expected columns for Titanic; used to score CSV parses
const EXPECTED_MIN = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
const NICE_TO_HAVE = ['Name'];

// UI helpers
const el  = id => document.getElementById(id);
const opt = id => document.getElementById(id) || null; // optional element

// ----------------------------- CSV Parsing (robust) -----------------------------
function parseWithPapa(file, delimiter, quoteChar){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, {
      header:true,
      dynamicTyping:true,
      skipEmptyLines:'greedy',
      delimiter,     // ',', ';', or '\t'
      quoteChar,     // '"' or '\''
      complete: res => resolve(res.data),
      error: reject
    });
  });
}

function roughMissingPct(rows){
  if(!rows || !rows.length) return 100;
  const cols = Object.keys(rows[0]);
  let miss = 0, total = rows.length * cols.length;
  for(const r of rows){ for(const c of cols){ const v=r[c]; if(v===''||v==null||v===undefined) miss++; } }
  return +(100*miss/total).toFixed(1);
}

function humanDelim(d){ return d === '\t' ? '\\t' : d; }

function scoreParsed(rows, delimiter, quoteChar){
  if(!rows || !rows.length) return {score:-1e9, diag:'No rows'};
  const headers = Object.keys(rows[0] ?? {});
  const headerSet = new Set(headers);
  let have=0; for(const c of EXPECTED_MIN) if(headerSet.has(c)) have++;
  const expectedHit = have/EXPECTED_MIN.length; // 0..1
  const widthCounts={};
  for(const r of rows){ const w=Object.keys(r).length; widthCounts[w]=(widthCounts[w]||0)+1; }
  const modeW = Object.entries(widthCounts).sort((a,b)=>b[1]-a[1])[0];
  const consistency = modeW? modeW[1]/rows.length : 0;
  const bonus = NICE_TO_HAVE.every(c=>headerSet.has(c)) ? 0.1 : 0;
  const missPct = roughMissingPct(rows)/100;
  const missPenalty = -0.2*missPct;
  const score = 3*expectedHit + 2*consistency + bonus + missPenalty;
  const diag = `Parsed with delimiter='${humanDelim(delimiter)}', quote='${quoteChar}' | expectedHit=${(100*expectedHit).toFixed(0)}% | rowConsistency=${(100*consistency).toFixed(0)}% | ` + (bonus>0?'Name:OK | ':'') + `missing≈${(missPct*100).toFixed(1)}%`;
  return {score, diag};
}

async function robustParseCSV(file, manualDelim, manualQuote){
  const tryDelims = manualDelim && manualDelim !== 'auto' ? [manualDelim] : [',',';','\t'];
  const tryQuotes = manualQuote && manualQuote !== 'auto' ? [manualQuote] : ['"',"'"];
  let best = {score:-Infinity, rows:[], cfg:null, diag:''};
  for(const d of tryDelims){
    for(const q of tryQuotes){
      const rows = await parseWithPapa(file, d, q);
      const {score, diag} = scoreParsed(rows, d, q);
      if(score > best.score) best = {score, rows, cfg:{delimiter:d, quoteChar:q}, diag};
    }
  }
  return best;
}

// ----------------------------- Row Normalization -----------------------------
function normalizeRow(row){
  const out={};
  for(const [k,v] of Object.entries(row)){
    if(v===''){ out[k]=null; continue; }
    if(typeof v==='string'){ const t=v.trim(); out[k]=t===''? null : t; }
    else { out[k]=v; }
  }
  return out;
}

// ----------------------------- EDA & Preview -----------------------------
function previewTable(rows, limit=8){
  if(!rows || !rows.length){ el('previewTable') && (el('previewTable').innerHTML=''); return; }
  const cols = Object.keys(rows[0]);
  const head = '<thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
  const body = '<tbody>' + rows.slice(0,limit).map(r=>'<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>').join('') + '</tbody>';
  el('previewTable') && (el('previewTable').innerHTML = `<table>${head}${body}</table>`);
}

async function edaCharts(rows){
  if(!rows || !rows.length || !window.tfvis) return;
  const bySex = ['female','male'].map(s=>{
    const g = rows.filter(r=> r[SCHEMA.target]!=null && r.Sex===s);
    const rate = g.length? g.filter(r=> +r[SCHEMA.target]===1).length/g.length : 0;
    return {index:s, value:+(100*rate).toFixed(1)};
  });
  el('edaSex') && await tfvis.render.barchart(el('edaSex'), bySex, {xLabel:'Sex', yLabel:'Survival %', width:450, height:220});

  const byClass = [1,2,3].map(c=>{
    const g = rows.filter(r=> r[SCHEMA.target]!=null && +r.Pclass===c);
    const rate = g.length? g.filter(r=> +r[SCHEMA.target]===1).length/g.length : 0;
    return {index:String(c), value:+(100*rate).toFixed(1)};
  });
  el('edaClass') && await tfvis.render.barchart(el('edaClass'), byClass, {xLabel:'Pclass', yLabel:'Survival %', width:450, height:220});
}

// ----------------------------- Preprocessing (SAFE) -----------------------------
function median(arr){ const a=arr.filter(v=>v!=null && !Number.isNaN(+v)).map(Number).sort((x,y)=>x-y); if(!a.length) return null; const m=Math.floor(a.length/2); return a.length%2? a[m] : (a[m-1]+a[m])/2; }
function mode(arr){ const m=new Map(); let best=null,cnt=0; for(const v of arr){ if(v==null||v==='') continue; const c=(m.get(v)||0)+1; m.set(v,c); if(c>cnt){cnt=c;best=v;} } return best; }
function oneHot(value, cats){ const v=new Array(cats.length).fill(0); const i=cats.indexOf(value); if(i>=0) v[i]=1; return v; }
function jsMean(a){ const b=a.filter(Number.isFinite); return b.length? b.reduce((s,x)=>s+x,0)/b.length : 0; }
function jsStd(a){ const b=a.filter(Number.isFinite); if (b.length<2) return 0; const mu=jsMean(b); const v=b.reduce((s,x)=>s+(x-mu)**2,0)/(b.length-1); return Math.sqrt(v); }
function clampFinite(x, def=0){ return Number.isFinite(x) ? x : def; }

function buildPreprocessor(trainRows){
  const ageMedRaw = median(trainRows.map(r=> r.Age));
  const ageMed    = Number.isFinite(ageMedRaw) ? ageMedRaw : 30;
  const embMode   = (mode(trainRows.map(r=> r.Embarked))) ?? 'S';

  const sexCats    = ['female','male'];
  const pclassCats = [1,2,3];
  const embCats    = ['C','Q','S','UNKNOWN'];

  const ageVals  = trainRows.map(r => { const v=(r.Age!=null && !Number.isNaN(+r.Age))? +r.Age : ageMed; return Number.isFinite(v)? v : ageMed; });
  const fareVals = trainRows.map(r => { const v=(r.Fare!=null && !Number.isNaN(+r.Fare))? +r.Fare : 0;     return Number.isFinite(v)? v : 0;     });

  const muAge  = jsMean(ageVals);  const sdAge  = jsStd(ageVals);
  const muFare = jsMean(fareVals); const sdFare = jsStd(fareVals);

  const useFamily = !!(opt('featFamily')?.checked ?? true);
  const useAlone  = !!(opt('featAlone')?.checked  ?? true);

  const mapRowBase = (r)=>{
    const age  = (r.Age!=null && !Number.isNaN(+r.Age)) ? +r.Age : ageMed;
    const emb  = (r.Embarked==null || r.Embarked==='') ? 'UNKNOWN' : r.Embarked;
    const fare = (r.Fare!=null && !Number.isNaN(+r.Fare)) ? +r.Fare : 0;
    const fam   = (+r.SibSp||0) + (+r.Parch||0) + 1;
    const alone = (fam===1) ? 1 : 0;
    const ageZ  = sdAge  ? (age - muAge)/sdAge   : 0;
    const fareZ = sdFare ? (fare - muFare)/sdFare: 0;
    const sexOH = oneHot(r.Sex, sexCats);
    const pclOH = oneHot(+r.Pclass, pclassCats);
    const embOH = oneHot(emb, embCats);
    let feats = [ageZ, fareZ, ...sexOH, ...pclOH, ...embOH];
    if (useFamily) feats.push(fam);
    if (useAlone)  feats.push(alone);
    return feats.map(x=>clampFinite(+x,0));
  };

  const FEAT_LEN = mapRowBase(trainRows[0] || {}).length;

  return {
    ageMed, embMode, sexCats, pclassCats, embCats, muAge, sdAge, muFare, sdFare,
    useFamily, useAlone, featLen: FEAT_LEN,
    mapRow: (r)=>{
      const v = mapRowBase(r);
      if (v.length !== FEAT_LEN) { // enforce fixed width
        if (v.length < FEAT_LEN) v.push(...Array(FEAT_LEN - v.length).fill(0));
        else v.length = FEAT_LEN;
      }
      return v;
    }
  };
}

function tensorize(rows, pre){
  const X=[]; const Y=[];
  for (const r of rows){
    const f = pre.mapRow(r);
    if (f.every(Number.isFinite)) {
      X.push(f);
      if (SCHEMA.target in r) Y.push(+r[SCHEMA.target]);
    }
  }
  if (!X.length) throw new Error('No valid rows after preprocessing.');
  const xs = tf.tensor2d(X, [X.length, pre.featLen], 'float32');
  let ys = null;
  if (Y.length) ys = tf.tensor2d(Y, [Y.length, 1], 'float32'); // labels as (N,1)
  return { xs, ys, nFeat: pre.featLen };
}

function stratifiedSplit(rows, valRatio=0.2){
  const zeros = rows.filter(r=> +r[SCHEMA.target]===0);
  const ones  = rows.filter(r=> +r[SCHEMA.target]===1);
  function split(g){ const a=g.slice(); tf.util.shuffle(a); const n=Math.max(1,Math.floor(a.length*valRatio)); return {val:a.slice(0,n), tr:a.slice(n)}; }
  const a=split(zeros), b=split(ones);
  const train=a.tr.concat(b.tr), val=a.val.concat(b.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  return {train, val};
}

// ----------------------------- Model -----------------------------
function buildModel(inputDim){
  const m = tf.sequential();
  m.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  m.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  return m;
}

function summaryToText(model){ const lines=[]; model.summary(line=>lines.push(line)); return lines.join('\n'); }

// ----------------------------- Metrics: ROC/AUC & PRF -----------------------------
function rocPoints(yTrue, yProb, steps=200){
  const T=[]; for(let i=0;i<=steps;i++) T.push(i/steps);
  const pts=T.map(th=>{ let TP=0,FP=0,TN=0,FN=0; for(let i=0;i<yTrue.length;i++){ const y=yTrue[i],p=yProb[i]>=th?1:0; if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++; } const TPR=TP/(TP+FN||1), FPR=FP/(FP+TN||1); return {x:FPR,y:TPR,th}; });
  const sorted=pts.slice().sort((a,b)=>a.x-b.x); let auc=0; for(let i=1;i<sorted.length;i++){ const x1=sorted[i-1].x,y1=sorted[i-1].y,x2=sorted[i].x,y2=sorted[i].y; auc+=(x2-x1)*(y1+y2)/2; } return {points:sorted, auc};
}

function confusionStats(yTrue, yProb, th){
  let TP=0,FP=0,TN=0,FN=0; for(let i=0;i<yTrue.length;i++){ const y=yTrue[i],p=yProb[i]>=th?1:0; if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++; }
  const prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=(2*prec*rec)/((prec+rec)||1); return {TP,FP,TN,FN,prec,rec,f1};
}

async function plotROC(container, points, auc){
  if(!window.tfvis) return; const series=[{name:'ROC', values: points.map(p=>({x:p.x,y:p.y}))}];
  await tfvis.render.linechart(container, series, {width:520,height:300,xLabel:'FPR',yLabel:'TPR',seriesColors:['#8aa3ff']});
  el('prf') && (el('prf').textContent = `AUC = ${auc.toFixed(4)}\nMove the slider to change threshold and see PR/F1.`);
}

function showConfMat(container, stats){
  if(!window.tfvis) return; const data=[[stats.TP,stats.FN],[stats.FP,stats.TN]]; // [[tp,fn],[fp,tn]]
  tfvis.render.confusionMatrix(container,{values:data,tickLabels:['1','0']},{shadeDiagonal:true,width:260,height:220});
}

// ----------------------------- Downloads -----------------------------
function downloadCSV(filename, rows){
  if(!rows.length) return; const cols=Object.keys(rows[0]);
  const esc=v=>{ if(v==null) return ''; const s=String(v); return /[",\n]/.test(s)? '"'+s.replace(/\"/g,'\"\"')+'"' : s; };
  const csv=[cols.join(',')].concat(rows.map(r=> cols.map(c=>esc(r[c])).join(','))).join('\n');
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'}); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}

// ----------------------------- Event Handlers -----------------------------
async function onLoadFiles(){
  try{
    const fTrain = el('trainFile')?.files?.[0];
    const fTest  = el('testFile')?.files?.[0];
    if(!fTrain){ alert('Please choose train.csv'); return; }

    const manualDelim = opt('csvDelimiter')? opt('csvDelimiter').value : 'auto';
    const manualQuote = opt('csvQuote')?     opt('csvQuote').value     : 'auto';

    const bestTrain = await robustParseCSV(fTrain, manualDelim, manualQuote);
    state.rawTrain = bestTrain.rows.map(normalizeRow);
    opt('parseDiag') && (opt('parseDiag').textContent = `Train: ${bestTrain.diag}`);

    if(fTest){
      const testRows = await parseWithPapa(fTest, bestTrain.cfg.delimiter, bestTrain.cfg.quoteChar);
      state.rawTest = testRows.map(normalizeRow);
    } else state.rawTest=[];

    const headers = Object.keys(state.rawTrain[0] || {});
    const missingExpected = EXPECTED_MIN.filter(c=> !headers.includes(c));
    if(missingExpected.length){
      alert('Parsed, but some expected columns are missing: '+missingExpected.join(', ')+'\nTry different Delimiter/Quote or re-download Kaggle CSV.');
    }

    el('kTrain') && (el('kTrain').textContent = state.rawTrain.length);
    el('kTest')  && (el('kTest').textContent  = state.rawTest.length || '—');
    el('kMiss')  && (el('kMiss').textContent  = roughMissingPct(state.rawTrain) + '%');

    previewTable(state.rawTrain);
    await edaCharts(state.rawTrain);
  }catch(err){ console.error(err); alert('Failed to load CSV. Try different Delimiter/Quote settings.'); }
}

function onPreprocess(){
  try{
    if(!state.rawTrain.length){ alert('Load train.csv first'); return; }
    state.pre = buildPreprocessor(state.rawTrain);
    const {train,val} = stratifiedSplit(state.rawTrain, 0.2);
    const tTrain = tensorize(train, state.pre);
    const tVal   = tensorize(val,   state.pre);
    state.xsTrain=tTrain.xs; state.ysTrain=tTrain.ys;
    state.xsVal=tVal.xs;     state.ysVal=tVal.ys;
    const info=[
      `Features count: ${tTrain.nFeat}`,
      `Train shape: ${state.xsTrain.shape}  |  Val shape: ${state.xsVal.shape}`,
      `Impute Age: median = ${state.pre.ageMed}`,
      `Impute Embarked: mode = ${state.pre.embMode}`,
      `Standardize Age/Fare with TRAIN means/stdevs` ,
      `One‑hot: Sex=[${state.pre.sexCats}], Pclass=[${state.pre.pclassCats}], Embarked=[${state.pre.embCats}]`,
      `Engineered: FamilySize=${state.pre.useFamily?'ON':'OFF'}, IsAlone=${state.pre.useAlone?'ON':'OFF'}`
    ].join('\n');
    el('preInfo') && (el('preInfo').textContent = info);
  }catch(err){ console.error(err); alert('Preprocessing failed: ' + (err?.message || err)); }
}

function onBuild(){
  try{
    if(!state.xsTrain){ alert('Run Preprocessing first'); return; }
    state.model = buildModel(state.xsTrain.shape[1]);
    alert('Model built. Click "Show Summary" to view layers.');
  }catch(err){ console.error(err); alert('Build failed.'); }
}

function onSummary(){ if(!state.model){ alert('Build the model first'); return; } el('modelSummary') && (el('modelSummary').textContent = summaryToText(state.model)); }

let stopFlag=false;
async function onTrain(){
  try{
    if(!state.model){ alert('Build the model first'); return; }
    stopFlag=false;

    // tfjs-vis (optional)
    const cbs = [];
    if (window.tfvis) {
      const surface = {name:'Training', tab:'Fit'};
      const fitCbs = tfvis.show.fitCallbacks(surface,
        ['loss','val_loss','acc','val_acc','accuracy','val_accuracy'],
        {callbacks:['onEpochEnd']}
      );
      cbs.push(fitCbs);
    }

    // Custom EarlyStopping with restore-best-weights (since tfjs doesn't support restoreBestWeights)
    function earlyStopWithRestore(patience=5, monitor='val_loss'){
      let best = Infinity, wait = 0, snapshot = null;
      return new tf.CustomCallback({
        onEpochEnd: async (epoch, logs)=>{
          const cur = logs?.[monitor];
          if (cur != null) {
            if (cur < best - 1e-12) {
              best = cur; wait = 0;
              if (snapshot) snapshot.forEach(t=>t.dispose());
              snapshot = state.model.getWeights().map(w=>w.clone());
            } else {
              wait += 1;
              if (wait >= patience) {
                if (snapshot) {
                  const restored = snapshot.map(w=>w.clone());
                  state.model.setWeights(restored);
                  snapshot.forEach(t=>t.dispose());
                  snapshot = null;
                }
                state.model.stopTraining = true;
              }
            }
          }
          if (stopFlag) state.model.stopTraining = true;
        }
      });
    }
    cbs.push(earlyStopWithRestore(5, 'val_loss'));

    await state.model.fit(state.xsTrain, state.ysTrain, {
      epochs:50, batchSize:32,
      validationData:[state.xsVal, state.ysVal],
      callbacks: cbs
    });

    const valPred = state.model.predict(state.xsVal).dataSync();
    state.valProbs = Float32Array.from(valPred);

    const yTrue = Array.from(state.ysVal.dataSync()).map(v=> +v);
    const {points, auc} = rocPoints(yTrue, state.valProbs, 200);
    await plotROC(el('rocPlot'), points, auc);
    updateThreshold(state.thresh);
  }catch(err){
    console.error(err);
    alert('Training failed: ' + (err?.message || err));
  }
}

function onStop(){ stopFlag=true; alert('Early stop requested. Will stop after this epoch.'); }

function updateThreshold(th){
  if(state.valProbs==null){ alert('Train first to compute validation probabilities.'); return; }
  state.thresh = th; el('thVal') && (el('thVal').textContent = th.toFixed(2));
  const yTrue = Array.from(state.ysVal.dataSync()).map(v=> +v);
  const st = confusionStats(yTrue, state.valProbs, th);
  showConfMat(el('confMat'), st);
  el('prf') && (el('prf').textContent = `Precision: ${(st.prec*100).toFixed(2)}%\nRecall: ${(st.rec*100).toFixed(2)}%\nF1: ${st.f1.toFixed(4)}\nTP:${st.TP}  FP:${st.FP}  TN:${st.TN}  FN:${st.FN}`);
}

async function onPredict(){
  try{
    if(!state.model){ alert('Train the model first'); return; }
    if(!state.rawTest.length){ alert('Load test.csv'); return; }
    const X = state.rawTest.map(state.pre.mapRow);
    const xs = tf.tensor2d(X, [X.length, state.pre.featLen]);
    const probs = state.model.predict(xs).dataSync();
    state.testProbs = Float32Array.from(probs);
    el('predInfo') && (el('predInfo').textContent = `Predicted ${state.rawTest.length} rows. You can now download submission.csv or probabilities.csv.`);
  }catch(err){ console.error(err); alert('Prediction failed.'); }
}

function onDownloadSubmission(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const th=state.thresh;
    const out = state.rawTest.map((r,i)=> ({ PassengerId:r[SCHEMA.id], Survived:(state.testProbs[i] >= th ? 1 : 0) }));
    downloadCSV('submission.csv', out);
  }catch(err){ console.error(err); alert('Download failed.'); }
}

function onDownloadProbs(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const out = state.rawTest.map((r,i)=> ({ PassengerId:r[SCHEMA.id], ProbSurvived:state.testProbs[i] }));
    downloadCSV('probabilities.csv', out);
  }catch(err){ console.error(err); alert('Download failed.'); }
}

async function onSaveModel(){ try{ if(!state.model){ alert('Train the model first'); return; } await state.model.save('downloads://titanic-tfjs'); } catch(err){ console.error(err); alert('Save failed.'); } }

// ----------------------------- Wire Up -----------------------------
window.addEventListener('DOMContentLoaded', ()=>{
  el('btnLoad')      && el('btnLoad').addEventListener('click', onLoadFiles);
  el('btnPre')       && el('btnPre').addEventListener('click', onPreprocess);
  el('btnBuild')     && el('btnBuild').addEventListener('click', onBuild);
  el('btnSummary')   && el('btnSummary').addEventListener('click', onSummary);
  el('btnTrain')     && el('btnTrain').addEventListener('click', onTrain);
  el('btnStop')      && el('btnStop').addEventListener('click', onStop);
  el('thSlider')     && el('thSlider').addEventListener('input', e=> updateThreshold(+e.target.value));
  el('btnPredict')   && el('btnPredict').addEventListener('click', onPredict);
  el('btnSub')       && el('btnSub').addEventListener('click', onDownloadSubmission);
  el('btnProb')      && el('btnProb').addEventListener('click', onDownloadProbs);
  el('btnSaveModel') && el('btnSaveModel').addEventListener('click', onSaveModel);
});
