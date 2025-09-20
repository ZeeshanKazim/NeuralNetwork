/*
  Titanic Binary Classifier — Week 2 (TensorFlow.js)
  Unified app.js with robust CSV parsing (auto-detect delimiter/quote) and optional manual overrides.
  - Works even if your index.html DOES NOT have the extra Delimiter/Quote controls.
  - If present, elements with ids #csvDelimiter, #csvQuote, #parseDiag will be used.
  - Schema: Target Survived (0/1). Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. Ignore: PassengerId.
*/

// ----------------------------- Global State -----------------------------
const state = {
  rawTrain: [],
  rawTest: [],
  pre: null,        // preprocessing artifacts
  xsTrain: null, ysTrain: null,
  xsVal: null, ysVal: null,
  model: null,
  valProbs: null,   // Float32Array of probabilities for validation set
  testProbs: null,  // Float32Array of probabilities for test set
  thresh: 0.5,
};

// Feature schema config (adjust if you reuse on another dataset)
const SCHEMA = {
  target: 'Survived',
  id: 'PassengerId',
  cols: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],
};

// Expected columns for Titanic (used only to validate correct parsing).
const EXPECTED_MIN = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
const NICE_TO_HAVE = ['Name']; // signals quotes around commas handled

// UI helpers
const el = (id)=> document.getElementById(id);
const opt = (id)=> document.getElementById(id) || null; // optional element

// ----------------------------- CSV Parsing (robust) -----------------------------
function parseWithPapa(file, delimiter, quoteChar){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: 'greedy',
      delimiter,      // ',', ';', or '\t'
      quoteChar,      // '"' or '\''
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
  if(!rows || !rows.length) return {score: -1e9, diag: 'No rows'};
  const headers = Object.keys(rows[0] ?? {});
  const headerSet = new Set(headers);
  // Expected columns present
  let have = 0; for(const c of EXPECTED_MIN) if(headerSet.has(c)) have++;
  const expectedHit = have/EXPECTED_MIN.length; // 0..1
  // Row width consistency
  const widthCounts = {};
  for(const r of rows){ const w = Object.keys(r).length; widthCounts[w] = (widthCounts[w]||0)+1; }
  const modeW = Object.entries(widthCounts).sort((a,b)=>b[1]-a[1])[0];
  const consistency = modeW ? modeW[1]/rows.length : 0; // 0..1
  // Bonus if Name column present (common comma field)
  const bonus = NICE_TO_HAVE.every(c => headerSet.has(c)) ? 0.1 : 0;
  // Light missingness penalty
  const missPct = roughMissingPct(rows)/100;
  const missPenalty = -0.2*missPct;
  const score = 3*expectedHit + 2*consistency + bonus + missPenalty;
  const diag = `Parsed with delimiter='${humanDelim(delimiter)}', quote='${quoteChar}' | `+
               `expectedHit=${(100*expectedHit).toFixed(0)}% | `+
               `rowConsistency=${(100*consistency).toFixed(0)}% | `+
               (bonus>0 ? 'Name:OK | ' : '')+
               `missing≈${(missPct*100).toFixed(1)}%`;
  return {score, diag};
}

async function robustParseCSV(file, manualDelim, manualQuote){
  const tryDelims = manualDelim && manualDelim !== 'auto' ? [manualDelim] : [',',';','\t'];
  const tryQuotes = manualQuote && manualQuote !== 'auto' ? [manualQuote] : ['"', "'"];
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

// ----------------------------- UI: Preview & EDA -----------------------------
function previewTable(rows, limit=8){
  if(!rows || !rows.length){ el('previewTable') && (el('previewTable').innerHTML = ''); return; }
  const cols = Object.keys(rows[0]);
  const head = '<thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
  const body = '<tbody>' + rows.slice(0,limit).map(r=>'<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>').join('') + '</tbody>';
  el('previewTable') && (el('previewTable').innerHTML = `<table>${head}${body}</table>`);
}

async function edaCharts(rows){
  if(!rows || !rows.length) return;
  if(!window.tfvis) return; // tfjs-vis is required for charts
  // Survival by Sex
  const bySex = ['female','male'].map(s=>{
    const g = rows.filter(r=> r[SCHEMA.target]!=null && r.Sex===s);
    const rate = g.length? g.filter(r=> +r[SCHEMA.target]===1).length / g.length : 0;
    return {index:s, value: +(100*rate).toFixed(1)};
  });
  el('edaSex') && await tfvis.render.barchart(el('edaSex'), bySex, { xLabel:'Sex', yLabel:'Survival %', width:450, height:220});
  // Survival by Pclass
  const byClass = [1,2,3].map(c=>{
    const g = rows.filter(r=> r[SCHEMA.target]!=null && +r.Pclass===c);
    const rate = g.length? g.filter(r=> +r[SCHEMA.target]===1).length / g.length : 0;
    return {index: c+'', value: +(100*rate).toFixed(1)};
  });
  el('edaClass') && await tfvis.render.barchart(el('edaClass'), byClass, { xLabel:'Pclass', yLabel:'Survival %', width:450, height:220});
}

// ----------------------------- Preprocessing -----------------------------
function median(arr){ const a=arr.filter(v=>v!=null && !Number.isNaN(v)).sort((x,y)=>x-y); if(!a.length) return null; const m=Math.floor(a.length/2); return a.length%2? a[m] : (a[m-1]+a[m])/2; }
function mode(arr){ const m=new Map(); let best=null, cnt=0; for(const v of arr){ if(v==null||v==='') continue; const c=(m.get(v)||0)+1; m.set(v,c); if(c>cnt){cnt=c;best=v;} } return best; }
function oneHot(value, cats){ const vec=new Array(cats.length).fill(0); const idx=cats.indexOf(value); if(idx>=0) vec[idx]=1; return vec; }
function standardize(x, mu, sd){ return sd? (x-mu)/sd : 0; }

function buildPreprocessor(trainRows){
  const ageMed = median(trainRows.map(r=> r.Age));
  const embMode = mode(trainRows.map(r=> r.Embarked));
  const sexCats = ['female','male'];
  const pclassCats = [1,2,3];
  const embCats = ['C','Q','S','UNKNOWN'];

  const ageVals = trainRows.map(r=> (r.Age!=null? r.Age : ageMed));
  const fareVals = trainRows.map(r=> (r.Fare!=null? r.Fare : 0));
  const muAge = tf.mean(ageVals).arraySync();
  const sdAge = tf.moments(ageVals).variance.sqrt().arraySync();
  const muFare = tf.mean(fareVals).arraySync();
  const sdFare = tf.moments(fareVals).variance.sqrt().arraySync();

  const useFamily = !!(opt('featFamily')?.checked ?? true);
  const useAlone  = !!(opt('featAlone')?.checked ?? true);

  return {
    ageMed, embMode, sexCats, pclassCats, embCats, muAge, sdAge, muFare, sdFare, useFamily, useAlone,
    mapRow: (r)=>{
      const age = (r.Age!=null && !Number.isNaN(r.Age)) ? r.Age : ageMed;
      const emb = (r.Embarked==null || r.Embarked==='') ? 'UNKNOWN' : r.Embarked;
      const fare = (r.Fare!=null && !Number.isNaN(r.Fare)) ? r.Fare : 0;
      const fam = (+r.SibSp||0) + (+r.Parch||0) + 1;
      const alone = (fam===1) ? 1 : 0;
      const ageZ = standardize(age, muAge, sdAge);
      const fareZ = standardize(fare, muFare, sdFare);
      const sexOH = oneHot(r.Sex, sexCats);
      const pclOH = oneHot(+r.Pclass, pclassCats);
      const embOH = oneHot(emb, embCats);
      const feats = [ageZ, fareZ, ...sexOH, ...pclOH, ...embOH];
      if(useFamily) feats.push(fam);
      if(useAlone) feats.push(alone);
      return feats;
    }
  };
}

function tensorize(rows, pre){
  const X = rows.map(pre.mapRow);
  const xs = tf.tensor2d(X);
  let ys = null;
  if(rows.length && SCHEMA.target in rows[0]){ ys = tf.tensor1d(rows.map(r=> +r[SCHEMA.target])); }
  return {xs, ys, nFeat: X[0]?.length || 0};
}

function stratifiedSplit(rows, valRatio=0.2){
  const zeros = rows.filter(r=> +r[SCHEMA.target]===0);
  const ones  = rows.filter(r=> +r[SCHEMA.target]===1);
  function splitGroup(g){ const arr=g.slice(); tf.util.shuffle(arr); const nVal=Math.max(1, Math.floor(arr.length*valRatio)); return {val:arr.slice(0,nVal), tr:arr.slice(nVal)}; }
  const a=splitGroup(zeros), b=splitGroup(ones);
  const train = a.tr.concat(b.tr), val = a.val.concat(b.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  return {train, val};
}

// ----------------------------- Model -----------------------------
function buildModel(inputDim){
  const model = tf.sequential();
  model.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputDim]}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  model.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  return model;
}

function summaryToText(model){
  const lines = [];
  model.summary( (line)=> lines.push(line) );
  return lines.join('\n');
}

// ----------------------------- Metrics: ROC/AUC & PRF -----------------------------
function rocPoints(yTrue, yProb, steps=200){
  const t = []; for(let i=0;i<=steps;i++) t.push(i/steps);
  const points = t.map(th=>{ let TP=0, FP=0, TN=0, FN=0; for(let i=0;i<yTrue.length;i++){ const y=yTrue[i], p=yProb[i]>=th?1:0; if(y===1&&p===1) TP++; else if(y===0&&p===1) FP++; else if(y===0&&p===0) TN++; else FN++; } const TPR=TP/(TP+FN||1); const FPR=FP/(FP+TN||1); return {x:FPR,y:TPR,th}; });
  const sorted = points.slice().sort((a,b)=> a.x-b.x);
  let auc=0; for(let i=1;i<sorted.length;i++){ const x1=sorted[i-1].x,y1=sorted[i-1].y,x2=sorted[i].x,y2=sorted[i].y; auc+=(x2-x1)*(y1+y2)/2; }
  return {points:sorted, auc};
}

function confusionStats(yTrue, yProb, th){
  let TP=0, FP=0, TN=0, FN=0;
  for(let i=0;i<yTrue.length;i++){
    const y=yTrue[i]; const p=yProb[i]>=th?1:0;
    if(y===1&&p===1) TP++; else if(y===0&&p===1) FP++; else if(y===0&&p===0) TN++; else FN++;
  }
  const prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=(2*prec*rec)/((prec+rec)||1);
  return {TP,FP,TN,FN,prec,rec,f1};
}

async function plotROC(container, points, auc){
  if(!window.tfvis) return;
  const series = [{name:'ROC', values: points.map(p=> ({x:p.x, y:p.y})) }];
  await tfvis.render.linechart(container, series, { width:520, height:300, xLabel:'FPR', yLabel:'TPR', seriesColors:['#8aa3ff'] });
  el('prf') && (el('prf').textContent = `AUC = ${auc.toFixed(4)}\nMove the slider to change threshold and see PR/F1.`);
}

function showConfMat(container, stats){
  if(!window.tfvis) return;
  const data = [[stats.TP, stats.FN],[stats.FP, stats.TN]]; // [[tp, fn],[fp, tn]]
  tfvis.render.confusionMatrix(container, {values:data, tickLabels:['1','0']}, { shadeDiagonal:true, width:260, height:220 });
}

// ----------------------------- Inference & Downloads -----------------------------
function downloadCSV(filename, rows){
  if(!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = v=>{ if(v==null) return ''; const s=String(v); return /[",\n]/.test(s) ? '"'+s.replace(/\"/g,'\"\"')+'"' : s; };
  const csv = [cols.join(',')].concat(rows.map(r=> cols.map(c=> esc(r[c])).join(','))).join('\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
}

// ----------------------------- Event Handlers -----------------------------
async function onLoadFiles(){
  try{
    const fTrain = el('trainFile')?.files?.[0];
    const fTest  = el('testFile')?.files?.[0];
    if(!fTrain){ alert('Please choose train.csv'); return; }

    const manualDelim = opt('csvDelimiter') ? opt('csvDelimiter').value : 'auto';
    const manualQuote = opt('csvQuote') ? opt('csvQuote').value : 'auto';

    const bestTrain = await robustParseCSV(fTrain, manualDelim, manualQuote);
    state.rawTrain = bestTrain.rows;
    opt('parseDiag') && (opt('parseDiag').textContent = `Train: ${bestTrain.diag}`);

    if(fTest){
      // Use SAME config for test to keep columns aligned
      const testRows = await parseWithPapa(fTest, bestTrain.cfg.delimiter, bestTrain.cfg.quoteChar);
      state.rawTest = testRows;
    } else {
      state.rawTest = [];
    }

    // Sanity: expected columns
    const headers = Object.keys(state.rawTrain[0] || {});
    const missingExpected = EXPECTED_MIN.filter(c => !headers.includes(c));
    if(missingExpected.length){
      alert('Parsed, but some expected columns are missing: ' + missingExpected.join(', ') + '\nTry changing Delimiter/Quote manually, then Load again.');
    }

    el('kTrain') && (el('kTrain').textContent = state.rawTrain.length);
    el('kTest')  && (el('kTest').textContent  = state.rawTest.length || '—');
    el('kMiss')  && (el('kMiss').textContent  = roughMissingPct(state.rawTrain) + '%');

    previewTable(state.rawTrain);
    await edaCharts(state.rawTrain);
  }catch(err){
    console.error(err);
    alert('Failed to load CSV. Try different Delimiter/Quote settings.');
  }
}

function onPreprocess(){
  try{
    if(!state.rawTrain.length){ alert('Load train.csv first'); return; }
    state.pre = buildPreprocessor(state.rawTrain);

    const {train, val} = stratifiedSplit(state.rawTrain, 0.2);
    const tTrain = tensorize(train, state.pre);
    const tVal   = tensorize(val,   state.pre);

    state.xsTrain = tTrain.xs; state.ysTrain = tTrain.ys;
    state.xsVal   = tVal.xs;   state.ysVal   = tVal.ys;

    const info = [
      `Features count: ${tTrain.nFeat}`,
      `Train shape: ${state.xsTrain.shape}  |  Val shape: ${state.xsVal.shape}`,
      `Impute Age: median = ${state.pre.ageMed?.toFixed?.(2)}`,
      `Impute Embarked: mode = ${state.pre.embMode}`,
      `Standardize Age/Fare with TRAIN means/stdevs`,
      `One‑hot: Sex=[${state.pre.sexCats}], Pclass=[${state.pre.pclassCats}], Embarked=[${state.pre.embCats}]`,
      `Engineered: FamilySize=${state.pre.useFamily?'ON':'OFF'}, IsAlone=${state.pre.useAlone?'ON':'OFF'}`
    ].join('\n');
    el('preInfo') && (el('preInfo').textContent = info);
  }catch(err){ console.error(err); alert('Preprocessing failed.'); }
}

function onBuild(){
  try{
    if(!state.xsTrain){ alert('Run Preprocessing first'); return; }
    state.model = buildModel(state.xsTrain.shape[1]);
    alert('Model built. Click "Show Summary" to view layers.');
  }catch(err){ console.error(err); alert('Build failed.'); }
}

function onSummary(){
  if(!state.model){ alert('Build the model first'); return; }
  el('modelSummary') && (el('modelSummary').textContent = summaryToText(state.model));
}

let stopFlag = false;
async function onTrain(){
  try{
    if(!state.model){ alert('Build the model first'); return; }
    stopFlag = false;
    const surface = { name:'Training', tab:'Fit' };
    const metrics = ['loss','val_loss','acc','val_acc'];
    const fitCallbacks = window.tfvis ? tfvis.show.fitCallbacks(surface, metrics, {callbacks:[ 'onEpochEnd' ]}) : {};

    const es = tf.callbacks.earlyStopping({ monitor:'val_loss', patience:5, restoreBestWeights:true });
    const stopCb = new tf.CustomCallback({ onEpochEnd: async()=>{ if(stopFlag){ state.model.stopTraining = true; } } });

    await state.model.fit(state.xsTrain, state.ysTrain, {
      epochs:50, batchSize:32, validationData:[state.xsVal, state.ysVal], callbacks:[fitCallbacks, es, stopCb]
    });

    const valPred = state.model.predict(state.xsVal).dataSync();
    state.valProbs = Float32Array.from(valPred);

    const yTrue = Array.from(state.ysVal.dataSync()).map(v=> +v);
    const {points, auc} = rocPoints(yTrue, state.valProbs, 200);
    await plotROC(el('rocPlot'), points, auc);

    updateThreshold(state.thresh);
  }catch(err){ console.error(err); alert('Training failed.'); }
}

function onStop(){ stopFlag = true; alert('Early stop requested. Will stop after current epoch.'); }

function updateThreshold(th){
  if(state.valProbs==null){ alert('Train first to compute validation probabilities.'); return; }
  state.thresh = th;
  el('thVal') && (el('thVal').textContent = th.toFixed(2));
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
    const xs = tf.tensor2d(X);
    const probs = state.model.predict(xs).dataSync();
    state.testProbs = Float32Array.from(probs);
    el('predInfo') && (el('predInfo').textContent = `Predicted ${state.rawTest.length} rows. You can now download submission.csv or probabilities.csv.`);
  }catch(err){ console.error(err); alert('Prediction failed.'); }
}

function onDownloadSubmission(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const th = state.thresh;
    const out = state.rawTest.map( (r,i)=> ({ PassengerId: r[SCHEMA.id], Survived: (state.testProbs[i] >= th ? 1 : 0) }) );
    downloadCSV('submission.csv', out);
  }catch(err){ console.error(err); alert('Download failed.'); }
}

function onDownloadProbs(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const out = state.rawTest.map( (r,i)=> ({ PassengerId:r[SCHEMA.id], ProbSurvived: state.testProbs[i] }) );
    downloadCSV('probabilities.csv', out);
  }catch(err){ console.error(err); alert('Download failed.'); }
}

async function onSaveModel(){
  try{
    if(!state.model){ alert('Train the model first'); return; }
    await state.model.save('downloads://titanic-tfjs');
  }catch(err){ console.error(err); alert('Save failed.'); }
}

// ----------------------------- Wire Up -----------------------------
window.addEventListener('DOMContentLoaded', ()=>{
  el('btnLoad')      && el('btnLoad').addEventListener('click', onLoadFiles);
  el('btnPre')       && el('btnPre').addEventListener('click', onPreprocess);
  el('btnBuild')     && el('btnBuild').addEventListener('click', onBuild);
  el('btnSummary')   && el('btnSummary').addEventListener('click', onSummary);
  el('btnTrain')     && el('btnTrain').addEventListener('click', onTrain);
  el('btnStop')      && el('btnStop').addEventListener('click', onStop);
  el('thSlider')     && el('thSlider').addEventListener('input', (e)=> updateThreshold(+e.target.value));
  el('btnPredict')   && el('btnPredict').addEventListener('click', onPredict);
  el('btnSub')       && el('btnSub').addEventListener('click', onDownloadSubmission);
  el('btnProb')      && el('btnProb').addEventListener('click', onDownloadProbs);
  el('btnSaveModel') && el('btnSaveModel').addEventListener('click', onSaveModel);
});
