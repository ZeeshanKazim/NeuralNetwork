/* Titanic Binary Classifier — FINAL (stable UI, working summary, safe training)
   - Robust CSV parse + normalization
   - Safe preprocessing (finite features, fixed width)
   - 2-D labels to match model output
   - Early stopping with best-weight restore (custom)
   - tfjs-vis optional; yields to UI each epoch
*/

const state = {
  rawTrain: [], rawTest: [],
  pre: null,
  xsTrain: null, ysTrain: null,
  xsVal: null,   ysVal: null,
  model: null,
  valProbs: null, testProbs: null,
  thresh: 0.5
};

const SCHEMA = {
  target: 'Survived',
  id: 'PassengerId',
  cols: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],
};

const EXPECTED_MIN = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
const NICE_TO_HAVE = ['Name'];

const el  = id => document.getElementById(id);
const opt = id => document.getElementById(id) || null;

// ---------- CSV parsing ----------
function parseWithPapa(file, delimiter, quoteChar){
  return new Promise((resolve,reject)=> {
    Papa.parse(file, {
      header:true, dynamicTyping:true, skipEmptyLines:'greedy',
      delimiter, quoteChar,
      complete: r => resolve(r.data),
      error: reject
    });
  });
}
function roughMissingPct(rows){
  if(!rows.length) return 100;
  const cols = Object.keys(rows[0]);
  let miss=0, total=rows.length*cols.length;
  for(const r of rows){ for(const c of cols){ const v=r[c]; if(v===''||v==null||v===undefined) miss++; } }
  return +(100*miss/total).toFixed(1);
}
function scoreParsed(rows, d, q){
  if(!rows.length) return {score:-1e9, diag:'No rows'};
  const headers = Object.keys(rows[0]); const S = new Set(headers);
  let have=0; for(const c of EXPECTED_MIN) if(S.has(c)) have++;
  const expectedHit = have/EXPECTED_MIN.length;
  const widthCnt={}; for(const r of rows){ const w=Object.keys(r).length; widthCnt[w]=(widthCnt[w]||0)+1; }
  const modeW = Object.entries(widthCnt).sort((a,b)=>b[1]-a[1])[0];
  const consistency = modeW? modeW[1]/rows.length : 0;
  const bonus = NICE_TO_HAVE.every(c=>S.has(c)) ? 0.1 : 0;
  const missPenalty = -0.2*(roughMissingPct(rows)/100);
  const score = 3*expectedHit + 2*consistency + bonus + missPenalty;
  const human = d==='\t' ? '\\t' : d;
  const diag = `Parsed with delimiter='${human}', quote='${q}' | expectedHit=${(100*expectedHit).toFixed(0)}% | rowConsistency=${(100*consistency).toFixed(0)}% | ${bonus>0?'Name:OK | ':''}missing≈${roughMissingPct(rows)}%`;
  return {score, diag};
}
async function robustParseCSV(file){
  const delims=[',',';','\t'], quotes=['"',"\'"];
  let best={score:-Infinity, rows:[], cfg:null, diag:''};
  for(const d of delims){
    for(const q of quotes){
      const rows = await parseWithPapa(file, d, q);
      const {score, diag} = scoreParsed(rows, d, q);
      if(score>best.score) best={score, rows, cfg:{delimiter:d, quoteChar:q}, diag};
    }
  }
  return best;
}

// ---------- normalization ----------
function normalizeRow(row){
  const out={};
  for(const [k,v] of Object.entries(row)){
    if(v===''){ out[k]=null; continue; }
    if(typeof v==='string'){ const t=v.trim(); out[k]=t===''? null : t; }
    else out[k]=v;
  }
  return out;
}

// ---------- quick EDA ----------
async function edaCharts(rows){
  if(!rows.length || !window.tfvis) return;
  const bySex = ['female','male'].map(s=>{
    const g = rows.filter(r=> r[SCHEMA.target]!=null && r.Sex===s);
    const rate = g.length? g.filter(r=> +r[SCHEMA.target]===1).length/g.length : 0;
    return {index:s, value:+(100*rate).toFixed(1)};
  });
  await tfvis.render.barchart(el('edaSex'), bySex, {xLabel:'Sex', yLabel:'Survival %', width:480, height:220});

  const byClass = [1,2,3].map(c=>{
    const g = rows.filter(r=> r[SCHEMA.target]!=null && +r.Pclass===c);
    const rate = g.length? g.filter(r=> +r[SCHEMA.target]===1).length/g.length : 0;
    return {index:String(c), value:+(100*rate).toFixed(1)};
  });
  await tfvis.render.barchart(el('edaClass'), byClass, {xLabel:'Pclass', yLabel:'Survival %', width:480, height:220});
}
function previewTable(rows, limit=8){
  if(!rows.length){ el('previewTable').innerHTML=''; return; }
  const cols = Object.keys(rows[0]);
  const head = '<thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
  const body = '<tbody>' + rows.slice(0,limit).map(r=>'<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>').join('') + '</tbody>';
  el('previewTable').innerHTML = `<table>${head}${body}</table>`;
}

// ---------- preprocessing ----------
function median(arr){ const a=arr.filter(v=>v!=null&&!Number.isNaN(+v)).map(Number).sort((x,y)=>x-y); if(!a.length) return null; const m=Math.floor(a.length/2); return a.length%2?a[m]:(a[m-1]+a[m])/2; }
function mode(arr){ const m=new Map(); let best=null,cnt=0; for(const v of arr){ if(v==null||v==='') continue; const c=(m.get(v)||0)+1; m.set(v,c); if(c>cnt){cnt=c;best=v;} } return best; }
function oneHot(v, cats){ const a=new Array(cats.length).fill(0); const i=cats.indexOf(v); if(i>=0) a[i]=1; return a; }
const jsMean = a => { const b=a.filter(Number.isFinite); return b.length? b.reduce((s,x)=>s+x,0)/b.length : 0; };
const jsStd  = a => { const b=a.filter(Number.isFinite); if(b.length<2) return 0; const m=jsMean(b); const v=b.reduce((s,x)=>s+(x-m)**2,0)/(b.length-1); return Math.sqrt(v); };
const clampFinite = (x,def=0)=> Number.isFinite(x)? x : def;

function buildPreprocessor(trainRows){
  const ageMed = Number.isFinite(median(trainRows.map(r=>r.Age))) ? median(trainRows.map(r=>r.Age)) : 30;
  const embMode = mode(trainRows.map(r=>r.Embarked)) ?? 'S';

  const sexCats=['female','male'], pclassCats=[1,2,3], embCats=['C','Q','S','UNKNOWN'];

  const ageVals  = trainRows.map(r=> { const v=(r.Age!=null&&!Number.isNaN(+r.Age))? +r.Age : ageMed; return clampFinite(v, ageMed); });
  const fareVals = trainRows.map(r=> { const v=(r.Fare!=null&&!Number.isNaN(+r.Fare))? +r.Fare : 0;   return clampFinite(v, 0); });

  const muAge=jsMean(ageVals), sdAge=jsStd(ageVals);
  const muFare=jsMean(fareVals), sdFare=jsStd(fareVals);

  const useFamily = !!(opt('featFamily')?.checked ?? true);
  const useAlone  = !!(opt('featAlone')?.checked  ?? true);

  const baseMap = (r)=>{
    const age  = (r.Age!=null&&!Number.isNaN(+r.Age))? +r.Age : ageMed;
    const emb  = (r.Embarked==null||r.Embarked==='')? 'UNKNOWN' : r.Embarked;
    const fare = (r.Fare!=null&&!Number.isNaN(+r.Fare))? +r.Fare : 0;

    const fam  = (+r.SibSp||0) + (+r.Parch||0) + 1;
    const alone= (fam===1)?1:0;

    const ageZ  = sdAge?  (age - muAge)/sdAge   : 0;
    const fareZ = sdFare? (fare - muFare)/sdFare: 0;

    const sexOH=oneHot(r.Sex, sexCats);
    const pclOH=oneHot(+r.Pclass, pclassCats);
    const embOH=oneHot(emb, embCats);

    let feats=[ageZ, fareZ, ...sexOH, ...pclOH, ...embOH];
    if(useFamily) feats.push(fam);
    if(useAlone)  feats.push(alone);
    return feats.map(x=>clampFinite(+x,0));
  };
  const FEAT_LEN = baseMap(trainRows[0] || {}).length;

  return {
    ageMed, embMode, sexCats, pclassCats, embCats, muAge, sdAge, muFare, sdFare,
    useFamily, useAlone, featLen: FEAT_LEN,
    mapRow: r => {
      const v=baseMap(r);
      if(v.length!==FEAT_LEN){
        if(v.length<FEAT_LEN) v.push(...Array(FEAT_LEN - v.length).fill(0));
        else v.length=FEAT_LEN;
      }
      return v;
    }
  };
}

function tensorize(rows, pre){
  const X=[], Y=[];
  for(const r of rows){
    const f = pre.mapRow(r);
    if(f.every(Number.isFinite)){
      X.push(f);
      if(SCHEMA.target in r) Y.push(+r[SCHEMA.target]);
    }
  }
  if(!X.length) throw new Error('No valid rows after preprocessing.');
  const xs=tf.tensor2d(X, [X.length, pre.featLen], 'float32');
  let ys=null; if(Y.length) ys=tf.tensor2d(Y, [Y.length, 1], 'float32');
  return {xs, ys, nFeat: pre.featLen};
}

function stratifiedSplit(rows, valRatio=0.2){
  const zeros=rows.filter(r=> +r[SCHEMA.target]===0);
  const ones =rows.filter(r=> +r[SCHEMA.target]===1);
  function split(g){ const a=g.slice(); tf.util.shuffle(a); const n=Math.max(1,Math.floor(a.length*valRatio)); return {val:a.slice(0,n), tr:a.slice(n)}; }
  const a=split(zeros), b=split(ones);
  const train=a.tr.concat(b.tr), val=a.val.concat(b.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  return {train, val};
}

// ---------- model ----------
function buildModel(inputDim){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  m.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  return m;
}
function summaryToText(model){
  const lines=[]; model.summary(undefined, undefined, (msg)=>lines.push(msg));
  return lines.join('\n');
}

// ---------- metrics ----------
function rocPoints(yTrue, yProb, steps=200){
  const T=[]; for(let i=0;i<=steps;i++) T.push(i/steps);
  const pts=T.map(th=>{ let TP=0,FP=0,TN=0,FN=0;
    for(let i=0;i<yTrue.length;i++){ const y=yTrue[i],p=yProb[i]>=th?1:0;
      if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++;
    }
    const TPR=TP/(TP+FN || 1), FPR=FP/(FP+TN || 1); return {x:FPR,y:TPR,th};
  });
  const s=pts.slice().sort((a,b)=>a.x-b.x); let auc=0;
  for(let i=1;i<s.length;i++){ const {x:x1,y:y1}=s[i-1], {x:x2,y:y2}=s[i]; auc+=(x2-x1)*(y1+y2)/2; }
  return {points:s, auc};
}
function confusionStats(yTrue, yProb, th){
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<yTrue.length;i++){ const y=yTrue[i],p=yProb[i]>=th?1:0;
    if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++;
  }
  const prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=(2*prec*rec)/((prec+rec)||1);
  return {TP,FP,TN,FN,prec,rec,f1};
}
async function plotROC(container, points, auc){
  if(!window.tfvis) return;
  const series=[{name:'ROC', values: points.map(p=>({x:p.x,y:p.y}))}];
  await tfvis.render.linechart(container, series, {width:520,height:300,xLabel:'FPR',yLabel:'TPR',seriesColors:['#8aa3ff']});
  el('prf').textContent = `AUC = ${auc.toFixed(4)}\nMove the slider to change threshold and see PR/F1.`;
}
function showConfMat(container, stats){
  if(!window.tfvis) return;
  const data=[[stats.TP,stats.FN],[stats.FP,stats.TN]];
  tfvis.render.confusionMatrix(container,{values:data,tickLabels:['1','0']},{shadeDiagonal:true,width:260,height:220});
}

// ---------- downloads ----------
function downloadCSV(filename, rows){
  if(!rows.length) return;
  const cols=Object.keys(rows[0]);
  const esc=v=>{ if(v==null) return ''; const s=String(v); return /[",\n]/.test(s)? '"'+s.replace(/"/g,'""')+'"' : s; };
  const csv=[cols.join(',')].concat(rows.map(r=>cols.map(c=>esc(r[c])).join(','))).join('\n');
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'}); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}

// ---------- UI actions ----------
async function onLoadFiles(){
  try{
    const fTrain=el('trainFile')?.files?.[0]; const fTest=el('testFile')?.files?.[0];
    if(!fTrain){ alert('Please choose train.csv'); return; }

    const best = await robustParseCSV(fTrain);
    state.rawTrain = best.rows.map(normalizeRow);
    if(fTest){
      const trows = await parseWithPapa(fTest, best.cfg.delimiter, best.cfg.quoteChar);
      state.rawTest = trows.map(normalizeRow);
    } else state.rawTest=[];

    const headers = Object.keys(state.rawTrain[0] || {});
    const missingExpected = EXPECTED_MIN.filter(c=>!headers.includes(c));
    if(missingExpected.length) alert('Parsed, but some expected columns are missing: '+missingExpected.join(', ')+'\nRe-download from Kaggle if needed.');

    el('kTrain').textContent = state.rawTrain.length;
    el('kTest').textContent  = state.rawTest.length || '—';
    el('kMiss').textContent  = roughMissingPct(state.rawTrain) + '%';

    previewTable(state.rawTrain);
    await edaCharts(state.rawTrain);
  }catch(err){
    console.error(err); alert('Failed to load CSV: ' + (err?.message || err));
  }
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
      `Standardize Age/Fare (train mean/stdev)`,
      `One-hot: Sex=[${state.pre.sexCats}], Pclass=[${state.pre.pclassCats}], Embarked=[${state.pre.embCats}]`,
      `Engineered: FamilySize=${state.pre.useFamily?'ON':'OFF'}, IsAlone=${state.pre.useAlone?'ON':'OFF'}`
    ].join('\n');
    el('preInfo').textContent = info;
  }catch(err){ console.error(err); alert('Preprocessing failed: '+(err?.message||err)); }
}

function onBuild(){
  try{
    if(!state.xsTrain){ alert('Run Preprocessing first'); return; }
    state.model = buildModel(state.xsTrain.shape[1]);
    el('modelSummary').textContent = 'Model built. Click "Show Summary" to view layers.';
  }catch(err){ console.error(err); alert('Build failed: '+(err?.message||err)); }
}
function onSummary(){
  try{
    if(!state.model){ alert('Build the model first'); return; }
    el('modelSummary').textContent = summaryToText(state.model);
  }catch(err){ console.error(err); alert('Summary failed: '+(err?.message||err)); }
}

let stopFlag=false;
function earlyStopWithRestore(patience=5, monitor='val_loss'){
  let best=Infinity, wait=0, snapshot=null;
  return new tf.CustomCallback({
    onEpochEnd: async (epoch, logs)=>{
      const cur = logs?.[monitor];
      // yield to UI so page stays responsive
      await tf.nextFrame();

      if(cur!=null){
        if(cur < best - 1e-12){
          best = cur; wait = 0;
          if(snapshot) snapshot.forEach(t=>t.dispose());
          snapshot = state.model.getWeights().map(w=>w.clone());
        }else{
          wait += 1;
          if(wait>=patience){
            if(snapshot){
              const restored = snapshot.map(w=>w.clone());
              state.model.setWeights(restored);
              snapshot.forEach(t=>t.dispose());
            }
            state.model.stopTraining = true;
          }
        }
      }
      if(stopFlag) state.model.stopTraining = true;
    }
  });
}

async function onTrain(){
  try{
    if(!state.model){ alert('Build the model first'); return; }
    stopFlag=false;

    const callbacks = [];
    if(window.tfvis){
      callbacks.push(tfvis.show.fitCallbacks({name:'Training',tab:'Fit'},
        ['loss','val_loss','acc','val_acc','accuracy','val_accuracy'],
        {callbacks:['onEpochEnd']}
      ));
    }
    callbacks.push(earlyStopWithRestore(5,'val_loss'));

    await state.model.fit(state.xsTrain, state.ysTrain, {
      epochs:50, batchSize:32,
      validationData:[state.xsVal, state.ysVal],
      callbacks
    });

    const valPred = state.model.predict(state.xsVal).dataSync();
    state.valProbs = Float32Array.from(valPred);

    const yTrue = Array.from(state.ysVal.dataSync()).map(v=>+v);
    const {points, auc} = rocPoints(yTrue, state.valProbs, 200);
    await plotROC(el('rocPlot'), points, auc);
    updateThreshold(state.thresh);
  }catch(err){ console.error(err); alert('Training failed: '+(err?.message||err)); }
}
function onStop(){ stopFlag=true; alert('Early stop requested (will stop after this epoch).'); }

function updateThreshold(th){
  if(state.valProbs==null){ el('thVal').textContent = th.toFixed(2); return; }
  state.thresh = th; el('thVal').textContent = th.toFixed(2);
  const yTrue = Array.from(state.ysVal.dataSync()).map(v=>+v);
  const st = confusionStats(yTrue, state.valProbs, th);
  showConfMat(el('confMat'), st);
  el('prf').textContent = `Precision: ${(st.prec*100).toFixed(2)}%\nRecall: ${(st.rec*100).toFixed(2)}%\nF1: ${st.f1.toFixed(4)}\nTP:${st.TP}  FP:${st.FP}  TN:${st.TN}  FN:${st.FN}`;
}

async function onPredict(){
  try{
    if(!state.model){ alert('Train the model first'); return; }
    if(!state.rawTest.length){ alert('Load test.csv'); return; }
    const X = state.rawTest.map(state.pre.mapRow);
    const xs = tf.tensor2d(X, [X.length, state.pre.featLen], 'float32');
    const probs = state.model.predict(xs).dataSync();
    state.testProbs = Float32Array.from(probs);
    el('predInfo').textContent = `Predicted ${state.rawTest.length} rows. You can now download submission.csv or probabilities.csv.`;
  }catch(err){ console.error(err); alert('Prediction failed: '+(err?.message||err)); }
}
function onDownloadSubmission(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const th=state.thresh;
    const out = state.rawTest.map((r,i)=> ({ PassengerId:r[SCHEMA.id], Survived:(state.testProbs[i] >= th ? 1 : 0) }));
    downloadCSV('submission.csv', out);
  }catch(err){ console.error(err); alert('Download failed: '+(err?.message||err)); }
}
function onDownloadProbs(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const out = state.rawTest.map((r,i)=> ({ PassengerId:r[SCHEMA.id], ProbSurvived:state.testProbs[i] }));
    downloadCSV('probabilities.csv', out);
  }catch(err){ console.error(err); alert('Download failed: '+(err?.message||err)); }
}
async function onSaveModel(){
  try{ if(!state.model){ alert('Train the model first'); return; } await state.model.save('downloads://titanic-tfjs'); }
  catch(err){ console.error(err); alert('Save failed: '+(err?.message||err)); }
}

// ---------- wire up ----------
window.addEventListener('DOMContentLoaded', ()=>{
  el('btnLoad').addEventListener('click', onLoadFiles);
  el('btnPre').addEventListener('click', onPreprocess);
  el('btnBuild').addEventListener('click', onBuild);
  el('btnSummary').addEventListener('click', onSummary);
  el('btnTrain').addEventListener('click', onTrain);
  el('btnStop').addEventListener('click', onStop);
  el('thSlider').addEventListener('input', e=> updateThreshold(+e.target.value));
  el('btnPredict').addEventListener('click', onPredict);
  el('btnSub').addEventListener('click', onDownloadSubmission);
  el('btnProb').addEventListener('click', onDownloadProbs);
  el('btnSaveModel').addEventListener('click', onSaveModel);
});
