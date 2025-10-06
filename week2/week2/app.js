/* =========================================================================
   Titanic Binary Classifier — TensorFlow.js
   FINAL (V3): robust CSV parsing with smart-quote normalization + validation
   ======================================================================= */

/* ---------------------------- Global State ---------------------------- */
const S = {
  rawTrain: [],
  rawTest: [],
  pre: null,
  xsTrain: null, ysTrain: null,
  xsVal: null,   ysVal: null,
  model: null,
  valProbs: null,
  testProbs: null,
  thresh: 0.5
};

const SCHEMA = {
  id: 'PassengerId',
  target: 'Survived',
  features: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
};

const $ = id => document.getElementById(id);

/* ------------------------------ Backend ------------------------------ */
(async () => { try { await tf.setBackend('cpu'); } catch {} await tf.ready(); })();

/* ------------------------------ Helpers ------------------------------ */
function previewTable(rows, n=8){
  const el = $('previewTable'); if(!rows.length){ el.innerHTML=''; return; }
  const cols = Object.keys(rows[0]);
  const head = '<thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
  const body = '<tbody>'+rows.slice(0,n).map(r=>'<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>').join('')+'</tbody>';
  el.innerHTML = `<table>${head}${body}</table>`;
}
const fin = (x,d=0)=>Number.isFinite(x)?x:d;
const mean = a=>{const b=a.filter(Number.isFinite);return b.length?b.reduce((s,x)=>s+x,0)/b.length:0};
const sd = a=>{const b=a.filter(Number.isFinite);if(b.length<2)return 0;const m=mean(b);return Math.sqrt(b.reduce((s,x)=>s+(x-m)**2,0)/(b.length-1))};
const median = a=>{const b=a.filter(v=>v!=null&&!Number.isNaN(+v)).map(Number).sort((x,y)=>x-y); if(!b.length)return null; const m=Math.floor(b.length/2); return b.length%2?b[m]:(b[m-1]+b[m])/2;};
const mode = a=>{const m=new Map();let best=null,cnt=0;for(const v of a){if(v==null||v==='')continue;const c=(m.get(v)||0)+1;m.set(v,c);if(c>cnt){cnt=c;best=v}}return best};
const oneHot=(v,c)=>{const r=new Array(c.length).fill(0);const i=c.indexOf(v);if(i>=0)r[i]=1;return r};
function missingPct(rows){ if(!rows.length)return 100; const cols=Object.keys(rows[0]); let miss=0,total=rows.length*cols.length; for(const r of rows)for(const c of cols){const v=r[c]; if(v==null||v==='')miss++;} return +(100*miss/total).toFixed(1); }
function normalizeRow(r){const o={};for(const[k,v]of Object.entries(r)){o[k]=(v===''||v===undefined)?null:(typeof v==='string'?v.trim():v);}return o}

/* ----------------------------- CSV Loader ---------------------------- */
/** Normalize text:
 *  - remove BOM
 *  - unify newline
 *  - convert smart quotes “ ” „ ‟ to "
 */
function normalizeCSVText(txt){
  return txt
    .replace(/\uFEFF/g,'')
    .replace(/\r\n/g,'\n')
    .replace(/[\u201C\u201D\u201E\u201F]/g,'"')   // smart double quotes → "
    .replace(/[\u2018\u2019]/g,"'");            // smart single quotes → '
}

function looksCorrect(rows){
  if(!rows.length) return false;
  const n = Math.min(rows.length, 120);
  const sample = rows.slice(0,n);
  const sexOK = sample.filter(r=>/^(male|female)$/i.test(r.Sex)).length/n > 0.8;
  const ageOK = sample.filter(r=>typeof r.Age==='number' && Number.isFinite(r.Age)).length/n > 0.8;
  const extraBad = sample.some(r => r && r.__parsed_extra !== undefined);
  return sexOK && ageOK && !extraBad;
}

function parseWithPapaText(text, delimiter=',', quoteChar='"'){
  return new Promise((resolve,reject)=>{
    Papa.parse(text,{
      header:true,dynamicTyping:true,skipEmptyLines:'greedy',
      delimiter, quoteChar, escapeChar: '"',
      complete:r=>resolve(r.data.map(normalizeRow)), error:reject
    });
  });
}

/** Robust parser:
 *  1) Read as text, normalize quotes/newlines.
 *  2) Try a small grid of (delimiter × quoteChar).
 *  3) Validate (Sex/ Age / no __parsed_extra).
 *  4) Pick first passing config; else best score fallback.
 */
async function robustParseFile(file){
  let text = await file.text();
  text = normalizeCSVText(text);

  const delims = [',',';','\t','|'];
  const quotes = ['"']; // after normalization, double-quote is what we want
  let best = {score:-Infinity, rows:[], cfg:null};

  function score(rows){
    if(!rows.length) return -1e9;
    const n=Math.min(rows.length,120), sample=rows.slice(0,n);
    const sex = sample.filter(r=>/^(male|female)$/i.test(r.Sex)).length/n;
    const age = sample.filter(r=>typeof r.Age==='number'&&Number.isFinite(r.Age)).length/n;
    const extra = sample.some(r=>r && r.__parsed_extra!==undefined) ? -1 : 0;
    return 5*sex + 5*age + 1*extra; // extra penalizes
  }

  for(const d of delims){
    for(const q of quotes){
      const rows = await parseWithPapaText(text,d,q);
      const sc = score(rows);
      if(sc > best.score){ best = {score:sc, rows, cfg:{delimiter:d, quoteChar:q}}; }
      if(looksCorrect(rows)) return rows.map(r=>{delete r.__parsed_extra; return r;});
    }
  }
  // fallback: keep the best and strip extras
  best.rows.forEach(r=>{if(r && '__parsed_extra' in r) delete r.__parsed_extra;});
  return best.rows;
}

/* --------------------------- Preprocessing --------------------------- */
function buildPreprocessor(trainRows){
  const useFamily = $('featFamily')?.checked ?? true;
  const useAlone  = $('featAlone')?.checked ?? true;

  const ageMed = Number(median(trainRows.map(r=>r.Age))) || 30;
  const embMode = mode(trainRows.map(r=>r.Embarked)) ?? 'S';

  const sexCats=['female','male'], pclassCats=[1,2,3], embCats=['C','Q','S','UNKNOWN'];

  const ageVals=trainRows.map(r=>fin((r.Age!=null&&!isNaN(+r.Age))?+r.Age:ageMed,ageMed));
  const fareVals=trainRows.map(r=>fin((r.Fare!=null&&!isNaN(+r.Fare))?+r.Fare:0,0));
  const muA=mean(ageVals), sdA=sd(ageVals), muF=mean(fareVals), sdF=sd(fareVals);

  function mapRow(r){
    if('__parsed_extra' in r) delete r.__parsed_extra;

    const age=(r.Age!=null&&!isNaN(+r.Age))?+r.Age:ageMed;
    const emb=(r.Embarked==null||r.Embarked==='')?'UNKNOWN':r.Embarked;
    const fare=(r.Fare!=null&&!isNaN(+r.Fare))?+r.Fare:0;
    const fam=(+r.SibSp||0)+(+r.Parch||0)+1, alone=fam===1?1:0;
    const ageZ=sdA? (age-muA)/sdA : 0, fareZ=sdF? (fare-muF)/sdF : 0;

    let f=[
      ageZ, fareZ,
      ...oneHot(r.Sex, sexCats),
      ...oneHot(+r.Pclass, pclassCats),
      ...oneHot(emb, embCats)
    ];
    if(useFamily) f.push(fam);
    if(useAlone)  f.push(alone);
    return f.map(x=>fin(+x,0));
  }

  return { mapRow, featLen: mapRow(trainRows[0]||{}).length,
           ageMed, embMode, useFamily, useAlone };
}

function tensorize(rows, pre){
  const X=[], Y=[];
  for(const r of rows){
    const f=pre.mapRow(r);
    if(!f.every(Number.isFinite)) continue;
    X.push(f);
    if(SCHEMA.target in r) Y.push(+r[SCHEMA.target]);
  }
  if(!X.length) throw new Error('No valid rows after preprocessing.');
  return {
    xs: tf.tensor2d(X,[X.length, pre.featLen],'float32'),
    ys: Y.length? tf.tensor2d(Y,[Y.length,1],'float32') : null
  };
}

function stratifiedSplit(rows, ratio=0.2){
  const z=rows.filter(r=>+r[SCHEMA.target]===0);
  const o=rows.filter(r=>+r[SCHEMA.target]===1);
  const cut=g=>{const a=g.slice(); tf.util.shuffle(a); const n=Math.max(1,Math.floor(a.length*ratio)); return {val:a.slice(0,n), train:a.slice(n)};}
  const A=cut(z), B=cut(o);
  const train=A.train.concat(B.train), val=A.val.concat(B.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  return {train,val};
}

/* ------------------------------- Model ------------------------------- */
function buildModel(inputDim){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:16,activation:'relu',inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  m.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  return m;
}
function modelSummaryText(m){ const L=[]; m.summary(undefined,undefined,s=>L.push(s)); return L.join('\n'); }

/* --------------------------- Metrics & ROC --------------------------- */
function rocPoints(yTrue,yProb,steps=200){
  const T=[]; for(let i=0;i<=steps;i++) T.push(i/steps);
  const pts=T.map(th=>{
    let TP=0,FP=0,TN=0,FN=0;
    for(let i=0;i<yTrue.length;i++){const y=yTrue[i],p=yProb[i]>=th?1:0;
      if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++;
      else if(y===0&&p===0)TN++; else FN++; }
    const TPR=TP/(TP+FN||1), FPR=FP/(FP+TN||1); return {x:FPR,y:TPR};
  });
  const s=pts.slice().sort((a,b)=>a.x-b.x); let auc=0;
  for(let i=1;i<s.length;i++){const a=s[i-1],b=s[i]; auc+=(b.x-a.x)*(a.y+b.y)/2;}
  return {points:s, auc};
}
function drawROC(cv,pts){
  const ctx=cv.getContext('2d'),W=cv.width,H=cv.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#0f1628'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#233350'; for(let i=0;i<=5;i++){const x=40+i*(W-60)/5;ctx.beginPath();ctx.moveTo(x,H-30);ctx.lineTo(x,20);ctx.stroke();}
  for(let i=0;i<=5;i++){const y=20+i*(H-50)/5;ctx.beginPath();ctx.moveTo(40,y);ctx.lineTo(W-20,y);ctx.stroke();}
  ctx.strokeStyle='#3a4760'; ctx.beginPath(); ctx.moveTo(40,H-30); ctx.lineTo(W-20,20); ctx.stroke();
  ctx.strokeStyle='#8aa3ff'; ctx.lineWidth=2; ctx.beginPath();
  pts.forEach((p,i)=>{const x=40+p.x*(W-60), y=H-30-p.y*(H-50); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);}); ctx.stroke();
}

/* --------------------------- Early Stopping -------------------------- */
let stopFlag=false;
function earlyStopWithRestore(pat=5, key='val_loss'){
  let best=Infinity, wait=0, snap=null;
  return new tf.CustomCallback({
    onBatchEnd: async()=>{ await new Promise(r=>setTimeout(r,0)); },
    onEpochEnd: async(_e,logs)=>{
      await tf.nextFrame();
      const cur=logs?.[key];
      if(cur!=null){
        if(cur<best-1e-12){ best=cur; wait=0; if(snap) snap.forEach(t=>t.dispose()); snap=S.model.getWeights().map(w=>w.clone()); }
        else if(++wait>=pat){ if(snap){S.model.setWeights(snap); snap=null;} S.model.stopTraining=true; }
      }
      if(stopFlag) S.model.stopTraining=true;
    }
  });
}

/* ----------------------------- UI: Load ------------------------------ */
async function onLoadFiles(){
  try{
    const fT=$('trainFile').files[0]; if(!fT){ alert('Please choose train.csv'); return; }
    const fX=$('testFile').files[0]||null;

    S.rawTrain = (await robustParseFile(fT)).map(r=>normalizeRow(r));
    S.rawTest  = fX ? (await robustParseFile(fX)).map(r=>normalizeRow(r)) : [];

    $('kTrain').textContent = S.rawTrain.length;
    $('kTest').textContent  = S.rawTest.length || '—';
    $('kMiss').textContent  = missingPct(S.rawTrain)+'%';

    previewTable(S.rawTrain);
  }catch(e){
    console.error(e); alert('Failed to load CSV: '+(e?.message||e));
  }
}

/* --------------------------- UI: Preprocess -------------------------- */
function onPreprocess(){
  try{
    if(!S.rawTrain.length){ alert('Load train.csv first'); return; }
    S.pre = buildPreprocessor(S.rawTrain);
    const {train,val} = stratifiedSplit(S.rawTrain,0.2);
    const tTr = tensorize(train,S.pre); const tVa = tensorize(val,S.pre);
    S.xsTrain=tTr.xs; S.ysTrain=tTr.ys; S.xsVal=tVa.xs; S.ysVal=tVa.ys;

    $('preInfo').textContent = [
      `features: ${S.pre.featLen}`,
      `Train: ${S.xsTrain.shape} | Val: ${S.xsVal.shape}`,
      `Impute Age median=${S.pre.ageMed} | Embarked mode=${S.pre.embMode}`,
      `One-hot: Sex, Pclass, Embarked | Engineered: FamilySize=${S.pre.useFamily}, IsAlone=${S.pre.useAlone}`
    ].join('\n');

  }catch(e){ console.error(e); alert('Preprocessing failed: '+(e?.message||e)); }
}

/* ---------------------------- UI: Model ------------------------------ */
function onBuild(){
  try{
    if(!S.xsTrain){ alert('Run Preprocessing first'); return; }
    S.model = buildModel(S.xsTrain.shape[1]);
    $('modelSummary').textContent = 'Model built. Click "Show Summary" to view layers.';
  }catch(e){ console.error(e); alert('Build failed: '+(e?.message||e)); }
}
function onSummary(){
  try{
    if(!S.model){ alert('Build the model first'); return; }
    $('modelSummary').textContent = modelSummaryText(S.model);
  }catch(e){ console.error(e); alert('Summary failed: '+(e?.message||e)); }
}

/* --------------------------- UI: Training ---------------------------- */
async function onTrain(){
  try{
    if(!S.model){ alert('Build the model first'); return; }
    if(!S.xsTrain){ alert('Run Preprocessing first'); return; }
    stopFlag=false; $('trainLog').textContent='';

    const cb = earlyStopWithRestore(5,'val_loss');

    await S.model.fit(S.xsTrain,S.ysTrain,{
      epochs:40,batchSize:16,validationData:[S.xsVal,S.ysVal],
      callbacks:[{
        onEpochEnd: async(ep,logs)=>{
          $('trainLog').textContent += `epoch ${ep+1}: loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} acc=${(logs.acc??logs.accuracy??0).toFixed(4)}\n`;
          await cb.onEpochEnd?.(ep,logs);
        },
        onBatchEnd: async(b,l)=>{ await cb.onBatchEnd?.(b,l); }
      }]
    });

    const val = tf.tidy(()=>S.model.predict(S.xsVal).dataSync());
    S.valProbs = Float32Array.from(val);
    const yTrue = Array.from(S.ysVal.dataSync()).map(v=>+v);
    const {points,auc} = rocPoints(yTrue,S.valProbs,200);
    drawROC($('rocCanvas'),points); $('aucText').textContent = `AUC = ${auc.toFixed(4)}`;
    updateThreshold(S.thresh);

  }catch(e){ console.error(e); alert('Training failed: '+(e?.message||e)); }
}
function onStop(){ stopFlag=true; alert('Early stop requested.'); }

/* ----------------------------- Metrics UI ---------------------------- */
function confusion(yTrue,yProb,th){
  let TP=0,FP=0,TN=0,FN=0; for(let i=0;i<yTrue.length;i++){const y=yTrue[i],p=yProb[i]>=th?1:0;
    if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++; }
  const prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=(2*prec*rec)/((prec+rec)||1);
  return {TP,FP,TN,FN,prec,rec,f1};
}
function updateThreshold(th){
  S.thresh=+th; $('thVal').textContent=(+th).toFixed(2);
  if(!S.valProbs) return;
  const yTrue = Array.from(S.ysVal.dataSync()).map(v=>+v);
  const st = confusion(yTrue,S.valProbs,S.thresh);
  $('cmTP').textContent=st.TP; $('cmFN').textContent=st.FN; $('cmFP').textContent=st.FP; $('cmTN').textContent=st.TN;
  $('prf').textContent = `Precision: ${(st.prec*100).toFixed(2)}%\nRecall: ${(st.rec*100).toFixed(2)}%\nF1: ${st.f1.toFixed(4)}`;
}

/* ------------------------ Predict & Export UI ------------------------ */
async function onPredict(){
  try{
    if(!S.model){ alert('Train the model first'); return; }
    if(!S.rawTest.length){ alert('Load test.csv'); return; }
    const out=tf.tidy(()=>{const X=S.rawTest.map(S.pre.mapRow); const xs=tf.tensor2d(X,[X.length,S.pre.featLen],'float32'); const p=S.model.predict(xs).dataSync(); xs.dispose(); return p;});
    S.testProbs=Float32Array.from(out); $('predInfo').textContent=`Predicted ${S.rawTest.length} rows.`;
  }catch(e){ console.error(e); alert('Prediction failed: '+(e?.message||e)); }
}
function downloadCSV(name,rows){
  if(!rows.length) return; const cols=Object.keys(rows[0]);
  const esc=v=>{if(v==null)return'';const s=String(v);return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s};
  const csv=[cols.join(',')].concat(rows.map(r=>cols.map(c=>esc(r[c])).join(','))).join('\n');
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'}); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}
function onDownloadSubmission(){
  if(!S.testProbs){ alert('Run Predict first'); return; }
  const out=S.rawTest.map((r,i)=>({[SCHEMA.id]:r[SCHEMA.id],[SCHEMA.target]: (S.testProbs[i]>=S.thresh?1:0)}));
  downloadCSV('submission.csv',out);
}
function onDownloadProbs(){
  if(!S.testProbs){ alert('Run Predict first'); return; }
  const out=S.rawTest.map((r,i)=>({[SCHEMA.id]:r[SCHEMA.id],ProbSurvived:S.testProbs[i]}));
  downloadCSV('probabilities.csv',out);
}
async function onSaveModel(){ if(!S.model){ alert('Train the model first'); return; } await S.model.save('downloads://titanic-tfjs'); }

/* ----------------------------- Wire Events --------------------------- */
window.addEventListener('DOMContentLoaded', ()=>{
  $('btnLoad').addEventListener('click', onLoadFiles);
  $('btnPre').addEventListener('click', onPreprocess);
  $('btnBuild').addEventListener('click', onBuild);
  $('btnSummary').addEventListener('click', onSummary);
  $('btnTrain').addEventListener('click', onTrain);
  $('btnStop').addEventListener('click', onStop);
  $('thSlider').addEventListener('input', e=>updateThreshold(+e.target.value));
  $('btnPredict').addEventListener('click', onPredict);
  $('btnSub').addEventListener('click', onDownloadSubmission);
  $('btnProb').addEventListener('click', onDownloadProbs);
  $('btnSaveModel').addEventListener('click', onSaveModel);
});
