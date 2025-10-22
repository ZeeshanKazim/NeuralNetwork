/* =========================================================================
   Airline Passenger Satisfaction — Wide & Deep (TF.js) + EDA (ECharts)
   FAST BUILD (GPU, fewer epochs, larger batch, smaller deep path)
   - WebGL backend preferred (set in index.html)
   - Deep path no longer duplicates categorical one-hots (smaller input)
   - Deep layers 32→8, dropout 0.1
   - fit(): epochs 25, batchSize 64, early stop patience 3
   - Multi-input passed as arrays everywhere ([wide, deep])
   - Concurrency guard, stop button, clear logs
   - New: EDA panel (missingness, categorical rates, histograms, correlation)
   ========================================================================= */

const S = {
  rawTrain: [], rawTest: [],
  map: null,
  xsWTr: null, xsDTr: null, ysTr: null,
  xsWVa: null, xsDVa: null, ysVa: null,
  model: null,
  valProbs: null, testProbs: null, testIDs: null,
  thresh: 0.5,
  isTraining: false,
  charts: {} // for ECharts instances (EDA)
};

const $ = id => document.getElementById(id);

/* ---------------- UI helpers ---------------- */
const TRAIN_BTNS = ['btnLoad','btnPre','btnBuild','btnSummary','btnTrain','btnStop','btnPredict','btnSub','btnProb','btnSaveModel'];
function setBusy(on){
  S.isTraining = !!on;
  TRAIN_BTNS.forEach(id=>{
    const el=$(id); if(!el) return;
    if(id==='btnStop'){ el.disabled = !on; } else { el.disabled = !!on; }
  });
  const dot=$('trainDot'), txt=$('trainText');
  if(dot&&txt){ dot.classList.toggle('busy', !!on); txt.textContent = on ? 'Training…' : 'Idle'; }
}
function appendLog(id, line){
  const el=$(id); if(!el) return;
  if(el.textContent==='—') el.textContent='';
  el.textContent += line;
  el.scrollTop = el.scrollHeight;
}

/* --------------- CSV parsing (Kaggle defaults) --------------- */
async function parseWithPapa(file, delimiter=',', quoteChar='"'){
  const text = (await file.text()).replace(/^\uFEFF/, '');
  return new Promise((resolve, reject)=>{
    Papa.parse(text, {
      header:true, dynamicTyping:true, skipEmptyLines:'greedy',
      delimiter, quoteChar,
      complete: r => resolve(r.data),
      error: reject
    });
  });
}
function normalizeRow(row){
  const out={};
  for(const [k,v] of Object.entries(row)){
    if(v===''||v===undefined) out[k]=null;
    else if(typeof v==='string'){ const t=v.trim(); out[k]=t===''? null : t; }
    else out[k]=v;
  }
  return out;
}
function missingPct(rows){
  if(!rows.length) return 100;
  const cols=Object.keys(rows[0]); let miss=0, tot=rows.length*cols.length;
  for(const r of rows){ for(const c of cols){ const v=r[c]; if(v==null||v==='') miss++; } }
  return +(100*miss/tot).toFixed(1);
}
function previewTable(rows, limit=8){
  if(!rows.length){ $('previewTable').innerHTML=''; return; }
  const cols=Object.keys(rows[0]);
  const head='<thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
  const body='<tbody>'+rows.slice(0,limit).map(r=>(
    '<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>'
  )).join('')+'</tbody>';
  $('previewTable').innerHTML = `<table>${head}${body}</table>`;
}

/* --------------- Column resolver --------------- */
const norm = s => (s||'').toLowerCase().replace(/[^a-z0-9]/g,'');
const ALIAS = new Map(Object.entries({
  gender:'gender', sex:'gender',
  customertype:'customertype','customer_type':'customertype',
  typeoftravel:'typeoftravel','type_of_travel':'typeoftravel',
  class:'class',
  satisfaction:'satisfaction','satisfied':'satisfaction',
  age:'age',
  flightdistance:'flightdistance','flight_distance':'flightdistance',
  inflightwifiservice:'inflightwifiservice','inflight_wifi_service':'inflightwifiservice','wifi':'inflightwifiservice',
  departurearrivaltimeconvenient:'timeconvenient','departuretimeconvenient':'timeconvenient','arrivaltimeconvenient':'timeconvenient',
  easeofonlinebooking:'easeonline','ease_online_booking':'easeonline',
  gatelocation:'gatelocation','gate_location':'gatelocation',
  foodanddrink:'fooddrink','food_drink':'fooddrink',
  onlineboarding:'onlineboarding','online_boarding':'onlineboarding',
  seatcomfort:'seatcomfort','seat_comfort':'seatcomfort',
  inflightentertainment:'inflightentertainment','inflight_entertainment':'inflightentertainment',
  onboardservice:'onboardservice','on_board_service':'onboardservice',
  legroomservice:'legroomservice','leg_room_service':'legroomservice',
  baggagehandling:'baggagehandling','baggage_handling':'baggagehandling',
  checkinservice:'checkinservice','check_in_service':'checkinservice',
  inflightservice:'inflightservice','inflight_service':'inflightservice',
  cleanliness:'cleanliness',
  departuredelayinminutes:'departuredelay','departuredelay':'departuredelay',
  arrivaldelayinminutes:'arrivaldelay','arrivaldelay':'arrivaldelay',
  id:'id', passengerid:'id'
}));
function unifyRow(row){
  const u={};
  for(const [k,v] of Object.entries(row)){
    const key = ALIAS.get(norm(k)) || null;
    if(key){ u[key]=v; }
  }
  if('satisfaction' in u){
    const val=u.satisfaction;
    if(typeof val==='string'){
      const t=val.toLowerCase();
      u.satisfaction = (t.includes('satisfied') && !t.includes('neutral')) ? 1 : 0;
    } else u.satisfaction = +val;
  }
  return u;
}

/* --------------- Stats --------------- */
const median = a => { const b=a.filter(x=>x!=null&&!Number.isNaN(+x)).map(Number).sort((x,y)=>x-y);
  if(!b.length) return null; const m=Math.floor(b.length/2); return b.length%2? b[m] : (b[m-1]+b[m])/2; };
const mode = a => { const m=new Map(); let best=null, c=0; for(const v of a){ if(v==null) continue; const k=String(v); const n=(m.get(k)||0)+1; m.set(k,n); if(n>c){c=n; best=k;} } return best; };
const mean = a => { const b=a.filter(Number.isFinite); return b.length? b.reduce((s,x)=>s+x,0)/b.length : 0; };
const sd   = a => { const b=a.filter(Number.isFinite); if(b.length<2) return 0; const mu=mean(b); return Math.sqrt(b.reduce((s,x)=>s+(x-mu)**2,0)/(b.length-1)); };
const finite = (x, d=0)=> Number.isFinite(+x) ? +x : d;
const pearson=(x,y)=>{ const P=[]; for(let i=0;i<Math.min(x.length,y.length);i++){const xi=x[i],yi=y[i]; if(Number.isFinite(+xi)&&Number.isFinite(+yi)) P.push([+xi,+yi]);}
  const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]); const mx=mean(xs), my=mean(ys); const sx=sd(xs)||0, sy=sd(ys)||0; if(!sx||!sy) return 0;
  const cov=xs.reduce((s,xi,j)=>s+(xi-mx)*(ys[j]-my),0)/(xs.length-1); return cov/(sx*sy); };

/* --------------- Preprocess mapping --------------- */
function buildMapping(trainRows){
  const useCase = $('useCase').value;
  const useCross = $('featCross').checked;
  const useTot   = $('featDelay').checked;

  const T = trainRows.map(unifyRow);

  const CAT_KEYS = ['gender','customertype','typeoftravel','class'];
  const NUM_KEYS_BASE = [
    'age','flightdistance','timeconvenient','easeonline','gatelocation','fooddrink','onlineboarding',
    'seatcomfort','inflightentertainment','onboardservice','legroomservice','baggagehandling',
    'checkinservice','inflightservice','cleanliness','departuredelay'
  ];
  const NUM_KEYS = [...NUM_KEYS_BASE, ...(useCase==='post'? ['arrivaldelay']:[])];

  const med = {}, mu={}, sig={};
  for(const k of NUM_KEYS){
    const arr = T.map(r => finite(r[k], null)).filter(v=>v!=null);
    const m = median(arr);
    med[k] = Number.isFinite(m) ? m : 0;
    const vals = T.map(r=> finite(r[k], med[k]));
    mu[k] = mean(vals); sig[k] = sd(vals) || 1;
  }
  const catVals = {};
  for(const k of CAT_KEYS){
    const m = mode(T.map(r=> r[k]==null? null : String(r[k])));
    const uniq = Array.from(new Set(T.map(r=> r[k]==null ? m : String(r[k]))))
                      .map(v=> v==null?'UNK':String(v));
    const set = new Set(uniq.concat(['UNK']));
    catVals[k] = Array.from(set);
  }

  let crossVals = [];
  const crossKey = 'classXtypeoftravel';
  if(useCross){
    crossVals = Array.from(new Set(T.map(r=>{
      const c = (r.class ?? 'UNK') + '|' + (r.typeoftravel ?? 'UNK');
      return String(c);
    })));
    if(!crossVals.includes('UNK|UNK')) crossVals.push('UNK|UNK');
  }

  const DEEP_NUM = [...NUM_KEYS];
  if(useTot){ DEEP_NUM.push('totaldelay'); }

  const WIDE_SPEC = { cats:[...CAT_KEYS], cross:{enabled:useCross, key:crossKey, values:crossVals} };
  // SPEED: do not include categorical one-hots in deep path
  const DEEP_SPEC = { nums:DEEP_NUM, includeCats:false, cats:[...CAT_KEYS] };
  const idKey = 'id';

  return { useCase, useCross, useTot, med, mu, sig, catVals, WIDE_SPEC, DEEP_SPEC, idKey };
}

function imputeAndVec(rows, M){
  const Xw=[], Xd=[], Y=[], IDS=[];
  for(let i=0;i<rows.length;i++){
    const r0 = unifyRow(rows[i]);
    const rc = {...r0};

    for(const k of Object.keys(M.catVals)){
      const vals=M.catVals[k];
      const v = rc[k];
      rc[k] = (v==null) ? vals[0] : String(v);
      if(!vals.includes(rc[k])) rc[k]='UNK';
    }
    for(const k of M.DEEP_SPEC.nums){
      if(k==='totaldelay') continue;
      const raw=rc[k];
      const val=finite(raw, null);
      rc[k] = (val==null) ? M.med[k] : val;
    }
    if(M.useTot){
      const dep = finite(rc['departuredelay'], M.med['departuredelay']);
      const arr = (M.useCase==='post') ? finite(rc['arrivaldelay'], M.med['arrivaldelay']) : 0;
      rc['totaldelay'] = dep + arr;
    }

    const w=[];
    for(const k of M.WIDE_SPEC.cats){
      const cats=M.catVals[k];
      w.push(...cats.map(cv => rc[k]===cv ? 1 : 0));
    }
    if(M.WIDE_SPEC.cross.enabled){
      const key = (rc['class']??'UNK') + '|' + (rc['typeoftravel']??'UNK');
      const cats = M.WIDE_SPEC.cross.values;
      w.push(...cats.map(cv => key===cv ? 1 : 0));
    }

    const d=[];
    for(const k of M.DEEP_SPEC.nums){
      const val = finite(rc[k], M.med[k] ?? 0);
      const z = (val - (M.mu[k]??0)) / (M.sig[k]||1);
      d.push(Number.isFinite(z)? z : 0);
    }

    Xw.push(w); Xd.push(d);
    if('satisfaction' in rc) Y.push(+rc.satisfaction);
    const id = ('id' in r0) ? r0['id'] : (i+1);
    IDS.push(id);
  }
  return { Xwide:Xw, Xdeep:Xd, y: Y.length? Y : null, ids: IDS };
}

/* --------------- Split --------------- */
function stratifiedSplitRows(rows, rate=0.2){
  const U = rows.map(unifyRow).map((r,idx)=>({...rows[idx], __y: +r.satisfaction}));
  const z = U.filter(r=>r.__y===0), o=U.filter(r=>r.__y===1);
  const split = arr => { const a=arr.slice(); tf.util.shuffle(a);
    const n=Math.max(1, Math.floor(a.length*rate)); return {val:a.slice(0,n), tr:a.slice(n)}; };
  const A=split(z), B=split(o);
  const train=A.tr.concat(B.tr), val=A.val.concat(B.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  train.forEach(r=>delete r.__y); val.forEach(r=>delete r.__y);
  return { train, val };
}

/* --------------- Model (smaller deep) --------------- */
function buildWideDeep(wideLen, deepLen){
  const wideIn = tf.input({shape:[wideLen], name:'wide'});
  const deepIn = tf.input({shape:[deepLen], name:'deep'});

  const wideLogit = tf.layers.dense({units:1, useBias:false, activation:'linear', name:'wide_logit'}).apply(wideIn);

  let x = tf.layers.dense({units:32, activation:'relu'}).apply(deepIn);
  x = tf.layers.dropout({rate:0.1}).apply(x);
  x = tf.layers.dense({units:8, activation:'relu'}).apply(x);
  const deepLogit = tf.layers.dense({units:1, activation:'linear', name:'deep_logit'}).apply(x);

  const sum = tf.layers.add().apply([wideLogit, deepLogit]);
  const out = tf.layers.activation({activation:'sigmoid', name:'out'}).apply(sum);

  const model = tf.model({inputs:[wideIn, deepIn], outputs:out, name:'wide_deep'});
  model.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  return model;
}
function modelSummaryText(m){ const lines=[]; m.summary(undefined, undefined, s=>lines.push(s)); return lines.join('\n'); }

/* --------------- Metrics --------------- */
function rocPoints(yTrue, yProb, steps=200){
  const T=[]; for(let i=0;i<=steps;i++) T.push(i/steps);
  const pts=T.map(th=>{
    let TP=0,FP=0,TN=0,FN=0;
    for(let i=0;i<yTrue.length;i++){
      const y=yTrue[i], p=yProb[i]>=th?1:0;
      if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++;
    }
    const TPR=TP/(TP+FN||1), FPR=FP/(FP+TN||1);
    return {x:FPR,y:TPR,th};
  });
  const s=pts.slice().sort((a,b)=>a.x-b.x);
  let auc=0; for(let i=1;i<s.length;i++){ const a=s[i-1], b=s[i]; auc += (b.x-a.x)*(a.y+b.y)/2; }
  return {points:s, auc};
}
function drawROC(canvas, pts){
  const ctx=canvas.getContext('2d'); const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#0f1628'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#233350'; ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const x=i/5; ctx.beginPath(); ctx.moveTo(40+x*(W-60), H-30); ctx.lineTo(40+x*(W-60), 20); ctx.stroke();}
  for(let i=0;i<=5;i++){const y=i/5; ctx.beginPath(); ctx.moveTo(40, 20+y*(H-50)); ctx.lineTo(W-20, 20+y*(H-50)); ctx.stroke();}
  ctx.strokeStyle='#8aa3ff'; ctx.lineWidth=2; ctx.beginPath();
  pts.forEach((p,i)=>{ const x=40+p.x*(W-60), y=H-30-p.y*(H-50); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
  ctx.stroke();
}
function confusionStats(yTrue, yProb, th){
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<yTrue.length;i++){
    const y=yTrue[i], p=yProb[i]>=th?1:0;
    if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++;
  }
  const prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=(2*prec*rec)/((prec+rec)||1);
  return {TP,FP,TN,FN,prec,rec,f1};
}

/* --------------- Early stop --------------- */
let stopFlag=false;
function earlyStopWithRestore(patience=3, monitor='val_loss'){
  let best=Infinity, wait=0, snap=null;
  return new tf.CustomCallback({
    onBatchEnd: async ()=>{ await new Promise(r=>setTimeout(r,0)); },
    onEpochEnd: async (_e, logs)=>{
      await tf.nextFrame();
      const cur=logs?.[monitor];
      if(cur!=null){
        if(cur<best-1e-12){
          best=cur; wait=0;
          if(snap) snap.forEach(t=>t.dispose());
          snap = S.model.getWeights().map(w=>w.clone());
        }else if(++wait>=patience){
          if(snap){ S.model.setWeights(snap); snap=null; }
          S.model.stopTraining=true;
        }
      }
      if(stopFlag) S.model.stopTraining=true;
    }
  });
}

/* --------------- Sanity checks --------------- */
function ensureReadyForTraining(){
  if(!(S.xsWTr && S.xsDTr && S.ysTr && S.xsWVa && S.xsDVa && S.ysVa)) {
    throw new Error('Tensors not built. Run “Run Preprocessing” again.');
  }
  const nTr = S.xsWTr.shape[0];
  if(nTr===0) throw new Error('No training rows after preprocessing.');
  if(S.xsWTr.shape[0]!==S.xsDTr.shape[0] || S.xsWTr.shape[0]!==S.ysTr.shape[0]) {
    throw new Error(`Train shape mismatch: wide=${S.xsWTr.shape} deep=${S.xsDTr.shape} y=${S.ysTr.shape}`);
  }
  if(S.xsWVa.shape[0]!==S.xsDVa.shape[0] || S.xsWVa.shape[0]!==S.ysVa.shape[0]) {
    throw new Error(`Val shape mismatch: wide=${S.xsWVa.shape} deep=${S.xsDVa.shape} y=${S.ysVa.shape}`);
  }
  if(S.xsWTr.shape[1]===0 || S.xsDTr.shape[1]===0){
    throw new Error('Feature vectors are empty (0 columns). Check categorical mapping.');
  }
}

/* --------------- Handlers: Load / Preprocess / Build / Train / Predict --------------- */
async function onLoad(){
  try{
    const fT = $('trainFile')?.files?.[0];
    const fX = $('testFile')?.files?.[0];
    if(!fT){ alert('Please choose train.csv'); return; }

    const rawTrain = (await parseWithPapa(fT, ',', '"')).map(normalizeRow);
    const rawTest  = fX ? (await parseWithPapa(fX, ',', '"')).map(normalizeRow) : [];

    S.rawTrain = rawTrain; S.rawTest = rawTest;

    $('kTrain').textContent = rawTrain.length;
    $('kTest').textContent  = rawTest.length || '—';
    $('kMiss').textContent  = missingPct(rawTrain) + '%';
    $('fixNote').textContent= '—';
    previewTable(rawTrain);
  }catch(e){ console.error(e); alert('Load failed: '+(e?.message||e)); }
}
function onPreprocess(){
  try{
    if(!S.rawTrain.length){ alert('Load train.csv first'); return; }
    S.map = buildMapping(S.rawTrain);

    const {train, val} = stratifiedSplitRows(S.rawTrain, 0.2);
    const TV = imputeAndVec(train, S.map);
    const VV = imputeAndVec(val,   S.map);

    S.xsWTr = tf.tensor2d(TV.Xwide, [TV.Xwide.length, TV.Xwide[0].length], 'float32');
    S.xsDTr = tf.tensor2d(TV.Xdeep, [TV.Xdeep.length, TV.Xdeep[0].length], 'float32');
    S.ysTr  = tf.tensor2d(TV.y, [TV.y.length, 1], 'float32');

    S.xsWVa = tf.tensor2d(VV.Xwide, [VV.Xwide.length, VV.Xwide[0].length], 'float32');
    S.xsDVa = tf.tensor2d(VV.Xdeep, [VV.Xdeep.length, VV.Xdeep[0].length], 'float32');
    S.ysVa  = tf.tensor2d(VV.y, [VV.y.length, 1], 'float32');

    $('preInfo').textContent = [
      `Use case: ${S.map.useCase==='pre'?'Pre-flight (ArrivalDelay excluded)':'Post-flight (ArrivalDelay included)'}`,
      `Wide len: ${S.xsWTr.shape[1]} | Deep len: ${S.xsDTr.shape[1]}`,
      `Train: wide ${S.xsWTr.shape} / deep ${S.xsDTr.shape} | Val: wide ${S.xsWVa.shape} / deep ${S.xsDVa.shape}`,
      `Impute: medians for numerics; modes for categoricals`,
      `Engineered: Cross=${S.map.useCross}, TotalDelay=${S.map.useTot}`
    ].join('\n');
  }catch(e){ console.error(e); alert('Preprocessing failed: '+(e?.message||e)); }
}
function onBuild(){
  try{
    if(!S.xsWTr){ alert('Run Preprocessing first'); return; }
    S.model = buildWideDeep(S.xsWTr.shape[1], S.xsDTr.shape[1]);
    $('modelSummary').textContent = 'Model built. Click “Show Summary”.';
  }catch(e){ console.error(e); alert('Build failed: '+(e?.message||e)); }
}
function onSummary(){
  try{
    if(!S.model){ alert('Build model first'); return; }
    $('modelSummary').textContent = modelSummaryText(S.model);
  }catch(e){ console.error(e); alert('Summary failed: '+(e?.message||e)); }
}

async function onTrain(){
  try{
    if(!S.model){ alert('Build the model first'); return; }
    if(S.isTraining){ alert('Training is already running. Please wait or click Early Stop.'); return; }

    ensureReadyForTraining();
    setBusy(true); stopFlag=false;
    $('trainLog').textContent = 'epoch 0: starting…\n';
    appendLog('trainLog', `Train shapes: wide ${S.xsWTr.shape}, deep ${S.xsDTr.shape}, y ${S.ysTr.shape}\n`);
    appendLog('trainLog', `Val shapes:   wide ${S.xsWVa.shape}, deep ${S.xsDVa.shape}, y ${S.ysVa.shape}\n`);

    const cb = earlyStopWithRestore(3,'val_loss');
    await S.model.fit(
      [S.xsWTr, S.xsDTr], S.ysTr,
      {
        epochs: 25,
        batchSize: 64,
        validationData: [[S.xsWVa, S.xsDVa], S.ysVa],
        callbacks: [{
          onEpochEnd: async (ep, logs)=>{
            appendLog('trainLog', `epoch ${ep+1}: loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} acc=${(logs.acc??logs.accuracy??0).toFixed(4)}\n`);
            await cb.onEpochEnd?.(ep, logs);
          },
          onBatchEnd: async (b, logs)=>{ await cb.onBatchEnd?.(b, logs); }
        }]
      }
    );

    const valProbs = tf.tidy(()=> S.model.predict([S.xsWVa, S.xsDVa]).dataSync());
    S.valProbs = Float32Array.from(valProbs);
    const yTrue = Array.from(S.ysVa.dataSync()).map(v=>+v);
    const {points, auc} = rocPoints(yTrue, S.valProbs, 200);
    drawROC($('rocCanvas'), points);
    $('aucText').textContent = auc.toFixed(4);
    updateThreshold(S.thresh);
  }catch(e){
    console.error(e);
    alert('Training failed: ' + (e?.message || e));
  } finally {
    setBusy(false);
  }
}
function onStop(){ if(S.isTraining){ stopFlag=true; appendLog('trainLog', 'Early stop requested…\n'); } }

/* --------------- Threshold --------------- */
function updateThreshold(th){
  $('thVal').textContent=(+th).toFixed(2);
  if(S.valProbs==null) return;
  const yTrue=Array.from(S.ysVa.dataSync()).map(v=>+v);
  const st=confusionStats(yTrue,S.valProbs,+th);
  $('cmTP').textContent=st.TP; $('cmFN').textContent=st.FN;
  $('cmFP').textContent=st.FP; $('cmTN').textContent=st.TN;
  $('prf').textContent=`Precision: ${(st.prec*100).toFixed(2)}%\nRecall: ${(st.rec*100).toFixed(2)}%\nF1: ${st.f1.toFixed(4)}`;
  S.thresh=+th;
}

/* --------------- Predict & Export --------------- */
function onPredict(){
  try{
    if(!S.model){ alert('Train the model first'); return; }
    if(!S.rawTest.length){ alert('Load test.csv'); return; }

    const V = imputeAndVec(S.rawTest, S.map);
    const xsW = tf.tensor2d(V.Xwide, [V.Xwide.length, V.Xwide[0].length], 'float32');
    const xsD = tf.tensor2d(V.Xdeep, [V.Xdeep.length, V.Xdeep[0].length], 'float32');
    const probs = tf.tidy(()=> S.model.predict([xsW, xsD]).dataSync());
    xsW.dispose(); xsD.dispose();

    S.testProbs = Float32Array.from(probs);
    S.testIDs   = V.ids;
    $('predInfo').textContent = `Predicted ${S.testProbs.length} rows. Ready to download.`;
  }catch(e){ console.error(e); alert('Prediction failed: '+(e?.message||e)); }
}
function downloadCSV(name, rows){
  if(!rows.length) return;
  const cols=Object.keys(rows[0]);
  const esc=v=>{ if(v==null) return ''; const s=String(v); return /[",\n]/.test(s)? '"'+s.replace(/"/g,'""')+'"' : s; };
  const csv=[cols.join(',')].concat(rows.map(r=> cols.map(c=>esc(r[c])).join(','))).join('\n');
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'}); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}
function onDownloadSubmission(){
  try{
    if(S.testProbs==null){ alert('Run Predict first'); return; }
    const out = S.testIDs.map((id,i)=>({ id, satisfaction: (S.testProbs[i]>=S.thresh? 1 : 0) }));
    downloadCSV('submission.csv', out);
  }catch(e){ console.error(e); alert('Download failed: '+(e?.message||e)); }
}
function onDownloadProbs(){
  try{
    if(S.testProbs==null){ alert('Run Predict first'); return; }
    const out = S.testIDs.map((id,i)=>({ id, prob_satisfied: S.testProbs[i] }));
    downloadCSV('probabilities.csv', out);
  }catch(e){ console.error(e); alert('Download failed: '+(e?.message||e)); }
}
async function onSaveModel(){
  try{
    if(!S.model){ alert('Train the model first'); return; }
    await S.model.save('downloads://airline_wide_deep_tfjs');
  }catch(e){ console.error(e); alert('Save failed: '+(e?.message||e)); }
}

/* ===================== EDA (ECharts) ===================== */
function eda_rows(){
  if(!S.rawTrain.length) throw new Error('Load train.csv first');
  return S.rawTrain.map(unifyRow);
}
function chart(id){ // get or create echarts instance
  if(S.charts[id]) return S.charts[id];
  const el=$(id);
  if(!el) return null;
  S.charts[id] = echarts.init(el);
  return S.charts[id];
}
function destroyCharts(){
  for(const k in S.charts){ try{ S.charts[k].dispose(); }catch{} }
  S.charts = {};
}
function drawBar(id, labels, values, title, percent=false){
  const c = chart(id); if(!c) return;
  c.setOption({
    title:{text:title, left:'center', textStyle:{fontSize:12,color:'#aab3c6'}},
    tooltip:{trigger:'axis', axisPointer:{type:'shadow'}, valueFormatter:v=> percent? v+'%' : v},
    grid:{left:60,right:20,top:30,bottom:40},
    xAxis:{type:'category', data:labels},
    yAxis:{type:'value', max: percent? 100 : null, axisLabel:{formatter:(v)=> percent? v+'%': v}},
    series:[{type:'bar', data:values, stack:null, label:{show:true, position:'top', formatter:(d)=> percent? d.value+'%': d.value }}],
  });
}
function drawHist(id, values, bins, title){
  const v = values.filter(Number.isFinite);
  if(!v.length){ drawBar(id, ['No data'], [0], title); return; }
  const min = Math.min(...v), max=Math.max(...v);
  const nb = Math.max(5, Math.min(40, bins||20));
  const step = (max-min)/nb || 1;
  const edges = Array.from({length:nb}, (_,i)=> min + i*step);
  const counts = new Array(nb).fill(0);
  for(const x of v){
    let idx = Math.floor((x-min)/step);
    if(idx>=nb) idx=nb-1; if(idx<0) idx=0;
    counts[idx]++;
  }
  const labels = edges.map((e,i)=> i===nb-1 ? `${e.toFixed(0)}+` : `${e.toFixed(0)}–${(e+step).toFixed(0)}`);
  drawBar(id, labels, counts, title, false);
}
function drawMissing(id, rows){
  const cols = new Set();
  rows.forEach(r=> Object.keys(r).forEach(k=> cols.add(k)));
  const labels=[], vals=[];
  for(const c of cols){
    let miss=0, tot=rows.length;
    for(const r of rows){ const v=r[c]; if(v==null||v==='') miss++; }
    labels.push(c); vals.push(Math.round(100*miss/tot));
  }
  // show top 12 by missing%
  const order = vals.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,12);
  const L=order.map(x=>labels[x[1]]), V=order.map(x=>x[0]);
  drawBar(id, L, V, 'Missing values (%)', true);
}
function groupRate(rows, key){
  const map = new Map();
  rows.forEach(r=>{
    const k = r[key]==null ? 'UNK' : String(r[key]);
    const y = +r.satisfaction===1 ? 1 : 0;
    const rec = map.get(k) || {tot:0, pos:0};
    rec.tot++; rec.pos+=y; map.set(k,rec);
  });
  const labels = Array.from(map.keys());
  const rates  = labels.map(k=> +(100*map.get(k).pos/Math.max(1,map.get(k).tot)).toFixed(1));
  return {labels, rates};
}
function drawCorr(id, rows){
  const NUM = ['age','flightdistance','timeconvenient','easeonline','gatelocation','fooddrink','onlineboarding',
    'seatcomfort','inflightentertainment','onboardservice','legroomservice','baggagehandling',
    'checkinservice','inflightservice','cleanliness','departuredelay','arrivaldelay'].filter(k=> rows.some(r=> r[k]!=null));
  if(!NUM.length){ drawBar(id, ['No numeric cols'], [0], 'Correlation'); return; }
  const cols=NUM;
  const mat = cols.map(c=> rows.map(r=> r[c]!=null? +r[c] : NaN));
  const Z = cols.map((_,i)=> cols.map((_,j)=> +pearson(mat[i],mat[j]).toFixed(2)));
  const data=[]; for(let i=0;i<cols.length;i++){ for(let j=0;j<cols.length;j++){ data.push([i,j,Z[i][j]]); } }
  const c = chart(id); if(!c) return;
  c.setOption({
    title:{text:'Correlation (Pearson)', left:'center', textStyle:{fontSize:12,color:'#aab3c6'}},
    tooltip:{},
    grid:{left:80,right:20,top:30,bottom:40},
    xAxis:{type:'category', data:cols},
    yAxis:{type:'category', data:cols},
    visualMap:{min:-1,max:1, calculable:true, orient:'horizontal', left:'center', bottom:0},
    series:[{type:'heatmap', data, label:{show:true, formatter:(p)=> (p.data[2]).toFixed(2)}}]
  });
}
function onEDA(){
  try{
    const rows = eda_rows(); // unified keys + normalized
    // Charts
    drawMissing('edMissing', rows);

    const gGender = groupRate(rows,'gender');
    drawBar('edGender', gGender.labels, gGender.rates, 'Satisfaction by Gender (%)', true);

    const gTravel = groupRate(rows,'typeoftravel');
    drawBar('edTravel', gTravel.labels, gTravel.rates, 'Satisfaction by Type of Travel (%)', true);

    const gClass = groupRate(rows,'class');
    drawBar('edClass', gClass.labels, gClass.rates, 'Satisfaction by Class (%)', true);

    // Histograms (cap dep delay to 300 for readability)
    const ages = rows.map(r=> Number.isFinite(+r.age)? +r.age : NaN).filter(Number.isFinite);
    drawHist('edAge', ages, 24, 'Age — histogram');

    const dists = rows.map(r=> Number.isFinite(+r.flightdistance)? +r.flightdistance : NaN).filter(Number.isFinite);
    drawHist('edDist', dists, 24, 'Flight Distance — histogram');

    const deps = rows.map(r=> Number.isFinite(+r.departuredelay)? Math.min(+r.departuredelay, 300) : NaN).filter(Number.isFinite);
    drawHist('edDep', deps, 24, 'Departure Delay (capped at 300) — histogram');

    drawCorr('edCorr', rows);

    // Resize on window change
    const ids=['edMissing','edGender','edTravel','edClass','edAge','edDist','edDep','edCorr'];
    window.addEventListener('resize', ()=> ids.forEach(id=> S.charts[id]?.resize()));
  }catch(e){ console.error(e); alert('EDA failed: '+(e?.message||e)); }
}

/* --------------- Wire-up --------------- */
window.addEventListener('DOMContentLoaded', ()=>{
  $('btnLoad').addEventListener('click', onLoad);
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
  $('btnEDA').addEventListener('click', ()=>{ destroyCharts(); onEDA(); });
});
