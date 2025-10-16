// app.js
// Data loading, model training with manual early stopping (fixed callbacks), per-output threshold tuning on validation,
// and test evaluation/visualization. Everything runs fully client-side.

import { prepareDatasetFromFile, DEFAULT_SEQ_LEN, DEFAULT_HORIZONS } from './data-loader.js';
import { GRUClassifier } from './gru.js';

const ui = {
  file: document.getElementById('csvFile'),
  fileName: document.getElementById('fileName'),
  seqLen: document.getElementById('seqLen'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  load: document.getElementById('btnLoad'),
  train: document.getElementById('btnTrain'),
  predict: document.getElementById('btnPredict'),
  save: document.getElementById('btnSave'),
  shapes: document.getElementById('shapes'),
  tunedThreshold: document.getElementById('tunedThreshold'),
  log: document.getElementById('log'),
  progressBar: document.getElementById('progressBar'),
  progressText: document.getElementById('progressText'),
  accCanvas: document.getElementById('accChart'),
  timelineContainer: document.getElementById('timelineContainer'),
  confusionContainer: document.getElementById('confusionContainer'),
};

let ds = null;
let model = null;
let accChart = null;
let tunedThresholds = null; // Array length = outDim

function setDisabled(el, v) { el.disabled = !!v; }
function setState(state) {
  if (state === 'init') { setDisabled(ui.train, true); setDisabled(ui.predict, true); setDisabled(ui.save, true); }
  if (state === 'loaded') { setDisabled(ui.train, false); setDisabled(ui.predict, true); setDisabled(ui.save, true); }
  if (state === 'trained') { setDisabled(ui.train, false); setDisabled(ui.predict, false); setDisabled(ui.save, false); }
}
function log(msg){ const t=new Date().toLocaleTimeString(); ui.log.textContent += `[${t}] ${msg}\n`; ui.log.scrollTop = ui.log.scrollHeight; }
function progress(p, txt){ const v=Math.max(0,Math.min(100,Math.round(p))); ui.progressBar.style.width=`${v}%`; ui.progressText.textContent = txt || `${v}%`; }
function showShapes(meta, symbols){
  ui.shapes.textContent = `SeqLen=${meta.seqLen}, FeatureDim=${meta.featureDim} (${meta.featuresPerStock}/stock), Samples=${meta.totalSamples} (train ${meta.numTrain} | val ${meta.numVal} | test ${meta.numTest}), Stocks=${symbols.length}`;
}

function computePerStockMetrics(pred, truth, symbols, horizons, thresholds) {
  const S=symbols.length, H=horizons.length, N=truth.length;
  const per=symbols.map(()=>({correct:0,total:0,timeline:Array.from({length:H},()=>[]),confusion:[0,0,0,0]}));
  for(let i=0;i<N;i++){
    const gt=truth[i], pp=pred[i];
    for(let s=0;s<S;s++){
      for(let h=0;h<H;h++){
        const idx=s*H+h;
        const thr = Array.isArray(thresholds) ? thresholds[idx] : thresholds;
        const y = gt[idx] >= 0.5 ? 1 : 0;
        const p = pp[idx] >= thr ? 1 : 0;
        const ok = (y===p);
        per[s].correct += ok?1:0; per[s].total += 1; per[s].timeline[h].push(ok);
        if (y===0&&p===0) per[s].confusion[0]++; else if (y===0&&p===1) per[s].confusion[1]++;
        else if (y===1&&p===0) per[s].confusion[2]++; else per[s].confusion[3]++;
      }
    }
  }
  return per.map((st,i)=>({ symbol: ds.symbols[i], accuracy: st.total?st.correct/st.total:0, timeline: st.timeline, confusion: st.confusion }));
}
function overallAccuracy(pred, truth, thresholds){
  let c=0,n=0; const outDim=truth[0].length;
  for(let i=0;i<truth.length;i++){
    const t=truth[i], p=pred[i];
    for(let j=0;j<outDim;j++){
      const thr = Array.isArray(thresholds) ? thresholds[j] : thresholds;
      c += ((t[j]>=0.5) === (p[j]>=thr)) ? 1:0; n++;
    }
  }
  return c/Math.max(1,n);
}
function tuneThresholds(valPred, yVal, lo=0.30, hi=0.70, step=0.01){
  const outDim = yVal[0].length;
  const N = yVal.length;
  const th = new Array(outDim).fill(0.5);

  for (let j=0;j<outDim;j++){
    let bestAcc=-1, bestThr=0.5;
    for (let thr=lo; thr<=hi+1e-9; thr+=step){
      let c=0;
      for (let i=0;i<N;i++){ c += ((yVal[i][j]>=0.5) === (valPred[i][j]>=thr)) ? 1 : 0; }
      const acc = c/N;
      if (acc > bestAcc) { bestAcc=acc; bestThr=+thr.toFixed(3); }
    }
    th[j]=bestThr;
  }
  return th;
}

function drawAccuracyChart(results){
  const sorted=[...results].sort((a,b)=>b.accuracy-a.accuracy);
  const labels=sorted.map(r=>r.symbol);
  const data=sorted.map(r=>(r.accuracy*100).toFixed(2));
  if (accChart){ accChart.destroy(); accChart=null; }
  const ctx=ui.accCanvas.getContext('2d');
  accChart = new Chart(ctx, {
    type:'bar',
    data:{ labels, datasets:[{ label:'Accuracy (%)', data }] },
    options:{ indexAxis:'y', scales:{ x:{ min:0,max:100, ticks:{ callback:v=>`${v}%` } } }, plugins:{ legend:{ display:false } }, responsive:true }
  });
}
function drawTimelines(results, horizons, baseDates){
  ui.timelineContainer.innerHTML='';
  const H=horizons.length;
  results.forEach(r=>{
    const wrap=document.createElement('div'); wrap.className='timeline';
    const title=document.createElement('div'); title.className='timeline-title';
    title.textContent=`${r.symbol} — prediction correctness (rows: +1d, +2d, +3d)`; wrap.appendChild(title);
    const canvas=document.createElement('canvas'); canvas.width=Math.max(640,r.timeline[0].length*6); canvas.height=H*18+26; wrap.appendChild(canvas);
    const legend=document.createElement('div'); legend.className='legend'; legend.innerHTML='<span class="ok" style="background:#27ae60;display:inline-block;width:12px;height:12px;margin-right:4px;"></span>Correct &nbsp; <span class="bad" style="background:#e74c3c;display:inline-block;width:12px;height:12px;margin-right:4px;"></span>Wrong'; wrap.appendChild(legend);
    const ctx=canvas.getContext('2d'); const cellW=6,cellH=12,padL=58,padT=8; ctx.font='12px system-ui, sans-serif';
    for(let h=0;h<H;h++){
      const seq=r.timeline[h]; ctx.fillStyle='#444'; ctx.fillText(`+${h+1}d`,6,padT+h*(cellH+6)+cellH-2);
      for(let i=0;i<seq.length;i++){ ctx.fillStyle=seq[i]?'#27ae60':'#e74c3c'; ctx.fillRect(padL+i*cellW, padT+h*(cellH+6), cellW-1, cellH); }
    }
    const N=r.timeline[0].length, step=Math.max(1,Math.floor(N/12)); ctx.fillStyle='#222';
    for(let i=0;i<N;i+=step){ const x=padL+i*cellW; ctx.fillRect(x,canvas.height-10,1,8); ctx.save(); ctx.translate(x+2,canvas.height-2); ctx.rotate(-Math.PI/4); ctx.fillText(baseDates[i]||'',0,0); ctx.restore(); }
    ui.timelineContainer.appendChild(wrap);
  });
}
function drawConfusions(results){
  ui.confusionContainer.innerHTML='';
  const grid=document.createElement('div'); grid.className='cm-grid';
  results.forEach(r=>{
    const [tn,fp,fn,tp]=r.confusion;
    const card=document.createElement('div'); card.className='cm-card';
    card.innerHTML=`<div class="cm-title">${r.symbol}</div>
      <table class="cm-table">
        <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
        <tbody><tr><td>True 0</td><td>${tn}</td><td>${fp}</td></tr>
               <tr><td>True 1</td><td>${fn}</td><td>${tp}</td></tr></tbody>
      </table>`;
    grid.appendChild(card);
  });
  ui.confusionContainer.appendChild(grid);
}

// ---------- Event handlers ----------
async function onLoad(){
  try {
    const f = ui.file.files?.[0]; if (!f) { alert('Choose a CSV file first.'); return; }
    ui.fileName.textContent = f.name;
    progress(5,'Preparing dataset...');
    ds?.X_train?.dispose(); ds?.y_train?.dispose(); ds?.X_val?.dispose(); ds?.y_val?.dispose(); ds?.X_test?.dispose(); ds?.y_test?.dispose();

    const seqLen = Math.max(16, parseInt(ui.seqLen.value || DEFAULT_SEQ_LEN, 10));

    ds = await prepareDatasetFromFile(f, {
      seqLen,
      horizons: DEFAULT_HORIZONS,
      testSplit: 0.2,
      valSplitWithinTrain: 0.12
    });
    showShapes(ds.meta, ds.symbols);
    log('Dataset prepared.');
    progress(25,'Dataset ready');
    setState('loaded');
  } catch (e) {
    console.error(e);
    alert(`Data load error: ${e.message}`);
    log(`Error: ${e.message}`);
    progress(0,'');
    setState('init');
  }
}

async function onTrain(){
  if (!ds) { alert('Load data first.'); return; }
  try {
    const epochs = Math.max(1, parseInt(ui.epochs.value || '40', 10));
    const batch  = Math.max(1, parseInt(ui.batch.value  || '32', 10));

    model?.dispose();
    model = new GRUClassifier({
      seqLen: ds.meta.seqLen,
      featureDim: ds.meta.featureDim,
      numStocks: ds.symbols.length,
      horizons: ds.horizons,
      convFilters: 64,
      gruUnits: 96,
      denseUnits: 192,
      learningRate: 5e-4,
      dropout: 0.25
    });

    log('Training...');
    progress(28,'Training...');
    await model.fit(
      ds.X_train, ds.y_train, ds.X_val, ds.y_val,
      { epochs, batchSize: batch, shuffle: false, patience: 6, minDelta: 1e-4 },
      (epoch, logs) => {
        const pct = 28 + Math.round(((epoch + 1) / epochs) * 56); // -> 84%
        progress(pct, `Epoch ${epoch+1}/${epochs} • loss ${logs.loss?.toFixed(4)} • acc ${(logs.binaryAccuracy*100).toFixed(1)}%`);
        log(`Epoch ${epoch+1}/${epochs} — loss=${logs.loss?.toFixed(4)} valLoss=${logs.val_loss?.toFixed(4)} acc=${(logs.binaryAccuracy*100).toFixed(2)}%`);
      }
    );
    progress(86,'Training complete');

    // Threshold tuning on validation per-output
    log('Tuning thresholds on validation...');
    const valPredT = model.predict(ds.X_val);
    const valPred = await valPredT.array(); valPredT.dispose();
    const yValArr = await ds.y_val.array();
    tunedThresholds = tuneThresholds(valPred, yValArr, 0.30, 0.70, 0.01);
    const avgThr = tunedThresholds.reduce((a,b)=>a+b,0)/tunedThresholds.length;
    const valAcc = overallAccuracy(valPred, yValArr, tunedThresholds);
    ui.tunedThreshold.textContent = `avg threshold ${avgThr.toFixed(2)} (val acc ${(valAcc*100).toFixed(2)}%)`;
    log(`Validation accuracy ${(valAcc*100).toFixed(2)}% with tuned thresholds (avg ${avgThr.toFixed(2)}).`);

    setState('trained');
  } catch (e) {
    console.error(e);
    alert(`Training error: ${e.message}`);
    log(`Error: ${e.message}`);
    progress(0,'');
  }
}

async function onPredict(){
  if (!ds || !model) { alert('Train model first.'); return; }
  try {
    log('Predicting on test set...');
    progress(90,'Predicting...');
    const predT = model.predict(ds.X_test);
    const predArr = await predT.array(); predT.dispose();
    const yTrueArr = await ds.y_test.array();

    const results = computePerStockMetrics(predArr, yTrueArr, ds.symbols, ds.horizons, tunedThresholds || 0.5);
    drawAccuracyChart(results);
    drawTimelines(results, ds.horizons, ds.baseDatesTest);
    drawConfusions(results);

    const overall = overallAccuracy(predArr, yTrueArr, tunedThresholds || 0.5);
    log(`Test accuracy (overall) ${(overall*100).toFixed(2)}% using tuned thresholds.`);
    progress(100,'Done');
  } catch (e) {
    console.error(e);
    alert(`Prediction error: ${e.message}`);
    log(`Error: ${e.message}`);
    progress(0,'');
  }
}

async function onSave(){ if (!model){ alert('No trained model to save.'); return; } await model.save('tfjs_gru_stock_demo'); log('Model weights downloaded.'); }

function init(){
  setState('init');
  ui.load.addEventListener('click', onLoad);
  ui.train.addEventListener('click', onTrain);
  ui.predict.addEventListener('click', onPredict);
  ui.save.addEventListener('click', onSave);
}
init();
