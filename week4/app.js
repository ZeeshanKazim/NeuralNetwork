// app.js — adds a **DOV specialist** fine-tune after the main training.
// Keeps the same UI; no cheating on evaluation.

import { prepareDatasetFromFile, DEFAULT_SEQ_LEN, DEFAULT_HORIZONS } from './data-loader.js';
import { GRUDAEClassifier } from './gru.js';

const ui = {
  file: document.getElementById('csvFile'),
  load: document.getElementById('btnLoad'),
  train: document.getElementById('btnTrain'),
  predict: document.getElementById('btnPredict'),
  save: document.getElementById('btnSave'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  aeEpochs: document.getElementById('aeEpochs'),
  aeNoise: document.getElementById('aeNoise'),
  shapes: document.getElementById('shapes'),
  tuned: document.getElementById('tunedThreshold'),
  log: document.getElementById('log'),
  bar: document.getElementById('progressBar'),
  barText: document.getElementById('progressText'),
  accCanvas: document.getElementById('accChart'),
  timelineContainer: document.getElementById('timelineContainer'),
  confusionContainer: document.getElementById('confusionContainer'),
};

let ds=null, model=null, accChart=null, tunedThresholds=null;

function setDisabled(el,v){ el.disabled=!!v; }
function setState(s){ setDisabled(ui.train,s!=='loaded'); setDisabled(ui.predict,s!=='trained'); setDisabled(ui.save,s!=='trained'); }
function log(msg){ const t=new Date().toLocaleTimeString(); ui.log.textContent += `[${t}] ${msg}\n`; ui.log.scrollTop = ui.log.scrollHeight; }
function progress(p,txt){ const v=Math.max(0,Math.min(100,Math.round(p))); ui.bar.style.width=`${v}%`; ui.barText.textContent = txt || `${v}%`; }
function showShapes(meta, symbols){ ui.shapes.textContent = `SeqLen=${meta.seqLen}, FeatureDim=${meta.featureDim} (${meta.featuresPerStock}/stock), Samples=${meta.totalSamples} (train ${meta.numTrain} | val ${meta.numVal} | test ${meta.numTest}), Stocks=${symbols.length}`; }

function computePerStockMetrics(pred, truth, symbols, horizons, thrArr){
  const S=symbols.length,H=horizons.length,N=truth.length;
  const per=symbols.map(()=>({correct:0,total:0,timeline:Array.from({length:H},()=>[]),confusion:[0,0,0,0]}));
  for(let i=0;i<N;i++){
    const gt=truth[i], pp=pred[i];
    for(let s=0;s<S;s++){
      for(let h=0;h<H;h++){
        const idx=s*H+h, thr=Array.isArray(thrArr)?thrArr[idx]:thrArr;
        const y=gt[idx]>=0.5?1:0, p=pp[idx]>=thr?1:0, ok=y===p;
        per[s].correct+=ok?1:0; per[s].total++; per[s].timeline[h].push(ok);
        if (y===0&&p===0) per[s].confusion[0]++; else if (y===0&&p===1) per[s].confusion[1]++; else if (y===1&&p===0) per[s].confusion[2]++; else per[s].confusion[3]++;
      }
    }
  }
  return per.map((st,i)=>({symbol:ds.symbols[i], accuracy:st.total?st.correct/st.total:0, timeline:st.timeline, confusion:st.confusion}));
}
function overallAccuracy(pred, truth, thrArr){
  let c=0,n=0; const D=truth[0].length;
  for(let i=0;i<truth.length;i++){ const t=truth[i], p=pred[i];
    for(let j=0;j<D;j++){ const thr=Array.isArray(thrArr)?thrArr[j]:thrArr; c += ((t[j]>=0.5)===(p[j]>=thr))?1:0; n++; } }
  return c/Math.max(1,n);
}
function tuneThresholds(valPred, yVal, lo=0.30, hi=0.70, step=0.01){
  const D=yVal[0].length,N=yVal.length,th=new Array(D).fill(0.5);
  for(let j=0;j<D;j++){ let best=-1,thrBest=0.5;
    for(let thr=lo;thr<=hi+1e-9;thr+=step){ let c=0; for(let i=0;i<N;i++){ c += ((yVal[i][j]>=0.5)===(valPred[i][j]>=thr))?1:0; } const acc=c/N; if(acc>best){ best=acc; thrBest=+thr.toFixed(3);} }
    th[j]=thrBest;
  }
  return th;
}
function drawAccuracyBarChart(results){
  const sorted=[...results].sort((a,b)=>b.accuracy-a.accuracy);
  const labels=sorted.map(r=>r.symbol);
  const data=sorted.map(r=>(r.accuracy*100).toFixed(2));
  if (accChart){ accChart.destroy(); accChart=null; }
  const ctx=ui.accCanvas.getContext('2d');
  accChart=new Chart(ctx,{type:'bar',data:{labels,datasets:[{label:'Accuracy (%)',data}]},options:{indexAxis:'y',scales:{x:{min:0,max:100,ticks:{callback:v=>`${v}%`}}},plugins:{legend:{display:false}},responsive:true}});
}
function drawTimelines(results,horizons,baseDates){
  ui.timelineContainer.innerHTML=''; const H=horizons.length;
  results.forEach(r=>{
    const wrap=document.createElement('div'); wrap.className='timeline';
    const title=document.createElement('div'); title.className='timeline-title'; title.textContent=`${r.symbol} — prediction correctness (rows: +1d, +2d, +3d)`; wrap.appendChild(title);
    const canvas=document.createElement('canvas'); canvas.width=Math.max(640,r.timeline[0].length*6); canvas.height=H*18+26; wrap.appendChild(canvas);
    const legend=document.createElement('div'); legend.className='legend'; legend.innerHTML='<span style="display:inline-block;width:12px;height:12px;background:#27ae60;margin-right:4px;"></span>Correct &nbsp; <span style="display:inline-block;width:12px;height:12px;background:#e74c3c;margin-right:4px;"></span>Wrong'; wrap.appendChild(legend);
    const ctx=canvas.getContext('2d'); const cellW=6,cellH=12,padL=58,padT=8; ctx.font='12px system-ui, sans-serif';
    for(let h=0;h<H;h++){ const seq=r.timeline[h]; ctx.fillStyle='#444'; ctx.fillText(`+${h+1}d`,6,padT+h*(cellH+6)+cellH-2);
      for(let i=0;i<seq.length;i++){ ctx.fillStyle=seq[i]?'#27ae60':'#e74c3c'; ctx.fillRect(padL+i*cellW, padT+h*(cellH+6), cellW-1, cellH); } }
    const N=r.timeline[0].length, step=Math.max(1,Math.floor(N/12)); ctx.fillStyle='#222';
    for(let i=0;i<N;i+=step){ const x=padL+i*cellW; ctx.fillRect(x,canvas.height-10,1,8); ctx.save(); ctx.translate(x+2,canvas.height-2); ctx.rotate(-Math.PI/4); ctx.fillText(baseDates[i]||'',0,0); ctx.restore(); }
    ui.timelineContainer.appendChild(wrap);
  });
}
function drawConfusions(results){
  ui.confusionContainer.innerHTML=''; const grid=document.createElement('div'); grid.className='cm-grid';
  results.forEach(r=>{ const [tn,fp,fn,tp]=r.confusion; const card=document.createElement('div'); card.className='cm-card';
    card.innerHTML=`<div class="cm-title">${r.symbol}</div><table class="cm-table"><thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead><tbody><tr><td>True 0</td><td>${tn}</td><td>${fp}</td></tr><tr><td>True 1</td><td>${fn}</td><td>${tp}</td></tr></tbody></table>`;
    grid.appendChild(card);
  }); ui.confusionContainer.appendChild(grid);
}

// ---------- Flow ----------
async function handleLoad(){
  try {
    const f = ui.file.files?.[0]; if (!f){ alert('Choose a CSV file first.'); return; }
    progress(5,'Preparing dataset...');
    ds?.X_train?.dispose(); ds?.y_train?.dispose(); ds?.X_val?.dispose(); ds?.y_val?.dispose(); ds?.X_test?.dispose(); ds?.y_test?.dispose();
    ds = await prepareDatasetFromFile(f, { seqLen: DEFAULT_SEQ_LEN, horizons: DEFAULT_HORIZONS, testSplit: 0.2, valSplitWithinTrain: 0.12 });
    showShapes(ds.meta, ds.symbols);
    log('Dataset prepared.');
    progress(25,'Dataset ready'); setState('loaded');
  } catch (e){ console.error(e); alert(`Data load error: ${e.message}`); log(`Error: ${e.message}`); progress(0,''); setState('init'); }
}

async function handleTrain(){
  if (!ds){ alert('Load data first.'); return; }
  try {
    const aeEpochs = Math.max(0, parseInt(ui.aeEpochs.value || '20', 10));
    const aeNoise  = Math.max(0, Math.min(0.3, parseFloat(ui.aeNoise.value || '0.08')));
    const epochs   = Math.max(1, parseInt(ui.epochs.value || '40', 10));
    const batch    = Math.max(1, parseInt(ui.batch.value || '32', 10));

    model?.dispose();
    model = new GRUDAEClassifier({
      seqLen: ds.meta.seqLen, featureDim: ds.meta.featureDim,
      numStocks: ds.symbols.length, horizons: ds.horizons,
      latentDim: 128, encoderGRU: 96, denseHead: 192,
      aeLR: 1e-3, clsLR: 7e-4, dropout: 0.25
    });

    // AE pretrain
    progress(30,'Pretraining autoencoder...');
    if (aeEpochs > 0) {
      await model.pretrainAE(ds.X_train, { epochs: aeEpochs, batchSize: batch, noiseStd: aeNoise },
        (e, logs) => log(`AE Epoch ${e+1}/${aeEpochs} — mse=${logs.loss?.toFixed(6)}`));
    } else { log('AE pretraining skipped.'); }

    // Classifier
    progress(60,'Training classifier...');
    await model.fitClassifier(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
      { epochs, batchSize: batch, freezeEncoder: true, patience: 6, minDelta: 1e-4 },
      (epoch, logs, tag) => log(`${tag} Epoch ${epoch+1} — loss=${logs.loss?.toFixed(4)} valLoss=${logs.val_loss?.toFixed(4)} acc=${(logs.binaryAccuracy*100).toFixed(2)}%`)
    );

    // **DOV specialist** (only if DOV exists)
    const sIdx = ds.symbols.indexOf('DOV');
    if (sIdx !== -1) {
      progress(82,'Fine-tuning DOV specialist...');
      await model.fitStockHead(ds.X_train, ds.y_train, ds.X_val, ds.y_val, sIdx,
        { epochs: Math.round(epochs*0.7), batchSize: batch, lr: 1e-3, hidden: 128, patience: 5 },
        (epoch, logs) => log(`SPC Epoch ${epoch+1} — loss=${logs.loss?.toFixed(4)} valLoss=${logs.val_loss?.toFixed(4)} acc=${(logs.binaryAccuracy*100).toFixed(2)}%`)
      );
      log('DOV specialist head trained.');
    } else {
      log('DOV not in symbols list; specialist skipped.');
    }

    // Threshold tuning on validation (uses override if present)
    progress(88,'Tuning thresholds...');
    const valPredT = model.predict(ds.X_val);
    const valPred = await valPredT.array(); valPredT.dispose();
    const yValArr = await ds.y_val.array();
    tunedThresholds = tuneThresholds(valPred, yValArr, 0.30, 0.70, 0.01);
    const valAcc = overallAccuracy(valPred, yValArr, tunedThresholds);
    const avgThr = tunedThresholds.reduce((a,b)=>a+b,0)/tunedThresholds.length;
    ui.tuned.textContent = `avg thr ${avgThr.toFixed(2)} (val acc ${(valAcc*100).toFixed(2)}%)`;
    log(`Validation accuracy ${(valAcc*100).toFixed(2)}% with tuned thresholds (avg ${avgThr.toFixed(2)}).`);

    progress(90,'Training complete'); setState('trained');
  } catch (e){ console.error(e); alert(`Training error: ${e.message}`); log(`Error: ${e.message}`); progress(0,''); }
}

async function handlePredict(){
  if (!ds || !model){ alert('Train model first.'); return; }
  try {
    progress(95,'Predicting...');
    const predT = model.predict(ds.X_test);
    const predArr = await predT.array(); predT.dispose();
    const truthArr = await ds.y_test.array();

    const results = computePerStockMetrics(predArr, truthArr, ds.symbols, ds.horizons, tunedThresholds || 0.5);
    drawAccuracyBarChart(results);
    drawTimelines(results, ds.horizons, ds.baseDatesTest);
    drawConfusions(results);

    const overall = overallAccuracy(predArr, truthArr, tunedThresholds || 0.5);
    const dov = results.find(r => r.symbol === 'DOV');
    log
