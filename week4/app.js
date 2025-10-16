// app.js
// Drop-in replacement for your app controller.
// Changes:
//  - Uses new loader.featuresPerStock to set input size
//  - Trains base model, then **fine-tunes DOV specialist**
//  - Tunes per-output thresholds on validation and uses them in evaluation
//  - Merges specialist predictions for DOV before scoring

import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.model = null;
    this.currentPredictions = null;
    this.accuracyChart = null;
    this.isTraining = false;

    this.horizons = 3;
    this.targetTicker = 'DOV'; // specialist target

    this.init();
  }

  init(){
    const fileInput = document.getElementById('csvFile');
    const trainBtn  = document.getElementById('trainBtn');
    const predictBtn = document.getElementById('predictBtn');

    fileInput.addEventListener('change', (e)=>this.handleFileUpload(e));
    trainBtn.addEventListener('click', ()=>this.trainModel());
    predictBtn.addEventListener('click', ()=>this.runPrediction());
  }

  async handleFileUpload(e){
    const file=e.target.files[0]; if(!file) return;
    const status = document.getElementById('status');
    try{
      status.textContent='Loading CSV...';
      await this.dataLoader.loadCSV(file);

      status.textContent='Preparing sequences...';
      const prepared = this.dataLoader.createSequences(12, this.horizons);

      document.getElementById('trainBtn').disabled=false;
      status.textContent='Data ready. Click Train Model.';
    }catch(err){
      status.textContent=`Error: ${err.message}`;
      console.error(err);
    }
  }

  async trainModel(){
    if (this.isTraining) return;
    this.isTraining=true;
    document.getElementById('trainBtn').disabled=true;
    document.getElementById('predictBtn').disabled=true;

    const st=document.getElementById('status');
    try{
      const { X_train, y_train, X_test, y_test, symbols, featuresPerStock } = this.dataLoader;

      // Chronological split: use part of X_test as validation (20% of whole time is already test)
      // We'll carve 25% of train as val:
      const nTrain = X_train.shape[0];
      const nVal = Math.max(1, Math.floor(nTrain*0.2));
      const nTr  = nTrain - nVal;

      const X_tr = X_train.slice([0,0,0],[nTr,-1,-1]);
      const y_tr = y_train.slice([0,0],[nTr,-1]);
      const X_val = X_train.slice([nTr,0,0],[nVal,-1,-1]);
      const y_val = y_train.slice([nTr,0],[nVal,-1]);

      const inputShape=[12, symbols.length*featuresPerStock];
      this.model = new GRUModel(inputShape, symbols.length*this.horizons, this.horizons);

      st.textContent='Training base model...';
      await this.model.train(X_tr, y_tr, X_val, y_val, 50, 32);

      // Specialist for DOV (if present)
      const dovIndex = symbols.indexOf(this.targetTicker);
      if (dovIndex !== -1){
        st.textContent='Fine-tuning DOV specialist...';
        await this.model.trainSpecialist(X_tr, y_tr, X_val, y_val, dovIndex, 30, 32);
        st.textContent='Specialist done.';
      } else {
        st.textContent='DOV not found — specialist skipped.';
      }

      // Threshold tuning (per-output) on validation
      const baseVal = this.model.model.predict(X_val);
      let mergedVal = baseVal;
      if (this.model.specialist && dovIndex !== -1) {
        mergedVal = await this.model.predictWithSpecialist(X_val, baseVal, dovIndex);
      }
      this.model.tunedThresholds = GRUModel.tuneThresholds(mergedVal, y_val, 0.30, 0.70, 0.01);
      mergedVal.dispose?.();

      document.getElementById('predictBtn').disabled=false;
      st.textContent='Training complete. Run Prediction to evaluate.';

      // Cleanup
      X_tr.dispose(); y_tr.dispose(); X_val.dispose(); y_val.dispose();
    }catch(err){
      st.textContent=`Training error: ${err.message}`;
      console.error(err);
    }finally{
      this.isTraining=false;
    }
  }

  async runPrediction(){
    if (!this.model){ alert('Train the model first'); return; }
    const st=document.getElementById('status');
    try{
      st.textContent='Running predictions...';
      const { X_test, y_test, symbols } = this.dataLoader;

      const base = await this.model.predict(X_test);
      const dovIdx = symbols.indexOf(this.targetTicker);
      const merged = (this.model.specialist && dovIdx !== -1)
        ? await this.model.predictWithSpecialist(X_test, base, dovIdx)
        : base;

      const thresholds = this.model.tunedThresholds || 0.5;

      // ---- per-stock accuracy using tuned thresholds ----
      const stockAcc = GRUModel.perStockAccuracy(y_test, merged, symbols, this.horizons, thresholds);

      // Build stockPredictions timeline (correct/wrong) for charts
      const arrY = y_test.arraySync();
      const arrP = merged.arraySync();
      const predictions = {};
      symbols.forEach((sym, sIdx)=>{
        const list=[];
        for (let i=0;i<arrY.length;i++){
          for (let h=0;h<this.horizons;h++){
            const j=sIdx*this.horizons+h;
            const ok = (arrY[i][j]>=0.5) === (arrP[i][j] >= (Array.isArray(thresholds)?thresholds[j]:thresholds));
            list.push({ correct: ok, true: arrY[i][j], pred: arrP[i][j] });
          }
        }
        predictions[sym]=list;
      });

      this.visualizeResults(stockAcc, predictions);

      const overall = Object.values(stockAcc).reduce((a,b)=>a+b,0)/symbols.length;
      const dov = stockAcc[this.targetTicker];
      st.textContent = `Prediction complete. Overall ${(overall*100).toFixed(2)}% | DOV ${(dov*100||0).toFixed(2)}%`;

      // Cleanup
      merged.dispose?.(); base.dispose?.();
    }catch(err){
      st.textContent=`Prediction error: ${err.message}`;
      console.error(err);
    }
  }

  // ---------- Charts (same as your original, minor tweaks) ----------
  visualizeResults(accuracies, predictions){
    this.createAccuracyChart(accuracies);
    this.createTimelineCharts(predictions);
  }

  createAccuracyChart(accuracies){
    const ctx=document.getElementById('accuracyChart').getContext('2d');

    const pairs = Object.entries(accuracies).sort((a,b)=>b[1]-a[1]);
    const labels=pairs.map(p=>p[0]);
    const data  =pairs.map(p=>+(p[1]*100).toFixed(2));

    if (this.accuracyChart) this.accuracyChart.destroy();

    this.accuracyChart = new Chart(ctx,{
      type:'bar',
      data:{ labels, datasets:[{ label:'Accuracy (%)', data, backgroundColor: data.map(v=> v>=70?'rgba(34,197,94,.85)': v>=60?'rgba(59,130,246,.7)':'rgba(239,68,68,.7)') }] },
      options:{ indexAxis:'y', scales:{ x:{ beginAtZero:true, max:100 } }, plugins:{ legend:{ display:false } } }
    });
  }

  createTimelineCharts(predictions){
    const container=document.getElementById('timelineContainer');
    container.innerHTML='';

    // Show DOV first if present
    const ordered = Object.keys(predictions).sort((a,b)=> a==='DOV'?-1 : b==='DOV'?1 : 0).slice(0,3);

    ordered.forEach(symbol=>{
      const sample=predictions[symbol];
      const div=document.createElement('div');
      div.className='stock-chart';
      div.innerHTML=`<h4>${symbol} Prediction Timeline</h4><canvas id="tl-${symbol}"></canvas>`;
      container.appendChild(div);

      const size=Math.min(50,sample.length);
      const s=sample.slice(0,size);
      const labels=s.map((_,i)=>`Pred ${i+1}`);
      const correct=s.map(p=>p.correct?1:0);

      new Chart(document.getElementById(`tl-${symbol}`).getContext('2d'),{
        type:'line',
        data:{ labels, datasets:[{ label:'Correct', data:correct, borderColor:'rgb(34,197,94)', backgroundColor:'rgba(34,197,94,.2)', fill:true, tension:.35 }] },
        options:{ scales:{ y:{ min:0,max:1, ticks:{ callback:v=>v===1?'Correct':v===0?'Wrong':'' } } }, plugins:{ legend:{ display:false } } }
      });
    });
  }
}

// Init
document.addEventListener('DOMContentLoaded', ()=> new StockPredictionApp());
