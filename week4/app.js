// app.js
// App glue: uses user's original UI flow but adds validation split + per-stock threshold tuning. :contentReference[oaicite:5]{index=5}
import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.model = null;
    this.thresholds = null;

    this.accuracyChart = null;
    this.trainChart = null;

    this.isTraining = false;

    this.cache = { seqLen:12, horizon:3 };

    this.init();
  }

  init() {
    const fileInput = document.getElementById('csvFile');
    const trainBtn = document.getElementById('trainBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');

    fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    trainBtn.addEventListener('click', () => this.trainModel());
    predictBtn.addEventListener('click', () => this.runEvaluation());
    resetBtn.addEventListener('click', () => this.reset());

    this.setStatus('Upload CSV file to begin');
  }

  setStatus(msg) {
    const el = document.getElementById('status');
    if (el) el.textContent = msg;
  }

  setProgress(v) {
    const p = document.getElementById('trainingProgress');
    if (p) p.value = v;
  }

  readNum(id, def) {
    const el = document.getElementById(id);
    const v = parseFloat(el?.value);
    return Number.isFinite(v) ? v : def;
  }

  readUnits(id, def) {
    const el = document.getElementById(id);
    const arr = (el?.value||'').split(',').map(s=>parseInt(s.trim(),10)).filter(x=>Number.isFinite(x)&&x>0);
    return arr.length?arr:def;
  }

  async handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      this.setStatus('Loading CSV...');
      await this.dataLoader.loadCSV(file);

      const seqLen = Math.max(6, Math.min(60, this.readNum('seqLen', 12)));
      const horizon = Math.max(1, Math.min(5, this.readNum('horizon', 3)));
      this.cache.seqLen = seqLen; this.cache.horizon = horizon;

      this.setStatus('Building sequences…');
      this.dataLoader.createSequences(seqLen, horizon, { testSplit: 0.2, valSplit: 0.1 });

      document.getElementById('trainBtn').disabled = false;
      this.setStatus('Data ready. Click Train.');
    } catch (err) {
      console.error(err);
      this.setStatus('Error: ' + err.message);
    }
  }

  async trainModel() {
    if (this.isTraining) return;
    if (!this.dataLoader?.X_train) return alert('Load CSV first.');

    this.isTraining = true;
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('predictBtn').disabled = true;

    const { X_train, y_train, X_val, y_val, symbols } = this.dataLoader;
    const seqLen = this.cache.seqLen;
    const featDim = symbols.length * 2;
    const horizon = this.cache.horizon;
    const outputSize = symbols.length * horizon;

    const units = this.readUnits('units', [96,64]);
    const dropout = this.readNum('dropout', 0.25);
    const lr = this.readNum('lr', 1e-3);
    const bidi = document.getElementById('bidi')?.checked ?? true;
    const convFilters = this.readNum('convFilters', 32);
    const epochs = Math.max(5, this.readNum('epochs', 40));
    const batch = Math.max(8, this.readNum('batch', 32));

    this.model = new GRUModel([seqLen, featDim], outputSize, {
      units, dropout, learningRate: lr, bidirectional: bidi, convFilters
    });

    const lossSeries=[]; const accSeries=[];
    this.setStatus('Training…');
    this.setProgress(0);

    await this.model.train(X_train, y_train, X_val, y_val, epochs, batch, (epoch, logs) => {
      const pct = ((epoch+1)/epochs)*100;
      this.setProgress(pct);
      this.setStatus(`Epoch ${epoch+1}/${epochs} — loss ${logs.loss.toFixed(4)} acc ${(logs.binaryAccuracy*100).toFixed(1)}% val_acc ${(logs.val_binaryAccuracy*100).toFixed(1)}%`);
      lossSeries.push({e:epoch+1, loss:logs.loss, val_loss:logs.val_loss});
      accSeries.push({e:epoch+1, acc:logs.binaryAccuracy, val_acc:logs.val_binaryAccuracy});
      this.renderTrainChart(lossSeries, accSeries);
    });

    // Threshold tuning on validation set
    const valPred = await this.model.predict(X_val);
    this.thresholds = this.model.tuneThresholds(y_val, valPred, symbols, horizon);
    valPred.dispose();

    document.getElementById('predictBtn').disabled = false;
    this.setStatus('Training complete. Click Evaluate.');
    this.isTraining = false;
  }

  async runEvaluation() {
    if (!this.model) return alert('Train first.');
    const { X_test, y_test, symbols, testDates } = this.dataLoader;

    this.setStatus('Predicting on test set…');
    const preds = await this.model.predict(X_test);

    const evalRes = this.model.evaluatePerStock(y_test, preds, symbols, this.cache.horizon, this.thresholds);

    preds.dispose();

    // Accuracy chart
    this.renderAccuracyChart(evalRes.stockAccuracies);

    // Timelines for all symbols
    this.renderTimelines(evalRes.stockPredictions, symbols, testDates);

    this.setStatus('Evaluation complete.');
  }

  renderAccuracyChart(accuracies) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    if (this.accuracyChart) this.accuracyChart.destroy();

    const entries = Object.entries(accuracies).sort((a,b)=>b[1]-a[1]);
    const labels = entries.map(([s])=>s);
    const data = entries.map(([,a])=>+(a*100).toFixed(2));

    this.accuracyChart = new Chart(ctx, {
      type:'bar',
      data:{ labels, datasets:[{ label:'Accuracy (%)', data, borderWidth:1 }] },
      options:{
        indexAxis:'y',
        scales:{ x:{ min:0, max:100, ticks:{ callback:v=>v+'%' } } },
        plugins:{ legend:{display:false} }
      }
    });
  }

  renderTrainChart(lossSeries, accSeries) {
    const ctx = document.getElementById('trainChart').getContext('2d');
    if (this.trainChart) this.trainChart.destroy();
    this.trainChart = new Chart(ctx, {
      type:'line',
      data:{
        labels: lossSeries.map(d=>d.e),
        datasets:[
          { label:'loss', data:lossSeries.map(d=>d.loss), yAxisID:'y' },
          { label:'val_loss', data:lossSeries.map(d=>d.val_loss), yAxisID:'y' },
          { label:'acc', data:accSeries.map(d=>d.acc), yAxisID:'y1' },
          { label:'val_acc', data:accSeries.map(d=>d.val_acc), yAxisID:'y1' },
        ]
      },
      options:{
        interaction:{ mode:'index', intersect:false },
        scales:{ y:{ position:'left' }, y1:{ position:'right', min:0, max:1, grid:{ drawOnChartArea:false } } },
        plugins:{ legend:{ position:'bottom' } }
      }
    });
  }

  renderTimelines(predictions, symbols, dates) {
    const container = document.getElementById('timelineContainer');
    container.innerHTML = '';
    const horizon = this.cache.horizon;

    symbols.forEach(sym => {
      const card = document.createElement('div');
      card.className = 'stock';
      const title = document.createElement('h4');
      title.textContent = sym;
      card.appendChild(title);

      for (let h=1; h<=horizon; h++) {
        const row = document.createElement('div');
        row.className = 'row';
        const label = document.createElement('div');
        label.className = 'label'; label.textContent = `D+${h}`;
        row.appendChild(label);

        const cells = document.createElement('div');
        cells.className = 'cells';

        const list = predictions[sym].filter(p=>p.horizon===h);
        for (let i=0;i<list.length;i++) {
          const cell=document.createElement('div');
          cell.className = 'cell ' + (list[i].correct?'ok':'bad');
          cell.title = `${sym} ${dates[i] ?? ''} D+${h} • ${list[i].correct?'Correct':'Wrong'}`;
          cells.appendChild(cell);
        }
        row.appendChild(cells);
        card.appendChild(row);
      }
      container.appendChild(card);
    });
  }

  reset() {
    try {
      if (this.accuracyChart) this.accuracyChart.destroy();
      if (this.trainChart) this.trainChart.destroy();
      this.accuracyChart = null; this.trainChart = null;
      if (this.model) this.model.dispose();
      if (this.dataLoader) this.dataLoader.dispose();
      this.model = null; this.thresholds = null;
      document.getElementById('trainBtn').disabled = true;
      document.getElementById('predictBtn').disabled = true;
      document.getElementById('timelineContainer').innerHTML = '';
      this.setProgress(0);
      this.setStatus('Reset. Upload CSV to start again.');
    } catch {}
  }
}

document.addEventListener('DOMContentLoaded', ()=> new StockPredictionApp());
