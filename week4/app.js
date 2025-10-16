// app.js
// Glue UI: data loading, model training, evaluation, and visualizations.

import { DataLoader } from './data-loader.js';
import { GRUStockModel, computePerStockMetrics } from './gru.js';

const els = {
  file: document.getElementById('csvFile'),
  prepareBtn: document.getElementById('prepareBtn'),
  trainBtn: document.getElementById('trainBtn'),
  predictBtn: document.getElementById('predictBtn'),
  runAllBtn: document.getElementById('runAllBtn'),
  saveBtn: document.getElementById('saveBtn'),
  loadBtn: document.getElementById('loadBtn'),
  resetBtn: document.getElementById('resetBtn'),
  status: document.getElementById('status'),
  prog: document.getElementById('prog'),
  log: document.getElementById('log'),
  dataInfo: document.getElementById('dataInfo'),
  accChart: document.getElementById('accChart'),
  trainChart: document.getElementById('trainChart'),
  timeline: document.getElementById('timeline'),
  confusions: document.getElementById('confusions'),
  seqLen: document.getElementById('seqLen'),
  horizon: document.getElementById('horizon'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  valsplit: document.getElementById('valsplit'),
  lr: document.getElementById('lr'),
  units: document.getElementById('units'),
  dropout: document.getElementById('dropout'),
  bidi: document.getElementById('bidi'),
};

let loader = null;
let dataset = null;
let model = null;
let charts = { acc: null, train: null };

function log(msg) {
  const time = new Date().toLocaleTimeString();
  els.log.textContent += `[${time}] ${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(msg, progress = null) {
  els.status.textContent = msg;
  if (progress != null) {
    els.prog.value = progress;
  }
}

function enableControls({ prepared = false, trained = false } = {}) {
  els.trainBtn.disabled = !prepared;
  els.predictBtn.disabled = !(prepared && trained);
  els.runAllBtn.disabled = !prepared;
  els.saveBtn.disabled = !trained;
}

function parseUnits(str) {
  const arr = (str || '').split(',').map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n) && n > 0);
  return arr.length ? arr : [64, 32];
}

function releaseTensors(obj) {
  for (const k of Object.keys(obj)) {
    if (obj[k] && typeof obj[k].dispose === 'function') {
      try { obj[k].dispose(); } catch {}
    }
  }
}

function clearVisuals() {
  if (charts.acc) { charts.acc.destroy(); charts.acc = null; }
  if (charts.train) { charts.train.destroy(); charts.train = null; }
  els.timeline.innerHTML = '';
  els.confusions.innerHTML = '';
}

async function prepare() {
  try {
    clearVisuals();
    if (!els.file.files[0]) {
      alert('Please choose a CSV file first.');
      return;
    }
    const seqLen = Math.max(6, Math.min(60, parseInt(els.seqLen.value, 10) || 12));
    const horizon = Math.max(1, Math.min(5, parseInt(els.horizon.value, 10) || 3));

    loader = new DataLoader({ seqLen, horizon, testSplit: 0.2 });
    setStatus('Parsing CSV & preparing dataset…', 0.05);
    log('Parsing CSV…');

    dataset = await loader.loadFromFile(els.file.files[0], (info) => {
      els.dataInfo.innerHTML = `
        <div>Symbols: <b>${info.symbols.join(', ')}</b></div>
        <div>Dates: <b>${info.start}</b> → <b>${info.end}</b></div>
        <div>Samples: <b>${info.samples}</b> (Train ${info.train}, Test ${info.test})</div>
        <div>Shapes: X[${info.train}, ${info.seqLen}, ${info.featDim}], y[${info.train}, ${info.outDim}]</div>
      `;
      log(`Aligned ${info.symbols.length} symbols across ${info.samples} samples.`);
    });

    setStatus('Dataset ready.', 0.2);
    enableControls({ prepared: true, trained: false });
    log('Dataset prepared.');
  } catch (err) {
    console.error(err);
    alert('Error preparing data: ' + err.message);
    setStatus('Error.');
  }
}

async function train() {
  if (!dataset) return;
  clearVisuals();

  const params = {
    seqLen: dataset.seqLen,
    featDim: dataset.featDim,
    horizon: dataset.horizon,
    numStocks: dataset.symbols.length,
    units: parseUnits(els.units.value),
    dropout: parseFloat(els.dropout.value || '0.2'),
    learningRate: parseFloat(els.lr.value || '0.001'),
    bidirectional: !!els.bidi.checked,
  };
  model = new GRUStockModel(params);
  model.summary();

  const epochs = parseInt(els.epochs.value || '25', 10);
  const batchSize = parseInt(els.batch.value || '32', 10);
  const validationSplit = Math.min(0.4, Math.max(0, parseFloat(els.valsplit.value || '0.1')));

  const lossSeries = [];
  const accSeries = [];
  setStatus('Training…', 0.25);

  const history = await model.fit(dataset.X_train, dataset.y_train, {
    epochs,
    batchSize,
    validationSplit,
    onEpoch: (epoch, logs) => {
      const { loss, val_loss, binaryAccuracy, val_binaryAccuracy } = logs;
      lossSeries.push({ epoch, loss, val_loss });
      accSeries.push({ epoch, acc: binaryAccuracy, val_acc: val_binaryAccuracy });
      setStatus(`Epoch ${epoch + 1}/${epochs} — loss ${loss.toFixed(4)} acc ${(binaryAccuracy * 100).toFixed(2)}%`, 0.25 + 0.65 * ((epoch + 1) / epochs));
      log(`Epoch ${epoch + 1}: loss=${loss.toFixed(4)}, val_loss=${val_loss?.toFixed(4)}, acc=${(binaryAccuracy * 100).toFixed(2)}%, val_acc=${(val_binaryAccuracy * 100).toFixed(2)}%`);
      renderTrainChart(lossSeries, accSeries);
    },
  });

  const final = history.history;
  log(`Training complete. Final acc=${(final.binaryAccuracy.slice(-1)[0] * 100).toFixed(2)}%, val_acc=${(final.val_binaryAccuracy.slice(-1)[0] * 100).toFixed(2)}%`);
  setStatus('Training complete.', 0.9);
  enableControls({ prepared: true, trained: true });

  // Free training tensors (keep test set)
  releaseTensors({ X_train: dataset.X_train, y_train: dataset.y_train });
  dataset.X_train = null; dataset.y_train = null;
}

async function evaluate() {
  if (!model || !dataset) return;

  setStatus('Evaluating…', 0.92);
  const evalStats = await model.evaluate(dataset.X_test, dataset.y_test);
  log(`Eval — loss=${evalStats.loss.toFixed(4)}, acc=${(evalStats.acc * 100).toFixed(2)}%`);

  const predsT = await model.predict(dataset.X_test);
  const yTrue = await dataset.y_test.array();
  const yPred = await predsT.array();

  // Metrics & timelines
  const { accuracies, confusions, timelines } = computePerStockMetrics({
    yTrue, yPred, symbols: dataset.symbols, horizon: dataset.horizon, threshold: 0.5,
  });

  // Render Accuracy Ranking
  renderAccuracyChart(dataset.symbols, accuracies);

  // Render Timelines
  renderTimelines(dataset.symbols, timelines, dataset.testDates, dataset.horizon);

  // Render Confusions
  renderConfusions(confusions);

  // Cleanup
  predsT.dispose();
  setStatus('Done.', 1);
}

function renderAccuracyChart(symbols, accuracies) {
  const pairs = symbols.map((s, i) => ({ s, a: accuracies[i] })).sort((a, b) => b.a - a.a);
  const labels = pairs.map(p => p.s);
  const data = pairs.map(p => +(p.a * 100).toFixed(2));

  if (charts.acc) charts.acc.destroy();
  charts.acc = new Chart(els.accChart.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Accuracy (%)', data, borderWidth: 1 }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: { min: 0, max: 100, ticks: { callback: v => v + '%' } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw}%` } },
      },
    },
  });
}

function renderTrainChart(lossSeries, accSeries) {
  const ctx = els.trainChart.getContext('2d');
  if (charts.train) charts.train.destroy();

  charts.train = new Chart(ctx, {
    type: 'line',
    data: {
      labels: lossSeries.map(d => d.epoch + 1),
      datasets: [
        { label: 'loss', data: lossSeries.map(d => d.loss), yAxisID: 'y' },
        { label: 'val_loss', data: lossSeries.map(d => d.val_loss ?? null), yAxisID: 'y' },
        { label: 'acc', data: accSeries.map(d => d.acc), yAxisID: 'y1' },
        { label: 'val_acc', data: accSeries.map(d => d.val_acc ?? null), yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { type: 'linear', position: 'left' },
        y1: { type: 'linear', position: 'right', min: 0, max: 1, grid: { drawOnChartArea: false } },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function renderTimelines(symbols, timelines, dates, horizon) {
  els.timeline.innerHTML = '';
  symbols.forEach((sym, si) => {
    const card = document.createElement('div');
    card.className = 'stock-timeline';
    const title = document.createElement('h4');
    title.textContent = sym;
    card.appendChild(title);

    const H = horizon;
    for (let h = 0; h < H; h++) {
      const row = document.createElement('div');
      row.className = 'row';
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = `D+${h + 1}`;
      row.appendChild(label);

      const cells = document.createElement('div');
      cells.className = 'cells';
      const arr = timelines[sym][h];
      for (let i = 0; i < arr.length; i++) {
        const cell = document.createElement('div');
        cell.className = `cell ${arr[i] ? 'ok' : 'bad'}`;
        cell.title = `${sym} @ ${dates[i]} • D+${h + 1} • ${arr[i] ? 'Correct' : 'Wrong'}`;
        cells.appendChild(cell);
      }
      row.appendChild(cells);
      card.appendChild(row);
    }
    els.timeline.appendChild(card);
  });
}

function renderConfusions(confusions) {
  const table = document.createElement('table');
  table.style.borderCollapse = 'collapse';
  table.style.width = '100%';
  const thead = document.createElement('thead');
  thead.innerHTML = `<tr>
      <th style="text-align:left;padding:6px;border-bottom:1px solid #1f2937">Symbol</th>
      <th style="padding:6px;border-bottom:1px solid #1f2937">TP</th>
      <th style="padding:6px;border-bottom:1px solid #1f2937">FP</th>
      <th style="padding:6px;border-bottom:1px solid #1f2937">TN</th>
      <th style="padding:6px;border-bottom:1px solid #1f2937">FN</th>
    </tr>`;
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const sym of Object.keys(confusions)) {
    const { TP, FP, TN, FN } = confusions[sym];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="padding:6px;border-bottom:1px solid #1f2937">${sym}</td>
      <td style="padding:6px;border-bottom:1px solid #1f2937">${TP}</td>
      <td style="padding:6px;border-bottom:1px solid #1f2937">${FP}</td>
      <td style="padding:6px;border-bottom:1px solid #1f2937">${TN}</td>
      <td style="padding:6px;border-bottom:1px solid #1f2937">${FN}</td>
    `;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  els.confusions.innerHTML = '';
  els.confusions.appendChild(table);
}

els.prepareBtn.addEventListener('click', prepare);
els.trainBtn.addEventListener('click', train);
els.predictBtn.addEventListener('click', evaluate);

els.runAllBtn.addEventListener('click', async () => {
  if (!dataset) await prepare();
  await train();
  await evaluate();
});

els.saveBtn.addEventListener('click', async () => {
  if (!model) return;
  setStatus('Saving model…');
  await model.save('multi_stock_gru');
  setStatus('Model saved to browser storage.');
  log('Model saved (localstorage://multi_stock_gru).');
});

els.loadBtn.addEventListener('click', async () => {
  try {
    setStatus('Loading saved model…');
    model = await GRUStockModel.load('multi_stock_gru');
    setStatus('Model loaded.');
    log('Loaded model from localstorage://multi_stock_gru');
    enableControls({ prepared: !!dataset, trained: true });
  } catch (e) {
    alert('No saved model found or load failed.');
    setStatus('Load failed.');
  }
});

els.resetBtn.addEventListener('click', () => {
  try {
    if (model?.model) model.model.dispose();
    model = null;
    if (dataset) {
      releaseTensors(dataset);
      dataset = null;
    }
    clearVisuals();
    els.log.textContent = '';
    els.dataInfo.textContent = '';
    setStatus('Reset.');
    enableControls({ prepared: false, trained: false });
  } catch {}
});

// Auto-enable prepare when file chosen
els.file.addEventListener('change', () => {
  setStatus('CSV selected. Click "Load CSV & Prepare".');
});
