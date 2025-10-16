// app.js
// Orchestrates UI, data loading, model training, prediction, and visualization

import { prepareDatasetFromFile, DEFAULT_SEQ_LEN, DEFAULT_HORIZONS } from './data-loader.js';
import { GRUClassifier } from './gru.js';

const ui = {
  fileInput: document.getElementById('csvFile'),
  loadBtn: document.getElementById('btnLoad'),
  trainBtn: document.getElementById('btnTrain'),
  predictBtn: document.getElementById('btnPredict'),
  saveBtn: document.getElementById('btnSave'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  progress: document.getElementById('progress'),
  progressBar: document.getElementById('progressBar'),
  progressText: document.getElementById('progressText'),
  shapes: document.getElementById('shapes'),
  fileName: document.getElementById('fileName'),
  log: document.getElementById('log'),
  accCanvas: document.getElementById('accChart'),
  timelineContainer: document.getElementById('timelineContainer'),
  confusionContainer: document.getElementById('confusionContainer'),
};

let dataset = null;
let model = null;
let accChart = null;
let latestResults = null;

function setDisabled(el, flag) {
  el.disabled = !!flag;
}

function log(msg) {
  const time = new Date().toLocaleTimeString();
  ui.log.textContent += `[${time}] ${msg}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function showShapes(meta, symbols) {
  ui.shapes.textContent = `SeqLen=${meta.seqLen}, FeatureDim=${meta.featureDim}, Samples: ${meta.totalSamples} (train ${meta.numTrain} / test ${meta.numTest}), Stocks=${symbols.length}`;
}

function updateProgress(pct, text) {
  const v = Math.max(0, Math.min(100, Math.round(pct)));
  ui.progressBar.style.width = `${v}%`;
  ui.progressText.textContent = text || `${v}%`;
}

function enableControls(state) {
  // state: 'init' | 'loaded' | 'trained'
  if (state === 'init') {
    setDisabled(ui.trainBtn, true);
    setDisabled(ui.predictBtn, true);
    setDisabled(ui.saveBtn, true);
  } else if (state === 'loaded') {
    setDisabled(ui.trainBtn, false);
    setDisabled(ui.predictBtn, true);
    setDisabled(ui.saveBtn, true);
  } else if (state === 'trained') {
    setDisabled(ui.trainBtn, false);
    setDisabled(ui.predictBtn, false);
    setDisabled(ui.saveBtn, false);
  }
}

function computePerStockMetrics(predProbs, yTrue, symbols, horizons, threshold = 0.5) {
  const S = symbols.length;
  const H = horizons.length;
  const numTest = yTrue.length;

  const perStock = symbols.map(() => ({
    correct: 0,
    total: 0,
    timeline: Array.from({ length: H }, () => []),
    confusion: [0, 0, 0, 0], // TN, FP, FN, TP
  }));

  for (let i = 0; i < numTest; i++) {
    const gt = yTrue[i];
    const pp = predProbs[i];
    for (let s = 0; s < S; s++) {
      for (let h = 0; h < H; h++) {
        const idx = s * H + h;
        const y = gt[idx] >= 0.5 ? 1 : 0;
        const p = pp[idx] >= threshold ? 1 : 0;
        const ok = y === p;
        perStock[s].correct += ok ? 1 : 0;
        perStock[s].total += 1;
        perStock[s].timeline[h].push(ok);

        if (y === 0 && p === 0) perStock[s].confusion[0]++;
        else if (y === 0 && p === 1) perStock[s].confusion[1]++;
        else if (y === 1 && p === 0) perStock[s].confusion[2]++;
        else if (y === 1 && p === 1) perStock[s].confusion[3]++;
      }
    }
  }

  return perStock.map((st, idx) => ({
    symbol: symbols[idx],
    accuracy: st.total ? st.correct / st.total : 0,
    timeline: st.timeline,
    confusion: st.confusion
  }));
}

function drawAccuracyBarChart(results) {
  const sorted = [...results].sort((a, b) => b.accuracy - a.accuracy);
  const labels = sorted.map(r => r.symbol);
  const data = sorted.map(r => +(r.accuracy * 100).toFixed(2));

  if (accChart) {
    accChart.destroy();
    accChart = null;
  }
  const ctx = ui.accCanvas.getContext('2d');
  accChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Accuracy (%)', data, borderWidth: 1 }]
    },
    options: {
      responsive: true,
      indexAxis: 'y',
      scales: {
        x: { min: 0, max: 100, ticks: { callback: v => `${v}%` } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: (it) => `Accuracy: ${it.parsed.x.toFixed(2)}%` }
        }
      }
    }
  });
}

function drawTimelines(results, horizons, baseDates) {
  ui.timelineContainer.innerHTML = '';
  const H = horizons.length;

  results.forEach(r => {
    const wrapper = document.createElement('div');
    wrapper.className = 'timeline';

    const title = document.createElement('div');
    title.className = 'timeline-title';
    title.textContent = `${r.symbol} Prediction Timeline`;
    wrapper.appendChild(title);

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(600, r.timeline[0].length * 6);
    canvas.height = 24 + H * 18;
    wrapper.appendChild(canvas);

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = `<span class="ok"></span> Correct &nbsp;&nbsp; <span class="bad"></span> Wrong`;
    wrapper.appendChild(legend);

    const ctx = canvas.getContext('2d');
    const cellW = 6, cellH = 12, padLeft = 60, padTop = 6;

    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#333';
    ctx.fillText('Correct', 5, 12);
    ctx.fillText('Wrong', 5, 30);

    for (let h = 0; h < H; h++) {
      const seq = r.timeline[h];
      ctx.fillStyle = '#555';
      ctx.fillText(`+${h + 1}d`, 5, padTop + h * (cellH + 6) + cellH - 2);
      for (let i = 0; i < seq.length; i++) {
        ctx.fillStyle = seq[i] ? '#27ae60' : '#e74c3c';
        ctx.fillRect(padLeft + i * cellW, padTop + h * (cellH + 6), cellW - 1, cellH);
      }
    }

    // Date ticks
    ctx.fillStyle = '#222';
    const N = r.timeline[0].length;
    const step = Math.max(1, Math.floor(N / 12));
    for (let i = 0; i < N; i += step) {
      const x = padLeft + i * cellW;
      ctx.fillRect(x, canvas.height - 10, 1, 8);
      ctx.save();
      ctx.translate(x + 2, canvas.height - 2);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(baseDates[i] || '', 0, 0);
      ctx.restore();
    }

    ui.timelineContainer.appendChild(wrapper);
  });
}

function renderConfusionMatrices(results) {
  ui.confusionContainer.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'cm-grid';

  results.forEach(r => {
    const [tn, fp, fn, tp] = r.confusion;
    const card = document.createElement('div');
    card.className = 'cm-card';
    card.innerHTML = `
      <div class="cm-title">${r.symbol}</div>
      <table class="cm-table">
        <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
        <tbody>
          <tr><td>True 0</td><td>${tn}</td><td>${fp}</td></tr>
          <tr><td>True 1</td><td>${fn}</td><td>${tp}</td></tr>
        </tbody>
      </table>`;
    grid.appendChild(card);
  });

  ui.confusionContainer.appendChild(grid);
}

// --- Event Handlers ----------------------------------------------------------

async function handleLoad() {
  try {
    const f = ui.fileInput.files?.[0];
    if (!f) {
      alert('Choose a CSV file first.');
      return;
    }
    ui.fileName.textContent = f.name;
    log('Loading & preparing dataset...');
    updateProgress(2, 'Preparing...');
    // Dispose old dataset tensors
    dataset?.X_train?.dispose();
    dataset?.y_train?.dispose();
    dataset?.X_test?.dispose();
    dataset?.y_test?.dispose();

    dataset = await prepareDatasetFromFile(f, {
      seqLen: DEFAULT_SEQ_LEN,
      horizons: DEFAULT_HORIZONS,
      testSplit: 0.2
    });

    showShapes(dataset.meta, dataset.symbols);
    updateProgress(10, 'Dataset ready');
    log('Dataset prepared successfully.');
    enableControls('loaded');
  } catch (err) {
    console.error(err);
    alert(`Data load error: ${err.message}`);
    log(`Error: ${err.message}`);
    enableControls('init');
    updateProgress(0, '');
  }
}

async function handleTrain() {
  if (!dataset) {
    alert('Load data first.');
    return;
  }
  try {
    const epochs = Math.max(1, parseInt(ui.epochs.value || '25', 10));
    const batch = Math.max(1, parseInt(ui.batch.value || '32', 10));

    model?.dispose();
    model = new GRUClassifier({
      seqLen: dataset.meta.seqLen,
      featureDim: dataset.meta.featureDim,
      numStocks: dataset.symbols.length,
      horizons: dataset.horizons,
      units: 64,
      learningRate: 1e-3
    });

    log('Training model...');
    updateProgress(12, 'Training...');
    await model.fit(
      dataset.X_train, dataset.y_train,
      {
        epochs, batchSize: batch, validationSplit: 0.1, shuffle: false
      },
      (epoch, logs) => {
        const pct = 12 + Math.round(((epoch + 1) / epochs) * 78); // 12%→90%
        const txt = `Epoch ${epoch + 1}/${epochs} — loss ${logs.loss?.toFixed(4)} acc ${(logs.binaryAccuracy * 100).toFixed(2)}%`;
        updateProgress(pct, txt);
        log(txt);
      }
    );

    updateProgress(92, 'Training complete');
    log('Training complete.');
    enableControls('trained');
  } catch (err) {
    console.error(err);
    alert(`Training error: ${err.message}`);
    log(`Error: ${err.message}`);
    updateProgress(0, '');
  }
}

async function handlePredict() {
  if (!dataset || !model) {
    alert('Load data and train model first.');
    return;
  }
  try {
    log('Running prediction on test split...');
    updateProgress(94, 'Predicting...');
    const predTensor = model.predict(dataset.X_test);
    const predArray = await predTensor.array();
    const yTrueArray = await dataset.y_test.array();
    predTensor.dispose();

    latestResults = computePerStockMetrics(
      predArray, yTrueArray,
      dataset.symbols, dataset.horizons, 0.5
    );

    drawAccuracyBarChart(latestResults);
    drawTimelines(latestResults, dataset.horizons, dataset.baseDatesTest);
    renderConfusionMatrices(latestResults);
    updateProgress(100, 'Prediction completed');
    log('Prediction completed. Results displayed below.');
  } catch (err) {
    console.error(err);
    alert(`Prediction error: ${err.message}`);
    log(`Error: ${err.message}`);
    updateProgress(0, '');
  }
}

async function handleSave() {
  if (!model) {
    alert('Train a model first.');
    return;
  }
  await model.save('tfjs_gru_stock_demo');
  log('Weights saved (downloaded).');
}

// --- Init --------------------------------------------------------------------

function init() {
  enableControls('init');
  ui.loadBtn.addEventListener('click', handleLoad);
  ui.trainBtn.addEventListener('click', handleTrain);
  ui.predictBtn.addEventListener('click', handlePredict);
  ui.saveBtn.addEventListener('click', handleSave);
}

init();
