// app.js
// UI orchestration, model training, evaluation, and interactive visualizations.

import { prepareDatasetFromFile, DEFAULT_SEQ_LEN, DEFAULT_HORIZONS } from './data-loader.js';
import { GRUClassifier } from './gru.js';

const ui = {
  fileInput: document.getElementById('csvFile'),
  fileName: document.getElementById('fileName'),
  loadBtn: document.getElementById('btnLoad'),
  trainBtn: document.getElementById('btnTrain'),
  predictBtn: document.getElementById('btnPredict'),
  saveBtn: document.getElementById('btnSave'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  shapes: document.getElementById('shapes'),
  log: document.getElementById('log'),
  progressBar: document.getElementById('progressBar'),
  progressText: document.getElementById('progressText'),
  accCanvas: document.getElementById('accChart'),
  timelineContainer: document.getElementById('timelineContainer'),
  confusionContainer: document.getElementById('confusionContainer'),
};

let dataset = null;
let model = null;
let accChart = null;

function setDisabled(el, v) { el.disabled = !!v; }
function enableState(state) {
  // 'init' | 'loaded' | 'trained'
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

function log(msg) {
  const time = new Date().toLocaleTimeString();
  ui.log.textContent += `[${time}] ${msg}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function progress(pct, text) {
  const v = Math.max(0, Math.min(100, Math.round(pct)));
  ui.progressBar.style.width = `${v}%`;
  ui.progressText.textContent = text || `${v}%`;
}

function showShapes(meta, symbols) {
  ui.shapes.textContent = `SeqLen=${meta.seqLen}, FeatureDim=${meta.featureDim}, Samples=${meta.totalSamples} (train ${meta.numTrain} / test ${meta.numTest}), Stocks=${symbols.length}`;
}

function computePerStockMetrics(predProbs, yTrue, symbols, horizons, threshold = 0.5) {
  const S = symbols.length, H = horizons.length, N = yTrue.length;
  const per = symbols.map(() => ({
    correct: 0, total: 0,
    timeline: Array.from({ length: H }, () => []),
    confusion: [0, 0, 0, 0] // TN FP FN TP
  }));

  for (let i = 0; i < N; i++) {
    const t = yTrue[i], p = predProbs[i];
    for (let s = 0; s < S; s++) {
      for (let h = 0; h < H; h++) {
        const idx = s * H + h;
        const y = t[idx] >= 0.5 ? 1 : 0;
        const ph = p[idx] >= threshold ? 1 : 0;
        const ok = y === ph;
        per[s].correct += ok ? 1 : 0;
        per[s].total += 1;
        per[s].timeline[h].push(ok);

        if (y === 0 && ph === 0) per[s].confusion[0]++;
        else if (y === 0 && ph === 1) per[s].confusion[1]++;
        else if (y === 1 && ph === 0) per[s].confusion[2]++;
        else per[s].confusion[3]++;
      }
    }
  }
  return per.map((st, i) => ({
    symbol: symbols[i],
    accuracy: st.total ? st.correct / st.total : 0,
    timeline: st.timeline,
    confusion: st.confusion
  }));
}

function drawAccuracyBarChart(results) {
  const sorted = [...results].sort((a, b) => b.accuracy - a.accuracy);
  const labels = sorted.map(r => r.symbol);
  const data = sorted.map(r => +(r.accuracy * 100).toFixed(2));

  if (accChart) { accChart.destroy(); accChart = null; }
  const ctx = ui.accCanvas.getContext('2d');
  accChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Accuracy (%)', data }] },
    options: {
      indexAxis: 'y',
      scales: { x: { min: 0, max: 100, ticks: { callback: v => `${v}%` } } },
      plugins: { legend: { display: false } },
      responsive: true
    }
  });
}

function drawTimelines(results, horizons, baseDates) {
  ui.timelineContainer.innerHTML = '';
  const H = horizons.length;
  results.forEach(r => {
    const wrap = document.createElement('div');
    wrap.className = 'timeline';

    const title = document.createElement('div');
    title.className = 'timeline-title';
    title.textContent = `${r.symbol} — prediction correctness (rows: +1d, +2d, +3d)`;
    wrap.appendChild(title);

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(600, r.timeline[0].length * 6);
    canvas.height = H * 18 + 28;
    wrap.appendChild(canvas);

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = `<span class="green"></span>Correct &nbsp; <span class="red"></span>Wrong`;
    wrap.appendChild(legend);

    const ctx = canvas.getContext('2d');
    const cellW = 6, cellH = 12, padL = 60, padT = 8;

    ctx.font = '12px system-ui, sans-serif';

    for (let h = 0; h < H; h++) {
      ctx.fillStyle = '#444';
      ctx.fillText(`+${h + 1}d`, 8, padT + h * (cellH + 6) + cellH - 2);
      const seq = r.timeline[h];
      for (let i = 0; i < seq.length; i++) {
        ctx.fillStyle = seq[i] ? '#2ecc71' : '#e74c3c';
        ctx.fillRect(padL + i * cellW, padT + h * (cellH + 6), cellW - 1, cellH);
      }
    }

    // bottom date ticks
    const N = r.timeline[0].length;
    const step = Math.max(1, Math.floor(N / 12));
    ctx.fillStyle = '#222';
    for (let i = 0; i < N; i += step) {
      const x = padL + i * cellW;
      ctx.fillRect(x, canvas.height - 10, 1, 8);
      ctx.save();
      ctx.translate(x + 2, canvas.height - 2);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(baseDates[i] || '', 0, 0);
      ctx.restore();
    }

    ui.timelineContainer.appendChild(wrap);
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

// --- Handlers ----------------------------------------------------------------

async function handleLoad() {
  try {
    const f = ui.fileInput.files?.[0];
    if (!f) { alert('Choose a CSV file first.'); return; }
    ui.fileName.textContent = f.name;
    progress(5, 'Preparing dataset...');
    dataset?.X_train?.dispose();
    dataset?.y_train?.dispose();
    dataset?.X_test?.dispose();
    dataset?.y_test?.dispose();

    dataset = await prepareDatasetFromFile(f, {
      seqLen: DEFAULT_SEQ_LEN,
      horizons: DEFAULT_HORIZONS,
      testSplit: 0.2
    });
    progress(20, 'Dataset ready');
    showShapes(dataset.meta, dataset.symbols);
    log('Dataset prepared successfully.');
    enableState('loaded');
  } catch (e) {
    console.error(e);
    alert(`Data load error: ${e.message}`);
    log(`Error: ${e.message}`);
    enableState('init');
    progress(0, '');
  }
}

async function handleTrain() {
  if (!dataset) { alert('Load data first.'); return; }
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

    log('Training started...');
    progress(25, 'Training...');

    await model.fit(
      dataset.X_train, dataset.y_train,
      { epochs, batchSize: batch, validationSplit: 0.1, shuffle: false },
      (epoch, logs) => {
        const pct = 25 + Math.round(((epoch + 1) / epochs) * 60); // -> 85%
        progress(pct, `Epoch ${epoch + 1}/${epochs} • loss ${logs.loss?.toFixed(4)} • acc ${(logs.binaryAccuracy * 100).toFixed(1)}%`);
        log(`Epoch ${epoch + 1}/${epochs} — loss=${logs.loss?.toFixed(4)} valLoss=${logs.val_loss?.toFixed(4)} acc=${(logs.binaryAccuracy * 100).toFixed(2)}%`);
      }
    );

    log('Training complete.');
    progress(88, 'Training complete');
    enableState('trained');
  } catch (e) {
    console.error(e);
    alert(`Training error: ${e.message}`);
    log(`Error: ${e.message}`);
    progress(0, '');
  }
}

async function handlePredict() {
  if (!dataset || !model) { alert('Train a model first.'); return; }
  try {
    log('Predicting on test set...');
    progress(92, 'Predicting...');
    const predT = model.predict(dataset.X_test);
    const predArr = await predT.array();
    const yTrueArr = await dataset.y_test.array();
    predT.dispose();

    const results = computePerStockMetrics(predArr, yTrueArr, dataset.symbols, dataset.horizons, 0.5);
    drawAccuracyBarChart(results);
    drawTimelines(results, dataset.horizons, dataset.baseDatesTest);
    renderConfusionMatrices(results);

    log('Prediction completed. Results displayed below.');
    progress(100, 'Done');
  } catch (e) {
    console.error(e);
    alert(`Prediction error: ${e.message}`);
    log(`Error: ${e.message}`);
    progress(0, '');
  }
}

async function handleSave() {
  if (!model) { alert('No trained model to save.'); return; }
  await model.save('tfjs_gru_stock_demo');
  log('Weights downloaded.');
}

function init() {
  enableState('init');
  ui.loadBtn.addEventListener('click', handleLoad);
  ui.trainBtn.addEventListener('click', handleTrain);
  ui.predictBtn.addEventListener('click', handlePredict);
  ui.saveBtn.addEventListener('click', handleSave);
}
init();
