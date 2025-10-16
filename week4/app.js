// app.js
// Orchestrates UI, data loading, model training, evaluation, and visualization

import { prepareDatasetFromFile, DEFAULT_SEQ_LEN, DEFAULT_HORIZONS } from './data-loader.js';
import { GRUClassifier } from './gru.js';

const ui = {
  fileInput: document.getElementById('csvFile'),
  loadBtn: document.getElementById('btnLoad'),
  trainBtn: document.getElementById('btnTrain'),
  saveBtn: document.getElementById('btnSave'),
  epochsInput: document.getElementById('epochs'),
  batchInput: document.getElementById('batch'),
  log: document.getElementById('log'),
  shapes: document.getElementById('shapes'),
  accCanvas: document.getElementById('accChart'),
  timelineContainer: document.getElementById('timelineContainer'),
  confusionContainer: document.getElementById('confusionContainer'),
};

let dataset = null;
let model = null;
let accChart = null;

function log(msg) {
  const time = new Date().toLocaleTimeString();
  ui.log.textContent += `[${time}] ${msg}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function showShapes(meta, symbols) {
  ui.shapes.textContent =
    `SeqLen=${meta.seqLen}, FeatureDim=${meta.featureDim}, Samples(Total/Train/Test)=${meta.totalSamples}/${meta.numTrain}/${meta.numTest}, Symbols=${symbols.length}`;
}

function enableControls(loaded) {
  ui.trainBtn.disabled = !loaded;
  ui.saveBtn.disabled = true;
}

function computePerStockMetrics(predProbs, yTrue, symbols, horizons, threshold = 0.5) {
  // predProbs: [numTest, outputDim] (array)
  // yTrue: same shape (array)
  const S = symbols.length;
  const H = horizons.length;
  const numTest = yTrue.length;

  const perStock = symbols.map(() => ({
    correct: 0, total: 0,
    timeline: Array.from({ length: H }, () => []), // each horizon has a sequence of booleans per test sample
    confusion: [0, 0, 0, 0], // [TN, FP, FN, TP]
  }));

  for (let i = 0; i < numTest; i++) {
    const gt = yTrue[i];
    const pp = predProbs[i];
    for (let s = 0; s < S; s++) {
      for (let h = 0; h < H; h++) {
        const idx = s * H + h;
        const y = gt[idx] >= 0.5 ? 1 : 0;
        const p = pp[idx] >= threshold ? 1 : 0;
        const correct = (y === p);
        perStock[s].correct += correct ? 1 : 0;
        perStock[s].total += 1;
        perStock[s].timeline[h].push(correct);

        // Confusion matrix counts
        if (y === 1 && p === 1) perStock[s].confusion[3]++;       // TP
        else if (y === 0 && p === 0) perStock[s].confusion[0]++;  // TN
        else if (y === 0 && p === 1) perStock[s].confusion[1]++;  // FP
        else if (y === 1 && p === 0) perStock[s].confusion[2]++;  // FN
      }
    }
  }

  const results = perStock.map((st, idx) => ({
    symbol: symbols[idx],
    accuracy: st.total ? st.correct / st.total : 0,
    timeline: st.timeline,
    confusion: st.confusion
  }));

  return results;
}

function drawAccuracyBarChart(results) {
  // Sort by accuracy descending
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
      datasets: [{
        label: 'Accuracy (%)',
        data
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: { min: 0, max: 100, ticks: { callback: v => v + '%' } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.x.toFixed(2)}%`
          }
        }
      }
    }
  });
}

function drawTimelines(results, horizons, baseDates) {
  // Clear container
  ui.timelineContainer.innerHTML = '';
  const H = horizons.length;

  results.forEach(r => {
    const wrapper = document.createElement('div');
    wrapper.className = 'timeline-block';

    const title = document.createElement('div');
    title.className = 'timeline-title';
    title.textContent = `${r.symbol} — prediction correctness timelines (rows: +1d, +2d, +3d)`;
    wrapper.appendChild(title);

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(600, r.timeline[0].length * 6);
    canvas.height = H * 16 + 24;
    wrapper.appendChild(canvas);

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = `<span class="greenBox"></span> correct &nbsp;&nbsp; <span class="redBox"></span> wrong`;
    wrapper.appendChild(legend);

    const ctx = canvas.getContext('2d');
    ctx.font = '12px sans-serif';

    const cellW = 6;
    const cellH = 12;
    const padLeft = 70;
    const padTop = 6;

    for (let h = 0; h < H; h++) {
      ctx.fillStyle = '#444';
      ctx.fillText(`+${h + 1}d`, 5, padTop + h * (cellH + 6) + cellH - 2);

      const seq = r.timeline[h];
      for (let i = 0; i < seq.length; i++) {
        ctx.fillStyle = seq[i] ? '#2ecc71' : '#e74c3c';
        ctx.fillRect(padLeft + i * cellW, padTop + h * (cellH + 6), cellW - 1, cellH);
      }
    }

    // Optional: draw some date ticks across the bottom (every ~10th)
    ctx.fillStyle = '#222';
    const N = r.timeline[0].length;
    const tickEvery = Math.max(1, Math.floor(N / 12));
    for (let i = 0; i < N; i += tickEvery) {
      const x = padLeft + i * cellW;
      ctx.fillRect(x, canvas.height - 10, 1, 8);
      ctx.save();
      ctx.translate(x + 2, canvas.height - 2);
      ctx.rotate(-Math.PI / 4);
      const dateLabel = baseDates[i] || '';
      ctx.fillText(dateLabel, 0, 0);
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
      </table>
    `;
    grid.appendChild(card);
  });

  ui.confusionContainer.appendChild(grid);
}

async function handleLoad() {
  try {
    const file = ui.fileInput.files?.[0];
    if (!file) {
      alert('Please choose a CSV file first.');
      return;
    }
    log('Loading & preparing dataset...');
    dataset?.X_train?.dispose();
    dataset?.y_train?.dispose();
    dataset?.X_test?.dispose();
    dataset?.y_test?.dispose();

    dataset = await prepareDatasetFromFile(file, {
      seqLen: DEFAULT_SEQ_LEN,
      horizons: DEFAULT_HORIZONS,
      testSplit: 0.2
    });

    showShapes(dataset.meta, dataset.symbols);
    log('Dataset ready.');
    enableControls(true);
  } catch (err) {
    console.error(err);
    alert(`Error loading data: ${err.message}`);
    log(`Error: ${err.message}`);
    enableControls(false);
  }
}

async function handleTrain() {
  if (!dataset) {
    alert('Load data first.');
    return;
  }
  try {
    const epochs = Math.max(1, parseInt(ui.epochsInput.value || '25', 10));
    const batch = Math.max(1, parseInt(ui.batchInput.value || '32', 10));

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
    await model.fit(
      dataset.X_train, dataset.y_train,
      { epochs, batchSize: batch, validationSplit: 0.1, shuffle: false },
      (epoch, logs) => {
        log(`Epoch ${epoch + 1}/${epochs} - loss=${logs.loss?.toFixed(4)} valLoss=${logs.val_loss?.toFixed(4)} acc=${(logs.binaryAccuracy * 100).toFixed(2)}%`);
      }
    );
    log('Training complete.');

    ui.saveBtn.disabled = false;

    // Evaluate & visualize
    log('Evaluating on test set...');
    const predTensor = model.predict(dataset.X_test);

    const predArray = await predTensor.array();
    const yTrueArray = await dataset.y_test.array();
    predTensor.dispose();

    const perStock = computePerStockMetrics(predArray, yTrueArray, dataset.symbols, dataset.horizons, 0.5);
    drawAccuracyBarChart(perStock);
    drawTimelines(perStock, dataset.horizons, dataset.baseDatesTest);
    renderConfusionMatrices(perStock);
    log('Evaluation and visualization updated.');
  } catch (err) {
    console.error(err);
    alert(`Training/Evaluation error: ${err.message}`);
    log(`Error: ${err.message}`);
  }
}

async function handleSave() {
  if (!model) return;
  await model.save('tfjs_gru_stock_demo');
  log('Model weights downloaded.');
}

function init() {
  enableControls(false);
  ui.loadBtn.addEventListener('click', handleLoad);
  ui.trainBtn.addEventListener('click', handleTrain);
  ui.saveBtn.addEventListener('click', handleSave);
}

init();
