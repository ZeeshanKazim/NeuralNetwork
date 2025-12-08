let rawData = [];
let featureMatrix = [];
let labels = [];
let predictions = [];
let model = null;

const MAX_TRAIN_ROWS = 8000;   // cap training size for speed
const EPOCHS = 5;

const fileInput = document.getElementById("file-input");
const dataStatus = document.getElementById("data-status");
const modelStatus = document.getElementById("model-status");
const thrSlider = document.getElementById("threshold-slider");
const thrValue = document.getElementById("threshold-value");
const metricsDiv = document.getElementById("metrics");
const tableEl = document.getElementById("ranking-table");

let trainingStartTime = null;  // will store performance.now() at start

console.log("app.js loaded");

if (fileInput) {
  fileInput.addEventListener("change", handleFile);
}
const trainBtn = document.getElementById("btn-train");
if (trainBtn) {
  trainBtn.addEventListener("click", () => {
    modelStatus.textContent = "Train button clicked…";
    trainAndPredict();
  });
}
if (thrSlider) {
  thrSlider.addEventListener("input", () => {
    thrValue.textContent = parseFloat(thrSlider.value).toFixed(2);
    if (predictions.length) computeMetrics();
  });
}

/**
 * Handle CSV upload: parse, grab numeric features, and target_class
 */
async function handleFile(evt) {
  try {
    const file = evt.target.files[0];
    if (!file) return;

    const text = await file.text();
    const rows = text.trim().split("\n").map(r => r.split(","));

    if (rows.length < 2) {
      dataStatus.textContent = "File is empty.";
      return;
    }

    const header = rows[0].map(h => h.trim());
    const dataRows = rows.slice(1);

    rawData = dataRows.map(cols => {
      const obj = {};
      header.forEach((h, i) => {
        obj[h] = cols[i] !== undefined ? cols[i].trim() : "";
      });
      return obj;
    });

    if (!rawData.length) {
      dataStatus.textContent = "No rows found in CSV.";
      return;
    }

    if (!header.includes("target_class")) {
      dataStatus.textContent = "CSV must contain a 'target_class' column.";
      return;
    }

    // Numeric feature columns (exclude IDs + target)
    const sample = rawData[0];
    const numericCols = Object.keys(sample).filter(k => {
      if (k === "target_class" || k === "visitorid") return false;
      const v = sample[k];
      return v !== "" && !isNaN(parseFloat(v));
    });

    featureMatrix = rawData.map(row =>
      numericCols.map(c => {
        const v = row[c];
        const num = parseFloat(v);
        return isNaN(num) ? 0 : num;
      })
    );

    labels = rawData.map(row => {
      const v = parseInt(row["target_class"]);
      return isNaN(v) ? 0 : v;
    });

    dataStatus.textContent =
      `Loaded ${rawData.length} rows, using ${numericCols.length} numeric features.`;
    modelStatus.textContent = "Model not trained yet.";
    predictions = [];
    metricsDiv.innerHTML =
      '<p class="muted">Upload data, train the model, and run predictions to see metrics.</p>';
    tableEl.innerHTML = "";
  } catch (err) {
    console.error("Error in handleFile:", err);
    dataStatus.textContent = "Error reading file: " + err;
  }
}

/**
 * Train shallow NN in browser on a subset for speed, then predict on all rows
 */
async function trainAndPredict() {
  try {
    if (!featureMatrix.length || !labels.length) {
      alert("Upload a CSV with 'visitorid' and 'target_class' first.");
      modelStatus.textContent = "No data available to train.";
      return;
    }

    const nSamples = featureMatrix.length;
    const nFeatures = featureMatrix[0].length;

    trainingStartTime = performance.now();  // start timer
    modelStatus.textContent = "Preparing training data…";

    // Select a random subset for training
    const indices = tf.util
      .createShuffledIndices(nSamples)
      .slice(0, Math.min(MAX_TRAIN_ROWS, nSamples));

    const featSubset = indices.map(i => featureMatrix[i]);
    const labelSubset = indices.map(i => labels[i]);

    // Allow UI to update before heavy work
    await tf.nextFrame();

    // Convert subset to tensors
    const Xtrain = tf.tensor2d(featSubset);               // [N_sub, D]
    const ytrain = tf.tensor2d(labelSubset.map(v => [v])); // [N_sub, 1]

    // Normalize features: (x - mean) / std  (computed on subset)
    const moments = tf.moments(Xtrain, 0);
    const mean = moments.mean;
    const variance = moments.variance;
    const std = tf.sqrt(variance).add(1e-6);
    const XtrainNorm = Xtrain.sub(mean).div(std);

    modelStatus.textContent =
      `Training on ${indices.length} rows × ${nFeatures} features…`;

    // Build model: Dense(16, relu) -> Dense(1, sigmoid)
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [nFeatures] }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    // Train (few epochs)
    await model.fit(XtrainNorm, ytrain, {
      epochs: EPOCHS,
      batchSize: 256,
      validationSplit: 0.2,
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const elapsed = ((performance.now() - trainingStartTime) / 1000).toFixed(1);
          modelStatus.textContent =
            `Epoch ${epoch + 1}/${EPOCHS} – loss: ${logs.loss.toFixed(4)}, ` +
            `val_acc: ${logs.val_accuracy.toFixed(4)} (elapsed: ${elapsed}s)`;
        },
      },
    });

    // Predict on ALL rows (normalize with same mean/std from subset)
    modelStatus.textContent = "Training finished. Scoring all customers…";
    const Xall = tf.tensor2d(featureMatrix);     // [N_all, D]
    const XallNorm = Xall.sub(mean).div(std);
    const preds = model.predict(XallNorm);
    const probs = await preds.data();
    predictions = Array.from(probs);

    // Clean up tensors
    Xtrain.dispose();
    ytrain.dispose();
    XtrainNorm.dispose();
    Xall.dispose();
    XallNorm.dispose();
    preds.dispose();
    mean.dispose();
    variance.dispose();
    std.dispose();

    const totalElapsed = ((performance.now() - trainingStartTime) / 1000).toFixed(1);
    modelStatus.textContent =
      `Model trained on ${indices.length} rows; scored all ${nSamples} customers ` +
      `in ${totalElapsed}s.`;

    computeMetrics();
    renderRankingTable();
  } catch (err) {
    console.error("Error in trainAndPredict:", err);
    const totalElapsed = trainingStartTime
      ? ` (elapsed: ${((performance.now() - trainingStartTime) / 1000).toFixed(1)}s)`
      : "";
    modelStatus.textContent = "Error during training: " + err + totalElapsed;
  }
}

/**
 * Compute confusion matrix + precision/recall/F1
 */
function computeMetrics() {
  const thr = parseFloat(thrSlider.value);
  thrValue.textContent = thr.toFixed(2);

  if (!predictions.length || !labels.length) {
    metricsDiv.innerHTML = "<p class=\"muted\">No predictions yet.</p>";
    return;
  }

  const n = predictions.length;
  const avgProb =
    predictions.reduce((s, p) => s + p, 0) / (predictions.length || 1);

  let tp = 0, fp = 0, tn = 0, fn = 0;
  predictions.forEach((p, i) => {
    const y = labels[i];
    const pred = p >= thr ? 1 : 0;
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else if (pred === 0 && y === 1) fn++;
  });

  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  metricsDiv.innerHTML = `
    <p>Total rows: <strong>${n}</strong></p>
    <p>Average churn probability: <strong>${avgProb.toFixed(3)}</strong></p>
    <p>TP: <strong>${tp}</strong>, FP: <strong>${fp}</strong>,
       TN: <strong>${tn}</strong>, FN: <strong>${fn}</strong></p>
    <p>Precision: <strong>${precision.toFixed(3)}</strong></p>
    <p>Recall: <strong>${recall.toFixed(3)}</strong></p>
    <p>F1-score: <strong>${f1.toFixed(3)}</strong></p>
  `;
}

/**
 * Show top-risk customers sorted by predicted probability
 */
function renderRankingTable() {
  if (!rawData.length || !predictions.length) {
    tableEl.innerHTML = "";
    return;
  }

  const rows = rawData.map((r, i) => ({
    visitorid: r["visitorid"] || `row_${i}`,
    label: labels[i],
    prob: predictions[i],
  }));

  rows.sort((a, b) => b.prob - a.prob);

  tableEl.innerHTML = "";

  const header = document.createElement("tr");
  ["Rank", "visitorid", "true_label", "churn_prob"].forEach(h => {
    const th = document.createElement("th");
    th.textContent = h;
    header.appendChild(th);
  });
  tableEl.appendChild(header);

  rows.slice(0, 50).forEach((r, idx) => {
    const tr = document.createElement("tr");

    const rankCell = document.createElement("td");
    rankCell.textContent = idx + 1;

    const idCell = document.createElement("td");
    idCell.textContent = r.visitorid;

    const yCell = document.createElement("td");
    yCell.textContent = r.label;

    const pCell = document.createElement("td");
    pCell.textContent = r.prob.toFixed(3);

    if (r.prob >= 0.8) pCell.classList.add("prob-high");
    else if (r.prob >= 0.5) pCell.classList.add("prob-med");
    else pCell.classList.add("prob-low");

    tr.appendChild(rankCell);
    tr.appendChild(idCell);
    tr.appendChild(yCell);
    tr.appendChild(pCell);
    tableEl.appendChild(tr);
  });
}
