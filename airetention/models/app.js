let rawData = [];
let featureMatrix = [];
let labels = [];
let predictions = [];
let model = null;

const fileInput = document.getElementById("file-input");
const dataStatus = document.getElementById("data-status");
const modelStatus = document.getElementById("model-status");
const thrSlider = document.getElementById("threshold-slider");
const thrValue = document.getElementById("threshold-value");
const metricsDiv = document.getElementById("metrics");
const tableEl = document.getElementById("ranking-table");

fileInput.addEventListener("change", handleFile);
document.getElementById("btn-train").addEventListener("click", trainAndPredict);
thrSlider.addEventListener("input", () => {
  thrValue.textContent = parseFloat(thrSlider.value).toFixed(2);
  if (predictions.length) computeMetrics();
});

/**
 * Handle CSV upload: parse, grab numeric features, and target_class
 */
async function handleFile(evt) {
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

  // Check that target_class exists
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
}

/**
 * Train shallow NN in browser and run predictions
 */
async function trainAndPredict() {
  if (!featureMatrix.length || !labels.length) {
    alert("Upload a CSV with 'visitorid' and 'target_class' first.");
    return;
  }

  const nSamples = featureMatrix.length;
  const nFeatures = featureMatrix[0].length;

  modelStatus.textContent = "Training model in browser…";

  // Convert to tensors
  const X = tf.tensor2d(featureMatrix);          // [N, D]
  const y = tf.tensor2d(labels.map(v => [v]));   // [N, 1]

  // Standardize features: (x - mean) / std
  const { mean, variance } = tf.moments(X, 0);
  const std = tf.sqrt(variance).add(1e-6);
  const Xnorm = X.sub(mean).div(std);

  // Build model: Dense(16, relu) -> Dense(1, sigmoid)
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [nFeatures] }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  // Train
  await model.fit(Xnorm, y, {
    epochs: 10,
    batchSize: 256,
    validationSplit: 0.2,
    shuffle: true,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        modelStatus.textContent =
          `Training… epoch ${epoch + 1}/10 – loss: ${logs.loss.toFixed(4)}, val_acc: ${logs.val_accuracy.toFixed(4)}`;
      },
    },
  });

  modelStatus.textContent = "Training finished. Running predictions…";

  // Predict on all rows
  const preds = model.predict(Xnorm);
  const probs = await preds.data();
  predictions = Array.from(probs);

  // Clean up big tensors
  X.dispose();
  y.dispose();
  Xnorm.dispose();
  preds.dispose();
  mean.dispose();
  std.dispose();

  modelStatus.textContent =
    `Model trained on ${nSamples} rows with ${nFeatures} features. Predictions ready.`;

  computeMetrics();
  renderRankingTable();
}

/**
 * Compute confusion matrix + precision/recall/F1 vs. ground truth
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
