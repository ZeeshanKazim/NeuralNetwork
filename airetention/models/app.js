let rawData = [];
let featureMatrix = [];
let model = null;
let predictions = [];

const fileInput = document.getElementById("file-input");
const dataStatus = document.getElementById("data-status");
const modelStatus = document.getElementById("model-status");
const thrSlider = document.getElementById("threshold-slider");
const thrValue = document.getElementById("threshold-value");
const metricsDiv = document.getElementById("metrics");
const tableEl = document.getElementById("ranking-table");

fileInput.addEventListener("change", handleFile);
document.getElementById("btn-load-model").addEventListener("click", loadModel);
document.getElementById("btn-predict").addEventListener("click", predict);
thrSlider.addEventListener("input", () => {
  thrValue.textContent = parseFloat(thrSlider.value).toFixed(2);
  if (predictions.length) computeMetrics();
});

/**
 * Handle CSV upload
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

  const header = rows[0];
  const dataRows = rows.slice(1);

  rawData = dataRows.map(cols => {
    const obj = {};
    header.forEach((h, i) => {
      obj[h.trim()] = cols[i] !== undefined ? cols[i].trim() : "";
    });
    return obj;
  });

  if (!rawData.length) {
    dataStatus.textContent = "No rows found in CSV.";
    return;
  }

  // Auto-detect numeric columns (ignore visitorid + target_class)
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

  dataStatus.textContent =
    `Loaded ${rawData.length} rows, using ${numericCols.length} numeric features.`;
}

/**
 * Load TF.js model from models/tfjs_main_model/model.json
 */
async function loadModel() {
  try {
    modelStatus.textContent = "Loading model…";
    model = await tf.loadLayersModel("models/tfjs_main_model/model.json");
    modelStatus.textContent = "Model loaded and ready.";
  } catch (err) {
    console.error(err);
    modelStatus.textContent = "Error loading model. Check console.";
  }
}

/**
 * Run predictions on uploaded data
 */
async function predict() {
  if (!model) {
    alert("Load the model first.");
    return;
  }
  if (!featureMatrix.length) {
    alert("Upload a CSV file first.");
    return;
  }

  const X = tf.tensor2d(featureMatrix);
  const preds = model.predict(X);
  const probs = await preds.data();
  predictions = Array.from(probs);
  X.dispose();
  preds.dispose();

  computeMetrics();
  renderRankingTable();
}

/**
 * Compute confusion matrix + precision/recall/F1 if labels present.
 * If target_class is missing, only show summary of probabilities.
 */
function computeMetrics() {
  const thr = parseFloat(thrSlider.value);
  thrValue.textContent = thr.toFixed(2);

  if (!rawData.length || !predictions.length) {
    metricsDiv.innerHTML = "<p class=\"muted\">No data or predictions.</p>";
    return;
  }

  const hasLabels = rawData.some(r => r["target_class"] !== undefined && r["target_class"] !== "");
  const n = predictions.length;
  const avgProb =
    predictions.reduce((s, p) => s + p, 0) / (predictions.length || 1);

  if (!hasLabels) {
    metricsDiv.innerHTML = `
      <p>Total rows scored: <strong>${n}</strong></p>
      <p>Average churn probability: <strong>${avgProb.toFixed(3)}</strong></p>
      <p class="muted">No <code>target_class</code> column found, so metrics vs. ground truth are not shown.</p>
    `;
    return;
  }

  const yTrue = rawData.map(r => {
    const v = parseInt(r["target_class"]);
    return isNaN(v) ? 0 : v;
  });

  let tp = 0, fp = 0, tn = 0, fn = 0;
  predictions.forEach((p, i) => {
    const y = yTrue[i];
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
    <p>TP: <strong>${tp}</strong>, FP: <strong>${fp}</strong>, TN: <strong>${tn}</strong>, FN: <strong>${fn}</strong></p>
    <p>Precision: <strong>${precision.toFixed(3)}</strong></p>
    <p>Recall: <strong>${recall.toFixed(3)}</strong></p>
    <p>F1-score: <strong>${f1.toFixed(3)}</strong></p>
  `;
}

/**
 * Show top-risk customers in a table
 */
function renderRankingTable() {
  if (!rawData.length || !predictions.length) {
    tableEl.innerHTML = "";
    return;
  }

  const rows = rawData.map((r, i) => ({
    visitorid: r["visitorid"] || `row_${i}`,
    label: r["target_class"] !== undefined && r["target_class"] !== "" ? parseInt(r["target_class"]) : null,
    prob: predictions[i]
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
    yCell.textContent = r.label === null || isNaN(r.label) ? "-" : r.label;

    const pCell = document.createElement("td");
    pCell.textContent = r.prob.toFixed(3);

    // Color-code probability
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
