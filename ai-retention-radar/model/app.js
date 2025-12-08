// paths based on your current repo layout (inside ai-retention-radar/model/)
const CONFIG_URL = "config/preprocessing_config.json";
const MODEL_URL = "model.json";
const SAMPLE_TRAIN_URL = "sample_data/train_web.csv";
const SAMPLE_SCORE_URL = "sample_data/scoring_web.csv";

let preprocessingConfig = null;
let model = null;

let rawTrainRows = [];
let rawScoreRows = [];

let trainSet = null;
let scoreSet = null;

let lastEval = null; // { yTrue, yScore }
let lastScorePreds = null;

let rocChart = null;

/* basic helpers */

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.text();
}

function parseCsvText(text) {
  const result = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return result.data;
}

function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data),
      error: (err) => reject(err),
    });
  });
}

function logTrain(msg) {
  const el = document.getElementById("training-log");
  const div = document.createElement("div");
  div.textContent = msg;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

/* preprocessing */

function transformRow(row) {
  const cfg = preprocessingConfig;
  if (!cfg) return null;

  const cols = cfg.numeric_cols;
  const means = cfg.numeric_means || {};
  const stds = cfg.numeric_stds || {};

  const features = [];
  for (const col of cols) {
    let v = Number(row[col]);
    if (Number.isNaN(v)) v = means[col]; // simple impute with mean

    const mean = means[col] ?? 0;
    const std = stds[col] ?? 1 || 1;

    const scaled = (v - mean) / std;
    if (Number.isNaN(scaled)) return null;
    features.push(scaled);
  }

  return features;
}

function buildDataset(rows, isTrain) {
  const cfg = preprocessingConfig;
  const ids = [];
  const X = [];
  const y = [];

  for (const row of rows) {
    if (!row) continue;

    if (isTrain) {
      const label = row[cfg.target];
      if (label === undefined || label === null || label === "") continue;
    }

    const feats = transformRow(row);
    if (!feats) continue;

    ids.push(cfg.id_col ? row[cfg.id_col] : null);
    X.push(feats);
    if (isTrain) y.push(Number(row[cfg.target]));
  }

  if (!X.length) return null;

  const X_tensor = tf.tensor2d(X);
  const y_tensor = isTrain ? tf.tensor2d(y, [y.length, 1]) : null;

  return { ids, X, y, X_tensor, y_tensor };
}

/* metrics */

function computeConfusion(yTrue, yScore, threshold) {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const y = Number(yTrue[i]);
    const p = yScore[i] >= threshold ? 1 : 0;

    if (y === 1 && p === 1) tp++;
    else if (y === 0 && p === 1) fp++;
    else if (y === 0 && p === 0) tn++;
    else if (y === 1 && p === 0) fn++;
  }

  return { tp, fp, tn, fn };
}

function confusionToMetrics({ tp, fp, tn, fn }) {
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 =
    precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1 };
}

// simple ROC + AUC
function computeRocAuc(yTrue, yScore) {
  const pairs = yTrue.map((y, i) => ({ y: Number(y), score: yScore[i] }));
  pairs.sort((a, b) => b.score - a.score);

  const P = pairs.filter((p) => p.y === 1).length;
  const N = pairs.length - P;
  if (P === 0 || N === 0) {
    return { fprs: [0, 1], tprs: [0, 1], auc: 0.5 };
  }

  let tp = 0;
  let fp = 0;
  const tprs = [0];
  const fprs = [0];

  for (const p of pairs) {
    if (p.y === 1) tp++;
    else fp++;
    tprs.push(tp / P);
    fprs.push(fp / N);
  }

  let auc = 0;
  for (let i = 1; i < tprs.length; i++) {
    const x1 = fprs[i - 1];
    const x2 = fprs[i];
    const y1 = tprs[i - 1];
    const y2 = tprs[i];
    auc += (x2 - x1) * ((y1 + y2) / 2);
  }

  return { fprs, tprs, auc };
}

function renderRocCurve(fprs, tprs) {
  const ctx = document.getElementById("rocChart").getContext("2d");

  const data = {
    datasets: [
      {
        label: "ROC curve",
        data: fprs.map((fpr, i) => ({ x: fpr, y: tprs[i] })),
        fill: false,
        tension: 0.25,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: "linear",
        min: 0,
        max: 1,
        title: { display: true, text: "False Positive Rate" },
      },
      y: {
        min: 0,
        max: 1,
        title: { display: true, text: "True Positive Rate" },
      },
    },
    plugins: {
      legend: { display: false },
    },
  };

  if (rocChart) {
    rocChart.data = data;
    rocChart.options = options;
    rocChart.update();
  } else {
    rocChart = new Chart(ctx, { type: "line", data, options });
  }
}

/* UI helpers */

function updateMetricsDisplay(auc, metrics, cm) {
  document.getElementById("auc-value").textContent = auc.toFixed(3);
  document.getElementById("precision-value").textContent =
    metrics.precision.toFixed(3);
  document.getElementById("recall-value").textContent =
    metrics.recall.toFixed(3);
  document.getElementById("f1-value").textContent = metrics.f1.toFixed(3);

  document.getElementById("cm-tp").textContent = cm.tp;
  document.getElementById("cm-tn").textContent = cm.tn;
  document.getElementById("cm-fp").textContent = cm.fp;
  document.getElementById("cm-fn").textContent = cm.fn;
}

function updateThresholdLabel(value) {
  document.getElementById("threshold-value").textContent =
    Number(value).toFixed(2);
}

function renderRankingTable(items) {
  const tbody = document.querySelector("#ranking-table tbody");
  tbody.innerHTML = "";

  for (const item of items) {
    const tr = document.createElement("tr");

    const row = item.row || {};
    const rec = row["DaySinceLastOrder"] ?? row["Recency"] ?? "";
    const freq = row["OrderCount"] ?? row["Frequency"] ?? "";
    const mon =
      row["OrderAmountHikeFromlastYear"] ??
      row["CashbackAmount"] ??
      row["Monetary"] ??
      "";

    tr.innerHTML = `
      <td>${item.id ?? ""}</td>
      <td>${item.prob.toFixed(3)}</td>
      <td>${rec}</td>
      <td>${freq}</td>
      <td>${mon}</td>
    `;
    tbody.appendChild(tr);
  }
}

/* CSV export */

function downloadCsv(filename, rows, header) {
  const lines = [];
  lines.push(header.join(","));

  for (const row of rows) {
    const values = header.map((key) => {
      const v = row[key];
      if (v === null || v === undefined) return "";
      const s = String(v);
      return s.includes(",") ? `"${s.replace(/"/g, '""')}"` : s;
    });
    lines.push(values.join(","));
  }

  const blob = new Blob([lines.join("\n")], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* actions */

async function init() {
  const statusEl = document.getElementById("model-status");

  try {
    preprocessingConfig = await fetchJSON(CONFIG_URL);
    statusEl.textContent = "Preprocessing config loaded ✓";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Config missing – check config/";
  }

  try {
    model = await tf.loadLayersModel(MODEL_URL);
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
    statusEl.textContent = "Model loaded ✓";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Model missing – check model.json";
  }

  const thresholdSlider = document.getElementById("threshold-slider");
  updateThresholdLabel(thresholdSlider.value);
}

async function handleTrainUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  rawTrainRows = await parseCsvFile(file);
  document.getElementById(
    "train-summary"
  ).textContent = `${file.name} · ${rawTrainRows.length} rows`;
}

async function handleScoreUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  rawScoreRows = await parseCsvFile(file);
  document.getElementById(
    "score-summary"
  ).textContent = `${file.name} · ${rawScoreRows.length} rows`;
}

async function handleLoadSampleTrain() {
  try {
    const text = await fetchText(SAMPLE_TRAIN_URL);
    rawTrainRows = parseCsvText(text);
    document.getElementById(
      "train-summary"
    ).textContent = `sample_data/train_web.csv · ${rawTrainRows.length} rows`;
  } catch (err) {
    alert("Could not load sample train_web.csv");
    console.error(err);
  }
}

async function handleLoadSampleScore() {
  try {
    const text = await fetchText(SAMPLE_SCORE_URL);
    rawScoreRows = parseCsvText(text);
    document.getElementById(
      "score-summary"
    ).textContent = `sample_data/scoring_web.csv · ${rawScoreRows.length} rows`;
  } catch (err) {
    alert("Could not load sample scoring_web.csv");
    console.error(err);
  }
}

function handlePreprocess() {
  if (!preprocessingConfig) {
    alert("Preprocessing config not loaded.");
    return;
  }
  if (!rawTrainRows.length) {
    alert("Load a labeled train CSV first (or use the sample).");
    return;
  }

  if (trainSet?.X_tensor) {
    trainSet.X_tensor.dispose();
    trainSet.y_tensor?.dispose();
  }

  trainSet = buildDataset(rawTrainRows, true);
  if (!trainSet) {
    alert("Could not build training dataset. Check numeric columns.");
    return;
  }

  if (rawScoreRows.length) {
    if (scoreSet?.X_tensor) scoreSet.X_tensor.dispose();
    scoreSet = buildDataset(rawScoreRows, false);
  }

  document.getElementById(
    "feature-info"
  ).textContent = `Features: ${preprocessingConfig.numeric_cols.length}`;
  document.getElementById(
    "shape-info"
  ).textContent = `Train rows: ${trainSet.X.length}`;

  alert("Preprocessing complete.");
}

async function handleTrain() {
  if (!model) {
    alert("Model not loaded.");
    return;
  }
  if (!trainSet || !trainSet.X_tensor) {
    alert("Preprocess your train data first.");
    return;
  }

  logTrain("Starting training…");

  const { X_tensor, y_tensor } = trainSet;

  await model.fit(X_tensor, y_tensor, {
    epochs: 15,
    batchSize: 256,
    validationSplit: 0.2,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;
        const valLoss = logs.val_loss ?? 0;
        logTrain(
          `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(
            4
          )}, val_loss=${valLoss.toFixed(4)}, acc=${acc.toFixed(4)}`
        );
      },
    },
  });

  logTrain("Training complete ✓");
}

async function handleMetrics() {
  if (!model) {
    alert("Model not loaded.");
    return;
  }
  if (!trainSet || !trainSet.X_tensor) {
    alert("Preprocess train data first.");
    return;
  }

  const predsTensor = model.predict(trainSet.X_tensor);
  const predsArr = Array.from(await predsTensor.data());
  predsTensor.dispose();

  lastEval = {
    yTrue: trainSet.y,
    yScore: predsArr,
  };

  const slider = document.getElementById("threshold-slider");
  const threshold = parseFloat(slider.value);

  const cm = computeConfusion(lastEval.yTrue, lastEval.yScore, threshold);
  const metrics = confusionToMetrics(cm);
  const { fprs, tprs, auc } = computeRocAuc(lastEval.yTrue, lastEval.yScore);

  renderRocCurve(fprs, tprs);
  updateMetricsDisplay(auc, metrics, cm);
}

function handleThresholdChange(e) {
  const val = parseFloat(e.target.value);
  updateThresholdLabel(val);

  if (!lastEval) return;

  const cm = computeConfusion(lastEval.yTrue, lastEval.yScore, val);
  const metrics = confusionToMetrics(cm);
  const { fprs, tprs, auc } = computeRocAuc(lastEval.yTrue, lastEval.yScore);

  renderRocCurve(fprs, tprs);
  updateMetricsDisplay(auc, metrics, cm);
}

async function handlePredict() {
  if (!model) {
    alert("Model not loaded.");
    return;
  }
  if (!scoreSet || !scoreSet.X_tensor) {
    alert("Load + preprocess a scoring CSV first.");
    return;
  }

  const predsTensor = model.predict(scoreSet.X_tensor);
  const predsArr = Array.from(await predsTensor.data());
  predsTensor.dispose();

  const threshold = parseFloat(
    document.getElementById("threshold-slider").value
  );

  const items = scoreSet.ids.map((id, idx) => ({
    id,
    prob: predsArr[idx],
    label: predsArr[idx] >= threshold ? 1 : 0,
    row: rawScoreRows[idx],
  }));

  items.sort((a, b) => b.prob - a.prob);
  lastScorePreds = items;

  renderRankingTable(items.slice(0, 100));
}

function handleExportProbabilities() {
  if (!lastScorePreds) {
    alert("Run prediction first.");
    return;
  }

  const rows = lastScorePreds.map((item) => ({
    customer_id: item.id,
    churn_probability: item.prob,
  }));

  downloadCsv("probabilities.csv", rows, [
    "customer_id",
    "churn_probability",
  ]);
}

function handleExportSubmission() {
  if (!lastScorePreds) {
    alert("Run prediction first.");
    return;
  }

  const threshold = parseFloat(
    document.getElementById("threshold-slider").value
  );

  const rows = lastScorePreds.map((item) => ({
    customer_id: item.id,
    churn_prediction: item.prob >= threshold ? 1 : 0,
  }));

  downloadCsv("submission.csv", rows, [
    "customer_id",
    "churn_prediction",
  ]);
}

/* wiring */

document.addEventListener("DOMContentLoaded", () => {
  init();

  document
    .getElementById("train-upload")
    .addEventListener("change", handleTrainUpload);
  document
    .getElementById("score-upload")
    .addEventListener("change", handleScoreUpload);

  document
    .getElementById("load-sample-train")
    .addEventListener("click", handleLoadSampleTrain);
  document
    .getElementById("load-sample-score")
    .addEventListener("click", handleLoadSampleScore);

  document
    .getElementById("preprocess-btn")
    .addEventListener("click", handlePreprocess);
  document
    .getElementById("train-btn")
    .addEventListener("click", handleTrain);
  document
    .getElementById("metrics-btn")
    .addEventListener("click", handleMetrics);

  document
    .getElementById("threshold-slider")
    .addEventListener("input", handleThresholdChange);

  document
    .getElementById("predict-btn")
    .addEventListener("click", handlePredict);
  document
    .getElementById("export-proba-btn")
    .addEventListener("click", handleExportProbabilities);
  document
    .getElementById("export-submission-btn")
    .addEventListener("click", handleExportSubmission);
});
