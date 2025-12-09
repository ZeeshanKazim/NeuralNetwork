/*
ASSUMPTIONS:
1) train_web.csv contains 'CustomerID', 'Churn' and all numeric_cols from config/preprocessing_config.json.
2) scoring_web.csv contains 'CustomerID' and the same numeric_cols (label column optional).
3) We start from the built-in CSVs; uploads are optional for the data team, not required for end users.
*/

const CONFIG_URL = "config/preprocessing_config.json";
const MODEL_URL = "model.json";
const SAMPLE_TRAIN_URL = "sample_data/train_web.csv";
const SAMPLE_SCORE_URL = "sample_data/scoring_web.csv";

const RISK_THRESHOLDS = {
  high: 0.7,
  medium: 0.4, // [medium, high) = medium, < medium = low
};

let preprocessingConfig = null;
let model = null;

let rawTrainRows = [];
let rawScoreRows = [];
let trainSet = null;
let scoreSet = null;

let lastEval = null; // { yTrue, yScore }
let lastMetrics = null; // { threshold, auc, accuracy, precision, recall, f1, cm, ... }
let lastScorePreds = null; // [{ id, prob, pred, row }]
let datasetSummary = null;
let rocChart = null;
let lastSegments = null; // { high, medium, low, highRevenue }

let baseStatusText = "Initializing…";

const BUTTON_IDS = [
  "btn-load-data",
  "btn-preprocess",
  "btn-reset",
  "btn-train",
  "btn-evaluate",
  "btn-score",
  "btn-export-model",
  "btn-export-metrics-json",
  "btn-export-metrics-csv",
  "btn-export-scores",
];

/* ---------- DOM & status ---------- */

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  baseStatusText = text || baseStatusText;
  const el = $("app-status");
  if (el) el.textContent = baseStatusText;
}

function setBusy(isBusy, label) {
  BUTTON_IDS.forEach((id) => {
    const btn = $(id);
    if (btn) btn.disabled = isBusy;
  });
  if (isBusy && label) {
    $("app-status").textContent = label;
  } else {
    $("app-status").textContent = baseStatusText;
  }
}

function log(message, level = "info") {
  const consoleEl = $("log-console");
  if (!consoleEl) return;

  const line = document.createElement("div");
  line.className = "log-line";

  const time = document.createElement("span");
  time.className = "log-time";
  const now = new Date();
  time.textContent = now.toLocaleTimeString("en-US", { hour12: false });

  const text = document.createElement("span");
  text.className = `log-text ${level}`;
  text.textContent = message;

  line.appendChild(time);
  line.appendChild(text);
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

/* ---------- Generic helpers ---------- */

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadCsv(filename, rows, header) {
  const lines = [];
  lines.push(header.join(","));
  for (const row of rows) {
    const values = header.map((key) => {
      const v = row[key];
      if (v === undefined || v === null) return "";
      const s = String(v);
      return s.includes(",") ? `"${s.replace(/"/g, '""')}"` : s;
    });
    lines.push(values.join(","));
  }
  downloadBlob(filename, lines.join("\n"), "text/csv;charset=utf-8;");
}

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
  if (result.errors && result.errors.length) {
    log(`CSV parse warning: ${result.errors[0].message}`, "error");
  }
  return result.data;
}

function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors && results.errors.length) {
          reject(results.errors[0]);
        } else {
          resolve(results.data);
        }
      },
      error: (err) => reject(err),
    });
  });
}

/* ---------- Dataset summary ---------- */

function updateDatasetSummary() {
  const rowsEl = $("summary-rows");
  const colsEl = $("summary-cols");
  const missEl = $("summary-missing");

  if (!rawTrainRows || rawTrainRows.length === 0) {
    datasetSummary = null;
    if (rowsEl) rowsEl.textContent = "–";
    if (colsEl) colsEl.textContent = "–";
    if (missEl) missEl.textContent = "–";
    return;
  }

  const rows = rawTrainRows.length;
  const cols = Object.keys(rawTrainRows[0]).length;
  const colNames = Object.keys(rawTrainRows[0]);
  let missing = 0;

  rawTrainRows.forEach((r) => {
    colNames.forEach((c) => {
      const v = r[c];
      if (
        v === null ||
        v === undefined ||
        v === "" ||
        (typeof v === "number" && Number.isNaN(v))
      ) {
        missing += 1;
      }
    });
  });

  const totalCells = rows * cols;
  const missingPct = totalCells ? (missing / totalCells) * 100 : 0;

  datasetSummary = { rows, cols, missing, missingPct };

  if (rowsEl) rowsEl.textContent = rows.toString();
  if (colsEl) colsEl.textContent = cols.toString();
  if (missEl) missEl.textContent = `${missing} (${missingPct.toFixed(1)}%)`;
}

/* ---------- Preprocessing ---------- */

function transformRow(row) {
  const cfg = preprocessingConfig;
  if (!cfg) return null;

  const cols = cfg.numeric_cols;
  const means = cfg.numeric_means || {};
  const stds = cfg.numeric_stds || {};

  const features = [];
  for (let i = 0; i < cols.length; i++) {
    const col = cols[i];
    if (!(col in row)) {
      return null; // missing column
    }
    let v = Number(row[col]);
    if (Number.isNaN(v)) v = means[col];

    const mean = means[col] != null ? means[col] : 0;
    const std = stds[col] != null && stds[col] !== 0 ? stds[col] : 1;
    const scaled = (v - mean) / std;
    if (Number.isNaN(scaled)) return null;
    features.push(scaled);
  }
  return features;
}

function buildDataset(rows, isTrain) {
  const cfg = preprocessingConfig;
  if (!cfg) return null;

  const ids = [];
  const X = [];
  const y = [];

  for (const row of rows) {
    if (!row) continue;

    if (isTrain) {
      const label = row[cfg.target];
      if (
        label === undefined ||
        label === null ||
        label === "" ||
        Number.isNaN(Number(label))
      ) {
        continue;
      }
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

/* ---------- Model ---------- */

function createDefaultModel() {
  if (!preprocessingConfig || !preprocessingConfig.numeric_cols) {
    throw new Error("Preprocessing config not loaded; cannot build model.");
  }

  const inputDim = preprocessingConfig.numeric_cols.length;

  const m = tf.sequential();
  m.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
      inputShape: [inputDim],
    })
  );
  m.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    })
  );

  m.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return m;
}

async function loadConfigAndModel() {
  try {
    preprocessingConfig = await fetchJSON(CONFIG_URL);
    log("Preprocessing config loaded.", "success");
  } catch (err) {
    log(`Failed to load preprocessing config: ${err.message}`, "error");
    setStatus("Config failed to load – app limited.");
    return;
  }

  // Try to load the pre-trained model.json first.
  try {
    model = await tf.loadLayersModel(MODEL_URL);
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
    setStatus("Pretrained model loaded and ready.");
    log("TF.js model loaded from model.json.", "success");
  } catch (err) {
    log(
      `Failed to load TF.js model from model.json (${err.message}). Falling back to a fresh model.`,
      "error"
    );
    try {
      model = createDefaultModel();
      setStatus("Fresh model created – ready to train.");
      log(
        `New shallow NN model created with input dim ${preprocessingConfig.numeric_cols.length}.`,
        "success"
      );
    } catch (innerErr) {
      model = null;
      setStatus("Could not create model – training disabled.");
      log(`Also failed to create fresh model: ${innerErr.message}`, "error");
    }
  }

  const slider = $("threshold-slider");
  if (slider) {
    $("threshold-display").textContent = Number(slider.value || 0.5).toFixed(2);
  }
}

/* ---------- Metrics & ROC ---------- */

function computeConfusion(yTrue, yScore, threshold) {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const y = Number(yTrue[i]);
    const pred = yScore[i] >= threshold ? 1 : 0;

    if (y === 1 && pred === 1) tp++;
    else if (y === 0 && pred === 1) fp++;
    else if (y === 0 && pred === 0) tn++;
    else if (y === 1 && pred === 0) fn++;
  }

  return { tp, fp, tn, fn };
}

function confusionToMetrics(cm) {
  const { tp, fp, tn, fn } = cm;
  const total = tp + fp + tn + fn;
  const accuracy = total === 0 ? 0 : (tp + tn) / total;
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 =
    precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { accuracy, precision, recall, f1 };
}

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
  const canvas = $("rocChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

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

function updateMetricsUI(auc, metrics, cm) {
  $("metric-auc").textContent = auc != null ? auc.toFixed(3) : "–";
  $("metric-accuracy").textContent =
    metrics && metrics.accuracy != null ? metrics.accuracy.toFixed(3) : "–";
  $("metric-precision").textContent =
    metrics && metrics.precision != null ? metrics.precision.toFixed(3) : "–";
  $("metric-recall").textContent =
    metrics && metrics.recall != null ? metrics.recall.toFixed(3) : "–";
  $("metric-f1").textContent =
    metrics && metrics.f1 != null ? metrics.f1.toFixed(3) : "–";

  $("cm-tn").textContent = cm ? cm.tn : "–";
  $("cm-fp").textContent = cm ? cm.fp : "–";
  $("cm-fn").textContent = cm ? cm.fn : "–";
  $("cm-tp").textContent = cm ? cm.tp : "–";
}

/* ---------- Overview & segments ---------- */

function computeSegmentsFromScores(items) {
  if (!items || !items.length) return null;

  let high = 0;
  let medium = 0;
  let low = 0;
  let highRevenue = 0;

  items.forEach((item) => {
    const p = item.prob;
    const row = item.row || {};
    if (p >= RISK_THRESHOLDS.high) {
      high++;
      const revenue = Number(row["CashbackAmount"]);
      if (!Number.isNaN(revenue)) {
        highRevenue += revenue;
      }
    } else if (p >= RISK_THRESHOLDS.medium) {
      medium++;
    } else {
      low++;
    }
  });

  return { high, medium, low, highRevenue };
}

function updateOverviewUI() {
  const totalCustomers =
    (rawScoreRows && rawScoreRows.length) ||
    (rawTrainRows && rawTrainRows.length) ||
    0;

  const churnLabel = (() => {
    if (!rawTrainRows || !rawTrainRows.length) return "–";
    let count = 0;
    let valid = 0;
    rawTrainRows.forEach((r) => {
      if ("Churn" in r) {
        const v = Number(r.Churn);
        if (!Number.isNaN(v)) {
          valid++;
          count += v;
        }
      }
    });
    if (!valid) return "–";
    const rate = (count / valid) * 100;
    return `${rate.toFixed(1)}%`;
  })();

  const seg = lastSegments;
  let highText = "–";
  let revenueText = "–";

  if (seg && totalCustomers > 0) {
    const pct = (seg.high / totalCustomers) * 100;
    highText = `${seg.high} (${pct.toFixed(1)}%)`;
    if (seg.highRevenue > 0) {
      const approx = Math.round(seg.highRevenue);
      revenueText = approx.toLocaleString();
    }
  }

  const totalEl = $("overview-total");
  const churnEl = $("overview-churn");
  const highEl = $("overview-high");
  const revEl = $("overview-revenue");

  if (totalEl) totalEl.textContent = totalCustomers ? totalCustomers.toString() : "–";
  if (churnEl) churnEl.textContent = churnLabel;
  if (highEl) highEl.textContent = highText;
  if (revEl) revEl.textContent = revenueText;
}

/* ---------- Scored table & filters ---------- */

function renderScoreTable(items) {
  const tbody = $("score-table").querySelector("tbody");
  tbody.innerHTML = "";
  if (!items || !items.length) return;

  const maxRows = 200;
  const toShow = items.slice(0, maxRows);

  toShow.forEach((item) => {
    const row = item.row || {};
    const tr = document.createElement("tr");

    const rec =
      row["DaySinceLastOrder"] != null
        ? row["DaySinceLastOrder"]
        : row["Recency"] != null
        ? row["Recency"]
        : "";
    const freq =
      row["OrderCount"] != null
        ? row["OrderCount"]
        : row["Frequency"] != null
        ? row["Frequency"]
        : "";
    const mon =
      row["OrderAmountHikeFromlastYear"] != null
        ? row["OrderAmountHikeFromlastYear"]
        : row["CashbackAmount"] != null
        ? row["CashbackAmount"]
        : row["Monetary"] != null
        ? row["Monetary"]
        : "";

    tr.innerHTML = `
      <td>${item.id != null ? item.id : ""}</td>
      <td>${item.prob.toFixed(3)}</td>
      <td>${item.pred}</td>
      <td>${rec}</td>
      <td>${freq}</td>
      <td>${mon}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAndRenderTable() {
  if (!lastScorePreds) {
    renderScoreTable([]);
    return;
  }

  const riskFilter = ($("filter-risk") || {}).value || "all";
  const search = ($("search-id") || {}).value || "";
  const trimmedSearch = search.trim();

  const filtered = lastScorePreds.filter((item) => {
    const p = item.prob;
    let keep = true;

    if (riskFilter === "high") {
      keep = p >= RISK_THRESHOLDS.high;
    } else if (riskFilter === "medium") {
      keep = p >= RISK_THRESHOLDS.medium && p < RISK_THRESHOLDS.high;
    } else if (riskFilter === "low") {
      keep = p < RISK_THRESHOLDS.medium;
    }

    if (!keep) return false;

    if (trimmedSearch) {
      const idStr = item.id != null ? String(item.id) : "";
      if (!idStr.includes(trimmedSearch)) return false;
    }

    return true;
  });

  renderScoreTable(filtered);
}

/* ---------- Action handlers ---------- */

async function handleLoadSampleData(auto = false) {
  if (!auto) setBusy(true, "Loading sample data…");
  try {
    const [trainText, scoreText] = await Promise.all([
      fetchText(SAMPLE_TRAIN_URL),
      fetchText(SAMPLE_SCORE_URL),
    ]);

    rawTrainRows = parseCsvText(trainText);
    rawScoreRows = parseCsvText(scoreText);

    const trainSummary = $("train-summary");
    const scoreSummary = $("score-summary");
    if (trainSummary) {
      trainSummary.textContent = `Using sample train_web.csv · ${rawTrainRows.length} rows`;
    }
    if (scoreSummary) {
      scoreSummary.textContent = `Using sample scoring_web.csv · ${rawScoreRows.length} rows`;
    }

    updateDatasetSummary();
    log("Sample train & scoring data loaded.", "success");
  } catch (err) {
    log(`Failed to load sample data: ${err.message}`, "error");
  } finally {
    if (!auto) setBusy(false);
  }
}

async function handleTrainFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;

  setBusy(true, "Parsing train CSV…");
  try {
    rawTrainRows = await parseCsvFile(file);
    const trainSummary = $("train-summary");
    if (trainSummary) {
      trainSummary.textContent = `${file.name} · ${rawTrainRows.length} rows`;
    }
    updateDatasetSummary();
    log(`Train file "${file.name}" loaded.`, "success");
  } catch (err) {
    log(`Failed to parse train CSV: ${err.message}`, "error");
  } finally {
    setBusy(false);
  }
}

async function handleScoreFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;

  setBusy(true, "Parsing scoring CSV…");
  try {
    rawScoreRows = await parseCsvFile(file);
    const scoreSummary = $("score-summary");
    if (scoreSummary) {
      scoreSummary.textContent = `${file.name} · ${rawScoreRows.length} rows`;
    }
    log(`Scoring file "${file.name}" loaded.`, "success");
  } catch (err) {
    log(`Failed to parse scoring CSV: ${err.message}`, "error");
  } finally {
    setBusy(false);
  }
}

function handlePreprocess(auto = false) {
  if (!preprocessingConfig) {
    log("Preprocessing config not loaded yet.", "error");
    return;
  }
  if (!rawTrainRows || rawTrainRows.length === 0) {
    log("No train data loaded. Using sample data or upload a train CSV first.", "error");
    return;
  }

  if (!auto) setBusy(true, "Building datasets…");
  try {
    const cols = preprocessingConfig.numeric_cols;
    const firstTrainRow = rawTrainRows[0];
    const missingCols = cols.filter((c) => !(c in firstTrainRow));
    if (missingCols.length) {
      log(
        `Train data is missing required numeric columns: ${missingCols.join(", ")}`,
        "error"
      );
      return;
    }

    if (trainSet && trainSet.X_tensor) {
      trainSet.X_tensor.dispose();
      if (trainSet.y_tensor) trainSet.y_tensor.dispose();
    }
    trainSet = buildDataset(rawTrainRows, true);
    if (!trainSet) {
      log(
        "Could not build training dataset – check numeric columns and label column.",
        "error"
      );
      return;
    }

    if (rawScoreRows && rawScoreRows.length) {
      const firstScoreRow = rawScoreRows[0];
      const missingScoreCols = cols.filter((c) => !(c in firstScoreRow));
      if (missingScoreCols.length) {
        log(
          `Scoring data is missing required numeric columns: ${missingScoreCols.join(
            ", "
          )}`,
          "error"
        );
      } else {
        if (scoreSet && scoreSet.X_tensor) scoreSet.X_tensor.dispose();
        scoreSet = buildDataset(rawScoreRows, false);
      }
    }

    log(
      `Preprocessing complete. Train rows used: ${trainSet.X.length}${
        scoreSet ? ` · Scoring rows: ${scoreSet.X.length}` : ""
      }.`,
      "success"
    );
  } finally {
    if (!auto) setBusy(false);
  }
}

async function handleTrain(auto = false) {
  if (!model) {
    log("Model is not loaded; training is unavailable.", "error");
    return;
  }
  if (!trainSet || !trainSet.X_tensor || !trainSet.y_tensor) {
    log("Preprocess the training data before training.", "error");
    return;
  }

  if (!auto) {
    setBusy(true, "Training model in browser…");
    log("Starting training (15 epochs, validation split 0.2)…", "info");
  }

  try {
    await model.fit(trainSet.X_tensor, trainSet.y_tensor, {
      epochs: 15,
      batchSize: 256,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const valLoss = logs.val_loss ?? 0;
          log(
            `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(
              4
            )} · val_loss=${valLoss.toFixed(4)} · acc=${acc.toFixed(4)}`,
            "info"
          );
        },
      },
    });
    log("Training complete.", "success");
  } catch (err) {
    log(`Training failed: ${err.message}`, "error");
  } finally {
    if (!auto) setBusy(false);
  }
}

async function handleEvaluate(auto = false) {
  if (!model) {
    log("Model is not loaded.", "error");
    return;
  }
  if (!trainSet || !trainSet.X_tensor || !trainSet.y || trainSet.y.length === 0) {
    log("Preprocess the training data before evaluation.", "error");
    return;
  }

  if (!auto) setBusy(true, "Running evaluation…");
  try {
    const predsTensor = model.predict(trainSet.X_tensor);
    const predsArr = Array.from(await predsTensor.data());
    predsTensor.dispose();

    lastEval = { yTrue: trainSet.y, yScore: predsArr };

    const { fprs, tprs, auc } = computeRocAuc(lastEval.yTrue, lastEval.yScore);
    renderRocCurve(fprs, tprs);

    const threshold = parseFloat($("threshold-slider").value) || 0.5;
    const cm = computeConfusion(lastEval.yTrue, lastEval.yScore, threshold);
    const metrics = confusionToMetrics(cm);

    lastMetrics = {
      threshold,
      auc,
      ...metrics,
      cm,
      timestamp: new Date().toISOString(),
      datasetSummary,
    };

    updateMetricsUI(auc, metrics, cm);
    log(
      `Evaluation complete. AUC=${auc.toFixed(3)}, Accuracy=${metrics.accuracy.toFixed(
        3
      )}.`,
      "success"
    );
  } catch (err) {
    log(`Evaluation failed: ${err.message}`, "error");
  } finally {
    if (!auto) setBusy(false);
  }
}

function handleThresholdChange(e) {
  const value = parseFloat(e.target.value) || 0.5;
  $("threshold-display").textContent = value.toFixed(2);

  if (!lastEval) return;

  const cm = computeConfusion(lastEval.yTrue, lastEval.yScore, value);
  const metrics = confusionToMetrics(cm);
  const { auc } = computeRocAuc(lastEval.yTrue, lastEval.yScore);

  lastMetrics = {
    threshold: value,
    auc,
    ...metrics,
    cm,
    timestamp: new Date().toISOString(),
    datasetSummary,
  };

  updateMetricsUI(auc, metrics, cm);
}

async function handleScore(auto = false) {
  if (!model) {
    log("Model is not loaded.", "error");
    return;
  }
  if (!scoreSet || !scoreSet.X_tensor || !scoreSet.ids) {
    log("No scoring dataset. Load and preprocess scoring data first.", "error");
    return;
  }

  if (!auto) setBusy(true, "Scoring customers…");
  try {
    const predsTensor = model.predict(scoreSet.X_tensor);
    const predsArr = Array.from(await predsTensor.data());
    predsTensor.dispose();

    const threshold =
      lastMetrics && lastMetrics.threshold != null
        ? lastMetrics.threshold
        : parseFloat($("threshold-slider").value) || 0.5;

    const items = scoreSet.ids.map((id, idx) => ({
      id,
      prob: predsArr[idx],
      pred: predsArr[idx] >= threshold ? 1 : 0,
      row: rawScoreRows[idx],
    }));

    items.sort((a, b) => b.prob - a.prob);
    lastScorePreds = items;

    lastSegments = computeSegmentsFromScores(items);
    updateOverviewUI();
    filterAndRenderTable();

    log(
      `Scoring complete. ${items.length} customers scored (threshold=${threshold.toFixed(
        2
      )}).`,
      "success"
    );
  } catch (err) {
    log(`Scoring failed: ${err.message}`, "error");
  } finally {
    if (!auto) setBusy(false);
  }
}

async function handleExportModel() {
  if (!model) {
    log("Model is not loaded; nothing to export.", "error");
    return;
  }
  try {
    await model.save("downloads://ai_retention_radar_tfjs_model");
    log("Model exported as TF.js files.", "success");
  } catch (err) {
    log(`Model export failed: ${err.message}`, "error");
  }
}

function handleExportMetricsJson() {
  if (!lastMetrics) {
    log("No metrics to export. Run evaluation first.", "error");
    return;
  }
  const payload = {
    model: "AI Retention Radar – shallow NN",
    generatedAt: new Date().toISOString(),
    metrics: lastMetrics,
  };
  downloadBlob(
    "metrics.json",
    JSON.stringify(payload, null, 2),
    "application/json;charset=utf-8;"
  );
  log("Metrics exported as metrics.json.", "success");
}

function handleExportMetricsCsv() {
  if (!lastMetrics) {
    log("No metrics to export. Run evaluation first.", "error");
    return;
  }

  const { threshold, auc, accuracy, precision, recall, f1, cm } = lastMetrics;
  const header = [
    "threshold",
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
  ];
  const rows = [
    {
      threshold,
      auc,
      accuracy,
      precision,
      recall,
      f1,
      tn: cm.tn,
      fp: cm.fp,
      fn: cm.fn,
      tp: cm.tp,
    },
  ];
  downloadCsv("metrics.csv", rows, header);
  log("Metrics exported as metrics.csv.", "success");
}

function handleExportScores() {
  if (!lastScorePreds) {
    log("No scored customers to export. Run prediction first.", "error");
    return;
  }

  const rows = lastScorePreds.map((item) => ({
    customer_id: item.id,
    churn_probability: item.prob,
    churn_prediction: item.pred,
  }));
  const header = ["customer_id", "churn_probability", "churn_prediction"];
  downloadCsv("scores.csv", rows, header);
  log("Scores exported as scores.csv.", "success");
}

function handleReset() {
  if (trainSet && trainSet.X_tensor) {
    trainSet.X_tensor.dispose();
    if (trainSet.y_tensor) trainSet.y_tensor.dispose();
  }
  if (scoreSet && scoreSet.X_tensor) {
    scoreSet.X_tensor.dispose();
  }

  rawTrainRows = [];
  rawScoreRows = [];
  trainSet = null;
  scoreSet = null;
  lastEval = null;
  lastMetrics = null;
  lastScorePreds = null;
  datasetSummary = null;
  lastSegments = null;

  const trainSummary = $("train-summary");
  const scoreSummary = $("score-summary");
  if (trainSummary) {
    trainSummary.textContent =
      "Using built-in train_web.csv (5630 rows) by default.";
  }
  if (scoreSummary) {
    scoreSummary.textContent =
      "Using built-in scoring_web.csv (5630 rows) by default.";
  }

  updateDatasetSummary();
  updateMetricsUI(null, null, null);
  const tbody = $("score-table").querySelector("tbody");
  tbody.innerHTML = "";

  const slider = $("threshold-slider");
  if (slider) {
    slider.value = 0.5;
    $("threshold-display").textContent = "0.50";
  }

  if (rocChart) {
    rocChart.destroy();
    rocChart = null;
  }

  const consoleEl = $("log-console");
  if (consoleEl) consoleEl.innerHTML = "";

  updateOverviewUI();

  setStatus("Dashboard reset – reload to start again.");
  log("State reset. You can reload data and start over.", "info");
}

/* ---------- Auto pipeline ---------- */

async function runAutoPipeline() {
  try {
    setBusy(true, "Loading data & building dashboard…");
    await handleLoadSampleData(true);
    handlePreprocess(true);
    if (model) {
      await handleTrain(true);
      await handleEvaluate(true);
    }
    await handleScore(true);
    setStatus("Dashboard ready – data loaded.");
  } catch (err) {
    log(`Auto-init pipeline failed: ${err.message}`, "error");
    setStatus("Dashboard error – see log.");
  } finally {
    setBusy(false);
  }
}

/* ---------- Init ---------- */

document.addEventListener("DOMContentLoaded", async () => {
  const slider = $("threshold-slider");
  if (slider) {
    slider.value = 0.5;
    $("threshold-display").textContent = "0.50";
  }

  log("App initializing…", "info");
  setStatus("Loading model & config…");

  await loadConfigAndModel();
  await runAutoPipeline();

  // Wire manual/advanced controls
  const loadBtn = $("btn-load-data");
  if (loadBtn) loadBtn.addEventListener("click", () => handleLoadSampleData());

  const preBtn = $("btn-preprocess");
  if (preBtn) preBtn.addEventListener("click", () => handlePreprocess());

  const resetBtn = $("btn-reset");
  if (resetBtn) resetBtn.addEventListener("click", handleReset);

  const trainBtn = $("btn-train");
  if (trainBtn) trainBtn.addEventListener("click", () => handleTrain());

  const evalBtn = $("btn-evaluate");
  if (evalBtn) evalBtn.addEventListener("click", () => handleEvaluate());

  const scoreBtn = $("btn-score");
  if (scoreBtn) scoreBtn.addEventListener("click", () => handleScore());

  const exportModelBtn = $("btn-export-model");
  if (exportModelBtn)
    exportModelBtn.addEventListener("click", handleExportModel);

  const exportMetricsJsonBtn = $("btn-export-metrics-json");
  if (exportMetricsJsonBtn)
    exportMetricsJsonBtn.addEventListener("click", handleExportMetricsJson);

  const exportMetricsCsvBtn = $("btn-export-metrics-csv");
  if (exportMetricsCsvBtn)
    exportMetricsCsvBtn.addEventListener("click", handleExportMetricsCsv);

  const exportScoresBtn = $("btn-export-scores");
  if (exportScoresBtn)
    exportScoresBtn.addEventListener("click", handleExportScores);

  const fileTrain = $("file-train");
  if (fileTrain) fileTrain.addEventListener("change", handleTrainFileChange);

  const fileScore = $("file-score");
  if (fileScore) fileScore.addEventListener("change", handleScoreFileChange);

  const riskFilter = $("filter-risk");
  if (riskFilter) riskFilter.addEventListener("change", filterAndRenderTable);

  const searchId = $("search-id");
  if (searchId) searchId.addEventListener("input", filterAndRenderTable);

  if (slider) slider.addEventListener("input", handleThresholdChange);

  log("Dashboard initialized. Data has been loaded and scored automatically.", "info");
});
