/* =========================================================================
   Titanic Binary Classifier — TensorFlow.js
   FINAL APP.JS (CSV parsing fixed; responsive training; full pipeline)
   ======================================================================= */

/* ---------------------------- Global State ---------------------------- */
const S = {
  rawTrain: [],
  rawTest: [],
  pre: null,
  xsTrain: null, ysTrain: null,
  xsVal: null,   ysVal: null,
  model: null,
  valProbs: null,
  testProbs: null,
  thresh: 0.5
};

// Fixed schema we use for modeling (ignore Name, Ticket, Cabin, etc.)
const SCHEMA = {
  id: 'PassengerId',
  target: 'Survived',
  features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] // baseline features
};

const $ = id => document.getElementById(id);

/* ------------------------------ Backend ------------------------------ */
(async () => {
  try { await tf.setBackend('cpu'); } catch {}
  await tf.ready();
})();

/* ------------------------------ Helpers ------------------------------ */
function previewTable(rows, limit = 8) {
  const host = $('previewTable');
  if (!rows.length) { host.innerHTML = ''; return; }
  const cols = Object.keys(rows[0]);
  const thead = '<thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead>';
  const tbody = '<tbody>' + rows.slice(0, limit).map(r =>
    '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>'
  ).join('') + '</tbody>';
  host.innerHTML = `<table>${thead}${tbody}</table>`;
}

function normalizeRow(row) {
  const o = {};
  for (const [k, v] of Object.entries(row)) {
    if (v === '' || v === undefined) o[k] = null;
    else if (typeof v === 'string') o[k] = v.trim();
    else o[k] = v;
  }
  return o;
}

function missingPct(rows) {
  if (!rows.length) return 100;
  const cols = Object.keys(rows[0]); let miss = 0, total = rows.length * cols.length;
  for (const r of rows) for (const c of cols) {
    const v = r[c]; if (v == null || v === '') miss++;
  }
  return +(100 * miss / total).toFixed(1);
}

/* ----------------------------- CSV Loader ---------------------------- */
/* Force Kaggle defaults: comma delimiter + double-quote.
   This prevents the Name column (with commas) from splitting and
   avoids the "__parsed_extra" column that indicates a malformed row. */
function parseCSVFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: 'greedy',
      delimiter: ',',         // <— FIXED
      quoteChar: '"',         // <— FIXED
      complete: r => resolve(r.data.map(normalizeRow)),
      error: reject
    });
  });
}

/* --------------------------- Preprocessing --------------------------- */
const median = a => {
  const b = a.filter(v => v != null && !Number.isNaN(+v)).map(Number).sort((x, y) => x - y);
  if (!b.length) return null;
  const m = Math.floor(b.length / 2);
  return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
};
const mode = a => {
  const m = new Map(); let best = null, cnt = 0;
  for (const v of a) { if (v == null || v === '') continue;
    const c = (m.get(v) || 0) + 1; m.set(v, c); if (c > cnt) { cnt = c; best = v; } }
  return best;
};
const mean = a => { const b = a.filter(Number.isFinite); return b.length ? b.reduce((s, x) => s + x, 0) / b.length : 0; };
const sd   = a => { const b = a.filter(Number.isFinite); if (b.length < 2) return 0; const m = mean(b); return Math.sqrt(b.reduce((s, x) => s + (x - m) ** 2, 0) / (b.length - 1)); };
const oneHot = (v, cats) => { const r = new Array(cats.length).fill(0); const i = cats.indexOf(v); if (i >= 0) r[i] = 1; return r; };
const fin = (x, d=0) => Number.isFinite(x) ? x : d;

function buildPreprocessor(trainRows) {
  // add engineered toggles
  const useFamily = $('featFamily')?.checked ?? true;
  const useAlone  = $('featAlone')?.checked ?? true;

  // imputation anchors
  const ageMed = Number(median(trainRows.map(r => r.Age))) || 30;
  const embMode = mode(trainRows.map(r => r.Embarked)) ?? 'S';

  // categories for one-hot
  const sexCats = ['female', 'male'];
  const pclassCats = [1, 2, 3];
  const embCats = ['C', 'Q', 'S', 'UNKNOWN'];

  // standardization stats
  const ageVals = trainRows.map(r => fin((r.Age != null && !isNaN(+r.Age)) ? +r.Age : ageMed, ageMed));
  const fareVals= trainRows.map(r => fin((r.Fare!= null && !isNaN(+r.Fare)) ? +r.Fare : 0, 0));
  const muA = mean(ageVals), sdA = sd(ageVals), muF = mean(fareVals), sdF = sd(fareVals);

  // final mapping from a raw row to feature vector
  function mapRow(r) {
    // sanitize extra column (if present) so it never leaks into features
    if ('__parsed_extra' in r) delete r.__parsed_extra;

    const age = (r.Age != null && !isNaN(+r.Age)) ? +r.Age : ageMed;
    const emb = (r.Embarked == null || r.Embarked === '') ? 'UNKNOWN' : r.Embarked;
    const fare = (r.Fare != null && !isNaN(+r.Fare)) ? +r.Fare : 0;
    const fam = (+r.SibSp || 0) + (+r.Parch || 0) + 1;
    const alone = fam === 1 ? 1 : 0;

    // z-scores
    const ageZ  = sdA ? (age  - muA) / sdA : 0;
    const fareZ = sdF ? (fare - muF) / sdF : 0;

    // only use the schema features (ignore Name, Ticket, Cabin, etc.)
    let f = [
      ageZ, fareZ,
      ...oneHot(r.Sex, sexCats),
      ...oneHot(+r.Pclass, pclassCats),
      ...oneHot(emb, embCats)
    ];
    if (useFamily) f.push(fam);
    if (useAlone)  f.push(alone);

    return f.map(x => fin(+x, 0));
  }

  const featLen = mapRow(trainRows[0] || {}).length;

  return {
    ageMed, embMode, muA, sdA, muF, sdF,
    sexCats, pclassCats, embCats,
    useFamily, useAlone,
    featLen,
    mapRow
  };
}

function tensorize(rows, pre) {
  const X = [], Y = [];
  for (const r of rows) {
    const f = pre.mapRow(r);
    if (!f.every(Number.isFinite)) continue;
    X.push(f);
    if (SCHEMA.target in r) Y.push(+r[SCHEMA.target]);
  }
  if (!X.length) throw new Error('No valid rows after preprocessing.');
  const xs = tf.tensor2d(X, [X.length, pre.featLen], 'float32');
  const ys = Y.length ? tf.tensor2d(Y, [Y.length, 1], 'float32') : null; // shape (N,1)
  return { xs, ys };
}

function stratifiedSplit(rows, ratio = 0.2) {
  const z = rows.filter(r => +r[SCHEMA.target] === 0);
  const o = rows.filter(r => +r[SCHEMA.target] === 1);
  const cut = g => {
    const a = g.slice(); tf.util.shuffle(a);
    const n = Math.max(1, Math.floor(a.length * ratio));
    return { val: a.slice(0, n), train: a.slice(n) };
  };
  const a = cut(z), b = cut(o);
  const train = a.train.concat(b.train), val = a.val.concat(b.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  return { train, val };
}

/* ------------------------------- Model ------------------------------- */
function buildModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

function modelSummaryText(m) {
  const lines = [];
  m.summary(undefined, undefined, s => lines.push(s));
  return lines.join('\n');
}

/* --------------------------- Metrics & Plots -------------------------- */
function rocPoints(yTrue, yProb, steps = 200) {
  const T = []; for (let i = 0; i <= steps; i++) T.push(i / steps);
  const pts = T.map(th => {
    let TP=0,FP=0,TN=0,FN=0;
    for (let i = 0; i < yTrue.length; i++) {
      const y = yTrue[i], p = yProb[i] >= th ? 1 : 0;
      if (y===1 && p===1) TP++; else if (y===0 && p===1) FP++;
      else if (y===0 && p===0) TN++; else FN++;
    }
    const TPR = TP / (TP + FN || 1), FPR = FP / (FP + TN || 1);
    return { x: FPR, y: TPR };
  });
  const s = pts.slice().sort((a, b) => a.x - b.x);
  let auc = 0; for (let i = 1; i < s.length; i++) { const a = s[i-1], b = s[i]; auc += (b.x - a.x) * (a.y + b.y) / 2; }
  return { points: s, auc };
}

function drawROC(canvas, pts) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0f1628'; ctx.fillRect(0, 0, W, H);
  // grid
  ctx.strokeStyle = '#233350'; ctx.lineWidth = 1;
  for (let i=0;i<=5;i++){const x=40+i*(W-60)/5;ctx.beginPath();ctx.moveTo(x,H-30);ctx.lineTo(x,20);ctx.stroke();}
  for (let i=0;i<=5;i){const y=20+i*(H-50)/5;ctx.beginPath();ctx.moveTo(40,y);ctx.lineTo(W-20,y);ctx.stroke();}
  // diagonal (random)
  ctx.strokeStyle = '#3a4760'; ctx.beginPath();
  ctx.moveTo(40, H-30); ctx.lineTo(W-20, 20); ctx.stroke();
  // roc
  ctx.strokeStyle = '#8aa3ff'; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((p,i)=>{ const x = 40 + p.x*(W-60), y = H-30 - p.y*(H-50); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
  ctx.stroke();
}

/* --------------------------- Early Stopping -------------------------- */
let stopFlag = false;
function earlyStopWithRestore(patience = 5, monitor = 'val_loss') {
  let best = Infinity, wait = 0, snap = null;
  return new tf.CustomCallback({
    onBatchEnd: async () => { await new Promise(r => setTimeout(r, 0)); },
    onEpochEnd: async (_e, logs) => {
      await tf.nextFrame();
      const cur = logs?.[monitor];
      if (cur != null) {
        if (cur < best - 1e-12) {
          best = cur; wait = 0;
          if (snap) snap.forEach(t => t.dispose());
          snap = S.model.getWeights().map(w => w.clone());
        } else if (++wait >= patience) {
          if (snap) { S.model.setWeights(snap); snap = null; }
          S.model.stopTraining = true;
        }
      }
      if (stopFlag) S.model.stopTraining = true;
    }
  });
}

/* ------------------------------- UI: Load ---------------------------- */
async function onLoadFiles() {
  try {
    const fTrain = $('trainFile').files[0];
    if (!fTrain) { alert('Please choose train.csv'); return; }

    S.rawTrain = await parseCSVFile(fTrain);
    const fTest = $('testFile').files[0];
    S.rawTest  = fTest ? await parseCSVFile(fTest) : [];

    $('kTrain').textContent = S.rawTrain.length;
    $('kTest').textContent  = S.rawTest.length || '—';
    $('kMiss').textContent  = missingPct(S.rawTrain) + '%';

    previewTable(S.rawTrain);
  } catch (e) {
    console.error(e);
    alert('Failed to load CSV: ' + (e?.message || e));
  }
}

/* --------------------------- UI: Preprocess -------------------------- */
function onPreprocess() {
  try {
    if (!S.rawTrain.length) { alert('Load train.csv first'); return; }

    // Build and apply preprocessor
    S.pre = buildPreprocessor(S.rawTrain);
    const { train, val } = stratifiedSplit(S.rawTrain, 0.2);
    const tTr = tensorize(train, S.pre);
    const tVa = tensorize(val,   S.pre);
    S.xsTrain = tTr.xs; S.ysTrain = tTr.ys;
    S.xsVal   = tVa.xs; S.ysVal   = tVa.ys;

    $('preInfo').textContent = [
      `features: ${S.pre.featLen}`,
      `Train: ${S.xsTrain.shape} | Val: ${S.xsVal.shape}`,
      `Impute Age median=${S.pre.ageMed} | Embarked mode=${S.pre.embMode}`,
      `One-hot: Sex, Pclass, Embarked | Engineered: FamilySize=${S.pre.useFamily}, IsAlone=${S.pre.useAlone}`
    ].join('\n');

  } catch (e) {
    console.error(e);
    alert('Preprocessing failed: ' + (e?.message || e));
  }
}

/* ---------------------------- UI: Model ------------------------------ */
function onBuild() {
  try {
    if (!S.xsTrain) { alert('Run Preprocessing first'); return; }
    S.model = buildModel(S.xsTrain.shape[1]);
    $('modelSummary').textContent = 'Model built. Click "Show Summary" to view layers.';
  } catch (e) {
    console.error(e);
    alert('Build failed: ' + (e?.message || e));
  }
}

function onSummary() {
  try {
    if (!S.model) { alert('Build the model first'); return; }
    $('modelSummary').textContent = modelSummaryText(S.model);
  } catch (e) {
    console.error(e);
    alert('Summary failed: ' + (e?.message || e));
  }
}

/* --------------------------- UI: Training ---------------------------- */
async function onTrain() {
  try {
    if (!S.model) { alert('Build the model first'); return; }
    if (!S.xsTrain) { alert('Run Preprocessing first'); return; }

    stopFlag = false;
    $('trainLog').textContent = '';

    const cb = earlyStopWithRestore(5, 'val_loss');

    await S.model.fit(S.xsTrain, S.ysTrain, {
      epochs: 40, batchSize: 16,
      validationData: [S.xsVal, S.ysVal],
      callbacks: [{
        onEpochEnd: async (ep, logs) => {
          $('trainLog').textContent += `epoch ${ep+1}: `
            + `loss=${logs.loss.toFixed(4)} `
            + `val_loss=${logs.val_loss.toFixed(4)} `
            + `acc=${(logs.acc ?? logs.accuracy ?? 0).toFixed(4)}\n`;
          await cb.onEpochEnd?.(ep, logs);
        },
        onBatchEnd: async (b, logs) => { await cb.onBatchEnd?.(b, logs); }
      }]
    });

    // validation probabilities for metrics/ROC
    const val = tf.tidy(() => S.model.predict(S.xsVal).dataSync());
    S.valProbs = Float32Array.from(val);

    const yTrue = Array.from(S.ysVal.dataSync()).map(v => +v);
    const { points, auc } = rocPoints(yTrue, S.valProbs, 200);
    drawROC($('rocCanvas'), points);
    $('aucText').textContent = `AUC = ${auc.toFixed(4)}`;

    updateThreshold(S.thresh);

  } catch (e) {
    console.error(e);
    alert('Training failed: ' + (e?.message || e));
  }
}

function onStop() {
  stopFlag = true;
  alert('Early stop requested (will stop after current epoch).');
}

/* ----------------------------- Metrics UI ---------------------------- */
function confusion(yTrue, yProb, th) {
  let TP=0,FP=0,TN=0,FN=0;
  for (let i=0;i<yTrue.length;i++){
    const y=yTrue[i], p = yProb[i] >= th ? 1 : 0;
    if (y===1 && p===1) TP++;
    else if (y===0 && p===1) FP++;
    else if (y===0 && p===0) TN++;
    else FN++;
  }
  const prec = TP / (TP + FP || 1);
  const rec  = TP / (TP + FN || 1);
  const f1   = (2 * prec * rec) / ((prec + rec) || 1);
  return { TP, FP, TN, FN, prec, rec, f1 };
}

function updateThreshold(th) {
  S.thresh = +th;
  $('thVal').textContent = (+th).toFixed(2);
  if (!S.valProbs) return;
  const yTrue = Array.from(S.ysVal.dataSync()).map(v => +v);
  const st = confusion(yTrue, S.valProbs, +th);
  $('cmTP').textContent = st.TP; $('cmFN').textContent = st.FN;
  $('cmFP').textContent = st.FP; $('cmTN').textContent = st.TN;
  $('prf').textContent = `Precision: ${(st.prec*100).toFixed(2)}%\n`
                       + `Recall: ${(st.rec*100).toFixed(2)}%\n`
                       + `F1: ${st.f1.toFixed(4)}`;
}

/* ------------------------ Predict & Export UI ------------------------ */
async function onPredict() {
  try {
    if (!S.model) { alert('Train the model first'); return; }
    if (!S.rawTest.length) { alert('Load test.csv'); return; }

    const out = tf.tidy(() => {
      const X = S.rawTest.map(S.pre.mapRow);
      const xs = tf.tensor2d(X, [X.length, S.pre.featLen], 'float32');
      const probs = S.model.predict(xs).dataSync();
      xs.dispose(); return probs;
    });

    S.testProbs = Float32Array.from(out);
    $('predInfo').textContent = `Predicted ${S.rawTest.length} rows. You can now download CSVs.`;

  } catch (e) {
    console.error(e);
    alert('Prediction failed: ' + (e?.message || e));
  }
}

function downloadCSV(name, rows) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = v => {
    if (v == null) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => esc(r[c])).join(','))).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob); const a = document.createElement('a');
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}

function onDownloadSubmission() {
  try {
    if (!S.testProbs) { alert('Run Predict first'); return; }
    const out = S.rawTest.map((r, i) => ({
      [SCHEMA.id]: r[SCHEMA.id],
      [SCHEMA.target]: (S.testProbs[i] >= S.thresh ? 1 : 0)
    }));
    downloadCSV('submission.csv', out);
  } catch (e) {
    console.error(e);
    alert('Download failed: ' + (e?.message || e));
  }
}

function onDownloadProbs() {
  try {
    if (!S.testProbs) { alert('Run Predict first'); return; }
    const out = S.rawTest.map((r, i) => ({
      [SCHEMA.id]: r[SCHEMA.id],
      ProbSurvived: S.testProbs[i]
    }));
    downloadCSV('probabilities.csv', out);
  } catch (e) {
    console.error(e);
    alert('Download failed: ' + (e?.message || e));
  }
}

async function onSaveModel() {
  try {
    if (!S.model) { alert('Train the model first'); return; }
    await S.model.save('downloads://titanic-tfjs');
  } catch (e) {
    console.error(e);
    alert('Save failed: ' + (e?.message || e));
  }
}

/* ----------------------------- Wire Events --------------------------- */
window.addEventListener('DOMContentLoaded', () => {
  $('btnLoad').addEventListener('click', onLoadFiles);

  $('btnPre').addEventListener('click', onPreprocess);

  $('btnBuild').addEventListener('click', onBuild);
  $('btnSummary').addEventListener('click', onSummary);

  $('btnTrain').addEventListener('click', onTrain);
  $('btnStop').addEventListener('click', onStop);

  $('thSlider').addEventListener('input', e => updateThreshold(+e.target.value));

  $('btnPredict').addEventListener('click', onPredict);
  $('btnSub').addEventListener('click', onDownloadSubmission);
  $('btnProb').addEventListener('click', onDownloadProbs);
  $('btnSaveModel').addEventListener('click', onSaveModel);
});
