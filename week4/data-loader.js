// data-loader.js
// Improved feature engineering for higher accuracy.
// - Parses CSV (Date, Symbol, Open, Close).
// - Aligns dates across 10 symbols (complete panel only).
// - Builds rich per-stock features (10 features/stock) and 3-step-ahead binary labels.
// - Normalizes per stock & feature using only the training split (min-max).
// - Produces chronological train/test tensors ready for TF.js GRU/CNN models.

export const DEFAULT_SEQ_LEN = 24;                 // longer history usually improves accuracy
export const DEFAULT_HORIZONS = [1, 2, 3];
export const FEATURES_PER_STOCK = 10;              // Open, Close, OCret, Ret1, LogRet1, Mom5, Mom10, RSI14, BB%20, Vol10
export const STOCK_COUNT_EXPECTED = 10;

const INDICATOR_WARMUP = 20;                       // to safely compute 20-day stats

// ---------- CSV parsing ----------
function detectDelimiter(firstLine) {
  const comma = (firstLine.match(/,/g) || []).length;
  const semi  = (firstLine.match(/;/g) || []).length;
  return semi > comma ? ';' : ',';
}

function parseCSV(text) {
  const raw = text.replace(/\uFEFF/g, '');
  const firstNL = raw.indexOf('\n');
  const headerLine = firstNL === -1 ? raw : raw.slice(0, firstNL);
  const delim = detectDelimiter(headerLine);

  const lines = raw.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV has no data rows.');

  const header = lines[0].split(delim).map(h => h.trim());
  const idx = {
    Date:   header.findIndex(h => /^date$/i.test(h)),
    Symbol: header.findIndex(h => /^symbol$/i.test(h)),
    Open:   header.findIndex(h => /^open$/i.test(h)),
    Close:  header.findIndex(h => /^close$/i.test(h)),
  };
  for (const k of Object.keys(idx)) {
    if (idx[k] === -1) throw new Error(`Missing "${k}" column.`);
  }

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delim).map(c => c.trim());
    if (!cols.length) continue;
    const d = new Date(cols[idx.Date]); if (isNaN(+d)) continue;
    const dateIso = d.toISOString().slice(0, 10);
    const sym = cols[idx.Symbol];
    const o = parseFloat(cols[idx.Open]);
    const c = parseFloat(cols[idx.Close]);
    if (!sym || !Number.isFinite(o) || !Number.isFinite(c)) continue;
    rows.push({ date: dateIso, symbol: sym, open: o, close: c });
  }
  if (!rows.length) throw new Error('No valid rows parsed.');
  return rows;
}

// ---------- Panel alignment ----------
function pivotPanel(records) {
  const symbolSet = new Set(records.map(r => r.symbol));
  const symbols = Array.from(symbolSet).sort();

  const byDate = new Map();
  for (const r of records) {
    if (!byDate.has(r.date)) byDate.set(r.date, {});
    byDate.get(r.date)[r.symbol] = r;
  }

  const allDates = Array.from(byDate.keys()).sort();
  // Keep only dates where all symbols exist
  const dates = allDates.filter(d => {
    const row = byDate.get(d);
    return row && symbols.every(s => row[s] != null);
  });

  if (dates.length < INDICATOR_WARMUP + 5) {
    throw new Error('Not enough complete days across all symbols.');
  }

  // Build series per symbol
  const openSeries  = symbols.map(() => []);
  const closeSeries = symbols.map(() => []);
  for (const d of dates) {
    const m = byDate.get(d);
    for (let s = 0; s < symbols.length; s++) {
      const sym = symbols[s];
      openSeries[s].push(m[sym].open);
      closeSeries[s].push(m[sym].close);
    }
  }

  return { symbols, dates, openSeries, closeSeries };
}

// ---------- Math helpers ----------
function mean(arr, i0, i1) {
  let s = 0, n = 0;
  for (let i = i0; i <= i1; i++) { s += arr[i]; n++; }
  return s / Math.max(1, n);
}
function std(arr, i0, i1) {
  const m = mean(arr, i0, i1);
  let s2 = 0, n = 0;
  for (let i = i0; i <= i1; i++) { const d = arr[i] - m; s2 += d*d; n++; }
  return Math.sqrt(s2 / Math.max(1, n));
}
function safeDiv(a, b, fallback = 0) { return b === 0 ? fallback : a / b; }

// Wilder’s RSI(14)
function rsi14(prices) {
  const n = prices.length;
  const out = new Array(n).fill(NaN);
  if (n < 15) return out;

  let gains = 0, losses = 0;
  for (let i = 1; i <= 14; i++) {
    const diff = prices[i] - prices[i - 1];
    if (diff > 0) gains += diff; else losses -= diff;
  }
  let avgG = gains / 14;
  let avgL = losses / 14;
  out[14] = avgL === 0 ? 1 : 1 - 1 / (1 + avgG / avgL); // scale to 0..1

  for (let i = 15; i < n; i++) {
    const diff = prices[i] - prices[i - 1];
    const g = diff > 0 ? diff : 0;
    const l = diff < 0 ? -diff : 0;
    avgG = (avgG * 13 + g) / 14;
    avgL = (avgL * 13 + l) / 14;
    const rs = avgL === 0 ? 1e6 : avgG / avgL;
    const rsi = 1 - 1 / (1 + rs); // 0..1
    out[i] = rsi;
  }
  return out;
}

// ---------- Feature engineering ----------
function buildFeatureMatrix(symbols, openSeries, closeSeries) {
  const nDays = closeSeries[0].length;

  // Precompute per symbol arrays
  const featuresPerSymbol = symbols.map((_, s) => {
    const O = openSeries[s];
    const C = closeSeries[s];

    const ocRet = new Array(nDays).fill(NaN);       // (C/O - 1)
    const ret1  = new Array(nDays).fill(NaN);       // (C_t/C_{t-1} - 1)
    const logR  = new Array(nDays).fill(NaN);       // ln(C_t/C_{t-1})
    const mom5  = new Array(nDays).fill(NaN);       // C/MA5 - 1
    const mom10 = new Array(nDays).fill(NaN);       // C/MA10 - 1
    const rsi   = rsi14(C);                         // 0..1
    const bbp   = new Array(nDays).fill(NaN);       // Bollinger %B (20)
    const vol10 = new Array(nDays).fill(NaN);       // std of logR (10)

    // compute once to reuse
    const logRarr = new Array(nDays).fill(NaN);

    for (let t = 0; t < nDays; t++) {
      ocRet[t] = safeDiv(C[t], O[t], 1) - 1;
      if (t > 0) {
        const ratio = safeDiv(C[t], C[t - 1], 1);
        ret1[t] = ratio - 1;
        logR[t] = Math.log(Math.max(1e-8, ratio));
        logRarr[t] = logR[t];
      }
      if (t >= 4) {
        const ma5 = mean(C, t - 4, t);
        mom5[t] = safeDiv(C[t], ma5, 1) - 1;
      }
      if (t >= 9) {
        const ma10 = mean(C, t - 9, t);
        mom10[t] = safeDiv(C[t], ma10, 1) - 1;
      }
      if (t >= 19) {
        const ma20 = mean(C, t - 19, t);
        const sd20 = std(C, t - 19, t);
        const denom = 2 * (sd20 === 0 ? 1 : sd20);
        bbp[t] = 0.5 + (C[t] - ma20) / denom; // keep in ~[0,1]
      }
      if (t >= 9) {
        vol10[t] = std(logRarr, t - 9, t);
      }
    }
    return { O, C, ocRet, ret1, logR, mom5, mom10, rsi, bbp, vol10 };
  });

  // Assemble rows: [ ... per-symbol 10 features ... ]
  const rows = [];
  for (let t = 0; t < nDays; t++) {
    const row = [];
    for (let s = 0; s < symbols.length; s++) {
      const f = featuresPerSymbol[s];
      row.push(
        f.O[t], f.C[t],
        f.ocRet[t] ?? 0,
        f.ret1[t]  ?? 0,
        f.logR[t]  ?? 0,
        f.mom5[t]  ?? 0,
        f.mom10[t] ?? 0,
        f.rsi[t]   ?? 0.5,
        f.bbp[t]   ?? 0.5,
        f.vol10[t] ?? 0
      );
    }
    rows.push(row);
  }
  return rows; // [days, symbols*10]
}

// ---------- Normalization ----------
function computeMinMax(featureRows, symbols, featuresPerStock, lastTrainDay) {
  const mins = symbols.map(() => new Array(featuresPerStock).fill(Number.POSITIVE_INFINITY));
  const maxs = symbols.map(() => new Array(featuresPerStock).fill(Number.NEGATIVE_INFINITY));

  for (let t = 0; t <= lastTrainDay; t++) {
    const row = featureRows[t];
    for (let s = 0; s < symbols.length; s++) {
      for (let f = 0; f < featuresPerStock; f++) {
        const v = row[s * featuresPerStock + f];
        if (v < mins[s][f]) mins[s][f] = v;
        if (v > maxs[s][f]) maxs[s][f] = v;
      }
    }
  }
  // guard degenerate ranges
  for (let s = 0; s < symbols.length; s++) {
    for (let f = 0; f < featuresPerStock; f++) {
      if (!isFinite(mins[s][f]) || !isFinite(maxs[s][f])) { mins[s][f] = 0; maxs[s][f] = 1; }
      if (maxs[s][f] === mins[s][f]) maxs[s][f] = mins[s][f] + 1;
    }
  }
  return { mins, maxs };
}
function minmax(v, min, max) { const r = max - min; return (v - min) / (r === 0 ? 1 : r); }

// ---------- Public API ----------
export async function prepareDatasetFromFile(file, options = {}) {
  const {
    seqLen = DEFAULT_SEQ_LEN,
    horizons = DEFAULT_HORIZONS,
    testSplit = 0.2,
    expectStocks = STOCK_COUNT_EXPECTED,
  } = options;

  const text = await file.text();
  const records = parseCSV(text);
  const { symbols, dates, openSeries, closeSeries } = pivotPanel(records);

  if (expectStocks && symbols.length !== expectStocks) {
    console.warn(`Expected ${expectStocks} symbols, found ${symbols.length}. Proceeding.`);
  }

  const featureRows = buildFeatureMatrix(symbols, openSeries, closeSeries);
  const featuresPerStock = FEATURES_PER_STOCK;
  const nDays = dates.length;

  const maxH = Math.max(...horizons);
  // Earliest base index must respect both sequence history and indicator warmup
  const earliestBase = Math.max(seqLen - 1, INDICATOR_WARMUP);
  const latestBase   = nDays - 1 - maxH;

  const baseIndices = [];
  for (let i = earliestBase; i <= latestBase; i++) baseIndices.push(i);

  const totalSamples = baseIndices.length;
  const numTrain = Math.max(1, Math.floor(totalSamples * (1 - testSplit)));
  const numTest  = totalSamples - numTrain;

  const trainBase = baseIndices.slice(0, numTrain);
  const testBase  = baseIndices.slice(numTrain);

  const { mins, maxs } = computeMinMax(featureRows, symbols, featuresPerStock, trainBase[trainBase.length - 1]);

  // Build normalized windows
  const X = new Array(totalSamples);
  const Y = new Array(totalSamples);
  const baseDates = new Array(totalSamples);

  for (let k = 0; k < totalSamples; k++) {
    const base = baseIndices[k];

    // Sequence slice
    const seq = [];
    for (let t = base - seqLen + 1; t <= base; t++) {
      const raw = featureRows[t];
      const normRow = [];
      for (let s = 0; s < symbols.length; s++) {
        for (let f = 0; f < featuresPerStock; f++) {
          const v = raw[s * featuresPerStock + f];
          normRow.push(minmax(v, mins[s][f], maxs[s][f]));
        }
      }
      seq.push(normRow);
    }
    X[k] = seq;

    // Labels: future close > base close
    const labels = [];
    for (let s = 0; s < symbols.length; s++) {
      const baseClose = closeSeries[s][base];
      for (const h of horizons) {
        const fClose = closeSeries[s][base + h];
        labels.push(fClose > baseClose ? 1 : 0);
      }
    }
    Y[k] = labels;
    baseDates[k] = dates[base];
  }

  // Tensors & split
  const featureDim = symbols.length * featuresPerStock;
  const outDim     = symbols.length * horizons.length;

  const XTensor = tf.tensor3d(X);  // [total, seqLen, featureDim]
  const YTensor = tf.tensor2d(Y);  // [total, outDim]

  const X_train = XTensor.slice([0, 0, 0], [numTrain, seqLen, featureDim]);
  const y_train = YTensor.slice([0, 0], [numTrain, outDim]);

  const X_test  = XTensor.slice([numTrain, 0, 0], [numTest, seqLen, featureDim]);
  const y_test  = YTensor.slice([numTrain, 0], [numTest, outDim]);

  XTensor.dispose();
  YTensor.dispose();

  return {
    X_train, y_train, X_test, y_test,
    symbols, horizons,
    baseDatesTest: baseDates.slice(numTrain),
    norms: { mins, maxs },
    meta: { seqLen, featureDim, totalSamples, numTrain, numTest, featuresPerStock }
  };
}
