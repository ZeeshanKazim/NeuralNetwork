// data-loader.js
// Rich features + chronological train/val/test split + train-only normalization.
//
// CSV columns (case-insensitive): Date, Symbol, Open, Close
// Inputs  : [samples, seqLen, 10 stocks × 10 features]  => featureDim = 100
// Labels  : [samples, 10 stocks × 3 horizons]           => 30 binaries
// Splits  : chronological (train -> val -> test)
// Exports : X_train, y_train, X_val, y_val, X_test, y_test, symbols, horizons, baseDatesTest

export const DEFAULT_SEQ_LEN = 48;
export const DEFAULT_HORIZONS = [1, 2, 3];
export const FEATURES_PER_STOCK = 10; // Open, Close, OCret, Ret1, LogRet1, Mom5, Mom10, RSI14, BB%20, Vol10
export const STOCK_COUNT_EXPECTED = 10;
const INDICATOR_WARMUP = 20;

function detectDelimiter(firstLine) {
  const comma = (firstLine.match(/,/g) || []).length;
  const semi = (firstLine.match(/;/g) || []).length;
  return semi > comma ? ';' : ',';
}

function parseCSV(text) {
  const raw = text.replace(/\uFEFF/g, '');
  const headerLine = raw.slice(0, raw.indexOf('\n') === -1 ? undefined : raw.indexOf('\n'));
  const delim = detectDelimiter(headerLine);
  const lines = raw.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV has no data rows.');

  const header = lines[0].split(delim).map(h => h.trim());
  const idx = {
    Date: header.findIndex(h => /^date$/i.test(h)),
    Symbol: header.findIndex(h => /^symbol$/i.test(h)),
    Open: header.findIndex(h => /^open$/i.test(h)),
    Close: header.findIndex(h => /^close$/i.test(h)),
  };
  for (const k of Object.keys(idx)) if (idx[k] === -1) throw new Error(`Missing "${k}" column.`);

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delim).map(c => c.trim());
    const d = new Date(cols[idx.Date]); if (isNaN(+d)) continue;
    const iso = d.toISOString().slice(0, 10);
    const symbol = cols[idx.Symbol];
    const open = parseFloat(cols[idx.Open]);
    const close = parseFloat(cols[idx.Close]);
    if (!symbol || !Number.isFinite(open) || !Number.isFinite(close)) continue;
    rows.push({ date: iso, symbol, open, close });
  }
  if (!rows.length) throw new Error('No valid rows parsed.');
  return rows;
}

function pivotPanel(records) {
  const symbolSet = new Set(records.map(r => r.symbol));
  const symbols = Array.from(symbolSet).sort();

  const byDate = new Map();
  for (const r of records) {
    if (!byDate.has(r.date)) byDate.set(r.date, {});
    byDate.get(r.date)[r.symbol] = r;
  }

  const allDates = Array.from(byDate.keys()).sort();
  const dates = allDates.filter(d => {
    const m = byDate.get(d);
    return m && symbols.every(s => m[s] != null);
  });
  if (dates.length < INDICATOR_WARMUP + 5) throw new Error('Not enough complete days across all symbols.');

  const openSeries = symbols.map(() => []);
  const closeSeries = symbols.map(() => []);

  for (const d of dates) {
    const row = byDate.get(d);
    for (let s = 0; s < symbols.length; s++) {
      const sym = symbols[s];
      openSeries[s].push(row[sym].open);
      closeSeries[s].push(row[sym].close);
    }
  }
  return { symbols, dates, openSeries, closeSeries };
}

// math utils
function mean(a, i0, i1) { let s = 0; let n = 0; for (let i = i0; i <= i1; i++) { s += a[i]; n++; } return s / Math.max(1, n); }
function std(a, i0, i1) { const m = mean(a, i0, i1); let s2 = 0; let n = 0; for (let i = i0; i <= i1; i++) { const d = a[i] - m; s2 += d*d; n++; } return Math.sqrt(s2 / Math.max(1, n)); }
function div(a, b, fb = 0) { return b === 0 ? fb : a / b; }

// RSI(14) in 0..1
function rsi14(C) {
  const n = C.length, out = new Array(n).fill(NaN);
  if (n < 15) return out;
  let g = 0, l = 0;
  for (let i = 1; i <= 14; i++) { const d = C[i] - C[i-1]; if (d > 0) g += d; else l -= d; }
  let avgG = g / 14, avgL = l / 14;
  out[14] = avgL === 0 ? 1 : 1 - 1 / (1 + avgG / avgL);
  for (let i = 15; i < n; i++) {
    const d = C[i] - C[i-1];
    const G = d > 0 ? d : 0, L = d < 0 ? -d : 0;
    avgG = (avgG * 13 + G) / 14;
    avgL = (avgL * 13 + L) / 14;
    const rs = avgL === 0 ? 1e6 : avgG / avgL;
    out[i] = 1 - 1 / (1 + rs);
  }
  return out;
}

function buildFeatureRows(symbols, openSeries, closeSeries) {
  const n = closeSeries[0].length;
  const rows = [];

  const per = symbols.map((_, s) => {
    const O = openSeries[s], C = closeSeries[s];
    const ocRet = new Array(n).fill(NaN);
    const ret1  = new Array(n).fill(NaN);
    const logR  = new Array(n).fill(NaN);
    const mom5  = new Array(n).fill(NaN);
    const mom10 = new Array(n).fill(NaN);
    const rsi   = rsi14(C);
    const bbp   = new Array(n).fill(NaN);
    const vol10 = new Array(n).fill(NaN);

    const logArr = new Array(n).fill(NaN);

    for (let t = 0; t < n; t++) {
      ocRet[t] = div(C[t], O[t], 1) - 1;
      if (t > 0) {
        const r = div(C[t], C[t-1], 1);
        ret1[t] = r - 1;
        logR[t] = Math.log(Math.max(1e-8, r));
        logArr[t] = logR[t];
      }
      if (t >= 4)  { const ma5  = mean(C, t-4,  t);  mom5[t]  = div(C[t], ma5, 1)  - 1; }
      if (t >= 9)  { const ma10 = mean(C, t-9,  t);  mom10[t] = div(C[t], ma10,1) - 1; }
      if (t >= 19) {
        const ma20 = mean(C, t-19, t); const sd20 = std(C, t-19, t) || 1;
        bbp[t] = 0.5 + (C[t] - ma20) / (2 * sd20);
      }
      if (t >= 9)  vol10[t] = std(logArr, t-9, t) || 0;
    }
    return { O, C, ocRet, ret1, logR, mom5, mom10, rsi, bbp, vol10 };
  });

  for (let t = 0; t < n; t++) {
    const row = [];
    for (let s = 0; s < symbols.length; s++) {
      const f = per[s];
      row.push(
        f.O[t], f.C[t],
        f.ocRet[t]  ?? 0,
        f.ret1[t]   ?? 0,
        f.logR[t]   ?? 0,
        f.mom5[t]   ?? 0,
        f.mom10[t]  ?? 0,
        f.rsi[t]    ?? 0.5,
        f.bbp[t]    ?? 0.5,
        f.vol10[t]  ?? 0
      );
    }
    rows.push(row);
  }
  return rows;
}

function computeMinMax(featureRows, symbols, featuresPerStock, lastTrainDay) {
  const mins = symbols.map(() => new Array(featuresPerStock).fill(Number.POSITIVE_INFINITY));
  const maxs = symbols.map(() => new Array(featuresPerStock).fill(Number.NEGATIVE_INFINITY));
  for (let t = 0; t <= lastTrainDay; t++) {
    const r = featureRows[t];
    for (let s = 0; s < symbols.length; s++) {
      for (let f = 0; f < featuresPerStock; f++) {
        const v = r[s*featuresPerStock + f];
        if (v < mins[s][f]) mins[s][f] = v;
        if (v > maxs[s][f]) maxs[s][f] = v;
      }
    }
  }
  for (let s = 0; s < symbols.length; s++) {
    for (let f = 0; f < featuresPerStock; f++) {
      if (!isFinite(mins[s][f]) || !isFinite(maxs[s][f])) { mins[s][f] = 0; maxs[s][f] = 1; }
      if (maxs[s][f] === mins[s][f]) maxs[s][f] = mins[s][f] + 1;
    }
  }
  return { mins, maxs };
}
const mm = (v, min, max) => (v - min) / ((max - min) || 1);

// MAIN
export async function prepareDatasetFromFile(file, options = {}) {
  const {
    seqLen = DEFAULT_SEQ_LEN,
    horizons = DEFAULT_HORIZONS,
    testSplit = 0.2,
    valSplitWithinTrain = 0.12,
    expectStocks = STOCK_COUNT_EXPECTED,
  } = options;

  const text = await file.text();
  const recs = parseCSV(text);
  const { symbols, dates, openSeries, closeSeries } = pivotPanel(recs);
  if (expectStocks && symbols.length !== expectStocks) console.warn(`Expected ${expectStocks} symbols, found ${symbols.length}.`);

  const featureRows = buildFeatureRows(symbols, openSeries, closeSeries);
  const F = FEATURES_PER_STOCK;
  const maxH = Math.max(...horizons);
  const earliestBase = Math.max(seqLen - 1, INDICATOR_WARMUP);
  const latestBase = dates.length - 1 - maxH;

  const baseIdx = [];
  for (let i = earliestBase; i <= latestBase; i++) baseIdx.push(i);

  const total = baseIdx.length;
  const testCount = Math.max(1, Math.floor(total * testSplit));
  const trainValCount = total - testCount;

  const valCount = Math.max(1, Math.floor(trainValCount * valSplitWithinTrain));
  const trainCount = trainValCount - valCount;

  const trainBase = baseIdx.slice(0, trainCount);
  const valBase   = baseIdx.slice(trainCount, trainCount + valCount);
  const testBase  = baseIdx.slice(trainCount + valCount);

  const { mins, maxs } = computeMinMax(featureRows, symbols, F, trainBase[trainBase.length - 1]);

  const make = (bases) => {
    const X = [], Y = [], baseDates = [];
    for (const base of bases) {
      // window
      const seq = [];
      for (let t = base - seqLen + 1; t <= base; t++) {
        const raw = featureRows[t];
        const row = [];
        for (let s = 0; s < symbols.length; s++) {
          for (let f = 0; f < F; f++) {
            row.push(mm(raw[s*F + f], mins[s][f], maxs[s][f]));
          }
        }
        seq.push(row);
      }
      X.push(seq);
      // labels vs base close
      const lab = [];
      for (let s = 0; s < symbols.length; s++) {
        const baseC = closeSeries[s][base];
        for (const h of horizons) {
          const fut = closeSeries[s][base + h];
          lab.push(fut > baseC ? 1 : 0);
        }
      }
      Y.push(lab);
      baseDates.push(dates[base]);
    }
    return { X, Y, baseDates };
  };

  const tr = make(trainBase);
  const va = make(valBase);
  const te = make(testBase);

  const featureDim = symbols.length * F;
  const outDim = symbols.length * horizons.length;

  const X_train = tf.tensor3d(tr.X);
  const y_train = tf.tensor2d(tr.Y);
  const X_val   = tf.tensor3d(va.X);
  const y_val   = tf.tensor2d(va.Y);
  const X_test  = tf.tensor3d(te.X);
  const y_test  = tf.tensor2d(te.Y);

  return {
    X_train, y_train, X_val, y_val, X_test, y_test,
    symbols, horizons,
    baseDatesTest: te.baseDates,
    meta: { seqLen, featuresPerStock: F, featureDim, totalSamples: total, numTrain: trainCount, numVal: valCount, numTest: testCount }
  };
}
