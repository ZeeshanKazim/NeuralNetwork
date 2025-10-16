// data-loader.js
// Loads, pivots, normalizes, and windows stock CSV data for TensorFlow.js.
// Expected columns (case-insensitive): Date, Symbol, Open, Close
// Returns tensors split chronologically into train/test and useful metadata.

export const DEFAULT_SEQ_LEN = 12;
export const DEFAULT_HORIZONS = [1, 2, 3];
export const FEATURES_PER_STOCK = 2; // [Open, Close]
export const STOCK_COUNT_EXPECTED = 10;

// --- CSV parsing -------------------------------------------------------------

function detectDelimiter(firstLine) {
  const comma = (firstLine.match(/,/g) || []).length;
  const semi = (firstLine.match(/;/g) || []).length;
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
  const findIdx = (name) => header.findIndex(h => h.toLowerCase() === name.toLowerCase());

  const idxDate = findIdx('Date');
  const idxSymbol = findIdx('Symbol');
  const idxOpen = findIdx('Open');
  const idxClose = findIdx('Close');
  if (idxDate === -1 || idxSymbol === -1 || idxOpen === -1 || idxClose === -1) {
    throw new Error('CSV must contain Date, Symbol, Open, Close columns.');
  }

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delim).map(c => c.trim());
    if (!cols.length) continue;

    const d = new Date(cols[idxDate]);
    if (isNaN(+d)) continue;
    const iso = d.toISOString().slice(0, 10);

    const symbol = cols[idxSymbol];
    const open = parseFloat(cols[idxOpen]);
    const close = parseFloat(cols[idxClose]);
    if (!symbol || !Number.isFinite(open) || !Number.isFinite(close)) continue;

    rows.push({ date: iso, symbol, open, close });
  }
  if (!rows.length) throw new Error('No valid data parsed from CSV.');
  return rows;
}

// --- Pivot to complete panel -------------------------------------------------

function pivotByDate(records) {
  const symbolSet = new Set(records.map(r => r.symbol));
  const symbols = Array.from(symbolSet).sort();

  const byDate = new Map();
  for (const r of records) {
    if (!byDate.has(r.date)) byDate.set(r.date, {});
    byDate.get(r.date)[r.symbol] = r;
  }

  const allDates = Array.from(byDate.keys()).sort();
  const validDates = allDates.filter(d => {
    const row = byDate.get(d);
    return row && symbols.every(s => row[s] != null);
  });

  if (validDates.length < DEFAULT_SEQ_LEN + Math.max(...DEFAULT_HORIZONS) + 1) {
    throw new Error('Not enough complete days across all symbols.');
  }

  const featuresByDate = validDates.map(d => {
    const row = byDate.get(d);
    const v = [];
    for (const s of symbols) {
      v.push(row[s].open, row[s].close);
    }
    return v; // length = symbols.length * 2
  });

  const closeSeries = symbols.map(s => validDates.map(d => byDate.get(d)[s].close));

  return { symbols, dates: validDates, featuresByDate, closeSeries };
}

// --- Normalization -----------------------------------------------------------

function computeMinMax(featuresByDate, symbols, lastTrainIdx) {
  const mins = symbols.map(() => [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]);
  const maxs = symbols.map(() => [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]);

  for (let t = 0; t <= lastTrainIdx; t++) {
    const row = featuresByDate[t];
    for (let s = 0; s < symbols.length; s++) {
      const o = row[2 * s];
      const c = row[2 * s + 1];
      if (o < mins[s][0]) mins[s][0] = o;
      if (o > maxs[s][0]) maxs[s][0] = o;
      if (c < mins[s][1]) mins[s][1] = c;
      if (c > maxs[s][1]) maxs[s][1] = c;
    }
  }

  // Guard degenerate
  for (let s = 0; s < symbols.length; s++) {
    for (let f = 0; f < 2; f++) {
      if (!isFinite(mins[s][f]) || !isFinite(maxs[s][f])) {
        mins[s][f] = 0; maxs[s][f] = 1;
      }
      if (maxs[s][f] === mins[s][f]) maxs[s][f] = mins[s][f] + 1;
    }
  }
  return { mins, maxs };
}

function norm(val, min, max) {
  const r = max - min;
  return (val - min) / (r === 0 ? 1 : r);
}

// --- Public API --------------------------------------------------------------

export async function prepareDatasetFromFile(file, options = {}) {
  const {
    seqLen = DEFAULT_SEQ_LEN,
    horizons = DEFAULT_HORIZONS,
    testSplit = 0.2,
    expectStocks = STOCK_COUNT_EXPECTED,
  } = options;

  const text = await file.text();
  const records = parseCSV(text);
  const { symbols, dates, featuresByDate, closeSeries } = pivotByDate(records);

  if (expectStocks && symbols.length !== expectStocks) {
    console.warn(`Expected ${expectStocks} stocks but found ${symbols.length}. Proceeding.`);
  }

  const maxH = Math.max(...horizons);
  const earliestBase = seqLen - 1;                  // inclusive
  const latestBase = dates.length - 1 - maxH;       // inclusive

  const baseIdxs = [];
  for (let i = earliestBase; i <= latestBase; i++) baseIdxs.push(i);

  const totalSamples = baseIdxs.length;
  const numTrain = Math.max(1, Math.floor(totalSamples * (1 - testSplit)));
  const numTest = totalSamples - numTrain;

  const trainBase = baseIdxs.slice(0, numTrain);
  const testBase = baseIdxs.slice(numTrain);

  const lastTrainBase = trainBase[trainBase.length - 1];
  const { mins, maxs } = computeMinMax(featuresByDate, symbols, lastTrainBase);

  const X = new Array(totalSamples);
  const Y = new Array(totalSamples);
  const baseDates = new Array(totalSamples);

  for (let k = 0; k < totalSamples; k++) {
    const base = baseIdxs[k];

    // Sequence window
    const seq = [];
    for (let t = base - seqLen + 1; t <= base; t++) {
      const row = featuresByDate[t];
      const nr = [];
      for (let s = 0; s < symbols.length; s++) {
        nr.push(
          norm(row[2 * s], mins[s][0], maxs[s][0]),
          norm(row[2 * s + 1], mins[s][1], maxs[s][1])
        );
      }
      seq.push(nr);
    }
    X[k] = seq;

    // Labels
    const lab = [];
    for (let s = 0; s < symbols.length; s++) {
      const baseClose = closeSeries[s][base];
      for (const h of horizons) {
        const fClose = closeSeries[s][base + h];
        lab.push(fClose > baseClose ? 1 : 0);
      }
    }
    Y[k] = lab;
    baseDates[k] = dates[base];
  }

  // Tensors & splits
  const featureDim = symbols.length * FEATURES_PER_STOCK;
  const outDim = symbols.length * horizons.length;

  const XTensor = tf.tensor3d(X); // [total, seqLen, featureDim]
  const YTensor = tf.tensor2d(Y); // [total, outDim]

  const X_train = XTensor.slice([0, 0, 0], [numTrain, seqLen, featureDim]);
  const y_train = YTensor.slice([0, 0], [numTrain, outDim]);

  const X_test = XTensor.slice([numTrain, 0, 0], [numTest, seqLen, featureDim]);
  const y_test = YTensor.slice([numTrain, 0], [numTest, outDim]);

  XTensor.dispose();
  YTensor.dispose();

  return {
    X_train, y_train, X_test, y_test,
    symbols,
    horizons,
    baseDatesTest: baseDates.slice(numTrain),
    norms: { mins, maxs },
    meta: { seqLen, featureDim, totalSamples, numTrain, numTest }
  };
}
