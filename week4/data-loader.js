// data-loader.js
// Utility to load, pivot, normalize, and window stock CSV data for TensorFlow.js
// Expected CSV header: Date,Symbol,Open,Close
// Output tensors:
//   X_train: [numTrainSamples, seqLen, 20]  (10 stocks × [Open, Close])
//   y_train: [numTrainSamples, 30]          (10 stocks × 3 horizons)
//   X_test, y_test with same shapes for the test split

export const DEFAULT_SEQ_LEN = 12;
export const DEFAULT_HORIZONS = [1, 2, 3];
export const FEATURES_PER_STOCK = 2; // Open, Close
export const STOCK_COUNT_EXPECTED = 10;

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV has no data rows.');
  const header = lines[0].split(',').map(h => h.trim());
  const colIdx = {
    Date: header.indexOf('Date'),
    Symbol: header.indexOf('Symbol'),
    Open: header.indexOf('Open'),
    Close: header.indexOf('Close'),
  };
  for (const k of Object.keys(colIdx)) {
    if (colIdx[k] === -1) throw new Error(`Missing column "${k}" in CSV header.`);
  }

  const records = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    if (cols.length < header.length) continue;
    const dateStr = cols[colIdx.Date];
    const symbol = cols[colIdx.Symbol];
    const open = parseFloat(cols[colIdx.Open]);
    const close = parseFloat(cols[colIdx.Close]);
    if (!dateStr || !symbol || !Number.isFinite(open) || !Number.isFinite(close)) continue;
    // Normalize date format to YYYY-MM-DD for stable sort
    const d = new Date(dateStr);
    if (isNaN(+d)) continue;
    const iso = d.toISOString().slice(0, 10);
    records.push({ date: iso, symbol, open, close });
  }
  if (records.length === 0) throw new Error('No valid rows parsed from CSV.');
  return records;
}

function pivotByDate(records) {
  // Collect symbols and dates
  const symbolSet = new Set();
  const dateSet = new Set();
  for (const r of records) {
    symbolSet.add(r.symbol);
    dateSet.add(r.date);
  }
  const symbols = Array.from(symbolSet).sort();
  const dates = Array.from(dateSet).sort();

  // Map date -> { symbol -> row }
  const byDate = new Map();
  for (const r of records) {
    if (!byDate.has(r.date)) byDate.set(r.date, {});
    byDate.get(r.date)[r.symbol] = r;
  }

  // Keep only dates where all symbols are present (complete panel)
  const validDates = dates.filter(d => {
    const m = byDate.get(d);
    return m && symbols.every(s => m[s] != null);
  });

  if (validDates.length < DEFAULT_SEQ_LEN + 3) {
    throw new Error('Not enough complete days for required sequence/horizons.');
  }

  // Build feature rows per date in symbol order
  const featuresByDate = validDates.map(d => {
    const m = byDate.get(d);
    const row = [];
    for (const s of symbols) {
      const rec = m[s];
      row.push(rec.open, rec.close);
    }
    return row; // length = symbols.length * 2
  });

  // Build close series per symbol for label creation
  const closeSeries = symbols.map((s) => validDates.map(d => byDate.get(d)[s].close));

  return { symbols, dates: validDates, featuresByDate, closeSeries };
}

function computeMinMax(featuresByDate, symbols, trainLastIdxInclusive) {
  // Per-stock, per-feature min/max from training portion only
  const mins = symbols.map(() => [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]); // [openMin, closeMin]
  const maxs = symbols.map(() => [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]); // [openMax, closeMax]
  for (let t = 0; t <= trainLastIdxInclusive; t++) {
    const row = featuresByDate[t];
    for (let s = 0; s < symbols.length; s++) {
      const open = row[2 * s];
      const close = row[2 * s + 1];
      if (open < mins[s][0]) mins[s][0] = open;
      if (open > maxs[s][0]) maxs[s][0] = open;
      if (close < mins[s][1]) mins[s][1] = close;
      if (close > maxs[s][1]) maxs[s][1] = close;
    }
  }
  return { mins, maxs };
}

function norm(val, min, max) {
  const range = (max - min);
  return (val - min) / (range === 0 ? 1 : range);
}

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
    console.warn(`Expected ${expectStocks} symbols but found ${symbols.length}. Proceeding anyway.`);
  }

  const horizonMax = Math.max(...horizons);
  const earliestBase = seqLen - 1;
  const latestBase = dates.length - 1 - horizonMax;

  const baseIndices = [];
  for (let i = earliestBase; i <= latestBase; i++) {
    baseIndices.push(i);
  }
  const totalSamples = baseIndices.length;
  const numTrain = Math.max(1, Math.floor(totalSamples * (1 - testSplit)));
  const numTest = totalSamples - numTrain;
  const trainBaseIndices = baseIndices.slice(0, numTrain);
  const testBaseIndices = baseIndices.slice(numTrain);

  // Compute min-max using all feature rows up to the last training base index
  const lastTrainBase = trainBaseIndices[trainBaseIndices.length - 1];
  const { mins, maxs } = computeMinMax(featuresByDate, symbols, lastTrainBase);

  // Build normalized sample windows and labels
  const X = new Array(totalSamples);
  const Y = new Array(totalSamples);
  const baseDates = new Array(totalSamples);

  for (let k = 0; k < totalSamples; k++) {
    const base = baseIndices[k];
    const seq = [];
    for (let t = base - seqLen + 1; t <= base; t++) {
      const rawRow = featuresByDate[t];
      const normRow = [];
      for (let s = 0; s < symbols.length; s++) {
        normRow.push(
          norm(rawRow[2 * s], mins[s][0], maxs[s][0]),
          norm(rawRow[2 * s + 1], mins[s][1], maxs[s][1]),
        );
      }
      seq.push(normRow);
    }
    X[k] = seq;

    const labels = [];
    for (let s = 0; s < symbols.length; s++) {
      const baseClose = closeSeries[s][base];
      for (let h of horizons) {
        const fClose = closeSeries[s][base + h];
        labels.push(fClose > baseClose ? 1 : 0);
      }
    }
    Y[k] = labels;
    baseDates[k] = dates[base];
  }

  // Convert to tensors and split
  const XTensor = tf.tensor3d(X); // [total, seqLen, symbols*2]
  const YTensor = tf.tensor2d(Y); // [total, symbols*len(horizons)]

  const X_train = XTensor.slice([0, 0, 0], [numTrain, seqLen, symbols.length * FEATURES_PER_STOCK]);
  const y_train = YTensor.slice([0, 0], [numTrain, symbols.length * horizons.length]);
  const X_test = XTensor.slice([numTrain, 0, 0], [numTest, seqLen, symbols.length * FEATURES_PER_STOCK]);
  const y_test = YTensor.slice([numTrain, 0], [numTest, symbols.length * horizons.length]);

  // Free temporaries
  XTensor.dispose();
  YTensor.dispose();

  return {
    X_train,
    y_train,
    X_test,
    y_test,
    symbols,
    horizons,
    baseDatesTest: baseDates.slice(numTrain),
    norms: { mins, maxs },
    meta: {
      seqLen,
      featureDim: symbols.length * FEATURES_PER_STOCK,
      totalSamples,
      numTrain,
      numTest,
      datesCount: dates.length,
    }
  };
}
