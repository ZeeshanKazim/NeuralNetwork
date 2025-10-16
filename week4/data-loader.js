// data-loader.js
// Utility to load, pivot, normalize, and window stock CSV data for TensorFlow.js
// Expected CSV header contains (at minimum): Date, Symbol, Open, Close
// Outputs:
//   X_train: [numTrain, seqLen, 20]   // 10 stocks × [Open, Close]
//   y_train: [numTrain, 30]           // 10 stocks × 3 horizons (binary up/down)
//   X_test, y_test with the same shapes
//   symbols: ordered list of stock tickers used
//   baseDatesTest: base dates for each test sample (for timeline labels)

export const DEFAULT_SEQ_LEN = 12;
export const DEFAULT_HORIZONS = [1, 2, 3];
export const FEATURES_PER_STOCK = 2; // Open, Close
export const STOCK_COUNT_EXPECTED = 10;

// --- CSV Parsing & Pivoting -------------------------------------------------

function detectDelimiter(firstLine) {
  // Simple heuristic to handle common delimiters: comma or semicolon
  const comma = (firstLine.match(/,/g) || []).length;
  const semi = (firstLine.match(/;/g) || []).length;
  return semi > comma ? ';' : ',';
}

function parseCSV(text) {
  const raw = text.replace(/\uFEFF/g, ''); // strip BOM if present
  const firstNewline = raw.indexOf('\n');
  const headerLine = firstNewline === -1 ? raw : raw.slice(0, firstNewline);
  const delim = detectDelimiter(headerLine);

  const lines = raw.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV has no data rows.');

  const header = lines[0].split(delim).map(h => h.trim());
  const colIdx = {
    Date: header.findIndex(h => /^date$/i.test(h)),
    Symbol: header.findIndex(h => /^symbol$/i.test(h)),
    Open: header.findIndex(h => /^open$/i.test(h)),
    Close: header.findIndex(h => /^close$/i.test(h)),
  };
  for (const k of Object.keys(colIdx)) {
    if (colIdx[k] === -1) throw new Error(`Missing column "${k}" in CSV header.`);
  }

  const records = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delim).map(c => c.trim());
    if (cols.length < header.length) continue;

    const dateStr = cols[colIdx.Date];
    const symbol = cols[colIdx.Symbol];
    const open = parseFloat(cols[colIdx.Open]);
    const close = parseFloat(cols[colIdx.Close]);

    if (!dateStr || !symbol || !Number.isFinite(open) || !Number.isFinite(close)) continue;

    const d = new Date(dateStr);
    if (isNaN(+d)) continue;
    const iso = d.toISOString().slice(0, 10);

    records.push({ date: iso, symbol, open, close });
  }
  if (records.length === 0) throw new Error('No valid rows parsed from CSV.');
  return records;
}

function pivotByDate(records) {
  // Symbols sorted alphabetically for stable ordering
  const symbolSet = new Set(records.map(r => r.symbol));
  const symbols = Array.from(symbolSet).sort();

  // Map: date -> { symbol -> record }
  const byDate = new Map();
  for (const r of records) {
    if (!byDate.has(r.date)) byDate.set(r.date, {});
    byDate.get(r.date)[r.symbol] = r;
  }

  // Keep only dates where all symbols are present (complete panel)
  const dates = Array.from(byDate.keys()).sort();
  const validDates = dates.filter(d => {
    const row = byDate.get(d);
    return row && symbols.every(s => row[s] != null);
  });

  if (validDates.length < DEFAULT_SEQ_LEN + Math.max(...DEFAULT_HORIZONS) + 1) {
    throw new Error('Not enough complete days for required sequence/horizons.');
  }

  // Build feature rows per date in symbol order
  const featuresByDate = validDates.map(d => {
    const row = byDate.get(d);
    const out = [];
    for (const s of symbols) {
      out.push(row[s].open, row[s].close);
    }
    return out; // length = symbols.length * 2
  });

  // Close series per symbol (for labels)
  const closeSeries = symbols.map(s => validDates.map(d => byDate.get(d)[s].close));

  return { symbols, dates: validDates, featuresByDate, closeSeries };
}

// --- Normalization -----------------------------------------------------------

function computeMinMax(featuresByDate, symbols, lastTrainIndex) {
  const mins = symbols.map(() => [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]);
  const maxs = symbols.map(() => [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]);

  for (let t = 0; t <= lastTrainIndex; t++) {
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
  // Guard against degenerate ranges
  for (let s = 0; s < symbols.length; s++) {
    for (let f = 0; f < 2; f++) {
      if (!isFinite(mins[s][f]) || !isFinite(maxs[s][f])) {
        mins[s][f] = 0;
        maxs[s][f] = 1;
      }
      if (maxs[s][f] === mins[s][f]) maxs[s][f] = mins[s][f] + 1;
    }
  }
  return { mins, maxs };
}

function norm(val, min, max) {
  const range = max - min;
  return (val - min) / (range === 0 ? 1 : range);
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
    console.warn(`Expected ${expectStocks} symbols but found ${symbols.length}. Proceeding with ${symbols.length}.`);
  }

  const maxH = Math.max(...horizons);
  const earliestBase = seqLen - 1;                  // inclusive
  const latestBase = dates.length - 1 - maxH;       // inclusive

  const baseIndices = [];
  for (let i = earliestBase; i <= latestBase; i++) baseIndices.push(i);

  const totalSamples = baseIndices.length;
  const numTrain = Math.max(1, Math.floor(totalSamples * (1 - testSplit)));
  const numTest = totalSamples - numTrain;

  const trainBase = baseIndices.slice(0, numTrain);
  const testBase = baseIndices.slice(numTrain);

  const lastTrainBase = trainBase[trainBase.length - 1];
  const { mins, maxs } = computeMinMax(featuresByDate, symbols, lastTrainBase);

  const X = new Array(totalSamples);
  const Y = new Array(totalSamples);
  const baseDates = new Array(totalSamples);

  for (let k = 0; k < totalSamples; k++) {
    const base = baseIndices[k];

    // Build normalized sequence window [seqLen, symbols*2]
    const seq = [];
    for (let t = base - seqLen + 1; t <= base; t++) {
      const row = featuresByDate[t];
      const normRow = [];
      for (let s = 0; s < symbols.length; s++) {
        normRow.push(
          norm(row[2 * s], mins[s][0], maxs[s][0]),      // Open
          norm(row[2 * s + 1], mins[s][1], maxs[s][1])   // Close
        );
      }
      seq.push(normRow);
    }
    X[k] = seq;

    // Labels: for each stock, for each horizon
    const lab = [];
    for (let s = 0; s < symbols.length; s++) {
      const baseClose = closeSeries[s][base];
      for (let h of horizons) {
        const futureClose = closeSeries[s][base + h];
        lab.push(futureClose > baseClose ? 1 : 0);
      }
    }
    Y[k] = lab;
    baseDates[k] = dates[base];
  }

  // Convert to tensors & split chronologically
  const XTensor = tf.tensor3d(X);     // [total, seqLen, symbols*2]
  const YTensor = tf.tensor2d(Y);     // [total, symbols*horizons]

  const featureDim = symbols.length * FEATURES_PER_STOCK;
  const outDim = symbols.length * horizons.length;

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
