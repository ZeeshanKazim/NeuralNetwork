// data-loader.js
// Denoising-AE + GRU pipeline data prep (browser-only, TF.js).
// CSV columns (case-insensitive): Date, Symbol, Open, High, Low, Close, Volume
// Output tensors are chronologically split: train → val → test.
// Features: per stock (12): O,H,L,C, Vol(log), Ret1, LogRet1, HL%Range, CO%Range, Mom5, Mom10, RSI14, BB%20, Vol10
// (Vol10 is volatility of log returns). All normalized using train split only.

export const DEFAULT_SEQ_LEN = 12;             // 12-day context (as required)
export const DEFAULT_HORIZONS = [1, 2, 3];     // predict +1d, +2d, +3d
export const MAX_STOCKS = 10;                  // keep 10 for browser memory
export const FEATURES_PER_STOCK = 14;          // see above

const INDICATOR_WARMUP = 20;                   // to compute 20-day indicators

// ---------- CSV parsing ----------
function detectDelimiter(firstLine) {
  const c = (firstLine.match(/,/g) || []).length;
  const s = (firstLine.match(/;/g) || []).length;
  return s > c ? ';' : ',';
}
function parseCSV(text) {
  const raw = text.replace(/\uFEFF/g, '');
  const headLine = raw.slice(0, raw.indexOf('\n') === -1 ? undefined : raw.indexOf('\n'));
  const delim = detectDelimiter(headLine);
  const lines = raw.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV has no data rows.');

  const header = lines[0].split(delim).map(h => h.trim());
  const idx = {
    Date:   header.findIndex(h => /^date$/i.test(h)),
    Symbol: header.findIndex(h => /^symbol$/i.test(h)),
    Open:   header.findIndex(h => /^open$/i.test(h)),
    High:   header.findIndex(h => /^high$/i.test(h)),
    Low:    header.findIndex(h => /^low$/i.test(h)),
    Close:  header.findIndex(h => /^close$/i.test(h)),
    Volume: header.findIndex(h => /^volume$/i.test(h)),
  };
  for (const k of Object.keys(idx)) if (idx[k] === -1) throw new Error(`Missing column "${k}".`);

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delim).map(c => c.trim());
    const d = new Date(cols[idx.Date]); if (isNaN(+d)) continue;
    const iso = d.toISOString().slice(0,10);
    const sym = cols[idx.Symbol];
    const o = parseFloat(cols[idx.Open]);
    const h = parseFloat(cols[idx.High]);
    const l = parseFloat(cols[idx.Low]);
    const c = parseFloat(cols[idx.Close]);
    const v = parseFloat(cols[idx.Volume]);
    if (!sym || ![o,h,l,c,v].every(Number.isFinite)) continue;
    rows.push({ date: iso, symbol: sym, open: o, high: h, low: l, close: c, volume: v });
  }
  if (!rows.length) throw new Error('No valid rows parsed.');
  return rows;
}

// ---------- Panel alignment (pick first MAX_STOCKS symbols, alphabetical) ----------
function alignPanel(records) {
  const allSymbols = Array.from(new Set(records.map(r => r.symbol))).sort().slice(0, MAX_STOCKS);
  const mapByDate = new Map();
  for (const r of records) {
    if (!allSymbols.includes(r.symbol)) continue;
    if (!mapByDate.has(r.date)) mapByDate.set(r.date, {});
    mapByDate.get(r.date)[r.symbol] = r;
  }
  const allDates = Array.from(mapByDate.keys()).sort();
  const dates = allDates.filter(d => {
    const m = mapByDate.get(d);
    return m && allSymbols.every(s => m[s]);
  });

  if (dates.length < INDICATOR_WARMUP + DEFAULT_SEQ_LEN + 4) {
    throw new Error('Not enough complete days for selected symbols.');
  }

  // Build raw series per symbol
  const O = allSymbols.map(() => []);
  const H = allSymbols.map(() => []);
  const L = allSymbols.map(() => []);
  const C = allSymbols.map(() => []);
  const V = allSymbols.map(() => []);

  for (const d of dates) {
    const row = mapByDate.get(d);
    for (let s = 0; s < allSymbols.length; s++) {
      const r = row[allSymbols[s]];
      O[s].push(r.open);
      H[s].push(r.high);
      L[s].push(r.low);
      C[s].push(r.close);
      V[s].push(r.volume);
    }
  }

  return { symbols: allSymbols, dates, O, H, L, C, V };
}

// ---------- helpers ----------
function mean(a, i0, i1) { let s=0,n=0; for (let i=i0;i<=i1;i++){ s+=a[i]; n++; } return s/Math.max(1,n); }
function std(a, i0, i1) { const m=mean(a,i0,i1); let s2=0,n=0; for (let i=i0;i<=i1;i++){ const d=a[i]-m; s2+=d*d; n++; } return Math.sqrt(s2/Math.max(1,n)); }
const div = (a,b,fb=0) => b===0?fb:a/b;

// Wilder’s RSI(14) scaled to [0,1]
function rsi14(C) {
  const n=C.length, out=new Array(n).fill(NaN);
  if (n<15) return out;
  let g=0,l=0;
  for (let i=1;i<=14;i++){ const d=C[i]-C[i-1]; if(d>0) g+=d; else l-=d; }
  let avgG=g/14, avgL=l/14;
  out[14] = avgL===0 ? 1 : 1 - 1/(1 + avgG/avgL);
  for (let i=15;i<n;i++){
    const d=C[i]-C[i-1], G=d>0?d:0, L=d<0?-d:0;
    avgG=(avgG*13+G)/14; avgL=(avgL*13+L)/14;
    const rs = avgL===0?1e6:avgG/avgL;
    out[i] = 1 - 1/(1 + rs);
  }
  return out;
}

// Build per-day feature rows [stocks*FEATURES_PER_STOCK]
function buildFeatureRows(symbols, O, H, L, C, V) {
  const n = C[0].length;
  const per = symbols.map((_, s) => {
    const o=O[s], h=H[s], l=L[s], c=C[s], v=V[s];
    const ret1 = new Array(n).fill(NaN);
    const logr = new Array(n).fill(NaN);
    const hlp  = new Array(n).fill(NaN);
    const cop  = new Array(n).fill(NaN);
    const mom5 = new Array(n).fill(NaN);
    const mom10= new Array(n).fill(NaN);
    const rs   = rsi14(c);
    const bbp  = new Array(n).fill(NaN);
    const vol10= new Array(n).fill(NaN);
    const logArr=new Array(n).fill(NaN);

    for (let t=0;t<n;t++){
      if (t>0){
        const r=div(c[t],c[t-1],1);
        ret1[t]=r-1;
        logr[t]=Math.log(Math.max(1e-8,r));
        logArr[t]=logr[t];
      }
      hlp[t]=div(h[t]-l[t], c[t]||1, 0);            // high-low % of close
      cop[t]=div(c[t]-o[t], o[t]||1, 0);            // close-open % of open
      if (t>=4) { const ma5=mean(c,t-4,t);  mom5[t] = div(c[t],ma5,1)-1; }
      if (t>=9) { const ma10=mean(c,t-9,t); mom10[t]= div(c[t],ma10,1)-1; }
      if (t>=19){ const ma20=mean(c,t-19,t); const sd20=std(c,t-19,t)||1; bbp[t] = 0.5 + (c[t]-ma20)/(2*sd20); }
      if (t>=9) { vol10[t]= std(logArr,t-9,t)||0; }
    }
    // log volume to compress heavy tails
    const vLog = v.map(x => Math.log(1 + Math.max(0,x)));
    return { o,h,l,c, vLog, ret1, logr, hlp, cop, mom5, mom10, rs, bbp, vol10 };
  });

  const rows=[];
  for (let t=0;t<n;t++){
    const row=[];
    for (let s=0;s<symbols.length;s++){
      const f=per[s];
      row.push(
        f.o[t], f.h[t], f.l[t], f.c[t],
        f.vLog[t] ?? 0,
        f.ret1[t] ?? 0,
        f.logr[t] ?? 0,
        f.hlp[t]  ?? 0,
        f.cop[t]  ?? 0,
        f.mom5[t] ?? 0,
        f.mom10[t]?? 0,
        f.rs[t]   ?? 0.5,
        f.bbp[t]  ?? 0.5,
        f.vol10[t]?? 0
      );
    }
    rows.push(row);
  }
  return rows;
}

// Per-stock, per-feature train-only min-max
function computeMinMax(featureRows, symbols, F, lastTrainDay) {
  const mins = symbols.map(()=> new Array(F).fill(Number.POSITIVE_INFINITY));
  const maxs = symbols.map(()=> new Array(F).fill(Number.NEGATIVE_INFINITY));
  for (let t=0;t<=lastTrainDay;t++){
    const r = featureRows[t];
    for (let s=0;s<symbols.length;s++){
      for (let f=0;f<F;f++){
        const v = r[s*F + f];
        if (v < mins[s][f]) mins[s][f]=v;
        if (v > maxs[s][f]) maxs[s][f]=v;
      }
    }
  }
  for (let s=0;s<symbols.length;s++){
    for (let f=0;f<F;f++){
      if (!isFinite(mins[s][f]) || !isFinite(maxs[s][f])) { mins[s][f]=0; maxs[s][f]=1; }
      if (maxs[s][f]===mins[s][f]) maxs[s][f]=mins[s][f]+1;
    }
  }
  return { mins, maxs };
}
const mm = (v,min,max)=> (v-min)/((max-min)||1);

// ---------- Public API ----------
export async function prepareDatasetFromFile(file, options = {}) {
  const {
    seqLen = DEFAULT_SEQ_LEN,
    horizons = DEFAULT_HORIZONS,
    testSplit = 0.2,
    valSplitWithinTrain = 0.12
  } = options;

  const text = await file.text();
  const recs = parseCSV(text);
  const { symbols, dates, O, H, L, C, V } = alignPanel(recs);

  const F = FEATURES_PER_STOCK;
  const featureRows = buildFeatureRows(symbols, O, H, L, C, V);

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
  const valBase   = baseIdx.slice(trainCount, trainCount+valCount);
  const testBase  = baseIdx.slice(trainCount+valCount);

  const { mins, maxs } = computeMinMax(featureRows, symbols, F, trainBase[trainBase.length-1]);

  function build(bases){
    const X=[], Y=[], baseDates=[];
    for (const base of bases){
      const seq=[];
      for (let t=base-seqLen+1;t<=base;t++){
        const raw = featureRows[t];
        const row = [];
        for (let s=0;s<symbols.length;s++){
          for (let f=0;f<F;f++){
            row.push( mm(raw[s*F+f], mins[s][f], maxs[s][f]) );
          }
        }
        seq.push(row);
      }
      X.push(seq);
      const lab=[];
      for (let s=0;s<symbols.length;s++){
        const baseClose = C[s][base];
        for (const h of horizons) lab.push( C[s][base+h] > baseClose ? 1 : 0 );
      }
      Y.push(lab);
      baseDates.push(dates[base]);
    }
    return { X, Y, baseDates };
  }

  const tr = build(trainBase);
  const va = build(valBase);
  const te = build(testBase);

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
    meta: { seqLen, featureDim, featuresPerStock: F, totalSamples: total, numTrain: trainCount, numVal: valCount, numTest: testCount }
  };
}
