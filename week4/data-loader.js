// data-loader.js
// Client-side CSV loader & time-series dataset builder for multi-stock GRU.
// - Expects CSV columns: Date, Symbol, Open, Close
// - Produces tensors:
//   X_train [N_train, seqLen, 20], y_train [N_train, 30]
//   X_test  [N_test,  seqLen, 20], y_test  [N_test,  30]
// Notes: 10 symbols × 2 features = 20; 10 symbols × 3 horizons = 30.
// Reference assignment: RNN tutorial / prompt PDF. :contentReference[oaicite:1]{index=1}

export class DataLoader {
  constructor({ seqLen = 12, horizon = 3, testSplit = 0.2 } = {}) {
    this.seqLen = seqLen;
    this.horizon = horizon;
    this.testSplit = testSplit;
    this.symbols = [];
    this.norm = {}; // {SYM: {Open:{min,max}, Close:{min,max}}}
    this._prepared = false;
  }

  /** Parse CSV file using PapaParse; returns array of {Date,Symbol,Open,Close} */
  async parseCSVFile(file) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (res) => resolve(res.data),
        error: (err) => reject(err),
      });
    });
  }

  /** Prepare tensors from a parsed CSV file object */
  async loadFromFile(file, onInfo = () => {}) {
    const rows = await this.parseCSVFile(file);
    const clean = rows
      .map(r => ({
        Date: (r.Date ?? r.date ?? r.DATE ?? '').toString().slice(0, 10),
        Symbol: (r.Symbol ?? r.symbol ?? r.SYMBOL ?? '').toString().trim(),
        Open: Number(r.Open ?? r.open ?? r.OPEN),
        Close: Number(r.Close ?? r.close ?? r.CLOSE),
      }))
      .filter(r => r.Date && r.Symbol && Number.isFinite(r.Open) && Number.isFinite(r.Close));

    if (clean.length === 0) throw new Error('Parsed CSV is empty or columns are missing (need Date, Symbol, Open, Close).');

    // Collect symbols and per-date grouping
    const symbolSet = new Set();
    const byDate = new Map(); // dateStr -> {SYM: {Open,Close}}
    for (const r of clean) {
      symbolSet.add(r.Symbol);
      if (!byDate.has(r.Date)) byDate.set(r.Date, {});
      byDate.get(r.Date)[r.Symbol] = { Open: r.Open, Close: r.Close };
    }

    this.symbols = Array.from(symbolSet).sort().slice(0, 10);
    if (this.symbols.length !== 10) {
      throw new Error(`Expected 10 symbols; found ${this.symbols.length}. Symbols: ${this.symbols.join(', ')}`);
    }

    // Only keep dates where all 10 symbols are present
    const dates = Array.from(byDate.keys())
      .filter(d => this.symbols.every(sym => byDate.get(d)[sym] != null))
      .sort((a, b) => new Date(a) - new Date(b));

    if (dates.length < this.seqLen + this.horizon + 5) {
      throw new Error(`Not enough aligned dates across all 10 symbols. Need at least ${this.seqLen + this.horizon + 5}, got ${dates.length}.`);
    }

    // Compute min/max per symbol & feature for MinMax normalization
    this.norm = {};
    for (const sym of this.symbols) {
      let oMin = +Infinity, oMax = -Infinity, cMin = +Infinity, cMax = -Infinity;
      for (const d of dates) {
        const { Open, Close } = byDate.get(d)[sym];
        if (Open < oMin) oMin = Open; if (Open > oMax) oMax = Open;
        if (Close < cMin) cMin = Close; if (Close > cMax) cMax = Close;
      }
      this.norm[sym] = { Open: { min: oMin, max: oMax }, Close: { min: cMin, max: cMax } };
    }
    const eps = 1e-8;
    const normVal = (v, min, max) => (v - min) / Math.max(max - min, eps);

    // Build sliding windows
    const featDim = this.symbols.length * 2; // 20
    const outDim = this.symbols.length * this.horizon; // 30
    const X = [];
    const Y = [];
    const sampleDates = []; // current day (D) for each sample
    for (let i = this.seqLen - 1; i < dates.length - this.horizon; i++) {
      // Input window: indices [i - seqLen + 1 .. i]
      const window = [];
      for (let t = i - this.seqLen + 1; t <= i; t++) {
        const feats = [];
        const dateT = dates[t];
        for (const sym of this.symbols) {
          const { Open, Close } = byDate.get(dateT)[sym];
          const nOpen = normVal(Open, this.norm[sym].Open.min, this.norm[sym].Open.max);
          const nClose = normVal(Close, this.norm[sym].Close.min, this.norm[sym].Close.max);
          feats.push(nOpen, nClose);
        }
        window.push(feats);
      }
      // Targets: for each sym, compare Close(D+offset) > Close(D)
      const y = [];
      const baseDate = dates[i];
      for (const sym of this.symbols) {
        const baseClose = byDate.get(baseDate)[sym].Close;
        for (let k = 1; k <= this.horizon; k++) {
          const futureClose = byDate.get(dates[i + k])[sym].Close;
          y.push(futureClose > baseClose ? 1 : 0);
        }
      }
      X.push(window);
      Y.push(y);
      sampleDates.push(baseDate);
    }

    // Chronological split
    const total = X.length;
    const nTest = Math.max(1, Math.floor(total * this.testSplit));
    const nTrain = total - nTest;
    const X_train = tf.tensor3d(X.slice(0, nTrain));
    const y_train = tf.tensor2d(Y.slice(0, nTrain));
    const X_test = tf.tensor3d(X.slice(nTrain));
    const y_test = tf.tensor2d(Y.slice(nTrain));
    const trainDates = sampleDates.slice(0, nTrain);
    const testDates = sampleDates.slice(nTrain);

    this._prepared = true;

    onInfo({
      samples: total,
      train: nTrain,
      test: nTest,
      seqLen: this.seqLen,
      featDim,
      outDim,
      start: dates[0],
      end: dates[dates.length - 1],
      symbols: this.symbols.slice(),
    });

    return {
      X_train, y_train, X_test, y_test,
      symbols: this.symbols.slice(),
      testDates,
      seqLen: this.seqLen,
      featDim,
      outDim,
      horizon: this.horizon,
    };
  }
}
