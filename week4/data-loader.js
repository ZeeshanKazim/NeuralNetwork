// data-loader.js
// Default export to match the user's original import style. :contentReference[oaicite:2]{index=2}
export default class DataLoader {
  constructor() {
    this.stocksData = null;     // {SYM: {date: {Open,Close}}}
    this.symbols = [];
    this.dates = [];
    this.alignedDates = [];

    this.X_train = null; this.y_train = null;
    this.X_val   = null; this.y_val   = null;
    this.X_test  = null; this.y_test  = null;

    this.testDates = [];
    this.seqLen = 12;
    this.horizon = 3;
  }

  async loadCSV(file) {
    const text = await file.text();
    this._parseCSV(text);
  }

  _parseCSV(csvText) {
    const lines = csvText.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.trim());
    const colIdx = (name) => headers.findIndex(h => h.toLowerCase() === name.toLowerCase());

    const iDate = colIdx('Date');
    const iSym  = colIdx('Symbol');
    const iOpen = colIdx('Open');
    const iClose= colIdx('Close');

    if ([iDate,iSym,iOpen,iClose].some(i => i < 0)) {
      throw new Error('CSV must include columns: Date, Symbol, Open, Close');
    }

    const data = {};
    const symSet = new Set();
    const dateSet = new Set();

    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length < headers.length) continue;

      const date = row[iDate]?.trim().slice(0,10);
      const sym  = row[iSym]?.trim();
      const open = Number(row[iOpen]);
      const close= Number(row[iClose]);
      if (!date || !sym || !Number.isFinite(open) || !Number.isFinite(close)) continue;

      if (!data[sym]) data[sym] = {};
      data[sym][date] = { Open: open, Close: close };
      symSet.add(sym); dateSet.add(date);
    }

    // Keep exactly 10 symbols (alphabetical if more)
    this.symbols = Array.from(symSet).sort().slice(0,10);
    if (this.symbols.length !== 10) {
      throw new Error(`Expected 10 symbols, found ${this.symbols.length}.`);
    }

    // Keep only dates where all 10 symbols exist
    const allDates = Array.from(dateSet).sort();
    this.alignedDates = allDates.filter(d => this.symbols.every(s => data[s]?.[d]));
    if (this.alignedDates.length < 40) {
      throw new Error('Not enough aligned dates across all symbols.');
    }

    this.stocksData = data;
    this.dates = this.alignedDates;
  }

  _minMaxPerSymbol() {
    const mm = {};
    for (const s of this.symbols) {
      let omin=Infinity, omax=-Infinity, cmin=Infinity, cmax=-Infinity;
      for (const d of this.dates) {
        const {Open,Close} = this.stocksData[s][d];
        omin=Math.min(omin,Open); omax=Math.max(omax,Open);
        cmin=Math.min(cmin,Close); cmax=Math.max(cmax,Close);
      }
      mm[s]={Open:{min:omin,max:omax}, Close:{min:cmin,max:cmax}};
    }
    return mm;
  }

  createSequences(sequenceLength = 12, predictionHorizon = 3, { testSplit = 0.2, valSplit = 0.1 } = {}) {
    this.seqLen = sequenceLength;
    this.horizon = predictionHorizon;

    const mm = this._minMaxPerSymbol();
    const eps = 1e-8;
    const norm = (v, min, max) => (v - min) / Math.max(max - min, eps);

    const sequences = [];
    const targets = [];
    const baseDates = [];

    for (let i = sequenceLength - 1; i < this.dates.length - predictionHorizon; i++) {
      const window = [];
      let valid = true;

      for (let t = i - sequenceLength + 1; t <= i; t++) {
        const feats = [];
        const dateT = this.dates[t];
        for (const s of this.symbols) {
          const {Open,Close} = this.stocksData[s][dateT];
          feats.push(norm(Open, mm[s].Open.min, mm[s].Open.max));
          feats.push(norm(Close, mm[s].Close.min, mm[s].Close.max));
        }
        window.push(feats);
      }

      const y = [];
      const baseDate = this.dates[i];
      for (const s of this.symbols) {
        const baseClose = this.stocksData[s][baseDate].Close;
        for (let k = 1; k <= predictionHorizon; k++) {
          const futureDate = this.dates[i + k];
          if (!futureDate) { valid = false; break; }
          const futureClose = this.stocksData[s][futureDate]?.Close;
          if (!Number.isFinite(futureClose)) { valid = false; break; }
          y.push(futureClose > baseClose ? 1 : 0);
        }
      }
      if (!valid) continue;

      sequences.push(window);
      targets.push(y);
      baseDates.push(baseDate);
    }

    const N = sequences.length;
    const nTest = Math.max(1, Math.floor(N * testSplit));
    const nVal  = Math.max(1, Math.floor((N - nTest) * valSplit));
    const nTrain= N - nTest - nVal;

    const X = tf.tensor3d(sequences);
    const Y = tf.tensor2d(targets);

    this.X_train = X.slice([0,0,0],[nTrain,sequenceLength,this.symbols.length*2]);
    this.y_train = Y.slice([0,0],[nTrain,this.symbols.length*predictionHorizon]);

    this.X_val   = X.slice([nTrain,0,0],[nVal,sequenceLength,this.symbols.length*2]);
    this.y_val   = Y.slice([nTrain,0],[nVal,this.symbols.length*predictionHorizon]);

    this.X_test  = X.slice([nTrain+nVal,0,0],[nTest,sequenceLength,this.symbols.length*2]);
    this.y_test  = Y.slice([nTrain+nVal,0],[nTest,this.symbols.length*predictionHorizon]);

    this.testDates = baseDates.slice(nTrain + nVal);

    // Free big tensors we sliced from
    X.dispose(); Y.dispose();

    return {
      X_train:this.X_train, y_train:this.y_train,
      X_val:this.X_val, y_val:this.y_val,
      X_test:this.X_test, y_test:this.y_test,
      symbols:this.symbols, testDates:this.testDates,
      seqLen:this.seqLen, horizon:this.horizon
    };
  }

  dispose() {
    for (const k of ['X_train','y_train','X_val','y_val','X_test','y_test']) {
      if (this[k]?.dispose) try { this[k].dispose(); } catch {}
      this[k]=null;
    }
  }
}
