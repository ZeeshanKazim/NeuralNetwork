// data-loader.js
// Drop-in replacement for your loader.
// Changes:
//  - Parse OHLCV
//  - Engineer richer features per stock (14 total)
//  - Chronological split (80/20)
//  - Train-only min–max normalization to avoid leakage
//  - Expose featuresPerStock so app/model know input size

export default class DataLoader {
  constructor() {
    this.symbols = [];
    this.dates = [];
    this.stocksData = null;

    this.featuresPerStock = 14; // O,H,L,C, logVol, Ret1, LogRet1, HL%, CO%, Mom5, Mom10, RSI14, BB%20, Vol10

    this.X_train = null; this.y_train = null;
    this.X_val = null;   this.y_val = null;  // not used by your UI, but helpful if you add it
    this.X_test = null;  this.y_test = null;
    this.testDates = [];
  }

  async loadCSV(file) {
    const text = await file.text();
    this.parseCSV(text);
    return this.stocksData;
  }

  parseCSV(csvText) {
    const lines = csvText.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.trim());
    const idx = {
      Date: headers.indexOf('Date'),
      Symbol: headers.indexOf('Symbol'),
      Open: headers.indexOf('Open'),
      High: headers.indexOf('High'),
      Low: headers.indexOf('Low'),
      Close: headers.indexOf('Close'),
      Volume: headers.indexOf('Volume')
    };
    for (const k of Object.keys(idx)) {
      if (idx[k] === -1) throw new Error(`Missing column "${k}" in CSV`);
    }

    const data = {};
    const symSet = new Set();
    const dateSet = new Set();

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',');
      if (cols.length < headers.length) continue;
      const row = {
        date: cols[idx.Date],
        symbol: cols[idx.Symbol],
        open: parseFloat(cols[idx.Open]),
        high: parseFloat(cols[idx.High]),
        low: parseFloat(cols[idx.Low]),
        close: parseFloat(cols[idx.Close]),
        volume: parseFloat(cols[idx.Volume])
      };
      if (!row.symbol || !row.date) continue;
      if (![row.open,row.high,row.low,row.close,row.volume].every(Number.isFinite)) continue;

      symSet.add(row.symbol);
      dateSet.add(row.date);
      if (!data[row.symbol]) data[row.symbol] = {};
      data[row.symbol][row.date] = row;
    }

    this.symbols = Array.from(symSet).sort();
    this.dates = Array.from(dateSet).sort();
    this.stocksData = data;
  }

  // --------- feature helpers ----------
  mean(a, i0, i1){ let s=0,n=0; for(let i=i0;i<=i1;i++){ s+=a[i]; n++; } return s/Math.max(1,n); }
  stdev(a, i0, i1){ const m=this.mean(a,i0,i1); let s2=0,n=0; for(let i=i0;i<=i1;i++){ const d=a[i]-m; s2+=d*d; n++; } return Math.sqrt(s2/Math.max(1,n)); }

  rsi14(C){
    const n=C.length, out=new Array(n).fill(NaN);
    if (n<15) return out;
    let g=0,l=0;
    for (let i=1;i<=14;i++){ const d=C[i]-C[i-1]; if(d>0) g+=d; else l-=d; }
    let avgG=g/14, avgL=l/14;
    out[14] = avgL===0 ? 1 : 1 - 1/(1+avgG/avgL);
    for (let i=15;i<n;i++){
      const d=C[i]-C[i-1]; const G=d>0?d:0, L=d<0?-d:0;
      avgG=(avgG*13+G)/14; avgL=(avgL*13+L)/14;
      const rs=avgL===0?1e6:avgG/avgL;
      out[i]=1-1/(1+rs);
    }
    return out;
  }

  // --------- sequences ----------
  createSequences(sequenceLength = 12, predictionHorizon = 3) {
    if (!this.stocksData) throw new Error('No data loaded');

    // Build aligned OHLCV arrays per symbol
    const S = this.symbols.length;
    const N = this.dates.length;

    const O = Array.from({length:S},()=>[]);
    const H = Array.from({length:S},()=>[]);
    const L = Array.from({length:S},()=>[]);
    const C = Array.from({length:S},()=>[]);
    const V = Array.from({length:S},()=>[]);

    for (const d of this.dates) {
      for (let s=0;s<S;s++){
        const sym=this.symbols[s];
        const row=this.stocksData[sym][d];
        if (!row) throw new Error(`Missing ${sym} at ${d}`);
        O[s].push(row.open); H[s].push(row.high); L[s].push(row.low); C[s].push(row.close); V[s].push(row.volume);
      }
    }

    // Engineer features per stock (14 total)
    const F = this.featuresPerStock;
    const featRows = []; // [N][S*F]
    for (let t=0;t<N;t++){
      const row=[];
      for (let s=0;s<S;s++){
        const o=O[s][t], h=H[s][t], l=L[s][t], c=C[s][t];
        const vLog = Math.log(1+Math.max(0,V[s][t]));
        const prevC = t>0 ? C[s][t-1] : C[s][t];
        const ret1  = prevC!==0 ? (c/prevC - 1) : 0;
        const logR  = Math.log(Math.max(1e-8, c/prevC));
        const hlp   = c!==0 ? (h-l)/c : 0;
        const cop   = o!==0 ? (c-o)/o : 0;

        const mom5  = t>=4  ? (c/this.mean(C[s],t-4,t) - 1) : 0;
        const mom10 = t>=9  ? (c/this.mean(C[s],t-9,t) - 1) : 0;
        const rsi   = this.rsi14(C[s])[t] ?? 0.5;
        const bbp   = t>=19 ? (()=>{ const ma20=this.mean(C[s],t-19,t); const sd20=this.stdev(C[s],t-19,t)||1; return 0.5 + (c-ma20)/(2*sd20); })() : 0.5;

        // volatility of log returns last 10
        let vol10=0;
        if (t>=9){
          const logs=[];
          for (let k=t-9;k<=t;k++){ const p=C[s][k-1]??C[s][k]; logs.push(Math.log(Math.max(1e-8,C[s][k]/p))); }
          vol10=this.stdev(logs,0,logs.length-1)||0;
        }

        row.push(o,h,l,c,vLog,ret1,logR,hlp,cop,mom5,mom10,rsi,bbp,vol10);
      }
      featRows.push(row);
    }

    // Build samples (base index i). We’ll split windows chronologically (80/20)
    const earliest = Math.max(11, 20); // to allow indicators
    const latest   = N - 1 - predictionHorizon;
    const bases = [];
    for (let i=earliest; i<=latest; i++) bases.push(i);

    const split = Math.floor(bases.length * 0.8);
    const trainBases = bases.slice(0, split);
    const testBases  = bases.slice(split);

    // Train-only min–max (per stock, per feature)
    const mins = Array.from({length:S},()=>Array(F).fill(Number.POSITIVE_INFINITY));
    const maxs = Array.from({length:S},()=>Array(F).fill(Number.NEGATIVE_INFINITY));
    const lastTrainDay = trainBases[trainBases.length-1]; // last base used in train
    for (let t=0;t<=lastTrainDay;t++){
      const r=featRows[t];
      for (let s=0;s<S;s++){
        for (let f=0;f<F;f++){
          const v=r[s*F+f];
          if (v<mins[s][f]) mins[s][f]=v;
          if (v>maxs[s][f]) maxs[s][f]=v;
        }
      }
    }
    for (let s=0;s<S;s++) for (let f=0;f<F;f++){ if (!isFinite(mins[s][f])||!isFinite(maxs[s][f])||mins[s][f]===maxs[s][f]) { mins[s][f]=0; maxs[s][f]=1; } }
    const norm = (v,min,max)=> (v-min)/((max-min)||1);

    const build = (basesArr) => {
      const X=[]; const Y=[]; const baseDates=[];
      for (const i of basesArr){
        const seq=[];
        for (let t=i-sequenceLength+1; t<=i; t++){
          const raw=featRows[t]; const row=[];
          for (let s=0;s<S;s++) for (let f=0;f<F;f++) row.push(norm(raw[s*F+f], mins[s][f], maxs[s][f]));
          seq.push(row);
        }
        X.push(seq);

        // labels (+1,+2,+3)
        const y=[];
        for (let s=0;s<S;s++){
          const baseC = C[s][i];
          for (let h=1; h<=predictionHorizon; h++){
            const fut = C[s][i+h];
            y.push(fut>baseC ? 1 : 0);
          }
        }
        Y.push(y);
        baseDates.push(this.dates[i]);
      }
      return { X, Y, baseDates };
    };

    const tr = build(trainBases);
    const te = build(testBases);

    this.X_train = tf.tensor3d(tr.X);
    this.y_train = tf.tensor2d(tr.Y);
    this.X_test  = tf.tensor3d(te.X);
    this.y_test  = tf.tensor2d(te.Y);
    this.testDates = te.baseDates;

    return {
      X_train: this.X_train, y_train: this.y_train,
      X_test: this.X_test,   y_test: this.y_test,
      symbols: this.symbols, testDates: this.testDates,
      featuresPerStock: this.featuresPerStock
    };
  }

  dispose(){
    this.X_train?.dispose(); this.y_train?.dispose();
    this.X_test?.dispose();  this.y_test?.dispose();
  }
}
