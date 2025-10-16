// gru.js
// Drop-in replacement for your model.
// Changes:
//  - Stronger backbone (Conv1D + GRU stack + Dense)
//  - Manual early stopping (no callback API issues)
//  - Validation threshold tuning helpers
//  - **Specialist head** that fine-tunes one stock (DOV) on top of the encoder

class GRUModel {
  constructor(inputShape, outputSize, horizons = 3) {
    this.inputShape = inputShape;     // [seqLen, featureDim]
    this.outputSize = outputSize;     // stocks * horizons
    this.horizons = horizons;

    this.backbone = null;   // encoder model (inp -> latent)
    this.model = null;      // full classifier (inp -> outputSize)
    this.specialist = null; // DOV-only head (inp -> horizons)

    this.tunedThresholds = null;
  }

  buildBackbone() {
    const inp = tf.input({ shape: this.inputShape });

    let x = tf.layers.conv1d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(inp);
    x = tf.layers.dropout({ rate: 0.1 }).apply(x);
    x = tf.layers.gru({ units: 96, returnSequences: true, dropout: 0.1 }).apply(x);
    x = tf.layers.gru({ units: 64, returnSequences: false, dropout: 0.1 }).apply(x);
    const latent = tf.layers.dense({ units: 192, activation: 'relu' }).apply(x);

    this.backbone = tf.model({ inputs: inp, outputs: latent });
    return this.backbone;
  }

  buildModel() {
    if (!this.backbone) this.buildBackbone();
    const inp = this.backbone.inputs[0];
    let x = this.backbone.apply(inp);
    x = tf.layers.dropout({ rate: 0.25 }).apply(x);
    const out = tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }).apply(x);

    this.model = tf.model({ inputs: inp, outputs: out });
    this.model.compile({
      optimizer: tf.train.adam(7e-4),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return this.model;
  }

  async train(X_train, y_train, X_val, y_val, epochs = 50, batchSize = 32) {
    if (!this.model) this.buildModel();

    let best = Number.POSITIVE_INFINITY;
    let bestW = null, wait = 0, patience = 6, minDelta = 1e-4;

    await this.model.fit(X_train, y_train, {
      epochs, batchSize, shuffle: false, validationData: [X_val, y_val],
      callbacks: [ new tf.CustomCallback({
        onEpochEnd: async (epoch, logs) => {
          const p = ((epoch + 1) / epochs) * 100;
          const msg = `Epoch ${epoch+1}/${epochs} - loss ${logs.loss?.toFixed(4)} acc ${(logs.binaryAccuracy*100).toFixed(1)}% val ${logs.val_loss?.toFixed(4)}`;
          const bar = document.getElementById('trainingProgress');
          const st  = document.getElementById('status');
          if (bar) bar.value = p; if (st) st.textContent = msg;
          if (Number.isFinite(logs.val_loss) && logs.val_loss < best - minDelta) {
            best = logs.val_loss; wait = 0;
            if (bestW) bestW.forEach(w=>w.dispose());
            bestW = this.model.getWeights().map(w=>w.clone());
          } else {
            wait++;
            if (wait >= patience) this.model.stopTraining = true;
          }
          await tf.nextFrame();
        }
      })]
    });

    if (bestW) { const cloned=bestW.map(w=>w.clone()); this.model.setWeights(cloned); bestW.forEach(w=>w.dispose()); }
  }

  // Fine-tune a single stock (e.g., DOV) on top of frozen backbone
  async trainSpecialist(X_train, y_train, X_val, y_val, stockIndex, epochs = 30, batchSize = 32) {
    if (!this.backbone) this.buildBackbone();

    // Backbone frozen
    this.backbone.layers.forEach(l => l.trainable = false);

    const inp = this.backbone.inputs[0];
    let x = this.backbone.apply(inp);
    x = tf.layers.dropout({ rate: 0.25 }).apply(x);
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
    x = tf.layers.dropout({ rate: 0.25 }).apply(x);
    const out = tf.layers.dense({ units: this.horizons, activation: 'sigmoid' }).apply(x);

    this.specialist = tf.model({ inputs: inp, outputs: out });
    this.specialist.compile({ optimizer: tf.train.adam(1e-3), loss: 'binaryCrossentropy', metrics: ['binaryAccuracy'] });

    // Slice y for the target stock (columns stockIndex*h..stockIndex*h+h-1)
    const start = stockIndex * this.horizons;
    const yTr = y_train.slice([0, start], [-1, this.horizons]);
    const yVa = y_val.slice([0, start], [-1, this.horizons]);

    await this.specialist.fit(X_train, yTr, {
      epochs, batchSize, shuffle: false, validationData: [X_val, yVa],
      callbacks: [ new tf.CustomCallback({
        onEpochEnd: async (epoch, logs) => {
          const st = document.getElementById('status');
          if (st) st.textContent = `Specialist Epoch ${epoch+1}/${epochs} – loss ${logs.loss?.toFixed(4)} val ${logs.val_loss?.toFixed(4)}`;
          await tf.nextFrame();
        }
      })]
    });

    yTr.dispose(); yVa.dispose();
  }

  async predict(X) {
    if (!this.model) throw new Error('Model not trained');
    const base = this.model.predict(X); // [N, outDim]

    if (this.specialist && typeof this.specialist._stockIndex === 'number') {
      // If you later attach the index, you could splice; we’ll keep base-only here,
      // the app handles merging specialist output below.
    }
    return base;
  }

  // Merge base predictions with specialist for one stock
  async predictWithSpecialist(X, basePred, stockIndex) {
    if (!this.specialist) return basePred;
    const N = basePred.shape[0];
    const h = this.horizons;
    const start = stockIndex * h;

    const left  = start>0 ? basePred.slice([0,0],[N,start]) : null;
    const right = (start+h < basePred.shape[1]) ? basePred.slice([0,start+h],[N, basePred.shape[1]-start-h]) : null;
    const mid   = this.specialist.predict(X); // [N,h]

    const parts = [];
    if (left) parts.push(left);
    parts.push(mid);
    if (right) parts.push(right);

    const out = tf.concat(parts,1);
    basePred.dispose();
    if (left) left.dispose();
    if (right) right.dispose();
    return out;
  }

  // ---- evaluation helpers ----
  static tuneThresholds(valPred, yVal, lo=0.30, hi=0.70, step=0.01){
    const arrP = valPred.arraySync();
    const arrY = yVal.arraySync();
    const D = arrY[0].length;
    const N = arrY.length;
    const thr = new Array(D).fill(0.5);
    for (let j=0;j<D;j++){
      let best=-1, bestT=0.5;
      for (let t=lo;t<=hi+1e-12;t+=step){
        let c=0; for (let i=0;i<N;i++){ c += ((arrY[i][j]>=0.5) === (arrP[i][j]>=t)) ? 1:0; }
        const acc = c/N; if (acc>best){ best=acc; bestT=+t.toFixed(3); }
      }
      thr[j]=bestT;
    }
    return thr;
  }

  static perStockAccuracy(yTrue, yPred, symbols, horizons, thresholds=0.5){
    const arrY = yTrue.arraySync(), arrP = yPred.arraySync();
    const S = symbols.length, H = horizons;
    const out = {};
    for (let s=0;s<S;s++){
      let c=0,n=0;
      for (let i=0;i<arrY.length;i++){
        for (let h=0;h<H;h++){
          const j=s*H+h; const thr=Array.isArray(thresholds)?thresholds[j]:thresholds;
          const ok = (arrY[i][j]>=0.5) === (arrP[i][j]>=thr); c+=ok?1:0; n++;
        }
      }
      out[symbols[s]] = c/n;
    }
    return out;
  }

  dispose(){ this.model?.dispose(); this.backbone?.dispose(); this.specialist?.dispose(); }
}

export default GRUModel;
