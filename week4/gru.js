// gru.js — Denoising AE + GRU classifier + **per-stock specialist head**
// Keeps the clean callback usage (tf.CustomCallback), avoids setParams issues.

export class GRUDAEClassifier {
  constructor(cfg = {}) {
    const {
      seqLen = 12,
      featureDim = 140,           // 10 stocks × 14 features
      numStocks = 10,
      horizons = [1,2,3],
      latentDim = 128,
      encoderGRU = 96,
      denseHead = 192,
      aeLR = 1e-3,
      clsLR = 7e-4,
      dropout = 0.25
    } = cfg;

    this.seqLen = seqLen;
    this.featureDim = featureDim;
    this.numStocks = numStocks;
    this.horizons = horizons;
    this.outDim = numStocks * horizons.length;

    this.latentDim = latentDim;
    this.encoderGRU = encoderGRU;
    this.denseHead = denseHead;
    this.aeLR = aeLR;
    this.clsLR = clsLR;
    this.dropout = dropout;

    this.ae = null;
    this.encoder = null;     // inp -> latent
    this.classifier = null;  // inp -> outDim
    this.stockHead = null;   // specialist head (inp -> H) for one stock
    this.stockIndex = null;  // which stock the specialist targets
  }

  // ---------- Build models ----------
  _buildAE() {
    const inp = tf.input({ shape: [this.seqLen, this.featureDim] });
    // Encoder
    let x = tf.layers.conv1d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(inp);
    x = tf.layers.gru({ units: this.encoderGRU, returnSequences: false, dropout: 0.1 }).apply(x);
    const code = tf.layers.dense({ units: this.latentDim, activation: 'relu' }).apply(x);

    // Decoder
    let z = tf.layers.dense({ units: this.seqLen * this.encoderGRU, activation: 'relu' }).apply(code);
    z = tf.layers.reshape({ targetShape: [this.seqLen, this.encoderGRU] }).apply(z);
    z = tf.layers.gru({ units: this.encoderGRU, returnSequences: true, dropout: 0.1 }).apply(z);
    const recon = tf.layers.dense({ units: this.featureDim, activation: 'sigmoid' }).apply(z);

    this.ae = tf.model({ inputs: inp, outputs: recon });
    this.ae.compile({ optimizer: tf.train.adam(this.aeLR), loss: 'meanSquaredError' });

    // Separate encoder model (inp -> code)
    this.encoder = tf.model({ inputs: inp, outputs: code });
  }

  _buildClassifier(freezeEncoder = true, lr = this.clsLR) {
    if (!this.encoder) throw new Error('Encoder not built. Call pretrainAE first.');
    const inp = tf.input({ shape: [this.seqLen, this.featureDim] });

    let x = this.encoder.apply(inp);
    this.encoder.layers.forEach(ly => ly.trainable = !freezeEncoder);

    x = tf.layers.dropout({ rate: this.dropout }).apply(x);
    x = tf.layers.dense({ units: this.denseHead, activation: 'relu' }).apply(x);
    x = tf.layers.dropout({ rate: this.dropout }).apply(x);
    const out = tf.layers.dense({ units: this.outDim, activation: 'sigmoid' }).apply(x);

    this.classifier = tf.model({ inputs: inp, outputs: out });
    this.classifier.compile({
      optimizer: tf.train.adam(lr),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
  }

  // ---------- AE pretraining with noise ----------
  async pretrainAE(X_train, { epochs = 25, batchSize = 32, noiseStd = 0.08 }, onEpochEnd) {
    if (!this.ae) this._buildAE();
    for (let e = 0; e < epochs; e++) {
      const noise = tf.randomNormal(X_train.shape, 0, noiseStd);
      const noisy = tf.clipByValue(tf.add(X_train, noise), 0, 1);
      const cb = new tf.CustomCallback({
        onEpochEnd: async (_epoch, logs) => {
          if (onEpochEnd) onEpochEnd(e, logs, 'AE');
          await tf.nextFrame();
        }
      });
      await this.ae.fit(noisy, X_train, { epochs: 1, batchSize, shuffle: false, callbacks: [cb] });
      noise.dispose(); noisy.dispose();
      await tf.nextFrame();
    }
  }

  // ---------- Classifier training (freeze → unfreeze) ----------
  async fitClassifier(X_train, y_train, X_val, y_val, { epochs = 40, batchSize = 32, freezeEncoder = true, patience = 6, minDelta = 1e-4 }, onEpochEnd) {
    if (!this.classifier) this._buildClassifier(freezeEncoder, this.clsLR);

    let bestVal = Number.POSITIVE_INFINITY;
    let bestWeights = null;
    let wait = 0;

    const cb = new tf.CustomCallback({
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) onEpochEnd(epoch, logs, 'CLS');
        const v = logs.val_loss;
        if (Number.isFinite(v) && v < bestVal - minDelta) {
          bestVal = v; wait = 0;
          if (bestWeights) bestWeights.forEach(w => w.dispose());
          bestWeights = this.classifier.getWeights().map(w => w.clone());
        } else {
          wait++;
          if (wait >= patience) this.classifier.stopTraining = true;
        }
        await tf.nextFrame();
      }
    });

    await this.classifier.fit(X_train, y_train, {
      epochs, batchSize, shuffle: false, validationData: [X_val, y_val], callbacks: [cb]
    });

    if (bestWeights) {
      const cloned = bestWeights.map(w => w.clone());
      this.classifier.setWeights(cloned);
      bestWeights.forEach(w => w.dispose());
    }

    // light fine-tune (unfreeze)
    this._buildClassifier(false, this.clsLR * 0.5);
    await this.classifier.fit(X_train, y_train, {
      epochs: Math.max(4, Math.floor(epochs * 0.25)),
      batchSize,
      shuffle: false,
      validationData: [X_val, y_val],
      callbacks: [new tf.CustomCallback({ onEpochEnd: async (e, logs) => { if (onEpochEnd) onEpochEnd(e, logs, 'FT'); await tf.nextFrame(); } })]
    });
  }

  // ---------- NEW: per-stock specialist head (e.g., DOV) ----------
  async fitStockHead(X_train, y_train, X_val, y_val, stockIndex, {
    epochs = 28, batchSize = 32, lr = 1e-3, hidden = 128, patience = 5, minDelta = 1e-4
  } = {}, onEpochEnd) {
    if (!this.encoder) throw new Error('Encoder not ready.');
    this.stockIndex = stockIndex;

    // Build specialist head on top of the shared encoder (frozen)
    const inp = tf.input({ shape: [this.seqLen, this.featureDim] });
    const latent = this.encoder.apply(inp);
    this.encoder.layers.forEach(ly => ly.trainable = false);

    let h = tf.layers.dropout({ rate: this.dropout }).apply(latent);
    h = tf.layers.dense({ units: hidden, activation: 'relu' }).apply(h);
    h = tf.layers.dropout({ rate: this.dropout }).apply(h);

    const H = this.horizons.length;
    const out = tf.layers.dense({ units: H, activation: 'sigmoid' }).apply(h);

    this.stockHead = tf.model({ inputs: inp, outputs: out });
    this.stockHead.compile({ optimizer: tf.train.adam(lr), loss: 'binaryCrossentropy', metrics: ['binaryAccuracy'] });

    // Slice y for the target stock (columns s*H .. s*H+H-1)
    const start = stockIndex * H;
    const yTr = y_train.slice([0, start], [-1, H]);
    const yVa = y_val.slice([0, start], [-1, H]);

    let bestVal = Number.POSITIVE_INFINITY;
    let bestW = null, wait = 0;

    const cb = new tf.CustomCallback({
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) onEpochEnd(epoch, logs, 'SPC');
        const v = logs.val_loss;
        if (Number.isFinite(v) && v < bestVal - minDelta) {
          bestVal = v; wait = 0;
          if (bestW) bestW.forEach(w => w.dispose());
          bestW = this.stockHead.getWeights().map(w => w.clone());
        } else {
          wait++;
          if (wait >= patience) this.stockHead.stopTraining = true;
        }
        await tf.nextFrame();
      }
    });

    await this.stockHead.fit(X_train, yTr, {
      epochs, batchSize, shuffle: false, validationData: [X_val, yVa], callbacks: [cb]
    });

    if (bestW) {
      const cloned = bestW.map(w => w.clone());
      this.stockHead.setWeights(cloned);
      bestW.forEach(w => w.dispose());
    }

    yTr.dispose(); yVa.dispose();
  }

  // Predict with optional specialist override (replaces that stock’s 3 columns)
  predict(X) {
    const base = this.classifier.predict(X);                  // [N, outDim]
    if (!this.stockHead || this.stockIndex == null) return base;

    const N = base.shape[0];
    const H = this.horizons.length;
    const start = this.stockIndex * H;

    const left  = start > 0 ? base.slice([0,0],[N,start]) : null;
    const rightLen = this.outDim - (start + H);
    const right = rightLen > 0 ? base.slice([0,start+H],[N,rightLen]) : null;
    const mid   = this.stockHead.predict(X);                  // [N, H]

    const parts = [];
    if (left)  parts.push(left);
    parts.push(mid);
    if (right) parts.push(right);

    const out = tf.concat(parts, 1);
    base.dispose();
    if (left) left.dispose();
    if (right) right.dispose();
    // mid is kept till concat copies; safe to let GC reclaim after return or explicitly dispose here:
    // (do not dispose mid before concat)
    return out;
  }

  async save(prefix='tfjs_gru_ae'){ if (this.classifier) await this.classifier.save(`downloads://${prefix}_classifier`); if (this.ae) await this.ae.save(`downloads://${prefix}_ae`); if (this.stockHead) await this.stockHead.save(`downloads://${prefix}_specialist`); }
  dispose(){ this.classifier?.dispose(); this.encoder?.dispose(); this.ae?.dispose(); this.stockHead?.dispose(); }
}
