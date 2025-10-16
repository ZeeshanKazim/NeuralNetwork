// gru.js
// Denoising Autoencoder (DAE) + GRU classifier (browser, TF.js).
// No unsupported callbacks -> avoids "setParams is not a function" error.
// Uses tf.CustomCallback for logging, and manual early-stopping with best-weight restore.

export class GRUDAEClassifier {
  constructor(cfg = {}) {
    const {
      seqLen = 12,
      featureDim = 140,           // MAX_STOCKS(10) × FEATURES_PER_STOCK(14)
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
    this.encoder = null;
    this.classifier = null;
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

    // Reuse encoder layers (by calling encoder on input)
    let x = this.encoder.apply(inp);
    // Optionally freeze encoder by marking its layers non-trainable
    this.encoder.layers.forEach(ly => ly.trainable = !freezeEncoder);

    // Head
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

  // ---------- Training ----------
  async pretrainAE(X_train, { epochs = 25, batchSize = 32, noiseStd = 0.08 }, onEpochEnd) {
    if (!this.ae) this._buildAE();

    // Manual per-epoch noise injection to avoid requiring special layers
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
          if (wait >= patience) {
            this.classifier.stopTraining = true;
          }
        }
        await tf.nextFrame();
      }
    });

    await this.classifier.fit(X_train, y_train, {
      epochs, batchSize, shuffle: false, validationData: [X_val, y_val], callbacks: [cb]
    });

    // Restore best weights
    if (bestWeights) {
      const cloned = bestWeights.map(w => w.clone());
      this.classifier.setWeights(cloned);
      bestWeights.forEach(w => w.dispose());
    }

    // Optional fine-tune: unfreeze encoder and run a few small-LR epochs
    this._buildClassifier(false, this.clsLR * 0.5); // unfreeze + lower LR, reuse same encoder graph
    await this.classifier.fit(X_train, y_train, {
      epochs: Math.max(4, Math.floor(epochs * 0.25)),
      batchSize,
      shuffle: false,
      validationData: [X_val, y_val],
      callbacks: [new tf.CustomCallback({ onEpochEnd: async (e, logs) => { if (onEpochEnd) onEpochEnd(e, logs, 'FT'); await tf.nextFrame(); } })]
    });
  }

  predict(X) {
    if (!this.classifier) throw new Error('Classifier is not built.');
    return this.classifier.predict(X);
  }

  async save(prefix='tfjs_gru_ae') {
    if (this.classifier) await this.classifier.save(`downloads://${prefix}_classifier`);
    if (this.ae)         await this.ae.save(`downloads://${prefix}_ae`);
  }

  dispose() {
    this.classifier?.dispose();
    this.encoder?.dispose();
    this.ae?.dispose();
  }
}
