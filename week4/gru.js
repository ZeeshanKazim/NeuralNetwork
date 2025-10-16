// gru.js
// Conv1D + BiGRU + GRU + Attention + Dense head (sigmoid).
// Early stopping supported.

export class GRUClassifier {
  constructor(cfg = {}) {
    const {
      seqLen = 48,
      featureDim = 100,
      numStocks = 10,
      horizons = [1,2,3],
      convFilters = 64,
      gruUnits = 96,
      denseUnits = 192,
      learningRate = 5e-4,
      dropout = 0.25
    } = cfg;

    this.seqLen = seqLen;
    this.featureDim = featureDim;
    this.numStocks = numStocks;
    this.horizons = horizons;
    this.outputDim = numStocks * horizons.length;

    this.model = this._build({ convFilters, gruUnits, denseUnits, learningRate, dropout });
  }

  _attentionOverTime(seqTensor) {
    // seqTensor: [B, T, F]
    const T = this.seqLen;
    // score_t = tanh(W*h_t + b) -> [B,T,1]
    let scores = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(seqTensor);
    // -> [B, T]
    scores = tf.layers.reshape({ targetShape: [T] }).apply(scores);
    // softmax over time -> [B, T]
    const weights = tf.layers.activation({ activation: 'softmax' }).apply(scores);
    // context = weights · seqTensor over time axis -> [B, F]
    const context = tf.layers.dot({ axes: [1, 1] }).apply([weights, seqTensor]);
    return context; // [B, F]
  }

  _build({ convFilters, gruUnits, denseUnits, learningRate, dropout }) {
    const input = tf.input({ shape: [this.seqLen, this.featureDim] });

    let x = tf.layers.conv1d({ filters: convFilters, kernelSize: 5, padding: 'same', activation: 'relu' }).apply(input);
    x = tf.layers.conv1d({ filters: convFilters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);

    x = tf.layers.bidirectional({
      layer: tf.layers.gru({ units: gruUnits, returnSequences: true, dropout: 0.15 }),
      mergeMode: 'concat'
    }).apply(x);

    // second GRU (sequence)
    x = tf.layers.gru({ units: gruUnits, returnSequences: true, dropout: 0.15 }).apply(x);

    // attention over time -> fixed length
    const ctx = this._attentionOverTime(x);

    // dense head
    let out = tf.layers.dense({ units: denseUnits, activation: 'relu' }).apply(ctx);
    out = tf.layers.dropout({ rate: dropout }).apply(out);
    out = tf.layers.dense({ units: this.outputDim, activation: 'sigmoid' }).apply(out);

    const model = tf.model({ inputs: input, outputs: out });
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return model;
  }

  async fit(X_train, y_train, X_val, y_val, trainOpts = {}, onEpochEnd) {
    const {
      epochs = 40,
      batchSize = 32,
      shuffle = false,
      patience = 6
    } = trainOpts;

    const cbs = [
      tf.callbacks.earlyStopping({ monitor: 'val_loss', patience, verbose: 0 })
    ];
    if (onEpochEnd) {
      cbs.push({ onEpochEnd: async (epoch, logs) => { onEpochEnd(epoch, logs); await tf.nextFrame(); } });
    }

    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle,
      validationData: [X_val, y_val],
      callbacks: cbs
    });
  }

  predict(X) { return this.model.predict(X); }

  dispose() { this.model?.dispose(); }

  async save(name = 'tfjs_gru_stock_demo') { await this.model.save(`downloads://${name}`); }
}
