// gru.js
// TensorFlow.js GRU model for multi-output (10 stocks × 3-day horizon) binary classification.

export class GRUStockModel {
  constructor({
    seqLen = 12,
    featDim = 20,
    horizon = 3,
    numStocks = 10,
    units = [64, 32],
    dropout = 0.2,
    learningRate = 1e-3,
    bidirectional = true,
  } = {}) {
    this.seqLen = seqLen;
    this.featDim = featDim;
    this.horizon = horizon;
    this.numStocks = numStocks;
    this.outDim = numStocks * horizon;
    this.units = units;
    this.dropout = dropout;
    this.learningRate = learningRate;
    this.bidirectional = bidirectional;
    this.model = this._build();
  }

  _maybeBi(layer) {
    return this.bidirectional ? tf.layers.bidirectional({ layer, mergeMode: 'concat' }) : layer;
  }

  _build() {
    const inputs = tf.input({ shape: [this.seqLen, this.featDim] });

    // Stacked GRU (optionally bidirectional)
    const gru1 = this._maybeBi(tf.layers.gru({
      units: this.units[0],
      returnSequences: true,
      dropout: this.dropout,
      recurrentDropout: 0,
    })).apply(inputs);

    const gru2 = tf.layers.gru({
      units: this.units[1],
      returnSequences: false,
      dropout: this.dropout * 0.5,
      recurrentDropout: 0,
    }).apply(gru1);

    const dense1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(gru2);
    const drop1 = tf.layers.dropout({ rate: Math.min(this.dropout, 0.5) }).apply(dense1);

    const outputs = tf.layers.dense({
      units: this.outDim,
      activation: 'sigmoid', // multi-label binary
    }).apply(drop1);

    const model = tf.model({ inputs, outputs });
    model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy'],
    });
    return model;
  }

  summary() {
    this.model.summary();
  }

  async fit(X_train, y_train, {
    epochs = 25,
    batchSize = 32,
    validationSplit = 0.1,
    onEpoch = () => {},
  } = {}) {
    const es = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: Math.max(2, Math.floor(epochs * 0.2)),
      restoreBestWeight: true,
    });

    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit,
      shuffle: false, // respect chronology
      callbacks: [
        {
          onEpochEnd: async (epoch, logs) => {
            onEpoch(epoch, logs);
            await tf.nextFrame();
          },
        },
        es,
      ],
    });
    return history;
  }

  async predict(X) {
    return this.model.predict(X);
  }

  async evaluate(X_test, y_test) {
    const [loss, acc] = await this.model.evaluate(X_test, y_test, { batchSize: 64, verbose: 0 });
    const scalars = await Promise.all([loss.data(), acc.data()]);
    return { loss: scalars[0][0], acc: scalars[1][0] };
  }

  async save(name = 'multi_stock_gru') {
    await this.model.save(`localstorage://${name}`);
  }

  static async load(name = 'multi_stock_gru') {
    const model = await tf.loadLayersModel(`localstorage://${name}`);
    const m = new GRUStockModel();
    m.model = model;
    // Try to infer shapes from loaded model if possible
    const inShape = model.inputs?.[0]?.shape;
    const outUnits = model.outputs?.[0]?.shape?.[1];
    if (Array.isArray(inShape)) {
      m.seqLen = inShape[1] ?? m.seqLen;
      m.featDim = inShape[2] ?? m.featDim;
    }
    if (typeof outUnits === 'number') {
      m.outDim = outUnits;
      m.numStocks = 10;
      m.horizon = outUnits / m.numStocks;
    }
    return m;
  }
}

/** Utility: compute per-stock metrics and timelines (on arrays, not tensors) */
export function computePerStockMetrics({
  yTrue, // Array [N][numStocks*horizon]
  yPred, // Array [N][numStocks*horizon] values 0..1
  symbols,
  horizon = 3,
  threshold = 0.5,
}) {
  const numStocks = symbols.length;
  const N = yTrue.length;
  const correctCounts = new Array(numStocks).fill(0);
  const totalCounts = new Array(numStocks).fill(0);
  const confusions = {}; // {SYM: {TP,FP,TN,FN}}
  const timelines = {};  // {SYM: [ [bool*N] * horizon ]}

  for (let s = 0; s < numStocks; s++) {
    confusions[symbols[s]] = { TP: 0, FP: 0, TN: 0, FN: 0 };
    timelines[symbols[s]] = Array.from({ length: horizon }, () => new Array(N).fill(false));
  }

  for (let i = 0; i < N; i++) {
    for (let s = 0; s < numStocks; s++) {
      for (let h = 0; h < horizon; h++) {
        const idx = s * horizon + h;
        const gt = yTrue[i][idx] > 0.5 ? 1 : 0;
        const pr = yPred[i][idx] >= threshold ? 1 : 0;
        timelines[symbols[s]][h][i] = (gt === pr);
        correctCounts[s] += (gt === pr) ? 1 : 0;
        totalCounts[s] += 1;

        if (gt === 1 && pr === 1) confusions[symbols[s]].TP++;
        else if (gt === 0 && pr === 1) confusions[symbols[s]].FP++;
        else if (gt === 0 && pr === 0) confusions[symbols[s]].TN++;
        else if (gt === 1 && pr === 0) confusions[symbols[s]].FN++;
      }
    }
  }

  const accuracies = correctCounts.map((c, s) => c / Math.max(1, totalCounts[s]));
  return { accuracies, confusions, timelines };
}
