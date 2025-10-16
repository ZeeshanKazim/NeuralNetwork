// gru.js
// Enhanced GRU model with optional Conv1D front-end to improve accuracy.
// Based on the user's baseline class API; adds validation-aware threshold tuning. :contentReference[oaicite:3]{index=3}
/*
 Prompt allows changing model or adding CNN before GRU to extract features (page 24). :contentReference[oaicite:4]{index=4}
*/
export default class GRUModel {
  constructor(inputShape, outputSize, {
    units = [96, 64],
    dropout = 0.25,
    learningRate = 1e-3,
    bidirectional = true,
    convFilters = 32,
    convKernel = 3,
  } = {}) {
    this.model = null;
    this.inputShape = inputShape;   // [seqLen, featDim]
    this.outputSize = outputSize;   // 10 * horizon
    this.units = units;
    this.dropout = dropout;
    this.learningRate = learningRate;
    this.bidirectional = bidirectional;
    this.convFilters = convFilters;
    this.convKernel = convKernel;

    this.history = null;
    this.perStockThresholds = null;
  }

  buildModel() {
    const inp = tf.input({ shape: this.inputShape });

    // 1) Temporal conv front-end to extract short-range patterns
    let x = tf.layers.conv1d({
      filters: this.convFilters,
      kernelSize: this.convKernel,
      padding: 'same',
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-5 })
    }).apply(inp);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.dropout({ rate: Math.min(this.dropout, 0.5) }).apply(x);

    // 2) Stacked GRU (first layer optionally bidirectional)
    const gru1 = tf.layers.gru({ units: this.units[0], returnSequences: true, dropout: this.dropout * 0.6 });
    x = (this.bidirectional ? tf.layers.bidirectional({ layer: gru1, mergeMode: 'concat' }) : gru1).apply(x);

    x = tf.layers.gru({ units: this.units[1], returnSequences: false, dropout: this.dropout * 0.4 }).apply(x);

    // 3) Head
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
    x = tf.layers.dropout({ rate: Math.min(this.dropout, 0.5) }).apply(x);

    const out = tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }).apply(x);
    this.model = tf.model({ inputs: inp, outputs: out });

    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy'],
    });
    return this.model;
  }

  async train(X_train, y_train, X_val=null, y_val=null, epochs = 40, batchSize = 32, onEpoch = null) {
    if (!this.model) this.buildModel();

    const callbacks = [{
      onEpochEnd: async (epoch, logs) => {
        if (typeof onEpoch === 'function') onEpoch(epoch, logs);
        await tf.nextFrame();
      }
    }, tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: Math.max(2, Math.floor(epochs * 0.2)), restoreBestWeight: true })];

    const fitOpts = {
      epochs,
      batchSize,
      shuffle: false,
      validationData: X_val && y_val ? [X_val, y_val] : undefined,
      callbacks
    };

    this.history = await this.model.fit(X_train, y_train, fitOpts);
    return this.history;
  }

  async predict(X) {
    if (!this.model) throw new Error('Model not trained');
    return this.model.predict(X);
  }

  // Find per-stock thresholds on validation set to maximize accuracy
  tuneThresholds(yTrue, yPred, symbols, horizon = 3) {
    const T = yTrue.arraySync();
    const P = yPred.arraySync();
    const thresholds = {};
    const numStocks = symbols.length;

    const scan = (gt, pr) => {
      let bestTh = 0.5, bestAcc = -1;
      for (let t = 0.3; t <= 0.7; t += 0.02) {
        let ok = 0, tot = 0;
        for (let i = 0; i < gt.length; i++) {
          const y = gt[i] >= 0.5 ? 1 : 0;
          const p = pr[i] >= t ? 1 : 0;
          ok += (y === p) ? 1 : 0; tot++;
        }
        const acc = ok / Math.max(1, tot);
        if (acc > bestAcc) { bestAcc = acc; bestTh = t; }
      }
      return bestTh;
    };

    for (let s = 0; s < numStocks; s++) {
      const gtS = [], prS = [];
      for (let i = 0; i < T.length; i++) {
        for (let h = 0; h < horizon; h++) {
          const idx = s * horizon + h;
          gtS.push(T[i][idx]);
          prS.push(P[i][idx]);
        }
      }
      thresholds[symbols[s]] = scan(gtS, prS);
    }
    this.perStockThresholds = thresholds;
    return thresholds;
  }

  evaluatePerStock(yTrue, yPred, symbols, horizon = 3, thresholds = null) {
    const yTrueArray = yTrue.arraySync();
    const yPredArray = yPred.arraySync();
    const numStocks = symbols.length;

    const stockAccuracies = {};
    const stockPredictions = {};
    const confusions = {};

    symbols.forEach((symbol, stockIdx) => {
      let correct = 0, total = 0;
      const predictions = [];
      const th = thresholds?.[symbol] ?? 0.5;

      let TP=0,FP=0,TN=0,FN=0;

      for (let i = 0; i < yTrueArray.length; i++) {
        for (let offset = 0; offset < horizon; offset++) {
          const targetIdx = stockIdx * horizon + offset;
          const trueVal = yTrueArray[i][targetIdx] >= 0.5 ? 1 : 0;
          const predVal = yPredArray[i][targetIdx] >= th ? 1 : 0;

          predictions.push({ true: trueVal, pred: predVal, correct: trueVal === predVal, horizon: offset+1 });

          if (trueVal === predVal) correct++;
          total++;

          if (trueVal===1 && predVal===1) TP++;
          else if (trueVal===0 && predVal===1) FP++;
          else if (trueVal===0 && predVal===0) TN++;
          else if (trueVal===1 && predVal===0) FN++;
        }
      }
      stockAccuracies[symbol] = correct / Math.max(1,total);
      stockPredictions[symbol] = predictions;
      confusions[symbol] = {TP,FP,TN,FN};
    });

    return { stockAccuracies, stockPredictions, confusions };
  }

  async save(name = 'multi_stock_gru') { await this.model.save(`localstorage://${name}`); }
  static async load(name = 'multi_stock_gru') { return tf.loadLayersModel(`localstorage://${name}`); }

  dispose() { if (this.model) this.model.dispose(); }
}
