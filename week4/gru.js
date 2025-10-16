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
        for (let i = 0; i < gt.l
