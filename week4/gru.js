// gru.js
// GRU-based multi-output classifier using TensorFlow.js (client-side, browser)

export class GRUClassifier {
  constructor(config = {}) {
    const {
      seqLen = 12,
      featureDim = 20,
      numStocks = 10,
      horizons = [1, 2, 3],
      units = 64,
      learningRate = 1e-3,
    } = config;

    this.seqLen = seqLen;
    this.featureDim = featureDim;
    this.numStocks = numStocks;
    this.horizons = horizons;
    this.outputDim = numStocks * horizons.length;

    this.model = this._build(units, learningRate);
  }

  _build(units, lr) {
    const input = tf.input({ shape: [this.seqLen, this.featureDim] });

    let x = tf.layers.gru({
      units,
      returnSequences: true,
      dropout: 0.1,
      recurrentDropout: 0.0,
      kernelInitializer: 'glorotUniform'
    }).apply(input);

    x = tf.layers.gru({
      units: Math.max(32, Math.floor(units * 0.75)),
      returnSequences: false,
      dropout: 0.1,
      recurrentDropout: 0.0,
      kernelInitializer: 'glorotUniform'
    }).apply(x);

    x = tf.layers.dropout({ rate: 0.2 }).apply(x);

    const output = tf.layers.dense({
      units: this.outputDim,
      activation: 'sigmoid',
      kernelInitializer: 'glorotUniform'
    }).apply(x);

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return model;
  }

  async fit(X_train, y_train, options = {}, onEpochEnd) {
    const {
      epochs = 25,
      batchSize = 32,
      validationSplit = 0.1,
      shuffle = false // chronological
    } = options;

    return await this.model.fit(X_train, y_train, {
      epochs, batchSize, validationSplit, shuffle,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpochEnd) onEpochEnd(epoch, logs);
          await tf.nextFrame();
        }
      }
    });
  }

  predict(X) {
    return this.model.predict(X);
  }

  async evaluateOverallAccuracy(X, yTrue, threshold = 0.5) {
    const yPred = this.predict(X);
    const preds = await yPred.greaterEqual(threshold).toInt().array();
    const truth = await yTrue.toInt().array();
    yPred.dispose();

    let correct = 0, total = 0;
    for (let i = 0; i < truth.length; i++) {
      const t = truth[i], p = preds[i];
      for (let j = 0; j < t.length; j++) {
        if (t[j] === p[j]) correct++;
        total++;
      }
    }
    return correct / Math.max(1, total);
  }

  async save(name = 'tfjs_gru_stock_demo') {
    await this.model.save(`downloads://${name}`);
  }

  dispose() {
    if (this.model) this.model.dispose();
  }
}
