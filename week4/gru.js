// gru.js
// Enhanced architecture for higher accuracy:
// Conv1D feature extractor -> Bidirectional GRU -> GRU -> Dense head.
// Loss: binaryCrossentropy; metrics: binaryAccuracy.

export class GRUClassifier {
  constructor(config = {}) {
    const {
      seqLen = 24,
      featureDim = 100,            // 10 stocks × 10 features (auto from data)
      numStocks = 10,
      horizons = [1, 2, 3],
      convFilters = 64,
      gruUnits = 64,
      denseUnits = 128,
      learningRate = 8e-4,
      dropout = 0.25
    } = config;

    this.seqLen = seqLen;
    this.featureDim = featureDim;
    this.numStocks = numStocks;
    this.horizons = horizons;
    this.outputDim = numStocks * horizons.length;

    this.model = this._build({ convFilters, gruUnits, denseUnits, learningRate, dropout });
  }

  _build({ convFilters, gruUnits, denseUnits, learningRate, dropout }) {
    const input = tf.input({ shape: [this.seqLen, this.featureDim] });

    // Temporal feature extractor
    let x = tf.layers.conv1d({
      filters: convFilters,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'glorotUniform'
    }).apply(input);

    x = tf.layers.conv1d({
      filters: convFilters,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'glorotUniform'
    }).apply(x);

    // Sequence modeling
    x = tf.layers.bidirectional({
      layer: tf.layers.gru({
        units: gruUnits,
        returnSequences: true,
        dropout: 0.15,
        recurrentDropout: 0.0,
        kernelInitializer: 'glorotUniform'
      }),
      mergeMode: 'concat'
    }).apply(x);

    x = tf.layers.gru({
      units: gruUnits,
      returnSequences: false,
      dropout: 0.15,
      recurrentDropout: 0.0,
      kernelInitializer: 'glorotUniform'
    }).apply(x);

    // Dense head
    x = tf.layers.dense({ units: denseUnits, activation: 'relu' }).apply(x);
    x = tf.layers.dropout({ rate: dropout }).apply(x);

    const output = tf.layers.dense({
      units: this.outputDim,
      activation: 'sigmoid',
      kernelInitializer: 'glorotUniform'
    }).apply(x);

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return model;
  }

  async fit(X_train, y_train, options = {}, onEpochEnd) {
    const {
      epochs = 35,
      batchSize = 32,
      validationSplit = 0.12,
      shuffle = false
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

  async save(name = 'tfjs_gru_stock_demo') {
    await this.model.save(`downloads://${name}`);
  }

  dispose() {
    if (this.model) this.model.dispose();
  }
}
