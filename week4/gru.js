// gru.js
// Conv1D feature extractor + BiGRU + GRU + Dense head (sigmoid).
// Manual early stopping implemented via a CustomCallback to avoid callback API issues.

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

  _build({ convFilters, gruUnits, denseUnits, learningRate, dropout }) {
    const input = tf.input({ shape: [this.seqLen, this.featureDim] });

    let x = tf.layers.conv1d({ filters: convFilters, kernelSize: 5, padding: 'same', activation: 'relu' }).apply(input);
    x = tf.layers.conv1d({ filters: convFilters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);

    x = tf.layers.bidirectional({
      layer: tf.layers.gru({ units: gruUnits, returnSequences: true, dropout: 0.15 }),
      mergeMode: 'concat'
    }).apply(x);

    x = tf.layers.gru({ units: gruUnits, returnSequences: false, dropout: 0.15 }).apply(x);

    x = tf.layers.dense({ units: denseUnits, activation: 'relu' }).apply(x);
    x = tf.layers.dropout({ rate: dropout }).apply(x);

    const output = tf.layers.dense({ units: this.outputDim, activation: 'sigmoid' }).apply(x);

    const model = tf.model({ inputs: input, outputs: output });
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return model;
  }

  async fit(X_train, y_train, X_val, y_val, opts = {}, onEpochEnd) {
    const {
      epochs = 40,
      batchSize = 32,
      shuffle = false,
      patience = 6,
      minDelta = 1e-4
    } = opts;

    let bestVal = Number.POSITIVE_INFINITY;
    let bestWeights = null;
    let wait = 0;
    const self = this;

    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle,
      validationData: [X_val, y_val],
      callbacks: {
        onEpochEnd: async function(epoch, logs) {
          if (onEpochEnd) onEpochEnd(epoch, logs);

          const v = logs.val_loss;
          if (Number.isFinite(v) && v < bestVal - minDelta) {
            bestVal = v;
            wait = 0;
            if (bestWeights) bestWeights.forEach(w => w.dispose());
            bestWeights = self.model.getWeights().map(w => w.clone());
          } else {
            wait++;
            if (wait >= patience) {
              self.model.stopTraining = true;
              if (bestWeights) {
                const cloned = bestWeights.map(w => w.clone());
                self.model.setWeights(cloned);
                bestWeights.forEach(w => w.dispose());
                bestWeights = null;
              }
            }
          }
          await tf.nextFrame();
        }
      }
    });

    // Ensure best weights restored at the end if training didn't early-stop.
    if (bestWeights) {
      const cloned = bestWeights.map(w => w.clone());
      this.model.setWeights(cloned);
      bestWeights.forEach(w => w.dispose());
      bestWeights = null;
    }

    return history;
  }

  predict(X) { return this.model.predict(X); }

  async save(name = 'tfjs_gru_stock_demo') { await this.model.save(`downloads://${name}`); }

  dispose() { if (this.model) this.model.dispose(); }
}
