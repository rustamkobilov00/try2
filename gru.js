// gru.js
// ES6 module: defines and trains a multi-output GRU network with tf.js

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js';

export class GRUModel {
  constructor(inputShape=[12,20], outputSize=30, units=64) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.units = units;
    this.model = this.buildModel();
  }

  buildModel() {
    const model = tf.sequential();
    model.add(tf.layers.gru({
      units: this.units,
      inputShape: this.inputShape,
      returnSequences: true,
      activation: 'tanh'
    }));
    model.add(tf.layers.gru({
      units: Math.floor(this.units/2),
      returnSequences: false,
      activation: 'tanh'
    }));
    model.add(tf.layers.dense({
      units: this.outputSize,
      activation: 'sigmoid'
    }));
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
    return model;
  }

  async fit(X, y, {epochs=30, batchSize=32, validationSplit=0.1, callbacks}={}) {
    if(this.model == null) throw new Error('GRU model not initialized');
    return await this.model.fit(X, y, {
      epochs,
      batchSize,
      validationSplit,
      shuffle:true,
      callbacks
    });
  }

  async predict(X) {
    if(this.model == null) throw new Error('GRU model not initialized');
    return this.model.predict(X);
  }

  async evaluate(X_test, y_test) {
    const evalOut = await this.model.evaluate(X_test, y_test, {batchSize:32});
    return {
      loss: evalOut[0].dataSync()[0],
      acc: evalOut[1].dataSync()[0]
    };
  }

  // Compute binary accuracy per stock (averaged across 3 output days)
  async accuracyByStock(X_test, y_test, symbolNames) {
    const preds = await this.model.predict(X_test);
    const y_pred = preds.arraySync();
    const y_true = y_test.arraySync();
    const nStocks = symbolNames.length;
    const nSteps = 3;
    const byStock = [];
    for(let s=0; s<nStocks; s++) {
      let correct = 0, total = 0;
      for(let i=0; i<y_pred.length; i++) {
        for(let t=0; t<nSteps; t++) {
          const idx = s*nSteps + t;
          const gt = y_true[i][idx];
          const pr = y_pred[i][idx] >= 0.5 ? 1 : 0;
          if(gt === pr) correct++;
          total++;
        }
      }
      byStock.push({symbol:symbolNames[s], acc: correct/total});
    }
    byStock.sort((a,b)=>b.acc-a.acc);
    return {perStock: byStock, y_true, y_pred};
  }

  async saveModel(name='gru-stock-model') {
    return await this.model.save(`localstorage://${name}`);
  }
  async loadModel(name='gru-stock-model') {
    this.model = await tf.loadLayersModel(`localstorage://${name}`);
  }
  dispose() {
    this.model?.dispose();
  }
}
