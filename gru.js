class GRUStockPredictor {
    constructor() {
        this.model = null;
        this.history = null;
        this.isTraining = false;
        this.trainingCallback = null;
    }

    buildModel(inputShape, outputDim) {
        console.log('Building model with input shape:', inputShape, 'output dim:', outputDim);
        
        this.model = tf.sequential();
        
        // First GRU layer
        this.model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: inputShape
        }));
        
        // Second GRU layer
        this.model.add(tf.layers.gru({
            units: 32,
            returnSequences: false
        }));
        
        // Dropout for regularization
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer - 30 binary classifications (10 stocks × 3 days)
        this.model.add(tf.layers.dense({
            units: outputDim,
            activation: 'sigmoid'
        }));

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        console.log('Model built successfully');
        return this.model.summary();
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) {
            throw new Error('Model must be built before training');
        }

        this.isTraining = true;
        
        try {
            this.history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        if (this.trainingCallback) {
                            this.trainingCallback(epoch, logs);
                        }
                    }
                }
            });
        } catch (error) {
            this.isTraining = false;
            throw error;
        }

        this.isTraining = false;
        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model must be built before prediction');
        }
        return this.model.predict(X);
    }

    evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model must be built before evaluation');
        }
        return this.model.evaluate(X_test, y_test);
    }

    calculateStockAccuracies(predictions, y_test, symbols) {
        const predArray = predictions.arraySync();
        const trueArray = y_test.arraySync();
        const stocksCount = symbols.length;
        const daysAhead = 3;
        
        const accuracies = {};
        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            
            for (let sampleIdx = 0; sampleIdx < predArray.length; sampleIdx++) {
                for (let day = 0; day < daysAhead; day++) {
                    const predIdx = stockIdx + (day * stocksCount);
                    const prediction = predArray[sampleIdx][predIdx] > 0.5 ? 1 : 0;
                    const trueVal = trueArray[sampleIdx][predIdx];
                    
                    if (prediction === trueVal) {
                        correct++;
                    }
                    total++;
                }
            }
            
            accuracies[symbol] = total > 0 ? correct / total : 0;
        });
        
        return accuracies;
    }

    setTrainingCallback(callback) {
        this.trainingCallback = callback;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    async saveModel() {
        if (!this.model) return null;
        const saveResult = await this.model.save('indexeddb://stock-gru-model');
        return saveResult;
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://stock-gru-model');
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.warn('No saved model found:', error);
            return false;
        }
    }
}
