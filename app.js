class StockPredictionApp {
    constructor() {
        this.dataLoader = new StockDataLoader();
        this.model = new GRUStockPredictor();
        this.isInitialized = false;
        this.predictions = null;
        this.accuracies = null;
        this.accuracyChart = null;
        this.timelineChart = null;
        
        this.initializeUI();
    }

    initializeUI() {
        // File input handler
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Train button handler
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.trainModel();
        });

        // Evaluate button handler
        document.getElementById('evaluateBtn').addEventListener('click', () => {
            this.evaluateModel();
        });

        // Load model button handler
        document.getElementById('loadModelBtn').addEventListener('click', () => {
            this.loadSavedModel();
        });

        console.log('UI initialized');
    }

    async handleFileUpload(file) {
        if (!file) return;
        
        try {
            this.updateStatus('Loading CSV data...');
            await this.dataLoader.loadCSV(file);
            this.updateStatus('Data loaded successfully!');
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            this.updateStatus('Error loading file: ' + error.message, true);
            console.error('File loading error:', error);
        }
    }

    async trainModel() {
        try {
            const data = this.dataLoader.getData();
            const { X_train, y_train, X_test, y_test, symbols } = data;
            
            this.updateStatus('Building model...');
            this.model.buildModel([12, symbols.length * 2], symbols.length * 3);
            
            // Set up training progress updates
            this.model.setTrainingCallback((epoch, logs) => {
                this.updateTrainingProgress(epoch, logs);
            });

            this.updateStatus('Training model... (this may take a few minutes)');
            await this.model.train(X_train, y_train, X_test, y_test, 30, 32);
            
            // Save model automatically
            await this.model.saveModel();
            
            this.updateStatus('Training completed!');
            document.getElementById('evaluateBtn').disabled = false;
            
        } catch (error) {
            this.updateStatus('Training error: ' + error.message, true);
            console.error('Training error:', error);
        }
    }

    async evaluateModel() {
        try {
            const { X_test, y_test, symbols } = this.dataLoader.getData();
            
            this.updateStatus('Making predictions...');
            this.predictions = await this.model.predict(X_test);
            
            this.updateStatus('Calculating accuracies...');
            this.accuracies = this.model.calculateStockAccuracies(this.predictions, y_test, symbols);
            
            this.visualizeResults();
            this.updateStatus('Evaluation completed!');
            
        } catch (error) {
            this.updateStatus('Evaluation error: ' + error.message, true);
            console.error('Evaluation error:', error);
        }
    }

    async loadSavedModel() {
        try {
            this.updateStatus('Loading saved model...');
            const success = await this.model.loadModel();
            
            if (success) {
                this.updateStatus('Model loaded successfully!');
                document.getElementById('evaluateBtn').disabled = false;
            } else {
                this.updateStatus('No saved model found. Please train a new model.');
            }
        } catch (error) {
            this.updateStatus('Error loading model: ' + error.message, true);
            console.error('Model loading error:', error);
        }
    }

    visualizeResults() {
        if (!this.accuracies) return;
        
        this.createAccuracyChart();
        this.createPredictionTimeline();
    }

    createAccuracyChart() {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        // Sort stocks by accuracy
        const sortedEntries = Object.entries(this.accuracies)
            .sort(([, a], [, b]) => b - a);
        
        const symbols = sortedEntries.map(([symbol]) => symbol);
        const accuracyValues = sortedEntries.map(([, accuracy]) => accuracy * 100);
        
        // Clear previous chart
        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }
        
        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: symbols,
                datasets: [{
                    label: 'Accuracy (%)',
                    data: accuracyValues,
                    backgroundColor: accuracyValues.map(acc => 
                        acc > 50 ? 'rgba(75, 192, 192, 0.8)' : 'rgba(255, 99, 132, 0.8)'
                    ),
                    borderColor: accuracyValues.map(acc => 
                        acc > 50 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Prediction Accuracy (Sorted)'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    }

    createPredictionTimeline() {
        const ctx = document.getElementById('timelineChart').getContext('2d');
        const { symbols } = this.dataLoader.getData();
        
        if (symbols.length === 0) return;
        
        // For demonstration, show first 50 test samples for the best stock
        const bestStock = Object.entries(this.accuracies)
            .sort(([, a], [, b]) => b - a)[0][0];
        
        const stockIndex = symbols.indexOf(bestStock);
        const predData = this.predictions.arraySync();
        const trueData = this.dataLoader.y_test.arraySync();
        
        const sampleCount = Math.min(50, predData.length);
        const correctness = [];
        
        for (let i = 0; i < sampleCount; i++) {
            // Check next day prediction for the best stock
            const pred = predData[i][stockIndex] > 0.5 ? 1 : 0;
            const trueVal = trueData[i][stockIndex];
            correctness.push(pred === trueVal ? 1 : 0);
        }
        
        // Clear previous chart
        if (this.timelineChart) {
            this.timelineChart.destroy();
        }
        
        this.timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: sampleCount}, (_, i) => `Sample ${i + 1}`),
                datasets: [{
                    label: `Prediction Correctness - ${bestStock}`,
                    data: correctness,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: correctness.map(correct => 
                        correct ? 'rgba(75, 192, 192, 0.2)' : 'rgba(255, 99, 132, 0.2)'
                    ),
                    pointBackgroundColor: correctness.map(correct => 
                        correct ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                    ),
                    pointBorderColor: correctness.map(correct => 
                        correct ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                    ),
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `Prediction Timeline - ${bestStock} (Green=Correct, Red=Wrong)`
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value === 1 ? 'Correct' : 'Wrong';
                            }
                        }
                    }
                }
            }
        });
    }

    updateStatus(message, isError = false) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = isError ? 'status error' : 'status';
        
        console.log(message);
    }

    updateTrainingProgress(epoch, logs) {
        const progressElement = document.getElementById('trainingProgress');
        if (logs) {
            progressElement.textContent = 
                `Epoch ${epoch + 1} - Loss: ${logs.loss?.toFixed(4) || 'N/A'}, ` +
                `Accuracy: ${logs.acc?.toFixed(4) || 'N/A'}, ` +
                `Val Loss: ${logs.val_loss?.toFixed(4) || 'N/A'}, ` +
                `Val Accuracy: ${logs.val_acc?.toFixed(4) || 'N/A'}`;
        }
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
        if (this.predictions) {
            tf.dispose(this.predictions);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.stockApp = new StockPredictionApp();
});
