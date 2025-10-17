// Remove import statement since we're loading tf globally
class StockDataLoader {
    constructor() {
        this.rawData = [];
        this.normalizedData = {};
        this.symbols = [];
        this.dates = [];
        this.X = null;
        this.y = null;
        this.trainTestSplit = 0.8;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve();
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        this.rawData = lines.slice(1).map(line => {
            const values = line.split(',').map(v => v.trim());
            const entry = {};
            headers.forEach((header, idx) => {
                entry[header] = values[idx];
            });
            return entry;
        });

        console.log('Parsed', this.rawData.length, 'records');
        this.prepareData();
    }

    prepareData() {
        // Extract unique symbols and dates
        this.symbols = [...new Set(this.rawData.map(d => d.Symbol))].sort();
        this.dates = [...new Set(this.rawData.map(d => d.Date))].sort();
        
        console.log('Found symbols:', this.symbols);
        console.log('Found dates:', this.dates.length);

        // Pivot data: date -> symbol -> {Open, Close}
        const pivoted = {};
        this.dates.forEach(date => {
            pivoted[date] = {};
            this.symbols.forEach(symbol => {
                const record = this.rawData.find(d => d.Date === date && d.Symbol === symbol);
                if (record) {
                    pivoted[date][symbol] = {
                        Open: parseFloat(record.Open),
                        Close: parseFloat(record.Close)
                    };
                }
            });
        });

        this.normalizeData(pivoted);
        this.createSequences();
    }

    normalizeData(pivoted) {
        // Calculate min-max per stock
        const stockStats = {};
        this.symbols.forEach(symbol => {
            const opens = this.dates.map(d => pivoted[d]?.[symbol]?.Open).filter(Boolean);
            const closes = this.dates.map(d => pivoted[d]?.[symbol]?.Close).filter(Boolean);
            
            if (opens.length > 0 && closes.length > 0) {
                stockStats[symbol] = {
                    openMin: Math.min(...opens),
                    openMax: Math.max(...opens),
                    closeMin: Math.min(...closes),
                    closeMax: Math.max(...closes)
                };
            }
        });

        // Normalize data
        this.normalizedData = {};
        this.dates.forEach(date => {
            this.normalizedData[date] = {};
            this.symbols.forEach(symbol => {
                const data = pivoted[date]?.[symbol];
                if (data && stockStats[symbol]) {
                    const stats = stockStats[symbol];
                    this.normalizedData[date][symbol] = {
                        Open: (data.Open - stats.openMin) / (stats.openMax - stats.openMin),
                        Close: (data.Close - stats.closeMin) / (stats.closeMax - stats.closeMin)
                    };
                }
            });
        });
    }

    createSequences() {
        const sequenceLength = 12;
        const predictionHorizon = 3;
        const featuresPerStock = 2; // Open, Close
        const totalFeatures = this.symbols.length * featuresPerStock;

        const sequences = [];
        const targets = [];

        for (let i = 0; i < this.dates.length - sequenceLength - predictionHorizon; i++) {
            const sequence = [];
            const currentDate = this.dates[i + sequenceLength - 1];
            
            // Get 12-day sequence for all stocks
            for (let j = 0; j < sequenceLength; j++) {
                const date = this.dates[i + j];
                const featureVector = [];
                
                this.symbols.forEach(symbol => {
                    const stockData = this.normalizedData[date]?.[symbol];
                    if (stockData) {
                        featureVector.push(stockData.Open, stockData.Close);
                    } else {
                        featureVector.push(0, 0); // Padding for missing data
                    }
                });
                
                sequence.push(featureVector);
            }

            // Create target: 3-day ahead binary labels for each stock
            const target = [];
            const currentClosePrices = {};
            
            // Get current close prices for comparison
            this.symbols.forEach(symbol => {
                currentClosePrices[symbol] = this.normalizedData[currentDate]?.[symbol]?.Close || 0;
            });

            // Calculate binary labels for next 3 days
            for (let offset = 1; offset <= predictionHorizon; offset++) {
                const futureDate = this.dates[i + sequenceLength - 1 + offset];
                this.symbols.forEach(symbol => {
                    const futureClose = this.normalizedData[futureDate]?.[symbol]?.Close;
                    const currentClose = currentClosePrices[symbol];
                    
                    if (futureClose !== undefined && currentClose !== undefined) {
                        target.push(futureClose > currentClose ? 1 : 0);
                    } else {
                        target.push(0); // Default for missing data
                    }
                });
            }

            sequences.push(sequence);
            targets.push(target);
        }

        console.log('Created', sequences.length, 'sequences');
        
        // Convert to tensors
        this.X = tf.tensor3d(sequences);
        this.y = tf.tensor2d(targets);

        this.splitData();
    }

    splitData() {
        const splitIndex = Math.floor(this.X.shape[0] * this.trainTestSplit);
        
        this.X_train = this.X.slice([0, 0, 0], [splitIndex, -1, -1]);
        this.X_test = this.X.slice([splitIndex, 0, 0], [-1, -1, -1]);
        this.y_train = this.y.slice([0, 0], [splitIndex, -1]);
        this.y_test = this.y.slice([splitIndex, 0], [-1, -1]);

        console.log('Training data shape:', this.X_train.shape);
        console.log('Test data shape:', this.X_test.shape);

        // Clean up intermediate tensors
        tf.dispose([this.X, this.y]);
    }

    getData() {
        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            symbols: this.symbols
        };
    }

    dispose() {
        const tensors = [this.X_train, this.y_train, this.X_test, this.y_test, this.X, this.y];
        tensors.forEach(tensor => {
            if (tensor) {
                tf.dispose(tensor);
            }
        });
    }
}
