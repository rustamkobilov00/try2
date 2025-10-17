// data-loader.js
// ES6 module for client-side data loading and preprocessing. Requires TensorFlow.js.

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js';

export class DataLoader {
  constructor() {
    this.symbols = [];
    this.symbolIndex = {};
    this.dates = [];
    this.dataMatrix = null;
    this.min = {};
    this.max = {};
    this.ready = false;
  }

  // Parse CSV file: expects columns [Date,Symbol,Open,Close]
  async loadCSV(file) {
    const csvText = await file.text();
    const rows = csvText.trim().split('\n')
      .map(row => row.split(','));
    const headers = rows[0];
    if (!['Date','Symbol','Open','Close'].every(h => headers.includes(h))) {
      throw new Error('CSV columns missing. Required: Date,Symbol,Open,Close');
    }
    const dateIdx = headers.indexOf('Date');
    const symbolIdx = headers.indexOf('Symbol');
    const openIdx = headers.indexOf('Open');
    const closeIdx = headers.indexOf('Close');
    const data = [];
    const symbolSet = new Set();
    const dateSet = new Set();

    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      if (!r[dateIdx] || !r[symbolIdx]) continue;
      symbolSet.add(r[symbolIdx]);
      dateSet.add(r[dateIdx]);
      data.push({
        date: r[dateIdx],
        symbol: r[symbolIdx],
        open: Number(r[openIdx]),
        close: Number(r[closeIdx]),
      });
    }
    this.symbols = Array.from(symbolSet).sort();
    this.symbolIndex = Object.fromEntries(this.symbols.map((s,i)=>[s,i]));
    this.dates = Array.from(dateSet).sort();

    // Pivot: [date][symbol] → {open, close}
    const symbolN = this.symbols.length;
    const dateN = this.dates.length;
    // Fill with NaN by default
    this.dataMatrix = Array(dateN).fill(null).map(() =>
      Array(symbolN).fill(null).map(() => ({open:NaN, close:NaN}))
    );
    for(const row of data) {
      const dIdx = this.dates.indexOf(row.date);
      const sIdx = this.symbolIndex[row.symbol];
      this.dataMatrix[dIdx][sIdx] = {
        open: row.open,
        close: row.close
      };
    }
    this._normalize(); // calc min/max and normalize data
    this.ready = true;
  }

  _normalize() {
    // Compute min/max for normalization, per (symbol, feature)
    // Features: [Open, Close]
    this.min = {}, this.max = {};
    const symbolN = this.symbols.length;
    const openArrs = Array(symbolN).fill(null).map(()=>[]);
    const closeArrs = Array(symbolN).fill(null).map(()=>[]);
    for(const day of this.dataMatrix) {
      day.forEach((d, sIdx) => {
        openArrs[sIdx].push(d.open);
        closeArrs[sIdx].push(d.close);
      });
    }
    for(let i=0;i<symbolN;i++) {
      this.min[this.symbols[i]] = {
        open: Math.min(...openArrs[i]),
        close: Math.min(...closeArrs[i])
      };
      this.max[this.symbols[i]] = {
        open: Math.max(...openArrs[i]),
        close: Math.max(...closeArrs[i])
      };
    }
    // Normalize in-place to [0,1] per symbol/feature
    for (let d=0; d<this.dataMatrix.length; d++) {
      for (let s=0; s<symbolN; s++) {
        const sym = this.symbols[s];
        this.dataMatrix[d][s].open = this._scale(
          this.dataMatrix[d][s].open,
          this.min[sym].open, this.max[sym].open
        );
        this.dataMatrix[d][s].close = this._scale(
          this.dataMatrix[d][s].close,
          this.min[sym].close, this.max[sym].close
        );
      }
    }
  }

  _scale(val, min, max) {
    if (isNaN(val)) return 0.5; // fallback
    if (max === min) return 0.5;
    return (val - min) / (max - min);
  }

  // Prepare sliding window samples
  // X: [samples, 12, 20] (12 days, 10 stocks, 2 features per stock)
  // y: [samples, 30] (10 stocks × 3 day binary)
  buildDataset(windowLen=12, ahead=3, testSplit=0.2) {
    if (!this.ready) throw new Error('Data not loaded');
    const datesLen = this.dates.length;
    const symbolN = this.symbols.length;
    let X = [], y = [];
    for(let d=windowLen; d<datesLen - ahead; d++) {
      const xSeq = [];
      for(let w=d-windowLen; w<d; w++) {
        const dayVec = [];
        for(let s=0; s<symbolN; s++) {
          dayVec.push(this.dataMatrix[w][s].open);
          dayVec.push(this.dataMatrix[w][s].close);
        }
        xSeq.push(dayVec);
      }
      // Output: Binary labels per stock × day (future)
      const label = [];
      for(let s=0;s<symbolN;s++) {
        const closeD = this.dataMatrix[d-1][s].close; // prev day close
        for(let t=1; t<=ahead; t++) {
          const closeF = this.dataMatrix[d-1+t][s].close;
          label.push(closeF > closeD ? 1 : 0);
        }
      }
      X.push(xSeq);
      y.push(label);
    }
    // Chronological split
    const total = X.length;
    const splitIdx = Math.floor(total*(1-testSplit));
    const X_train = tf.tensor(X.slice(0,splitIdx));
    const y_train = tf.tensor(y.slice(0,splitIdx));
    const X_test = tf.tensor(X.slice(splitIdx));
    const y_test = tf.tensor(y.slice(splitIdx));
    // Return symbols for mapping indices
    return {
      X_train, y_train, X_test, y_test,
      symbolNames: this.symbols,
      disposeAll:()=>{
        X_train.dispose();
        y_train.dispose();
        X_test.dispose();
        y_test.dispose();
      }
    };
  }
}
