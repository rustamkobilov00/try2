// app.js
// ES6 module. Coordinates UI, training, visualization. No external libs except tf.js.

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js';
import {DataLoader} from './data-loader.js';
import {GRUModel} from './gru.js';

// Simple UI helpers
function setStatus(msg) {
  document.getElementById('status-msg').textContent = msg;
}
function clearCharts() {
  document.getElementById('bar-chart').innerHTML = '';
  document.getElementById('timeline-rows').innerHTML = '';
}

function renderBarChart(data) {
  const container = document.getElementById('bar-chart');
  container.innerHTML = '';
  const maxAcc = Math.max(...data.map(x=>x.acc));
  data.forEach((d, i) => {
    const bar = document.createElement('div');
    bar.style.display = 'flex';
    bar.style.alignItems = 'center';
    bar.style.margin = '4px';
    const label = document.createElement('span');
    label.style.width = '90px';
    label.textContent = d.symbol;
    const bbox = document.createElement('div');
    bbox.style.height = '18px';
    bbox.style.background = '#ddd';
    bbox.style.width = '130px';
    bbox.style.margin = '0 10px';
    const barfill = document.createElement('div');
    barfill.style.height = '100%';
    barfill.style.background = '#278a40';
    barfill.style.width = `${Math.round((d.acc/maxAcc) * 120)}px`;
    bbox.appendChild(barfill);
    const val = document.createElement('span');
    val.textContent = (d.acc*100).toFixed(1)+'%';
    bar.appendChild(label);
    bar.appendChild(bbox);
    bar.appendChild(val);
    container.appendChild(bar);
  });
}

function renderTimelines(perStock, y_true, y_pred, nSteps=3) {
  const timelineRows = document.getElementById('timeline-rows');
  timelineRows.innerHTML = '';
  perStock.forEach((s, idx) => {
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.marginBottom = '3px';
    const name = document.createElement('span');
    name.textContent = s.symbol;
    name.style.width = '75px';
    name.style.fontSize = '0.96em';
    row.appendChild(name);
    const timeline = document.createElement('div');
    timeline.style.display = 'flex';
    timeline.style.margin = '0 6px';
    let stockId = perStock.map(x=>x.symbol).indexOf(s.symbol);
    for(let i=0;i<y_pred.length;i++) {
      for(let t=0;t<nSteps;t++) {
        const idxFlat = stockId*nSteps+t;
        const pr = y_pred[i][idxFlat] >= 0.5 ? 1 : 0;
        const gt = y_true[i][idxFlat];
        const cell = document.createElement('span');
        cell.style.width = '8px';
        cell.style.height = '14px';
        cell.style.display = 'inline-block';
        cell.style.marginRight = '1px';
        cell.style.background = pr === gt ? '#2fb631' : '#d0303c';
        row.appendChild(cell);
      }
    }
    timelineRows.appendChild(row);
  });
}

let dataLoader, model, trainData, inTraining = false;

// UI Bindings
function setupApp() {
  document.getElementById('csv-input').addEventListener('change', async (e) => {
    setStatus('Loading CSV...');
    try {
      dataLoader = new DataLoader();
      await dataLoader.loadCSV(e.target.files[0]);
      setStatus('CSV loaded: '+dataLoader.symbols.join(', '));
    } catch (err) {
      setStatus('Failed: '+err.message);
    }
  });

  document.getElementById('train-btn').addEventListener('click', async ()=>{
    if (inTraining) return;
    if (!dataLoader?.ready) { setStatus('Load a CSV first!'); return; }
    inTraining = true;
    setStatus('Preparing data...');
    clearCharts();
    trainData = dataLoader.buildDataset();
    model = new GRUModel([12, 20], 30, 64);
    setStatus('Training model...');
    await model.fit(
      trainData.X_train, trainData.y_train, {
        epochs: 32, batchSize: 32,
        validationSplit: 0.12,
        callbacks: {
          onEpochEnd:(epoch, logs)=>setStatus(
            `Epoch ${epoch+1}: val_acc ${(logs.val_binaryAccuracy*100).toFixed(1)}%...`
          )
        }
      });
    setStatus('Evaluating...');
    const stats = await model.accuracyByStock(trainData.X_test, trainData.y_test, trainData.symbolNames);
    // Bar chart
    renderBarChart(stats.perStock);
    // Timelines
    renderTimelines(stats.perStock, stats.y_true, stats.y_pred, 3);
    setStatus('Completed');
    inTraining = false;
    // Clean up memory
    tf.dispose([trainData.X_train,trainData.y_train,trainData.X_test,trainData.y_test]);
  });

  // (Optional) save/load
  document.getElementById('save-btn').addEventListener('click', async()=>{
    if (model) await model.saveModel();
    setStatus('Model saved locally');
  });
  document.getElementById('load-btn').addEventListener('click', async()=>{
    model = new GRUModel([12,20],30,64);
    await model.loadModel();
    setStatus('Model loaded from storage');
  });
}

window.addEventListener('DOMContentLoaded', setupApp);
