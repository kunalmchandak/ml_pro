
import React, { useState } from 'react';
import axios from 'axios';
import { Line, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ClusterTools = ({ selectedDataset, isUserDataset }) => {
  console.log('ClusterTools rendered with:', { selectedDataset, isUserDataset });
  
  const [loading, setLoading] = useState(false);
  const [elbowData, setElbowData] = useState([]);
  const [nClusters, setNClusters] = useState(3);
  const [algorithm, setAlgorithm] = useState('kmeans');
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState('');
  const [clusterData, setClusterData] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [debugLog, setDebugLog] = useState([]);

  // Evaluate model status from clustering metrics
  const evaluateModelStatus = (m) => {
    if (!m) return null;
    const s = m.silhouette_score ?? null;
    const ch = m.calinski_harabasz_score ?? null;
    const db = m.davies_bouldin_score ?? null;

    // Simple scoring: normalize available metrics and average
    let parts = [];
    if (typeof s === 'number') {
      // silhouette: -1..1 -> 0..1
      parts.push((s + 1) / 2);
    }
    if (typeof ch === 'number') {
      // CH score scale varies; scale by a heuristic cap
      parts.push(Math.min(ch / 100, 1));
    }
    if (typeof db === 'number') {
      // DB lower is better; invert and clamp assuming typical 0..3 range
      parts.push(Math.max(0, Math.min(1, 1 - (db / 3))));
    }
    if (parts.length === 0) return null;
    const avg = parts.reduce((a, b) => a + b, 0) / parts.length;
    if (avg >= 0.85) return { label: 'Best', className: 'text-green-700 bg-green-100' };
    if (avg >= 0.65) return { label: 'Good', className: 'text-green-600 bg-green-50' };
    if (avg >= 0.4) return { label: 'Average', className: 'text-yellow-700 bg-yellow-100' };
    return { label: 'Poor', className: 'text-red-700 bg-red-100' };
  };

  const fetchElbow = async (maxK = 10) => {
    if (!selectedDataset) {
      setError('Select a dataset first');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const resp = await axios.post('http://localhost:8000/cluster/elbow', {
        dataset_name: selectedDataset,
        is_user_dataset: !!isUserDataset,
        max_k: maxK
      });
      const elbow = resp.data.elbow || [];
      setElbowData(elbow);

      // Auto-compute optimal k (elbow point)
      if (elbow.length > 2) {
        // Find where improvement rate drops significantly
        let bestK = elbow[0].k;
        for (let i = 1; i < elbow.length; i++) {
          const improvement = elbow[i].improvement;
          // Choose k where improvement drops below 20%
          if (improvement && improvement < 0.20) {
            bestK = elbow[i].k;
            break;
          }
          // If no clear drop found, fallback to k=5 for Mall_Customers.csv
          bestK = selectedDataset === 'Mall_Customers.csv' ? 5 : 3;
        }
        setNClusters(bestK);
      }
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to fetch elbow data');
    } finally {
      setLoading(false);
    }
  };

  const runMetrics = async () => {
    if (!selectedDataset) {
      setError('Select a dataset first');
      return;
    }
    console.log('Running metrics for dataset:', selectedDataset, 'isUserDataset:', isUserDataset);
    setLoading(true);
    setError('');
    setMetrics(null);
    try {
      const payload = {
        dataset_name: selectedDataset,
        is_user_dataset: !!isUserDataset,
        n_clusters: Number(nClusters),
        params: { algorithm }
      };
      setDebugLog(l => [...l, { time: Date.now(), type: 'request-metrics', payload }]);
      const resp = await axios.post('http://localhost:8000/cluster/metrics', payload);
      setMetrics(resp.data);
      setDebugLog(l => [...l, { time: Date.now(), type: 'response-metrics', data: resp.data }]);
      // request visualization data (2D PCA points + centroids) if backend supports it
      try {
        const vizPayload = payload; // same shape
        setDebugLog(l => [...l, { time: Date.now(), type: 'request-visualize', payload: vizPayload }]);
        const viz = await axios.post('http://localhost:8000/cluster/visualize', vizPayload);
        setDebugLog(l => [...l, { time: Date.now(), type: 'response-visualize', data: viz.data }]);
        console.log('Visualization response:', viz.data);
        setClusterData(viz.data);
      } catch (e) {
        // show visualization failure in UI debug
        const errDetail = e?.response?.data || e.message || String(e);
        setDebugLog(l => [...l, { time: Date.now(), type: 'error-visualize', error: errDetail }]);
        console.warn('Visualization endpoint failed:', errDetail);
      }
      // compute model status
      if (resp.data && resp.data.metrics) {
        setModelStatus(evaluateModelStatus(resp.data.metrics));
      }
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to run clustering');
    } finally {
      setLoading(false);
    }
  };

  const chart = {
    labels: elbowData.map(d => d.k),
    datasets: [
      {
        label: 'Inertia',
        data: elbowData.map(d => d.inertia),
        fill: false,
        borderColor: 'rgba(59,130,246,0.8)'
      }
    ]
  };

  const animationClasses = {
    card: "transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-purple-200/50",
    button: "transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 shadow-lg hover:shadow-xl",
    input: "transition-all duration-200 focus:ring-2 focus:ring-purple-400",
  };

  return (
    <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
        Clustering Tools
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className={`${animationClasses.card} bg-white/80 p-4 rounded-xl border-2 border-purple-200`}>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Algorithm</label>
          <select 
            value={algorithm} 
            onChange={e => setAlgorithm(e.target.value)} 
            className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`}
          >
            <option value="kmeans">K-Means</option>
            <option value="agglomerative">Agglomerative</option>
            <option value="birch">Birch</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </div>

        <div className={`${animationClasses.card} bg-white/80 p-4 rounded-xl border-2 border-purple-200`}>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Clusters (k)</label>
          <input 
            type="number" 
            min={2} 
            value={nClusters} 
            onChange={e => setNClusters(e.target.value)} 
            className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`} 
          />
        </div>

        <div className="flex items-end gap-3">
          <button 
            onClick={() => fetchElbow(10)} 
            className={`${animationClasses.button} flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
              text-white font-bold py-3 px-4 rounded-lg`}
          >
            Compute Elbow
          </button>
          <button 
            onClick={runMetrics} 
            className={`${animationClasses.button} flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
              text-white font-bold py-3 px-4 rounded-lg`}
          >
            Run Clustering
          </button>
        </div>
      </div>

      {loading && (
        <div className="text-sm font-medium text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 animate-pulse">
          Processing...
        </div>
      )}
      {error && (
        <div className="text-sm font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
          {error}
        </div>
      )}

      {elbowData && elbowData.length > 0 && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Elbow Plot
          </h4>
          <Line data={chart} />
        </div>
      )}

      {metrics && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            Cluster Analysis
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Algorithm</div>
              <div className="text-lg font-semibold">{metrics.algorithm}</div>
            </div>
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Clusters</div>
              <div className="text-lg font-semibold">{metrics.n_clusters}</div>
            </div>
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
              <div className="text-sm text-gray-600 mb-1">Samples</div>
              <div className="text-lg font-semibold">{metrics.n_samples}</div>
            </div>
          </div>

          {modelStatus && (
            <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100 mb-6`}>
              <div className="text-sm text-gray-600 mb-1">Model Status</div>
              <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${modelStatus.className}`}>
                {modelStatus.label}
              </div>
            </div>
          )}

          <div className={`${animationClasses.card} bg-white/90 p-4 rounded-xl border border-purple-100`}>
            <div className="text-sm text-gray-600 mb-3">Detailed Metrics</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(metrics.metrics).map(([k,v]) => (
                <div key={k} className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-600">{k}:</span>
                  <span className="text-sm font-bold">
                    {v === null ? 'n/a' : (typeof v === 'number' ? v.toFixed(3) : String(v))}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6">
            <button
              onClick={async () => {
                try {
                  const resp = await axios.get('http://localhost:8000/download_model', { responseType: 'blob' });
                  const url = window.URL.createObjectURL(new Blob([resp.data]));
                  const link = document.createElement('a');
                  link.href = url;
                  link.setAttribute('download', `cluster_model_${metrics.algorithm}_${metrics.n_clusters}.joblib`);
                  document.body.appendChild(link);
                  link.click();
                  link.parentNode.removeChild(link);
                  window.URL.revokeObjectURL(url);
                } catch (e) {
                  console.error('Download failed', e);
                  setError(e?.response?.data?.detail || 'Model download failed');
                }
              }}
              className={`${animationClasses.button} bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                text-white font-bold py-3 px-6 rounded-lg`}
            >
              Download Model
            </button>
          </div>
        </div>
      )}

      {clusterData && (
        <div className={`${animationClasses.card} mt-8 bg-white/80 p-6 rounded-xl border-2 border-purple-200`}>
          <h4 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
            {selectedDataset === 'Mall_Customers.csv' ? 
              'Customer Segments (Income vs Spending)' : 
              'Cluster Visualization (PCA 2D)'}
          </h4>
          <div className="h-[400px]">
            <Scatter
              data={{
                datasets: [
                  {
                    label: 'Customers',
                    data: clusterData.points || [],
                    pointBackgroundColor: clusterData.colors || 'rgba(147,51,234,0.6)', // purple-600
                    pointRadius: 4,
                    showLine: false
                  },
                  {
                    label: 'Centroids',
                    data: clusterData.centroids || [],
                    pointBackgroundColor: 'rgba(79,70,229,0.9)', // indigo-600
                    pointRadius: 9,
                    pointStyle: 'triangle',
                    showLine: false
                  }
                ]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: { 
                    title: { 
                      display: true, 
                      text: selectedDataset === 'Mall_Customers.csv' ? 'Annual Income (k$)' : 'PC1',
                      font: { weight: 'bold' }
                    },
                    ticks: {
                      callback: function(value) {
                        return selectedDataset === 'Mall_Customers.csv' ? value + 'k$' : value;
                      }
                    },
                    grid: {
                      color: 'rgba(107,114,128,0.1)' // gray-500 with low opacity
                    }
                  },
                  y: { 
                    title: { 
                      display: true, 
                      text: selectedDataset === 'Mall_Customers.csv' ? 'Spending Score (1-100)' : 'PC2',
                      font: { weight: 'bold' }
                    },
                    grid: {
                      color: 'rgba(107,114,128,0.1)' // gray-500 with low opacity
                    }
                  }
                },
                plugins: {
                  tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    titleColor: '#1F2937', // gray-800
                    bodyColor: '#1F2937', // gray-800
                    borderColor: '#E5E7EB', // gray-200
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 8,
                    callbacks: {
                      label: function(context) {
                        const dsLabel = context.dataset.label || '';
                        if (dsLabel === 'Centroids') {
                          return `Centroid (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                        }
                        const idx = context.dataIndex;
                        const clusterLabel = (clusterData.labels && clusterData.labels[idx] !== undefined) ? clusterData.labels[idx] : 'n/a';
                        return `Cluster ${clusterLabel}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                      }
                    }
                  },
                  legend: { 
                    position: 'top',
                    labels: {
                      font: { weight: 'bold' }
                    }
                  }
                }
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ClusterTools;
