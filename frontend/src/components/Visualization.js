import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Bar, Scatter } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

import { PointElement } from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const Visualization = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const results = location.state?.results || {};

  // Detect association or cluster outputs
  const isAssociationResults = !!(results.rules || results.association || results.type === 'association');
  const clusterPoints = results.cluster_points || results.cluster_data?.points || null;

  // Determine task type
  const isClassification = 'accuracy' in results;
  const isRegression = 'r2_score' in results;
  const isUnsupervised = !isClassification && !isRegression;

  // Determine algorithm type for better display
  const isBoostingAlgorithm = results.algorithm?.includes('boost');
  const isNaiveBayes = results.algorithm?.includes('nb');

  // Get specific metrics for chart
  const metrics = isRegression
    ? ['mse', 'mae', 'r2_score']
    : isClassification
    ? ['accuracy', 'precision', 'recall', 'f1_score']
    : ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'inertia'].filter(key => key in results);

  // Choose chart color based on algorithm type
  const getChartColor = () => {
    if (isBoostingAlgorithm) return 'rgba(52, 211, 153, 0.5)'; // Green for boosting
    if (isNaiveBayes) return 'rgba(251, 146, 60, 0.5)'; // Orange for Naive Bayes
    if (isClassification) return 'rgba(59, 130, 246, 0.5)'; // Blue for classification
    if (isRegression) return 'rgba(139, 92, 246, 0.5)'; // Purple for regression
    return 'rgba(156, 163, 175, 0.5)'; // Gray for unsupervised
  };

  const chartData = {
    labels: metrics.map(m => m.replace(/_/g, ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')),
    datasets: [
      {
        label: isRegression ? 'Regression Metrics' : 
               isClassification ? `${isBoostingAlgorithm ? 'Boosting' : isNaiveBayes ? 'Naive Bayes' : ''} Classification Metrics` : 
               'Unsupervised Metrics',
        data: metrics.map(m => results[m] || 0),
        backgroundColor: getChartColor(),
      }
    ]
  };

  // Build scatter data when cluster points are available
  const buildClusterScatter = () => {
    if (!clusterPoints || !Array.isArray(clusterPoints)) return null;
    // Expect clusterPoints as array of {x, y, label}
    const groups = {};
    clusterPoints.forEach(p => {
      const lab = p.label ?? '0';
      if (!groups[lab]) groups[lab] = [];
      groups[lab].push({ x: p.x, y: p.y });
    });
    const palette = [
      'rgba(59,130,246,0.8)', 'rgba(234,88,12,0.8)', 'rgba(16,185,129,0.8)', 'rgba(168,85,247,0.8)', 'rgba(244,63,94,0.8)'
    ];
    const datasets = Object.entries(groups).map(([lab, points], i) => ({
      label: `Cluster ${lab}`,
      data: points,
      pointBackgroundColor: palette[i % palette.length],
      pointRadius: 4
    }));
    return { datasets };
  };

  const scatterData = buildClusterScatter();

  const handleDownload = async () => {
    try {
      const response = await axios.get(
        `http://localhost:8000/download_model/${results.dataset_name}/${results.algorithm}`,
        { responseType: 'blob' } // Important: set responseType to blob
      );
      
      // Create blob link to download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${results.dataset_name}_${results.algorithm}_model.pkl`);
      
      // Append to html link element page
      document.body.appendChild(link);
      
      // Start download
      link.click();
      
      // Clean up and remove the link
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      alert('Error downloading model');
    }
  };

  const formatMetric = (key, value) => {
    if (key === 'accuracy' || key === 'precision' || key === 'recall' || key === 'f1_score' || key === 'r2_score') {
      return `${(value * 100).toFixed(2)}%`;
    }
    if (key === 'silhouette_score') {
      return value.toFixed(3);
    }
    return value.toFixed(2);
  };

  const taskType = isRegression ? 'Regression' : isClassification ? 'Classification' : 'Unsupervised';

  return (
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-purple-100/40 to-indigo-100/40 relative">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 w-full h-full overflow-hidden -z-20 pointer-events-none">
        <div className="absolute w-96 h-96 -top-48 -left-48 bg-purple-400 rounded-full mix-blend-multiply filter blur-lg opacity-60 animate-blob" />
        <div className="absolute w-96 h-96 -top-48 -right-48 bg-indigo-400 rounded-full mix-blend-multiply filter blur-lg opacity-60 animate-blob-x animation-delay-2000" />
        <div className="absolute w-96 h-96 -bottom-48 -left-48 bg-pink-400 rounded-full mix-blend-multiply filter blur-lg opacity-60 animate-blob-y animation-delay-4000" />
        <div className="absolute w-80 h-80 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-pink-300 rounded-full mix-blend-multiply filter blur-lg opacity-55 animate-blob-xy animation-delay-2500" />
      </div>

      <div className="container mx-auto p-6 relative z-10">
        <h1 className="text-4xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">Results</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
            <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">Model Performance ({taskType})</h2>
            {isAssociationResults ? (
              <div className="text-sm text-gray-700">This run produced association rules — see Rules panel for details.</div>
            ) : scatterData ? (
              <div>
                <h4 className="font-medium mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">Cluster Visualization (PCA 2D)</h4>
                <div className="h-80 bg-gradient-to-br from-purple-100 via-indigo-100 to-white rounded-xl p-4 shadow-inner">
                  <Scatter
                    data={scatterData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'top',
                          labels: { color: '#6D28D9', font: { weight: 'bold', size: 14 } }
                        },
                        tooltip: {
                          backgroundColor: 'rgba(255,255,255,0.95)',
                          titleColor: '#7C3AED',
                          bodyColor: '#1F2937',
                          borderColor: '#E5E7EB',
                          borderWidth: 1,
                          padding: 12,
                          cornerRadius: 8
                        }
                      },
                      scales: {
                        x: {
                          grid: { color: 'rgba(147,51,234,0.08)' },
                          title: { display: true, text: 'PC1', color: '#7C3AED', font: { weight: 'bold', size: 16 } },
                          ticks: { color: '#6D28D9' }
                        },
                        y: {
                          grid: { color: 'rgba(99,102,241,0.08)' },
                          title: { display: true, text: 'PC2', color: '#6366F1', font: { weight: 'bold', size: 16 } },
                          ticks: { color: '#6366F1' }
                        }
                      }
                    }}
                  />
                </div>
              </div>
            ) : (
              <div className="bg-gradient-to-br from-purple-100 via-indigo-100 to-white rounded-xl p-4 shadow-inner">
                <Bar
                  data={chartData}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        display: true,
                        labels: { color: '#6D28D9', font: { weight: 'bold', size: 14 } }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(255,255,255,0.95)',
                        titleColor: '#7C3AED',
                        bodyColor: '#1F2937',
                        borderColor: '#E5E7EB',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8
                      }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(147,51,234,0.08)' },
                        ticks: { color: '#6D28D9', font: { weight: 'bold', size: 13 } }
                      },
                      y: {
                        grid: { color: 'rgba(99,102,241,0.08)' },
                        ticks: { color: '#6366F1', font: { weight: 'bold', size: 13 } }
                      }
                    }
                  }}
                />
              </div>
            )}
          </div>

          <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
            <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">Model Details</h2>
            <div className="space-y-2">
              <p><strong>Dataset:</strong> {results.dataset_name}</p>
              <p><strong>Algorithm:</strong> {results.algorithm?.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</p>
              <p><strong>Algorithm Type:</strong> {
                isBoostingAlgorithm ? 'Gradient Boosting' :
                isNaiveBayes ? 'Naive Bayes' :
                isClassification ? 'Classification' :
                isRegression ? 'Regression' : 'Unsupervised'
              }</p>
              <p><strong>Samples:</strong> {results.n_samples}</p>
              <p><strong>Features:</strong> {results.n_features}</p>
              <p>
                <strong>Status:</strong>{' '}
                <span 
                  className={`px-2 py-1 rounded-full font-semibold ${
                    results.status === 'good' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {results.status === 'good' ? 'Good Performance' : 'Needs Improvement'}
                </span>
              </p>

              {isClassification && (
                <>
                  <p><strong>Accuracy:</strong> {formatMetric('accuracy', results.accuracy)}</p>
                  <p><strong>Precision:</strong> {formatMetric('precision', results.precision)}</p>
                  <p><strong>Recall:</strong> {formatMetric('recall', results.recall)}</p>
                  <p><strong>F1 Score:</strong> {formatMetric('f1_score', results.f1_score)}</p>
                </>
              )}

              {isRegression && (
                <>
                  <p><strong>MSE:</strong> {formatMetric('mse', results.mse)}</p>
                  <p><strong>MAE:</strong> {formatMetric('mae', results.mae)}</p>
                  <p><strong>R² Score:</strong> {formatMetric('r2_score', results.r2_score)}</p>
                </>
              )}

              {isUnsupervised && (
                <>
                  {results.silhouette_score !== undefined && (
                    <p><strong>Silhouette Score:</strong> {formatMetric('silhouette_score', results.silhouette_score)}</p>
                  )}
                  {results.calinski_harabasz_score !== undefined && (
                    <p><strong>Calinski-Harabasz Score:</strong> {formatMetric('calinski_harabasz_score', results.calinski_harabasz_score)}</p>
                  )}
                  {results.davies_bouldin_score !== undefined && (
                    <p><strong>Davies-Bouldin Score:</strong> {formatMetric('davies_bouldin_score', results.davies_bouldin_score)}</p>
                  )}
                  {results.inertia !== undefined && (
                    <p><strong>Inertia:</strong> {formatMetric('inertia', results.inertia)}</p>
                  )}
                </>
              )}
            </div>

            <div className="mt-6 flex gap-3">
              <button
                onClick={handleDownload}
                className="bg-gradient-to-r from-green-500 to-green-700 text-white px-5 py-2 rounded-lg font-bold shadow hover:from-green-600 hover:to-green-800 transition-all duration-200"
              >
                Download Model
              </button>

              {results.status === 'bad' && (
                <button
                  onClick={() => navigate('/dashboard', { state: { algorithm: results.algorithm, dataset: results.dataset_name } })}
                  className="bg-gradient-to-r from-red-500 to-red-700 text-white px-5 py-2 rounded-lg font-bold shadow hover:from-red-600 hover:to-red-800 transition-all duration-200"
                >
                  Retrain Model
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="flex justify-end mt-10">
          <button
            onClick={() => navigate('/dashboard')}
            className="bg-gradient-to-r from-gray-600 to-gray-800 text-white px-6 py-3 rounded-lg font-bold shadow hover:from-gray-700 hover:to-gray-900 transition-all duration-200"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};

export default Visualization;
