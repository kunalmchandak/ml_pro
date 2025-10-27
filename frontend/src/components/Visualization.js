import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Bar } from 'react-chartjs-2';
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

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Visualization = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const results = location.state?.results || {};

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
        borderColor: getChartColor().replace('0.5', '1'),
        borderWidth: 2,
        borderRadius: 8,
        hoverBackgroundColor: getChartColor().replace('0.5', '0.7'),
        hoverBorderColor: getChartColor().replace('0.5', '1'),
        hoverBorderWidth: 3,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    animation: {
      duration: 2000,
      easing: 'easeInOutQuart'
    },
    plugins: {
      legend: {
        labels: {
          font: {
            family: "'Inter', sans-serif",
            weight: '600'
          },
          padding: 20,
          usePointStyle: true,
        }
      },
      tooltip: {
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        titleColor: '#1F2937',
        bodyColor: '#4B5563',
        borderColor: '#E5E7EB',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        titleFont: {
          family: "'Inter', sans-serif",
          weight: '600',
          size: 14
        },
        bodyFont: {
          family: "'Inter', sans-serif",
          size: 13
        },
        displayColors: false
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
          drawBorder: false
        },
        ticks: {
          font: {
            family: "'Inter', sans-serif"
          },
          padding: 10
        }
      },
      x: {
        grid: {
          display: false
        },
        ticks: {
          font: {
            family: "'Inter', sans-serif"
          },
          padding: 10
        }
      }
    }
  };

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
    if (value === undefined || value === null || Number.isNaN(value)) return 'N/A';
    if (key === 'accuracy' || key === 'precision' || key === 'recall' || key === 'f1_score' || key === 'r2_score') {
      return `${(value * 100).toFixed(2)}%`;
    }
    if (key === 'silhouette_score') {
      return Number(value).toFixed(3);
    }
    return Number(value).toFixed(3);
  };

  const taskType = isRegression ? 'Regression' : isClassification ? 'Classification' : 'Unsupervised';

  return (
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-purple-50/30 to-indigo-50/20 relative overflow-hidden">
      {/* Enhanced Background Effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute w-[600px] h-[600px] -top-64 -left-48 bg-purple-300/40 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob motion-safe:animate-pulse"></div>
        <div className="absolute w-[700px] h-[700px] -bottom-64 -right-48 bg-indigo-300/40 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob-reverse motion-safe:animate-pulse"></div>
        <div className="absolute w-[500px] h-[500px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-pink-300/30 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-float"></div>
        <div className="absolute w-[400px] h-[400px] top-1/4 right-1/4 bg-blue-300/30 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob-slow"></div>
      </div>

      <div className="container mx-auto p-6">
        <header className="mb-8 text-center relative">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-400/20 via-transparent to-indigo-400/20 blur-3xl -z-10"></div>
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-fuchsia-500 to-indigo-600 mb-2 animate-gradient-x">Model Results</h1>
          <p className="text-gray-600/90 backdrop-blur-sm py-2">Visual summary of model performance and artifacts</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Performance + Charts */}
          <div className="lg:col-span-8">
            <div className="bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-purple-200/50 shadow-lg mb-6
                          hover:shadow-2xl transition-all duration-500 hover:bg-white/90
                          before:absolute before:inset-0 before:bg-gradient-to-r before:from-purple-500/10 before:via-transparent before:to-indigo-500/10 
                          before:opacity-0 before:transition-opacity hover:before:opacity-100 relative">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Model Performance ({taskType})</h2>
                <div className="text-sm text-gray-600">Algorithm: <span className="font-medium text-gray-800">{results.algorithm?.split('_').map(w => w.charAt(0).toUpperCase()+w.slice(1)).join(' ') || '—'}</span></div>
              </div>

              <div className="mb-4">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 to-indigo-500/5 rounded-xl"></div>
                  <div className="relative">
                    <Bar data={chartData} options={chartOptions} />
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
                {['accuracy','precision','recall','f1_score'].map((k) => (
                  (k in results) ? (
                    <div key={k} className="bg-white/80 backdrop-blur-sm p-4 rounded-lg border border-purple-100/50 shadow-sm
                                 hover:shadow-md hover:border-purple-200 transition-all duration-300
                                 hover:bg-gradient-to-br hover:from-white/90 hover:to-purple-50/90">
                      <div className="text-sm text-gray-500/90">{k.replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase())}</div>
                      <div className="text-lg font-semibold mt-1 bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-indigo-600">
                        {formatMetric(k, results[k])}
                      </div>
                    </div>
                  ) : null
                ))}
                {isRegression && (
                  ['mse','mae','r2_score'].map(k => (
                    (k in results) ? (
                      <div key={k} className="bg-white/80 backdrop-blur-sm p-4 rounded-lg border border-indigo-100/50 shadow-sm
                                   hover:shadow-md hover:border-indigo-200 transition-all duration-300
                                   hover:bg-gradient-to-br hover:from-white/90 hover:to-indigo-50/90">
                        <div className="text-sm text-gray-500/90">{k.replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase())}</div>
                        <div className="text-lg font-semibold mt-1 bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
                          {formatMetric(k, results[k])}
                        </div>
                      </div>
                    ) : null
                  ))
                )}
                {isUnsupervised && (
                  ['silhouette_score','calinski_harabasz_score','davies_bouldin_score'].map(k => (
                    (k in results) ? (
                      <div key={k} className="bg-white/80 p-4 rounded-lg border border-gray-100 shadow-sm">
                        <div className="text-sm text-gray-500">{k.replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase())}</div>
                        <div className="text-lg font-semibold mt-1 text-gray-800">{formatMetric(k, results[k])}</div>
                      </div>
                    ) : null
                  ))
                )}
              </div>
            </div>

            {/* Optional: placeholder for confusion matrix / cluster viz */}
            <div className="bg-white/90 backdrop-blur-lg p-6 rounded-2xl border border-purple-200 shadow-lg">
              <h3 className="text-lg font-semibold mb-3">Detailed Visuals</h3>
              <p className="text-sm text-gray-600 mb-4">You can extend this area with a confusion matrix, ROC curve, or 2D cluster projection depending on the algorithm and data.</p>
              <div className="h-56 bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg border-2 border-dashed border-purple-100 flex items-center justify-center text-gray-400">Visualization placeholder</div>
            </div>
          </div>

          {/* Right: Model details and actions */}
          <aside className="lg:col-span-4">
            <div className="sticky top-6 space-y-4">
              <div className="bg-white/80 backdrop-blur-xl p-6 rounded-2xl border border-purple-200/50 shadow-lg
                         hover:shadow-2xl transition-all duration-500 hover:bg-white/90 relative
                         before:absolute before:inset-0 before:bg-gradient-to-r before:from-purple-500/10 before:via-transparent before:to-indigo-500/10 
                         before:opacity-0 before:transition-opacity hover:before:opacity-100">
                <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 mb-4">Model Details</h3>
                <div className="text-sm space-y-3 relative z-10">
                  <div className="p-3 rounded-lg bg-purple-50/50 border border-purple-100/50 hover:bg-purple-50/70 transition-colors">
                    <div className="font-medium text-purple-800">Dataset</div>
                    <div className="text-purple-600">{results.dataset_name || '—'}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-indigo-50/50 border border-indigo-100/50 hover:bg-indigo-50/70 transition-colors">
                    <div className="font-medium text-indigo-800">Model Statistics</div>
                    <div className="grid grid-cols-2 gap-2 mt-1">
                      <div className="text-indigo-600">
                        <div className="text-xs text-indigo-500">Samples</div>
                        <div className="font-medium">{results.n_samples ?? '—'}</div>
                      </div>
                      <div className="text-indigo-600">
                        <div className="text-xs text-indigo-500">Features</div>
                        <div className="font-medium">{results.n_features ?? '—'}</div>
                      </div>
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-fuchsia-50/50 border border-fuchsia-100/50 hover:bg-fuchsia-50/70 transition-colors">
                    <div className="font-medium text-fuchsia-800">Algorithm Type</div>
                    <div className="text-fuchsia-600">
                      {isBoostingAlgorithm ? 'Gradient Boosting' : 
                       isNaiveBayes ? 'Naive Bayes' : 
                       isClassification ? 'Classification' : 
                       isRegression ? 'Regression' : 'Unsupervised'}
                    </div>
                  </div>
                </div>

                <div className="mt-6 flex flex-col gap-3 relative z-10">
                  <button onClick={handleDownload} 
                          className="w-full inline-flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-emerald-600 
                                   text-white py-3 px-4 rounded-xl font-semibold shadow-lg hover:shadow-xl
                                   transition-all duration-300 transform hover:scale-[1.02] hover:from-green-600 hover:to-emerald-700">
                    Download Model
                  </button>
                  {results.status === 'bad' && (
                    <button onClick={() => navigate('/dashboard', { state: { algorithm: results.algorithm, dataset: results.dataset_name } })} 
                            className="w-full inline-flex items-center justify-center gap-2 bg-gradient-to-r from-red-500 to-rose-600 
                                     text-white py-3 px-4 rounded-xl font-semibold shadow-lg hover:shadow-xl
                                     transition-all duration-300 transform hover:scale-[1.02] hover:from-red-600 hover:to-rose-700">
                      Retrain Model
                    </button>
                  )}
                  <button onClick={() => navigate('/categories')} 
                          className="w-full inline-flex items-center justify-center gap-2 bg-gradient-to-r from-gray-100 to-gray-200 
                                   text-gray-800 py-3 px-4 rounded-xl font-medium hover:from-gray-200 hover:to-gray-300 
                                   transition-all duration-300 transform hover:scale-[1.02]">
                    Back to Categories
                  </button>
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50/80 to-indigo-50/80 backdrop-blur-xl p-6 rounded-2xl 
                         border border-purple-200/50 shadow-lg hover:shadow-xl transition-all duration-500">
                <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 mb-4">Notes</h3>
                <ul className="text-sm space-y-3">
                  <li className="flex items-start gap-3 p-2 hover:bg-white/50 rounded-lg transition-colors">
                    <div className="w-2 h-2 mt-1.5 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500"></div>
                    <span className="text-gray-700">Interpret metrics relative to your business need and data context.</span>
                  </li>
                  <li className="flex items-start gap-3 p-2 hover:bg-white/50 rounded-lg transition-colors">
                    <div className="w-2 h-2 mt-1.5 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500"></div>
                    <span className="text-gray-700">Consider using cross-validation for more robust performance estimates.</span>
                  </li>
                  <li className="flex items-start gap-3 p-2 hover:bg-white/50 rounded-lg transition-colors">
                    <div className="w-2 h-2 mt-1.5 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500"></div>
                    <span className="text-gray-700">Experiment with different feature sets to potentially improve performance.</span>
                  </li>
                </ul>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
};

export default Visualization;
