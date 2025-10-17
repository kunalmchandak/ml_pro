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
      }
    ]
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
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Results</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">
            Model Performance ({taskType})
          </h2>
          <Bar data={chartData} />
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Model Details</h2>
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
                className={`px-2 py-1 rounded ${
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

          <div className="mt-4 flex gap-2">
            <button
              onClick={handleDownload}
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
            >
              Download Model
            </button>

            {results.status === 'bad' && (
              <button
                onClick={() => navigate('/dashboard', { state: { algorithm: results.algorithm, dataset: results.dataset_name } })}
                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
              >
                Retrain Model
              </button>
            )}
          </div>
        </div>
      </div>

      <button
        onClick={() => navigate('/dashboard')}
        className="mt-6 bg-gray-600 text-white px-4 py-2 rounded"
      >
        Back to Dashboard
      </button>
    </div>
  );
};

export default Visualization;
