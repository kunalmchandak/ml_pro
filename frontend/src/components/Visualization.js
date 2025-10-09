import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Bar } from 'react-chartjs-2';
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

  const chartData = {
    labels: ['Accuracy'],
    datasets: [
      {
        label: 'Model Performance',
        data: [results.accuracy || 0],
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      }
    ]
  };

  const handleDownload = async () => {
    try {
      const response = await axios.get(
        `http://localhost:8000/download_model/${results.dataset_name}/${results.algorithm}`
      );
      window.location.href = response.data.model_path;
    } catch (error) {
      console.error('Download error:', error);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Results</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
          <Bar data={chartData} />
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Model Details</h2>
          <div className="space-y-2">
            <p>Dataset: {results.dataset_name}</p>
            <p>Algorithm: {results.algorithm}</p>
            <p>Samples: {results.n_samples}</p>
            <p>Features: {results.n_features}</p>
            <p>Accuracy: {(results.accuracy * 100).toFixed(2)}%</p>
          </div>
          
          <button
            onClick={handleDownload}
            className="mt-4 bg-green-600 text-white px-4 py-2 rounded"
          >
            Download Model
          </button>
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
