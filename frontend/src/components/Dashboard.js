import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Dashboard = () => {
  const [algorithm, setAlgorithm] = useState('');
  const [compatibleDatasets, setCompatibleDatasets] = useState({});
  const [selectedDataset, setSelectedDataset] = useState('');
  const [algorithmInfo, setAlgorithmInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Fetch compatible datasets when algorithm changes
  useEffect(() => {
    if (algorithm) {
      fetchCompatibleDatasets(algorithm);
    }
    // Reset dataset selection when algorithm changes
    setSelectedDataset('');
  }, [algorithm]);

  const fetchCompatibleDatasets = async (algo) => {
    try {
      const response = await axios.get(`http://localhost:8000/datasets/compatible/${algo}`);
      setCompatibleDatasets(response.data.compatible_datasets);
      setAlgorithmInfo(response.data.algorithm_info);
    } catch (error) {
      console.error('Error fetching compatible datasets:', error);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/train', {
        algorithm,
        dataset_name: selectedDataset,
        params: {}
      });
      navigate('/visualization', { state: { results: response.data } });
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">ML Dashboard</h1>
      
      <div className="grid grid-cols-1 gap-6">
        {/* Algorithm Selection */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">1. Select Algorithm</h2>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="">Choose an algorithm</option>
            <optgroup label="Supervised Learning">
              <option value="logistic_regression">Logistic Regression</option>
              <option value="random_forest">Random Forest</option>
              <option value="svm">SVM</option>
            </optgroup>
            <optgroup label="Unsupervised Learning">
              <option value="kmeans">K-Means Clustering</option>
            </optgroup>
          </select>
          
          {algorithmInfo && (
            <div className="mt-2 text-sm text-gray-600">
              {algorithmInfo.description}
            </div>
          )}
        </div>

        {/* Dataset Selection - Only shown if algorithm is selected */}
        {algorithm && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">2. Select Compatible Dataset</h2>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full p-2 border rounded"
            >
              <option value="">Select a dataset</option>
              {Object.entries(compatibleDatasets).map(([name, info]) => (
                <option key={name} value={name}>
                  {name} ({info.samples} samples, {info.features} features)
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={handleTrain}
            disabled={!selectedDataset || loading}
            className="mt-6 bg-blue-600 text-white px-6 py-2 rounded-lg 
                     hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Training...' : 'Train Model'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
