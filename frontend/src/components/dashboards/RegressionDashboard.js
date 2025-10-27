import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  PresentationChartLineIcon,
  AdjustmentsHorizontalIcon,
  ChartBarIcon,
  ArrowPathIcon,
  VariableIcon
} from '@heroicons/react/24/outline';

const RegressionDashboard = () => {
  const [algorithm, setAlgorithm] = useState('linear_regression');
  const [compatibleDatasets, setCompatibleDatasets] = useState({});
  const [selectedDataset, setSelectedDataset] = useState('');
  const [algorithmInfo, setAlgorithmInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    if (algorithm) {
      fetchCompatibleDatasets(algorithm);
    }
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
    setErrorMsg('');
    try {
      const response = await axios.post('http://localhost:8000/train', {
        algorithm,
        dataset_name: selectedDataset,
        params: {}
      });
      navigate('/visualization', { state: { results: response.data } });
    } catch (error) {
      console.error('Training error:', error);
      setErrorMsg(error?.response?.data?.detail || 'Training failed. Please check your selections and dataset.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-indigo-50/30 to-purple-50/20 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute w-[500px] h-[500px] -top-48 -right-48 bg-indigo-300/20 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob"></div>
        <div className="absolute w-[500px] h-[500px] bottom-0 left-0 bg-purple-300/20 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob-reverse"></div>
      </div>

      <div className="container mx-auto px-6 py-12">
        {/* Header Section */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-4">
            Regression Dashboard
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Select your regression algorithm and dataset to predict continuous values
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Main Content - Algorithm Selection */}
          <div className="lg:col-span-8">
            <div className="bg-white/90 backdrop-blur-lg p-8 rounded-2xl border border-indigo-200 shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <PresentationChartLineIcon className="w-6 h-6 text-indigo-600" />
                Select Regression Algorithm
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                <AlgorithmCard
                  title="Linear Regression"
                  description="Basic linear model for simple relationships"
                  selected={algorithm === 'linear_regression'}
                  onClick={() => setAlgorithm('linear_regression')}
                  icon={<ChartBarIcon className="w-6 h-6" />}
                />
                <AlgorithmCard
                  title="Random Forest"
                  description="Ensemble method for complex non-linear relationships"
                  selected={algorithm === 'random_forest_regressor'}
                  onClick={() => setAlgorithm('random_forest_regressor')}
                  icon={<VariableIcon className="w-6 h-6" />}
                />
                <AlgorithmCard
                  title="Gradient Boosting"
                  description="Boosted trees for high accuracy predictions"
                  selected={algorithm === 'gradient_boosting'}
                  onClick={() => setAlgorithm('gradient_boosting')}
                  icon={<ChartBarIcon className="w-6 h-6" />}
                />
                <AlgorithmCard
                  title="Support Vector Regression"
                  description="Kernel-based method for non-linear patterns"
                  selected={algorithm === 'svr'}
                  onClick={() => setAlgorithm('svr')}
                  icon={<AdjustmentsHorizontalIcon className="w-6 h-6" />}
                />
              </div>

              {algorithmInfo && (
                <div className="bg-indigo-50 rounded-xl p-6 border border-indigo-100">
                  <h3 className="font-semibold text-indigo-800 mb-2">Algorithm Information</h3>
                  <p className="text-indigo-600">{algorithmInfo.description}</p>
                </div>
              )}
            </div>

            {/* Dataset Selection */}
            <div className="bg-white/90 backdrop-blur-lg p-8 rounded-2xl border border-indigo-200 shadow-lg mt-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <ArrowPathIcon className="w-6 h-6 text-indigo-600" />
                Select Dataset
              </h2>
              
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full p-3 rounded-xl bg-white border border-indigo-200 
                         focus:border-indigo-400 focus:ring focus:ring-indigo-200 
                         focus:ring-opacity-50 mb-4"
              >
                <option value="">Choose a dataset</option>
                {Object.entries(compatibleDatasets).map(([key, info]) => (
                  <option key={key} value={key}>
                    {info && info.description
                      ? `${key} — ${info.description}`
                      : `${key}${info && info.samples ? ` (${info.samples} samples)` : ''}`}
                  </option>
                ))}
              </select>

              <button
                onClick={handleTrain}
                disabled={loading || !selectedDataset}
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white 
                         py-3 px-6 rounded-xl font-semibold shadow-lg
                         hover:from-indigo-700 hover:to-purple-700 
                         disabled:opacity-50 disabled:cursor-not-allowed
                         transition-all duration-300 transform hover:scale-[1.02]"
              >
                {loading ? 'Training...' : 'Train Model'}
              </button>

              {errorMsg && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl text-red-600">
                  {errorMsg}
                </div>
              )}
            </div>
          </div>

          {/* Sidebar - Additional Info */}
          <div className="lg:col-span-4">
            <div className="sticky top-6">
              <div className="bg-white/90 backdrop-blur-lg p-6 rounded-2xl border border-indigo-200 shadow-lg mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Regression Metrics</h3>
                <ul className="space-y-3 text-gray-600">
                  <MetricItem label="R² Score" description="Coefficient of determination" />
                  <MetricItem label="MAE" description="Mean Absolute Error" />
                  <MetricItem label="MSE" description="Mean Squared Error" />
                  <MetricItem label="RMSE" description="Root Mean Squared Error" />
                </ul>
              </div>

              <div className="bg-gradient-to-br from-indigo-50 to-purple-50 p-6 rounded-2xl border border-indigo-200 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Tips</h3>
                <ul className="space-y-3 text-gray-600">
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-600">•</span>
                    Check for linear relationships in your data
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-600">•</span>
                    Consider feature scaling for better results
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-600">•</span>
                    Handle outliers in your dataset
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const AlgorithmCard = ({ title, description, selected, onClick, icon }) => (
  <button
    onClick={onClick}
    className={`flex flex-col p-6 rounded-xl border-2 transition-all duration-300 ${
      selected
        ? 'border-indigo-400 bg-indigo-50/50 shadow-lg scale-[1.02]'
        : 'border-indigo-200 bg-white hover:border-indigo-300 hover:shadow-md'
    }`}
  >
    <div className={`p-2 rounded-lg mb-3 transition-colors ${
      selected ? 'bg-indigo-100 text-indigo-700' : 'bg-indigo-50 text-indigo-600'
    }`}>
      {icon}
    </div>
    <h3 className="text-lg font-semibold text-gray-800 mb-1">{title}</h3>
    <p className="text-sm text-gray-600">{description}</p>
  </button>
);

const MetricItem = ({ label, description }) => (
  <li className="flex flex-col">
    <span className="font-medium text-gray-800">{label}</span>
    <span className="text-sm">{description}</span>
  </li>
);

export default RegressionDashboard;