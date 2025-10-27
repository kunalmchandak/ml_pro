import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  BeakerIcon,
  ChartPieIcon,
  LightBulbIcon,
  DocumentChartBarIcon,
  ArrowPathIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

// AlgorithmCard is defined below to keep the file organized.

const ClassificationDashboard = () => {
  const [algorithm, setAlgorithm] = useState('logistic_regression');
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
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-purple-50/30 to-indigo-50/20 relative overflow-hidden">
      {/* Enhanced Background Effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute w-[500px] h-[500px] -top-48 -left-48 bg-purple-300/30 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob motion-safe:animate-pulse"></div>
        <div className="absolute w-[500px] h-[500px] top-1/2 right-0 bg-indigo-300/30 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob-reverse motion-safe:animate-pulse"></div>
        <div className="absolute w-[400px] h-[400px] bottom-0 left-1/4 bg-pink-300/20 rounded-full mix-blend-multiply filter blur-3xl opacity-40 animate-float"></div>
      </div>

      <div className="container mx-auto px-6 py-12">
        {/* Header Section */}
        <div className="mb-12 text-center relative">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-400/20 via-transparent to-indigo-400/20 blur-3xl -z-10"></div>
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-fuchsia-500 to-indigo-600 mb-4 animate-gradient-x">
            Classification Dashboard
          </h1>
          <p className="text-gray-600/90 text-lg max-w-2xl mx-auto backdrop-blur-sm py-2">
            Choose your classification algorithm and dataset to start training your model
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Main Content - Algorithm Selection */}
          <div className="lg:col-span-8">
            <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl border border-purple-200/50 shadow-lg
                         hover:shadow-2xl transition-all duration-500 hover:bg-white/90
                         before:absolute before:inset-0 before:bg-gradient-to-r before:from-purple-500/10 before:via-transparent before:to-indigo-500/10 
                         before:opacity-0 before:transition-opacity hover:before:opacity-100 relative">
              <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 mb-6 flex items-center gap-2">
                <BeakerIcon className="w-6 h-6 text-purple-600" />
                Select Classification Algorithm
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                <AlgorithmCard
                  title="Logistic Regression"
                  description="Basic linear classifier for binary/multiclass problems"
                  selected={algorithm === 'logistic_regression'}
                  onClick={() => setAlgorithm('logistic_regression')}
                  icon={<ChartPieIcon className="w-6 h-6" />}
                />
                <AlgorithmCard
                  title="Random Forest"
                  description="Ensemble of decision trees for robust classification"
                  selected={algorithm === 'random_forest'}
                  onClick={() => setAlgorithm('random_forest')}
                  icon={<DocumentChartBarIcon className="w-6 h-6" />}
                />
                <AlgorithmCard
                  title="SVM"
                  description="Support Vector Machine for complex decision boundaries"
                  selected={algorithm === 'svm'}
                  onClick={() => setAlgorithm('svm')}
                  icon={<SparklesIcon className="w-6 h-6" />}
                />
                <AlgorithmCard
                  title="Neural Network"
                  description="Deep learning for complex pattern recognition"
                  selected={algorithm === 'neural_network'}
                  onClick={() => setAlgorithm('neural_network')}
                  icon={<LightBulbIcon className="w-6 h-6" />}
                />
              </div>

              {algorithmInfo && (
                <div className="bg-purple-50 rounded-xl p-6 border border-purple-100">
                  <h3 className="font-semibold text-purple-800 mb-2">Algorithm Information</h3>
                  <p className="text-purple-600">{algorithmInfo.description}</p>
                </div>
              )}
            </div>

            {/* Dataset Selection */}
            <div className="bg-white/90 backdrop-blur-lg p-8 rounded-2xl border border-purple-200 shadow-lg mt-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <ArrowPathIcon className="w-6 h-6 text-purple-600" />
                Select Dataset
              </h2>
              
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full p-3 rounded-xl bg-white border border-purple-200 
                         focus:border-purple-400 focus:ring focus:ring-purple-200 
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
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white 
                         py-3 px-6 rounded-xl font-semibold shadow-lg
                         hover:from-purple-700 hover:to-indigo-700 
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
              <div className="bg-white/90 backdrop-blur-lg p-6 rounded-2xl border border-purple-200 shadow-lg mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Classification Metrics</h3>
                <ul className="space-y-3 text-gray-600">
                  <MetricItem label="Accuracy" description="Overall correctness of predictions" />
                  <MetricItem label="Precision" description="Ratio of true positives to predicted positives" />
                  <MetricItem label="Recall" description="Ratio of true positives to actual positives" />
                  <MetricItem label="F1 Score" description="Harmonic mean of precision and recall" />
                </ul>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-indigo-50 p-6 rounded-2xl border border-purple-200 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Tips</h3>
                <ul className="space-y-3 text-gray-600">
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600">•</span>
                    Choose algorithms based on your data size and complexity
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600">•</span>
                    Ensure your dataset is properly preprocessed
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600">•</span>
                    Consider cross-validation for robust evaluation
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
    className={`w-full text-left p-6 rounded-xl border-2 transition-all duration-300
                ${selected 
                  ? 'bg-gradient-to-br from-purple-50/90 to-indigo-50/90 border-purple-400 shadow-lg scale-[1.02]' 
                  : 'bg-white/80 border-purple-100 hover:border-purple-300 hover:shadow-md hover:scale-[1.01]'
                }
                backdrop-blur-sm group relative overflow-hidden`}
  >
    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-transparent to-indigo-500/10 opacity-0 
                    transition-opacity group-hover:opacity-100"></div>
    <div className="flex items-start gap-4 relative z-10">
      <div className={`p-3 rounded-lg transition-all duration-300 transform group-hover:scale-110
                      ${selected ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-600 group-hover:bg-purple-50 group-hover:text-purple-500'}`}>
        {icon}
      </div>
      <div>
        <h3 className={`font-semibold mb-2 transition-colors
                       ${selected ? 'text-purple-800' : 'text-gray-800'}`}>
          {title}
        </h3>
        <p className={`text-sm transition-colors
                      ${selected ? 'text-purple-600' : 'text-gray-600'}`}>
          {description}
        </p>
      </div>
    </div>
  </button>
);

const MetricItem = ({ label, description }) => (
  <li className="flex flex-col">
    <span className="font-medium text-gray-800">{label}</span>
    <span className="text-sm">{description}</span>
  </li>
);

export default ClassificationDashboard;