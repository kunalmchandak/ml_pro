import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import {
  CubeIcon,
  Squares2X2Icon,
  SparklesIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

const ClusteringDashboard = () => {
  const [algorithm, setAlgorithm] = useState('kmeans');
  const [compatibleDatasets, setCompatibleDatasets] = useState({});
  const [selectedDataset, setSelectedDataset] = useState('');
  const [algorithmInfo, setAlgorithmInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (location.state?.algorithm) setAlgorithm(location.state.algorithm);
  }, [location.state]);

  useEffect(() => {
    if (algorithm) fetchCompatibleDatasets(algorithm);
  }, [algorithm]);

  const fetchCompatibleDatasets = async (algo) => {
    try {
      const response = await axios.get(`http://localhost:8000/datasets/compatible/${algo}`);
      setCompatibleDatasets(response.data.compatible_datasets);
      setAlgorithmInfo(response.data.algorithm_info);
    } catch (err) {
      console.error(err);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await axios.post('http://localhost:8000/train', { algorithm, dataset_name: selectedDataset, params: {} });
      navigate('/visualization', { state: { results: res.data } });
    } catch (err) {
      console.error(err);
      setErrorMsg('Training failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-purple-50/30 to-indigo-50/20 relative overflow-hidden">
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute w-[500px] h-[500px] -top-48 -left-48 bg-purple-300/20 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob"></div>
        <div className="absolute w-[500px] h-[500px] top-1/2 right-0 bg-indigo-300/20 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob-reverse"></div>
      </div>

      <div className="container mx-auto px-6 py-12">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">Clustering Dashboard</h1>
          <p className="text-gray-600 mt-2">Choose a clustering algorithm and a dataset to explore structure in your data.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-8">
            <div className="bg-white/90 backdrop-blur-lg p-8 rounded-2xl border border-purple-200 shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <CubeIcon className="w-6 h-6 text-purple-600" /> Select Clustering Algorithm
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                <AlgorithmCard title="K-Means" description="Partition data into k clusters" selected={algorithm === 'kmeans'} onClick={() => setAlgorithm('kmeans')} icon={<Squares2X2Icon className="w-6 h-6" />} />
                <AlgorithmCard title="DBSCAN" description="Density-based clustering" selected={algorithm === 'dbscan'} onClick={() => setAlgorithm('dbscan')} icon={<SparklesIcon className="w-6 h-6" />} />
                <AlgorithmCard title="Agglomerative" description="Hierarchical clustering" selected={algorithm === 'agglomerative'} onClick={() => setAlgorithm('agglomerative')} icon={<Squares2X2Icon className="w-6 h-6" />} />
                <AlgorithmCard title="BIRCH" description="Balanced iterative reduction" selected={algorithm === 'birch'} onClick={() => setAlgorithm('birch')} icon={<SparklesIcon className="w-6 h-6" />} />
              </div>

              {algorithmInfo && (
                <div className="bg-purple-50 rounded-xl p-6 border border-purple-100">
                  <h3 className="font-semibold text-purple-800 mb-2">Algorithm Information</h3>
                  <p className="text-purple-600">{algorithmInfo.description}</p>
                </div>
              )}
            </div>

            <div className="bg-white/90 backdrop-blur-lg p-8 rounded-2xl border border-purple-200 shadow-lg mt-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <ArrowPathIcon className="w-6 h-6 text-purple-600" /> Select Dataset
              </h2>

              <select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)} className="w-full p-3 rounded-xl bg-white border border-purple-200 mb-4">
                <option value="">Choose a dataset</option>
                {Object.entries(compatibleDatasets).map(([key, info]) => (
                  <option key={key} value={key}>{info && info.description ? `${key} — ${info.description}` : `${key}${info && info.samples ? ` (${info.samples} samples)` : ''}`}</option>
                ))}
              </select>

              <button onClick={handleTrain} disabled={loading || !selectedDataset} className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 px-6 rounded-xl font-semibold shadow-lg hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 transition-all duration-300">
                {loading ? 'Running...' : 'Run Clustering'}
              </button>

              {errorMsg && <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl text-red-600">{errorMsg}</div>}
            </div>
          </div>

          <div className="lg:col-span-4">
            <div className="sticky top-6 bg-white/90 backdrop-blur-lg p-6 rounded-2xl border border-purple-200 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Clustering Tips</h3>
              <ul className="space-y-3 text-gray-600">
                <li>Scale your features before clustering</li>
                <li>Use silhouette score to compare clusterings</li>
                <li>For DBSCAN tune eps and min_samples</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const AlgorithmCard = ({ title, description, selected, onClick, icon }) => (
  <button onClick={onClick} className={`flex flex-col p-6 rounded-xl border-2 transition-all duration-300 ${selected ? 'border-purple-400 bg-purple-50/50 shadow-lg scale-[1.02]' : 'border-purple-200 bg-white hover:border-purple-300 hover:shadow-md'}`}>
    <div className={`p-2 rounded-lg mb-3 transition-colors ${selected ? 'bg-purple-100 text-purple-700' : 'bg-purple-50 text-purple-600'}`}>{icon}</div>
    <h3 className="text-lg font-semibold text-gray-800 mb-1">{title}</h3>
    <p className="text-sm text-gray-600">{description}</p>
  </button>
);

export default ClusteringDashboard;
