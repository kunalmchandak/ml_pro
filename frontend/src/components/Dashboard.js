
import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import DatasetUpload from './DatasetUpload';
import ClusterTools from './ClusterTools';
import AssociationRules from './AssociationRules';
import DatasetSelector from './DatasetSelector';
import AlgorithmSelector from './AlgorithmSelector';

const Dashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // core selections
  const [algorithm, setAlgorithm] = useState('');
  const [compatibleDatasets, setCompatibleDatasets] = useState({});
  const [selectedDataset, setSelectedDataset] = useState('');
  const [algorithmInfo, setAlgorithmInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // user datasets
  const [userDatasets, setUserDatasets] = useState([]);
  const [selectedUserDataset, setSelectedUserDataset] = useState(null);
  const [userTarget, setUserTarget] = useState('');
  const [userType, setUserType] = useState('');

  // UI flow
  const [category, setCategory] = useState('');
  const [subcategory, setSubcategory] = useState('');

  // animation classes
  const animationClasses = {
    card: "transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-purple-200/50",
    button: "transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 shadow-lg hover:shadow-xl",
    input: "transition-all duration-200 focus:ring-2 focus:ring-purple-400",
  };

  // algorithm groups for selector
  const groupedAlgorithms = {
    'Supervised - Classification': ['logistic_regression','random_forest','svm','knn','decision_tree','gaussian_nb','multinomial_nb','bernoulli_nb','gradient_boosting_classifier','xgboost_classifier','lightgbm_classifier','catboost_classifier'],
    'Supervised - Regression': ['linear_regression','ridge','lasso','decision_tree_regressor','random_forest_regressor','gradient_boosting','xgboost','lightgbm','svr','knn_regressor'],
    'Unsupervised - Clustering': ['kmeans','dbscan','agglomerative','birch'],
    'Association Rules': ['apriori','fp_growth']
  };

  const supervisedAlgorithms = Object.values(groupedAlgorithms).flat().filter(a => a && !['kmeans','dbscan','agglomerative','birch','apriori','fp_growth'].includes(a));
  const unsupervisedAlgorithms = groupedAlgorithms['Unsupervised - Clustering'];
  const associationAlgorithms = groupedAlgorithms['Association Rules'];

  const isSupervised = supervisedAlgorithms.includes(algorithm);
  const isUnsupervised = unsupervisedAlgorithms.includes(algorithm);
  const isAssociation = associationAlgorithms.includes(algorithm);

  useEffect(() => {
    if (location.state) {
      setAlgorithm(location.state.algorithm || '');
      setSelectedDataset(location.state.dataset || '');
    }
  }, [location.state]);

  useEffect(() => {
    if (!algorithm) return;
    (async () => {
      try {
        const res = await axios.get(`http://localhost:8000/datasets/compatible/${algorithm}`);
        setCompatibleDatasets(res.data.compatible_datasets || {});
        setAlgorithmInfo(res.data.algorithm_info || null);
      } catch (err) {
        console.error('fetchCompatibleDatasets error', err);
      }
    })();
    if (!location.state?.dataset) setSelectedDataset('');
  }, [algorithm]);

  const handleTrain = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const payload = {
        algorithm,
        dataset_name: selectedDataset,
        params: {}
      };
      if (selectedUserDataset) {
        payload.is_user_dataset = true;
        payload.target_column = userTarget;
      }
      const resp = await axios.post('http://localhost:8000/train', payload);
      navigate('/visualization', { state: { results: resp.data } });
    } catch (err) {
      console.error(err);
      setErrorMsg(err?.response?.data?.detail || 'Training failed');
    } finally {
      setLoading(false);
    }
  };

  // preserve existing dataset handler behavior
  const handleUserDataset = (analysis, file) => {
    // Kaggle import style: single object
    if (analysis && typeof analysis === 'object' && analysis.name && analysis.analysis) {
      const imported = analysis;
      const meta = imported.analysis || {};
      const name = imported.name;
      const fileObj = imported.file || { name, kaggle: true };
      const targetColumn = meta.suggestedTarget || meta.selectedTarget || meta.target_column || null;
      const taskType = meta.inferred_type || meta.task_type || 'unspecified';
      const newDataset = { name, analysis: { ...meta, selectedTarget: targetColumn, selectedType: taskType }, file: fileObj };
      setUserDatasets(prev => {
        const idx = prev.findIndex(d => d.name === name);
        if (idx >= 0) {
          const copy = [...prev]; copy[idx] = newDataset; return copy;
        }
        return [...prev, newDataset];
      });
      setSelectedDataset(name);
      setSelectedUserDataset(newDataset);
      setUserTarget(targetColumn || '');
      setUserType(taskType || '');
      return;
    }

    if (!file) return;
    const targetColumn = analysis.target_column || analysis.selected_target;
    const taskType = analysis.task_type || analysis.inferred_type;
    const newDataset = { name: file.name, analysis: { ...analysis, selectedTarget: targetColumn, selectedType: taskType }, file };
    setUserDatasets(prev => {
      const idx = prev.findIndex(d => d.name === file.name);
      if (idx >= 0) { const copy = [...prev]; copy[idx] = newDataset; return copy; }
      return [...prev, newDataset];
    });
    setSelectedDataset(file.name);
    setSelectedUserDataset(newDataset);
    setUserTarget(targetColumn || '');
    setUserType(taskType || '');
  };

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
        <h1 className="text-4xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">ML Dashboard</h1>

        <div className="grid grid-cols-1 gap-8">
          {/* Category selector */}
          <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
            <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">1. Choose a Category</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className={`${animationClasses.card} p-6 bg-white/80 rounded-xl border-2 border-purple-200 hover:border-purple-400`} onClick={() => { setCategory('supervised'); setSubcategory(''); }}>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Supervised Learning</h3>
                <p className="text-sm text-gray-600 mb-4">Classification & Regression</p>
                <div className="flex gap-3">
                  <button 
                    className={`${animationClasses.button} px-4 py-2 rounded-lg text-sm font-medium ${
                      subcategory === 'classification' 
                      ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' 
                      : 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                    }`} 
                    onClick={() => { setCategory('supervised'); setSubcategory('classification'); setAlgorithm('logistic_regression'); }}
                  >
                    Classification
                  </button>
                  <button 
                    className={`${animationClasses.button} px-4 py-2 rounded-lg text-sm font-medium ${
                      subcategory === 'regression' 
                      ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' 
                      : 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                    }`} 
                    onClick={() => { setCategory('supervised'); setSubcategory('regression'); setAlgorithm('linear_regression'); }}
                  >
                    Regression
                  </button>
                </div>
              </div>

              <div className={`${animationClasses.card} p-6 bg-white/80 rounded-xl border-2 border-purple-200 hover:border-purple-400`}>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Unsupervised Learning</h3>
                <p className="text-sm text-gray-600 mb-4">Clustering & Association</p>
                <div className="flex gap-3">
                  <button 
                    className={`${animationClasses.button} px-4 py-2 rounded-lg text-sm font-medium ${
                      subcategory === 'clustering' 
                      ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' 
                      : 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                    }`} 
                    onClick={(e) => {
                      e.stopPropagation();
                      setCategory('unsupervised');
                      setSubcategory('clustering');
                      setAlgorithm('kmeans');
                    }}
                  >
                    Clustering
                  </button>
                  <button 
                    className={`${animationClasses.button} px-4 py-2 rounded-lg text-sm font-medium ${
                      subcategory === 'association' 
                      ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' 
                      : 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                    }`} 
                    onClick={(e) => {
                      e.stopPropagation();
                      setCategory('unsupervised');
                      setSubcategory('association');
                      setAlgorithm('apriori');
                    }}
                  >
                    Association
                  </button>
                </div>
              </div>
            </div>

            <div className="mt-6 text-sm font-medium text-gray-600">
              Selected: <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600 font-semibold">
                {category ? (subcategory ? `${category} - ${subcategory}` : category) : 'None'}
              </span>
            </div>
          </div>

          {/* Algorithm selector */}
          {(subcategory || algorithm) && (
            <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
              <AlgorithmSelector algorithms={groupedAlgorithms} selected={algorithm} onSelect={(alg) => setAlgorithm(alg)} />
              {algorithmInfo && (
                <div className="mt-4 p-4 bg-purple-50 rounded-lg text-sm text-gray-700">
                  {algorithmInfo.description}
                </div>
              )}
            </div>
          )}

          {/* Dataset selector / uploader */}
          <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg">
            <DatasetSelector selectedDataset={selectedDataset} onDatasetSelect={(value) => {
              setSelectedDataset(value);
              const userDs = userDatasets.find(ds => ds.name === value);
              setSelectedUserDataset(userDs || null);
              if (userDs) { setUserTarget(userDs.analysis.selectedTarget || ''); setUserType(userDs.analysis.selectedType || ''); }
            }} compatibleDatasets={compatibleDatasets} userDatasets={userDatasets} onUserDataset={handleUserDataset} />
          </div>

          {selectedUserDataset && (
            <div className="bg-white/90 backdrop-blur-lg p-4 rounded-xl border-2 border-purple-200 shadow-lg">
              <div className="font-medium text-gray-700"><span className="text-purple-600">Target column:</span> {userTarget}</div>
              <div className="font-medium text-gray-700"><span className="text-purple-600">Task type:</span> {userType}</div>
            </div>
          )}

          {/* Tools */}
          {isUnsupervised && (
            <div className="bg-white/90 backdrop-blur-lg rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg overflow-hidden">
              <ClusterTools selectedDataset={selectedDataset} isUserDataset={!!selectedUserDataset} />
            </div>
          )}
          {isAssociation && (
            <div className="bg-white/90 backdrop-blur-lg rounded-xl border-4 border-purple-400 hover:border-purple-600 transition-all duration-300 shadow-lg overflow-hidden">
              <AssociationRules selectedDataset={selectedDataset} isUserDataset={!!selectedUserDataset} />
            </div>
          )}

          {/* Train button for supervised */}
          {isSupervised && (
            <div className="flex flex-col items-end">
              {errorMsg && <div className="mb-2 text-red-600 text-sm font-medium">{errorMsg}</div>}
              <button 
                onClick={handleTrain} 
                disabled={loading || !selectedDataset || (selectedUserDataset && (!userTarget || !userType))} 
                className={`${animationClasses.button} bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                  text-white font-bold py-3 px-8 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {loading ? 'Training...' : 'Train Model'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
