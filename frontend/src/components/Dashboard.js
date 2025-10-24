
import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import DatasetUpload from './DatasetUpload';
import ClusterTools from './ClusterTools';
import AssociationRules from './AssociationRules';
import DatasetSelector from './DatasetSelector';


const Dashboard = () => {

  // State declarations first
  const [algorithm, setAlgorithm] = useState('kmeans'); // Default to kmeans clustering
  const [compatibleDatasets, setCompatibleDatasets] = useState({});
  const [selectedDataset, setSelectedDataset] = useState('');
  const [algorithmInfo, setAlgorithmInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [userDatasets, setUserDatasets] = useState([]); // {name, analysis, file}
  const [selectedUserDataset, setSelectedUserDataset] = useState(null);
  const [userTarget, setUserTarget] = useState('');
  const [userType, setUserType] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  // Algorithm type detection
  const supervisedAlgorithms = [
    'logistic_regression', 'random_forest', 'svm', 'knn', 'decision_tree', 'gaussian_nb', 'multinomial_nb', 'bernoulli_nb',
    'gradient_boosting_classifier', 'xgboost_classifier', 'lightgbm_classifier', 'catboost_classifier',
    'linear_regression', 'ridge', 'lasso', 'decision_tree_regressor', 'random_forest_regressor', 'gradient_boosting',
    'xgboost', 'lightgbm', 'svr', 'knn_regressor'
  ];
  const unsupervisedAlgorithms = ['kmeans', 'dbscan', 'agglomerative', 'birch'];
  const associationAlgorithms = ['apriori', 'fp_growth'];
  
  // Determine algorithm type
  const isSupervised = supervisedAlgorithms.includes(algorithm);
  const isUnsupervised = unsupervisedAlgorithms.includes(algorithm);
  const isAssociation = associationAlgorithms.includes(algorithm);
  
  useEffect(() => {
    console.log('Current state:', {
      algorithm,
      selectedDataset,
      isSupervised,
      isUnsupervised,
      isAssociation
    });
  }, [algorithm, selectedDataset, isSupervised, isUnsupervised, isAssociation]);

  // Handle retrain state
  useEffect(() => {
    if (location.state) {
      setAlgorithm(location.state.algorithm || '');
      setSelectedDataset(location.state.dataset || '');
    }
  }, [location.state]);

  // Fetch compatible datasets when algorithm changes
  useEffect(() => {
    if (algorithm) {
      fetchCompatibleDatasets(algorithm);
    }
    // Reset dataset selection when algorithm changes (but not if from retrain)
    if (!location.state?.dataset) {
      setSelectedDataset('');
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
      let trainPayload = {
        algorithm,
        dataset_name: selectedDataset,
        params: {}
      };
      // If user dataset selected, add extra params
      if (selectedUserDataset) {
        trainPayload = {
          ...trainPayload,
          is_user_dataset: true,
          target_column: userTarget,
        };
      }
      const response = await axios.post('http://localhost:8000/train', trainPayload);
      navigate('/visualization', { state: { results: response.data } });
    } catch (error) {
      console.error('Training error:', error);
      setErrorMsg(error?.response?.data?.detail || 'Training failed. Please check your selections and dataset.');
    } finally {
      setLoading(false);
    }
  };

  // Handle user dataset upload/analysis
  // Supports two call styles:
  // 1) (analysis, file) - from local upload (DatasetUpload)
  // 2) (importedObject) - single object from Kaggle import (DatasetSelector/KaggleImport)
  const handleUserDataset = (analysis, file) => {
    // If caller passed a single object (from KaggleImport/DatasetSelector), normalize it
    if (analysis && typeof analysis === 'object' && analysis.name && analysis.analysis) {
      const imported = analysis; // rename for clarity
      // Use the embedded analysis and file (may be null)
      const meta = imported.analysis || {};
      const name = imported.name;
      const fileObj = imported.file || { name, kaggle: true };
      const targetColumn = meta.suggestedTarget || meta.selectedTarget || meta.target_column || meta.selected_target || null;
      const taskType = meta.inferred_type || meta.task_type || 'unspecified';

      setUserDatasets(prev => {
        const newDataset = {
          name,
          analysis: {
            ...meta,
            selectedTarget: targetColumn,
            selectedType: taskType
          },
          file: fileObj
        };
        const existingIndex = prev.findIndex(ds => ds.name === name);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = newDataset;
          return updated;
        }
        return [...prev, newDataset];
      });

      // Auto-select
      setSelectedDataset(name);
      setSelectedUserDataset({ name, analysis: { ...meta, selectedTarget: targetColumn, selectedType: taskType }, file: fileObj });
      setUserTarget(targetColumn || '');
      setUserType(taskType || '');
      return;
    }

    // Original path: analysis + file provided (local upload)
    if (!file) return;
    const targetColumn = analysis.target_column || analysis.selected_target;
    const taskType = analysis.task_type || analysis.inferred_type;

    setUserDatasets(prev => {
      // Avoid duplicates
      const newDataset = {
        name: file.name,
        analysis: {
          ...analysis,
          selectedTarget: targetColumn,
          selectedType: taskType
        },
        file
      };
      
      // Update existing dataset or add new one
      const existingIndex = prev.findIndex(ds => ds.name === file.name);
      if (existingIndex >= 0) {
        const updatedDatasets = [...prev];
        updatedDatasets[existingIndex] = newDataset;
        return updatedDatasets;
      }
      return [...prev, newDataset];
    });

    // Auto-select the newly uploaded dataset so tools can render immediately
    setSelectedDataset(file.name);
    setSelectedUserDataset({
      name: file.name,
      analysis: {
        ...analysis,
        selectedTarget: targetColumn,
        selectedType: taskType
      },
      file
    });
    setUserTarget(targetColumn || '');
    setUserType(taskType || '');
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">ML Dashboard</h1>
      <div className="grid grid-cols-1 gap-6">
        {/* 1. Select Algorithm */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">1. Select Algorithm</h2>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="">Choose an algorithm</option>
            <optgroup label="Supervised Learning - Classification">
              <option value="logistic_regression">Logistic Regression</option>
              <option value="random_forest">Random Forest</option>
              <option value="svm">SVM</option>
              <option value="knn">K-Nearest Neighbors</option>
              <option value="decision_tree">Decision Tree</option>
              <option value="gaussian_nb">Gaussian Naive Bayes</option>
              <option value="multinomial_nb">Multinomial Naive Bayes</option>
              <option value="bernoulli_nb">Bernoulli Naive Bayes</option>
              <option value="gradient_boosting_classifier">Gradient Boosting Classifier</option>
              <option value="xgboost_classifier">XGBoost Classifier</option>
              <option value="lightgbm_classifier">LightGBM Classifier</option>
              <option value="catboost_classifier">CatBoost Classifier</option>
            </optgroup>
            <optgroup label="Supervised Learning - Regression">
              <option value="linear_regression">Linear Regression</option>
              <option value="ridge">Ridge Regression</option>
              <option value="lasso">Lasso Regression</option>
              <option value="decision_tree_regressor">Decision Tree Regressor</option>
              <option value="random_forest_regressor">Random Forest Regressor</option>
              <option value="gradient_boosting">Gradient Boosting Regressor</option>
              <option value="xgboost">XGBoost Regressor</option>
              <option value="lightgbm">LightGBM Regressor</option>
              <option value="svr">Support Vector Regressor</option>
              <option value="knn_regressor">K-Nearest Neighbors Regressor</option>
            </optgroup>
            <optgroup label="Unsupervised Learning">
              <option value="kmeans">K-Means Clustering</option>
              <option value="dbscan">DBSCAN</option>
              <option value="agglomerative">Agglomerative Clustering</option>
              <option value="birch">BIRCH Clustering</option>
            </optgroup>
            <optgroup label="Association Rules">
              <option value="apriori">Apriori (Association Rules)</option>
              <option value="fp_growth">FP-Growth (Association Rules)</option>
            </optgroup>
          </select>
          {algorithmInfo && (
            <div className="mt-2 text-sm text-gray-600">
              {algorithmInfo.description}
            </div>
          )}
        </div>

        {/* 2. Upload or Select Dataset */}
        <DatasetSelector
          selectedDataset={selectedDataset}
          onDatasetSelect={(value) => {
            console.log('Dataset selected:', value);
            setSelectedDataset(value);
            // If user dataset, set extra info
            const userDs = userDatasets.find(ds => ds.name === value);
            console.log('User dataset found:', userDs);
            setSelectedUserDataset(userDs || null);
            if (userDs) {
              setUserTarget(userDs.analysis.selectedTarget || '');
              setUserType(userDs.analysis.selectedType || '');
            } else {
              setUserTarget('');
              setUserType('');
            }
          }}
          compatibleDatasets={compatibleDatasets}
          userDatasets={userDatasets}
          onUserDataset={handleUserDataset}
        />

        {selectedUserDataset && (
          <div className="mt-2 bg-white p-4 rounded-lg shadow">
            <div><strong>Target column:</strong> {userTarget}</div>
            <div><strong>Task type:</strong> {userType}</div>
          </div>
        )}
        </div>

          {/* Show relevant tools based on algorithm type */}
        {isUnsupervised && (
          <ClusterTools selectedDataset={selectedDataset} isUserDataset={!!selectedUserDataset} />
        )}
        {isAssociation && (
          <AssociationRules selectedDataset={selectedDataset} isUserDataset={!!selectedUserDataset} />
        )}

        {/* 3. Train Model button only for supervised/regression algorithms */}
        {isSupervised && (
          <div className="flex flex-col items-end">
            {errorMsg && (
              <div className="mb-2 text-red-600 text-sm">{errorMsg}</div>
            )}
            <button
              onClick={handleTrain}
              disabled={
                loading ||
                !selectedDataset ||
                (selectedUserDataset && (!userTarget || !userType))
              }
              className="mt-6 bg-blue-600 text-white px-6 py-2 rounded-lg 
                       hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Training...' : 'Train Model'}
            </button>
          </div>
        )}
      </div>
  );
};

export default Dashboard;
