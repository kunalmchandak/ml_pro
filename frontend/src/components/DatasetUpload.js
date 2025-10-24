import React, { useState } from 'react';
import axios from 'axios';

// Algorithm definitions
const ALGORITHMS = {
    'logistic_regression': {
        name: 'Logistic Regression',
        description: 'Best for binary/multiclass classification'
    },
    'random_forest': {
        name: 'Random Forest',
        description: 'Versatile classifier for all types'
    },
    'svm': {
        name: 'Support Vector Machine',
        description: 'Good for high-dimensional data'
    },
    'knn': {
        name: 'K-Nearest Neighbors',
        description: 'Instance-based learning algorithm'
    },
    'decision_tree': {
        name: 'Decision Tree',
        description: 'Tree-based model for classification'
    },
    'gaussian_nb': {
        name: 'Gaussian Naive Bayes',
        description: 'Probabilistic classifier based on Bayes theorem'
    },
    'multinomial_nb': {
        name: 'Multinomial Naive Bayes',
        description: 'Good for text classification'
    },
    'bernoulli_nb': {
        name: 'Bernoulli Naive Bayes',
        description: 'Naive Bayes for binary/boolean features'
    },
    'gradient_boosting_classifier': {
        name: 'Gradient Boosting Classifier',
        description: 'Powerful boosting algorithm for classification'
    },
    'xgboost_classifier': {
        name: 'XGBoost Classifier',
        description: 'High-performance implementation of gradient boosting'
    },
    'lightgbm_classifier': {
        name: 'LightGBM Classifier',
        description: 'Light and fast implementation of gradient boosting'
    },
    'catboost_classifier': {
        name: 'CatBoost Classifier',
        description: 'High-performance gradient boosting on decision trees'
    },
    'linear_regression': {
        name: 'Linear Regression',
        description: 'Linear model for continuous target prediction'
    },
    'ridge': {
        name: 'Ridge Regression',
        description: 'Linear regression with L2 regularization'
    },
    'lasso': {
        name: 'Lasso Regression',
        description: 'Linear regression with L1 regularization'
    },
    'decision_tree_regressor': {
        name: 'Decision Tree Regressor',
        description: 'Tree-based regression model'
    },
    'random_forest_regressor': {
        name: 'Random Forest Regressor',
        description: 'Ensemble of decision trees for regression'
    },
    'gradient_boosting': {
        name: 'Gradient Boosting Regressor',
        description: 'Gradient boosting for regression'
    },
    'xgboost': {
        name: 'XGBoost Regressor',
        description: 'Extreme gradient boosting for regression'
    },
    'lightgbm': {
        name: 'LightGBM Regressor',
        description: 'Light gradient boosting for regression'
    },
    'svr': {
        name: 'Support Vector Regressor',
        description: 'Support vector machine for regression'
    },
    'knn_regressor': {
        name: 'K-Nearest Neighbors Regressor',
        description: 'Instance-based regression'
    },
    'kmeans': {
        name: 'K-Means Clustering',
        description: 'Clustering algorithm for unlabeled data'
    },
    'dbscan': {
        name: 'DBSCAN',
        description: 'Density-based clustering algorithm'
    }
};

const DatasetUpload = ({ onAnalysis }) => {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const [analysis, setAnalysis] = useState(null);
    const [uploadedFile, setUploadedFile] = useState(null);
    const [applyCleaning, setApplyCleaning] = useState(true); // ask user whether to apply cleaning
    const [datasetType, setDatasetType] = useState('supervised'); // 'supervised' | 'unsupervised' | 'association'

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Check file type
        if (!file.name.endsWith('.csv') && !file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
            setError('Please upload a CSV or Excel file');
            return;
        }

        setUploading(true);
        setError('');
        
        const formData = new FormData();
        formData.append('file', file);
        // Include user's choice whether to apply automatic cleaning
    formData.append('apply_cleaning', applyCleaning ? 'true' : 'false');
    // Include dataset type so server can make heuristics for association data
    formData.append('dataset_type', datasetType);

        try {
            // Let axios set the Content-Type (including boundary) for multipart forms
            const response = await axios.post('http://localhost:8000/upload', formData);
            
            setUploadedFile(file);
            setAnalysis(response.data);
            // Include inferred task_type from datasetType so parent knows how to proceed
            onAnalysis({ ...response.data, target_column: response.data.selected_target || null, task_type: datasetType }, file);
        } catch (error) {
            console.error('Upload error:', error);
            setError(error.response?.data?.detail || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    const renderCleaningReport = (report) => {
        // Handle case where report is null or undefined
        if (!report) return null;

        // For association datasets or minimal reports, show a simplified message
        if (!report.missing_values && !report.encoded_columns && !report.removed_columns) {
            return (
                <div className="mt-2">
                    <h4 className="font-semibold">Dataset Processing:</h4>
                    <p className="text-sm text-gray-600">
                        Dataset loaded successfully. No cleaning actions were performed to preserve transaction data integrity.
                    </p>
                </div>
            );
        }

        return (
            <div className="mt-2">
                <h4 className="font-semibold">Cleaning Actions:</h4>
                {report.missing_values && Object.keys(report.missing_values).length > 0 && (
                    <div className="mt-1">
                        <strong>Missing Values Handled:</strong>
                        <ul className="list-disc pl-4">
                            {Object.entries(report.missing_values).map(([col, info]) => (
                                <li key={col} className="text-sm">
                                    {col}: {info.count} values {info.action}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                {report.encoded_columns && report.encoded_columns.length > 0 && (
                    <div className="mt-1">
                        <strong>Encoded Columns:</strong>
                        <ul className="list-disc pl-4">
                            {report.encoded_columns.map(info => (
                                <li key={info.column} className="text-sm">
                                    {info.column} ({info.unique_values} unique values)
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                {report.removed_columns && report.removed_columns.length > 0 && (
                    <div className="mt-1">
                        <strong>Removed Columns:</strong>
                        <ul className="list-disc pl-4">
                            {report.removed_columns.map(info => (
                                <li key={info.column} className="text-sm">
                                    {info.column} (reason: {info.reason})
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div>
            {/* Ask user if they want automatic cleaning */}
            <div className="mb-3 flex items-center gap-2">
                <input
                    id="applyCleaning"
                    type="checkbox"
                    checked={applyCleaning}
                    onChange={(e) => setApplyCleaning(e.target.checked)}
                    className="w-4 h-4"
                />
                <label htmlFor="applyCleaning" className="text-sm">Apply automatic cleaning & preprocessing (recommended)</label>
            </div>
            <div className="flex items-center space-x-4">
                <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileUpload}
                    className="block w-full text-sm text-gray-500
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-full file:border-0
                        file:text-sm file:font-semibold
                        file:bg-blue-50 file:text-blue-700
                        hover:file:bg-blue-100"
                />
                {uploading && <div className="text-blue-600">Uploading & Cleaning...</div>}
            </div>
            
            {error && (
                <div className="mt-2 text-red-600 text-sm">{error}</div>
            )}

            {analysis && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                    <h3 className="font-semibold mb-2">Dataset Analysis:</h3>
                    <div className="text-sm">
                        {!analysis.cleaned && (
                            <div className="mb-2 text-sm text-yellow-700">
                                Note: automatic cleaning was skipped — data shown is the raw uploaded data.
                            </div>
                        )}
                        <div>Shape: {analysis.shape[0]} rows × {analysis.shape[1]} columns</div>

                        {/* Dataset type selector: supervised/unsupervised/association */}
                        <div className="mt-4">
                            <h4 className="text-md font-semibold mb-2">Dataset Type</h4>
                            <div className="flex items-center gap-3">
                                <label className="text-sm">
                                    <input type="radio" name="datasetType" value="supervised" checked={datasetType === 'supervised'} onChange={() => setDatasetType('supervised')} className="mr-2" />
                                    Supervised
                                </label>
                                <label className="text-sm">
                                    <input type="radio" name="datasetType" value="unsupervised" checked={datasetType === 'unsupervised'} onChange={() => setDatasetType('unsupervised')} className="mr-2" />
                                    Unsupervised / Clustering
                                </label>
                                <label className="text-sm">
                                    <input type="radio" name="datasetType" value="association" checked={datasetType === 'association'} onChange={() => setDatasetType('association')} className="mr-2" />
                                    Association / Market Basket
                                </label>
                            </div>
                        </div>
                        
                        {/* Target Column Selection */}
                        {/* Only show target selection for supervised datasets */}
                        {datasetType === 'supervised' && (
                            <div className="mt-4 mb-2">
                                <h4 className="text-lg font-semibold mb-2">1. Select Target Variable</h4>
                                <p className="text-sm text-gray-600 mb-4">
                                    Choose the column you want to predict. This will determine what kind of machine learning task we'll perform.
                                </p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {analysis.target_candidates.map(col => {
                                        const colInfo = analysis.column_info[col];
                                        const isSelected = analysis.selected_target === col;
                                        return (
                                            <div key={col} 
                                                 className={`p-4 border rounded-lg cursor-pointer transition-all
                                                           ${isSelected ? 'border-blue-500 bg-blue-50' : 'hover:border-blue-300'}
                                                           ${colInfo.suggested_task.length > 0 ? 'border-green-100' : ''}`}
                                                 onClick={() => {
                                                    const updatedAnalysis = {
                                                        ...analysis,
                                                        selected_target: col,
                                                        inferred_type: colInfo.suggested_task[0] || 'unknown',
                                                        suggested_algorithms: colInfo.suggested_algorithms || [],
                                                        task_type: colInfo.suggested_task[0] || null // Add task type
                                                    };
                                                    setAnalysis(updatedAnalysis); // Update local state
                                                    if (onAnalysis) {
                                                        onAnalysis({
                                                            ...updatedAnalysis,
                                                            target_column: col, // Explicitly set target column
                                                            task_type: colInfo.suggested_task[0] || null // Explicitly set task type
                                                        }, uploadedFile);
                                                    }
                                                 }}>
                                                <div className="flex items-center justify-between">
                                                    <div className="font-medium text-lg">{col}</div>
                                                    {isSelected && (
                                                        <div className="text-blue-600">
                                                            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                                                                <path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"/>
                                                            </svg>
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="mt-2 text-sm text-gray-600">
                                                    <div className="grid grid-cols-2 gap-2">
                                                        <div>
                                                            <span className="font-medium">Type:</span> {colInfo.dtype}
                                                        </div>
                                                        <div>
                                                            <span className="font-medium">Unique:</span> {colInfo.unique_values}
                                                        </div>
                                                    </div>
                                                    {colInfo.suggested_task.length > 0 && (
                                                        <div className="mt-2">
                                                            <span className="font-medium">Suggested Tasks:</span>
                                                            <div className="flex gap-2 mt-1">
                                                                {colInfo.suggested_task.map(task => (
                                                                    <span key={task} 
                                                                          className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">
                                                                        {task}
                                                                    </span>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                    <div className="mt-2">
                                                        <span className="font-medium">Sample Values:</span>
                                                        <div className="mt-1 text-xs bg-gray-50 p-1 rounded">
                                                            {colInfo.sample_values.slice(0, 3).join(', ')}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* For unsupervised/association, allow proceeding without a target */}
                        {datasetType !== 'supervised' && (
                            <div className="mt-4 p-3 bg-yellow-50 rounded-lg">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <h4 className="font-semibold">No target needed</h4>
                                        <div className="text-sm text-gray-700">You've selected <strong>{datasetType}</strong> mode. No target column is required — proceed to analysis using clustering or association algorithms.</div>
                                    </div>
                                    <div>
                                        <button
                                            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                                            onClick={() => {
                                                // Inform parent that we're proceeding without a target
                                                const payload = {
                                                    ...analysis,
                                                    target_column: null,
                                                    task_type: datasetType === 'association' ? 'association' : 'unsupervised'
                                                };
                                                setAnalysis(payload);
                                                if (onAnalysis) onAnalysis(payload, uploadedFile);
                                            }}
                                        >
                                            Proceed without target
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                        
                        {/* Algorithm Suggestions */}
                        {analysis.selected_target && analysis.column_info[analysis.selected_target].suggested_algorithms.length > 0 && (
                            <div className="mt-6 p-4 bg-green-50 rounded-lg">
                                <h4 className="text-lg font-semibold mb-2">2. Recommended Algorithms</h4>
                                <p className="text-sm text-gray-600 mb-3">
                                    Based on your selected target variable "{analysis.selected_target}", 
                                    here are the recommended algorithms:
                                </p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {analysis.column_info[analysis.selected_target].suggested_algorithms.map(algo => (
                                        <div key={algo} className="p-3 bg-white rounded border">
                                            <div className="font-medium">
                                                {ALGORITHMS[algo]?.name || algo}
                                            </div>
                                            {ALGORITHMS[algo]?.description && (
                                                <div className="text-sm text-gray-600 mt-1">
                                                    {ALGORITHMS[algo].description}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        
                        {analysis.cleaning_report && renderCleaningReport(analysis.cleaning_report)}
                        <div className="mt-2">
                            <strong>Preview:</strong>
                            <div className="overflow-x-auto">
                                <table className="min-w-full mt-1">
                                    <thead>
                                        <tr>
                                            {analysis.columns.map(col => (
                                                <th key={col} className="px-2 py-1 bg-gray-100 text-left">{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysis.preview_data.map((row, i) => (
                                            <tr key={i}>
                                                {analysis.columns.map(col => (
                                                    <td key={col} className="px-2 py-1 border-t">{row[col]}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DatasetUpload;
