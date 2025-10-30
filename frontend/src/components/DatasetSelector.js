import React, { useState } from 'react';
import DatasetUpload from './DatasetUpload';
import KaggleImport from './KaggleImport';

const DatasetSelector = ({ 
    selectedDataset,
    onDatasetSelect,
    compatibleDatasets,
    userDatasets,
    onUserDataset
}) => {
    const [showKaggleImport, setShowKaggleImport] = useState(false);

    const handleKaggleImport = (importedDataset) => {
        if (onUserDataset) {
            onUserDataset({
                name: importedDataset.name,
                file: null,  // Kaggle imports are already on server
                analysis: {
                    shape: importedDataset.info.shape,
                    columns: importedDataset.info.columns,
                    dtypes: importedDataset.info.dtypes,
                    suggestedTarget: importedDataset.suggestedTarget
                }
            });
        }
        setShowKaggleImport(false);
    };

    const animationClasses = {
        card: "transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-purple-200/50",
        button: "transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 shadow-lg hover:shadow-xl",
        input: "transition-all duration-200 focus:ring-2 focus:ring-purple-400",
    };

    return (
        <>
            <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">3. Select or Import Dataset</h2>
            <div className="space-y-6">
                {/* Dataset Selection */}
                <div>
                    <label className="block mb-2 text-lg font-medium text-gray-700">Select a dataset:</label>
                    <select
                        value={selectedDataset}
                        onChange={(e) => onDatasetSelect(e.target.value)}
                        className={`w-full p-3 rounded-lg border-2 border-purple-200 bg-white/80 focus:border-purple-400 ${animationClasses.input}`}
                    >
                        <option value="">Select a dataset</option>
                        {Object.entries(compatibleDatasets).map(([name, info]) => (
                            <option key={name} value={name}>
                                {name} {info.types && info.types.includes('association') 
                                    ? '(association rule mining)' 
                                    : `(${info.samples} samples, ${info.features} features)`}
                            </option>
                        ))}
                        {userDatasets.map(ds => (
                            <option key={ds.name} value={ds.name}>
                                {ds.name} (user-uploaded)
                            </option>
                        ))}
                    </select>
                </div>

                {/* Import Options */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className={`${animationClasses.card} bg-white/80 p-6 rounded-xl border-2 border-purple-200 hover:border-purple-400`}>
                        <h3 className="text-xl font-semibold mb-4 text-gray-800">Upload Local File</h3>
                        <DatasetUpload onFileUpload={() => {}} onAnalysis={onUserDataset} />
                    </div>

                    <div className={`${animationClasses.card} bg-white/80 p-6 rounded-xl border-2 border-purple-200 hover:border-purple-400`}>
                        <h3 className="text-xl font-semibold mb-4 text-gray-800">Import from Kaggle</h3>
                        {showKaggleImport ? (
                            <KaggleImport onImportSuccess={handleKaggleImport} />
                        ) : (
                            <button
                                onClick={() => setShowKaggleImport(true)}
                                className={`${animationClasses.button} w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                                text-white font-bold py-3 px-6 rounded-lg`}
                            >
                                Import from Kaggle
                            </button>
                        )}
                    </div>
                </div>

                {/* Setup Instructions */}
                {showKaggleImport && (
                    <div className={`${animationClasses.card} mt-6 p-6 bg-purple-50 rounded-xl border-2 border-purple-200`}>
                        <h4 className="text-lg font-semibold mb-3 text-gray-800">Kaggle API Setup</h4>
                        <ol className="list-decimal list-inside space-y-3 text-gray-600">
                            <li>Go to your Kaggle account settings</li>
                            <li>Click "Create New API Token" to download kaggle.json</li>
                            <li>Place kaggle.json in ~/.kaggle/ directory</li>
                            <li>Find a dataset on Kaggle and note its slug (username/dataset-name)</li>
                            <li>Note the specific file name you want to import</li>
                        </ol>
                    </div>
                )}
            </div>
        </>
    );
};

export default DatasetSelector;
