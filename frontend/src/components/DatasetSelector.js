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

    return (
        <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">2. Select or Import Dataset</h2>
            
            <div className="space-y-6">
                {/* Dataset Selection */}
                <div>
                    <label className="block mb-2">Select a dataset:</label>
                    <select
                        value={selectedDataset}
                        onChange={(e) => onDatasetSelect(e.target.value)}
                        className="w-full p-2 border rounded"
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
                <div className="flex gap-4">
                    <div className="flex-1">
                        <h3 className="text-lg font-medium mb-3">Upload Local File</h3>
                        <DatasetUpload onFileUpload={() => {}} onAnalysis={onUserDataset} />
                    </div>

                    <div className="flex-1">
                        <h3 className="text-lg font-medium mb-3">Import from Kaggle</h3>
                        {showKaggleImport ? (
                            <KaggleImport onImportSuccess={handleKaggleImport} />
                        ) : (
                            <button
                                onClick={() => setShowKaggleImport(true)}
                                className="w-full bg-green-600 text-white p-2 rounded hover:bg-green-700"
                            >
                                Import from Kaggle
                            </button>
                        )}
                    </div>
                </div>

                {/* Setup Instructions */}
                {showKaggleImport && (
                    <div className="mt-4 p-4 bg-gray-50 rounded text-sm">
                        <h4 className="font-medium mb-2">Kaggle API Setup</h4>
                        <ol className="list-decimal list-inside space-y-2">
                            <li>Go to your Kaggle account settings</li>
                            <li>Click "Create New API Token" to download kaggle.json</li>
                            <li>Place kaggle.json in ~/.kaggle/ directory</li>
                            <li>Find a dataset on Kaggle and note its slug (username/dataset-name)</li>
                            <li>Note the specific file name you want to import</li>
                        </ol>
                    </div>
                )}
            </div>
        </div>
    );
};

export default DatasetSelector;
