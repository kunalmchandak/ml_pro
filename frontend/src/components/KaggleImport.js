import React, { useState } from 'react';
import axios from 'axios';

const KaggleImport = ({ onImportSuccess }) => {
    const [datasetSlug, setDatasetSlug] = useState('');
    const [fileName, setFileName] = useState('');
    const [targetColumn, setTargetColumn] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [importInfo, setImportInfo] = useState(null);

    const handleImport = async () => {
        if (!datasetSlug || !fileName) {
            setError('Please provide both dataset slug and file name');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const response = await axios.post('http://localhost:8000/import_kaggle', {
                dataset_slug: datasetSlug,
                file_name: fileName,
                target_column: targetColumn || undefined
            });

            if (response.data.success) {
                setImportInfo(response.data);
                if (onImportSuccess) {
                    onImportSuccess({
                        name: fileName,
                        info: response.data.info,
                        suggestedTarget: response.data.suggested_target
                    });
                }
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to import dataset');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Import from Kaggle</h3>
            
            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Dataset Slug
                    </label>
                    <input
                        type="text"
                        placeholder="e.g., uciml/iris"
                        value={datasetSlug}
                        onChange={(e) => setDatasetSlug(e.target.value)}
                        className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                    />
                    <p className="mt-1 text-sm text-gray-500">
                        Format: username/dataset-name
                    </p>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        File Name
                    </label>
                    <input
                        type="text"
                        placeholder="e.g., Iris.csv"
                        value={fileName}
                        onChange={(e) => setFileName(e.target.value)}
                        className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Target Column (optional)
                    </label>
                    <input
                        type="text"
                        placeholder="e.g., target or class"
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        className="w-full p-2 border rounded focus:ring-blue-500 focus:border-blue-500"
                    />
                </div>

                {error && (
                    <div className="text-red-600 text-sm">
                        {error}
                    </div>
                )}

                <button
                    onClick={handleImport}
                    disabled={loading || !datasetSlug || !fileName}
                    className="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                    {loading ? 'Importing...' : 'Import Dataset'}
                </button>

                {importInfo && (
                    <div className="mt-4 p-4 bg-gray-50 rounded">
                        <h4 className="font-medium mb-2">Import Summary</h4>
                        <div className="text-sm">
                            <p>Rows: {importInfo.info.shape[0]}</p>
                            <p>Columns: {importInfo.info.shape[1]}</p>
                            {importInfo.suggested_target && (
                                <p>Suggested Target: {importInfo.suggested_target}</p>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default KaggleImport;