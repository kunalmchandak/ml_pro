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

    const animationClasses = {
        card: "transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-purple-200/50",
        button: "transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 shadow-lg hover:shadow-xl",
        input: "transition-all duration-200 focus:ring-2 focus:ring-purple-400",
    };

    return (
        <div className="bg-white/90 backdrop-blur-lg p-6 rounded-xl border-2 border-purple-200 hover:border-purple-400 transition-all duration-300">
            <div className="space-y-5">
                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Dataset Slug
                    </label>
                    <input
                        type="text"
                        placeholder="e.g., uciml/iris"
                        value={datasetSlug}
                        onChange={(e) => setDatasetSlug(e.target.value)}
                        className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`}
                    />
                    <p className="mt-2 text-sm text-purple-600">
                        Format: username/dataset-name
                    </p>
                </div>

                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                        File Name
                    </label>
                    <input
                        type="text"
                        placeholder="e.g., Iris.csv"
                        value={fileName}
                        onChange={(e) => setFileName(e.target.value)}
                        className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`}
                    />
                </div>

                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Target Column (optional)
                    </label>
                    <input
                        type="text"
                        placeholder="e.g., target or class"
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        className={`w-full p-3 border-2 border-purple-200 rounded-lg bg-white/80 ${animationClasses.input}`}
                    />
                </div>

                {error && (
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm font-medium">
                        {error}
                    </div>
                )}

                <button
                    onClick={handleImport}
                    disabled={loading || !datasetSlug || !fileName}
                    className={`${animationClasses.button} w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                        text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                    {loading ? 'Importing...' : 'Import Dataset'}
                </button>

                {importInfo && (
                    <div className={`${animationClasses.card} mt-6 p-6 bg-purple-50 rounded-xl border-2 border-purple-200`}>
                        <h4 className="text-lg font-semibold mb-3 text-gray-800">Import Summary</h4>
                        <div className="space-y-2 text-gray-700">
                            <p className="flex justify-between">
                                <span className="font-medium">Rows:</span>
                                <span>{importInfo.info.shape[0]}</span>
                            </p>
                            <p className="flex justify-between">
                                <span className="font-medium">Columns:</span>
                                <span>{importInfo.info.shape[1]}</span>
                            </p>
                            {importInfo.suggested_target && (
                                <p className="flex justify-between">
                                    <span className="font-medium">Suggested Target:</span>
                                    <span className="text-purple-600">{importInfo.suggested_target}</span>
                                </p>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default KaggleImport;