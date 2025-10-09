import React, { useCallback } from 'react';
import axios from 'axios';

const DatasetUpload = ({ onFileUpload, currentFile }) => {
  const handleUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('http://localhost:8000/upload', formData);
      onFileUpload(file);
    } catch (error) {
      console.error('Upload error:', error);
    }
  }, [onFileUpload]);

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Dataset Upload</h2>
      
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={handleUpload}
          className="hidden"
          id="fileInput"
        />
        <label
          htmlFor="fileInput"
          className="cursor-pointer text-blue-600 hover:text-blue-800"
        >
          Click to upload dataset
        </label>
        
        {currentFile && (
          <div className="mt-4 text-sm text-gray-600">
            Selected: {currentFile.name}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetUpload;
