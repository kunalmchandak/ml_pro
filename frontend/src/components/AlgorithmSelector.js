import React from 'react';

const AlgorithmSelector = ({ algorithms, selected, onSelect }) => {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Select Algorithm</h2>
      
      {Object.entries(algorithms).map(([category, algs]) => (
        <div key={category} className="mb-4">
          <h3 className="text-lg font-medium mb-2 capitalize">
            {category.replace('_', ' ')}
          </h3>
          
          <div className="space-y-2">
            {algs.map(alg => (
              <div key={alg} className="flex items-center">
                <input
                  type="radio"
                  id={alg}
                  name="algorithm"
                  value={alg}
                  checked={selected === alg}
                  onChange={(e) => onSelect(e.target.value)}
                  className="mr-2"
                />
                <label htmlFor={alg} className="capitalize">
                  {alg.replace('_', ' ')}
                </label>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default AlgorithmSelector;
