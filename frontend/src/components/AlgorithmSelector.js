import React from 'react';

const AlgorithmSelector = ({ algorithms, selected, onSelect }) => {
  const animationClasses = {
    button: "transition-all duration-300 hover:-translate-y-1 hover:shadow-lg",
    radio: "text-purple-600 focus:ring-purple-400",
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">2. Select Algorithm</h2>
      
      <div className="space-y-6">
        {Object.entries(algorithms).map(([category, algs]) => (
          <div key={category} className="bg-white/80 p-4 rounded-lg border-2 border-purple-200">
            <h3 className="text-lg font-semibold mb-3 text-gray-800 capitalize">
              {category.replace('_', ' ')}
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {algs.map(alg => (
                <div 
                  key={alg} 
                  className={`${animationClasses.button} flex items-center p-3 rounded-lg ${
                    selected === alg 
                    ? 'bg-purple-50 border-2 border-purple-400' 
                    : 'bg-white border-2 border-transparent hover:border-purple-200'
                  }`}
                >
                  <input
                    type="radio"
                    id={alg}
                    name="algorithm"
                    value={alg}
                    checked={selected === alg}
                    onChange={(e) => onSelect(e.target.value)}
                    className={`${animationClasses.radio} w-4 h-4 mr-3`}
                  />
                  <label htmlFor={alg} className="capitalize font-medium text-gray-700 cursor-pointer">
                    {alg.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                  </label>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AlgorithmSelector;
