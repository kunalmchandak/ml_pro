import React from 'react';
import { useNavigate } from 'react-router-dom';

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-blue-900 mb-6">
            Welcome to Apna ML
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            No-code machine learning platform for everyone. Upload your data,
            choose an algorithm, and start training models in minutes.
          </p>
          <button
            onClick={() => navigate('/dashboard')}
            className="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg
                     hover:bg-blue-700 transition duration-200"
          >
            Get Started
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16">
          <FeatureCard
            title="Easy Upload"
            description="Upload your CSV or Excel files with just a few clicks"
          />
          <FeatureCard
            title="Multiple Algorithms"
            description="Choose from various ML algorithms across different categories"
          />
          <FeatureCard
            title="Instant Visualization"
            description="View model performance metrics and visualizations in real-time"
          />
        </div>
      </div>
    </div>
  );
};

const FeatureCard = ({ title, description }) => (
  <div className="bg-white p-6 rounded-lg shadow-md">
    <h3 className="text-xl font-semibold text-blue-800 mb-3">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

export default Landing;
