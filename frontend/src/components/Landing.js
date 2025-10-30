import React from 'react';
import { Link } from 'react-router-dom';
import { 
  ChartBarIcon, 
  CpuChipIcon, 
  BeakerIcon, 
  PresentationChartLineIcon,
  LightBulbIcon,
  RocketLaunchIcon,
  BoltIcon,
  CloudArrowUpIcon,
  CommandLineIcon,
  CursorArrowRaysIcon
} from '@heroicons/react/24/outline';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-purple-100/40 to-indigo-100/40 relative">
  {/* Animated Background Elements */}
      <div className="inset-0 w-full h-full overflow-hidden -z-20 pointer-events-none">
        <div className="absolute w-96 h-96 -top-48 -left-48 bg-purple-400 rounded-full mix-blend-multiply filter blur-lg opacity-60 animate-blob"></div>
        <div className="absolute w-96 h-96 -top-48 -right-48 bg-indigo-400 rounded-full mix-blend-multiply filter blur-lg opacity-60 animate-blob-x animation-delay-2000"></div>
        <div className="absolute w-96 h-96 -bottom-48 -left-48 bg-pink-400 rounded-full mix-blend-multiply filter blur-lg opacity-60 animate-blob-y animation-delay-4000"></div>
        {/* Center pink blob */}
        <div className="absolute w-80 h-80 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-pink-300 rounded-full mix-blend-multiply filter blur-lg opacity-55 animate-blob-xy animation-delay-2500" />
      </div>

      {/* Hero Section */}
      <div className="container mx-auto px-6 py-16">
        <div className="flex flex-col lg:flex-row items-center">
          <div className="lg:w-1/2 flex flex-col items-start">
            <div className="animate-fade-in-down w-full max-w-xl text-justify ml-4">
              <h1 className="text-6xl font-bold leading-tight mb-5 text-gray-900">
                Machine Learning
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                  Made Simple
                </span>
              </h1>
              <p className="text-xl text-gray-600 mb-8 max-w-md">
                Transform your data into insights with our powerful machine learning platform. <br />
                No coding required - just intuitive analytics at your fingertips.
              </p>
              <div className="flex">
                <Link
                  to="/dashboard"
                  className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                            text-white font-bold py-4 px-8 rounded-lg transition duration-300 ease-in-out 
                            transform hover:-translate-y-1 hover:scale-105 shadow-lg hover:shadow-xl"
                >
                  Get Started
                </Link>
              </div>
            </div>
          </div>
          
          {/* Feature Cards */}
          <div className="lg:w-1/2 grid grid-cols-2 gap-6 mt-12 lg:mt-0">
            <FeatureCard
              title="Upload Dataset"
              description="Import your own data or use our sample datasets"
              icon={<CloudArrowUpIcon className="w-8 h-8" />}
              delay="0"
            />
            <FeatureCard
              title="Choose Algorithm"
              description="Select from various ML algorithms"
              icon={<CpuChipIcon className="w-8 h-8" />}
              delay="200"
            />
            <FeatureCard
              title="Train Model"
              description="Automatic model training and evaluation"
              icon={<BeakerIcon className="w-8 h-8" />}
              delay="400"
            />
            <FeatureCard
              title="Visualize Results"
              description="Interactive charts and metrics"
              icon={<PresentationChartLineIcon className="w-8 h-8" />}
              delay="600"
            />
          </div>
        </div>
      </div>

      {/* Features Section with Gradient Border */}
      <div className="bg-gradient-to-b from-transparent via-white/30 to-white/40 py-24">
        <div className="container mx-auto px-6">
          <h2 className="text-4xl font-bold text-center text-gray-800 mb-16">
            Why Choose Our Platform?
          </h2>
          <div className="grid md:grid-cols-3 gap-12">
            <BenefitCard
              title="Intuitive Interface"
              description="No coding experience required. Our platform guides you through every step."
              icon={<CursorArrowRaysIcon className="w-12 h-12" />}
              delay="0"
            />
            <BenefitCard
              title="Advanced Algorithms"
              description="Access state-of-the-art machine learning algorithms with one click."
              icon={<CommandLineIcon className="w-12 h-12" />}
              delay="200"
            />
            <BenefitCard
              title="Real-time Analytics"
              description="Get instant insights with interactive visualizations and detailed metrics."
              icon={<ChartBarIcon className="w-12 h-12" />}
              delay="400"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const FeatureCard = ({ title, description, icon, delay }) => (
  <div 
    className="group bg-white/90 backdrop-blur-lg p-6 rounded-xl border-4 border-purple-400 
              hover:bg-white transition-all duration-300 transform hover:-translate-y-1 
              hover:shadow-xl hover:shadow-purple-200/50 hover:border-purple-600 animate-fade-in-up"
    style={{ animationDelay: `${delay}ms` }}
  >
    <div className="text-purple-600 group-hover:text-purple-700 transition-colors duration-300 mb-4">
      {icon}
    </div>
    <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
    <p className="text-gray-600 group-hover:text-gray-700 transition-colors duration-300">
      {description}
    </p>
  </div>
);

const BenefitCard = ({ title, description, icon, delay }) => (
  <div 
    className="text-center p-8 rounded-xl bg-white/90 backdrop-blur-lg border-4 border-purple-400 
              hover:bg-white hover:shadow-xl hover:shadow-purple-200/50 hover:border-purple-600 
              transition-all duration-300 animate-fade-in-up"
    style={{ animationDelay: `${delay}ms` }}
  >
    <div className="inline-block p-4 rounded-full bg-purple-100 text-purple-600 mb-4 
                    ring-2 ring-purple-200">
      {icon}
    </div>
    <h3 className="text-xl font-semibold text-gray-800 mb-3">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

export default LandingPage;