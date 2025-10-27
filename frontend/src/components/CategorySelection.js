import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BeakerIcon,
  ChartBarIcon,
  ChartPieIcon,
  CircleStackIcon,
  LightBulbIcon,
  PresentationChartLineIcon,
  Square3Stack3DIcon,
  TableCellsIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';

const CategorySelection = () => {
  const navigate = useNavigate();
  const [hoveredCategory, setHoveredCategory] = useState(null);

  const handleSelect = (algorithm, category, subCategory) => {
    // Route to specialized dashboard based on category
    const dashboardRoutes = {
      supervised: {
        classification: '/dashboard/classification',
        regression: '/dashboard/regression'
      },
      unsupervised: {
        clustering: '/dashboard/clustering',
        association: '/dashboard/association'
      }
    };

    const route = dashboardRoutes[category]?.[subCategory] || '/dashboard';
    navigate(route, { 
      state: { 
        algorithm,
        category,
        subCategory 
      } 
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white/40 via-purple-50/30 to-indigo-50/20 relative overflow-hidden">
      {/* Enhanced Animated Background Blobs with improved effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute w-[600px] h-[600px] -top-64 -left-48 bg-purple-300/40 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob motion-safe:animate-pulse"></div>
        <div className="absolute w-[700px] h-[700px] -bottom-64 -right-48 bg-indigo-300/40 rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob-reverse motion-safe:animate-pulse"></div>
        <div className="absolute w-[500px] h-[500px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-pink-300/30 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-float motion-safe:animate-pulse"></div>
        <div className="absolute w-[400px] h-[400px] top-1/4 right-1/4 bg-blue-300/30 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob-slow"></div>
      </div>

      <div className="container mx-auto px-6 py-20">
        <div className="text-center max-w-3xl mx-auto mb-16 animate-fade-in-down relative">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-400/20 via-transparent to-indigo-400/20 blur-3xl -z-10"></div>
          <h2 className="text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-fuchsia-500 to-indigo-600 mb-6 animate-gradient-x">
            Choose a Category
          </h2>
          <p className="text-xl text-gray-600/90 backdrop-blur-sm py-2">
            Select your machine learning approach based on your data and objectives
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 max-w-7xl mx-auto">
          {/* Supervised Card */}
          <div className="group/card animate-scale-up"
               style={{ animationDelay: '0.1s' }}
               onMouseEnter={() => setHoveredCategory('supervised')}
               onMouseLeave={() => setHoveredCategory(null)}>
            <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl border-2 border-purple-200/50 
                          group-hover/card:border-purple-400 shadow-lg group-hover/card:shadow-2xl 
                          transition-all duration-500 h-full relative overflow-hidden
                          group-hover/card:-translate-y-2 hover:bg-white/90
                          before:absolute before:inset-0 before:bg-gradient-to-r before:from-purple-500/10 before:via-transparent before:to-indigo-500/10 before:opacity-0 
                          before:transition-opacity group-hover/card:before:opacity-100">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-purple-100 rounded-xl group-hover/card:scale-110 transition-transform">
                  <BeakerIcon className="w-8 h-8 text-purple-600" />
                </div>
                <h3 className="text-3xl font-bold text-gray-800">Supervised Learning</h3>
              </div>
              <p className="text-gray-600 mb-8 text-lg">
                Train models using labeled data for classification or regression tasks.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <CategoryButton
                  icon={<ChartPieIcon className="w-6 h-6" />}
                  title="Classification"
                  subtitle="Binary & multiclass"
                  description="Perfect for categorizing data into predefined classes"
                  onClick={() => handleSelect('logistic_regression', 'supervised', 'classification')}
                  isHovered={hoveredCategory === 'supervised'}
                  delay={0}
                />
                <CategoryButton
                  icon={<PresentationChartLineIcon className="w-6 h-6" />}
                  title="Regression"
                  subtitle="Continuous targets"
                  description="Predict continuous numerical values"
                  onClick={() => handleSelect('linear_regression', 'supervised', 'regression')}
                  isHovered={hoveredCategory === 'supervised'}
                  delay={200}
                />
              </div>
            </div>
          </div>

          {/* Unsupervised Card */}
          <div className="group/card animate-scale-up"
               style={{ animationDelay: '0.3s' }}
               onMouseEnter={() => setHoveredCategory('unsupervised')}
               onMouseLeave={() => setHoveredCategory(null)}>
            <div className="bg-white/90 backdrop-blur-lg p-8 rounded-2xl border-2 border-purple-200 
                          group-hover/card:border-purple-400 shadow-lg group-hover/card:shadow-2xl 
                          transition-all duration-500 h-full relative overflow-hidden
                          group-hover/card:-translate-y-2">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-purple-100 rounded-xl group-hover/card:scale-110 transition-transform">
                  <LightBulbIcon className="w-8 h-8 text-purple-600" />
                </div>
                <h3 className="text-3xl font-bold text-gray-800">Unsupervised Learning</h3>
              </div>
              <p className="text-gray-600 mb-8 text-lg">
                Discover structure in unlabeled data using clustering or association rule mining.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <CategoryButton
                  icon={<Square3Stack3DIcon className="w-6 h-6" />}
                  title="Clustering"
                  subtitle="Group similar items"
                  description="Find natural groupings in your data"
                  onClick={() => handleSelect('kmeans', 'unsupervised', 'clustering')}
                  isHovered={hoveredCategory === 'unsupervised'}
                  delay={0}
                />
                <CategoryButton
                  icon={<TableCellsIcon className="w-6 h-6" />}
                  title="Association"
                  subtitle="Market-basket rules"
                  description="Discover patterns in transaction data"
                  onClick={() => handleSelect('apriori', 'unsupervised', 'association')}
                  isHovered={hoveredCategory === 'unsupervised'}
                  delay={200}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const CategoryButton = ({ icon, title, subtitle, description, onClick, isHovered, delay }) => (
  <button
    onClick={onClick}
    className="flex flex-col items-start p-6 bg-white/90 rounded-xl border border-purple-200 
               hover:border-purple-500 transition-all duration-300 shadow hover:shadow-xl 
               group/btn animate-slide-up hover:-translate-y-1 hover:scale-[1.02]
               relative overflow-hidden"
    style={{ animationDelay: `${delay}ms` }}
  >
    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-indigo-500/10 
                    opacity-0 group-hover/btn:opacity-100 transition-opacity duration-300" />
    <div className="relative">
      <div className="p-2 bg-purple-50 rounded-lg mb-3 group-hover/btn:bg-purple-100 
                    transition-all duration-300 group-hover/btn:scale-110">
        <div className="text-purple-600 group-hover/btn:text-purple-700">
          {icon}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="text-xl font-semibold text-gray-800 group-hover/btn:text-purple-700 transition-colors">
          {title}
        </div>
        <ArrowRightIcon className="w-4 h-4 text-purple-500 opacity-0 -translate-x-2 
                                 group-hover/btn:opacity-100 group-hover/btn:translate-x-0 
                                 transition-all duration-300" />
      </div>
      <div className="text-sm text-purple-600 font-medium mb-2">{subtitle}</div>
      <div className="text-sm text-gray-500 group-hover/btn:text-gray-600">{description}</div>
    </div>
  </button>
);

export default CategorySelection;
