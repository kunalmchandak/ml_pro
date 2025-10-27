import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Visualization from './components/Visualization';
import LandingPage from './components/LandingPage';
import CategorySelection from './components/CategorySelection';
import ClassificationDashboard from './components/dashboards/ClassificationDashboard';
import RegressionDashboard from './components/dashboards/RegressionDashboard';
import ClusteringDashboard from './components/dashboards/ClusteringDashboard';
import AssociationDashboard from './components/dashboards/AssociationDashboard';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dashboard/classification" element={<ClassificationDashboard />} />
          <Route path="/dashboard/regression" element={<RegressionDashboard />} />
          <Route path="/dashboard/clustering" element={<ClusteringDashboard />} />
          <Route path="/dashboard/association" element={<AssociationDashboard />} />
          <Route path="/categories" element={<CategorySelection />} />
          <Route path="/visualization" element={<Visualization />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
