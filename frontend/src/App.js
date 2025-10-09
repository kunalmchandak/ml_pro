import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Visualization from './components/Visualization';
import Landing from './components/Landing';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/visualization" element={<Visualization />} />
      </Routes>
    </Router>
  );
}

export default App;
