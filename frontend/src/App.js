// src/App.js
import React from 'react';
import DiseaseDetection from './components/DiseaseDetection';
import './App.css';
import TranslatePage from './components/TranslatePage';

function App() {
  return (
    <div className="App">
      <DiseaseDetection />
      <TranslatePage />
    </div>
  );
}

export default App;
