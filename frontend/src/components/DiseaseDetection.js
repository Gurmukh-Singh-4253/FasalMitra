import React, { useState, useEffect } from 'react';
import './DiseaseDetection.css';
import axios from 'axios';

function DiseaseDetection() {
  const [showDiseaseDetection, setShowDiseaseDetection] = useState(true);
  const [showFertilizerRecommendation, setShowFertilizerRecommendation] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [diseaseResult, setDiseaseResult] = useState(null);

  useEffect(() => {
    createBalls(15);
  }, []);

  const createBalls = (numBalls) => {
    try {
      const container = document.getElementById('ball-container');
      if (!container) return;
      for (let i = 0; i < numBalls; i++) {
        let ball = document.createElement('div');
        ball.classList.add('ball');
        ball.style.top = `${Math.random() * window.innerHeight}px`;
        ball.style.left = `${Math.random() * window.innerWidth}px`;
        ball.style.animationDuration = `${10 + Math.random() * 5}s, ${25 + Math.random() * 4}s`;
        ball.style.animationDelay = `${Math.random() * 5}s, ${Math.random() * 5}s`;
        container.appendChild(ball);
      }
    } catch (error) {
      console.error('Error in createBalls:', error);
    }
  };

  const handleDiseaseDetectionClick = () => {
    setShowDiseaseDetection(true);
    setShowFertilizerRecommendation(false);
    setShowFeedback(false);
  };

  const handleFertilizerRecommendationClick = () => {
    setShowDiseaseDetection(false);
    setShowFertilizerRecommendation(true);
    setShowFeedback(false);
  };

  const handleFeedbackClick = () => {
    setShowDiseaseDetection(false);
    setShowFertilizerRecommendation(false);
    setShowFeedback(true);
  };

  const handleImageChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedImage(event.target.files[0]);
    }
  };

  const handleDiseaseSubmit = async (event) => {
    event.preventDefault();
    if (!selectedImage) return;
    const formData = new FormData();
    formData.append('image', selectedImage);
    try {
      const response = await axios.post('https://api.example.com/disease', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setDiseaseResult(response.data);
    } catch (error) {
      console.error('API Error:', error);
      alert('Failed to detect disease. Please try again.');
    }
  };

  return (
    <div className="disease-detection-container">
      <div id="ball-container"></div>
      <div className="main-content-wrapper">
        <div className="dashboard">
          <div className="p-6 dashboard-content">
            <h1 className="text-xl font-bold mb-6 mt-4">Dashboard</h1>
            <ul className="dashboard-buttons">
              <li className="mb-4">
                <button className="dashboard-button" onClick={handleDiseaseDetectionClick}>
                  Disease Detection
                </button>
              </li>
              <li>
                <button className="dashboard-button" onClick={handleFertilizerRecommendationClick}>
                  Fertilizer Recommendation System
                </button>
              </li>
              <li className="mt-4">
                <button className="dashboard-button" onClick={handleFeedbackClick}>
                  Chatbot
                </button>
              </li>
            </ul>
          </div>
        </div>
        <div className="main-content">
          {showDiseaseDetection && (
            <div className="flex justify-center items-center h-full">
              <div className="glass-effect p-8 rounded-lg shadow-lg relative z-10 scrollable" id="disease-detection-content">
                <h2 className="text-2xl font-bold mb-4">Disease Detection System</h2>
                <form onSubmit={handleDiseaseSubmit}>
                  <div className="mb-4">
                    <label htmlFor="image-upload">Upload Crop Image</label>
                    <input className="hidden" id="image-upload" type="file" onChange={handleImageChange} />
                    {selectedImage && <p className="file-name">Selected: {selectedImage.name}</p>}
                  </div>
                  <button className="w-full btn-emerald text-white p-2 rounded-lg" type="submit">Submit</button>
                </form>
                {diseaseResult && (
                  <div className="mt-6">
                    <h3 className="text-xl font-bold mb-2">{diseaseResult.diseaseName}</h3>
                    <p>
                      <strong>Cause:</strong> {diseaseResult.cause}
                    </p>
                    <p>
                      <strong>Symptoms:</strong> {diseaseResult.symptoms}
                    </p>
                    <p>
                      <strong>Cure:</strong> {diseaseResult.cure}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
          {showFertilizerRecommendation && (
            <div className="glass-effect p-8 rounded-lg shadow-lg w-1/2 relative z-10 scrollable" id="fertiliser-recommendation-content">
              <h2 className="text-2xl font-bold mb-4">Fertilizer Recommendation System</h2>
              <p>Fertilizer recommendation form removed due to unused state variables. Please add the form and handlers if you wish to use this section.</p>
            </div>
          )}
          {showFeedback && (
            <div className="feedback-section glass-effect p-8 rounded-lg shadow-lg relative z-10">
              <div className="chatbot-image-container">
                <img alt="Chatbot interface with a friendly robot icon" className="chatbot-image" src="https://i.ibb.co/HfYqc5kw/hhhhhhh.jpg" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DiseaseDetection;