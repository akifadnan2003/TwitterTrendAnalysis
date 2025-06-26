// frontend/src/App.js
import React, { useState } from 'react';
import './App.css';

function App() {
  const [comment, setComment] = useState('');
  const [reach, setReach] = useState(''); // State for reach input
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setResults(null);
    setError(null);

    try {
      // Send both text and reach to the API
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: comment,
          reach: parseInt(reach) || 0 // Convert to number, default to 0
        }),
      });
      if (!response.ok) throw new Error('API call failed.');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Advanced Tweet Analyzer</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Enter your tweet text here..."
        />
        <input
          type="number"
          className="reach-input" // Make sure you have styling for this class in App.css
          value={reach}
          onChange={(e) => setReach(e.target.value)}
          placeholder="Enter your account's reach (e.g., 5000)"
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Analyze Tweet'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {results && (
        <div className="results-container">
          <div className="result-card">
            <h2>Predicted Likes</h2>
            <p className="prediction">{results.predicted_likes}</p>
          </div>
          <div className="result-card">
            <h2>Suggested Keywords</h2>
            <ul className="keyword-list">
              {results.suggested_keywords.map((keyword, index) => <li key={index}>{keyword}</li>)}
            </ul>
          </div>
          <div className="result-card">
            <h2>Similar Tweets from the Dataset</h2>
            {results.similar_comments.map((simComment, index) => <p key={index} className="similar-comment">"{simComment}"</p>)}
          </div>
        </div>
      )}
    </div>
  );
}
export default App;
