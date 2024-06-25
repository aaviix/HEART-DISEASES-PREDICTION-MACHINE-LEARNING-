import React, { useState } from 'react';
import './App.css';

function App() {
  const [label, setLabel] = useState('');
  const [number, setNumber] = useState('');
  const [result, setResult] = useState(null);

  const handleClick = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8080/get-options');
      const data = await response.json();
      setLabel(data.message);
    } catch (error) {
      console.error('Error fetching the data', error);
    }
  };

  const handleMultiply = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8080/multiply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ number: parseInt(number) })
      });
      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error('Error fetching the data', error);
    }
  };

  return (
    <div className="App">
      <h1>"Harsh bro are u dumb ? "</h1>
      <div className="container">
        <h1>Harsh:</h1>
        <label className="text-label">{label}</label>
        <button className="button" onClick={handleClick}>Click Me</button>
      </div>
      <div className="container">
        <h1>Multiply a number by 2:</h1>
        <input
          type="number"
          value={number}
          onChange={(e) => setNumber(e.target.value)}
        />
        <button className="button" onClick={handleMultiply}>Multiply</button>
        {result !== null && (
          <div className="result">
            Result: {result}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
