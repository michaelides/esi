import React, { useContext, useState } from 'react';
import { Context } from '../../context/Context';
import './TemperatureSlider.css';

const TemperatureSlider = () => {
  const { temperature, setTemperature, selectedModel, getMaxTemperatureForModel } = useContext(Context);
  const [showTooltip, setShowTooltip] = useState(false);

  const handleTemperatureChange = (event) => {
    const newTemperature = parseFloat(event.target.value);
    setTemperature(newTemperature);
  };

  const maxTemp = getMaxTemperatureForModel(selectedModel);

  return (
    <div className="temperature-slider">
      <label 
        htmlFor="temperature-slider"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className="slider-label"
      >
        Creativity
        {showTooltip && (
          <div className="tooltip">
            Controls the randomness of responses. Higher values make responses more creative and unpredictable, while lower values make them more consistent and deterministic.
          </div>
        )}
      </label>
      <input
        id="temperature-slider"
        type="range"
        min="0.0"
        max={maxTemp}
        step="0.1"
        value={temperature}
        onChange={handleTemperatureChange}
        className="slider"
      />
      <div className="slider-value">{temperature.toFixed(1)}</div>
    </div>
  );
};

export default TemperatureSlider;