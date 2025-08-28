import React, { useContext } from 'react';
import { Context } from '../../context/Context';
import './VerbositySlider.css';

const VerbositySlider = () => {
  const { verbosity, setVerbosity } = useContext(Context);

  const handleVerbosityChange = (event) => {
    setVerbosity(parseInt(event.target.value));
  };

  const getVerbosityLabel = (level) => {
    switch(level) {
      case 1: return 'Laconic';
      case 2: return 'Concise';
      case 3: return 'Moderate';
      case 4: return 'Verbose';
      case 5: return 'Very Verbose';
      default: return 'Moderate';
    }
  };

  return (
    <div className="verbosity-slider">
      <label htmlFor="verbosity-slider">Verbosity</label>
      <input
        id="verbosity-slider"
        type="range"
        min="1"
        max="5"
        step="1"
        value={verbosity}
        onChange={handleVerbosityChange}
        className="slider"
      />
      <div className="slider-value">{getVerbosityLabel(verbosity)}</div>
    </div>
  );
};

export default VerbositySlider;