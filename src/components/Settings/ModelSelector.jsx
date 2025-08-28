import React, { useContext } from 'react';
import { Context } from '../../context/Context';
import './ModelSelector.css';

const ModelSelector = () => {
  const { selectedModel, setSelectedModel, modelCategories } = useContext(Context);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const getSelectedModelInfo = () => {
    if (!modelCategories) {
      return { name: selectedModel || 'Unknown', description: '' };
    }
    
    for (const category of Object.values(modelCategories)) {
      const model = category.models.find(m => m.id === selectedModel);
      if (model) return model;
    }
    return { name: selectedModel || 'Unknown', description: '' };
  };

  const selectedModelInfo = getSelectedModelInfo();
  
  // Don't render if required props are missing
  if (!modelCategories || !selectedModel || !setSelectedModel) {
    return <div>Model selector loading...</div>;
  }

  return (
    <div className="model-selector">
      <div className="model-selector-header">
        <label htmlFor="model-select">Language Model</label>
        <div className="selected-model-info">
          <span className="model-name">{selectedModelInfo.name}</span>
          {selectedModelInfo.description && (
            <span className="model-description">{selectedModelInfo.description}</span>
          )}
        </div>
      </div>
      
      <select 
        id="model-select"
        value={selectedModel} 
        onChange={handleModelChange}
        className="model-dropdown"
      >
        {Object.entries(modelCategories).map(([categoryKey, category]) => (
          <optgroup key={categoryKey} label={category.label}>
            {category.models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
};

export default ModelSelector;