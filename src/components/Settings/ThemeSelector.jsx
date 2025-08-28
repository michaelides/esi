import React, { useContext } from 'react';
import { Context } from '../../context/Context';
import './ThemeSelector.css';

const ThemeSelector = () => {
  const { theme, setSelectedTheme } = useContext(Context);
  
  // Debug log to verify component is being rendered
  console.log('ThemeSelector rendered, current theme:', theme);

  const lightTheme = {
    id: 'light',
    name: 'Light',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="2"/>
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" stroke="currentColor" strokeWidth="2"/>
      </svg>
    )
  };

  const darkTheme = {
    id: 'dark',
    name: 'Dark',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" stroke="currentColor" strokeWidth="2"/>
      </svg>
    )
  };

  // Determine which theme option to show based on current theme
  const themeOption = theme === 'light' ? darkTheme : lightTheme;

  return (
    <div className="theme-selector">
      <div className="theme-toggle-container">
        <span className="theme-label">Theme:</span>
        <button
          className="theme-toggle-button"
          onClick={() => setSelectedTheme(themeOption.id)}
        >
          <span className="theme-icon">{themeOption.icon}</span>
          <span className="theme-name">{themeOption.name}</span>
        </button>
      </div>
    </div>
  );
};

export default ThemeSelector;