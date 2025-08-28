import React, { useState, useContext, useRef, useEffect } from 'react';
import Sidebar from './components/Sidebar/Sidebar'
import Main from './components/Main/Main'
import { Context } from './context/Context';
import './App.css'

const App = () => {
  const { closeSettings } = useContext(Context);
  const settingsRef = useRef(null);

  return (
    <div className="app">
      <Sidebar/>
      <div className="main-wrap">
        <Main />
      </div>
    </div>
  )
}

export default App