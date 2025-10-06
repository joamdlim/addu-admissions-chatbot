import React, { useState } from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import { ChatPage } from "./pages/ChatPage";
import { GuidedChatPage } from "./pages/GuidedChatPage";
import adduLogo from "./assets/addu.png"; // Make sure this path is correct!

function App() {
  const navigate = useNavigate();
  const [isGuidedMode, setIsGuidedMode] = useState(true); // Default to guided mode

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="bg-[#063970] text-white p-4 flex justify-between items-center shadow-md">
        <div className="flex items-center">
          {" "}
          {/* Added a flex container for the logo and text */}
          <img
            src={adduLogo} // Use the imported logo
            alt="Ateneo de Davao University Logo"
            className="w-16 h-16 rounded-full mr-4" // Tailwind classes for styling
          />
          <div>
            <h1 className="text-xl font-bold">ATENEO DE DAVAO UNIVERSITY</h1>
            <p className="text-sm">ADMISSIONS AI ASSISTANT</p>
          </div>
        </div>

        {/* Mode Toggle */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm">Chat Mode:</span>
            <button
              onClick={() => setIsGuidedMode(!isGuidedMode)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                isGuidedMode
                  ? "bg-white text-[#063970]"
                  : "bg-transparent border border-white text-white hover:bg-white hover:text-[#063970]"
              }`}
            >
              {isGuidedMode ? "Guided" : "Free Chat"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto">
        {isGuidedMode ? <GuidedChatPage /> : <ChatPage />}
      </div>
    </div>
  );
}

export default App;
