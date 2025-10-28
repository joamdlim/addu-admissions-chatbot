import React, { useState } from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import { ChatPage } from "./pages/ChatPage";
import { GuidedChatPage } from "./pages/GuidedChatPage";
import adduLogo from "./assets/addu blue.png";
import adduKnight from "./assets/addu-knight.png";

function App() {
  const navigate = useNavigate();
  const [isGuidedMode, setIsGuidedMode] = useState(true); // Default to guided mode

  return (
    <div className="flex h-screen">
      {/* Sidebar with Logo - only visible on extra large screens (1280px+) */}
      <div className="hidden xl:block xl:w-80 2xl:w-96 bg-white border-r border-gray-200">
        <div className="flex flex-col items-center py-8 px-6 h-full relative overflow-hidden">
          {/* Content */}
          <div className="relative z-10 flex flex-col items-center space-y-4">
            <img
              src={adduLogo}
              alt="Ateneo de Davao University Logo"
              className="w-20 h-20 rounded-full"
            />
            <div className="text-center space-y-1">
              <h1 className="text-lg font-light text-gray-700 tracking-wide">
                ATENEO DE DAVAO <br /> UNIVERSITY
              </h1>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                ADMISSIONS AI ASSISTANT
              </p>
            </div>
          </div>
          {/* Knight Background Image */}
          <div className="absolute inset-0 pointer-events-none">
            <img
              src={adduKnight}
              alt="ADDU Knight"
              className="w-full h-full object-cover opacity-10"
              style={{ objectPosition: "78% center" }}
            />
          </div>
        </div>
      </div>

      {/* Main Content Area - takes full width on small screens, remaining width on large screens */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile/Tablet Header - visible on small, medium, and large screens */}
        <div className="xl:hidden w-full bg-white border-b border-gray-200 px-3 sm:px-4 py-3 sm:py-4 flex-shrink-0">
          <div className="flex items-center max-w-7xl mx-auto">
            {/* Logo on the left */}
            <img
              src={adduLogo}
              alt="Ateneo de Davao University Logo"
              className="w-12 h-12 sm:w-14 sm:h-14 rounded-full flex-shrink-0"
            />
            {/* Text next to logo */}
            <div className="ml-3 sm:ml-4 flex flex-col justify-center">
              <h1 className="text-sm sm:text-base font-medium text-gray-800 leading-tight">
                ATENEO DE DAVAO UNIVERSITY
              </h1>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mt-0.5">
                ADMISSIONS AI ASSISTANT
              </p>
            </div>
          </div>
        </div>

        {/* Chat Area - takes remaining space */}
        <div className="flex-1 flex flex-col min-h-0">
          {isGuidedMode ? <GuidedChatPage /> : <ChatPage />}
        </div>
      </div>
    </div>
  );
}

export default App;
