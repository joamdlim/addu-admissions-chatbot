import React from "react";
import AdminPage from "../pages/AdminPage";
import adduLogo from "../assets/addu.png";

function App() {
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-[#063970] text-white p-4 flex justify-between items-center shadow-md">
        <div className="flex items-center">
          <img
            src={adduLogo}
            alt="Ateneo de Davao University Logo"
            className="w-16 h-16 rounded-full mr-4"
          />
          <div>
            <h1 className="text-xl font-bold">ATENEO DE DAVAO UNIVERSITY</h1>
            <p className="text-sm">ADMISSIONS AI ASSISTANT - ADMIN PANEL</p>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto">
        <AdminPage />
      </div>
    </div>
  );
}

export default App;
