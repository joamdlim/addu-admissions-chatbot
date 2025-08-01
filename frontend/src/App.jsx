import React from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import { ChatPage } from "./pages/ChatPage";
import AdminPage from "./pages/AdminPage";

function App() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="bg-[#063970] text-white p-4 flex justify-between items-center shadow-md">
        <div>
          <h1 className="text-xl font-bold">ATENEO DE DAVAO UNIVERSITY</h1>
          <p className="text-sm">ADMISSIONS AI ASSISTANT</p>
        </div>
        {/* Button to Admin Page */}
        <button
          onClick={() => navigate("/admin")}
          className="bg-white text-[#063970] px-4 py-2 rounded shadow hover:bg-gray-100 transition"
        >
          Admin Page
        </button>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;