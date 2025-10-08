import React, { useState } from "react";
import AdminPage from "./pages/AdminPage";
import TextEditorPage from "./pages/TextEditorPage";
import TopicManagementPage from "./pages/TopicManagementPage";
import adduLogo from "./assets/addu.png";

function App() {
  const [currentPage, setCurrentPage] = useState("admin");
  const [fileData, setFileData] = useState(null);

  const handleFileForReview = (data) => {
    setFileData(data);
    setCurrentPage("editor");
  };

  const handleConfirmUpload = () => {
    setCurrentPage("admin");
    setFileData(null);
  };

  const handleCancel = () => {
    setCurrentPage("admin");
    setFileData(null);
  };

  return (
    <div className="flex flex-col h-screen max-h-screen overflow-hidden bg-gray-100">
      {/* Header - Fixed Height */}
      <header className="bg-[#063970] text-white p-3 flex justify-between items-center shadow-md flex-shrink-0">
        <div className="flex items-center">
          <img
            src={adduLogo}
            alt="Ateneo de Davao University Logo"
            className="w-12 h-12 rounded-full mr-3"
          />
          <div>
            <h1 className="text-lg font-bold">ATENEO DE DAVAO UNIVERSITY</h1>
            <p className="text-xs">
              ADMISSIONS AI ASSISTANT - ADMIN PANEL
              {currentPage === "editor" && " - DOCUMENT EDITOR"}
              {currentPage === "topics" && " - TOPIC MANAGEMENT"}
            </p>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex items-center space-x-4">
          {currentPage !== "admin" && (
            <button
              onClick={() => setCurrentPage("admin")}
              className="text-white hover:text-gray-300 transition text-sm"
            >
              📄 Documents
            </button>
          )}
          {currentPage !== "topics" && (
            <button
              onClick={() => setCurrentPage("topics")}
              className="text-white hover:text-gray-300 transition text-sm"
            >
              🏷️ Topics
            </button>
          )}
          {(currentPage === "editor" || currentPage === "topics") && (
            <button
              onClick={handleCancel}
              className="text-white hover:text-gray-300 transition"
            >
              ← Back to Admin
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area - Remaining Height */}
      <div className="flex-1 overflow-hidden">
        {currentPage === "admin" && (
          <AdminPage onFileForReview={handleFileForReview} />
        )}
        {currentPage === "editor" && (
          <TextEditorPage
            fileData={fileData}
            onConfirmUpload={handleConfirmUpload}
            onCancel={handleCancel}
          />
        )}
        {currentPage === "topics" && <TopicManagementPage />}
      </div>
    </div>
  );
}

export default App;
