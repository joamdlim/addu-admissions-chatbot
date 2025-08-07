import React from "react";

const Sidebar = ({ promptHistory }) => (
  <div className="flex flex-col items-center bg-[#063970] text-white w-64 h-full py-6">
    {/* Logo placeholder */}
    <div className="mb-4">
      <div className="w-24 h-24 bg-white rounded-full flex items-center justify-center mb-2 overflow-hidden">
        <img
          src="/rc/assets/addu blue.png"
          alt="ADDU Logo"
          className="w-24 h-24"
        />
      </div>
      <div className="text-center font-semibold text-white">
        ATENEO DE DAVAO UNIVERSITY<br />
        ADMISSIONS AI ASSISTANT
      </div>
    </div>
    {/* Prompt History */}
    <div className="mt-8 w-full px-4">
      <div className="font-bold mb-2">Prompt History</div>
      <ul className="space-y-1">
        {promptHistory.map((prompt, idx) => (
          <li key={idx} className="truncate">{prompt}</li>
        ))}
      </ul>
    </div>
  </div>
);

export { Sidebar };