import React, { useRef } from "react";

const AdminPage = () => {
  const fileInputRef = useRef(null);

  const handleAddFileClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="p-6">
      {/* Admin Page Header - This will be covered by the main App header or you can customize App header based on route*/}
      <div className="mb-6">
        <h2 className="text-2xl text-white text-[#063970]">
          ATENEO DE DAVAO UNIVERSITY
        </h2>
        <p className="text-lg text-white">ADMISSIONS AI ASSISTANT - ADMIN</p>
      </div>

      {/* Add file button */}
      <button
        onClick={handleAddFileClick}
        className="bg-[#063970] text-white px-4 py-2 rounded mb-6 flex items-center space-x-2 hover:bg-blue-800 transition"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M8 4a3 3 0 00-3 3v4a5 5 0 0010 0V7a1 1 0 112 0v4a7 7 0 11-14 0V7a5 5 0 0110 0v4a3 3 0 11-6 0V7a1 1 0 012 0v4a1 1 0 102 0V7a3 3 0 00-3-3z"
            clipRule="evenodd"
          />
        </svg>
        <span>Add file</span>
      </button>
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        // No onChange handler yet, as you requested
      />

      {/* File List Table (Structure Only) */}
      <div className="bg-white shadow rounded overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-[#063970] text-white">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                File name
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-gray-100 divide-y divide-gray-200">
            {/* Example rows - replace with actual data */}
            <tr>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                Document1.pdf
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <button className="text-gray-600 hover:text-gray-900">...</button>
              </td>
            </tr>
            <tr>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                Document2.pdf
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <button className="text-gray-600 hover:text-gray-900">...</button>
              </td>
            </tr>
            <tr>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                Document3.txt
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <button className="text-gray-600 hover:text-gray-900">...</button>
              </td>
            </tr>
            {/* ... more rows ... */}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AdminPage;