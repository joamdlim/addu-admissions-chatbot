import React, { useState, useRef, useEffect } from "react";

const AdminPage = ({ onFileForReview }) => {
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [syncingFile, setSyncingFile] = useState(null);

  // Fetch files from backend
  const fetchFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/files/"
      );
      const data = await response.json();
      setFiles(data.files || []);
    } catch (error) {
      console.error("Error fetching files:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleAddFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setExtracting(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/extract-text/",
        {
          method: "POST",
          body: formData,
        }
      );

      const result = await response.json();

      if (response.ok) {
        // Pass the extracted data to the editor page
        if (onFileForReview) {
          onFileForReview(result);
        }
        fileInputRef.current.value = ""; // Clear input
      } else {
        alert(`Text extraction failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Extraction error:", error);
      alert("Text extraction failed. Please try again.");
    } finally {
      setExtracting(false);
    }
  };

  const handleDeleteFile = async (fileName) => {
    if (!confirm(`Are you sure you want to delete "${fileName}"?`)) return;

    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/delete/",
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ file_name: fileName }),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        fetchFiles(); // Refresh file list
      } else {
        alert(`Delete failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert("Delete failed. Please try again.");
    }
  };

  // Keep the sync function for manual syncing if needed
  const handleSyncToChroma = async (fileName) => {
    if (!fileName.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF files can be synced to ChromaDB");
      return;
    }

    setSyncingFile(fileName);
    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/sync-to-chroma/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ file_name: fileName }),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(
          `File synced successfully! ${result.documents_stored} documents stored in ChromaDB.`
        );
      } else {
        alert(`Sync failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Sync error:", error);
      alert("Sync failed. Please try again.");
    } finally {
      setSyncingFile(null);
    }
  };

  const handleDownloadFile = async (fileName) => {
    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/download/${encodeURIComponent(
          fileName
        )}/`,
        {
          method: "GET",
        }
      );

      if (response.ok) {
        // Create a blob from the response
        const blob = await response.blob();

        // Create a temporary URL for the blob
        const url = window.URL.createObjectURL(blob);

        // Create a temporary download link
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;

        // Trigger the download
        document.body.appendChild(link);
        link.click();

        // Clean up
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        console.log(`✅ Downloaded: ${fileName}`);
      } else {
        const error = await response.json();
        alert(`Download failed: ${error.error}`);
      }
    } catch (error) {
      console.error("Download error:", error);
      alert("Download failed. Please try again.");
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (dateString) => {
    if (!dateString) return "Unknown";
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div className="p-6">
      {/* Add file button */}
      <button
        onClick={handleAddFileClick}
        disabled={extracting}
        className={`px-4 py-2 rounded mb-6 flex items-center space-x-2 transition ${
          extracting
            ? "bg-gray-500 cursor-not-allowed"
            : "bg-[#063970] hover:bg-blue-800"
        } text-white`}
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
        <span>{extracting ? "Extracting text..." : "Add file"}</span>
      </button>

      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileUpload}
        accept=".pdf,.txt,.doc,.docx"
      />

      {/* Instruction text */}
      {extracting && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
          <p className="text-blue-700 text-sm">
            📄 Extracting text from your file... You'll be redirected to the
            editor to review the content before uploading.
          </p>
        </div>
      )}

      {/* File List Table */}
      <div className="bg-white shadow rounded overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-[#063970] text-white">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                File name
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Size
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Type
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Uploaded
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-gray-100 divide-y divide-gray-200">
            {loading ? (
              <tr>
                <td colSpan="5" className="px-6 py-4 text-center text-gray-500">
                  Loading files...
                </td>
              </tr>
            ) : files.length === 0 ? (
              <tr>
                <td colSpan="5" className="px-6 py-4 text-center text-gray-500">
                  No files uploaded yet
                </td>
              </tr>
            ) : (
              files.map((file, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {file.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatFileSize(file.size)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {file.content_type}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatDate(file.created_at)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium space-x-2">
                    {file.name.toLowerCase().endsWith(".pdf") && (
                      <button
                        onClick={() => handleSyncToChroma(file.name)}
                        disabled={syncingFile === file.name}
                        className="text-blue-600 hover:text-blue-900 transition"
                        title="Sync to ChromaDB"
                      >
                        {syncingFile === file.name ? "⏳" : "🔄"}
                      </button>
                    )}
                    <button
                      onClick={() => handleDownloadFile(file.name)}
                      className="text-green-600 hover:text-green-900 transition"
                      title="Download file"
                    >
                      📥
                    </button>
                    <button
                      onClick={() => handleDeleteFile(file.name)}
                      className="text-red-600 hover:text-red-900 transition"
                      title="Delete file"
                    >
                      🗑️
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Refresh button */}
      <button
        onClick={fetchFiles}
        disabled={loading}
        className="mt-4 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded transition"
      >
        {loading ? "Loading..." : "Refresh"}
      </button>

      {/* Info section */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          How it works:
        </h3>
        <ol className="list-decimal list-inside text-sm text-gray-600 space-y-1">
          <li>Click "Add file" to select a document (PDF, TXT, DOC, DOCX)</li>
          <li>The system will extract and clean the text content</li>
          <li>Review and edit the extracted text in the editor</li>
          <li>
            Confirm to upload the processed document to the knowledge base
          </li>
        </ol>
      </div>
    </div>
  );
};

export default AdminPage;
