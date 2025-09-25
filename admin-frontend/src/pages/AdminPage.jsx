import React, { useState, useRef, useEffect } from "react";

const AdminPage = ({ onFileForReview }) => {
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [folders, setFolders] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [syncingFile, setSyncingFile] = useState(null);

  // Folder management state
  const [showFolderForm, setShowFolderForm] = useState(false);
  const [newFolder, setNewFolder] = useState({
    name: "",
    description: "",
    color: "#063970",
  });
  const [editingFolder, setEditingFolder] = useState(null);

  // Document metadata state
  const [showMetadataPanel, setShowMetadataPanel] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);

  // Upload form state
  const [uploadMetadata, setUploadMetadata] = useState({
    folder_id: "",
    document_type: "other",
    target_program: "all",
    keywords: "",
  });

  // Fetch all data
  const fetchAllData = async () => {
    setLoading(true);
    try {
      // Fetch files, folders, and documents in parallel
      const [filesRes, foldersRes, docsRes] = await Promise.all([
        fetch("http://localhost:8000/chatbot/admin/files/"),
        fetch("http://localhost:8000/chatbot/admin/folders/"),
        fetch("http://localhost:8000/chatbot/admin/documents/"),
      ]);

      const [filesData, foldersData, docsData] = await Promise.all([
        filesRes.json(),
        foldersRes.json(),
        docsRes.json(),
      ]);

      setFiles(filesData.files || []);
      setFolders(foldersData.folders || []);
      setDocuments(docsData.documents || []);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  // Folder management functions
  const handleCreateFolder = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/folders/",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(newFolder),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        setNewFolder({ name: "", description: "", color: "#063970" });
        setShowFolderForm(false);
        fetchAllData();
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error creating folder:", error);
      alert("Failed to create folder");
    }
  };

  const handleUpdateDocumentMetadata = async (documentId, updates) => {
    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/documents/${documentId}/metadata/`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updates),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        fetchAllData();
        setShowMetadataPanel(false);
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error updating metadata:", error);
      alert("Failed to update metadata");
    }
  };

  // Enhanced file upload with metadata
  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setExtracting(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    // Add metadata to form data
    if (uploadMetadata.folder_id)
      formData.append("folder_id", uploadMetadata.folder_id);
    formData.append("document_type", uploadMetadata.document_type);
    formData.append("target_program", uploadMetadata.target_program);
    formData.append("keywords", uploadMetadata.keywords);

    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/upload/",
        {
          method: "POST",
          body: formData,
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        fetchAllData();
        fileInputRef.current.value = "";
        setUploadMetadata({
          folder_id: "",
          document_type: "other",
          target_program: "all",
          keywords: "",
        });
      } else {
        alert(`Upload failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed. Please try again.");
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
        fetchAllData(); // Refresh file list
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

  // Add this function after the handleUpdateDocumentMetadata function
  const handleDeleteDocument = async (documentId, filename) => {
    if (
      !confirm(
        `Are you sure you want to delete "${filename}"? This will remove it from all systems (database, Supabase, and ChromaDB).`
      )
    )
      return;

    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/documents/${documentId}/`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        fetchAllData(); // Refresh data
      } else {
        alert(`Delete failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert("Delete failed. Please try again.");
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Document Management
        </h1>
        <p className="text-gray-600">
          Organize documents into folders and manage metadata for better
          retrieval
        </p>
      </div>

      {/* Folder Management Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-800">Folders</h2>
          <button
            onClick={() => setShowFolderForm(!showFolderForm)}
            className="bg-[#063970] text-white px-4 py-2 rounded hover:bg-blue-800 transition"
          >
            {showFolderForm ? "Cancel" : "+ New Folder"}
          </button>
        </div>

        {/* New Folder Form */}
        {showFolderForm && (
          <form
            onSubmit={handleCreateFolder}
            className="mb-6 p-4 bg-gray-50 rounded border"
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Folder Name
                </label>
                <input
                  type="text"
                  value={newFolder.name}
                  onChange={(e) =>
                    setNewFolder((prev) => ({ ...prev, name: e.target.value }))
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  value={newFolder.description}
                  onChange={(e) =>
                    setNewFolder((prev) => ({
                      ...prev,
                      description: e.target.value,
                    }))
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Color
                </label>
                <input
                  type="color"
                  value={newFolder.color}
                  onChange={(e) =>
                    setNewFolder((prev) => ({ ...prev, color: e.target.value }))
                  }
                  className="w-full h-10 border border-gray-300 rounded"
                />
              </div>
            </div>
            <div className="mt-4">
              <button
                type="submit"
                className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition"
              >
                Create Folder
              </button>
            </div>
          </form>
        )}

        {/* Folders Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {folders.map((folder) => (
            <div
              key={folder.id}
              className="border rounded-lg p-4 hover:shadow-md transition"
            >
              <div className="flex items-center mb-2">
                <div
                  className="w-4 h-4 rounded mr-2"
                  style={{ backgroundColor: folder.color }}
                ></div>
                <h3 className="font-semibold text-gray-800">{folder.name}</h3>
              </div>
              <p className="text-sm text-gray-600 mb-2">{folder.description}</p>
              <p className="text-sm text-gray-500">
                {folder.document_count} documents
              </p>

              {/* Document type breakdown */}
              {folder.type_breakdown && folder.type_breakdown.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-gray-400 mb-1">Document types:</p>
                  <div className="flex flex-wrap gap-1">
                    {folder.type_breakdown.map((type) => (
                      <span
                        key={type.document_type}
                        className="text-xs bg-gray-100 px-2 py-1 rounded"
                      >
                        {type.document_type}: {type.count}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* File Upload Section with Metadata */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Upload New Document
        </h2>

        {/* Upload form with metadata */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Folder
            </label>
            <select
              value={uploadMetadata.folder_id}
              onChange={(e) =>
                setUploadMetadata((prev) => ({
                  ...prev,
                  folder_id: e.target.value,
                }))
              }
              className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select folder...</option>
              {folders.map((folder) => (
                <option key={folder.id} value={folder.id}>
                  {folder.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Document Type
            </label>
            <select
              value={uploadMetadata.document_type}
              onChange={(e) =>
                setUploadMetadata((prev) => ({
                  ...prev,
                  document_type: e.target.value,
                }))
              }
              className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="admission">Admission Requirements</option>
              <option value="enrollment">Enrollment Process</option>
              <option value="scholarship">Scholarships & Financial Aid</option>
              <option value="academic">Academic Programs</option>
              <option value="fees">Fees & Payments</option>
              <option value="policy">Policies & Procedures</option>
              <option value="contact">Contact Information</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Target Program
            </label>
            <select
              value={uploadMetadata.target_program}
              onChange={(e) =>
                setUploadMetadata((prev) => ({
                  ...prev,
                  target_program: e.target.value,
                }))
              }
              className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Programs</option>
              <option value="undergraduate">Undergraduate Programs</option>
              <option value="graduate">Graduate Programs</option>
              <option value="senior_high">Senior High School</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Keywords
            </label>
            <input
              type="text"
              value={uploadMetadata.keywords}
              onChange={(e) =>
                setUploadMetadata((prev) => ({
                  ...prev,
                  keywords: e.target.value,
                }))
              }
              placeholder="comma, separated, keywords"
              className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={extracting}
          className={`px-6 py-3 rounded-lg font-medium transition ${
            extracting
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-[#063970] hover:bg-blue-800"
          } text-white`}
        >
          {extracting ? "Processing..." : "📎 Upload Document"}
        </button>

        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          onChange={handleFileUpload}
          accept=".pdf,.txt,.doc,.docx"
        />
      </div>

      {/* Documents List with Metadata */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-800">Documents</h2>
          <button
            onClick={() => setShowMetadataPanel(!showMetadataPanel)}
            className="text-blue-600 hover:text-blue-800 transition"
          >
            {showMetadataPanel ? "Hide" : "Show"} Metadata Panel
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-[#063970] text-white">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                  Document
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                  Folder
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                  Program
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                  Keywords
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-gray-100 divide-y divide-gray-200">
              {documents.map((doc, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <div>
                      <div className="font-medium">{doc.filename}</div>
                      <div className="text-xs text-gray-500">
                        ID: {doc.document_id}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <div className="flex items-center">
                      <div
                        className="w-3 h-3 rounded mr-2"
                        style={{ backgroundColor: doc.folder.color }}
                      ></div>
                      {doc.folder.name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {doc.document_type_display}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {doc.target_program_display}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    <div className="max-w-xs overflow-hidden">
                      {doc.keywords_list.slice(0, 3).map((keyword) => (
                        <span
                          key={keyword}
                          className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mr-1 mb-1"
                        >
                          {keyword}
                        </span>
                      ))}
                      {doc.keywords_list.length > 3 && (
                        <span className="text-xs text-gray-400">
                          +{doc.keywords_list.length - 3} more
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium space-x-2">
                    <button
                      onClick={() => {
                        setSelectedDocument(doc);
                        setShowMetadataPanel(true);
                      }}
                      className="text-blue-600 hover:text-blue-900 transition"
                      title="Edit metadata"
                    >
                      ✏️
                    </button>
                    <button
                      onClick={() =>
                        handleDeleteDocument(doc.document_id, doc.filename)
                      }
                      className="text-red-600 hover:text-red-900 transition"
                      title="Delete document"
                    >
                      🗑️
                    </button>
                    <span
                      className={`inline-block w-2 h-2 rounded ${
                        doc.synced_to_chroma ? "bg-green-500" : "bg-red-500"
                      }`}
                      title={
                        doc.synced_to_chroma
                          ? "Synced to ChromaDB"
                          : "Not synced"
                      }
                    ></span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metadata Edit Panel */}
      {showMetadataPanel && selectedDocument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-4">
              Edit Document Metadata
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Document: {selectedDocument.filename}
            </p>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const updates = {
                  folder_id: parseInt(formData.get("folder_id")),
                  document_type: formData.get("document_type"),
                  target_program: formData.get("target_program"),
                  keywords: formData.get("keywords"),
                };
                handleUpdateDocumentMetadata(
                  selectedDocument.document_id,
                  updates
                );
              }}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Folder
                  </label>
                  <select
                    name="folder_id"
                    defaultValue={selectedDocument.folder.id}
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                  >
                    {folders.map((folder) => (
                      <option key={folder.id} value={folder.id}>
                        {folder.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Document Type
                  </label>
                  <select
                    name="document_type"
                    defaultValue={selectedDocument.document_type}
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                  >
                    <option value="admission">Admission Requirements</option>
                    <option value="enrollment">Enrollment Process</option>
                    <option value="scholarship">
                      Scholarships & Financial Aid
                    </option>
                    <option value="academic">Academic Programs</option>
                    <option value="fees">Fees & Payments</option>
                    <option value="policy">Policies & Procedures</option>
                    <option value="contact">Contact Information</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Target Program
                  </label>
                  <select
                    name="target_program"
                    defaultValue={selectedDocument.target_program}
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                  >
                    <option value="all">All Programs</option>
                    <option value="undergraduate">
                      Undergraduate Programs
                    </option>
                    <option value="graduate">Graduate Programs</option>
                    <option value="senior_high">Senior High School</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Keywords
                  </label>
                  <input
                    type="text"
                    name="keywords"
                    defaultValue={selectedDocument.keywords}
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                    placeholder="comma, separated, keywords"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowMetadataPanel(false);
                    setSelectedDocument(null);
                  }}
                  className="px-4 py-2 text-gray-600 border border-gray-300 rounded hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                >
                  Update Metadata
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminPage;
