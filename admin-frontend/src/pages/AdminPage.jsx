import React, { useState, useRef, useEffect } from "react";

const AdminPage = ({ onFileForReview }) => {
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [folders, setFolders] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [syncingFile, setSyncingFile] = useState(null);

  // Folder management state
  const [showFolderForm, setShowFolderForm] = useState(false);
  const [editingFolder, setEditingFolder] = useState(null);
  const [newFolder, setNewFolder] = useState({
    name: "",
    description: "",
    color: "#063970",
  });
  const [selectedFolder, setSelectedFolder] = useState(null);

  // Document metadata state
  const [showMetadataPanel, setShowMetadataPanel] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);

  // Enhanced upload state - staged upload
  const [stagedFile, setStagedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadMetadata, setUploadMetadata] = useState({
    folder_id: "",
    document_type: "other",
    keywords: "",
  });
  const [uploading, setUploading] = useState(false);

  // Keywords expansion state
  const [expandedKeywords, setExpandedKeywords] = useState({});

  // Fetch all data
  const fetchAllData = async () => {
    setLoading(true);
    try {
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

  const handleUpdateFolder = async (folderId, updates) => {
    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/folders/${folderId}/`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updates),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        setEditingFolder(null);
        fetchAllData();
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error updating folder:", error);
      alert("Failed to update folder");
    }
  };

  const handleDeleteFolder = async (folderId, folderName) => {
    if (
      !confirm(
        `Are you sure you want to delete "${folderName}"? This will also delete all documents in this folder.`
      )
    )
      return;

    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/folders/${folderId}/`,
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
        fetchAllData();
      } else {
        alert(`Delete failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert("Delete failed. Please try again.");
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

  // Drag and drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileStaging(files[0]);
    }
  };

  // Stage file (not upload yet)
  const handleFileStaging = (file) => {
    if (!file) return;

    // Validate file type
    const validTypes = [".pdf", ".txt", ".doc", ".docx"];
    const fileExtension = "." + file.name.split(".").pop().toLowerCase();

    if (!validTypes.includes(fileExtension)) {
      alert("Please upload only PDF, TXT, DOC, or DOCX files.");
      return;
    }

    setStagedFile(file);
    // Reset metadata for new file
    setUploadMetadata({
      folder_id: "",
      document_type: "other",
      keywords: "",
    });
  };

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleFileStaging(file);
    }
  };

  const handleClearStagedFile = () => {
    setStagedFile(null);
    setUploadMetadata({
      folder_id: "",
      document_type: "other",
      keywords: "",
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Final upload to Supabase and ChromaDB
  const handleFinalUpload = async () => {
    if (!stagedFile) {
      alert("No file staged for upload");
      return;
    }

    if (!uploadMetadata.folder_id) {
      alert("Please select a folder");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", stagedFile);
    formData.append("folder_id", uploadMetadata.folder_id);
    formData.append("document_type", uploadMetadata.document_type);
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
        handleClearStagedFile();
      } else {
        alert(`Upload failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId, filename) => {
    if (
      !confirm(
        `Are you sure you want to delete "${filename}"? This will remove it from all systems.`
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
        fetchAllData();
      } else {
        alert(`Delete failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert("Delete failed. Please try again.");
    }
  };

  // Toggle keywords expansion
  const toggleKeywordsExpansion = (documentId) => {
    setExpandedKeywords((prev) => ({
      ...prev,
      [documentId]: !prev[documentId],
    }));
  };

  // Filter documents by selected folder
  const filteredDocuments = selectedFolder
    ? documents.filter((doc) => doc.folder.id === selectedFolder.id)
    : documents;

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* LEFT SIDE - Upload Document Only */}
      <div className="w-1/4 border-r border-gray-300 bg-white p-4 overflow-y-auto">
        <h1 className="text-lg font-bold text-gray-900 mb-4">
          Upload Document
        </h1>

        {/* Drag & Drop Area */}
        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !stagedFile && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
            isDragging
              ? "border-blue-500 bg-blue-50"
              : stagedFile
              ? "border-green-500 bg-green-50"
              : "border-gray-300 hover:border-gray-400"
          }`}
        >
          {stagedFile ? (
            <div>
              <div className="text-green-600 text-3xl mb-2">✓</div>
              <p className="text-sm font-medium text-gray-700">
                {stagedFile.name}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {(stagedFile.size / 1024).toFixed(2)} KB
              </p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleClearStagedFile();
                }}
                className="mt-2 text-xs text-red-600 hover:text-red-800"
              >
                Remove file
              </button>
            </div>
          ) : (
            <div>
              <div className="text-gray-400 text-3xl mb-2">📎</div>
              <p className="text-sm text-gray-600 font-medium">
                Drop file here or click to upload
              </p>
              <p className="text-xs text-gray-500 mt-1">PDF, TXT, DOC, DOCX</p>
            </div>
          )}
        </div>

        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          onChange={handleFileInputChange}
          accept=".pdf,.txt,.doc,.docx"
        />

        {/* Metadata Form - Only shows when file is staged */}
        {stagedFile && (
          <div className="mt-4 space-y-3">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Folder *
              </label>
              <select
                value={uploadMetadata.folder_id}
                onChange={(e) =>
                  setUploadMetadata((prev) => ({
                    ...prev,
                    folder_id: e.target.value,
                  }))
                }
                className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
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
              <label className="block text-xs font-medium text-gray-700 mb-1">
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
                className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
              <label className="block text-xs font-medium text-gray-700 mb-1">
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
                className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <button
              onClick={handleFinalUpload}
              disabled={uploading || !uploadMetadata.folder_id}
              className={`w-full px-3 py-2 text-sm rounded font-medium transition ${
                uploading || !uploadMetadata.folder_id
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-[#063970] hover:bg-blue-800"
              } text-white`}
            >
              {uploading ? "Uploading..." : "Submit Upload"}
            </button>
          </div>
        )}
      </div>

      {/* RIGHT SIDE - Folders & Documents Table */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Folders Section - Horizontal Scrolling */}
        <div className="bg-white border-b border-gray-300 p-4">
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-lg font-bold text-gray-900">
              Document Management
            </h2>
            <button
              onClick={() => setShowFolderForm(!showFolderForm)}
              className="bg-[#063970] text-white px-3 py-1.5 text-xs rounded hover:bg-blue-800 transition"
            >
              {showFolderForm ? "Cancel" : "+ New Folder"}
            </button>
          </div>

          {/* New Folder Form */}
          {showFolderForm && (
            <form
              onSubmit={handleCreateFolder}
              className="mb-3 p-3 bg-gray-50 rounded border"
            >
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Folder Name
                  </label>
                  <input
                    type="text"
                    value={newFolder.name}
                    onChange={(e) =>
                      setNewFolder((prev) => ({
                        ...prev,
                        name: e.target.value,
                      }))
                    }
                    className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
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
                    className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Color
                  </label>
                  <input
                    type="color"
                    value={newFolder.color}
                    onChange={(e) =>
                      setNewFolder((prev) => ({
                        ...prev,
                        color: e.target.value,
                      }))
                    }
                    className="w-full h-7 border border-gray-300 rounded"
                  />
                </div>
              </div>
              <button
                type="submit"
                className="mt-2 bg-green-600 text-white px-3 py-1 text-xs rounded hover:bg-green-700 transition"
              >
                Create Folder
              </button>
            </form>
          )}

          <div className="flex items-start gap-3">
            {/* All Documents - Fixed (No scrolling) */}
            <div
              onClick={() => setSelectedFolder(null)}
              className={`flex-shrink-0 w-44 h-24 border-2 rounded-lg p-3 cursor-pointer transition ${
                selectedFolder === null
                  ? "bg-blue-50 border-blue-500"
                  : "hover:bg-gray-50 border-gray-300"
              }`}
            >
              <div className="flex items-center mb-1">
                <div className="w-3 h-3 rounded mr-1.5 bg-gray-400"></div>
                <h3 className="font-semibold text-sm text-gray-800">
                  All Documents
                </h3>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {documents.length} documents
              </p>
            </div>

            {/* Vertical Separator */}
            <div className="w-px h-24 bg-gray-300 flex-shrink-0"></div>

            {/* Scrollable Folders */}
            <div className="flex-1 overflow-x-auto">
              <div className="flex gap-3 pb-2">
                {folders.map((folder) => (
                  <div
                    key={folder.id}
                    className={`relative flex-shrink-0 w-44 h-24 border-2 rounded-lg p-3 transition ${
                      selectedFolder?.id === folder.id
                        ? "bg-blue-50 border-blue-500"
                        : "hover:bg-gray-50 border-gray-300"
                    }`}
                  >
                    {/* Edit and Delete buttons at top right */}
                    <div className="absolute top-1.5 right-1.5 flex gap-0.5">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingFolder(folder);
                        }}
                        className="text-blue-600 hover:text-blue-800 text-xs p-0.5"
                        title="Edit folder"
                      >
                        ✏️
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteFolder(folder.id, folder.name);
                        }}
                        className="text-red-600 hover:text-red-800 text-xs p-0.5"
                        title="Delete folder"
                      >
                        🗑️
                      </button>
                    </div>

                    <div
                      onClick={() => setSelectedFolder(folder)}
                      className="cursor-pointer h-full"
                    >
                      <div className="flex items-center mb-1">
                        <div
                          className="w-3 h-3 rounded mr-1.5"
                          style={{ backgroundColor: folder.color }}
                        ></div>
                        <h3 className="font-semibold text-sm text-gray-800 truncate pr-10">
                          {folder.name}
                        </h3>
                      </div>
                      <p className="text-xs text-gray-600 mb-1 line-clamp-1 h-4">
                        {folder.description}
                      </p>
                      <p className="text-xs text-gray-500 mt-auto">
                        {folder.document_count} documents
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Documents Table - Scrollable */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="mb-3 flex justify-between items-center">
            <div>
              <h3 className="text-base font-bold text-gray-900">Documents</h3>
              <p className="text-xs text-gray-600 mt-0.5">
                Showing {filteredDocuments.length} of {documents.length}{" "}
                documents
              </p>
            </div>
            {/* Filter indicator on the same line */}
            {selectedFolder && (
              <div className="flex items-center bg-blue-100 px-3 py-1.5 rounded">
                <div
                  className="w-2.5 h-2.5 rounded mr-1.5"
                  style={{ backgroundColor: selectedFolder.color }}
                ></div>
                <span className="text-xs font-medium text-blue-900">
                  Filtered by: {selectedFolder.name}
                </span>
                <button
                  onClick={() => setSelectedFolder(null)}
                  className="ml-2 text-blue-700 hover:text-blue-900 text-sm"
                >
                  ✕
                </button>
              </div>
            )}
          </div>

          {/* Documents Table */}
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-[#063970] text-white">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider">
                      Document
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider">
                      Folder
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider">
                      Keywords
                    </th>
                    <th className="px-4 py-2 text-center text-xs font-medium uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredDocuments.length === 0 ? (
                    <tr>
                      <td
                        colSpan="6"
                        className="px-4 py-6 text-center text-gray-500 text-sm"
                      >
                        {selectedFolder
                          ? `No documents in "${selectedFolder.name}" folder`
                          : "No documents uploaded yet"}
                      </td>
                    </tr>
                  ) : (
                    filteredDocuments.map((doc, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-xs text-gray-900">
                          <div className="font-medium">{doc.filename}</div>
                        </td>
                        <td className="px-4 py-3 text-xs">
                          <div className="flex items-center">
                            <div
                              className="w-2.5 h-2.5 rounded mr-1.5"
                              style={{ backgroundColor: doc.folder.color }}
                            ></div>
                            <span className="text-gray-700">
                              {doc.folder.name}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          {doc.document_type_display}
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          <div className="max-w-xs">
                            {doc.keywords_list.slice(0, 3).map((keyword) => (
                              <span
                                key={keyword}
                                className="inline-block bg-blue-100 text-blue-800 text-xs px-1.5 py-0.5 rounded mr-1 mb-1"
                              >
                                {keyword}
                              </span>
                            ))}
                            {doc.keywords_list.length > 3 && (
                              <button
                                onClick={() =>
                                  toggleKeywordsExpansion(doc.document_id)
                                }
                                className="text-xs text-blue-600 hover:text-blue-800 ml-1"
                              >
                                {expandedKeywords[doc.document_id]
                                  ? "Show less"
                                  : `+${doc.keywords_list.length - 3} more`}
                              </button>
                            )}
                            {expandedKeywords[doc.document_id] &&
                              doc.keywords_list.length > 3 && (
                                <div className="mt-1">
                                  {doc.keywords_list.slice(3).map((keyword) => (
                                    <span
                                      key={keyword}
                                      className="inline-block bg-blue-100 text-blue-800 text-xs px-1.5 py-0.5 rounded mr-1 mb-1"
                                    >
                                      {keyword}
                                    </span>
                                  ))}
                                </div>
                              )}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium whitespace-nowrap ${
                              doc.synced_to_chroma
                                ? "bg-green-100 text-green-800"
                                : "bg-red-100 text-red-800"
                            }`}
                            title={
                              doc.synced_to_chroma
                                ? "Synced to ChromaDB"
                                : "Not synced"
                            }
                          >
                            {doc.synced_to_chroma ? "✓ Synced" : "✗ Not synced"}
                          </span>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-right text-xs font-medium space-x-2">
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
                              handleDeleteDocument(
                                doc.document_id,
                                doc.filename
                              )
                            }
                            className="text-red-600 hover:text-red-900 transition"
                            title="Delete document"
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
          </div>
        </div>
      </div>

      {/* Folder Edit Modal */}
      {editingFolder && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-5 w-full max-w-md">
            <h3 className="text-base font-semibold mb-3">Edit Folder</h3>
            <p className="text-xs text-gray-600 mb-3">
              Folder: {editingFolder.name}
            </p>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const updates = {
                  name: formData.get("name"),
                  description: formData.get("description"),
                  color: formData.get("color"),
                };
                handleUpdateFolder(editingFolder.id, updates);
              }}
            >
              <div className="space-y-3 mb-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Folder Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    defaultValue={editingFolder.name}
                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                    required
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <input
                    type="text"
                    name="description"
                    defaultValue={editingFolder.description}
                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Color
                  </label>
                  <input
                    type="color"
                    name="color"
                    defaultValue={editingFolder.color}
                    className="w-full h-8 border border-gray-300 rounded"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setEditingFolder(null)}
                  className="px-3 py-1.5 text-xs text-gray-600 border border-gray-300 rounded hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                >
                  Update Folder
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Metadata Edit Panel (Modal) */}
      {showMetadataPanel && selectedDocument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-5 w-full max-w-2xl">
            <h3 className="text-base font-semibold mb-3">
              Edit Document Metadata
            </h3>
            <p className="text-xs text-gray-600 mb-3">
              Document: {selectedDocument.filename}
            </p>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const updates = {
                  folder_id: parseInt(formData.get("folder_id")),
                  document_type: formData.get("document_type"),
                  keywords: formData.get("keywords"),
                };
                handleUpdateDocumentMetadata(
                  selectedDocument.document_id,
                  updates
                );
              }}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Folder
                  </label>
                  <select
                    name="folder_id"
                    defaultValue={selectedDocument.folder.id}
                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                  >
                    {folders.map((folder) => (
                      <option key={folder.id} value={folder.id}>
                        {folder.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Document Type
                  </label>
                  <select
                    name="document_type"
                    defaultValue={selectedDocument.document_type}
                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
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

                <div className="md:col-span-2">
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Keywords
                  </label>
                  <input
                    type="text"
                    name="keywords"
                    defaultValue={selectedDocument.keywords}
                    className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                    placeholder="comma, separated, keywords"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => {
                    setShowMetadataPanel(false);
                    setSelectedDocument(null);
                  }}
                  className="px-3 py-1.5 text-xs text-gray-600 border border-gray-300 rounded hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition"
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
