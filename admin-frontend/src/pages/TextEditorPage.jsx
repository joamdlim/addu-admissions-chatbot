import React, { useState, useEffect } from "react";

const TextEditorPage = ({ fileData, onConfirmUpload, onCancel }) => {
  const [editedText, setEditedText] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    if (fileData?.extracted_text) {
      setEditedText(fileData.extracted_text);
    }
  }, [fileData]);

  const handleConfirmUpload = async () => {
    setIsUploading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/upload-processed/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_name: fileData.file_name,
            processed_text: editedText,
            file_type: fileData.file_type,
          }),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        onConfirmUpload();
      } else {
        alert(`Upload failed: ${result.error}`);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  if (!fileData) {
    return (
      <div className="p-6">
        <div className="text-center text-gray-500">No file data available</div>
      </div>
    );
  }

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-800">
              Review and Edit Document
            </h2>
            <p className="text-gray-600">
              File: <span className="font-medium">{fileData.file_name}</span>
            </p>
            <p className="text-sm text-gray-500">
              Original length: {fileData.text_length?.toLocaleString()}{" "}
              characters
            </p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={onCancel}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded transition"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirmUpload}
              disabled={isUploading || !editedText.trim()}
              className={`px-4 py-2 rounded transition ${
                isUploading || !editedText.trim()
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-[#063970] hover:bg-blue-800"
              } text-white`}
            >
              {isUploading ? "Uploading..." : "Confirm & Upload"}
            </button>
          </div>
        </div>

        {/* Text Stats */}
        <div className="flex space-x-4 text-sm text-gray-600 bg-gray-100 p-3 rounded">
          <span>
            Current length: {editedText.length.toLocaleString()} characters
          </span>
          <span>Lines: {editedText.split("\n").length}</span>
          <span>
            Words:{" "}
            {editedText.split(/\s+/).filter((word) => word.length > 0).length}
          </span>
        </div>
      </div>

      {/* Text Editor */}
      <div className="flex-1 flex flex-col">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Document Content (Edit as needed):
        </label>
        <textarea
          value={editedText}
          onChange={(e) => setEditedText(e.target.value)}
          className="flex-1 w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#063970] focus:border-transparent resize-none font-mono text-sm"
          placeholder="Document content will appear here..."
          style={{ minHeight: "400px" }}
        />
      </div>

      {/* Helper Text */}
      <div className="mt-4 text-sm text-gray-500">
        <p>
          💡 <strong>Tip:</strong> Review the extracted text for accuracy. You
          can edit any content before uploading. The document will be processed
          and stored in the knowledge base.
        </p>
      </div>
    </div>
  );
};

export default TextEditorPage;
