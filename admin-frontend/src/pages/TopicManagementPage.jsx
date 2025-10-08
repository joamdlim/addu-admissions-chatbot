import React, { useState, useEffect } from "react";

const TopicManagementPage = () => {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [keywords, setKeywords] = useState([]);
  const [newKeyword, setNewKeyword] = useState("");
  const [addingKeyword, setAddingKeyword] = useState(false);

  // Fetch topics
  const fetchTopics = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/chatbot/admin/topics/"
      );
      const data = await response.json();
      setTopics(data.topics || []);

      // Select first topic by default
      if (data.topics && data.topics.length > 0 && !selectedTopic) {
        setSelectedTopic(data.topics[0]);
      }
    } catch (error) {
      console.error("Error fetching topics:", error);
      alert("Failed to load topics");
    } finally {
      setLoading(false);
    }
  };

  // Fetch keywords for selected topic
  const fetchKeywords = async (topicId) => {
    if (!topicId) return;

    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/topics/${topicId}/keywords/`
      );
      const data = await response.json();
      setKeywords(data.keywords || []);
    } catch (error) {
      console.error("Error fetching keywords:", error);
      alert("Failed to load keywords");
    }
  };

  // Add new keyword
  const handleAddKeyword = async () => {
    if (!newKeyword.trim() || !selectedTopic) return;

    setAddingKeyword(true);
    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/topics/${selectedTopic.topic_id}/keywords/add/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keyword: newKeyword.trim() }),
        }
      );

      const result = await response.json();

      if (response.ok) {
        setNewKeyword("");
        fetchKeywords(selectedTopic.topic_id);
        fetchTopics(); // Refresh to update keyword counts
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error adding keyword:", error);
      alert("Failed to add keyword");
    } finally {
      setAddingKeyword(false);
    }
  };

  // Toggle keyword active status
  const handleToggleKeyword = async (keywordId, isActive) => {
    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/keywords/${keywordId}/update/`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_active: !isActive }),
        }
      );

      const result = await response.json();

      if (response.ok) {
        fetchKeywords(selectedTopic.topic_id);
        fetchTopics(); // Refresh to update keyword counts
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error updating keyword:", error);
      alert("Failed to update keyword");
    }
  };

  // Delete keyword
  const handleDeleteKeyword = async (keywordId, keyword) => {
    if (!confirm(`Are you sure you want to delete the keyword "${keyword}"?`))
      return;

    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/keywords/${keywordId}/delete/`,
        {
          method: "DELETE",
        }
      );

      const result = await response.json();

      if (response.ok) {
        fetchKeywords(selectedTopic.topic_id);
        fetchTopics(); // Refresh to update keyword counts
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error deleting keyword:", error);
      alert("Failed to delete keyword");
    }
  };

  // Bulk add keywords
  const handleBulkAddKeywords = async (keywordsText) => {
    if (!keywordsText.trim() || !selectedTopic) return;

    const keywordList = keywordsText
      .split(/[,\n]/)
      .map((k) => k.trim())
      .filter((k) => k.length > 0);

    if (keywordList.length === 0) return;

    setAddingKeyword(true);
    try {
      const response = await fetch(
        `http://localhost:8000/chatbot/admin/topics/${selectedTopic.topic_id}/keywords/bulk/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keywords: keywordList }),
        }
      );

      const result = await response.json();

      if (response.ok) {
        alert(
          `Successfully added ${result.added_count} keywords (${result.skipped_count} duplicates skipped)`
        );
        fetchKeywords(selectedTopic.topic_id);
        fetchTopics(); // Refresh to update keyword counts
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Error bulk adding keywords:", error);
      alert("Failed to add keywords");
    } finally {
      setAddingKeyword(false);
    }
  };

  useEffect(() => {
    fetchTopics();
  }, []);

  useEffect(() => {
    if (selectedTopic) {
      fetchKeywords(selectedTopic.topic_id);
    }
  }, [selectedTopic]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-600">Loading topics...</div>
      </div>
    );
  }

  return (
    <div className="flex h-full bg-gray-50 overflow-hidden">
      {/* LEFT SIDE - Topic Selection */}
      <div className="w-1/3 border-r border-gray-300 bg-white p-4 overflow-y-auto">
        <h1 className="text-lg font-bold text-gray-900 mb-4">
          Topic Management
        </h1>

        <div className="space-y-3">
          {topics.map((topic) => (
            <div
              key={topic.topic_id}
              onClick={() => setSelectedTopic(topic)}
              className={`p-4 border-2 rounded-lg cursor-pointer transition ${
                selectedTopic?.topic_id === topic.topic_id
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold text-sm text-gray-800">
                  {topic.label}
                </h3>
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                    topic.is_active
                      ? "bg-green-100 text-green-800"
                      : "bg-red-100 text-red-800"
                  }`}
                >
                  {topic.is_active ? "Active" : "Inactive"}
                </span>
              </div>

              <p className="text-xs text-gray-600 mb-2 line-clamp-2">
                {topic.description}
              </p>

              <div className="flex justify-between items-center text-xs text-gray-500">
                <span>ID: {topic.topic_id}</span>
                <span>{topic.keyword_count} keywords</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT SIDE - Keyword Management */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedTopic ? (
          <>
            {/* Header */}
            <div className="bg-white border-b border-gray-300 p-4">
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h2 className="text-lg font-bold text-gray-900">
                    {selectedTopic.label}
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">
                    {selectedTopic.description}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-700">
                    {keywords.filter((k) => k.is_active).length} active keywords
                  </div>
                  <div className="text-xs text-gray-500">
                    {keywords.length} total keywords
                  </div>
                </div>
              </div>

              {/* Add Single Keyword */}
              <div className="flex gap-2 mb-3">
                <input
                  type="text"
                  value={newKeyword}
                  onChange={(e) => setNewKeyword(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleAddKeyword()}
                  placeholder="Add new keyword..."
                  className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={addingKeyword}
                />
                <button
                  onClick={handleAddKeyword}
                  disabled={addingKeyword || !newKeyword.trim()}
                  className={`px-4 py-2 text-sm rounded font-medium transition ${
                    addingKeyword || !newKeyword.trim()
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-[#063970] hover:bg-blue-800"
                  } text-white`}
                >
                  {addingKeyword ? "Adding..." : "Add Keyword"}
                </button>
              </div>

              {/* Bulk Add Button */}
              <BulkAddModal
                onBulkAdd={handleBulkAddKeywords}
                disabled={addingKeyword}
              />
            </div>

            {/* Keywords List */}
            <div className="flex-1 overflow-y-auto p-4">
              {keywords.length === 0 ? (
                <div className="text-center text-gray-500 mt-8">
                  <div className="text-4xl mb-2">🏷️</div>
                  <p>No keywords added yet</p>
                  <p className="text-sm mt-1">
                    Add keywords to help the chatbot understand this topic
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {keywords.map((keyword) => (
                    <div
                      key={keyword.id}
                      className={`p-3 border rounded-lg transition ${
                        keyword.is_active
                          ? "border-green-300 bg-green-50"
                          : "border-gray-300 bg-gray-50"
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span
                          className={`font-medium text-sm ${
                            keyword.is_active
                              ? "text-green-800"
                              : "text-gray-600"
                          }`}
                        >
                          {keyword.keyword}
                        </span>
                        <div className="flex gap-1">
                          <button
                            onClick={() =>
                              handleToggleKeyword(keyword.id, keyword.is_active)
                            }
                            className={`text-xs px-2 py-1 rounded transition ${
                              keyword.is_active
                                ? "bg-yellow-100 text-yellow-800 hover:bg-yellow-200"
                                : "bg-green-100 text-green-800 hover:bg-green-200"
                            }`}
                            title={
                              keyword.is_active ? "Deactivate" : "Activate"
                            }
                          >
                            {keyword.is_active ? "⏸️" : "▶️"}
                          </button>
                          <button
                            onClick={() =>
                              handleDeleteKeyword(keyword.id, keyword.keyword)
                            }
                            className="text-xs px-2 py-1 rounded bg-red-100 text-red-800 hover:bg-red-200 transition"
                            title="Delete keyword"
                          >
                            🗑️
                          </button>
                        </div>
                      </div>

                      <div className="text-xs text-gray-500">
                        <div>
                          Added:{" "}
                          {new Date(keyword.created_at).toLocaleDateString()}
                        </div>
                        {keyword.created_by && (
                          <div>By: {keyword.created_by}</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-4xl mb-2">🏷️</div>
              <p>Select a topic to manage its keywords</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Bulk Add Modal Component
const BulkAddModal = ({ onBulkAdd, disabled }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [bulkKeywords, setBulkKeywords] = useState("");

  const handleSubmit = () => {
    onBulkAdd(bulkKeywords);
    setBulkKeywords("");
    setIsOpen(false);
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        disabled={disabled}
        className={`text-sm px-3 py-1.5 rounded border transition ${
          disabled
            ? "border-gray-300 text-gray-400 cursor-not-allowed"
            : "border-gray-300 text-gray-700 hover:bg-gray-50"
        }`}
      >
        📝 Bulk Add Keywords
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-4">Bulk Add Keywords</h3>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Keywords (comma or line separated)
              </label>
              <textarea
                value={bulkKeywords}
                onChange={(e) => setBulkKeywords(e.target.value)}
                placeholder="Enter keywords separated by commas or new lines:&#10;&#10;program, degree, course&#10;bachelor, undergraduate&#10;computer science, information technology"
                className="w-full h-40 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                You can paste keywords from the existing topics.py file here
              </p>
            </div>

            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setIsOpen(false);
                  setBulkKeywords("");
                }}
                className="px-4 py-2 text-sm text-gray-600 border border-gray-300 rounded hover:bg-gray-50 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={!bulkKeywords.trim()}
                className={`px-4 py-2 text-sm rounded font-medium transition ${
                  !bulkKeywords.trim()
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-[#063970] hover:bg-blue-800"
                } text-white`}
              >
                Add Keywords
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TopicManagementPage;
