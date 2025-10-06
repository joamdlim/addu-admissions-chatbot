import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { GuidedPromptArea } from "../components/GuidedPromptArea";

const GuidedChatPage = () => {
  // Conversation state
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  // Guided conversation state
  const [topics, setTopics] = useState([]);
  const [conversationState, setConversationState] = useState("topic_selection");
  const [currentTopic, setCurrentTopic] = useState(null);
  const [buttons, setButtons] = useState([]);
  const [inputEnabled, setInputEnabled] = useState(false);

  // Load topics on component mount
  useEffect(() => {
    const loadTopics = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/chatbot/topics/");
        if (response.ok) {
          const data = await response.json();
          setTopics(data.topics);
          console.log("✅ Topics loaded:", data.topics.length);
        } else {
          console.error("Failed to load topics:", response.status);
        }
      } catch (error) {
        console.error("Error loading topics:", error);
      }
    };

    loadTopics();
  }, []);

  // Initialize session
  useEffect(() => {
    const initializeSession = () => {
      const newSessionId = `guided_session_${Date.now()}_${Math.random()
        .toString(36)
        .substr(2, 9)}`;
      setSessionId(newSessionId);
      console.log("✅ Guided session initialized:", newSessionId);
    };

    initializeSession();
  }, []);

  // Add initial welcome message
  useEffect(() => {
    if (topics.length > 0 && messages.length === 0) {
      const welcomeMessage = {
        role: "bot",
        content:
          "Welcome to the ADDU Admissions Assistant! I'm here to help you with information about admissions, enrollment, fees, programs, and more. Please select a topic below to get started with focused, accurate answers.",
        timestamp: new Date().toISOString(),
      };
      setMessages([welcomeMessage]);
    }
  }, [topics, messages.length]);

  const handleGuidedRequest = async (
    userInput,
    actionType,
    actionData = null
  ) => {
    setIsLoading(true);

    try {
      const requestBody = {
        user_input: userInput,
        action_type: actionType,
        action_data: actionData,
        session_id: sessionId,
      };

      console.log("🚀 Guided request:", requestBody);

      const response = await fetch(
        "http://127.0.0.1:8000/chatbot/chat/guided/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("📨 Guided response:", data);

      // Update conversation state
      setConversationState(data.state || "topic_selection");
      setCurrentTopic(data.current_topic || null);
      setButtons(data.buttons || []);
      setInputEnabled(data.input_enabled || false);

      // Update session ID if provided
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
      }

      // Add messages if there's a response
      if (data.response) {
        const newMessages = [];

        // Add user message if there was user input
        if (userInput && userInput.trim()) {
          newMessages.push({
            role: "user",
            content: userInput,
            timestamp: new Date().toISOString(),
          });
        }

        // Add bot response
        newMessages.push({
          role: "bot",
          content: data.response,
          timestamp: new Date().toISOString(),
          sources: data.sources || [],
          autoDetectedTopic: data.auto_detected_topic,
        });

        setMessages((prev) => [...prev, ...newMessages]);
      }

      // Handle errors
      if (data.error) {
        const errorMessage = {
          role: "bot",
          content: `Error: ${data.error}`,
          timestamp: new Date().toISOString(),
          isError: true,
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error("Guided chat error:", error);
      const errorMessage = {
        role: "bot",
        content: "I'm sorry, I encountered an error. Please try again.",
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setQuery(""); // Clear input after sending
    }
  };

  const handleTopicSelect = (topicId) => {
    console.log("🎯 Topic selected:", topicId);
    handleGuidedRequest("", "topic_selection", topicId);
  };

  const handleAction = (actionId) => {
    console.log("🔄 Action triggered:", actionId);
    handleGuidedRequest("", "action", actionId);
  };

  const handleSend = (message) => {
    console.log("💬 Message sent:", message);
    handleGuidedRequest(message, "message");
  };

  return (
    <div className="w-full h-full bg-gray-50 flex flex-col items-center">
      {/* Messages */}
      <div className="flex-1 w-[900px] max-w-[900px] py-8 overflow-y-auto space-y-4 min-h-0">
        {messages.map((m, idx) => {
          const isUser = m.role === "user";
          return (
            <div
              key={idx}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-lg p-4 text-base leading-relaxed ${
                  isUser
                    ? "bg-[#063970] text-white rounded-br-sm"
                    : m.isError
                    ? "bg-red-50 text-red-800 border border-red-200 rounded-bl-sm"
                    : "bg-white text-gray-900 border border-gray-200 rounded-bl-sm shadow-sm"
                }`}
              >
                {/* Auto-detected topic indicator */}
                {m.autoDetectedTopic && (
                  <div className="mb-2 text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    Auto-detected topic: {m.autoDetectedTopic}
                  </div>
                )}

                {isUser ? (
                  m.content
                ) : (
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => (
                        <p className="mb-2 last:mb-0">{children}</p>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-disc list-inside mb-2 space-y-1">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal list-inside mb-2 space-y-1">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => (
                        <li className="mb-1">{children}</li>
                      ),
                      strong: ({ children }) => (
                        <strong className="font-semibold text-gray-900">
                          {children}
                        </strong>
                      ),
                      h1: ({ children }) => (
                        <h1 className="text-xl font-bold mb-2">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-lg font-semibold mb-2">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-md font-semibold mb-1">
                          {children}
                        </h3>
                      ),
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 underline hover:no-underline transition-colors duration-200"
                        >
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {m.content}
                  </ReactMarkdown>
                )}

                {/* Sources */}
                {m.sources && m.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <details className="text-sm">
                      <summary className="cursor-pointer text-gray-600 hover:text-gray-800">
                        Sources ({m.sources.length})
                      </summary>
                      <div className="mt-2 space-y-1">
                        {m.sources.map((source, sidx) => (
                          <div
                            key={sidx}
                            className="text-xs text-gray-500 bg-gray-50 p-2 rounded"
                          >
                            <div className="font-medium">{source.filename}</div>
                            <div>
                              Relevance: {(source.relevance * 100).toFixed(1)}%
                            </div>
                            {source._debug && (
                              <div>
                                Keywords:{" "}
                                {source._debug.matched_keywords?.join(", ")}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg p-4 text-gray-600 shadow-sm">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-[#063970]"></div>
                <span>Processing your request...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Guided Prompt Area - fixed at bottom */}
      <div className="w-[900px] max-w-[900px] pb-6 bg-gray-50 flex-shrink-0">
        <GuidedPromptArea
          conversationState={conversationState}
          topics={topics}
          buttons={buttons}
          inputEnabled={inputEnabled}
          currentTopic={currentTopic}
          query={query}
          setQuery={setQuery}
          onTopicSelect={handleTopicSelect}
          onAction={handleAction}
          onSend={handleSend}
          disabled={isLoading}
        />
      </div>
    </div>
  );
};

export { GuidedChatPage };
