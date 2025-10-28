import React from "react";
import { TopicSelector } from "./TopicSelector";
import { ActionButtons } from "./ActionButtons";

const GuidedPromptArea = ({
  conversationState,
  topics,
  buttons,
  inputEnabled,
  currentTopic,
  query,
  setQuery,
  onTopicSelect,
  onAction,
  onSend,
  onStop,
  disabled = false,
  isStreaming = false,
}) => {
  // Get current topic info for display
  const currentTopicInfo = topics?.find((t) => t.id === currentTopic);
  const currentTopicLabel = currentTopicInfo?.label;

  return (
    <div className="w-full flex flex-col items-center space-y-3">
      {/* Topic Selection State */}
      {conversationState === "topic_selection" && (
        <TopicSelector
          topics={topics || []}
          onTopicSelect={onTopicSelect}
          disabled={disabled}
        />
      )}

      {/* Action Buttons (for follow-up state) - pass all buttons except topic buttons */}
      {conversationState !== "topic_selection" && (
        <ActionButtons
          buttons={buttons ? buttons.filter((btn) => btn.type !== "topic") : []}
          onAction={onAction}
          disabled={disabled}
          currentTopic={currentTopic}
          currentTopicLabel={currentTopicLabel}
        />
      )}

      {/* Text Input (when enabled) */}
      {inputEnabled && (
        <div className="w-full">
          <form
            className="w-full flex shadow-lg rounded-lg overflow-hidden border-2 border-gray-200 hover:border-gray-300 transition-all duration-200"
            onSubmit={(e) => {
              e.preventDefault();
              if (isStreaming) {
                onStop();
              } else if (!disabled && query.trim()) {
                onSend(query);
              }
            }}
          >
            <input
              className={`text-gray-900 flex-1 px-4 py-3 border-0 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${
                disabled && !isStreaming
                  ? "bg-gray-100 text-gray-500"
                  : "bg-white"
              }`}
              placeholder={
                currentTopicLabel
                  ? `Ask about ${currentTopicLabel}...`
                  : "Ask anything about Ateneo de Davao's admissions..."
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={disabled && !isStreaming}
            />
            <div className="w-px bg-gray-300"></div>
            <button
              type="submit"
              className={`px-6 py-3 text-white text-base font-medium transition-all duration-200 ${
                isStreaming
                  ? "bg-red-600 hover:bg-red-700 hover:shadow-md active:scale-95"
                  : disabled || !query.trim()
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-[#063970] hover:bg-[#052a5a] hover:shadow-md active:scale-95"
              }`}
              disabled={!isStreaming && (disabled || !query.trim())}
            >
              {isStreaming ? "Stop" : disabled ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export { GuidedPromptArea };
