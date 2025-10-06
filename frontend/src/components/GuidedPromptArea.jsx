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
  disabled = false,
}) => {
  // Get current topic info for display
  const currentTopicInfo = topics?.find(t => t.id === currentTopic);
  const currentTopicLabel = currentTopicInfo?.label;

  return (
    <div className="w-full flex flex-col items-center space-y-4">
      {/* Topic Selection State */}
      {conversationState === 'topic_selection' && (
        <TopicSelector
          topics={topics || []}
          onTopicSelect={onTopicSelect}
          disabled={disabled}
        />
      )}

      {/* Action Buttons (for follow-up state) */}
      {buttons && buttons.length > 0 && (
        <ActionButtons
          buttons={buttons}
          onAction={onAction}
          disabled={disabled}
          currentTopic={currentTopicLabel}
        />
      )}

      {/* Text Input (when enabled) */}
      {inputEnabled && (
        <div className="w-full">
          {/* Show current topic context */}
          {currentTopicLabel && (
            <div className="mb-3 text-center">
              <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 text-[#063970] text-sm">
                <span className="w-2 h-2 bg-[#063970] rounded-full mr-2"></span>
                {currentTopicLabel}
              </div>
            </div>
          )}
          
          <form
            className="w-full flex"
            onSubmit={(e) => {
              e.preventDefault();
              if (!disabled && query.trim()) onSend(query);
            }}
          >
            <input
              className={`text-gray-900 flex-1 rounded-l px-4 py-3 border border-gray-300 text-base ${
                disabled ? "bg-gray-100" : ""
              }`}
              placeholder={
                currentTopicLabel 
                  ? `Ask about ${currentTopicLabel}...`
                  : "Ask anything about Ateneo de Davao's admissions"
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={disabled}
            />
            <button
              type="submit"
              className={`px-6 rounded-r text-white text-base ${
                disabled || !query.trim()
                  ? "bg-gray-400 cursor-not-allowed" 
                  : "bg-[#063970] hover:bg-[#052a5a]"
              }`}
              disabled={disabled || !query.trim()}
            >
              {disabled ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
      )}

      {/* Help text */}
      <div className="text-center text-xs text-gray-500 max-w-md">
        {conversationState === 'topic_selection' 
          ? "Select a topic above to start your conversation with focused, relevant answers."
          : inputEnabled
            ? "Type your question and press Enter or click Send."
            : "Use the buttons above to continue or change topics."
        }
      </div>
    </div>
  );
};

export { GuidedPromptArea };
