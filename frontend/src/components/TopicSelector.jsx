import React from "react";

const TopicSelector = ({ topics, onTopicSelect, disabled = false }) => {
  return (
    <div className="w-full flex flex-col items-center space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          What would you like to learn about?
        </h3>
        <p className="text-sm text-gray-600">
          Please select a topic to get started with your questions.
        </p>
      </div>

      <div className="flex justify-center gap-3 w-auto">
        {topics.map((topic) => (
          <button
            key={topic.id}
            onClick={() => !disabled && onTopicSelect(topic.id)}
            disabled={disabled}
            className={`
              w-70 px-5 py-3 rounded-lg border-2 transition-all duration-200 text-left
              ${
                disabled
                  ? "opacity-50 cursor-not-allowed bg-gray-100 border-gray-200"
                  : "bg-white border-gray-200 hover:border-[#063970] hover:bg-blue-50 hover:shadow-md cursor-pointer"
              }
            `}
          >
            <div className="flex flex-col">
              <h4 className="font-semibold text-gray-800 mb-1">
                {topic.label}
              </h4>
              {topic.description && (
                <p className="text-xs text-gray-600 leading-relaxed text-justify">
                  {topic.description}
                </p>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export { TopicSelector };
