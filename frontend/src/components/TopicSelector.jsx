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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
        {topics.map((topic) => (
          <button
            key={topic.id}
            onClick={() => !disabled && onTopicSelect(topic.id)}
            disabled={disabled}
            className={`
              p-4 rounded-lg border-2 text-left transition-all duration-200
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
              <p className="text-sm text-gray-600 leading-relaxed">
                {topic.description}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export { TopicSelector };
