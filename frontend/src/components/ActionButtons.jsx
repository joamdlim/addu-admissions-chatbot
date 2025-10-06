import React from "react";

const ActionButtons = ({ buttons, onAction, disabled = false, currentTopic = null }) => {
  if (!buttons || buttons.length === 0) return null;

  return (
    <div className="w-full flex flex-col items-center space-y-3">
      {/* Show current topic if available */}
      {currentTopic && (
        <div className="text-center">
          <p className="text-sm text-gray-600">
            Currently discussing: <span className="font-semibold text-[#063970]">{currentTopic}</span>
          </p>
        </div>
      )}
      
      <div className="flex flex-wrap justify-center gap-3">
        {buttons.map((button) => (
          <button
            key={button.id}
            onClick={() => !disabled && onAction(button.id)}
            disabled={disabled}
            className={`
              px-6 py-3 rounded-lg font-medium transition-all duration-200
              ${button.type === 'topic' 
                ? disabled 
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200'
                  : 'bg-white text-[#063970] border-2 border-[#063970] hover:bg-[#063970] hover:text-white'
                : disabled
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-[#063970] text-white hover:bg-[#052a5a] shadow-md hover:shadow-lg'
              }
            `}
          >
            {button.label}
          </button>
        ))}
      </div>
    </div>
  );
};

export { ActionButtons };
