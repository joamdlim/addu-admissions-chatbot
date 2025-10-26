import React from "react";

const ActionButtons = ({
  buttons,
  onAction,
  disabled = false,
  currentTopic = null,
}) => {
  if (!buttons || buttons.length === 0) return null;

  // Separate school buttons and action buttons
  const schoolButtons = buttons.filter((btn) => btn.type === "school");
  const actionButtons = buttons.filter(
    (btn) => btn.type !== "school" && btn.type !== "topic"
  );

  return (
    <div className="w-full flex flex-col items-center space-y-2">
      {/* School buttons (for Program Curriculum only) */}
      {schoolButtons.length > 0 && (
        <div className="flex flex-wrap justify-center gap-1 w-full">
          {schoolButtons.map((button) => (
            <button
              key={button.id}
              onClick={() => !disabled && onAction(button.id)}
              disabled={disabled}
              className={`
                px-2 py-1 rounded text-xs font-medium transition-all duration-200
                ${
                  disabled
                    ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                    : "bg-blue-50 text-[#063970] border border-blue-200 hover:bg-blue-100 hover:border-blue-300"
                }
              `}
            >
              {button.label}
            </button>
          ))}
        </div>
      )}

      {/* Action buttons (Change Topic, etc.) */}
      {actionButtons.length > 0 && (
        <div className="flex justify-center gap-2">
          {actionButtons.map((button) => (
            <button
              key={button.id}
              onClick={() => !disabled && onAction(button.id)}
              disabled={disabled}
              className={`
                px-4 py-2 rounded text-sm font-medium transition-all duration-200 whitespace-nowrap
                ${
                  disabled
                    ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                    : "bg-[#063970] text-white hover:bg-[#052a5a] shadow-md hover:shadow-lg"
                }
              `}
            >
              {button.label}
            </button>
          ))}
        </div>
      )}

      {/* Show current topic below action buttons */}
      {currentTopic && (
        <div className="text-center">
          <p className="text-sm text-gray-600">
            Currently discussing:{" "}
            <span className="font-semibold text-[#063970]">{currentTopic}</span>
          </p>
        </div>
      )}
    </div>
  );
};

export { ActionButtons };
