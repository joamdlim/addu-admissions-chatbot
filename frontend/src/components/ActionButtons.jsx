import React from "react";

const ActionButtons = ({
  buttons,
  onAction,
  disabled = false,
  currentTopic = null,
  currentTopicLabel = null,
}) => {
  // Define school buttons for program curriculum topic
  const schoolButtons = [
    {
      id: "school_arts_sciences",
      label: "School of Arts & Sciences",
      type: "school",
    },
    {
      id: "school_business_governance",
      label: "School of Business & Governance",
      type: "school",
    },
    { id: "school_education", label: "School of Education", type: "school" },
    {
      id: "school_engineering_architecture",
      label: "School of Engineering & Architecture",
      type: "school",
    },
    { id: "school_nursing", label: "School of Nursing", type: "school" },
  ];

  // Separate action buttons from the passed buttons
  const actionButtons = buttons
    ? buttons.filter((btn) => btn.type !== "school" && btn.type !== "topic")
    : [];

  // Show school buttons only for program curriculum topic
  const showSchoolButtons = currentTopic === "programs_courses";

  if (!showSchoolButtons && (!actionButtons || actionButtons.length === 0))
    return null;

  return (
    <div className="w-full flex flex-col items-center space-y-2">
      {/* School buttons (for Program Curriculum only) - Responsive grid */}
      {showSchoolButtons && (
        <div className="w-full max-w-6xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
            {schoolButtons.map((button) => (
              <button
                key={button.id}
                onClick={() => !disabled && onAction(button.id)}
                disabled={disabled}
                className={`
                  px-4 py-3 rounded text-sm font-medium transition-all duration-200 text-center
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
      {(currentTopic || currentTopicLabel) && (
        <div className="text-center">
          <p className="text-sm text-gray-600">
            Currently discussing:{" "}
            <span className="font-semibold text-[#063970]">
              {currentTopicLabel || currentTopic}
            </span>
          </p>
        </div>
      )}
    </div>
  );
};

export { ActionButtons };
