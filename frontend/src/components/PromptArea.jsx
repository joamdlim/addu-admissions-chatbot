import React from "react";

const PromptArea = ({
  faqs,
  onFaqClick,
  onSend,
  query,
  setQuery,
  disabled = false,
}) => (
  <div className="w-full flex flex-col items-center">
    {/* FAQs */}
    <div className="flex flex-wrap justify-start w-full mb-4 gap-2">
      {faqs.map((faq, idx) => (
        <button
          key={idx}
          className={`bg-white text-gray-900 rounded px-3 py-2 border border-gray-300 text-sm whitespace-nowrap ${
            disabled ? "opacity-50 cursor-not-allowed" : "hover:bg-gray-50"
          }`}
          onClick={() => !disabled && onFaqClick(faq)}
          disabled={disabled}
        >
          {faq}
        </button>
      ))}
    </div>
    {/* Query Input */}
    <form
      className="w-full flex shadow-lg rounded-lg overflow-hidden"
      onSubmit={(e) => {
        e.preventDefault();
        if (!disabled) onSend(query);
      }}
    >
      <input
        className={`text-gray-900 flex-1 px-4 py-3 border-0 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${
          disabled ? "bg-gray-100 text-gray-500" : "bg-white"
        }`}
        placeholder="Ask anything about Ateneo de Davao's admissions..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={disabled}
      />
      <button
        type="submit"
        className={`px-6 py-3 text-white text-base font-medium transition-all duration-200 ${
          disabled
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-[#063970] hover:bg-[#052a5a] hover:shadow-md active:scale-95"
        }`}
        disabled={disabled}
      >
        {disabled ? "Sending..." : "Send"}
      </button>
    </form>
  </div>
);

export { PromptArea };
