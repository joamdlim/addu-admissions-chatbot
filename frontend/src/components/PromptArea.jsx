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
    <div className="flex flex-row justify-start w-full mb-4 gap-2">
      {faqs.map((faq, idx) => (
        <button
          key={idx}
          className={`bg-white text-gray-900 rounded px-3 py-2 border border-gray-300 text-sm ${
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
      className="w-full flex"
      onSubmit={(e) => {
        e.preventDefault();
        if (!disabled) onSend(query);
      }}
    >
      <input
        className={`text-gray-900 flex-1 rounded-l px-4 py-3 border border-gray-300 text-base ${
          disabled ? "bg-gray-100" : ""
        }`}
        placeholder="Ask anything about Ateneo de Davao's admissions"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={disabled}
      />
      <button
        type="submit"
        className={`px-6 rounded-r text-white text-base ${
          disabled ? "bg-gray-400 cursor-not-allowed" : "bg-[#063970] hover:bg-[#052a5a]"
        }`}
        disabled={disabled}
      >
        {disabled ? "Sending..." : "Send"}
      </button>
    </form>
  </div>
);

export { PromptArea };