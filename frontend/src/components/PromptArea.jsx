import React from "react";

const PromptArea = ({
  faqs,
  onFaqClick,
  onSend,
  query,
  setQuery,
  disabled = false,
}) => (
  <div className="w-full flex flex-col items-center pb-8">
    {/* FAQs */}
    <div className="flex flex-row justify-end w-2/3 mb-4 space-x-2">
      {faqs.map((faq, idx) => (
        <button
          key={idx}
          className={`bg-white text-black rounded px-4 py-2 border border-gray-300 ${
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
      className="w-2/3 flex"
      onSubmit={(e) => {
        e.preventDefault();
        if (!disabled) onSend(query);
      }}
    >
      <input
        className={`text-black flex-1 rounded-l px-4 py-2 border border-gray-300 ${
          disabled ? "bg-gray-100" : ""
        }`}
        placeholder="Ask anything about Ateneo de Davao's admissions"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={disabled}
      />
      <button
        type="submit"
        className={`px-6 rounded-r text-white ${
          disabled
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-[#063970] hover:bg-[#052a5a]"
        }`}
        disabled={disabled}
      >
        {disabled ? "Sending..." : "Send"}
      </button>
    </form>
    <img
      src="/src/assets/addu.jpg"
      alt="ADDU Watermark"
      className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-50 w-96 h-96 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  </div>
);

export { PromptArea };
