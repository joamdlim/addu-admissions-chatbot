import React from "react";

const PromptArea = ({
  faqs, onFaqClick, onSend, query, setQuery
}) => (
  <div className="w-full flex flex-col items-center pb-8">
    {/* FAQs */}
    <div className="flex flex-row justify-end w-2/3 mb-4 space-x-2">
      {faqs.map((faq, idx) => (
        <button
          key={idx}
          className="bg-white text-[#063970] rounded px-4 py-2 shadow hover:bg-gray-100"
          onClick={() => onFaqClick(faq)}
        >
          {faq}
        </button>
      ))}
    </div>
    {/* Query Input */}
    <form
      className="w-2/3 flex"
      onSubmit={e => { e.preventDefault(); onSend(query); }}
    >
      <input
        className="flex-1 rounded-l px-4 py-2 border border-gray-300"
        placeholder="Ask anything about Ateneo de Davao's admissions"
        value={query}
        onChange={e => setQuery(e.target.value)}
      />
      <button
        type="submit"
        className="bg-[#063970] text-white px-6 rounded-r"
      >
        Send
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