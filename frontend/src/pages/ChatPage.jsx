
import React, { useState } from "react";
import { Sidebar } from "../components/Sidebar";
import { PromptArea } from "../components/PromptArea";

const mockFaqs = [
  "How much is the tuition for the course BSIT?",
  "What are the requirements needed for enrollment?",
  "Where is the admission office located in?",
  "Contact information of the Admissions"
];

const ChatPage = () => {
  const [promptHistory, setPromptHistory] = useState([]);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(""); // Store chatbot response

  const handleFaqClick = (faq) => {
    setQuery(faq);
  };

  const handleSend = async (q) => {
    if (!q.trim()) return;
    setPromptHistory([q, ...promptHistory]);
    setQuery("");
    setResponse(""); // Clear previous response

    try {
      const res = await fetch("http://127.0.0.1:8000/chatbot/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: q }),
      });

      if (!res.body) {
        setResponse("No response body.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let result = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
        setResponse(result); // Update UI as new data arrives
      }
    } catch (err) {
      setResponse("Error connecting to chatbot backend.");
    }
  };

  return (
    <div className="flex h-screen">
      <Sidebar promptHistory={promptHistory} />
      <div className="flex-1 relative bg-gray-50 flex flex-col">
        {/* Chatbot response at the top */}
        <div className="flex-1 flex flex-col items-center justify-start pt-12">
          {response && (
            <div className="bg-white shadow rounded p-6 w-2/3 text-lg text-gray-800">
              {response}
            </div>
          )}
        </div>
        {/* FAQs and textbox at the bottom */}
        <div className="w-full">
          <PromptArea
            faqs={mockFaqs}
            onFaqClick={handleFaqClick}
            onSend={handleSend}
            query={query}
            setQuery={setQuery}
          />
        </div>
      </div>
    </div>
  );
};

export { ChatPage };