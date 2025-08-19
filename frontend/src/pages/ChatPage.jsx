import React, { useState } from "react";
import { Sidebar } from "../components/Sidebar";
import { PromptArea } from "../components/PromptArea";

const mockFaqs = [
  "How much is the tuition for the course BSIT?",
  "What are the requirements needed for enrollment?",
  "Where is the admission office located in?",
  "Contact information of the Admissions",
];

const ChatPage = () => {
  const [promptHistory, setPromptHistory] = useState([]);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(""); // Store chatbot response
  const [isLoading, setIsLoading] = useState(false);

  const handleFaqClick = (faq) => {
    setQuery(faq);
  };

  const handleSend = async (q) => {
    if (!q.trim() || isLoading) return;

    setPromptHistory([q, ...promptHistory]);
    setQuery("");
    setResponse(""); // Clear previous response
    setIsLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chatbot/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ prompt: q }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      // Handle Server-Sent Events streaming
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the chunk
        buffer += decoder.decode(value, { stream: true });

        // Process complete lines from buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const jsonStr = line.slice(6); // Remove 'data: ' prefix
              if (jsonStr.trim()) {
                const data = JSON.parse(jsonStr);

                if (data.chunk) {
                  // Append chunk to current response
                  currentResponse += data.chunk;
                  setResponse(currentResponse);
                } else if (data.done) {
                  // Streaming complete
                  setIsLoading(false);
                  return;
                } else if (data.error) {
                  // Handle error
                  setResponse(`Error: ${data.error}`);
                  setIsLoading(false);
                  return;
                }
              }
            } catch (parseError) {
              console.error(
                "Error parsing SSE data:",
                parseError,
                "Line:",
                line
              );
            }
          }
        }
      }

      setIsLoading(false);
    } catch (err) {
      console.error("Streaming error:", err);
      setResponse("Error connecting to chatbot backend.");
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen">
      <Sidebar promptHistory={promptHistory} />
      <div className="flex-1 relative bg-gray-50 flex flex-col">
        {/* Chatbot response at the top */}
        <div className="flex-1 flex flex-col items-center justify-start pt-12">
          {(response || isLoading) && (
            <div className="bg-white shadow rounded p-6 w-2/3 text-lg text-gray-800">
              {response}
              {isLoading && <span className="animate-pulse">|</span>}
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
            disabled={isLoading}
          />
        </div>
      </div>
    </div>
  );
};

export { ChatPage };
