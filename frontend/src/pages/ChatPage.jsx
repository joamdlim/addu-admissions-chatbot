import React, { useState } from "react";
import { PromptArea } from "../components/PromptArea";

const mockFaqs = [
  "How much is the tuition for the course BSIT?",
  "What are the requirements needed for enrollment?",
  "Where is the admission office located in?",
  "Contact information of the Admissions",
];

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleFaqClick = (faq) => {
    setQuery(faq);
  };

  const handleSend = async (q) => {
    if (!q.trim() || isLoading) return;

    const userMsg = { role: "user", content: q };
    const botMsg = { role: "bot", content: "" };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    setQuery("");
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

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const jsonStr = line.slice(6);
              if (jsonStr.trim()) {
                const data = JSON.parse(jsonStr);

                if (data.chunk) {
                  currentResponse += data.chunk;
                  setMessages((prev) => {
                    const updated = [...prev];
                    // Update the last message (bot placeholder) with streamed content
                    const lastIndex = updated.length - 1;
                    if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
                      updated[lastIndex] = {
                        ...updated[lastIndex],
                        content: currentResponse,
                      };
                    }
                    return updated;
                  });
                } else if (data.done) {
                  setIsLoading(false);
                  return;
                } else if (data.error) {
                  setMessages((prev) => {
                    const updated = [...prev];
                    const lastIndex = updated.length - 1;
                    if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
                      updated[lastIndex] = {
                        ...updated[lastIndex],
                        content: `Error: ${data.error}`,
                      };
                    }
                    return updated;
                  });
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
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
          updated[lastIndex] = {
            ...updated[lastIndex],
            content: "Error connecting to chatbot backend.",
          };
        }
        return updated;
      });
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full h-full bg-gray-50 flex flex-col items-center">
      {/* Messages */}
      <div className="flex-1 w-[800px] max-w-[800px] py-8 overflow-y-auto space-y-4 min-h-0">
        {messages.map((m, idx) => {
          const isUser = m.role === "user";
          return (
            <div
              key={idx}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[75%] rounded-lg p-4 text-base leading-relaxed ${
                  isUser
                    ? "bg-[#063970] text-white rounded-br-sm"
                    : "bg-white text-gray-900 border border-gray-200 rounded-bl-sm"
                }`}
              >
                {m.content}
              </div>
            </div>
          );
        })}
      </div>

      {/* FAQs and input - fixed at bottom */}
      <div className="w-[800px] max-w-[800px] pb-6 bg-gray-50 flex-shrink-0">
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
  );
};

export { ChatPage };
