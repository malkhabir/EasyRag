import { useEffect, useRef, useState } from "react";
import "./ChatWindow.css";

const ChatWindow = ({
  messages,
  setMessages,
  loading,
  handleSubmit,
  onSourceClick,
}) => {
  const [userInput, setUserInput] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = () => {
    if (!userInput.trim()) return;
    const userMsg = { text: userInput.trim(), isBot: false };
    setMessages((msgs) => [...msgs, userMsg]);
    handleSubmit(userInput.trim());
    setUserInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Format source display based on content type
  const formatSourceDisplay = (src) => {
    const isTable = src.content_type === "table";
    
    if (isTable) {
      return (
        <span className="source-content">
          <span className="source-icon icon-table"></span>
          <span className="source-file">{src.file_name}</span>
          <span className="source-separator">—</span>
          <span className="source-detail">
            Page {src.page}
            {src.table_region && `, ${src.table_region}`}
            {src.table_index !== undefined && ` (Table #${src.table_index + 1})`}
          </span>
          {src.is_sub_table && <span className="sub-table-badge">Sub-table</span>}
        </span>
      );
    }
    
    return (
      <span className="source-content">
        <span className="source-icon icon-text"></span>
        <span className="source-file">{src.file_name}</span>
        <span className="source-separator">—</span>
        <span className="source-detail">
          Page {src.page}, Lines {src.start_line}-{src.end_line}
        </span>
      </span>
    );
  };

  return (
    <div className="chatbox-container">
      <div className="chatbox-messages">
        {messages?.map((m, i) => (
          <div key={i} className={`chat-message ${m.isBot ? "bot" : "user"}`}>
            <div className="chat-bubble">{m.text}</div>

            {/* Show clickable sources with enhanced display */}
            {m.isBot && m.sources && m.sources.length > 0 && (
              <div className="chat-sources">
                <div className="sources-header">
                  <span className="source-icon icon-sources"></span>
                  <strong>Sources ({m.sources.length})</strong>
                  <span className="sources-hint">Click to highlight in document</span>
                </div>
                <ul className="sources-list">
                  {m.sources
                    .slice()
                    .sort((a, b) => {
                      // Sort by: page, then table vs text, then position
                      if (a.page !== b.page) return (a.page || 0) - (b.page || 0);
                      // Tables first
                      if (a.content_type !== b.content_type) {
                        return a.content_type === "table" ? -1 : 1;
                      }
                      return (a.start_line || 0) - (b.start_line || 0);
                    })
                    .map((src, idx) => (
                      <li key={idx} className={`source-item ${src.content_type || "text"}`}>
                        <button
                          className="source-link"
                          onClick={() => onSourceClick(src)}
                          title={`Click to view ${src.content_type === "table" ? "table" : "text"} in document`}
                        >
                          {formatSourceDisplay(src)}
                        </button>
                      </li>
                    ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-loading">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <em>The Intern is analyzing documents...</em>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chatbox-input">
        <textarea
          rows={2}
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask the Intern about your documents..."
        />
        <button onClick={sendMessage} disabled={loading || !userInput.trim()}>
          <img src="send-icon.png" className="icon-lg" alt="" />
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;
