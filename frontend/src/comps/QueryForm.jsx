import { useEffect, useState } from "react";
import ChatWindow from "./chat/chatwindow";
import PDFViewer from "./chat/PDFViewer";

const QueryForm = ({ 
  selectedFiles, 
  setSelectedFiles, 
  viewingFile, 
  setViewingFile,
  viewingFileIndex,
  onNextFile,
  onPrevFile 
}) => {
  const [loading, setLoading] = useState(false);
  const [selectedPage, setSelectedPage] = useState(1);
  const [selectedSource, setSelectedSource] = useState(null); // Can be null for plain PDF viewing
  const [messages, setMessages] = useState([
    {
      text: "Hey, I'm Peter, the Intern. what do you want to know about your docs?",
      isBot: true,
    },
  ]);

  const handleSubmit = async (userQuery) => {
    if (!userQuery.trim()) return;
    setLoading(true);

    try {
      // Build files query params - FastAPI expects multiple 'files' params for List[str]
      const filesArray = Array.from(selectedFiles);
      const filesParams = filesArray.length > 0 
        ? filesArray.map(f => `files=${encodeURIComponent(f)}`).join("&")
        : "";
      const url = filesParams 
        ? `http://localhost:8080/api/v1/query?q=${encodeURIComponent(userQuery)}&${filesParams}`
        : `http://localhost:8080/api/v1/query?q=${encodeURIComponent(userQuery)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Server error: ${res.statusText}`);
      const data = await res.json();

      setMessages((msgs) => [
        ...msgs,
        {
          text: data.answer || "No answer returned.",
          isBot: true,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setMessages((msgs) => [...msgs, { text: `Error: ${err.message}`, isBot: true }]);
    } finally {
      setLoading(false);
    }
  };

  // Handle source click - set all relevant metadata for highlighting
  const handleSourceClick = (source) => {
    setViewingFile(source.file_name);
    setSelectedPage(source.page || 1);
    setSelectedSource(source);
  };

  // Parse table box from string if needed
  const parseTableBox = (boxData) => {
    if (!boxData) return null;
    if (Array.isArray(boxData)) return boxData;
    if (typeof boxData === "string") {
      try {
        // Handle "[x0, y0, x1, y1]" format
        const parsed = JSON.parse(boxData.replace(/'/g, '"'));
        return Array.isArray(parsed) ? parsed : null;
      } catch {
        return null;
      }
    }
    return null;
  };

  // When viewingFile changes externally (from UploadWindow), reset source
  useEffect(() => {
    if (viewingFile) {
      setSelectedPage(1);
      setSelectedSource(null);
    }
  }, [viewingFile]);

  return (
    <>
      <ChatWindow
        messages={messages}
        setMessages={setMessages}
        loading={loading}
        handleSubmit={handleSubmit}
        onSourceClick={handleSourceClick}
      />

      {viewingFile && (
        <div className="pdf-viewer-wrapper">
          {/* File navigation for multiple selected files */}
          {selectedFiles.size > 1 && (
            <div className="file-nav-controls">
              <button onClick={onPrevFile} className="file-nav-btn">◀ Prev</button>
              <span className="file-nav-info">
                File {viewingFileIndex + 1} of {selectedFiles.size}: {viewingFile}
              </span>
              <button onClick={onNextFile} className="file-nav-btn">Next ▶</button>
            </div>
          )}
          <PDFViewer
            file={`http://localhost:8080/api/v1/files/${viewingFile}`}
            pageNumber={selectedPage}
            // Only pass highlight props if we have a source selected
            startLine={selectedSource?.start_line}
            endLine={selectedSource?.end_line}
            tableBox={selectedSource ? parseTableBox(selectedSource.table_box) : null}
            tableRegion={selectedSource?.table_region}
            tableQuadrant={selectedSource?.table_quadrant}
            contentType={selectedSource?.content_type}
            imageWidth={selectedSource?.image_width}
            imageHeight={selectedSource?.image_height}
          />
        </div>
      )}
    </>
  );
};

export default QueryForm;
