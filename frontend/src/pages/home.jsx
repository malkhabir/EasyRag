import { useState } from "react";
import QueryForm from "../comps/QueryForm";
import UploadWindow from "../comps/UploadWindow";
import "./home.css";

const Home = () => {
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [viewingFileIndex, setViewingFileIndex] = useState(0); // Index in selectedFiles array
  const [viewingFile, setViewingFile] = useState(null); // Currently viewing file name

  // Called when user clicks a file to view it
  const handleFileClick = (fileName) => {
    // Always add file to selection when clicking to view
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      newSet.add(fileName);
      return newSet;
    });
    setViewingFile(fileName);
  };

  // Navigate between selected files
  const handleNextFile = () => {
    const filesArray = Array.from(selectedFiles);
    if (filesArray.length === 0) return;
    const nextIdx = (viewingFileIndex + 1) % filesArray.length;
    setViewingFileIndex(nextIdx);
    setViewingFile(filesArray[nextIdx]);
  };

  const handlePrevFile = () => {
    const filesArray = Array.from(selectedFiles);
    if (filesArray.length === 0) return;
    const prevIdx = (viewingFileIndex - 1 + filesArray.length) % filesArray.length;
    setViewingFileIndex(prevIdx);
    setViewingFile(filesArray[prevIdx]);
  };

  return (
    <div style={{ margin: "2rem auto", fontFamily: "sans-serif" }}>
      <div className="header">
        {/* <h1>TheIntern</h1> */}
      </div>
      <div className="main-view">
        <QueryForm 
          selectedFiles={selectedFiles} 
          setSelectedFiles={setSelectedFiles}
          viewingFile={viewingFile}
          setViewingFile={setViewingFile}
          viewingFileIndex={viewingFileIndex}
          onNextFile={handleNextFile}
          onPrevFile={handlePrevFile}
        />
        <UploadWindow 
          selectedFiles={selectedFiles} 
          setSelectedFiles={setSelectedFiles}
          onFileClick={handleFileClick}
        />
      </div>
    </div>
  );
};

export default Home;
