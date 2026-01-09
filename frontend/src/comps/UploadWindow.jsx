import { useCallback, useEffect, useRef, useState } from "react";
import Modal from "react-modal";
import FileDropzone from "./FileDropZone";
import UploadedFilesList from "./UploadedFilesList";

import "./UploadWindow.css";

Modal.setAppElement("#root");

const UploadWindow = ({ selectedFiles, setSelectedFiles, onFileClick }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [error, setError] = useState("");
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);
  const [modalMessage, setModalMessage] = useState(null);

  const fetchUploadedFiles = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:8080/api/v1/files");
      if (!res.ok) throw new Error("Failed to fetch files");
      const data = await res.json();
      setUploadedFiles(data.files || []);
    } catch (err) {
      console.error(err);
    }
  }, []);

  useEffect(() => {
    fetchUploadedFiles();
  }, [fetchUploadedFiles]);

  async function handleUpload(fileToUpload) {
    if (!fileToUpload) return;

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", fileToUpload);

      const res = await fetch("http://localhost:8080/api/v1/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);

      await res.json();

      if (fileInputRef.current) fileInputRef.current.value = null;

      fetchUploadedFiles();
      setShowSuccessModal(true);
      setModalMessage(<p>File uploaded successfully</p>);
      setTimeout(() => setShowSuccessModal(false), 1000);
      setSelectedFile(null);
    } catch (err) {
      setShowSuccessModal(true);
      setModalMessage(<p>{err.message}</p>);
      setTimeout(() => setShowSuccessModal(false), 1000);
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete() {
    if (!selectedFiles.size) return;

    const confirmed = window.confirm("Are you sure you want to delete the selected files?");
    if (!confirmed) return;

    setUploading(true);
    setError("");

    try {
      // Assuming selectedFiles is a Set or array of file names/ids
      const deletePromises = Array.from(selectedFiles).map((file) =>
        fetch(`http://localhost:8080/api/v1/files/${file}`, {
          method: "DELETE",
        })
      );

      const results = await Promise.allSettled(deletePromises);

      // Check if any delete failed
      const failed = results.find(
        (result) => result.status === "fulfilled" && !result.value.ok
      );

      if (failed) {
        const res = failed.value;
        throw new Error(`Delete failed: ${res.statusText}`);
      }
      setModalMessage(<p>Files deleted successfully</p>);
      setShowSuccessModal(true);
      setTimeout(() => {
        setShowSuccessModal(false);
      }, 1000);

      setSelectedFiles(new Set());
      fetchUploadedFiles();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  const toggleFileSelection = (filename) => {
    setSelectedFiles((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(filename)) newSet.delete(filename);
      else newSet.add(filename);
      return newSet;
    });
  };

  const filteredFiles = uploadedFiles.filter((f) =>
    f.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="upload-window-container">
      <div className="my-docs" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <img src="library.png" className="icon-lg" alt="Library Icon" />
        <h2 style={{ fontFamily: "'SF Mono', 'Fira Code', monospace" }}>Docs</h2>
      </div>
      <input
        type="search"
        placeholder="Search files..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="file-search-input"
      />

      <UploadedFilesList
        files={filteredFiles}
        selectedFiles={selectedFiles}
        toggleFileSelection={toggleFileSelection}
        onFileClick={onFileClick}
      />

      <FileDropzone
        selectedFile={selectedFile}
        onFileSelect={setSelectedFile}
      />

      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleUpload(selectedFile);
        }}
        className="upload-form"
      >
        <input
          type="file"
          onChange={(e) => setSelectedFile(e.target.files[0])}
          className="file-input"
          ref={fileInputRef}
        />

        <button
          type="submit"
          disabled={uploading || !selectedFile}
          className={`upload-button ${uploading || !selectedFile ? "disabled" : ""}`}
        >
          <span className="button-content">
            {uploading ? "Uploading..." : "Upload"}
            <img src="upload-icon.png" className="icon-lg" alt="" />
          </span>
        </button>

        <button
          type="button"
          onClick={() => setSelectedFile(null)}
          disabled={uploading || !selectedFile}
          className="clear-button"
        >
          Clear
        </button>
        <button
          type="button"
          onClick={handleDelete}
          disabled={!selectedFiles.size}
          className="delete-button"
        >
          <span className="button-content">
            <img src="delete-icon.png" className="icon-lg" alt="" />
          </span>

        </button>
      </form>

      <Modal
        isOpen={showSuccessModal}
        onRequestClose={() => setShowSuccessModal(false)}
        contentLabel="Upload Success"
        className="modal-content"
        overlayClassName="modal-overlay"
        shouldCloseOnOverlayClick={true}
        shouldCloseOnEsc={true}
      >
        {modalMessage}
      </Modal>
    </div>
  );
};

export default UploadWindow;
