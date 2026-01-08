import "./UploadedFilesList.css";

const UploadedFilesList = ({ files, selectedFiles, toggleFileSelection, onFileClick }) => {
  const handleRowClick = (file, e) => {
    // If clicking the checkbox, only toggle selection (don't open viewer)
    if (e.target.type === "checkbox") return;
    // Click on row: open in viewer (home.jsx handles selection)
    if (onFileClick) onFileClick(file);
  };

  return (
    <ul className="uploaded-files-list">
      {files.length === 0 && <li className="no-files">No files uploaded</li>}
      {files.map((file) => (
        <li
          key={file}
          className={`file-item ${selectedFiles.has(file) ? "selected" : ""}`}
          onClick={(e) => handleRowClick(file, e)}
          title="Click to select and view"
        >
          <input
            type="checkbox"
            checked={selectedFiles.has(file)}
            onChange={() => toggleFileSelection(file)}
            onClick={(e) => e.stopPropagation()}
            className="file-checkbox"
          />
          <span>{file}</span>
        </li>
      ))}
    </ul>
  );
};

export default UploadedFilesList;
