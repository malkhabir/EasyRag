import { useCallback, useRef, useState } from "react";
import "./FileDropZone.css";

const FileDropzone = ({ onFileSelect, selectedFile, setSelectedFile }) => {
  const [dragging, setDragging] = useState(false);

  // Counter to handle nested dragenter/dragleave events reliably
  const dragCounter = useRef(0);
  const inputRef = useRef(null);

  const onDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    // If files are present, show dragging state
    if (e.dataTransfer?.items?.length) setDragging(true);
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    // keep dragging true while over
    if (e.dataTransfer?.items?.length) setDragging(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = Math.max(0, dragCounter.current - 1);
    if (dragCounter.current === 0) setDragging(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setDragging(false);

    const dropped = e.dataTransfer?.files?.[0];
    if (dropped) {
      onFileSelect(dropped);
      // clear native file input value so re-selecting same file works
      if (inputRef.current) inputRef.current.value = "";
    }
  }, [onFileSelect]);

  const onClick = () => inputRef.current?.click();

  const onInputChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      onFileSelect(f);
      // keep input value clear to allow re-upload same file if needed
      e.target.value = "";
    }
  };

  const classes = `file-dropzone${dragging ? " dragging" : ""}${selectedFile ? " has-file" : ""}`;

  return (
    <div
      className={classes}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
      role="button"
      tabIndex={0}
      aria-label="File drop zone"
    >
      {/* two layered messages, same position; opacity toggles without layout change */}
      <span className="dropzone-text">
        {dragging ? "Drop here" : "Drag & drop, or click to select"}
      </span>

      {/* selected file area (renders above the messages) */}
      {selectedFile && (
        <div className="selected-file">
          <strong>Selected file:</strong> {selectedFile.name}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        onChange={onInputChange}
        style={{ display: "none" }}
      />
    </div>
  );
};

export default FileDropzone;
