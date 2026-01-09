import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import "./PDFViewer.css";

// Import worker locally using Vite's ?url loader
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

// Set pdfjs workerSrc to the local file URL
pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

/**
 * Enhanced PDF Viewer with support for:
 * - Line-based highlighting (text content)
 * - Region/box-based highlighting (tables)
 * - Quadrant visualization
 */
const PDFViewer = ({ 
    file, 
    pageNumber, 
    startLine, 
    endLine,
    // Table/region highlighting props
    tableBox,           // [x0, y0, x1, y1] - table bounding box (in image coordinates)
    tableRegion,        // "top", "middle", "bottom-left", etc.
    tableQuadrant,      // "top-left", "top-right", etc.
    contentType,        // "table" or "text"
    imageWidth,         // Original image width (for coordinate scaling)
    imageHeight,        // Original image height (for coordinate scaling)
    highlightMode = "auto" // "line", "region", or "auto"
}) => {
    const [numPages, setNumPages] = useState(null);
    const [highlightRect, setHighlightRect] = useState(null);
    const [regionOverlay, setRegionOverlay] = useState(null);
    const [pdfPageDimensions, setPdfPageDimensions] = useState({ width: 0, height: 0 });
    const containerRef = useRef(null);
    const pageRef = useRef(null);
    
    // Rendered width for the PDF
    const RENDER_WIDTH = 900;

    // Always use region/box highlight style for both text and tables
    // No highlight if no content type specified (just viewing PDF)
    const effectiveMode = highlightMode === "auto" 
        ? (contentType ? "region" : "none")
        : highlightMode;

    // Handle page load to get PDF dimensions
    const onPageLoadSuccess = useCallback((page) => {
        const { width, height } = page;
        setPdfPageDimensions({ width, height });
    }, []);

    // Line-based highlighting (for text content)
    useEffect(() => {
        if (effectiveMode !== "line" || !startLine || !endLine) {
            if (effectiveMode === "line") setHighlightRect(null);
            return;
        }

        const textLayer = containerRef.current?.querySelector(".react-pdf__Page__textContent");
        if (!textLayer) {
            setHighlightRect(null);
            return;
        }

        const spans = Array.from(textLayer.querySelectorAll("span"));
        if (spans.length === 0) {
            setHighlightRect(null);
            return;
        }

        const containerRect = textLayer.getBoundingClientRect();
        const tolerance = 4;

        // Group spans into lines
        const lines = [];
        spans.forEach(span => {
            const rect = span.getBoundingClientRect();
            const relTop = rect.top - containerRect.top;

            let lineFound = false;
            for (let line of lines) {
                if (Math.abs(line.top - relTop) < tolerance) {
                    line.spans.push(span);
                    line.top = (line.top * (line.spans.length - 1) + relTop) / line.spans.length;
                    lineFound = true;
                    break;
                }
            }
            if (!lineFound) {
                lines.push({ top: relTop, spans: [span] });
            }
        });

        lines.sort((a, b) => a.top - b.top);

        const startIdx = Math.max(0, startLine - 1);
        const endIdx = Math.min(lines.length - 1, endLine - 1);

        if (startIdx > endIdx) {
            setHighlightRect(null);
            return;
        }

        const selectedLines = lines.slice(startIdx, endIdx + 1);

        let top = Infinity;
        let bottom = -Infinity;

        selectedLines.forEach(line => {
            line.spans.forEach(span => {
                const rect = span.getBoundingClientRect();
                const relTop = rect.top - containerRect.top;
                const relBottom = rect.bottom - containerRect.top;
                if (relTop < top) top = relTop;
                if (relBottom > bottom) bottom = relBottom;
            });
        });

        setHighlightRect({
            left: 0,
            top,
            width: containerRect.width,
            height: bottom - top,
            type: "line"
        });

    }, [startLine, endLine, pageNumber, file, effectiveMode]);

    // Region/box-based highlighting (for tables)
    useEffect(() => {
        if (effectiveMode !== "region" || !tableBox || tableBox.length !== 4) {
            if (effectiveMode === "region") setRegionOverlay(null);
            return;
        }

        // tableBox is [x0, y0, x1, y1] in image pixel coordinates (from rasterization)
        // We need to scale from image coordinates to rendered PDF coordinates
        const [x0, y0, x1, y1] = tableBox;
        
        // Calculate scale factor from image to rendered PDF
        // Image was rasterized at some DPI, PDF is rendered at RENDER_WIDTH
        // We need: rendered_coord = image_coord * (RENDER_WIDTH / imageWidth)
        const scaleX = imageWidth ? RENDER_WIDTH / imageWidth : 1;
        const scaleY = imageHeight ? (RENDER_WIDTH / imageWidth) : 1; // Maintain aspect ratio
        
        setRegionOverlay({
            left: x0 * scaleX,
            top: y0 * scaleY,
            width: (x1 - x0) * scaleX,
            height: (y1 - y0) * scaleY,
            region: tableRegion,
            quadrant: tableQuadrant,
            type: "table"
        });

    }, [tableBox, imageWidth, imageHeight, tableRegion, tableQuadrant, effectiveMode]);

    // Get region indicator position
    const getRegionIndicator = () => {
        if (!tableRegion && !tableQuadrant) return null;
        
        const region = tableQuadrant || tableRegion;
        const positions = {
            "top-left": { top: "5%", left: "5%" },
            "top-right": { top: "5%", right: "5%" },
            "bottom-left": { bottom: "5%", left: "5%" },
            "bottom-right": { bottom: "5%", right: "5%" },
            "center": { top: "50%", left: "50%", transform: "translate(-50%, -50%)" },
            "top": { top: "5%", left: "50%", transform: "translateX(-50%)" },
            "middle": { top: "50%", left: "50%", transform: "translate(-50%, -50%)" },
            "bottom": { bottom: "5%", left: "50%", transform: "translateX(-50%)" },
        };
        
        return positions[region] || positions["center"];
    };

    return (
        <div ref={containerRef} className="pdf-viewer-container">
            <div className="pdf-viewer-header">
                <span className="pdf-page-info">Page {pageNumber} of {numPages || "?"}</span>
                {/* {contentType && (
                    <span className={`content-type-badge ${contentType}`}>
                        <span className={`badge-icon icon-${contentType}`}></span>
                        {contentType === "table" ? "Table" : "Text"}
                    </span>
                )} */}
                {tableRegion && (
                    <span className="region-badge">
                        <span className="badge-icon icon-location"></span>
                        {tableRegion}
                    </span>
                )}
            </div>

            <div className="pdf-document-wrapper" ref={pageRef}>
                <Document
                    file={file}
                    onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                >
                    <Page
                        pageNumber={pageNumber}
                        width={RENDER_WIDTH}
                        renderAnnotationLayer={false}
                        renderTextLayer={true}
                        onLoadSuccess={onPageLoadSuccess}
                    />
                </Document>

                {/* Line-based highlight (text) */}
                {highlightRect && highlightRect.type === "line" && (
                    <div
                        className="highlight-overlay line-highlight"
                        style={{
                            left: highlightRect.left,
                            top: highlightRect.top,
                            width: highlightRect.width,
                            height: highlightRect.height,
                        }}
                    />
                )}

                {/* Region-based highlight (table) */}
                {regionOverlay && regionOverlay.type === "table" && (
                    <>
                        <div
                            className="highlight-overlay table-highlight"
                            style={{
                                left: regionOverlay.left,
                                top: regionOverlay.top,
                                width: regionOverlay.width,
                                height: regionOverlay.height,
                            }}
                        >
                            <span className="table-label">
                                Table ({regionOverlay.region || regionOverlay.quadrant})
                            </span>
                        </div>
                        
                        {/* Region indicator */}
                        {tableQuadrant && (
                            <div 
                                className="region-indicator"
                                style={getRegionIndicator()}
                            >
                                {tableQuadrant}
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* Table metadata panel */}
            {contentType === "table" && tableBox && (
                <div className="table-metadata-panel">
                    <h4><span className="badge-icon icon-table"></span> Table Information</h4>
                    <div className="metadata-grid">
                        <div className="metadata-item">
                            <label>Region:</label>
                            <span>{tableRegion || "N/A"}</span>
                        </div>
                        <div className="metadata-item">
                            <label>Quadrant:</label>
                            <span>{tableQuadrant || "N/A"}</span>
                        </div>
                        <div className="metadata-item">
                            <label>Position:</label>
                            <span>({tableBox[0]}, {tableBox[1]})</span>
                        </div>
                        <div className="metadata-item">
                            <label>Size:</label>
                            <span>{tableBox[2] - tableBox[0]} × {tableBox[3] - tableBox[1]}px</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PDFViewer;
