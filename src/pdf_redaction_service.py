"""
PDF Redaction Service

Provides in-place redaction of sensitive information in PDFs while preserving
the original document layout, formatting, and structure. Uses PyMuPDF (fitz)
to extract text with positions and apply visual redaction annotations.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
import tempfile
import os
from dataclasses import dataclass


@dataclass
class PDFTextSpan:
    """Represents a text span in a PDF with position and content"""

    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    font_name: str


@dataclass
class PDFRedactionRect:
    """Represents a redaction rectangle to be applied to a PDF"""

    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    original_text: str
    redaction_type: str


class PDFRedactionService:
    """Service for redacting sensitive information in PDFs while preserving layout"""

    def __init__(self):
        self.redaction_color = (0, 0, 0)  # Black redaction
        self.redaction_text_color = (1, 1, 1)  # White text on black background

    def extract_text_with_positions(self, pdf_path: str) -> List[PDFTextSpan]:
        """
        Extract text from PDF with exact positions and formatting info

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PDFTextSpan objects with text content and positions
        """
        text_spans = []

        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Get text with detailed formatting information
                text_dict = page.get_text("dict")

                # Process each block
                for block in text_dict["blocks"]:
                    if "lines" not in block:  # Skip image blocks
                        continue

                    # Process each line
                    for line in block["lines"]:
                        # Process each span (text with same formatting)
                        for span in line["spans"]:
                            text = span["text"].strip()

                            if not text:  # Skip empty text
                                continue

                            # Create text span object
                            text_span = PDFTextSpan(
                                text=text,
                                page_num=page_num,
                                bbox=tuple(span["bbox"]),
                                font_size=span["size"],
                                font_name=span["font"],
                            )

                            text_spans.append(text_span)

            doc.close()
            return text_spans

        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return []

    def detect_sensitive_text(
        self, text_spans: List[PDFTextSpan], classifier_service
    ) -> List[PDFRedactionRect]:
        """
        Detect sensitive information in extracted text and create redaction rectangles

        Args:
            text_spans: List of PDFTextSpan objects from PDF
            classifier_service: The sensitivity classifier service

        Returns:
            List of PDFRedactionRect objects for sensitive text
        """
        redaction_rects = []

        # Combine all text for batch processing
        full_text = " ".join([span.text for span in text_spans])

        if not full_text.strip():
            return redaction_rects

        try:
            # Use the existing classifier service to detect sensitive information
            redaction_result = classifier_service.redact_sensitive_info(full_text)

            # Map detected sensitive text back to text spans
            for detection in redaction_result.detections:
                sensitive_text = detection["original"]
                detection_type = detection["type"]

                # Find text spans that contain this sensitive text
                for text_span in text_spans:
                    if self._text_contains_sensitive(text_span.text, sensitive_text):
                        redaction_rect = PDFRedactionRect(
                            page_num=text_span.page_num,
                            bbox=text_span.bbox,
                            original_text=sensitive_text,
                            redaction_type=detection_type,
                        )
                        redaction_rects.append(redaction_rect)

        except Exception as e:
            print(f"Error detecting sensitive text in PDF: {str(e)}")

        return redaction_rects

    def _text_contains_sensitive(self, span_text: str, sensitive_text: str) -> bool:
        """
        Check if a text span contains sensitive text (case-insensitive, partial match)

        Args:
            span_text: Text from the PDF text span
            sensitive_text: Sensitive text to look for

        Returns:
            True if the text span contains the sensitive text
        """
        return sensitive_text.lower() in span_text.lower()

    def apply_redactions(
        self, pdf_path: str, redaction_rects: List[PDFRedactionRect], output_path: str
    ) -> bool:
        """
        Apply redaction rectangles to the PDF and save the result

        Args:
            pdf_path: Path to the original PDF
            redaction_rects: List of redaction rectangles to apply
            output_path: Path to save the redacted PDF

        Returns:
            True if successful, False otherwise
        """
        try:
            # Open the original PDF
            doc = fitz.open(pdf_path)

            # Group redactions by page for efficient processing
            redactions_by_page = {}
            for redaction_rect in redaction_rects:
                page_num = redaction_rect.page_num
                if page_num not in redactions_by_page:
                    redactions_by_page[page_num] = []
                redactions_by_page[page_num].append(redaction_rect)

            # Apply redactions page by page
            for page_num, page_redactions in redactions_by_page.items():
                if page_num >= len(doc):  # Safety check
                    continue

                page = doc[page_num]

                # Add redaction annotations for each sensitive area
                for redaction_rect in page_redactions:
                    # Create redaction rectangle
                    rect = fitz.Rect(redaction_rect.bbox)

                    # Add redaction annotation (this will be a black box when applied)
                    redact_annot = page.add_redact_annot(rect)

                    # Optional: Add redaction text
                    redact_annot.set_info(
                        content="[REDACTED]",
                        title=f"Redacted {redaction_rect.redaction_type}",
                    )

                    # Set redaction appearance
                    redact_annot.set_colors(
                        stroke=self.redaction_color, fill=self.redaction_color
                    )
                    redact_annot.update()

                # Apply all redactions on this page
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            # Save the redacted PDF
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            return True

        except Exception as e:
            print(f"Error applying redactions to PDF: {str(e)}")
            return False

    def redact_pdf(self, pdf_path: str, classifier_service, output_path: str) -> Dict:
        """
        Complete PDF redaction pipeline: extract text, detect sensitive info, apply redactions

        Args:
            pdf_path: Path to the input PDF
            classifier_service: The sensitivity classifier service
            output_path: Path to save the redacted PDF

        Returns:
            Dictionary with redaction results and metadata
        """
        try:
            # Step 1: Extract text with positions
            text_spans = self.extract_text_with_positions(pdf_path)

            if not text_spans:
                return {
                    "success": False,
                    "error": "No text detected in PDF",
                    "text_spans_found": 0,
                    "redactions_applied": 0,
                }

            # Step 2: Detect sensitive information
            redaction_rects = self.detect_sensitive_text(text_spans, classifier_service)

            # Step 3: Apply redactions to PDF
            success = self.apply_redactions(pdf_path, redaction_rects, output_path)

            if not success:
                return {
                    "success": False,
                    "error": "Failed to apply redactions to PDF",
                    "text_spans_found": len(text_spans),
                    "redactions_applied": 0,
                }

            # Collect redaction summary
            redaction_summary = {}
            for redaction_rect in redaction_rects:
                redaction_type = redaction_rect.redaction_type
                if redaction_type not in redaction_summary:
                    redaction_summary[redaction_type] = 0
                redaction_summary[redaction_type] += 1

            return {
                "success": True,
                "text_spans_found": len(text_spans),
                "redactions_applied": len(redaction_rects),
                "redaction_summary": redaction_summary,
                "redacted_pdf_path": output_path,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"PDF redaction failed: {str(e)}",
                "text_spans_found": 0,
                "redactions_applied": 0,
            }


# Global instance for singleton pattern
_pdf_redaction_service_instance = None


def get_pdf_redaction_service() -> PDFRedactionService:
    """Get singleton instance of PDFRedactionService"""
    global _pdf_redaction_service_instance
    if _pdf_redaction_service_instance is None:
        _pdf_redaction_service_instance = PDFRedactionService()
    return _pdf_redaction_service_instance
