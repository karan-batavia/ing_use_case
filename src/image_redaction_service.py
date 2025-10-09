"""
Image Redaction Service

Provides in-place redaction of sensitive information in images while preserving
the original layout and visual structure. Uses OCR to detect text positions
and draws redaction boxes over sensitive content.
"""

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import re
import os
import tempfile
from dataclasses import dataclass


@dataclass
class TextBox:
    """Represents a detected text box with position and content"""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float


@dataclass
class RedactionBox:
    """Represents a redaction to be applied"""

    left: int
    top: int
    width: int
    height: int
    original_text: str
    redaction_type: str


class ImageRedactionService:
    """Service for redacting sensitive information in images while preserving layout"""

    def __init__(self):
        self.redaction_color = (0, 0, 0)  # Black redaction boxes
        self.min_confidence = 30  # Minimum OCR confidence threshold

    def extract_text_with_positions(self, image_path: str) -> List[TextBox]:
        """
        Extract text from image with bounding box positions using OCR

        Args:
            image_path: Path to the image file

        Returns:
            List of TextBox objects with text content and positions
        """
        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Use pytesseract to get detailed text data with positions
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",  # Uniform block of text
            )

            text_boxes = []

            # Process OCR results
            for i in range(len(ocr_data["text"])):
                text = ocr_data["text"][i].strip()
                confidence = int(ocr_data["conf"][i])

                # Skip empty text or low confidence detections
                if not text or confidence < self.min_confidence:
                    continue

                text_box = TextBox(
                    text=text,
                    left=ocr_data["left"][i],
                    top=ocr_data["top"][i],
                    width=ocr_data["width"][i],
                    height=ocr_data["height"][i],
                    confidence=confidence,
                )

                text_boxes.append(text_box)

            return text_boxes

        except Exception as e:
            print(f"Error extracting text from image: {str(e)}")
            return []

    def detect_sensitive_text(
        self, text_boxes: List[TextBox], classifier_service
    ) -> List[RedactionBox]:
        """
        Detect sensitive information in extracted text and create redaction boxes

        Args:
            text_boxes: List of TextBox objects from OCR
            classifier_service: The sensitivity classifier service

        Returns:
            List of RedactionBox objects for sensitive text
        """
        redaction_boxes = []

        if not text_boxes:
            return redaction_boxes

        try:
            # Method 1: Check each text box individually for better accuracy
            for text_box in text_boxes:
                if not text_box.text.strip():
                    continue

                # Use the classifier service on individual text chunks
                redaction_result = classifier_service.redact_sensitive_info(
                    text_box.text
                )

                # If this text box contains sensitive information, redact the entire box
                if redaction_result.detections:
                    for detection in redaction_result.detections:
                        redaction_box = RedactionBox(
                            left=text_box.left,
                            top=text_box.top,
                            width=text_box.width,
                            height=text_box.height,
                            original_text=detection["original"],
                            redaction_type=detection["type"],
                        )
                        redaction_boxes.append(redaction_box)

            # Method 2: Also check combined text for cross-box patterns
            full_text = " ".join([box.text for box in text_boxes])
            if full_text.strip():
                redaction_result = classifier_service.redact_sensitive_info(full_text)

                # Map detected sensitive text back to text boxes using fuzzy matching
                for detection in redaction_result.detections:
                    sensitive_text = detection["original"]
                    detection_type = detection["type"]

                    # Find text boxes that contain this sensitive text
                    for text_box in text_boxes:
                        if self._text_contains_sensitive(text_box.text, sensitive_text):
                            # Check if we already have a redaction for this box
                            already_redacted = any(
                                rb.left == text_box.left and rb.top == text_box.top
                                for rb in redaction_boxes
                            )

                            if not already_redacted:
                                redaction_box = RedactionBox(
                                    left=text_box.left,
                                    top=text_box.top,
                                    width=text_box.width,
                                    height=text_box.height,
                                    original_text=sensitive_text,
                                    redaction_type=detection_type,
                                )
                                redaction_boxes.append(redaction_box)

        except Exception as e:
            print(f"Error detecting sensitive text: {str(e)}")

        return redaction_boxes

    def _text_contains_sensitive(self, text_box_text: str, sensitive_text: str) -> bool:
        """
        Check if a text box contains sensitive text (case-insensitive, flexible matching)

        Args:
            text_box_text: Text from the OCR text box
            sensitive_text: Sensitive text to look for

        Returns:
            True if the text box contains the sensitive text
        """
        # Clean and normalize both texts
        box_text_clean = text_box_text.lower().strip()
        sensitive_clean = sensitive_text.lower().strip()

        # Direct substring match
        if sensitive_clean in box_text_clean:
            return True

        # Check if the text box is a subset of the sensitive text (for split OCR)
        if box_text_clean in sensitive_clean:
            return True

        # Check for partial matches with common OCR errors
        # Remove common punctuation and whitespace
        import re

        box_alphanumeric = re.sub(r"[^\w]", "", box_text_clean)
        sensitive_alphanumeric = re.sub(r"[^\w]", "", sensitive_clean)

        if len(box_alphanumeric) > 2 and box_alphanumeric in sensitive_alphanumeric:
            return True
        if (
            len(sensitive_alphanumeric) > 2
            and sensitive_alphanumeric in box_alphanumeric
        ):
            return True

        return False

    def debug_detection(self, image_path: str, classifier_service) -> Dict:
        """
        Debug method to show what text is detected and what gets classified as sensitive

        Args:
            image_path: Path to the image file
            classifier_service: The sensitivity classifier service

        Returns:
            Dictionary with debug information
        """
        try:
            # Extract text with positions
            text_boxes = self.extract_text_with_positions(image_path)

            debug_info = {
                "text_boxes_found": len(text_boxes),
                "extracted_texts": [
                    {
                        "text": box.text,
                        "confidence": box.confidence,
                        "position": (box.left, box.top, box.width, box.height),
                    }
                    for box in text_boxes
                ],
                "individual_classifications": [],
                "combined_classification": None,
                "redaction_boxes": [],
            }

            # Check individual text boxes
            for i, text_box in enumerate(text_boxes):
                if text_box.text.strip():
                    try:
                        result = classifier_service.redact_sensitive_info(text_box.text)
                        debug_info["individual_classifications"].append(
                            {
                                "text_box_index": i,
                                "original_text": text_box.text,
                                "detections": result.detections,
                                "total_redacted": result.total_redacted,
                            }
                        )
                    except Exception as e:
                        debug_info["individual_classifications"].append(
                            {
                                "text_box_index": i,
                                "original_text": text_box.text,
                                "error": str(e),
                            }
                        )

            # Check combined text
            full_text = " ".join([box.text for box in text_boxes])
            if full_text.strip():
                try:
                    result = classifier_service.redact_sensitive_info(full_text)
                    debug_info["combined_classification"] = {
                        "full_text": full_text,
                        "detections": result.detections,
                        "total_redacted": result.total_redacted,
                    }
                except Exception as e:
                    debug_info["combined_classification"] = {
                        "full_text": full_text,
                        "error": str(e),
                    }

            # Get redaction boxes
            redaction_boxes = self.detect_sensitive_text(text_boxes, classifier_service)
            debug_info["redaction_boxes"] = [
                {
                    "position": (rb.left, rb.top, rb.width, rb.height),
                    "original_text": rb.original_text,
                    "redaction_type": rb.redaction_type,
                }
                for rb in redaction_boxes
            ]

            return debug_info

        except Exception as e:
            return {"error": f"Debug failed: {str(e)}"}

    def apply_redactions(
        self, image_path: str, redaction_boxes: List[RedactionBox], output_path: str
    ) -> bool:
        """
        Apply redaction boxes to the image and save the result

        Args:
            image_path: Path to the original image
            redaction_boxes: List of redaction boxes to apply
            output_path: Path to save the redacted image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the original image
            image = Image.open(image_path)

            # Convert to RGBA for better redaction handling
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            # Create a drawing context
            draw = ImageDraw.Draw(image)

            # Apply each redaction box
            for redaction_box in redaction_boxes:
                # Calculate redaction rectangle coordinates
                left = redaction_box.left
                top = redaction_box.top
                right = left + redaction_box.width
                bottom = top + redaction_box.height

                # Draw black rectangle over sensitive text
                draw.rectangle(
                    [left, top, right, bottom],
                    fill=self.redaction_color + (255,),  # Add alpha channel
                    outline=self.redaction_color + (255,),
                )

                # Optional: Add redaction label
                try:
                    # Use default font for redaction label
                    font_size = min(redaction_box.height - 4, 12)
                    if font_size > 6:
                        redaction_label = "[REDACTED]"

                        # Calculate text position (centered)
                        text_bbox = draw.textbbox((0, 0), redaction_label)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        text_x = left + (redaction_box.width - text_width) // 2
                        text_y = top + (redaction_box.height - text_height) // 2

                        # Draw white text on black background
                        draw.text(
                            (text_x, text_y), redaction_label, fill=(255, 255, 255, 255)
                        )
                except Exception:
                    # If text rendering fails, just use solid black box
                    pass

            # Convert back to RGB for saving
            if image.mode == "RGBA":
                # Create white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[-1]
                )  # Use alpha channel as mask
                image = background

            # Save the redacted image
            image.save(output_path, format="PNG", quality=95)
            return True

        except Exception as e:
            print(f"Error applying redactions: {str(e)}")
            return False

    def redact_image(
        self, image_path: str, classifier_service, output_path: str
    ) -> Dict:
        """
        Complete image redaction pipeline: extract text, detect sensitive info, apply redactions

        Args:
            image_path: Path to the input image
            classifier_service: The sensitivity classifier service
            output_path: Path to save the redacted image

        Returns:
            Dictionary with redaction results and metadata
        """
        try:
            # Step 1: Extract text with positions
            text_boxes = self.extract_text_with_positions(image_path)

            if not text_boxes:
                return {
                    "success": False,
                    "error": "No text detected in image",
                    "text_boxes_found": 0,
                    "redactions_applied": 0,
                }

            # Step 2: Detect sensitive information
            redaction_boxes = self.detect_sensitive_text(text_boxes, classifier_service)

            # Step 3: Apply redactions to image
            success = self.apply_redactions(image_path, redaction_boxes, output_path)

            if not success:
                return {
                    "success": False,
                    "error": "Failed to apply redactions to image",
                    "text_boxes_found": len(text_boxes),
                    "redactions_applied": 0,
                }

            # Collect redaction summary
            redaction_summary = {}
            for redaction_box in redaction_boxes:
                redaction_type = redaction_box.redaction_type
                if redaction_type not in redaction_summary:
                    redaction_summary[redaction_type] = 0
                redaction_summary[redaction_type] += 1

            return {
                "success": True,
                "text_boxes_found": len(text_boxes),
                "redactions_applied": len(redaction_boxes),
                "redaction_summary": redaction_summary,
                "redacted_image_path": output_path,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Image redaction failed: {str(e)}",
                "text_boxes_found": 0,
                "redactions_applied": 0,
            }


# Global instance for singleton pattern
_image_redaction_service_instance = None


def get_image_redaction_service() -> ImageRedactionService:
    """Get singleton instance of ImageRedactionService"""
    global _image_redaction_service_instance
    if _image_redaction_service_instance is None:
        _image_redaction_service_instance = ImageRedactionService()
    return _image_redaction_service_instance
