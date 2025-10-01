from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import io
from typing import Optional


class PDFWriter:
    """Class to write text content to PDF format"""

    def __init__(self):
        self.page_width, self.page_height = A4
        self.margin = 72  # 1 inch margins

    def create_pdf_from_text(
        self, text_content: str, output_path: Optional[str] = None
    ) -> bytes:
        """
        Create a PDF document from text content

        Args:
            text_content: The text to write to the document
            output_path: Optional path to save the file

        Returns:
            bytes: The PDF file as bytes
        """
        try:
            # Create buffer
            buffer = io.BytesIO()

            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin,
            )

            # Get styles
            styles = getSampleStyleSheet()

            # Create custom style for body text
            normal_style = ParagraphStyle(
                "CustomNormal",
                parent=styles["Normal"],
                fontSize=12,
                spaceAfter=12,
                alignment=TA_LEFT,
            )

            # Create title style
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Title"],
                fontSize=16,
                spaceAfter=24,
                alignment=TA_LEFT,
            )

            # Build content
            story = []

            # Add title
            story.append(Paragraph("Scrubbed Document", title_style))
            story.append(Spacer(1, 12))

            # Split content into paragraphs
            paragraphs = text_content.split("\n\n")

            for paragraph_text in paragraphs:
                if paragraph_text.strip():
                    # Clean the text and add paragraph
                    clean_text = paragraph_text.strip().replace("\n", "<br/>")
                    story.append(Paragraph(clean_text, normal_style))
                    story.append(Spacer(1, 6))

            # Build PDF
            doc.build(story)

            # Get bytes
            buffer.seek(0)
            pdf_bytes = buffer.getvalue()

            # Optionally save to file
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(pdf_bytes)

            return pdf_bytes

        except Exception as e:
            raise Exception(f"Error creating PDF file: {str(e)}")

    def create_pdf_buffer(self, text_content: str) -> io.BytesIO:
        """
        Create a PDF document and return as BytesIO buffer

        Args:
            text_content: The text to write to the document

        Returns:
            io.BytesIO: Buffer containing the PDF file
        """
        try:
            buffer = io.BytesIO()

            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin,
            )

            styles = getSampleStyleSheet()
            normal_style = styles["Normal"]
            title_style = styles["Title"]

            story = []
            story.append(Paragraph("Scrubbed Document", title_style))
            story.append(Spacer(1, 12))

            # Add content
            paragraphs = text_content.split("\n\n")
            for paragraph_text in paragraphs:
                if paragraph_text.strip():
                    clean_text = paragraph_text.strip().replace("\n", "<br/>")
                    story.append(Paragraph(clean_text, normal_style))
                    story.append(Spacer(1, 6))

            doc.build(story)
            buffer.seek(0)

            return buffer

        except Exception as e:
            raise Exception(f"Error creating PDF buffer: {str(e)}")
