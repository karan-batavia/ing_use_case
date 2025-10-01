import pdfplumber
import os
from pathlib import Path


class PDFToTextConverter:
    def __init__(self, save_to_file=False):
        """
        Initialize the PDF to text converter.

        Args:
            save_to_file (bool): Whether to save extracted text to a file
        """
        self.save_to_file = save_to_file

    def extract_text_from_pdf(self, pdf_path, output_path=None):
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file
            output_path (str, optional): Path to save the extracted text

        Returns:
            str: Extracted text from the PDF
        """
        try:
            full_text = ""

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

            # Save to file if requested
            if self.save_to_file or output_path:
                if output_path is None:
                    output_path = str(Path(pdf_path).with_suffix("_text.txt"))

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

                print(f"✅ Text extracted and saved to: {output_path}")

            return full_text.strip()

        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            return ""

    def convert_pdf_to_txt(self, pdf_file, txt_file=None):
        """
        Convert PDF to text file (for compatibility with other converters).

        Args:
            pdf_file (str): Path to the input PDF file
            txt_file (str, optional): Path to the output TXT file

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            text_content = self.extract_text_from_pdf(pdf_file, txt_file)
            return len(text_content) > 0
        except Exception as e:
            print(f"❌ Error converting PDF: {e}")
            return False


# Example usage (for backward compatibility)
if __name__ == "__main__":
    pdf_path = r"kbo_project.pdf"  # this is the path of the upload file. to be modify when create application

    if os.path.exists(pdf_path):
        converter = PDFToTextConverter(save_to_file=True)
        text = converter.extract_text_from_pdf(pdf_path)
        print("Extracted text:\n")
        print(text)
    else:
        print(f"❌ PDF file not found: {pdf_path}")
