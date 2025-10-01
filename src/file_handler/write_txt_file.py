import io
from typing import Optional


class TXTWriter:
    """Class to write text content to TXT format"""

    def __init__(self):
        pass

    def create_txt_from_text(
        self, text_content: str, output_path: Optional[str] = None
    ) -> bytes:
        """
        Create a TXT file from text content

        Args:
            text_content: The text to write to the file
            output_path: Optional path to save the file

        Returns:
            bytes: The TXT file as bytes
        """
        try:
            # Encode text as UTF-8 bytes
            txt_bytes = text_content.encode("utf-8")

            # Optionally save to file
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text_content)

            return txt_bytes

        except Exception as e:
            raise Exception(f"Error creating TXT file: {str(e)}")

    def create_txt_buffer(self, text_content: str) -> io.BytesIO:
        """
        Create a TXT file and return as BytesIO buffer

        Args:
            text_content: The text to write to the file

        Returns:
            io.BytesIO: Buffer containing the TXT file
        """
        try:
            # Create buffer with text content
            buffer = io.BytesIO()
            buffer.write(text_content.encode("utf-8"))
            buffer.seek(0)

            return buffer

        except Exception as e:
            raise Exception(f"Error creating TXT buffer: {str(e)}")
