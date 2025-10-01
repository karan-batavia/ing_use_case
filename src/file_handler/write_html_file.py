from bs4 import BeautifulSoup, Tag
import io
from typing import Optional


class HTMLWriter:
    """Class to write text content to HTML format"""

    def __init__(self):
        pass

    def create_html_from_text(
        self, text_content: str, output_path: Optional[str] = None
    ) -> bytes:
        """
        Create an HTML document from text content

        Args:
            text_content: The text to write to the document
            output_path: Optional path to save the file

        Returns:
            bytes: The HTML file as bytes
        """
        try:
            # Create a new HTML document
            soup = BeautifulSoup("", "html.parser")

            # Create basic HTML structure
            html_tag = soup.new_tag("html")
            soup.append(html_tag)

            # Add head
            head_tag = soup.new_tag("head")
            html_tag.append(head_tag)

            # Add meta tags
            meta_charset = soup.new_tag("meta", charset="utf-8")
            head_tag.append(meta_charset)

            meta_viewport = soup.new_tag("meta")
            meta_viewport["name"] = "viewport"
            meta_viewport["content"] = "width=device-width, initial-scale=1.0"
            head_tag.append(meta_viewport)

            # Add title
            title_tag = soup.new_tag("title")
            title_tag.string = "Scrubbed Document"
            head_tag.append(title_tag)

            # Add basic CSS
            style_tag = soup.new_tag("style")
            style_tag.string = """
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #007acc;
                padding-bottom: 10px;
            }
            p {
                margin-bottom: 15px;
                text-align: justify;
            }
            """
            head_tag.append(style_tag)

            # Add body
            body_tag = soup.new_tag("body")
            html_tag.append(body_tag)

            # Add container div
            container_div = soup.new_tag("div")
            container_div["class"] = "container"
            body_tag.append(container_div)

            # Add title
            h1_tag = soup.new_tag("h1")
            h1_tag.string = "Scrubbed Document"
            container_div.append(h1_tag)

            # Split content into paragraphs
            paragraphs = text_content.split("\n\n")

            for paragraph_text in paragraphs:
                if paragraph_text.strip():
                    p_tag = soup.new_tag("p")
                    # Handle line breaks within paragraphs
                    lines = paragraph_text.strip().split("\n")
                    for i, line in enumerate(lines):
                        if i > 0:
                            # Add line break
                            br_tag = soup.new_tag("br")
                            p_tag.append(br_tag)
                        p_tag.append(line)
                    container_div.append(p_tag)

            # Convert to string and encode
            html_string = str(soup.prettify())
            html_bytes = html_string.encode("utf-8")

            # Optionally save to file
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_string)

            return html_bytes

        except Exception as e:
            raise Exception(f"Error creating HTML file: {str(e)}")

    def create_html_buffer(self, text_content: str) -> io.BytesIO:
        """
        Create an HTML document and return as BytesIO buffer

        Args:
            text_content: The text to write to the document

        Returns:
            io.BytesIO: Buffer containing the HTML file
        """
        try:
            # Create HTML content
            html_bytes = self.create_html_from_text(text_content)

            # Create buffer
            buffer = io.BytesIO(html_bytes)
            buffer.seek(0)

            return buffer

        except Exception as e:
            raise Exception(f"Error creating HTML buffer: {str(e)}")

    def create_simple_html_from_text(self, text_content: str) -> str:
        """
        Create a simple HTML string from text content

        Args:
            text_content: The text to convert

        Returns:
            str: HTML content as string
        """
        try:
            # Simple HTML template
            html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scrubbed Document</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        p {{ margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>Scrubbed Document</h1>
    {content}
</body>
</html>"""

            # Convert paragraphs
            paragraphs = text_content.split("\n\n")
            content_html = ""

            for paragraph_text in paragraphs:
                if paragraph_text.strip():
                    # Replace single newlines with <br> tags
                    formatted_text = paragraph_text.strip().replace("\n", "<br>")
                    content_html += f"    <p>{formatted_text}</p>\n"

            return html_template.format(content=content_html)

        except Exception as e:
            raise Exception(f"Error creating simple HTML: {str(e)}")
