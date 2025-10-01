from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import os
from pathlib import Path

# Configure Tesseract path for different environments
if os.path.exists("/usr/bin/tesseract"):
    # Linux/Docker environment
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
elif os.path.exists("/opt/homebrew/bin/tesseract"):
    # macOS with Homebrew
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


class ImageToTextConverter:
    def __init__(self, languages="eng+fra+nld", save_to_file=False, preprocess=True):
        """
        Initialize the image to text converter using OCR.

        Args:
            languages (str): Languages for OCR (e.g., "eng+fra+nld")
            save_to_file (bool): Whether to save extracted text to a file
            preprocess (bool): Whether to preprocess image for better OCR
        """
        self.languages = languages
        self.save_to_file = save_to_file
        self.preprocess = preprocess

    def preprocess_image(self, image_path):
        """
        Preprocess image to improve OCR accuracy using PIL only.

        Args:
            image_path (str): Path to the image file

        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Load image using PIL
            img = Image.open(image_path)

            # Convert to RGB if needed (handles RGBA, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to grayscale
            gray = img.convert("L")

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.5)

            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            sharpened = sharpness_enhancer.enhance(2.0)

            # Auto-adjust levels (similar to threshold)
            autocontrast = ImageOps.autocontrast(sharpened)

            return autocontrast

        except Exception as e:
            raise ValueError(f"Could not preprocess image {image_path}: {str(e)}")

    def extract_text_from_image(self, image_path, output_path=None):
        """
        Extract text from an image using OCR.

        Args:
            image_path (str): Path to the image file
            output_path (str, optional): Path to save the extracted text

        Returns:
            str: Extracted text from the image
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"❌ Image not found: {image_path}")

            # Preprocess image if enabled
            if self.preprocess:
                processed_image = self.preprocess_image(image_path)
            else:
                # Load image directly with PIL
                processed_image = Image.open(image_path)

            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(
                processed_image, lang=self.languages
            )

            # Clean up the text
            cleaned_text = extracted_text.strip()

            # Save to file if requested
            if self.save_to_file or output_path:
                if output_path is None:
                    output_path = str(Path(image_path).with_suffix("_text.txt"))

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                print(f"✅ Text extracted and saved to: {output_path}")

            return cleaned_text

        except Exception as e:
            print(f"❌ Error extracting text from image: {e}")
            return ""

    def convert_image_to_txt(self, image_file, txt_file=None):
        """
        Convert image to text file (for compatibility with other converters).

        Args:
            image_file (str): Path to the input image file
            txt_file (str, optional): Path to the output TXT file

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            text_content = self.extract_text_from_image(image_file, txt_file)
            return len(text_content) > 0
        except Exception as e:
            print(f"❌ Error converting image: {e}")
            return False


# Example usage (for backward compatibility)
if __name__ == "__main__":
    image_path = r"image2.png"  # this is the path of the upload file. to be modify when create application

    if os.path.exists(image_path):
        converter = ImageToTextConverter(save_to_file=True)
        text = converter.extract_text_from_image(image_path)
        print("Extracted text:\n")
        print(text)
    else:
        print(f"❌ Image file not found: {image_path}")
