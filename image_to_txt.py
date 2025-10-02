import os
import sys
import argparse
import glob
import re
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Image processing libraries
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

class ImageToTextConverter:
    def __init__(self, 
                 apply_preprocessing=True,
                 use_local_threshold=True,
                 remove_noise=True,
                 enhance_contrast=True,
                 preserve_word_order=True,
                 remove_special_chars=False,
                 clean_whitespace=True,
                 confidence_threshold=30.0,
                 languages=['eng'],
                 tesseract_config='--psm 6'):
        """
        Initialize the Image to Text converter with advanced OCR options.
        
        Args:
            apply_preprocessing (bool): Apply image preprocessing
            use_local_threshold (bool): Use adaptive thresholding instead of global
            remove_noise (bool): Remove noise from image
            enhance_contrast (bool): Enhance image contrast
            preserve_word_order (bool): Maintain proper word ordering
            remove_special_chars (bool): Remove unwanted special characters
            clean_whitespace (bool): Clean extra whitespace
            confidence_threshold (float): Minimum confidence for OCR results
            languages (list): Languages for OCR (e.g., ['eng', 'fra'])
            tesseract_config (str): Tesseract configuration string
        """
        self.apply_preprocessing = apply_preprocessing
        self.use_local_threshold = use_local_threshold
        self.remove_noise = remove_noise
        self.enhance_contrast = enhance_contrast
        self.preserve_word_order = preserve_word_order
        self.remove_special_chars = remove_special_chars
        self.clean_whitespace = clean_whitespace
        self.confidence_threshold = confidence_threshold
        self.languages = languages
        self.tesseract_config = tesseract_config
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect Tesseract path on Windows
        self._setup_tesseract()
        
        # Supported image formats
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']

    def _setup_tesseract(self):
        """Auto-detect and setup Tesseract path."""
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    self.logger.info(f"Found Tesseract at: {path}")
                    return
            
            self.logger.warning("Tesseract not found. Please install Tesseract-OCR or set the path manually.")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Apply comprehensive image preprocessing for better OCR results.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if not self.apply_preprocessing:
            return image
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()
        
        # Enhance contrast
        if self.enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        # Remove noise
        if self.remove_noise:
            processed = cv2.medianBlur(processed, 3)
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Apply thresholding
        if self.use_local_threshold:
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return processed

    def extract_text_from_image(self, image_path: str) -> Dict[str, any]:
        """
        Extract text from image with comprehensive processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(processed_image)
            
            # Extract text with configuration
            lang_string = '+'.join(self.languages)
            text = pytesseract.image_to_string(
                pil_image, 
                lang=lang_string,
                config=self.tesseract_config
            )
            
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Get confidence data
            try:
                ocr_data = pytesseract.image_to_data(
                    pil_image, 
                    lang=lang_string,
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                avg_confidence = self._calculate_average_confidence(ocr_data)
            except:
                avg_confidence = 0
            
            return {
                'text': cleaned_text,
                'confidence': avg_confidence,
                'total_characters': len(cleaned_text),
                'total_lines': len(cleaned_text.split('\n')) if cleaned_text else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                'text': '',
                'confidence': 0,
                'total_characters': 0,
                'total_lines': 0,
                'error': str(e)
            }

    def _calculate_average_confidence(self, ocr_data: Dict) -> float:
        """Calculate average confidence for OCR results."""
        confidences = [c for c in ocr_data['conf'] if c > 0]
        return np.mean(confidences) if confidences else 0

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing unwanted characters and formatting."""
        if not text:
            return text
        
        cleaned = text
        
        # Remove special characters if requested
        if self.remove_special_chars:
            # Keep letters, numbers, common punctuation, and whitespace
            cleaned = re.sub(r'[^\w\s\.,;:!?\-\'\"()\[\]{}/@#$%&*+=<>|\\~`]', '', cleaned)
        
        # Clean whitespace if requested
        if self.clean_whitespace:
            # Replace multiple spaces with single space
            cleaned = re.sub(r'\s+', ' ', cleaned)
            # Remove leading/trailing whitespace from each line
            lines = [line.strip() for line in cleaned.split('\n')]
            # Remove empty lines
            lines = [line for line in lines if line]
            cleaned = '\n'.join(lines)
        
        return cleaned.strip()

    def convert_image_to_txt(self, input_image_path: str, output_txt_path: str = None) -> bool:
        """
        Convert a single image file to text.
        
        Args:
            input_image_path: Path to the input image file
            output_txt_path: Path to the output text file (optional)
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Check if file exists and has supported extension
            if not os.path.exists(input_image_path):
                self.logger.error(f"File not found: {input_image_path}")
                return False
            
            ext = Path(input_image_path).suffix.lower()
            if ext not in self.supported_extensions:
                self.logger.error(f"Unsupported file type: {ext}")
                return False
            
            # Auto-generate output filename if not provided
            if output_txt_path is None:
                input_path = Path(input_image_path)
                output_txt_path = input_path.with_suffix('.txt')
            
            self.logger.info(f"Processing: {input_image_path}")
            
            # Extract text
            result = self.extract_text_from_image(input_image_path)
            
            if 'error' in result:
                self.logger.error(f"OCR failed: {result['error']}")
                return False
            
            # Save results
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
                
                # Add metadata as comments
                f.write(f"\n\n# OCR Metadata:\n")
                f.write(f"# Source image: {input_image_path}\n")
                f.write(f"# Total characters: {result['total_characters']}\n")
                f.write(f"# Total lines: {result['total_lines']}\n")
                f.write(f"# Average confidence: {result['confidence']:.1f}%\n")
                f.write(f"# Languages: {', '.join(self.languages)}\n")
                f.write(f"# Preprocessing applied: {self.apply_preprocessing}\n")
            
            self.logger.info(f"Saved text to: {output_txt_path}")
            self.logger.info(f"Extracted {result['total_characters']} characters with {result['confidence']:.1f}% confidence")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting {input_image_path}: {str(e)}")
            return False

    def batch_convert(self, input_pattern: str, output_dir: str = None) -> Dict[str, bool]:
        """
        Convert multiple image files to text.
        
        Args:
            input_pattern: Glob pattern for input files (e.g., "*.jpg")
            output_dir: Output directory for text files (optional)
            
        Returns:
            Dictionary mapping input files to conversion success status
        """
        results = {}
        
        # Find matching files
        image_files = glob.glob(input_pattern)
        if not image_files:
            self.logger.warning(f"No files found matching pattern: {input_pattern}")
            return results
        
        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for image_file in image_files:
            output_file = None
            if output_dir:
                filename = Path(image_file).stem + '.txt'
                output_file = Path(output_dir) / filename
            
            success = self.convert_image_to_txt(image_file, str(output_file) if output_file else None)
            results[image_file] = success
        
        return results


def main():
    """Command line interface for the Image to Text converter."""
    parser = argparse.ArgumentParser(
        description="Convert images to text using advanced OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_to_txt.py image.png
  python image_to_txt.py "*.jpg" --output-dir ./text_output
  python image_to_txt.py document.tiff --no-preprocessing
  python image_to_txt.py scan.png --languages eng+fra --confidence 50
        """
    )
    
    parser.add_argument('input', help='Input image file or glob pattern')
    parser.add_argument('--output-dir', '-o', help='Output directory for text files')
    parser.add_argument('--output-file', help='Output file for single input (ignored for patterns)')
    
    # OCR options
    parser.add_argument('--languages', '-l', default='eng', 
                       help='OCR languages (e.g., "eng", "eng+fra")')
    parser.add_argument('--confidence', '-c', type=float, default=30.0,
                       help='Minimum confidence threshold (0-100)')
    parser.add_argument('--psm', type=int, default=6,
                       help='Tesseract Page Segmentation Mode')
    
    # Preprocessing options
    parser.add_argument('--no-preprocessing', action='store_true',
                       help='Disable image preprocessing')
    parser.add_argument('--no-local-threshold', action='store_true',
                       help='Use global instead of adaptive thresholding')
    parser.add_argument('--no-noise-removal', action='store_true',
                       help='Disable noise removal')
    parser.add_argument('--no-contrast-enhancement', action='store_true',
                       help='Disable contrast enhancement')
    
    # Text processing options
    parser.add_argument('--remove-special-chars', action='store_true',
                       help='Remove special characters')
    parser.add_argument('--no-whitespace-cleaning', action='store_true',
                       help='Disable whitespace cleaning')
    
    args = parser.parse_args()
    
    # Parse languages
    languages = args.languages.replace('+', ' ').split()
    
    # Create converter with options
    converter = ImageToTextConverter(
        apply_preprocessing=not args.no_preprocessing,
        use_local_threshold=not args.no_local_threshold,
        remove_noise=not args.no_noise_removal,
        enhance_contrast=not args.no_contrast_enhancement,
        preserve_word_order=True,
        remove_special_chars=args.remove_special_chars,
        clean_whitespace=not args.no_whitespace_cleaning,
        confidence_threshold=args.confidence,
        languages=languages,
        tesseract_config=f'--psm {args.psm}'
    )
    
    # Check if input is a pattern or single file
    if '*' in args.input or '?' in args.input:
        # Batch processing
        results = converter.batch_convert(args.input, args.output_dir)
        
        # Print summary
        successful = sum(results.values())
        total = len(results)
        print(f"\nConversion complete: {successful}/{total} files processed successfully")
        
        # Print failed files
        failed_files = [f for f, success in results.items() if not success]
        if failed_files:
            print("\nFailed files:")
            for f in failed_files:
                print(f"  - {f}")
    else:
        # Single file processing
        output_file = args.output_file if args.output_file else None
        success = converter.convert_image_to_txt(args.input, output_file)
        
        if success:
            print("Conversion completed successfully")
        else:
            print("Conversion failed")
            sys.exit(1)


if __name__ == "__main__":
    main()