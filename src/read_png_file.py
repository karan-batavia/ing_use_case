from PIL import Image
import pytesseract
import os
import cv2

# === Optional: specify Tesseract path if not in PATH ===
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Input image ===
image_path = r"image2.png"  # 🔹 change this path

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

# === Preprocess image (optional but improves accuracy) ===
# Load image using OpenCV
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding (binarization)
bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# === Extract text using multiple languages ===
# Combine multiple languages with '+' (supported by Tesseract)
LANGS = "eng+fra+nld"  # English + French + Dutch

extracted_text = pytesseract.image_to_string(bw, lang=LANGS)

# === Display result ===
print("Extracted text:\n")
print(extracted_text.strip())

# === Save result to text file ===
output_file = os.path.splitext(image_path)[0] + "_text.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(extracted_text)

print(f"\n Text saved to: {output_file}")
