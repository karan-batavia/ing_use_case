# src/io_extract.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import io, mimetypes
import pandas as pd
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import docx
from PIL import Image
import pytesseract

def _ocr_tesseract(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img)

def extract_from_pdf(b: bytes) -> str:
    # Try PyPDF2 first; if empty, fallback to pdfplumber (optional)
    reader = PdfReader(io.BytesIO(b))
    txt = "\n".join((p.extract_text() or "") for p in reader.pages)
    return txt

def extract_from_csv(b: bytes) -> Tuple[str, pd.DataFrame]:
    df = pd.read_csv(io.BytesIO(b))
    blob = "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist())
    return blob, df

def extract_from_docx(b: bytes) -> str:
    d = docx.Document(io.BytesIO(b))
    return "\n".join(p.text for p in d.paragraphs)

def extract_from_html(b: bytes) -> str:
    soup = BeautifulSoup(b, "html.parser")
    return soup.get_text(" ", strip=True)

def extract_text(uploaded_file) -> Dict[str, Any]:
    """
    Returns:
      {"kind": "text", "text": str}  or
      {"kind": "csv",  "text": str, "df": DataFrame}
    """
    name = uploaded_file.name
    mime = uploaded_file.type or (mimetypes.guess_type(name)[0] or "")
    data = uploaded_file.getvalue()
    ext = Path(name).suffix.lower()

    # Plain text
    if mime.startswith("text/") and ext not in {".html", ".htm", ".csv"}:
        return {"kind": "text", "text": data.decode("utf-8", errors="ignore")}

    # HTML
    if ext in {".html", ".htm"} or mime in {"text/html", "application/xhtml+xml"}:
        return {"kind": "text", "text": extract_from_html(data)}

    # PDF
    if ext == ".pdf" or mime == "application/pdf":
        return {"kind": "text", "text": extract_from_pdf(data)}

    # DOCX
    if ext == ".docx" or mime.endswith("officedocument.wordprocessingml.document"):
        return {"kind": "text", "text": extract_from_docx(data)}

    # CSV
    if ext == ".csv" or mime in {"text/csv", "application/csv"}:
        blob, df = extract_from_csv(data)
        return {"kind": "csv", "text": blob, "df": df}

    # Images (png/jpg/jpeg)
    if ext in {".png", ".jpg", ".jpeg"} or mime.startswith("image/"):
        return {"kind": "text", "text": _ocr_tesseract(data)}

    # Fallback
    return {"kind": "text", "text": data.decode("utf-8", errors="ignore")}