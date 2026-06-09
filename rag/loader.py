import os
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def _load_pdf_file_with_pypdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""


def _load_pdf_file_with_fitz(file_path):
    if fitz is None:
        return ""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""


def _load_pdf_file(file_path):
    # Try pypdf first, then fall back to PyMuPDF if available and pypdf returned little content
    text = _load_pdf_file_with_pypdf(file_path)
    if not text or len(text.strip()) < 100:
        fb = _load_pdf_file_with_fitz(file_path)
        if fb and len(fb.strip()) > len(text.strip()):
            return fb
    return text


def load_pdf(file_path):
    if os.path.isdir(file_path):
        merged = ""
        for root, _, files in os.walk(file_path):
            for file_name in sorted(files):
                if file_name.lower().endswith(".pdf"):
                    merged += _load_pdf_file(os.path.join(root, file_name))
        return merged

    return _load_pdf_file(file_path)
    