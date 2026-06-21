import fitz  # PyMuPDF
from typing import List, Dict


def load_pdf_pages(path: str) -> List[Dict]:
    """Load text from each page of a PDF.

    Returns a list of dicts: {"page_num": int, "text": str, "path": str}
    """
    doc = fitz.open(path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text()
        pages.append({"page_num": i + 1, "text": text, "path": path})
    doc.close()
    return pages


def load_multiple_pdfs(paths: List[str]) -> List[Dict]:
    """Load pages from multiple PDFs and return a flattened list of pages."""
    all_pages = []
    for p in paths:
        all_pages.extend(load_pdf_pages(p))
    return all_pages
