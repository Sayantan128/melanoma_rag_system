from pypdf import PdfReader
from pathlib import Path
from dataclasses import dataclass
import typing as t
import logging

from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """
    A page-or-chunk of extracted text, plus metadata.
    """
    content: str
    metadata: dict

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        # hash by object identity so we can use Document as dict keys
        return id(self)


class PDFExtractor:
    """
    Extracts text from a PDF, returning a list of Document objects
    (one per non-empty page), each with source/page metadata.
    Falls back to OCR for pages below ocr_threshold chars.
    """
    def __init__(
        self,
        pdf_path: Path,
        ocr_threshold: int = 100,
        ocr_dpi: int = 200
    ):
        self.pdf_path = pdf_path
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        try:
            self.reader = PdfReader(str(pdf_path))
        except Exception as e:
            raise IOError(f"Could not read PDF {pdf_path}: {e}")
        self.ocr_threshold = ocr_threshold
        self.ocr_dpi = ocr_dpi

    def extract_text_with_metadata(self) -> t.List[Document]:
        docs: t.List[Document] = []
        total = len(self.reader.pages)
        for idx, page in enumerate(self.reader.pages, start=1):
            text = page.extract_text() or ""
            if len(text) < self.ocr_threshold:
                logger.warning(f"Page {idx} of {self.pdf_path.name} low-text ({len(text)} chars); running OCR.")
                text = self._ocr_page(idx)
            if text.strip():
                meta = {
                    "source": self.pdf_path.name,
                    "page_number": idx,
                    "total_pages": total
                }
                docs.append(Document(content=text, metadata=meta))
        return docs

    def _ocr_page(self, page_number: int) -> str:
        images = convert_from_path(
            str(self.pdf_path),
            dpi=self.ocr_dpi,
            first_page=page_number,
            last_page=page_number
        )
        # assume single image
        img = images[0]
        return pytesseract.image_to_string(img)
