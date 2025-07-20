# tests/test_ingestion_ocr.py
import os
import io
import pytest
import yaml
import numpy as np
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
import pytesseract

from src.knowledge_agent.ingestion import PDFExtractor, Document

# --- 1) Stub out PdfReader.extract_text and OCR calls ---
@pytest.fixture(autouse=True)
def stub_pdfreader_and_ocr(monkeypatch, tmp_path):
    # Dummy PdfReader: page 1 empty, page 2 has “Hello from PDF”
    class DummyPage:
        def __init__(self, text):
            self._text = text
        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, path):
            # ignore path, just build two pages
            self.pages = [DummyPage(""), DummyPage("Hello from PDF")]

    monkeypatch.setattr("src.knowledge_agent.ingestion.PdfReader", DummyReader)

    # Stub convert_from_path → return a dummy PIL Image that pytesseract will read
    class DummyImage:
        pass
    def fake_convert_from_path(pdf_path, dpi, first_page, last_page):
        # We only OCR page 1, so ignore args and return a one‐element list
        return [DummyImage()]
    monkeypatch.setattr("src.knowledge_agent.ingestion.convert_from_path", fake_convert_from_path)

    # Stub pytesseract.image_to_string
    monkeypatch.setattr("src.knowledge_agent.ingestion.pytesseract.image_to_string",
                        lambda img: "OCRed text!")

# --- 2) Fixture to write a two‐page blank PDF via PyPDF (no ReportLab needed) ---
@pytest.fixture
def blank_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    # add two blank pages
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path

def test_ocr_fallback(blank_pdf):
    """
    - Page 1 extract_text() → ''         → < ocr_threshold → goes through OCR → 'OCRed text!'
    - Page 2 extract_text() → 'Hello…'   → >= ocr_threshold → use direct extract
    """
    extractor = PDFExtractor(blank_pdf, ocr_threshold=10, ocr_dpi=200)
    docs = extractor.extract_text_with_metadata()

    # We should have two docs back
    assert len(docs) == 2

    # Page 1: OCR fallback
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].content == "OCRed text!"

    # Page 2: no OCR
    assert docs[1].metadata["page_number"] == 2
    assert docs[1].content == "Hello from PDF"
