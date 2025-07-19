import shutil
import pytest
from pathlib import Path
from src.knowledge_agent.ingestion import PDFExtractor

def test_real_pdf_extracts_text(tmp_path):
    # Copy the real PDF fixture into the tmp dir
    fixture = Path(__file__).parent / "fixtures" / "Current state of melanoma treatment.pdf"
    pdf = tmp_path / "Current state of melanoma treatment.pdf"
    shutil.copy(fixture, pdf)

    # Run the extractor
    extractor = PDFExtractor(pdf)
    docs = extractor.extract_text_with_metadata()

    # You should get at least one Document
    assert isinstance(docs, list)
    assert len(docs) > 0

    # Check metadata on the first page
    first = docs[0]
    assert first.metadata["page_number"] == 1
    assert first.metadata["total_pages"] >= 1

    # Check that some expected text appears (customize this!)
    assert "the" in first.content.lower()
