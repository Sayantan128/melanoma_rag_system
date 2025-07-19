import pytest
from pathlib import Path
from src.knowledge_agent.ingestion import PDFExtractor

def test_missing_file(tmp_path):
    missing = tmp_path / "no_such.pdf"
    with pytest.raises(FileNotFoundError):
        PDFExtractor(missing)

def test_invalid_pdf_raises(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4...")  # not a full PDF
    with pytest.raises(IOError) as excinfo:
        PDFExtractor(pdf)
    assert "Could not read PDF" in str(excinfo.value)
