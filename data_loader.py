import os
import re
import logging
from pathlib import Path
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Normalizes extracted text by removing excessive whitespace and non-ascii characters."""
    # Remove non-ascii characters
    text = text.encode("ascii", "ignore").decode()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts and cleans text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        doc.close()
        return clean_text(" ".join(full_text))
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def load_contracts(data_dir: Path) -> dict[str, str]:
    """Iterates through a directory and extracts text from all PDF files."""
    contracts = {}
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return contracts

    for file_name in os.listdir(data_dir):
        if file_name.lower().endswith(".pdf"):
            file_path = data_dir / file_name
            logger.info(f"Loading PDF: {file_name}")
            text = extract_text_from_pdf(file_path)
            if text:
                contracts[file_name] = text
            else:
                logger.warning(f"No text extracted from {file_name}")
    return contracts
