"""PDF text extraction"""
from pdfminer.high_level import extract_text as extract_pdf_text
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files"""
    
    @staticmethod
    def extract(filepath: Path) -> str:
        """Extract text from PDF"""
        try:
            text = extract_pdf_text(str(filepath))
            return text.strip() if text else ""
        except Exception as e:
            logger.error(f"Error extracting PDF {filepath}: {e}")
            return ""
