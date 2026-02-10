"""
PDF and document conversion utilities.
"""
import fitz  # PyMuPDF
import pypandoc
from pathlib import Path
from typing import Optional, List
from PIL import Image


class DocumentConverter:
    """Handles PDF and document conversion operations."""

    def __init__(self):
        """Initialize document converter."""
        pass

    def pdf_to_images(
        self,
        pdf_path: Path,
        max_pages: Optional[int] = None,
        label: str = ""
    ) -> Optional[List[Image.Image]]:
        """
        Convert PDF to list of PIL images.

        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to convert (None for all)
            label: Label for logging (e.g., "main", "supplementary")

        Returns:
            List of PIL images or None if not found
        """
        if not pdf_path.exists():
            print(f"PDF not found at {pdf_path}")
            return None

        try:
            print(f"Converting {label} PDF to images: {pdf_path.name}...")
            doc = fitz.open(str(pdf_path))
            images = []

            num_pages = len(doc) if max_pages is None else min(max_pages, len(doc))
            print(f"  Processing {num_pages} pages...")

            for page_num in range(num_pages):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)

            doc.close()
            print(f"  Converted {len(images)} pages to images")
            return images

        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return None

    def docx_to_pdf(self, docx_path: Path) -> Optional[Path]:
        """
        Convert DOCX to PDF using pypandoc.

        Args:
            docx_path: Path to the DOCX file

        Returns:
            Path to the converted PDF or None if conversion failed
        """
        if not docx_path.exists():
            print(f"DOCX not found at {docx_path}")
            return None

        pdf_path = docx_path.with_suffix('.pdf')

        if pdf_path.exists():
            print(f"  Using cached PDF: {pdf_path.name}")
            return pdf_path

        try:
            print(f"  Converting DOCX to PDF: {docx_path.name}...")

            pypandoc.convert_file(
                str(docx_path),
                'pdf',
                outputfile=str(pdf_path),
                extra_args=['--pdf-engine=tectonic']
            )

            if pdf_path.exists():
                print(f"  Successfully converted to: {pdf_path.name}")
                return pdf_path
            else:
                print(f"  Conversion completed but PDF not found")
                return None

        except ImportError:
            print("  Error: pypandoc not installed. Install with: pip install pypandoc")
            return None
        except OSError as e:
            if "pandoc" in str(e).lower():
                print("  Error: pandoc not installed. Install with: apt install pandoc")
            else:
                print(f"  Error: {e}")
            return None
        except Exception as e:
            print(f"  Error converting DOCX to PDF: {e}")
            return None

    def file_to_images(
        self,
        file_path: Path,
        max_pages: Optional[int] = None,
        label: str = ""
    ) -> Optional[List[Image.Image]]:
        """
        Convert a file (PDF or DOCX) to images based on its extension.

        Args:
            file_path: Path to the file
            max_pages: Maximum pages
            label: Label for logging

        Returns:
            List of PIL images or None
        """
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            return self.pdf_to_images(file_path, max_pages=max_pages, label=label)
        elif suffix == '.docx':
            pdf_path = self.docx_to_pdf(file_path)
            if pdf_path:
                return self.pdf_to_images(pdf_path, max_pages=max_pages, label=label)
            else:
                print(f"  Failed to convert DOCX to PDF: {file_path.name}")
                return None
        else:
            print(f"Unsupported file type: {suffix}")
            return None
