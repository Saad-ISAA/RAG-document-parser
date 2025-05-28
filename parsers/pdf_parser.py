# parsers/pdf_parser.py
"""
PDF parsing module with support for multiple PDF libraries.
Automatically selects the best available library for optimal performance.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import io

# Import PDF libraries with fallbacks
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    try:
        import PyPDF2 as pypdf
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False
        pypdf = None

from models.parse_result import ParseResult, TableData, ImageInfo, DocumentMetadata, FileInfo
from utils.config import ParserConfig

logger = logging.getLogger(__name__)


class PDFParser:
    """
    High-performance PDF parser with automatic library selection.
    Supports pdfplumber, PyMuPDF, and pypdf with fallback mechanisms.
    """
    
    def __init__(self, config: ParserConfig):
        """Initialize PDF parser with configuration."""
        self.config = config
        self.pdf_config = config.pdf
        
        # Determine available libraries and priority
        self.available_libs = []
        if HAS_PDFPLUMBER:
            self.available_libs.append('pdfplumber')
        if HAS_PYMUPDF:
            self.available_libs.append('pymupdf')
        if HAS_PYPDF:
            self.available_libs.append('pypdf')
        
        if not self.available_libs:
            raise ImportError("No PDF parsing library available. Install pdfplumber, PyMuPDF, or pypdf.")
        
        logger.info(f"PDF parser initialized with libraries: {', '.join(self.available_libs)}")
    
    def parse(self, file_path: Path, file_info: FileInfo) -> ParseResult:
        """
        Parse a PDF file using the best available method.
        
        Args:
            file_path: Path to the PDF file
            file_info: File information from detector
            
        Returns:
            ParseResult with extracted content
        """
        result = ParseResult(
            file_path=str(file_path),
            file_type=file_info.mime_type
        )
        
        # Try parsers in order of preference
        parsers_to_try = self._get_parser_priority()
        
        for parser_name in parsers_to_try:
            try:
                logger.debug(f"Trying {parser_name} for {file_path.name}")
                
                if parser_name == 'pdfplumber':
                    return self._parse_with_pdfplumber(file_path, file_info, result)
                elif parser_name == 'pymupdf':
                    return self._parse_with_pymupdf(file_path, file_info, result)
                elif parser_name == 'pypdf':
                    return self._parse_with_pypdf(file_path, file_info, result)
                    
            except Exception as e:
                logger.warning(f"{parser_name} failed for {file_path.name}: {str(e)}")
                result.error = f"{parser_name} failed: {str(e)}"
                continue
        
        # If all parsers failed
        result.success = False
        result.error = f"All PDF parsers failed. Last error: {result.error}"
        return result
    
    def _get_parser_priority(self) -> List[str]:
        """Get parser priority based on configuration and availability."""
        priority = []
        
        # If OCR is enabled, prefer PyMuPDF for better image extraction
        if self.config.enable_ocr and 'pymupdf' in self.available_libs:
            priority.append('pymupdf')
            logger.debug("Using PyMuPDF for OCR-enabled PDF parsing")
        
        # Honor preference in configuration
        if self.pdf_config.prefer_pdfplumber and 'pdfplumber' in self.available_libs:
            if 'pdfplumber' not in priority:
                priority.append('pdfplumber')
        
        # Add remaining parsers in optimal order
        optimal_order = ['pdfplumber', 'pymupdf', 'pypdf']
        for parser in optimal_order:
            if parser in self.available_libs and parser not in priority:
                priority.append(parser)
        
        return priority
    
    def _parse_with_pdfplumber(self, file_path: Path, file_info: FileInfo, result: ParseResult) -> ParseResult:
        """Parse PDF using pdfplumber (best for tables and structured content)."""
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            result.metadata = self._extract_metadata_pdfplumber(pdf)
            
            all_text = []
            page_number = 0
            
            for page in pdf.pages:
                page_number += 1
                
                # Extract text
                text = page.extract_text()
                if text:
                    all_text.append(text)
                
                # Extract tables if enabled
                if self.pdf_config.extract_tables:
                    tables = self._extract_tables_pdfplumber(page, page_number)
                    result.tables.extend(tables)
                
                # Extract images if enabled (limited in pdfplumber)
                if self.pdf_config.extract_images:
                    images = self._extract_images_pdfplumber(page, page_number)
                    result.images.extend(images)
            
            # Combine all text
            result.content = '\n\n'.join(all_text)
            result.success = True
            result.parser_used = 'pdfplumber'
            
            return result
    
    def _parse_with_pymupdf(self, file_path: Path, file_info: FileInfo, result: ParseResult) -> ParseResult:
        """Parse PDF using PyMuPDF (fast and comprehensive)."""
        doc = fitz.open(file_path)
        
        try:
            # Extract metadata
            result.metadata = self._extract_metadata_pymupdf(doc)
            
            all_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                if text:
                    all_text.append(text)
                
                # Extract tables if enabled
                if self.pdf_config.extract_tables:
                    tables = self._extract_tables_pymupdf(page, page_num + 1)
                    result.tables.extend(tables)
                
                # Extract images if enabled
                if self.pdf_config.extract_images:
                    images = self._extract_images_pymupdf(page, page_num + 1)
                    result.images.extend(images)
            
            # Combine all text
            result.content = '\n\n'.join(all_text)
            result.success = True
            result.parser_used = 'pymupdf'
            
        finally:
            doc.close()
        
        return result
    
    def _parse_with_pypdf(self, file_path: Path, file_info: FileInfo, result: ParseResult) -> ParseResult:
        """Parse PDF using pypdf (lightweight fallback)."""
        with open(file_path, 'rb') as file:
            if hasattr(pypdf, 'PdfReader'):
                # pypdf (newer version)
                pdf_reader = pypdf.PdfReader(file)
            else:
                # PyPDF2 (older version)
                pdf_reader = pypdf.PdfFileReader(file)
            
            # Extract metadata
            result.metadata = self._extract_metadata_pypdf(pdf_reader)
            
            all_text = []
            
            # Get pages
            if hasattr(pdf_reader, 'pages'):
                pages = pdf_reader.pages
            else:
                pages = [pdf_reader.getPage(i) for i in range(len(pdf_reader.pages))]
            
            for page_num, page in enumerate(pages):
                # Extract text
                if hasattr(page, 'extract_text'):
                    text = page.extract_text()
                else:
                    text = page.extractText()
                
                if text:
                    all_text.append(text)
            
            # Combine all text
            result.content = '\n\n'.join(all_text)
            result.success = True
            result.parser_used = 'pypdf'
            
            # Note: pypdf has limited table and image extraction capabilities
            if self.pdf_config.extract_tables or self.pdf_config.extract_images:
                logger.warning("Table and image extraction limited with pypdf. Consider installing pdfplumber or PyMuPDF.")
        
        return result
    
    def _extract_metadata_pdfplumber(self, pdf) -> DocumentMetadata:
        """Extract metadata using pdfplumber."""
        metadata = DocumentMetadata()
        
        if hasattr(pdf, 'metadata') and pdf.metadata:
            info = pdf.metadata
            metadata.title = info.get('Title')
            metadata.author = info.get('Author')
            metadata.subject = info.get('Subject')
            metadata.creator = info.get('Creator')
            
            # Handle dates
            if 'CreationDate' in info:
                metadata.creation_date = self._parse_pdf_date(info['CreationDate'])
            if 'ModDate' in info:
                metadata.modification_date = self._parse_pdf_date(info['ModDate'])
        
        metadata.page_count = len(pdf.pages)
        return metadata
    
    def _extract_metadata_pymupdf(self, doc) -> DocumentMetadata:
        """Extract metadata using PyMuPDF."""
        metadata = DocumentMetadata()
        info = doc.metadata
        
        metadata.title = info.get('title')
        metadata.author = info.get('author')
        metadata.subject = info.get('subject')
        metadata.creator = info.get('creator')
        
        # Handle dates
        if 'creationDate' in info:
            metadata.creation_date = self._parse_pdf_date(info['creationDate'])
        if 'modDate' in info:
            metadata.modification_date = self._parse_pdf_date(info['modDate'])
        
        metadata.page_count = len(doc)
        return metadata
    
    def _extract_metadata_pypdf(self, pdf_reader) -> DocumentMetadata:
        """Extract metadata using pypdf."""
        metadata = DocumentMetadata()
        
        if hasattr(pdf_reader, 'metadata'):
            info = pdf_reader.metadata
        elif hasattr(pdf_reader, 'getDocumentInfo'):
            info = pdf_reader.getDocumentInfo()
        else:
            info = {}
        
        if info:
            metadata.title = info.get('/Title')
            metadata.author = info.get('/Author')
            metadata.subject = info.get('/Subject')
            metadata.creator = info.get('/Creator')
            
            # Handle dates
            if '/CreationDate' in info:
                metadata.creation_date = self._parse_pdf_date(info['/CreationDate'])
            if '/ModDate' in info:
                metadata.modification_date = self._parse_pdf_date(info['/ModDate'])
        
        if hasattr(pdf_reader, 'pages'):
            metadata.page_count = len(pdf_reader.pages)
        elif hasattr(pdf_reader, 'getNumPages'):
            metadata.page_count = pdf_reader.getNumPages()
        
        return metadata
    
    def _extract_tables_pdfplumber(self, page, page_number: int) -> List[TableData]:
        """Extract tables using pdfplumber."""
        tables = []
        
        try:
            page_tables = page.extract_tables(self.pdf_config.table_settings)
            
            for table_index, table in enumerate(page_tables):
                if table and len(table) > 1:  # Must have header and at least one row
                    headers = table[0] if table[0] else []
                    rows = table[1:] if len(table) > 1 else []
                    
                    # Clean up None values
                    headers = [str(h) if h is not None else "" for h in headers]
                    cleaned_rows = []
                    for row in rows:
                        cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                        cleaned_rows.append(cleaned_row)
                    
                    if headers and cleaned_rows:
                        tables.append(TableData(
                            headers=headers,
                            rows=cleaned_rows,
                            page_number=page_number,
                            table_index=table_index
                        ))
        
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_number}: {str(e)}")
        
        return tables
    
    def _extract_tables_pymupdf(self, page, page_number: int) -> List[TableData]:
        """Extract tables using PyMuPDF."""
        tables = []
        
        try:
            # PyMuPDF table extraction (requires pymupdf >= 1.23.0)
            if hasattr(page, 'find_tables'):
                page_tables = page.find_tables()
                
                for table_index, table in enumerate(page_tables):
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 1:
                        headers = table_data[0] if table_data[0] else []
                        rows = table_data[1:] if len(table_data) > 1 else []
                        
                        # Clean up data
                        headers = [str(h) if h else "" for h in headers]
                        cleaned_rows = []
                        for row in rows:
                            cleaned_row = [str(cell) if cell else "" for cell in row]
                            cleaned_rows.append(cleaned_row)
                        
                        if headers and cleaned_rows:
                            tables.append(TableData(
                                headers=headers,
                                rows=cleaned_rows,
                                page_number=page_number,
                                table_index=table_index
                            ))
        
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_number}: {str(e)}")
        
        return tables
    
    def _extract_images_pdfplumber(self, page, page_number: int) -> List[ImageInfo]:
        """Extract images using pdfplumber (limited capability)."""
        images = []
        
        try:
            # pdfplumber has limited image extraction
            if hasattr(page, 'images'):
                for img_index, img in enumerate(page.images):
                    image_info = ImageInfo(
                        image_index=img_index,
                        page_number=page_number,
                        width=int(img.get('width', 0)),
                        height=int(img.get('height', 0))
                    )
                    
                    # Note: pdfplumber doesn't easily provide image data for OCR
                    # For OCR on PDF images, PyMuPDF is preferred
                    logger.debug(f"pdfplumber detected image but OCR requires PyMuPDF for better image extraction")
                    
                    images.append(image_info)
        
        except Exception as e:
            logger.warning(f"Image extraction failed on page {page_number}: {str(e)}")
        
        return images
    
    def _extract_images_pymupdf(self, page, page_number: int) -> List[ImageInfo]:
        """Extract images using PyMuPDF with OCR support."""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Get image info
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # Skip non-RGB images if needed
                    image_info = ImageInfo(
                        image_index=img_index,
                        page_number=page_number,
                        width=pix.width,
                        height=pix.height,
                        size_bytes=len(pix.tobytes())
                    )
                    
                    # Run OCR if enabled and image is large enough for text
                    if (self.config.enable_ocr and 
                        pix.width > 100 and pix.height > 50):  # Size threshold for text
                        try:
                            logger.debug(f"Running OCR on image {img_index} (page {page_number}): {pix.width}x{pix.height}")
                            ocr_text = self._run_ocr_on_pixmap(pix)
                            if ocr_text:
                                image_info.extracted_text = ocr_text
                                logger.info(f"OCR extracted {len(ocr_text)} chars from image {img_index} on page {page_number}")
                            else:
                                logger.debug(f"OCR found no text in image {img_index} on page {page_number}")
                        except Exception as e:
                            logger.warning(f"OCR failed for image {img_index} on page {page_number}: {str(e)}")
                    else:
                        if not self.config.enable_ocr:
                            logger.debug(f"OCR disabled, skipping image {img_index}")
                        else:
                            logger.debug(f"Image {img_index} too small for OCR: {pix.width}x{pix.height}")
                    
                    images.append(image_info)
                
                # Clean up pixmap memory
                if pix:
                    pix = None
        
        except Exception as e:
            logger.warning(f"Image extraction failed on page {page_number}: {str(e)}")
        
        return images
    
    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """Parse PDF date string to datetime object."""
        if not date_str:
            return None
    
    def _run_ocr_on_pixmap(self, pix) -> Optional[str]:
        """Run OCR on a PyMuPDF pixmap object."""
        converted_pix = None
        try:
            # Handle different pixmap formats and alpha channels
            if pix.alpha:
                # Remove alpha channel by converting to RGB
                converted_pix = fitz.Pixmap(fitz.csRGB, pix)
                working_pix = converted_pix
            else:
                working_pix = pix
            
            # Convert to RGB if not already
            if working_pix.colorspace and working_pix.colorspace.n != 3:
                if converted_pix:
                    converted_pix = None  # Clean up previous
                converted_pix = fitz.Pixmap(fitz.csRGB, working_pix)
                working_pix = converted_pix
            
            # Convert pixmap to bytes in a format PIL can handle
            try:
                # Try PNG format first (handles alpha better)
                img_data = working_pix.tobytes("png")
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(img_data))
            except Exception:
                # Fallback to PPM format (no alpha)
                img_data = working_pix.tobytes("ppm")
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(img_data))
            
            # Ensure RGB format for OCR
            if pil_img.mode not in ('RGB', 'L'):
                if pil_img.mode in ('RGBA', 'LA'):
                    # Create white background
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    if pil_img.mode == 'RGBA':
                        background.paste(pil_img, mask=pil_img.split()[3])  # Use alpha as mask
                    else:
                        background.paste(pil_img.convert('RGB'))
                    pil_img = background
                else:
                    pil_img = pil_img.convert('RGB')
            
            # Run OCR directly with EasyOCR if available
            try:
                import easyocr
                import numpy as np
                
                # Initialize EasyOCR reader with same config as image parser
                reader = easyocr.Reader(self.config.ocr.languages, verbose=False)
                img_array = np.array(pil_img)
                
                # Ensure the array is in the right format (no alpha channel)
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]  # Keep only RGB
                
                results = reader.readtext(img_array)
                
                # Filter by confidence and combine text
                text_parts = []
                low_confidence_parts = []
                for bbox, text, confidence in results:
                    if confidence >= self.config.ocr.confidence_threshold:
                        text_parts.append(text)
                    else:
                        low_confidence_parts.append((text, confidence))
                
                extracted_text = '\n'.join(text_parts) if text_parts else None
                
                # Debug logging
                if extracted_text:
                    logger.debug(f"OCR extracted {len(extracted_text)} chars with {len(text_parts)} high-confidence parts")
                else:
                    if low_confidence_parts:
                        logger.debug(f"OCR found {len(low_confidence_parts)} low-confidence results (threshold: {self.config.ocr.confidence_threshold})")
                        logger.debug(f"Low confidence samples: {low_confidence_parts[:3]}")
                    else:
                        logger.debug(f"OCR found no text in image")
                
                return extracted_text
                
            except ImportError:
                # Fallback to tesseract if available
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(pil_img).strip()
                    return text if text else None
                except ImportError:
                    logger.debug("No OCR engines available in PDF parser")
                    return None
                    
        except Exception as e:
            logger.debug(f"OCR processing failed: {str(e)}")
            return None
        finally:
            # Clean up any converted pixmaps
            if converted_pix:
                converted_pix = None
        
        try:
            # Remove D: prefix if present
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Parse various PDF date formats
            formats = [
                '%Y%m%d%H%M%S',
                '%Y%m%d%H%M',
                '%Y%m%d',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str[:len(fmt)], fmt)
                except ValueError:
                    continue
        
        except Exception:
            pass
        
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported PDF formats."""
        return ['pdf']
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about available PDF libraries."""
        info = {
            'available_libraries': self.available_libs,
            'preferred_library': self.available_libs[0] if self.available_libs else None,
            'capabilities': {
                'text_extraction': True,
                'table_extraction': 'pdfplumber' in self.available_libs or 'pymupdf' in self.available_libs,
                'image_extraction': 'pymupdf' in self.available_libs,
                'metadata_extraction': True
            }
        }
        
        return info