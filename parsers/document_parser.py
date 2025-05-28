# parsers/document_parser.py
"""
Document parser for DOCX, DOC, ODT, RTF and other word processing documents.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import io

# Import document libraries with fallbacks
try:
    from docx import Document as DocxDocument
    from docx.shared import Inches
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False
    DocxDocument = None

try:
    from odf import text, teletype
    from odf.opendocument import load as odf_load
    HAS_ODFPY = True
except ImportError:
    HAS_ODFPY = False

try:
    from striprtf.striprtf import rtf_to_text
    HAS_STRIPRTF = True
except ImportError:
    HAS_STRIPRTF = False

try:
    import olefile
    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False

from models.parse_result import ParseResult, TableData, ImageInfo, DocumentMetadata, FileInfo
from utils.config import ParserConfig

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parser for various document formats including DOCX, DOC, ODT, RTF.
    """
    
    def __init__(self, config: ParserConfig):
        """Initialize document parser with configuration."""
        self.config = config
        self.doc_config = config.document
        
        # Check available libraries
        self.available_libs = {}
        if HAS_PYTHON_DOCX:
            self.available_libs['docx'] = True
        if HAS_ODFPY:
            self.available_libs['odt'] = True
        if HAS_STRIPRTF:
            self.available_libs['rtf'] = True
        if HAS_OLEFILE:
            self.available_libs['doc'] = True
        
        logger.info(f"Document parser initialized with support for: {', '.join(self.available_libs.keys())}")
    
    def parse(self, file_path: Path, file_info: FileInfo) -> ParseResult:
        """
        Parse a document file based on its type.
        
        Args:
            file_path: Path to the document file
            file_info: File information from detector
            
        Returns:
            ParseResult with extracted content
        """
        result = ParseResult(
            file_path=str(file_path),
            file_type=file_info.mime_type
        )
        
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.docx':
                return self._parse_docx(file_path, result)
            elif extension == '.doc':
                return self._parse_doc(file_path, result)
            elif extension in ['.odt']:
                return self._parse_odt(file_path, result)
            elif extension == '.rtf':
                return self._parse_rtf(file_path, result)
            else:
                result.error = f"Unsupported document format: {extension}"
                return result
                
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}")
            result.error = str(e)
            return result
    
    def _parse_docx(self, file_path: Path, result: ParseResult) -> ParseResult:
        """Parse DOCX file using python-docx."""
        if not HAS_PYTHON_DOCX:
            result.error = "python-docx library not available"
            return result
        
        try:
            doc = DocxDocument(file_path)
            
            # Extract metadata
            result.metadata = self._extract_docx_metadata(doc)
            
            # Extract text content
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            
            result.content = '\n'.join(paragraphs)
            
            # Extract tables if enabled
            if self.doc_config.extract_tables:
                result.tables = self._extract_docx_tables(doc)
            
            # Extract images if enabled
            if self.doc_config.extract_images:
                result.images = self._extract_docx_images(doc)
            
            # Extract headers and footers if enabled
            if self.doc_config.include_headers_footers:
                header_footer_text = self._extract_docx_headers_footers(doc)
                if header_footer_text:
                    result.content = header_footer_text + '\n\n' + result.content
            
            result.success = True
            result.parser_used = 'python-docx'
            
        except Exception as e:
            result.error = f"DOCX parsing failed: {str(e)}"
            logger.error(f"DOCX parsing error: {str(e)}")
        
        return result
    
    def _parse_doc(self, file_path: Path, result: ParseResult) -> ParseResult:
        """Parse DOC file using available methods."""
        # DOC parsing is more complex and may require external tools
        # This is a basic implementation
        result.error = "DOC format parsing requires additional libraries (consider converting to DOCX)"
        result.parser_used = 'fallback'
        
        # Try basic text extraction if possible
        try:
            # This is a very basic approach - in production you might want to use
            # external tools like antiword, catdoc, or LibreOffice conversion
            with open(file_path, 'rb') as f:
                content = f.read()
                # Very basic text extraction (not reliable)
                text = content.decode('latin-1', errors='ignore')
                # Remove binary content (very rough approach)
                clean_text = ''.join(char for char in text if char.isprintable())
                if len(clean_text) > 100:  # Only if we got reasonable text
                    result.content = clean_text
                    result.success = True
                    result.parser_used = 'basic-binary'
        except Exception as e:
            result.error = f"DOC parsing failed: {str(e)}"
        
        return result
    
    def _parse_odt(self, file_path: Path, result: ParseResult) -> ParseResult:
        """Parse ODT file using odfpy."""
        if not HAS_ODFPY:
            result.error = "odfpy library not available for ODT parsing"
            return result
        
        try:
            doc = odf_load(file_path)
            
            # Extract text content
            paragraphs = []
            for paragraph in doc.getElementsByType(text.P):
                para_text = teletype.extractText(paragraph)
                if para_text.strip():
                    paragraphs.append(para_text)
            
            result.content = '\n'.join(paragraphs)
            
            # Extract basic metadata
            result.metadata = self._extract_odt_metadata(doc)
            
            # TODO: Add table and image extraction for ODT
            
            result.success = True
            result.parser_used = 'odfpy'
            
        except Exception as e:
            result.error = f"ODT parsing failed: {str(e)}"
            logger.error(f"ODT parsing error: {str(e)}")
        
        return result
    
    def _parse_rtf(self, file_path: Path, result: ParseResult) -> ParseResult:
        """Parse RTF file using striprtf."""
        if not HAS_STRIPRTF:
            result.error = "striprtf library not available for RTF parsing"
            return result
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            
            # Convert RTF to plain text
            plain_text = rtf_to_text(rtf_content)
            result.content = plain_text
            
            # Basic metadata
            result.metadata = DocumentMetadata()
            result.metadata.page_count = 1  # RTF doesn't have explicit pages
            
            result.success = True
            result.parser_used = 'striprtf'
            
        except Exception as e:
            result.error = f"RTF parsing failed: {str(e)}"
            logger.error(f"RTF parsing error: {str(e)}")
        
        return result
    
    def _extract_docx_metadata(self, doc) -> DocumentMetadata:
        """Extract metadata from DOCX document."""
        metadata = DocumentMetadata()
        
        try:
            core_props = doc.core_properties
            
            metadata.title = core_props.title
            metadata.author = core_props.author
            metadata.subject = core_props.subject
            metadata.creator = core_props.author
            metadata.creation_date = core_props.created
            metadata.modification_date = core_props.modified
            
            # Count words (approximate)
            word_count = 0
            for para in doc.paragraphs:
                word_count += len(para.text.split())
            metadata.word_count = word_count
            
            # Page count is harder to determine without rendering
            # This is an approximation
            char_count = sum(len(para.text) for para in doc.paragraphs)
            metadata.page_count = max(1, char_count // 2000)  # Rough estimate
            
        except Exception as e:
            logger.warning(f"Failed to extract DOCX metadata: {str(e)}")
        
        return metadata
    
    def _extract_docx_tables(self, doc) -> List[TableData]:
        """Extract tables from DOCX document."""
        tables = []
        
        try:
            for table_index, table in enumerate(doc.tables):
                if not table.rows:
                    continue
                
                # Extract headers (first row)
                headers = []
                if table.rows:
                    header_row = table.rows[0]
                    headers = [cell.text.strip() for cell in header_row.cells]
                
                # Extract data rows
                rows = []
                for row in table.rows[1:]:  # Skip header row
                    row_data = [cell.text.strip() for cell in row.cells]
                    rows.append(row_data)
                
                if headers and rows:
                    tables.append(TableData(
                        headers=headers,
                        rows=rows,
                        table_index=table_index
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to extract DOCX tables: {str(e)}")
        
        return tables
    
    def _extract_docx_images(self, doc) -> List[ImageInfo]:
        """Extract images from DOCX document."""
        images = []
        
        try:
            # Get document relationships
            rels = doc.part.rels
            
            image_index = 0
            for rel in rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_part = rel.target_part
                        images.append(ImageInfo(
                            image_index=image_index,
                            format=image_part.content_type.split('/')[-1],
                            size_bytes=len(image_part.blob)
                        ))
                        image_index += 1
                    except Exception as e:
                        logger.warning(f"Failed to process image {image_index}: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Failed to extract DOCX images: {str(e)}")
        
        return images
    
    def _extract_docx_headers_footers(self, doc) -> str:
        """Extract headers and footers from DOCX document."""
        header_footer_text = []
        
        try:
            # Extract headers
            for section in doc.sections:
                if section.header:
                    for para in section.header.paragraphs:
                        if para.text.strip():
                            header_footer_text.append(f"[HEADER] {para.text}")
                
                if section.footer:
                    for para in section.footer.paragraphs:
                        if para.text.strip():
                            header_footer_text.append(f"[FOOTER] {para.text}")
        
        except Exception as e:
            logger.warning(f"Failed to extract headers/footers: {str(e)}")
        
        return '\n'.join(header_footer_text)
    
    def _extract_odt_metadata(self, doc) -> DocumentMetadata:
        """Extract metadata from ODT document."""
        metadata = DocumentMetadata()
        
        try:
            # ODT metadata extraction is more complex
            # This is a basic implementation
            meta = doc.getElementsByType(text.P)
            if meta:
                char_count = sum(len(teletype.extractText(p)) for p in meta)
                metadata.page_count = max(1, char_count // 2000)
        
        except Exception as e:
            logger.warning(f"Failed to extract ODT metadata: {str(e)}")
        
        return metadata
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        formats = []
        if HAS_PYTHON_DOCX:
            formats.append('docx')
        if HAS_ODFPY:
            formats.append('odt')
        if HAS_STRIPRTF:
            formats.append('rtf')
        formats.append('doc')  # Basic support
        
        return formats
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about available document libraries."""
        return {
            'available_libraries': self.available_libs,
            'capabilities': {
                'docx': {
                    'text_extraction': HAS_PYTHON_DOCX,
                    'table_extraction': HAS_PYTHON_DOCX,
                    'image_extraction': HAS_PYTHON_DOCX,
                    'metadata_extraction': HAS_PYTHON_DOCX
                },
                'odt': {
                    'text_extraction': HAS_ODFPY,
                    'table_extraction': False,  # TODO: Implement
                    'image_extraction': False,  # TODO: Implement
                    'metadata_extraction': HAS_ODFPY
                },
                'rtf': {
                    'text_extraction': HAS_STRIPRTF,
                    'table_extraction': False,
                    'image_extraction': False,
                    'metadata_extraction': False
                },
                'doc': {
                    'text_extraction': True,  # Basic
                    'table_extraction': False,
                    'image_extraction': False,
                    'metadata_extraction': False
                }
            }
        }