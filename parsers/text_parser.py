# parsers/text_parser.py
"""
Text and markup parser for plain text, HTML, XML, Markdown, and other text-based formats.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import re
import chardet

# Import text processing libraries
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None

try:
    import markdown
    from markdown.extensions import tables, codehilite
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    markdown = None

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False
    ET = None

from models.parse_result import ParseResult, DocumentMetadata, FileInfo
from utils.config import ParserConfig

logger = logging.getLogger(__name__)


class TextParser:
    """
    Parser for text-based formats including plain text, HTML, XML, Markdown, JSON, etc.
    """
    
    def __init__(self, config: ParserConfig):
        """Initialize text parser with configuration."""
        self.config = config
        
        # Text processing settings
        self.preserve_whitespace = config.preserve_whitespace
        self.normalize_unicode = config.normalize_unicode
        self.remove_empty_lines = config.remove_empty_lines
        
        logger.info("Text parser initialized")
    
    def parse(self, file_path: Path, file_info: FileInfo) -> ParseResult:
        """
        Parse a text-based file.
        
        Args:
            file_path: Path to the text file
            file_info: File information from detector
            
        Returns:
            ParseResult with extracted content
        """
        result = ParseResult(
            file_path=str(file_path),
            file_type=file_info.mime_type
        )
        
        try:
            # Read file with proper encoding detection
            text_content = self._read_text_file(file_path, file_info.encoding)
            
            if text_content is None:
                result.error = "Failed to read text file or detect encoding"
                return result
            
            # Parse based on file type
            extension = file_path.suffix.lower()
            
            if extension in ['.html', '.htm']:
                result = self._parse_html(text_content, result)
            elif extension == '.xml':
                result = self._parse_xml(text_content, result)
            elif extension in ['.md', '.markdown']:
                result = self._parse_markdown(text_content, result)
            elif extension == '.json':
                result = self._parse_json(text_content, result)
            elif extension in ['.css', '.js', '.py', '.java', '.cpp', '.c', '.h']:
                result = self._parse_code(text_content, result, extension)
            else:
                result = self._parse_plain_text(text_content, result)
            
            # Post-process text if needed
            if result.content:
                result.content = self._post_process_text(result.content)
            
            # Create metadata
            result.metadata = self._create_text_metadata(text_content, file_path)
            
            result.success = True
            result.parser_used = 'text_parser'
            
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {str(e)}")
            result.error = str(e)
        
        return result
    
    def _read_text_file(self, file_path: Path, detected_encoding: Optional[str] = None) -> Optional[str]:
        """Read text file with proper encoding detection."""
        encodings_to_try = []
        
        # Add detected encoding first if available
        if detected_encoding:
            encodings_to_try.append(detected_encoding)
        
        # Add common encodings
        encodings_to_try.extend(['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1'])
        
        # Try chardet detection if available
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            detected = chardet.detect(raw_data)
            if detected['encoding'] and detected['confidence'] > 0.7:
                if detected['encoding'] not in encodings_to_try:
                    encodings_to_try.insert(0, detected['encoding'])
        except Exception as e:
            logger.debug(f"Chardet detection failed: {str(e)}")
        
        # Try each encoding
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                    logger.debug(f"Successfully read {file_path.name} with encoding: {encoding}")
                    return content
            except Exception as e:
                logger.debug(f"Failed to read with encoding {encoding}: {str(e)}")
                continue
        
        logger.error(f"Failed to read {file_path} with any encoding")
        return None
    
    def _parse_html(self, content: str, result: ParseResult) -> ParseResult:
        """Parse HTML content."""
        if not HAS_BS4:
            # Fallback: strip HTML tags with regex
            result.content = self._strip_html_tags(content)
            result.parser_used = 'regex_html'
            return result
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                result.metadata = DocumentMetadata(title=title_tag.get_text().strip())
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            result.content = text
            result.parser_used = 'beautifulsoup'
            
        except Exception as e:
            logger.warning(f"BeautifulSoup parsing failed, falling back to regex: {str(e)}")
            result.content = self._strip_html_tags(content)
            result.parser_used = 'regex_html_fallback'
        
        return result
    
    def _parse_xml(self, content: str, result: ParseResult) -> ParseResult:
        """Parse XML content."""
        if not HAS_XML:
            result.content = content
            result.parser_used = 'raw_xml'
            return result
        
        try:
            root = ET.fromstring(content)
            
            # Extract text content recursively
            def extract_text(element):
                text = element.text or ""
                for child in element:
                    text += extract_text(child)
                    if child.tail:
                        text += child.tail
                return text
            
            result.content = extract_text(root)
            result.parser_used = 'xml_parser'
            
            # Try to extract metadata from common XML elements
            title_elem = root.find('.//title') or root.find('.//Title')
            if title_elem is not None and title_elem.text:
                result.metadata = DocumentMetadata(title=title_elem.text.strip())
            
        except Exception as e:
            logger.warning(f"XML parsing failed, using raw content: {str(e)}")
            result.content = content
            result.parser_used = 'raw_xml_fallback'
        
        return result
    
    def _parse_markdown(self, content: str, result: ParseResult) -> ParseResult:
        """Parse Markdown content."""
        if not HAS_MARKDOWN:
            result.content = content
            result.parser_used = 'raw_markdown'
            return result
        
        try:
            # Convert Markdown to HTML first
            md = markdown.Markdown(extensions=['tables', 'codehilite', 'meta'])
            html = md.convert(content)
            
            # Extract metadata if available
            if hasattr(md, 'Meta') and md.Meta:
                title = md.Meta.get('title', [])
                if title:
                    result.metadata = DocumentMetadata(title=title[0])
            
            # Convert HTML to plain text
            if HAS_BS4:
                soup = BeautifulSoup(html, 'html.parser')
                result.content = soup.get_text()
            else:
                result.content = self._strip_html_tags(html)
            
            result.parser_used = 'markdown'
            
        except Exception as e:
            logger.warning(f"Markdown parsing failed, using raw content: {str(e)}")
            result.content = content
            result.parser_used = 'raw_markdown_fallback'
        
        return result
    
    def _parse_json(self, content: str, result: ParseResult) -> ParseResult:
        """Parse JSON content."""
        try:
            import json
            data = json.loads(content)
            
            # Convert JSON to readable text
            if isinstance(data, dict):
                text_parts = []
                self._json_to_text(data, text_parts)
                result.content = '\n'.join(text_parts)
            elif isinstance(data, list):
                result.content = '\n'.join(str(item) for item in data)
            else:
                result.content = str(data)
            
            result.parser_used = 'json_parser'
            
        except Exception as e:
            logger.warning(f"JSON parsing failed, using raw content: {str(e)}")
            result.content = content
            result.parser_used = 'raw_json'
        
        return result
    
    def _parse_code(self, content: str, result: ParseResult, extension: str) -> ParseResult:
        """Parse code files."""
        result.content = content
        result.parser_used = f'code_parser_{extension[1:]}'  # Remove the dot
        
        # Extract basic metadata for code files
        lines = content.split('\n')
        
        # Look for common comment patterns for metadata
        metadata = DocumentMetadata()
        
        # Count non-empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        metadata.word_count = len(non_empty_lines)
        
        # Look for title in comments (first few lines)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['#', '//', '/*', '--', '%']):
                # Remove comment markers
                clean_line = re.sub(r'^[#/\*\-% ]+', '', line).strip()
                if clean_line and len(clean_line) < 100 and not metadata.title:
                    metadata.title = clean_line
                    break
        
        result.metadata = metadata
        return result
    
    def _parse_plain_text(self, content: str, result: ParseResult) -> ParseResult:
        """Parse plain text content."""
        result.content = content
        result.parser_used = 'plain_text'
        return result
    
    def _strip_html_tags(self, html_content: str) -> str:
        """Strip HTML tags using regex (fallback method)."""
        # Remove scripts and styles
        html_content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', html_content)
        
        # Clean up whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def _json_to_text(self, obj: Any, text_parts: List[str], prefix: str = "") -> None:
        """Convert JSON object to readable text recursively."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(f"{prefix}{key}:")
                    self._json_to_text(value, text_parts, prefix + "  ")
                else:
                    text_parts.append(f"{prefix}{key}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    text_parts.append(f"{prefix}[{i}]:")
                    self._json_to_text(item, text_parts, prefix + "  ")
                else:
                    text_parts.append(f"{prefix}[{i}]: {item}")
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text based on configuration."""
        if not text:
            return text
        
        # Normalize unicode if enabled
        if self.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        
        # Remove empty lines if enabled
        if self.remove_empty_lines:
            lines = text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            text = '\n'.join(non_empty_lines)
        
        # Preserve or normalize whitespace
        if not self.preserve_whitespace:
            # Normalize multiple spaces to single space
            text = re.sub(r' +', ' ', text)
            # Normalize multiple newlines
            text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _create_text_metadata(self, content: str, file_path: Path) -> DocumentMetadata:
        """Create metadata for text file."""
        metadata = DocumentMetadata()
        
        lines = content.split('\n')
        words = content.split()
        
        metadata.word_count = len(words)
        metadata.page_count = max(1, len(lines) // 50)  # Approximate pages
        
        # Try to extract title from first line if it looks like a title
        if lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) < 100:
                # Check if it looks like a title (no punctuation at end, reasonable length)
                if not first_line.endswith('.') and len(first_line.split()) <= 10:
                    metadata.title = first_line
        
        # File stats
        try:
            stat = file_path.stat()
            from datetime import datetime
            metadata.modification_date = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            pass
        
        return metadata
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported text formats."""
        return [
            'txt', 'md', 'markdown', 'html', 'htm', 'xml',
            'json', 'js', 'css', 'py', 'java', 'cpp', 'c', 'h',
            'yaml', 'yml', 'ini', 'cfg', 'conf', 'log'
        ]
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about available text processing libraries."""
        return {
            'beautifulsoup': HAS_BS4,
            'markdown': HAS_MARKDOWN,
            'xml': HAS_XML,
            'capabilities': {
                'html_parsing': HAS_BS4,
                'markdown_conversion': HAS_MARKDOWN,
                'xml_parsing': HAS_XML,
                'encoding_detection': True,
                'json_parsing': True,
                'code_parsing': True
            }
        }