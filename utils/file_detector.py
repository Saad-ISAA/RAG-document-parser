# utils/file_detector.py
"""
File type detection utilities for the Universal Document Parser - Enhanced Edition.
"""

import mimetypes
from pathlib import Path
from typing import Optional
import logging

# Try to import python-magic for better MIME type detection
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    magic = None

from models.parse_result import FileInfo

logger = logging.getLogger(__name__)


class FileTypeDetector:
    """
    Detects file types and categorizes them for appropriate parser selection.
    Uses python-magic when available, falls back to extension-based detection.
    Enhanced for Regional requirements with PowerPoint and comprehensive format support.
    """
    
    # File type mappings - Enhanced for Regional requirements
    MIME_TO_CATEGORY = {
        # PDF files
        'application/pdf': 'pdf',
        
        # Document files
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',  # .docx
        'application/msword': 'document',  # .doc
        'application/vnd.oasis.opendocument.text': 'document',  # .odt
        'application/rtf': 'document',  # .rtf
        'text/rtf': 'document',  # .rtf
        
        # Presentation files - NEW
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'powerpoint',  # .pptx
        'application/vnd.ms-powerpoint': 'powerpoint',  # .ppt
        'application/vnd.oasis.opendocument.presentation': 'powerpoint',  # .odp
        
        # Spreadsheet files
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'spreadsheet',  # .xlsx
        'application/vnd.ms-excel': 'spreadsheet',  # .xls
        'application/vnd.oasis.opendocument.spreadsheet': 'spreadsheet',  # .ods
        'text/csv': 'spreadsheet',  # .csv
        
        # Text files
        'text/plain': 'text',
        'text/html': 'text',
        'text/xml': 'text',
        'application/xml': 'text',
        'text/markdown': 'text',
        'application/json': 'text',
        'application/javascript': 'text',
        'text/css': 'text',
        
        # Image files
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/gif': 'image',
        'image/bmp': 'image',
        'image/tiff': 'image',
        'image/webp': 'image',
        'image/svg+xml': 'image',
    }
    
    EXTENSION_TO_CATEGORY = {
        # PDF
        '.pdf': 'pdf',
        
        # Documents
        '.docx': 'document',
        '.doc': 'document',
        '.odt': 'document',
        '.rtf': 'document',
        
        # Presentations - NEW
        '.pptx': 'powerpoint',
        '.ppt': 'powerpoint',
        '.odp': 'powerpoint',
        '.key': 'powerpoint',  # Apple Keynote
        
        # Spreadsheets
        '.xlsx': 'spreadsheet',
        '.xls': 'spreadsheet',
        '.ods': 'spreadsheet',
        '.csv': 'spreadsheet',
        '.numbers': 'spreadsheet',  # Apple Numbers
        
        # Text files
        '.txt': 'text',
        '.md': 'text',
        '.markdown': 'text',
        '.html': 'text',
        '.htm': 'text',
        '.xml': 'text',
        '.json': 'text',
        '.js': 'text',
        '.css': 'text',
        '.py': 'text',
        '.java': 'text',
        '.cpp': 'text',
        '.c': 'text',
        '.h': 'text',
        '.yaml': 'text',
        '.yml': 'text',
        '.ini': 'text',
        '.cfg': 'text',
        '.conf': 'text',
        '.log': 'text',
        
        # Images
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.gif': 'image',
        '.bmp': 'image',
        '.tiff': 'image',
        '.tif': 'image',
        '.webp': 'image',
        '.svg': 'image',
        
        # Additional formats that may need conversion
        '.pages': 'document',  # Apple Pages
        '.wpd': 'document',    # WordPerfect
        '.wps': 'document',    # Microsoft Works
    }
    
    def __init__(self):
        """Initialize the file type detector."""
        self.magic_mime = None
        self.magic_mime_encoding = None
        
        if HAS_MAGIC:
            try:
                self.magic_mime = magic.Magic(mime=True)
                self.magic_mime_encoding = magic.Magic(mime_encoding=True)
                logger.info("python-magic initialized for enhanced file type detection")
            except Exception as e:
                logger.warning(f"Failed to initialize python-magic: {e}")
                self.magic_mime = None
                self.magic_mime_encoding = None
        else:
            logger.info("python-magic not available, using extension-based detection")
    
    def detect_file_type(self, file_path: Path) -> FileInfo:
        """
        Detect the file type and return FileInfo object.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileInfo object with detected type information
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Try magic-based detection first
        mime_type = self._detect_mime_type(file_path)
        encoding = self._detect_encoding(file_path)
        
        # Determine category
        category = self._get_category_from_mime(mime_type)
        confidence = 1.0
        
        # Fall back to extension-based detection if needed
        if category == 'unknown':
            category = self._get_category_from_extension(extension)
            confidence = 0.8  # Lower confidence for extension-based detection
            
            # If still unknown, try to guess MIME type from extension
            if category == 'unknown' and not mime_type:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type:
                    category = self._get_category_from_mime(mime_type)
                    confidence = 0.6
        
        # Final fallback - treat as text if it has a known text extension
        if category == 'unknown' and extension in ['.txt', '.log', '.cfg', '.conf']:
            category = 'text'
            mime_type = 'text/plain'
            confidence = 0.5
        
        logger.debug(f"Detected {file_path.name}: {mime_type} -> {category} (confidence: {confidence})")
        
        return FileInfo(
            mime_type=mime_type or 'application/octet-stream',
            category=category,
            extension=extension,
            encoding=encoding,
            confidence=confidence
        )
    
    def _detect_mime_type(self, file_path: Path) -> Optional[str]:
        """Detect MIME type using python-magic or mimetypes."""
        if self.magic_mime and file_path.exists():
            try:
                return self.magic_mime.from_file(str(file_path))
            except Exception as e:
                logger.debug(f"Magic MIME detection failed for {file_path}: {e}")
        
        # Fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type
    
    def _detect_encoding(self, file_path: Path) -> Optional[str]:
        """Detect file encoding using python-magic."""
        if self.magic_mime_encoding and file_path.exists():
            try:
                return self.magic_mime_encoding.from_file(str(file_path))
            except Exception as e:
                logger.debug(f"Magic encoding detection failed for {file_path}: {e}")
        
        return None
    
    def _get_category_from_mime(self, mime_type: str) -> str:
        """Get file category from MIME type."""
        if not mime_type:
            return 'unknown'
        
        # Direct mapping
        if mime_type in self.MIME_TO_CATEGORY:
            return self.MIME_TO_CATEGORY[mime_type]
        
        # Pattern matching for broader categories
        if mime_type.startswith('text/'):
            return 'text'
        elif mime_type.startswith('image/'):
            return 'image'
        elif 'pdf' in mime_type.lower():
            return 'pdf'
        elif any(word in mime_type.lower() for word in ['word', 'document', 'office']):
            return 'document'
        elif any(word in mime_type.lower() for word in ['excel', 'spreadsheet', 'sheet']):
            return 'spreadsheet'
        elif any(word in mime_type.lower() for word in ['powerpoint', 'presentation']):
            return 'powerpoint'
        
        return 'unknown'
    
    def _get_category_from_extension(self, extension: str) -> str:
        """Get file category from file extension."""
        return self.EXTENSION_TO_CATEGORY.get(extension.lower(), 'unknown')
    
    def is_supported(self, file_path: Path) -> bool:
        """Check if the file type is supported for parsing."""
        file_info = self.detect_file_type(file_path)
        return file_info.category != 'unknown'
    
    def get_supported_extensions(self) -> list:
        """Get list of all supported file extensions."""
        return list(self.EXTENSION_TO_CATEGORY.keys())
    
    def get_supported_mime_types(self) -> list:
        """Get list of all supported MIME types."""
        return list(self.MIME_TO_CATEGORY.keys())
    
    def get_category_info(self, category: str) -> dict:
        """Get information about a specific category."""
        category_info = {
            'pdf': {
                'description': 'Portable Document Format files',
                'extensions': ['.pdf'],
                'mime_types': ['application/pdf']
            },
            'document': {
                'description': 'Word processing documents',
                'extensions': ['.docx', '.doc', '.odt', '.rtf', '.pages', '.wpd', '.wps'],
                'mime_types': [
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/msword',
                    'application/vnd.oasis.opendocument.text',
                    'application/rtf'
                ]
            },
            'powerpoint': {  # NEW
                'description': 'Presentation files',
                'extensions': ['.pptx', '.ppt', '.odp', '.key'],
                'mime_types': [
                    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    'application/vnd.ms-powerpoint',
                    'application/vnd.oasis.opendocument.presentation'
                ]
            },
            'spreadsheet': {
                'description': 'Spreadsheet and tabular data files',
                'extensions': ['.xlsx', '.xls', '.ods', '.csv', '.numbers'],
                'mime_types': [
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'application/vnd.ms-excel',
                    'application/vnd.oasis.opendocument.spreadsheet',
                    'text/csv'
                ]
            },
            'text': {
                'description': 'Plain text and markup files',
                'extensions': ['.txt', '.md', '.html', '.xml', '.json'],
                'mime_types': ['text/plain', 'text/html', 'text/xml', 'application/json']
            },
            'image': {
                'description': 'Image files (with OCR support for Arabic/English)',
                'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
                'mime_types': ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff']
            }
        }
        
        return category_info.get(category, {})
    
    def analyze_directory(self, directory_path: Path, pattern: str = "*") -> dict:
        """Analyze all files in a directory and return type statistics."""
        directory = Path(directory_path)
        if not directory.exists():
            return {}
        
        files = list(directory.rglob(pattern))
        
        stats = {
            'total_files': len(files),
            'by_category': {},
            'by_extension': {},
            'unsupported': [],
            'errors': []
        }
        
        for file_path in files:
            try:
                if file_path.is_file():
                    file_info = self.detect_file_type(file_path)
                    
                    # Count by category
                    stats['by_category'][file_info.category] = stats['by_category'].get(file_info.category, 0) + 1
                    
                    # Count by extension
                    ext = file_path.suffix.lower() or 'no_extension'
                    stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
                    
                    # Track unsupported files
                    if file_info.category == 'unknown':
                        stats['unsupported'].append(str(file_path))
            
            except Exception as e:
                stats['errors'].append({'file': str(file_path), 'error': str(e)})
        
        return stats