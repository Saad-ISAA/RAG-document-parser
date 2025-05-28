# utils/config.py
"""
Configuration management for the Universal Document Parser.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class PDFConfig:
    """Configuration for PDF parsing."""
    prefer_pdfplumber: bool = True
    extract_tables: bool = True
    extract_images: bool = True
    table_settings: Dict[str, Any] = None
    pymupdf_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.table_settings is None:
            self.table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3
            }
        
        if self.pymupdf_settings is None:
            self.pymupdf_settings = {
                "get_text_flags": 0,  # Default text extraction flags
                "extract_images": True,
                "image_format": "png"
            }


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    enabled: bool = False
    languages: List[str] = None
    engine: str = "easyocr"  # "easyocr", "tesseract", "paddleocr"
    confidence_threshold: float = 0.3  # Lowered from 0.5 to catch more text
    preprocessing: bool = True
    
    # Tesseract specific
    tesseract_config: str = "--psm 6"
    tesseract_path: Optional[str] = None
    
    # EasyOCR specific
    easyocr_gpu: bool = False
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "ar"]  # English and Arabic by default


@dataclass
class ImageConfig:
    """Configuration for image processing."""
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: List[str] = None
    auto_rotate: bool = True
    enhance_quality: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"
            ]


@dataclass
class DocumentConfig:
    """Configuration for document processing (DOCX, etc.)."""
    extract_tables: bool = True
    extract_images: bool = True
    preserve_formatting: bool = False
    include_headers_footers: bool = True


@dataclass
class SpreadsheetConfig:
    """Configuration for spreadsheet processing."""
    max_rows: int = 10000
    max_columns: int = 100
    read_all_sheets: bool = False
    skip_empty_rows: bool = True
    skip_empty_columns: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 512
    timeout_seconds: int = 300
    chunk_size: int = 1024 * 1024  # 1MB chunks for large files
    
    def __post_init__(self):
        # Adjust max_workers based on available CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if self.max_workers > cpu_count:
            self.max_workers = max(1, cpu_count - 1)


@dataclass
class ParserConfig:
    """Main configuration class for the Universal Document Parser."""
    
    # Component configurations
    pdf: PDFConfig = None
    ocr: OCRConfig = None
    image: ImageConfig = None
    document: DocumentConfig = None
    spreadsheet: SpreadsheetConfig = None
    performance: PerformanceConfig = None
    
    # Global settings
    enable_ocr: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    verbose_logging: bool = False
    
    # File filtering
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = None
    blocked_extensions: List[str] = None
    
    # Output settings
    preserve_whitespace: bool = False
    normalize_unicode: bool = True
    remove_empty_lines: bool = True
    
    def __post_init__(self):
        # Initialize sub-configurations
        if self.pdf is None:
            self.pdf = PDFConfig()
        if self.ocr is None:
            self.ocr = OCRConfig(enabled=self.enable_ocr)
        else:
            # Update OCR enabled status if main config changed
            self.ocr.enabled = self.enable_ocr
        if self.image is None:
            self.image = ImageConfig()
        if self.document is None:
            self.document = DocumentConfig()
        if self.spreadsheet is None:
            self.spreadsheet = SpreadsheetConfig()
        if self.performance is None:
            self.performance = PerformanceConfig(
                parallel_processing=self.parallel_processing,
                max_workers=self.max_workers
            )
        
        # Set default allowed extensions
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                # PDF
                "pdf",
                # Documents
                "docx", "doc", "odt", "rtf",
                # Spreadsheets
                "xlsx", "xls", "ods", "csv",
                # Text
                "txt", "md", "html", "htm", "xml",
                # Images
                "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp",
                # Archives (if needed)
                # "zip", "tar", "gz"
            ]
        
        if self.blocked_extensions is None:
            self.blocked_extensions = [
                "exe", "dll", "so", "dylib", "bin", "dat"
            ]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ParserConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update main config
        for key, value in config_dict.items():
            if hasattr(config, key) and not key.startswith('_'):
                # Handle nested configurations
                if key in ['pdf', 'ocr', 'image', 'document', 'spreadsheet', 'performance']:
                    if isinstance(value, dict):
                        current_config = getattr(config, key)
                        for sub_key, sub_value in value.items():
                            if hasattr(current_config, sub_key):
                                setattr(current_config, sub_key, sub_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ParserConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> 'ParserConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Map environment variables to config
        env_mapping = {
            'PARSER_ENABLE_OCR': ('enable_ocr', bool),
            'PARSER_PARALLEL': ('parallel_processing', bool),
            'PARSER_MAX_WORKERS': ('max_workers', int),
            'PARSER_MAX_FILE_SIZE': ('max_file_size', int),
            'PARSER_VERBOSE': ('verbose_logging', bool),
            'OCR_LANGUAGES': ('ocr.languages', list),
            'OCR_ENGINE': ('ocr.engine', str),
            'PDF_EXTRACT_TABLES': ('pdf.extract_tables', bool),
            'PDF_EXTRACT_IMAGES': ('pdf.extract_images', bool),
        }
        
        for env_var, (attr_path, attr_type) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert value to appropriate type
                if attr_type == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif attr_type == int:
                    value = int(value)
                elif attr_type == list:
                    value = [item.strip() for item in value.split(',')]
                
                # Set nested attribute
                obj = config
                attrs = attr_path.split('.')
                for attr in attrs[:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, attrs[-1], value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'pdf': asdict(self.pdf),
            'ocr': asdict(self.ocr),
            'image': asdict(self.image),
            'document': asdict(self.document),
            'spreadsheet': asdict(self.spreadsheet),
            'performance': asdict(self.performance),
            'enable_ocr': self.enable_ocr,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'verbose_logging': self.verbose_logging,
            'max_file_size': self.max_file_size,
            'allowed_extensions': self.allowed_extensions,
            'blocked_extensions': self.blocked_extensions,
            'preserve_whitespace': self.preserve_whitespace,
            'normalize_unicode': self.normalize_unicode,
            'remove_empty_lines': self.remove_empty_lines
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate performance settings
        if self.performance.max_workers < 1:
            issues.append("max_workers must be at least 1")
        
        if self.max_file_size < 1024:  # 1KB minimum
            issues.append("max_file_size too small (minimum 1KB)")
        
        # Validate OCR settings
        if self.enable_ocr and not self.ocr.languages:
            issues.append("OCR enabled but no languages specified")
        
        # Validate file extensions
        if not self.allowed_extensions:
            issues.append("No allowed file extensions specified")
        
        return issues
    
    def is_file_allowed(self, file_path: str) -> bool:
        """Check if a file is allowed based on extension and size."""
        file_path = Path(file_path)
        
        # Check extension
        extension = file_path.suffix.lower().lstrip('.')
        if extension in self.blocked_extensions:
            return False
        
        if self.allowed_extensions and extension not in self.allowed_extensions:
            return False
        
        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
        
        return True