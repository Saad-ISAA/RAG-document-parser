# parsers/image_parser.py
"""
Image parser with OCR capabilities for extracting text from images.
Supports multiple OCR engines with automatic fallback.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import io
import tempfile

# Import image processing libraries
try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

# Import OCR libraries with fallbacks
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    easyocr = None

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None

from models.parse_result import ParseResult, ImageInfo, DocumentMetadata, FileInfo
from utils.config import ParserConfig

logger = logging.getLogger(__name__)


class ImageParser:
    """
    Image parser with OCR capabilities supporting multiple engines.
    """
    
    def __init__(self, config: ParserConfig):
        """Initialize image parser with configuration."""
        self.config = config
        self.image_config = config.image
        self.ocr_config = config.ocr
        
        # Initialize OCR readers
        self.easyocr_reader = None
        self.ocr_engines = []
        
        if HAS_EASYOCR and self.ocr_config.enabled:
            try:
                self.easyocr_reader = easyocr.Reader(
                    self.ocr_config.languages,
                    gpu=self.ocr_config.easyocr_gpu
                )
                self.ocr_engines.append('easyocr')
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {str(e)}")
        
        if HAS_TESSERACT and self.ocr_config.enabled:
            try:
                # Test tesseract availability
                pytesseract.get_tesseract_version()
                self.ocr_engines.append('tesseract')
                logger.info("Tesseract initialized")
            except Exception as e:
                logger.warning(f"Tesseract not available: {str(e)}")
        
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for image processing")
        
        logger.info(f"Image parser initialized with OCR engines: {', '.join(self.ocr_engines)}")
    
    def parse(self, file_path: Path, file_info: FileInfo) -> ParseResult:
        """
        Parse an image file and extract text using OCR.
        
        Args:
            file_path: Path to the image file
            file_info: File information from detector
            
        Returns:
            ParseResult with extracted text content
        """
        result = ParseResult(
            file_path=str(file_path),
            file_type=file_info.mime_type
        )
        
        try:
            # Load and validate image
            image = self._load_image(file_path)
            if image is None:
                result.error = "Failed to load image"
                return result
            
            # Create image info
            image_info = ImageInfo(
                image_index=0,
                width=image.width,
                height=image.height,
                format=image.format,
                size_bytes=file_path.stat().st_size
            )
            
            # Extract text using OCR if enabled
            if self.ocr_config.enabled and self.ocr_engines:
                extracted_text = self._extract_text_ocr(image, file_path)
                result.content = extracted_text
                image_info.extracted_text = extracted_text
                
                if extracted_text:
                    result.success = True
                    result.parser_used = self.ocr_engines[0]  # Primary engine used
            else:
                result.error = "OCR not enabled or no OCR engines available"
            
            # Add image to results
            result.images = [image_info]
            
            # Basic metadata
            result.metadata = self._create_image_metadata(image, file_path)
            
        except Exception as e:
            logger.error(f"Error parsing image {file_path}: {str(e)}")
            result.error = str(e)
        
        return result
    
    def _load_image(self, file_path: Path) -> Optional[Image.Image]:
        """Load and validate image file."""
        try:
            # Check file size
            if file_path.stat().st_size > self.image_config.max_image_size:
                logger.warning(f"Image {file_path.name} exceeds size limit")
                return None
            
            # Load image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Auto-rotate if enabled and EXIF data is available
            if self.image_config.auto_rotate:
                image = self._auto_rotate_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {str(e)}")
            return None
    
    def _auto_rotate_image(self, image: Image.Image) -> Image.Image:
        """Auto-rotate image based on EXIF orientation."""
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # Orientation tag
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        except Exception as e:
            logger.debug(f"Auto-rotation failed: {str(e)}")
        
        return image
    
    def _extract_text_ocr(self, image: Image.Image, file_path: Path) -> str:
        """Extract text from image using available OCR engines."""
        # Try OCR engines in order of preference
        for engine in self.ocr_engines:
            try:
                if engine == 'easyocr':
                    text = self._extract_text_easyocr(image)
                elif engine == 'tesseract':
                    text = self._extract_text_tesseract(image)
                else:
                    continue
                
                if text and len(text.strip()) > 0:
                    logger.debug(f"Successfully extracted text using {engine}")
                    return text
                    
            except Exception as e:
                logger.warning(f"OCR engine {engine} failed for {file_path.name}: {str(e)}")
                continue
        
        logger.warning(f"All OCR engines failed for {file_path.name}")
        return ""
    
    def _extract_text_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR."""
        if not self.easyocr_reader:
            raise Exception("EasyOCR not initialized")
        
        # Preprocess image if enabled
        if self.ocr_config.preprocessing:
            image = self._preprocess_image(image)
        
        # Convert PIL image to numpy array for EasyOCR
        image_array = np.array(image)
        
        # Extract text
        results = self.easyocr_reader.readtext(image_array)
        
        # Filter by confidence and combine text
        text_parts = []
        for bbox, text, confidence in results:
            if confidence >= self.ocr_config.confidence_threshold:
                text_parts.append(text)
        
        return '\n'.join(text_parts)
    
    def _extract_text_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract."""
        if not HAS_TESSERACT:
            raise Exception("Tesseract not available")
        
        # Preprocess image if enabled
        if self.ocr_config.preprocessing:
            image = self._preprocess_image(image)
        
        # Set tesseract path if specified
        if self.ocr_config.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.ocr_config.tesseract_path
        
        # Configure language
        lang = '+'.join(self.ocr_config.languages)
        
        # Extract text
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=self.ocr_config.tesseract_config
        )
        
        return text.strip()
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy."""
        try:
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast if image quality enhancement is enabled
            if self.image_config.enhance_quality:
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)
            
            # Additional OpenCV preprocessing if available
            if HAS_OPENCV:
                image = self._advanced_preprocessing(image)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
        
        return image
    
    def _advanced_preprocessing(self, image: Image.Image) -> Image.Image:
        """Advanced preprocessing using OpenCV."""
        try:
            # Convert PIL to OpenCV format
            cv_image = np.array(image)
            
            # Gaussian blur to reduce noise
            cv_image = cv2.GaussianBlur(cv_image, (1, 1), 0)
            
            # Threshold to get better contrast
            _, cv_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            image = Image.fromarray(cv_image)
            
        except Exception as e:
            logger.debug(f"Advanced preprocessing failed: {str(e)}")
        
        return image
    
    def _create_image_metadata(self, image: Image.Image, file_path: Path) -> DocumentMetadata:
        """Create metadata for image file."""
        metadata = DocumentMetadata()
        
        try:
            # Basic image info
            metadata.page_count = 1
            
            # Try to extract EXIF data
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    # Camera make/model
                    make = exif.get(271)  # Make
                    model = exif.get(272)  # Model
                    if make or model:
                        metadata.creator = f"{make or ''} {model or ''}".strip()
                    
                    # Date taken
                    date_taken = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
                    if date_taken:
                        try:
                            from datetime import datetime
                            metadata.creation_date = datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S")
                        except:
                            pass
            
            # File information
            stat = file_path.stat()
            metadata.modification_date = datetime.fromtimestamp(stat.st_mtime)
            
            # Custom properties
            metadata.custom_properties = {
                'image_width': image.width,
                'image_height': image.height,
                'image_format': image.format,
                'image_mode': image.mode,
                'file_size_bytes': stat.st_size
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract image metadata: {str(e)}")
        
        return metadata
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return self.image_config.supported_formats
    
    def get_ocr_info(self) -> Dict[str, Any]:
        """Get information about OCR capabilities."""
        return {
            'ocr_enabled': self.ocr_config.enabled,
            'available_engines': self.ocr_engines,
            'supported_languages': self.ocr_config.languages,
            'confidence_threshold': self.ocr_config.confidence_threshold,
            'preprocessing_enabled': self.ocr_config.preprocessing
        }
    
    def test_ocr_engine(self, engine_name: str) -> Dict[str, Any]:
        """Test specific OCR engine with a simple image."""
        if engine_name not in self.ocr_engines:
            return {'success': False, 'error': f'Engine {engine_name} not available'}
        
        try:
            # Create a simple test image with text
            test_image = Image.new('RGB', (200, 50), color='white')
            
            # You would normally draw text here, but for simplicity:
            # This is just a basic test of engine availability
            
            if engine_name == 'easyocr':
                # Test array conversion
                image_array = np.array(test_image)
                return {'success': True, 'engine': 'easyocr', 'ready': True}
            elif engine_name == 'tesseract':
                # Test tesseract
                pytesseract.image_to_string(test_image)
                return {'success': True, 'engine': 'tesseract', 'ready': True}
                
        except Exception as e:
            return {'success': False, 'engine': engine_name, 'error': str(e)}
        
        return {'success': False, 'error': 'Unknown engine'}
    
    def batch_process_images(self, image_paths: List[Path]) -> List[ParseResult]:
        """Process multiple images in batch."""
        results = []
        
        for image_path in image_paths:
            try:
                # Detect file type
                from utils.file_detector import FileTypeDetector
                detector = FileTypeDetector()
                file_info = detector.detect_file_type(image_path)
                
                if file_info.category != 'image':
                    continue
                
                # Parse image
                result = self.parse(image_path, file_info)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append(ParseResult(
                    success=False,
                    error=str(e),
                    file_path=str(image_path)
                ))
        
        return results