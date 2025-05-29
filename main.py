#!/usr/bin/env python3
"""
Universal Document Parser - Main Script - Arabic Enhanced Edition
High-performance document parsing service with automatic library selection
Enhanced with PowerPoint support, Arabic text processing, and conversion fallback
"""

import os
import sys
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import parser modules
from parsers.pdf_parser import PDFParser
from parsers.document_parser import DocumentParser
from parsers.powerpoint_parser import PowerPointParser  # NEW
from parsers.image_parser import ImageParser
from parsers.text_parser import TextParser
from parsers.spreadsheet_parser import SpreadsheetParser
from utils.file_detector import FileTypeDetector
from utils.config import ParserConfig
from models.parse_result import ParseResult

logger = logging.getLogger(__name__)


class UniversalDocumentParser:
    """
    High-performance universal document parser that automatically
    selects the best parsing library based on file type.
    
    Arabic Enhanced Edition with:
    - PowerPoint support (PPT/PPTX)
    - Enhanced DOC parsing with conversion fallback
    - Arabic text processing
    - Comprehensive format support
    """
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """Initialize the universal parser with configuration."""
        self.config = config or ParserConfig()
        self.file_detector = FileTypeDetector()
        
        # Initialize specialized parsers
        self.parsers = {
            'pdf': PDFParser(self.config),
            'document': DocumentParser(self.config),
            'powerpoint': PowerPointParser(self.config),  # NEW PowerPoint support
            'image': ImageParser(self.config),
            'text': TextParser(self.config),
            'spreadsheet': SpreadsheetParser(self.config)
        }
        
        # Initialize converter for unsupported formats
        try:
            from utils.document_converter import DocumentConverter
            self.converter = DocumentConverter(self.config)
            logger.info("Document converter initialized")
        except ImportError:
            self.converter = None
            logger.warning("Document converter not available")
        
        logger.info("Universal Document Parser initialized - Arabic Enhanced Edition")
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        Parse a single file using the most appropriate parser.
        Now with conversion fallback for unsupported formats.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParseResult object containing extracted content and metadata
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            return ParseResult(
                success=False,
                error=f"File not found: {file_path}",
                file_path=str(file_path)
            )
        
        try:
            # Detect file type
            file_info = self.file_detector.detect_file_type(file_path)
            logger.info(f"Detected file type: {file_info.category} for {file_path.name}")
            
            # Try direct parsing first
            parser = self._get_parser_for_category(file_info.category)
            if parser:
                result = parser.parse(file_path, file_info)
                
                # Add timing and file metadata
                result.parsing_time = time.time() - start_time
                result.file_hash = self._calculate_file_hash(file_path)
                result.file_size = file_path.stat().st_size
                
                if result.success:
                    logger.info(f"Successfully parsed {file_path.name} in {result.parsing_time:.2f}s")
                    return result
                else:
                    logger.warning(f"Direct parsing failed for {file_path.name}: {result.error}")
            
            # If direct parsing failed or no parser available, try conversion
            if self.converter and self.config.comprehensive_parsing:
                logger.info(f"Attempting conversion fallback for {file_path.name}")
                
                converted_result = self._try_conversion_fallback(file_path, file_info)
                if converted_result and converted_result.success:
                    converted_result.parsing_time = time.time() - start_time
                    converted_result.file_hash = self._calculate_file_hash(file_path)
                    converted_result.file_size = file_path.stat().st_size
                    logger.info(f"Successfully parsed {file_path.name} via conversion in {converted_result.parsing_time:.2f}s")
                    return converted_result
            
            # If all methods failed
            error_msg = f"No parser available for file type: {file_info.mime_type}"
            if not self.converter:
                error_msg += " (conversion not available)"
            
            return ParseResult(
                success=False,
                error=error_msg,
                file_path=str(file_path),
                file_type=file_info.mime_type,
                parsing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return ParseResult(
                success=False,
                error=str(e),
                file_path=str(file_path),
                parsing_time=time.time() - start_time
            )
    
    def _try_conversion_fallback(self, file_path: Path, file_info) -> Optional[ParseResult]:
        """
        Try to convert unsupported file to a supported format and parse.
        
        Args:
            file_path: Path to the original file
            file_info: File information from detector
            
        Returns:
            ParseResult if conversion and parsing succeeded, None otherwise
        """
        if not self.converter:
            return None
        
        try:
            # Try conversion to PDF first (most universal)
            if self.converter.can_convert(file_path, 'pdf'):
                logger.info(f"Converting {file_path.name} to PDF for parsing")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_file = self.converter.convert_to_pdf(file_path, Path(temp_dir))
                    
                    if pdf_file and pdf_file.exists():
                        # Parse the converted PDF
                        pdf_parser = self.parsers.get('pdf')
                        if pdf_parser:
                            from models.parse_result import FileInfo
                            pdf_info = FileInfo(
                                mime_type='application/pdf',
                                category='pdf',
                                extension='.pdf'
                            )
                            
                            result = pdf_parser.parse(pdf_file, pdf_info)
                            if result.success:
                                result.parser_used = f"conversion-pdf-{result.parser_used}"
                                result.file_path = str(file_path)  # Keep original path
                                return result
            
            # For PowerPoint files, try PPTX conversion
            if file_path.suffix.lower() == '.ppt' and self.converter.can_convert(file_path, 'pptx'):
                logger.info(f"Converting {file_path.name} to PPTX for parsing")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    pptx_file = self.converter.convert_to_pptx(file_path, Path(temp_dir))
                    
                    if pptx_file and pptx_file.exists():
                        # Parse the converted PPTX
                        ppt_parser = self.parsers.get('powerpoint')
                        if ppt_parser:
                            from models.parse_result import FileInfo
                            pptx_info = FileInfo(
                                mime_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                                category='powerpoint',
                                extension='.pptx'
                            )
                            
                            result = ppt_parser.parse(pptx_file, pptx_info)
                            if result.success:
                                result.parser_used = f"conversion-pptx-{result.parser_used}"
                                result.file_path = str(file_path)  # Keep original path
                                return result
            
            logger.warning(f"No suitable conversion method found for {file_path.name}")
            
        except Exception as e:
            logger.error(f"Conversion fallback failed for {file_path.name}: {str(e)}")
        
        return None
    
    def _get_parser_for_category(self, category: str):
        """Get the appropriate parser for a file category."""
        return self.parsers.get(category)
    
    def parse_directory(self, directory_path: str, pattern: str = "*") -> List[ParseResult]:
        """
        Parse all files in a directory matching the pattern.
        
        Args:
            directory_path: Directory to scan
            pattern: File pattern to match (e.g., "*.pdf", "*")
            
        Returns:
            List of ParseResult objects
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all matching files
        files = list(directory.rglob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern '{pattern}' in {directory}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files
        if self.config.parallel_processing:
            return self._parse_files_parallel(files)
        else:
            return self._parse_files_sequential(files)
    
    def _parse_files_parallel(self, files: List[Path]) -> List[ParseResult]:
        """Parse files in parallel using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(self.parse_file, file): file 
                for file in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to parse {file}: {str(e)}")
                    results.append(ParseResult(
                        success=False,
                        error=str(e),
                        file_path=str(file)
                    ))
        
        return results
    
    def _parse_files_sequential(self, files: List[Path]) -> List[ParseResult]:
        """Parse files sequentially."""
        results = []
        for file in files:
            result = self.parse_file(file)
            results.append(result)
        return results
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of the file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get all supported file formats grouped by parser type."""
        return {
            parser_name: parser.get_supported_formats()
            for parser_name, parser in self.parsers.items()
        }
    
    def generate_report(self, results: List[ParseResult]) -> Dict[str, Any]:
        """Generate a processing report from parse results."""
        total_files = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_files - successful
        
        # Group by file type
        by_type = {}
        total_time = 0
        total_size = 0
        
        for result in results:
            if result.file_type:
                by_type.setdefault(result.file_type, []).append(result)
            if result.parsing_time:
                total_time += result.parsing_time
            if result.file_size:
                total_size += result.file_size
        
        return {
            "summary": {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_files * 100) if total_files > 0 else 0,
                "total_processing_time": total_time,
                "total_file_size": total_size,
                "average_time_per_file": total_time / total_files if total_files > 0 else 0
            },
            "by_file_type": {
                file_type: {
                    "count": len(file_results),
                    "success_rate": sum(1 for r in file_results if r.success) / len(file_results) * 100
                }
                for file_type, file_results in by_type.items()
            },
            "failed_files": [
                {"file": r.file_path, "error": r.error}
                for r in results if not r.success
            ]
        }


def main():
    """Main entry point for the Arabic Enhanced Document Parser."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal Document Parser - Arabic Enhanced Edition",
        epilog="Supports: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, images, and more!"
    )
    parser.add_argument("path", help="File or directory path to parse")
    parser.add_argument("--pattern", default="*", help="File pattern for directory parsing")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--report", action="store_true", help="Generate processing report")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for images and PDFs")
    parser.add_argument("--arabic", action="store_true", help="Enable Arabic text processing")
    parser.add_argument("--comprehensive", action="store_true", help="Try all parsing methods including conversion")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = ParserConfig()
    if args.config and Path(args.config).exists():
        config = ParserConfig.from_file(args.config)
    
    # Override config with command line arguments
    if args.parallel:
        config.parallel_processing = True
    if args.ocr:
        config.enable_ocr = True
        config.ocr.enabled = True
    if args.arabic:
        config.arabic_support = True
        config.mixed_language_support = True
        if 'ar' not in config.ocr.languages:
            config.ocr.languages.append('ar')
    if args.comprehensive:
        config.comprehensive_parsing = True
    
    # Initialize parser
    print("🇸🇦 Initializing Arabic Enhanced Document Parser...")
    doc_parser = UniversalDocumentParser(config)
    
    # Parse file or directory
    path = Path(args.path)
    if path.is_file():
        results = [doc_parser.parse_file(path)]
        print(f"📄 Parsed file: {path}")
    elif path.is_dir():
        results = doc_parser.parse_directory(path, args.pattern)
        print(f"📁 Parsed {len(results)} files from directory: {path}")
    else:
        print(f"❌ Error: Path not found: {path}")
        sys.exit(1)
    
    # Generate report if requested
    if args.report:
        report = doc_parser.generate_report(results)
        print("\n📊 === PROCESSING REPORT ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        # Additional Arabic-specific analysis
        arabic_count = 0
        ocr_count = 0
        for result in results:
            if result.success and result.content:
                arabic_chars = sum(1 for c in result.content if '\u0600' <= c <= '\u06FF')
                if arabic_chars > 0:
                    arabic_count += 1
            if result.success and any(img.extracted_text for img in result.images):
                ocr_count += 1
        
        print(f"\n🇸🇦 Arabic-Specific Analysis:")
        print(f"   Arabic content detected: {arabic_count}/{len(results)} files")
        print(f"   OCR extractions performed: {ocr_count}/{len(results)} files")
    
    # Save results if output file specified
    if args.output:
        output_data = {
            "results": [result.to_dict() for result in results],
            "report": doc_parser.generate_report(results) if args.report else None,
            "Arabic_enhanced": True,
            "parser_version": "Arabic Enhanced Edition"
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {args.output}")
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"\n✅ Summary: {successful}/{len(results)} files parsed successfully")
    
    if successful < len(results):
        failed_files = [r.file_path for r in results if not r.success]
        print(f"❌ Failed files: {', '.join(Path(f).name for f in failed_files[:5])}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
    
    print("🎉 Arabic Enhanced Document Parser completed!")


if __name__ == "__main__":
    main()