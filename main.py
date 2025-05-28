#!/usr/bin/env python3
"""
Universal Document Parser - Main Script
High-performance document parsing service with automatic library selection
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import parser modules
from parsers.pdf_parser import PDFParser
from parsers.document_parser import DocumentParser
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
    """
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """Initialize the universal parser with configuration."""
        self.config = config or ParserConfig()
        self.file_detector = FileTypeDetector()
        
        # Initialize specialized parsers
        self.parsers = {
            'pdf': PDFParser(self.config),
            'document': DocumentParser(self.config),
            'image': ImageParser(self.config),
            'text': TextParser(self.config),
            'spreadsheet': SpreadsheetParser(self.config)
        }
        
        logger.info("Universal Document Parser initialized")
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        Parse a single file using the most appropriate parser.
        
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
            
            # Select appropriate parser
            parser = self._get_parser_for_category(file_info.category)
            if not parser:
                return ParseResult(
                    success=False,
                    error=f"No parser available for file type: {file_info.mime_type}",
                    file_path=str(file_path),
                    file_type=file_info.mime_type
                )
            
            # Parse the file
            result = parser.parse(file_path, file_info)
            
            # Add timing and file metadata
            result.parsing_time = time.time() - start_time
            result.file_hash = self._calculate_file_hash(file_path)
            result.file_size = file_path.stat().st_size
            
            logger.info(f"Successfully parsed {file_path.name} in {result.parsing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return ParseResult(
                success=False,
                error=str(e),
                file_path=str(file_path),
                parsing_time=time.time() - start_time
            )
    
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
    
    def _get_parser_for_category(self, category: str):
        """Get the appropriate parser for a file category."""
        return self.parsers.get(category)
    
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
    """Main entry point for the document parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Document Parser")
    parser.add_argument("path", help="File or directory path to parse")
    parser.add_argument("--pattern", default="*", help="File pattern for directory parsing")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--report", action="store_true", help="Generate processing report")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for images")
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
        # Also update the OCR sub-config
        config.ocr.enabled = True
    
    # Initialize parser
    doc_parser = UniversalDocumentParser(config)
    
    # Parse file or directory
    path = Path(args.path)
    if path.is_file():
        results = [doc_parser.parse_file(path)]
        print(f"Parsed file: {path}")
    elif path.is_dir():
        results = doc_parser.parse_directory(path, args.pattern)
        print(f"Parsed {len(results)} files from directory: {path}")
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    # Generate report if requested
    if args.report:
        report = doc_parser.generate_report(results)
        print("\n=== Processing Report ===")
        print(json.dumps(report, indent=2))
    
    # Save results if output file specified
    if args.output:
        output_data = {
            "results": [result.to_dict() for result in results],
            "report": doc_parser.generate_report(results) if args.report else None
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"\nSummary: {successful}/{len(results)} files parsed successfully")


if __name__ == "__main__":
    main()