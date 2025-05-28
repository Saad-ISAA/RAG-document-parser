#!/usr/bin/env python3
"""
Example usage scripts for the Universal Document Parser.
Demonstrates various use cases and integration patterns.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the parser to the path
sys.path.append(str(Path(__file__).parent))

from main import UniversalDocumentParser
from utils.config import ParserConfig, OCRConfig, PDFConfig, PerformanceConfig
from models.parse_result import ParseResult


def example_basic_usage():
    """Basic usage example - parse files in current directory."""
    print("=== Basic Usage Example ===")
    
    # Initialize with default configuration
    parser = UniversalDocumentParser()
    
    # Find all files in current directory
    current_dir = Path(".")
    supported_extensions = ['.pdf', '.docx', '.txt', '.jpg', '.png']
    
    files_to_parse = []
    for ext in supported_extensions:
        files_to_parse.extend(current_dir.glob(f"*{ext}"))
    
    if not files_to_parse:
        print("No supported files found in current directory")
        return
    
    print(f"Found {len(files_to_parse)} files to parse")
    
    # Parse each file
    for file_path in files_to_parse:
        print(f"\nParsing: {file_path.name}")
        result = parser.parse_file(file_path)
        
        if result.success:
            print(f"  ✓ Success ({result.parser_used})")
            print(f"  ✓ Content length: {len(result.content)} characters")
            print(f"  ✓ Tables: {len(result.tables)}")
            print(f"  ✓ Images: {len(result.images)}")
            print(f"  ✓ Processing time: {result.parsing_time:.2f}s")
        else:
            print(f"  ✗ Failed: {result.error}")


def example_advanced_configuration():
    """Advanced configuration example with custom settings."""
    print("\n=== Advanced Configuration Example ===")
    
    # Create custom configuration
    config = ParserConfig(
        enable_ocr=True,
        parallel_processing=True,
        max_workers=4,
        max_file_size=50 * 1024 * 1024,  # 50MB limit
        
        # OCR settings
        ocr=OCRConfig(
            enabled=True,
            languages=["en", "ar"],
            engine="easyocr",
            confidence_threshold=0.7,
            preprocessing=True
        ),
        
        # PDF settings
        pdf=PDFConfig(
            prefer_pdfplumber=True,
            extract_tables=True,
            extract_images=True
        ),
        
        # Performance settings
        performance=PerformanceConfig(
            parallel_processing=True,
            max_workers=4,
            memory_limit_mb=512,
            timeout_seconds=300
        )
    )
    
    # Initialize parser with custom config
    parser = UniversalDocumentParser(config)
    
    # Show configuration
    print("Configuration:")
    print(f"  OCR enabled: {config.enable_ocr}")
    print(f"  OCR languages: {config.ocr.languages}")
    print(f"  Max workers: {config.max_workers}")
    print(f"  Max file size: {config.max_file_size // (1024*1024)}MB")
    
    # Get supported formats
    supported_formats = parser.get_supported_formats()
    print(f"\nSupported formats: {supported_formats}")


def example_batch_processing():
    """Batch processing example with progress tracking."""
    print("\n=== Batch Processing Example ===")
    
    # Create a test directory structure if it doesn't exist
    test_dir = Path("test_documents")
    if not test_dir.exists():
        print(f"Directory {test_dir} not found. Create it and add some documents to test.")
        return
    
    # Initialize parser
    config = ParserConfig(parallel_processing=True, max_workers=4)
    parser = UniversalDocumentParser(config)
    
    # Get all files recursively
    all_files = []
    for pattern in ["*.pdf", "*.docx", "*.txt", "*.jpg", "*.png", "*.csv"]:
        all_files.extend(test_dir.rglob(pattern))
    
    if not all_files:
        print(f"No files found in {test_dir}")
        return
    
    print(f"Processing {len(all_files)} files...")
    
    # Process with progress tracking
    start_time = time.time()
    results = []
    
    for i, file_path in enumerate(all_files, 1):
        print(f"Processing [{i}/{len(all_files)}]: {file_path.name}")
        
        result = parser.parse_file(file_path)
        results.append(result)
        
        # Show progress
        if result.success:
            print(f"  ✓ {result.parser_used} - {len(result.content)} chars")
        else:
            print(f"  ✗ {result.error}")
    
    # Generate report
    processing_time = time.time() - start_time
    report = parser.generate_report(results)
    
    print(f"\n=== Batch Processing Report ===")
    print(f"Total time: {processing_time:.2f}s")
    print(f"Files processed: {report['summary']['total_files']}")
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"Average time per file: {report['summary']['average_time_per_file']:.2f}s")
    
    # Show results by file type
    print("\nResults by file type:")
    for file_type, stats in report['by_file_type'].items():
        print(f"  {file_type}: {stats['count']} files, {stats['success_rate']:.1f}% success")


def example_ocr_focused():
    """OCR-focused example for image documents."""
    print("\n=== OCR Processing Example ===")
    
    # OCR configuration
    ocr_config = OCRConfig(
        enabled=True,
        languages=["en", "ar"],  # English and Arabic
        engine="easyocr",  # or "tesseract"
        confidence_threshold=0.6,
        preprocessing=True
    )
    
    config = ParserConfig(
        enable_ocr=True,
        ocr=ocr_config
    )
    
    parser = UniversalDocumentParser(config)
    
    # Test OCR capabilities
    print("OCR Configuration:")
    ocr_info = parser.parsers['image'].get_ocr_info()
    print(f"  Engines available: {ocr_info['available_engines']}")
    print(f"  Languages: {ocr_info['supported_languages']}")
    print(f"  Confidence threshold: {ocr_info['confidence_threshold']}")
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(".").glob(f"*{ext}"))
    
    if not image_files:
        print("No image files found for OCR testing")
        return
    
    print(f"\nProcessing {len(image_files)} image files with OCR:")
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")
        result = parser.parse_file(image_file)
        
        if result.success and result.content:
            print(f"  ✓ Extracted text ({len(result.content)} chars):")
            print(f"    {result.content[:200]}...")
            if len(result.content) > 200:
                print("    [truncated]")
        elif result.success:
            print("  ✓ Processed but no text found")
        else:
            print(f"  ✗ Failed: {result.error}")


def example_data_extraction():
    """Example focusing on structured data extraction."""
    print("\n=== Structured Data Extraction Example ===")
    
    # Configuration optimized for data extraction
    config = ParserConfig(
        pdf=PDFConfig(
            prefer_pdfplumber=True,  # Better for tables
            extract_tables=True,
            extract_images=False  # Skip images for faster processing
        )
    )
    
    parser = UniversalDocumentParser(config)
    
    # Look for files with likely structured data
    data_files = []
    for pattern in ["*.pdf", "*.xlsx", "*.csv", "*.docx"]:
        data_files.extend(Path(".").glob(pattern))
    
    if not data_files:
        print("No data files found")
        return
    
    print(f"Analyzing {len(data_files)} files for structured data:")
    
    total_tables = 0
    total_images = 0
    
    for file_path in data_files:
        print(f"\nAnalyzing: {file_path.name}")
        result = parser.parse_file(file_path)
        
        if result.success:
            print(f"  ✓ Parser: {result.parser_used}")
            print(f"  ✓ Tables found: {len(result.tables)}")
            print(f"  ✓ Images found: {len(result.images)}")
            
            # Show table details
            for i, table in enumerate(result.tables):
                print(f"    Table {i+1}: {len(table.headers)} columns × {len(table.rows)} rows")
                if table.headers:
                    print(f"      Headers: {', '.join(table.headers[:5])}{'...' if len(table.headers) > 5 else ''}")
            
            total_tables += len(result.tables)
            total_images += len(result.images)
        else:
            print(f"  ✗ Failed: {result.error}")
    
    print(f"\nSummary:")
    print(f"  Total tables extracted: {total_tables}")
    print(f"  Total images found: {total_images}")


def example_export_results():
    """Example showing how to export results in different formats."""
    print("\n=== Export Results Example ===")
    
    parser = UniversalDocumentParser()
    
    # Parse some files
    files = list(Path(".").glob("*.pdf")) + list(Path(".").glob("*.docx"))
    if not files:
        print("No files to process for export example")
        return
    
    results = []
    for file_path in files[:3]:  # Limit to first 3 files
        result = parser.parse_file(file_path)
        results.append(result)
    
    # Export to JSON
    json_output = {
        "parsing_session": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": len(results),
            "successful": sum(1 for r in results if r.success)
        },
        "results": [result.to_dict() for result in results]
    }
    
    with open("parsing_results.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print("Results exported to parsing_results.json")
    
    # Export tables to CSV
    table_count = 0
    for i, result in enumerate(results):
        if result.success and result.tables:
            for j, table in enumerate(result.tables):
                csv_filename = f"table_{i}_{j}.csv"
                
                with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    
                    # Write headers
                    if table.headers:
                        writer.writerow(table.headers)
                    
                    # Write data rows
                    writer.writerows(table.rows)
                
                table_count += 1
                print(f"Table exported to {csv_filename}")
    
    print(f"Exported {table_count} tables to CSV files")


def example_error_handling():
    """Example demonstrating error handling and recovery."""
    print("\n=== Error Handling Example ===")
    
    parser = UniversalDocumentParser()
    
    # Test with various problematic scenarios
    test_cases = [
        "nonexistent_file.pdf",
        Path("."),  # Directory instead of file
        __file__,   # Python file (should work with text parser)
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case}")
        
        try:
            result = parser.parse_file(test_case)
            
            if result.success:
                print(f"  ✓ Success: {result.parser_used}")
                print(f"  ✓ Content length: {len(result.content)}")
            else:
                print(f"  ✗ Failed: {result.error}")
                print(f"  ✗ Parser attempted: {result.parser_used or 'None'}")
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
    
    # Test batch processing with mixed success/failure
    mixed_files = ["nonexistent1.pdf", __file__, "nonexistent2.docx"]
    
    print(f"\nBatch processing with mixed results:")
    results = []
    for file_path in mixed_files:
        try:
            result = parser.parse_file(file_path)
            results.append(result)
        except Exception as e:
            print(f"Exception processing {file_path}: {e}")
    
    # Generate report including failures
    if results:
        report = parser.generate_report(results)
        print(f"\nMixed batch report:")
        print(f"  Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"  Failed files: {len(report['failed_files'])}")
        
        for failed in report['failed_files']:
            print(f"    {failed['file']}: {failed['error']}")


def main():
    """Run all examples."""
    print("Universal Document Parser - Example Usage")
    print("=" * 50)
    
    examples = [
        example_basic_usage,
        example_advanced_configuration,
        example_batch_processing,
        example_ocr_focused,
        example_data_extraction,
        example_export_results,
        example_error_handling
    ]
    
    for example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"\nExample failed: {str(e)}")
        
        print("\n" + "-" * 50)
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()