#!/usr/bin/env python3
"""
Enhanced Bilingual Document Parser - Comprehensive Example Usage
Demonstrates all features optimized for Arabic/English bilingual document processing.
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
from utils.config import ParserConfig, OCRConfig, PDFConfig, PowerPointConfig, DocumentConfig, ConverterConfig
from models.parse_result import ParseResult


def create_bilingual_config() -> ParserConfig:
    """Create bilingual-optimized configuration for Arabic/English document processing."""
    print("🌐 Creating bilingual-optimized configuration...")
    
    # OCR configuration for Arabic/English
    ocr_config = OCRConfig(
        enabled=True,
        languages=["ar", "en"],  # Arabic and English
        engine="easyocr",
        confidence_threshold=0.25,  # Lower threshold for Arabic
        arabic_support=True,
        rtl_support=True,
        mixed_language_detection=True,
        preprocessing=True
    )
    
    # PDF configuration
    pdf_config = PDFConfig(
        prefer_pdfplumber=True,
        extract_tables=True,
        extract_images=True
    )
    
    # PowerPoint configuration
    powerpoint_config = PowerPointConfig(
        extract_tables=True,
        extract_images=True,
        extract_slide_notes=True,
        ocr_on_images=True,
        max_slides=200
    )
    
    # Document configuration
    document_config = DocumentConfig(
        extract_tables=True,
        extract_images=True,
        enhanced_doc_parsing=True,
        enable_conversion_fallback=True
    )
    
    # Converter configuration
    converter_config = ConverterConfig(
        enable_online=False,
        libreoffice_timeout=120,
        prefer_local_conversion=True,
        cleanup_temp_files=True
    )
    
    # Main configuration
    config = ParserConfig(
        enable_ocr=True,
        arabic_support=True,
        mixed_language_support=True,
        comprehensive_parsing=True,
        parallel_processing=True,
        max_workers=4,
        max_file_size=200 * 1024 * 1024,  # 200MB
        
        ocr=ocr_config,
        pdf=pdf_config,
        powerpoint=powerpoint_config,
        document=document_config,
        converter=converter_config
    )
    
    print("✅ Bilingual configuration created successfully!")
    return config


def demonstrate_arabic_text_processing():
    """Demonstrate Arabic text processing capabilities."""
    print("\n" + "="*60)
    print("🔤 ARABIC TEXT PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Sample Arabic text
    arabic_samples = [
        "مرحباً بكم في معالج المستندات المطور",
        "تحليل المستندات باللغة العربية",
        "Mixed Arabic النص العربي and English text",
        "التكنولوجيا الحديثة - Modern Technology"
    ]
    
    config = create_bilingual_config()
    
    # Test Arabic text processing
    try:
        # Check if Arabic reshaping is available
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            arabic_support = True
            print("✅ Arabic text processing libraries available")
        except ImportError:
            arabic_support = False
            print("⚠️  Arabic text processing libraries not installed")
            print("   Install with: pip install arabic-reshaper python-bidi")
        
        for i, text in enumerate(arabic_samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Original: {text}")
            
            if arabic_support:
                # Process Arabic text
                reshaped = arabic_reshaper.reshape(text)
                display_text = get_display(reshaped)
                print(f"Processed: {display_text}")
            
            # Detect language composition
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            latin_chars = sum(1 for c in text if c.isalpha() and not ('\u0600' <= c <= '\u06FF'))
            
            if arabic_chars > latin_chars:
                lang_type = "Primarily Arabic"
            elif latin_chars > arabic_chars:
                lang_type = "Primarily English"
            else:
                lang_type = "Mixed Arabic/English"
            
            print(f"Language: {lang_type} (Arabic: {arabic_chars}, Latin: {latin_chars})")
    
    except Exception as e:
        print(f"❌ Arabic processing demonstration failed: {e}")


def demonstrate_powerpoint_parsing():
    """Demonstrate PowerPoint parsing with bilingual support."""
    print("\n" + "="*60)
    print("📊 POWERPOINT PARSING DEMONSTRATION")
    print("="*60)
    
    config = create_bilingual_config()
    parser = UniversalDocumentParser(config)
    
    # Look for PowerPoint files
    ppt_extensions = ['*.pptx', '*.ppt']
    ppt_files = []
    
    for pattern in ppt_extensions:
        ppt_files.extend(Path('.').glob(pattern))
    
    if not ppt_files:
        print("ℹ️  No PowerPoint files found in current directory")
        print("   Create a sample PPTX file to test this feature")
        
        # Show what PowerPoint parsing can do
        print("\n🎯 PowerPoint Parsing Capabilities:")
        print("   ✅ Extract text from all slides")
        print("   ✅ Parse tables in presentations")
        print("   ✅ OCR text from images in slides")
        print("   ✅ Handle Arabic/English mixed content")
        print("   ✅ Convert legacy PPT files")
        print("   ✅ Preserve slide structure")
        return
    
    for ppt_file in ppt_files[:2]:  # Test first 2 files
        print(f"\n📊 Processing: {ppt_file.name}")
        
        try:
            result = parser.parse_file(ppt_file)
            
            if result.success:
                print(f"   ✅ Success with {result.parser_used}")
                print(f"   📝 Content length: {len(result.content)} characters")
                print(f"   📊 Tables found: {len(result.tables)}")
                print(f"   🖼️  Images processed: {len(result.images)}")
                
                # Analyze slides
                slides = result.content.split('=== Slide ')
                print(f"   📄 Slides detected: {len(slides) - 1}")
                
                # Show first slide preview
                if len(slides) > 1:
                    first_slide = slides[1].split('\n')[:3]
                    print(f"   📖 First slide preview:")
                    for line in first_slide:
                        if line.strip():
                            print(f"      {line.strip()[:80]}...")
                
                # Check for Arabic content
                arabic_found = any('\u0600' <= char <= '\u06FF' for char in result.content)
                print(f"   🔤 Arabic content: {'Yes' if arabic_found else 'No'}")
                
                # Show OCR results
                ocr_results = [img for img in result.images if img.extracted_text]
                if ocr_results:
                    print(f"   🔍 OCR extracted text from {len(ocr_results)} images")
                    for img in ocr_results[:2]:  # Show first 2
                        preview = img.extracted_text[:100].replace('\n', ' ')
                        print(f"      Image {img.image_index}: {preview}...")
            
            else:
                print(f"   ❌ Failed: {result.error}")
        
        except Exception as e:
            print(f"   ❌ Error processing {ppt_file.name}: {e}")


def demonstrate_enhanced_doc_parsing():
    """Demonstrate enhanced DOC file parsing."""
    print("\n" + "="*60)
    print("📄 ENHANCED DOC PARSING DEMONSTRATION")
    print("="*60)
    
    config = create_bilingual_config()
    parser = UniversalDocumentParser(config)
    
    # Look for DOC files
    doc_files = list(Path('.').glob('*.doc')) + list(Path('.').glob('*.docx'))
    
    if not doc_files:
        print("ℹ️  No DOC/DOCX files found in current directory")
        print("\n🎯 Enhanced DOC Parsing Features:")
        print("   ✅ Mammoth library for better DOC parsing")
        print("   ✅ Conversion fallback (DOC → PDF → Parse)")
        print("   ✅ Multiple extraction methods")
        print("   ✅ Arabic text processing")
        print("   ✅ Image OCR extraction")
        print("   ✅ Table structure preservation")
        return
    
    for doc_file in doc_files[:3]:  # Test first 3 files
        print(f"\n📄 Processing: {doc_file.name}")
        
        try:
            result = parser.parse_file(doc_file)
            
            if result.success:
                print(f"   ✅ Success with {result.parser_used}")
                print(f"   📝 Content: {len(result.content)} chars")
                print(f"   📊 Tables: {len(result.tables)}")
                print(f"   🖼️  Images: {len(result.images)}")
                print(f"   ⏱️  Time: {result.parsing_time:.2f}s")
                
                # Language analysis
                if result.content:
                    arabic_chars = sum(1 for c in result.content if '\u0600' <= c <= '\u06FF')
                    total_chars = len(result.content)
                    arabic_percentage = (arabic_chars / total_chars * 100) if total_chars > 0 else 0
                    
                    print(f"   🔤 Arabic content: {arabic_percentage:.1f}%")
                    
                    # Show content preview
                    preview = result.content[:150].replace('\n', ' ')
                    print(f"   📖 Preview: {preview}...")
                
                # Show table information
                for i, table in enumerate(result.tables[:2]):  # First 2 tables
                    print(f"   📊 Table {i+1}: {len(table.headers)} cols × {len(table.rows)} rows")
                    if table.headers:
                        headers_preview = ', '.join(table.headers[:3])
                        print(f"      Headers: {headers_preview}...")
            
            else:
                print(f"   ❌ Failed: {result.error}")
                
                # If it's a DOC file, explain the enhanced parsing attempt
                if doc_file.suffix.lower() == '.doc':
                    print("   ℹ️  DOC files use enhanced parsing with multiple fallback methods")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demonstrate_conversion_fallback():
    """Demonstrate conversion fallback for unsupported formats."""
    print("\n" + "="*60)
    print("🔄 CONVERSION FALLBACK DEMONSTRATION")
    print("="*60)
    
    config = create_bilingual_config()
    parser = UniversalDocumentParser(config)
    
    # Check converter availability
    if hasattr(parser, 'converter') and parser.converter:
        print("✅ Document converter available")
        
        # Show supported conversions
        try:
            conversions = parser.converter.get_supported_conversions()
            print("\n🔄 Supported conversions:")
            for target, formats in conversions.items():
                if formats:
                    print(f"   {target}: {', '.join(formats)}")
        except Exception as e:
            print(f"   ⚠️  Could not get conversion info: {e}")
    else:
        print("⚠️  Document converter not available")
        print("   Install LibreOffice or enable Office automation for conversion support")
        return
    
    # Look for files that might need conversion
    conversion_candidates = []
    for ext in ['.ppt', '.doc']:
        conversion_candidates.extend(Path('.').glob(f'*{ext}'))
    
    if conversion_candidates:
        print(f"\n📁 Found {len(conversion_candidates)} files that might benefit from conversion:")
        
        for file_path in conversion_candidates[:2]:  # Test first 2
            print(f"\n🔄 Testing conversion for: {file_path.name}")
            
            try:
                result = parser.parse_file(file_path)
                
                if result.success:
                    if 'conversion' in result.parser_used:
                        print(f"   ✅ Successfully converted and parsed with {result.parser_used}")
                        print(f"   📝 Extracted {len(result.content)} characters")
                    else:
                        print(f"   ✅ Parsed directly with {result.parser_used}")
                else:
                    print(f"   ❌ Conversion and parsing failed: {result.error}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    else:
        print("\nℹ️  No files found that typically require conversion (.ppt, .doc)")


def comprehensive_batch_analysis():
    """Perform comprehensive analysis of all documents in directory."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE BATCH ANALYSIS")
    print("="*60)
    
    config = create_bilingual_config()
    parser = UniversalDocumentParser(config)
    
    print("🔍 Scanning current directory for all supported documents...")
    
    # Find all supported files
    supported_extensions = [
        '*.pdf', '*.docx', '*.doc', '*.pptx', '*.ppt',
        '*.xlsx', '*.xls', '*.csv', '*.txt', '*.md',
        '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp'
    ]
    
    all_files = []
    for pattern in supported_extensions:
        files = list(Path('.').glob(pattern))
        all_files.extend(files)
    
    if not all_files:
        print("ℹ️  No supported documents found in current directory")
        return
    
    print(f"📁 Found {len(all_files)} supported files")
    
    # Process files (limit to first 10 for demo)
    files_to_process = all_files[:10]
    results = []
    
    print(f"\n🚀 Processing {len(files_to_process)} files...")
    
    start_time = time.time()
    
    for i, file_path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] Processing {file_path.name}...", end=' ')
        
        try:
            result = parser.parse_file(file_path)
            results.append(result)
            
            if result.success:
                print(f"✅ ({result.parser_used})")
            else:
                print(f"❌ ({result.error[:50]}...)")
                
        except Exception as e:
            print(f"❌ (Exception: {str(e)[:50]}...)")
    
    processing_time = time.time() - start_time
    
    # Generate comprehensive report
    print(f"\n📊 BATCH PROCESSING REPORT")
    print(f"⏱️  Total time: {processing_time:.2f} seconds")
    print(f"📈 Average time per file: {processing_time/len(files_to_process):.2f}s")
    
    # Success statistics
    successful = [r for r in results if r.success]
    print(f"✅ Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    # Parser usage statistics
    parser_usage = {}
    for result in successful:
        parser = result.parser_used or 'unknown'
        parser_usage[parser] = parser_usage.get(parser, 0) + 1
    
    if parser_usage:
        print(f"\n🔧 Parser usage:")
        for parser, count in sorted(parser_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"   {parser}: {count} files")
    
    # Content analysis
    arabic_docs = 0
    english_docs = 0
    mixed_docs = 0
    total_tables = 0
    total_images = 0
    ocr_extractions = 0
    
    for result in successful:
        if result.content:
            # Language analysis
            arabic_chars = sum(1 for c in result.content if '\u0600' <= c <= '\u06FF')
            latin_chars = sum(1 for c in result.content if c.isalpha() and not ('\u0600' <= c <= '\u06FF'))
            
            if arabic_chars > latin_chars * 2:
                arabic_docs += 1
            elif latin_chars > arabic_chars * 2:
                english_docs += 1
            else:
                mixed_docs += 1
        
        # Structure analysis
        total_tables += len(result.tables)
        total_images += len(result.images)
        
        # OCR analysis
        if any(img.extracted_text for img in result.images):
            ocr_extractions += 1
    
    print(f"\n🌍 Language Analysis:")
    print(f"   🔤 Primarily Arabic: {arabic_docs}")
    print(f"   🔤 Primarily English: {english_docs}")
    print(f"   🌐 Mixed language: {mixed_docs}")
    
    print(f"\n📊 Content Extraction:")
    print(f"   📋 Tables extracted: {total_tables}")
    print(f"   🖼️  Images processed: {total_images}")
    print(f"   🔍 OCR extractions: {ocr_extractions}")
    
    # File type breakdown
    file_types = {}
    for result in results:
        ext = Path(result.file_path).suffix.lower()
        file_types[ext] = file_types.get(ext, {'total': 0, 'success': 0})
        file_types[ext]['total'] += 1
        if result.success:
            file_types[ext]['success'] += 1
    
    if file_types:
        print(f"\n📄 File Type Performance:")
        for ext, stats in sorted(file_types.items()):
            success_rate = stats['success'] / stats['total'] * 100
            print(f"   {ext}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")


def main():
    """Main demonstration function."""
    print("🌐 ENHANCED BILINGUAL DOCUMENT PARSER - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("Welcome to the enhanced document parser optimized for bilingual processing!")
    print("This demo showcases all features including Arabic/English support, PowerPoint")
    print("parsing, enhanced DOC processing, and conversion capabilities.")
    print("=" * 70)
    
    # Run all demonstrations
    try:
        # 1. Arabic text processing
        demonstrate_arabic_text_processing()
        
        # 2. PowerPoint parsing
        demonstrate_powerpoint_parsing()
        
        # 3. Enhanced DOC parsing
        demonstrate_enhanced_doc_parsing()
        
        # 4. Conversion fallback
        demonstrate_conversion_fallback()
        
        # 5. Comprehensive batch analysis
        comprehensive_batch_analysis()
        
        print("\n" + "="*70)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print("Your enhanced bilingual document parser is ready for:")
        print("✅ Arabic/English document processing")
        print("✅ PowerPoint presentations with OCR")
        print("✅ Enhanced DOC file parsing")
        print("✅ Automatic format conversion")
        print("✅ Batch processing with analytics")
        print("✅ Enterprise-grade performance")
        print("\n🚀 Start using it with your bilingual documents today!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your setup and dependencies")


if __name__ == "__main__":
    main()