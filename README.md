# Universal Document Parser - Setup Guide

A high-performance, modular document parsing solution that automatically selects the best library for each file type.

## 🚀 Quick Start

### 1. Project Structure

Create the following directory structure:

```
universal_parser/
├── main.py                     # Main parser script
├── requirements.txt            # Dependencies
├── config.json                 # Configuration (optional)
├── models/
│   └── parse_result.py         # Data models
├── utils/
│   ├── config.py              # Configuration management
│   └── file_detector.py       # File type detection
└── parsers/
    ├── pdf_parser.py          # PDF parsing
    ├── document_parser.py     # DOCX, DOC, RTF, ODT
    ├── image_parser.py        # Images with OCR
    ├── text_parser.py         # Text, HTML, XML, Markdown
    └── spreadsheet_parser.py  # Excel, CSV, ODS
```

### 2. Installation

#### Basic Installation (Minimal Dependencies)
```bash
# Clone or download the parser files
# Navigate to the project directory
cd universal_parser

# Install core dependencies
pip install python-magic Pillow chardet beautifulsoup4 lxml

# For Windows users, also install:
pip install python-magic-bin
```

#### Full Installation (All Features)
```bash
# Install all dependencies
pip install -r requirements.txt

# Additional OCR setup (see OCR Setup section below)
```

#### Docker Installation (Recommended for Production)
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy parser code
COPY . .

# Run the parser
CMD ["python", "main.py", "--help"]
```

### 3. OCR Setup (Optional but Recommended)

#### EasyOCR (Recommended)
```bash
pip install easyocr
# No additional setup required - downloads models automatically
```

#### Tesseract OCR
```bash
# Install pytesseract
pip install pytesseract

# Install Tesseract engine:

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng

# macOS:
brew install tesseract tesseract-lang

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add tesseract.exe to PATH or set TESSDATA_PREFIX
```

### 4. Configuration

#### Basic Configuration
Create `config.json` in the project root:

```json
{
  "enable_ocr": true,
  "parallel_processing": true,
  "max_workers": 4,
  "max_file_size": 104857600,
  "ocr": {
    "enabled": true,
    "languages": ["en", "ar"],
    "engine": "easyocr",
    "confidence_threshold": 0.5
  },
  "pdf": {
    "prefer_pdfplumber": true,
    "extract_tables": true,
    "extract_images": true
  },
  "performance": {
    "parallel_processing": true,
    "max_workers": 4,
    "timeout_seconds": 300
  }
}
```

#### Environment Variables
```bash
# OCR Configuration
export PARSER_ENABLE_OCR=true
export OCR_LANGUAGES=en,ar
export OCR_ENGINE=easyocr

# Performance
export PARSER_PARALLEL=true
export PARSER_MAX_WORKERS=4

# File Processing
export PARSER_MAX_FILE_SIZE=104857600
export PDF_EXTRACT_TABLES=true
```

## 📖 Usage Examples

### Command Line Usage

#### Parse a Single File
```bash
python main.py document.pdf
```

#### Parse Directory with Pattern
```bash
python main.py /path/to/documents --pattern "*.pdf"
```

#### Generate Report
```bash
python main.py /path/to/documents --report --output results.json
```

#### Enable OCR and Parallel Processing
```bash
python main.py /path/to/images --ocr --parallel
```

### Python API Usage

#### Basic Usage
```python
from main import UniversalDocumentParser
from utils.config import ParserConfig

# Initialize parser
config = ParserConfig(enable_ocr=True, parallel_processing=True)
parser = UniversalDocumentParser(config)

# Parse single file
result = parser.parse_file("document.pdf")
print(f"Content: {result.content[:500]}...")
print(f"Tables: {len(result.tables)}")
print(f"Success: {result.success}")

# Parse directory
results = parser.parse_directory("/path/to/documents", "*.pdf")
successful = [r for r in results if r.success]
print(f"Parsed {len(successful)}/{len(results)} files successfully")
```

#### Advanced Usage with Custom Configuration
```python
from utils.config import ParserConfig, OCRConfig, PDFConfig

# Custom configuration
ocr_config = OCRConfig(
    enabled=True,
    languages=["en", "ar"],
    engine="easyocr",
    confidence_threshold=0.7
)

pdf_config = PDFConfig(
    prefer_pdfplumber=True,
    extract_tables=True,
    extract_images=True
)

config = ParserConfig(
    ocr=ocr_config,
    pdf=pdf_config,
    parallel_processing=True,
    max_workers=6
)

parser = UniversalDocumentParser(config)

# Process with custom settings
result = parser.parse_file("complex_document.pdf")

# Access detailed results
if result.success:
    print(f"Parser used: {result.parser_used}")
    print(f"Processing time: {result.parsing_time:.2f}s")
    print(f"Tables found: {len(result.tables)}")
    print(f"Images found: {len(result.images)}")
    
    # Access table data
    for i, table in enumerate(result.tables):
        print(f"Table {i}: {len(table.headers)} columns, {len(table.rows)} rows")
        print(f"Headers: {table.headers}")
```

## 🔧 Customization

### Adding Custom Parser
```python
# parsers/custom_parser.py
from models.parse_result import ParseResult, FileInfo
from utils.config import ParserConfig

class CustomParser:
    def __init__(self, config: ParserConfig):
        self.config = config
    
    def parse(self, file_path, file_info: FileInfo) -> ParseResult:
        result = ParseResult(file_path=str(file_path))
        
        # Your custom parsing logic here
        with open(file_path, 'r') as f:
            result.content = f.read()
        
        result.success = True
        result.parser_used = 'custom_parser'
        return result
    
    def get_supported_formats(self):
        return ['custom_ext']

# Register in main.py
self.parsers['custom'] = CustomParser(self.config)
```

### Custom File Type Detection
```python
# Add to utils/file_detector.py
EXTENSION_TO_CATEGORY = {
    # ... existing mappings ...
    '.custom': 'custom',
}
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Magic Library Not Found
```bash
# Linux
sudo apt-get install libmagic1

# macOS
brew install libmagic

# Windows
pip install python-magic-bin
```

#### 2. OCR Not Working
```bash
# Check EasyOCR installation
python -c "import easyocr; print('EasyOCR OK')"

# Check Tesseract installation
tesseract --version

# For Arabic OCR specifically:
sudo apt-get install tesseract-ocr-ara
```

#### 3. PDF Parsing Fails
```bash
# Install multiple PDF libraries for fallback
pip install pdfplumber PyMuPDF pypdf
```

#### 4. Memory Issues with Large Files
```python
# Adjust configuration
config = ParserConfig(
    max_file_size=50 * 1024 * 1024,  # 50MB limit
    performance=PerformanceConfig(
        memory_limit_mb=256,
        chunk_size=512 * 1024  # 512KB chunks
    )
)
```

### Performance Optimization

#### 1. Enable Parallel Processing
```python
config = ParserConfig(
    parallel_processing=True,
    max_workers=8  # Adjust based on CPU cores
)
```

#### 2. Use Appropriate Libraries
- **PDFs**: pdfplumber for tables, PyMuPDF for speed
- **Images**: EasyOCR for accuracy, Tesseract for speed
- **Spreadsheets**: pandas for complex files, openpyxl for simple ones

#### 3. Optimize OCR Settings
```python
ocr_config = OCRConfig(
    confidence_threshold=0.8,  # Higher = fewer false positives
    preprocessing=True,        # Better accuracy but slower
    languages=["en"]           # Limit to needed languages
)
```

## 📊 Monitoring and Logging

### Enable Detailed Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parser.log'),
        logging.StreamHandler()
    ]
)

config = ParserConfig(verbose_logging=True)
```

### Generate Processing Reports
```python
# Get detailed statistics
results = parser.parse_directory("/documents")
report = parser.generate_report(results)

print(f"Success rate: {report['summary']['success_rate']:.1f}%")
print(f"Total processing time: {report['summary']['total_processing_time']:.2f}s")
print(f"Average time per file: {report['summary']['average_time_per_file']:.2f}s")

# File type breakdown
for file_type, stats in report['by_file_type'].items():
    print(f"{file_type}: {stats['count']} files, {stats['success_rate']:.1f}% success")
```

## 🚀 Production Deployment

### Docker Compose for Scalable Deployment
```yaml
version: '3.8'
services:
  parser-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./documents:/app/documents
      - ./output:/app/output
    environment:
      - PARSER_ENABLE_OCR=true
      - PARSER_MAX_WORKERS=4
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

### API Wrapper
```python
# api_server.py
from fastapi import FastAPI, File, UploadFile
from main import UniversalDocumentParser

app = FastAPI()
parser = UniversalDocumentParser()

@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    # Save uploaded file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Parse document
    result = parser.parse_file(temp_path)
    
    return {
        "success": result.success,
        "content": result.content,
        "tables": len(result.tables),
        "parser_used": result.parser_used
    }
```

## 📝 License and Support

This parser is designed for enterprise use with modular, reusable components. Customize as needed for your specific requirements.

For issues or enhancements, check the configuration options and logging output first. The modular design allows easy debugging and extension of individual parser components.