# models/parse_result.py
"""
Data models for parse results and file information.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class FileInfo:
    """Information about a detected file."""
    mime_type: str
    category: str  # pdf, document, image, text, spreadsheet
    extension: str
    encoding: Optional[str] = None
    confidence: float = 1.0


@dataclass
class TableData:
    """Structured table data extracted from documents."""
    headers: List[str]
    rows: List[List[str]]
    page_number: Optional[int] = None
    table_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary format."""
        if not self.headers or not self.rows:
            return {}
        
        return {
            "headers": self.headers,
            "data": [
                {header: row[i] if i < len(row) else "" 
                 for i, header in enumerate(self.headers)}
                for row in self.rows
            ],
            "page_number": self.page_number,
            "table_index": self.table_index
        }


@dataclass
class ImageInfo:
    """Information about extracted images."""
    image_index: int
    page_number: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    extracted_text: Optional[str] = None  # OCR text if available


@dataclass
class DocumentMetadata:
    """Metadata extracted from documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of document parsing operation."""
    # Status information
    success: bool = False
    error: Optional[str] = None
    parser_used: Optional[str] = None
    
    # File information
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    
    # Extracted content
    content: str = ""
    tables: List[TableData] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
    
    # Processing information
    parsing_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Raw data (for advanced processing)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling non-serializable fields."""
        result = {
            "success": self.success,
            "error": self.error,
            "parser_used": self.parser_used,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "content": self.content,
            "tables": [table.to_dict() for table in self.tables],
            "images": [
                {
                    "image_index": img.image_index,
                    "page_number": img.page_number,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "size_bytes": img.size_bytes,
                    "extracted_text": img.extracted_text
                }
                for img in self.images
            ],
            "parsing_time": self.parsing_time,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "raw_data": self.raw_data
        }
        
        # Handle metadata
        if self.metadata:
            result["metadata"] = {
                "title": self.metadata.title,
                "author": self.metadata.author,
                "subject": self.metadata.subject,
                "creator": self.metadata.creator,
                "creation_date": self.metadata.creation_date.isoformat() if self.metadata.creation_date else None,
                "modification_date": self.metadata.modification_date.isoformat() if self.metadata.modification_date else None,
                "page_count": self.metadata.page_count,
                "word_count": self.metadata.word_count,
                "language": self.metadata.language,
                "custom_properties": self.metadata.custom_properties
            }
        
        return result
    
    def get_text_summary(self, max_length: int = 500) -> str:
        """Get a summary of the extracted text."""
        if not self.content:
            return "No text content extracted"
        
        if len(self.content) <= max_length:
            return self.content
        
        return self.content[:max_length] + "..."
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the parsed content."""
        return {
            "character_count": len(self.content),
            "word_count": len(self.content.split()) if self.content else 0,
            "line_count": self.content.count('\n') + 1 if self.content else 0,
            "table_count": len(self.tables),
            "image_count": len(self.images),
            "has_metadata": self.metadata is not None
        }
    
    def is_valid(self) -> bool:
        """Check if the parse result contains meaningful content."""
        return (
            self.success and 
            (
                (self.content and len(self.content.strip()) > 0) or
                len(self.tables) > 0 or
                len(self.images) > 0
            )
        )