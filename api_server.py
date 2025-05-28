#!/usr/bin/env python3
"""
FastAPI wrapper for the Universal Document Parser.
Provides REST API endpoints for document parsing and batch processing.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime

# FastAPI imports
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Add parser to path
sys.path.append(str(Path(__file__).parent))

from main import UniversalDocumentParser
from utils.config import ParserConfig
from models.parse_result import ParseResult


# Pydantic models for API
class ParseResponse(BaseModel):
    success: bool
    file_name: str
    parser_used: Optional[str] = None
    content_length: int = 0
    tables_count: int = 0
    images_count: int = 0
    processing_time: Optional[float] = None
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchParseResponse(BaseModel):
    job_id: str
    status: str
    files_count: int
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    files_processed: int
    files_total: int
    successful: int
    failed: int
    start_time: str
    results: Optional[List[ParseResponse]] = None
    error: Optional[str] = None


class ConfigRequest(BaseModel):
    enable_ocr: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    ocr_languages: List[str] = ["en", "ar"]
    ocr_engine: str = "easyocr"
    extract_tables: bool = True
    extract_images: bool = True


# Global variables
app = FastAPI(title="Universal Document Parser API", version="1.0.0")
parser_instance = None
active_jobs: Dict[str, Dict] = {}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_parser() -> UniversalDocumentParser:
    """Get or create parser instance."""
    global parser_instance
    if parser_instance is None:
        config = ParserConfig(
            enable_ocr=True,
            parallel_processing=True,
            max_workers=4
        )
        parser_instance = UniversalDocumentParser(config)
    return parser_instance


def parse_result_to_response(result: ParseResult, file_name: str) -> ParseResponse:
    """Convert ParseResult to API response."""
    return ParseResponse(
        success=result.success,
        file_name=file_name,
        parser_used=result.parser_used,
        content_length=len(result.content) if result.content else 0,
        tables_count=len(result.tables),
        images_count=len(result.images),
        processing_time=result.parsing_time,
        content=result.content[:1000] if result.content else None,  # Truncate for API
        error=result.error,
        metadata=result.metadata.to_dict() if result.metadata else None
    )


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Universal Document Parser API",
        "version": "1.0.0",
        "endpoints": {
            "parse": "/parse - Parse single document",
            "parse_batch": "/parse/batch - Parse multiple documents",
            "job_status": "/jobs/{job_id} - Get job status",
            "health": "/health - Health check",
            "supported_formats": "/formats - Get supported file formats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        parser = get_parser()
        supported_formats = parser.get_supported_formats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "supported_formats": supported_formats,
            "parsers_available": list(parser.parsers.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/formats")
async def get_supported_formats():
    """Get supported file formats."""
    try:
        parser = get_parser()
        return {
            "supported_formats": parser.get_supported_formats(),
            "parser_info": {
                name: parser_obj.get_supported_formats()
                for name, parser_obj in parser.parsers.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse", response_model=ParseResponse)
async def parse_document(
    file: UploadFile = File(...),
    include_content: bool = Query(False, description="Include full content in response"),
    max_content_length: int = Query(1000, description="Maximum content length to return")
):
    """Parse a single document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / file.filename
    
    try:
        # Save uploaded file
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse document
        parser = get_parser()
        result = parser.parse_file(temp_file)
        
        # Create response
        response = parse_result_to_response(result, file.filename)
        
        # Include full content if requested
        if include_content and result.content:
            response.content = result.content[:max_content_length]
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/parse/batch", response_model=BatchParseResponse)
async def parse_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    include_content: bool = Query(False, description="Include content in results")
):
    """Parse multiple documents in batch."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Too many files (max 100)")
    
    # Create job
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    active_jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "files_processed": 0,
        "files_total": len(files),
        "successful": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
        "results": [],
        "include_content": include_content
    }
    
    # Start background processing
    background_tasks.add_task(process_batch_job, job_id, files)
    
    return BatchParseResponse(
        job_id=job_id,
        status="processing",
        files_count=len(files),
        message=f"Batch processing started with {len(files)} files"
    )


async def process_batch_job(job_id: str, files: List[UploadFile]):
    """Process batch job in background."""
    job = active_jobs[job_id]
    temp_dir = tempfile.mkdtemp()
    
    try:
        parser = get_parser()
        
        # Save all files first
        temp_files = []
        for file in files:
            if file.filename:
                temp_file = Path(temp_dir) / file.filename
                content = await file.read()
                with open(temp_file, "wb") as f:
                    f.write(content)
                temp_files.append((temp_file, file.filename))
        
        # Process each file
        results = []
        for i, (temp_file, original_name) in enumerate(temp_files):
            try:
                result = parser.parse_file(temp_file)
                response = parse_result_to_response(result, original_name)
                
                # Include content if requested
                if job["include_content"] and result.content:
                    response.content = result.content
                
                results.append(response)
                
                if result.success:
                    job["successful"] += 1
                else:
                    job["failed"] += 1
                
            except Exception as e:
                # Create error response
                error_response = ParseResponse(
                    success=False,
                    file_name=original_name,
                    error=str(e)
                )
                results.append(error_response)
                job["failed"] += 1
            
            # Update progress
            job["files_processed"] = i + 1
            job["progress"] = (i + 1) / len(temp_files) * 100
        
        # Job completed
        job["status"] = "completed"
        job["results"] = [result.dict() for result in results]
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get batch job status."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        files_processed=job["files_processed"],
        files_total=job["files_total"],
        successful=job["successful"],
        failed=job["failed"],
        start_time=job["start_time"],
        results=job.get("results"),
        error=job.get("error")
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete completed job and free memory."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del active_jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


@app.post("/configure")
async def configure_parser(config: ConfigRequest):
    """Update parser configuration."""
    try:
        from utils.config import ParserConfig, OCRConfig, PDFConfig
        
        # Create new configuration
        ocr_config = OCRConfig(
            enabled=config.enable_ocr,
            languages=config.ocr_languages,
            engine=config.ocr_engine
        )
        
        pdf_config = PDFConfig(
            extract_tables=config.extract_tables,
            extract_images=config.extract_images
        )
        
        new_config = ParserConfig(
            enable_ocr=config.enable_ocr,
            parallel_processing=config.parallel_processing,
            max_workers=config.max_workers,
            ocr=ocr_config,
            pdf=pdf_config
        )
        
        # Recreate parser with new configuration
        global parser_instance
        parser_instance = UniversalDocumentParser(new_config)
        
        return {
            "message": "Configuration updated successfully",
            "config": config.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@app.get("/jobs")
async def list_jobs():
    """List all active jobs."""
    jobs_summary = []
    for job_id, job in active_jobs.items():
        jobs_summary.append({
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "files_total": job["files_total"],
            "start_time": job["start_time"]
        })
    
    return {
        "active_jobs": len(active_jobs),
        "jobs": jobs_summary
    }


@app.post("/parse/url")
async def parse_from_url(
    url: str,
    include_content: bool = Query(False),
    max_content_length: int = Query(1000)
):
    """Parse document from URL (for web-based documents)."""
    try:
        import requests
        
        # Download file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine filename from URL or Content-Disposition
        filename = url.split("/")[-1]
        if "." not in filename:
            filename += ".txt"  # Default extension
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / filename
        
        try:
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            # Parse document
            parser = get_parser()
            result = parser.parse_file(temp_file)
            
            # Create response
            api_response = parse_result_to_response(result, filename)
            
            if include_content and result.content:
                api_response.content = result.content[:max_content_length]
            
            return api_response
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL parsing failed: {str(e)}")


# For n8n integration
@app.post("/n8n/parse")
async def n8n_parse_endpoint(
    file: UploadFile = File(...),
    extract_tables: bool = Query(True),
    extract_images: bool = Query(True),
    enable_ocr: bool = Query(True)
):
    """
    N8N-optimized parsing endpoint.
    Returns structured data suitable for n8n workflows.
    """
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / (file.filename or "document")
    
    try:
        # Save file
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        
        # Parse with specific configuration
        config = ParserConfig(
            enable_ocr=enable_ocr,
            pdf=PDFConfig(
                extract_tables=extract_tables,
                extract_images=extract_images
            )
        )
        parser = UniversalDocumentParser(config)
        result = parser.parse_file(temp_file)
        
        # Format for n8n
        n8n_response = {
            "success": result.success,
            "fileName": file.filename,
            "parser": result.parser_used,
            "processingTime": result.parsing_time,
            "text": result.content,
            "wordCount": len(result.content.split()) if result.content else 0,
            "tables": [
                {
                    "index": i,
                    "headers": table.headers,
                    "rows": table.rows,
                    "rowCount": len(table.rows)
                }
                for i, table in enumerate(result.tables)
            ],
            "images": [
                {
                    "index": img.image_index,
                    "width": img.width,
                    "height": img.height,
                    "extractedText": img.extracted_text
                }
                for img in result.images
            ],
            "metadata": result.metadata.to_dict() if result.metadata else None,
            "error": result.error
        }
        
        return n8n_response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fileName": file.filename
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the API server."""
    print("Starting Universal Document Parser API Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker to maintain global state
    )


if __name__ == "__main__":
    main()