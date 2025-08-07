from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from backend.worker import process_blueprint_task
import os
import shutil
import pymongo
import json
import logging
from datetime import datetime
from typing import Optional

from backend.storage import MongoStorage, FileStorage, Storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emergency Lighting Detection API",
    description="AI Vision system for detecting emergency lighting from construction blueprints",
    version="1.0.0"
)

# Add CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Environment variables for production
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "output/uploads")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "emergency_lighting")

# Initialize storage based on MONGO_URL presence
storage: Storage
if MONGO_URL and MONGO_URL != "mongodb://localhost:27017/": # Check if MONGO_URL is explicitly set
    storage = MongoStorage(MONGO_URL, DB_NAME)
    if not storage.connect():
        logger.warning("Could not connect to MongoDB. Falling back to file storage.")
        storage = FileStorage(os.getenv("OUTPUT_DIR", "output"))
else:
    storage = FileStorage(os.getenv("OUTPUT_DIR", "output"))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Emergency Lighting Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "POST /blueprints/upload",
            "result": "GET /blueprints/result?pdf_name=<name>",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    storage_status = "connected" if storage.connect() else "not_found"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": storage_status
    }

@app.post("/blueprints/upload")
async def upload_blueprint(file: UploadFile = File(...)):
    """
    Upload a PDF blueprint and initiate background processing.
    """
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        pdf_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start background processing
        task = process_blueprint_task.delay(file.filename)
        
        # Store processing status
        storage.update_pdf_processing_status(
            pdf_name=file.filename,
            status="processing",
            task_id=task.id
        )
        
        logger.info(f"Started processing {file.filename} with task ID: {task.id}")
        
        return {
            "status": "uploaded",
            "pdf_name": file.filename,
            "message": "Processing started in background."
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blueprints/result")
async def get_blueprint_result(pdf_name: str = Query(..., description="Name of the uploaded PDF")):
    """
    Retrieve processing result by PDF name.
    """
    try:
        # Query storage for result
        row = storage.get_pdf_processing_status(pdf_name)
        
        if not row:
            raise HTTPException(status_code=404, detail=f"PDF '{pdf_name}' not found")
        
        status = row.get("status")
        result_data = row.get("result")
        task_id = row.get("task_id")
        
        if status == "complete" and result_data:
            # Return completed result
            # result_data is already a dict if from FileStorage, string if from MongoStorage
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            return {
                "pdf_name": pdf_name,
                "status": "complete",
                "result": result_data.get("grouped_results", {})
            }
        
        elif status == "processing":
            # Check actual task status
            if task_id:
                task_result = AsyncResult(task_id)
                if task_result.ready():
                    if task_result.successful():
                        # Update storage with completed result
                        full_result = task_result.get()
                        storage.update_pdf_processing_status(
                            pdf_name=pdf_name,
                            status="complete",
                            result=full_result
                        )
                        
                        # Also store extracted content
                        store_extracted_content(pdf_name, full_result)
                        
                        return {
                            "pdf_name": pdf_name,
                            "status": "complete", 
                            "result": full_result.get("grouped_results", {})
                        }
                    else:
                        return {
                            "pdf_name": pdf_name,
                            "status": "failed",
                            "message": f"Processing failed: {task_result.info}"
                        }
            
            return {
                "pdf_name": pdf_name,
                "status": "in_progress",
                "message": "Processing is still in progress. Please try again later."
            }
        
        else:
            return {
                "pdf_name": pdf_name,
                "status": status,
                "message": f"Current status: {status}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Result retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def store_extracted_content(pdf_name: str, result_data: dict):
    """Store extracted content."""
    try:
        content_to_store = []
        rulebook = result_data.get("rulebook", {})
        for entry in rulebook.get("rulebook", {}).get("rulebook", []):
            content_to_store.append({
                "type": entry.get("type"),
                "symbol": entry.get("symbol", ""),
                "description": entry.get("description", ""),
                "content": entry.get("text", entry.get("description", "")),
                "source_sheet": entry.get("source_sheet", ""),
                "created_at": datetime.now().isoformat()
            })
        storage.store_extracted_content(pdf_name, content_to_store)
        
    except Exception as e:
        logger.error(f"Error storing extracted content: {e}")

@app.get("/blueprints/content/{pdf_name}")
async def get_extracted_content(pdf_name: str):
    """Get all extracted content for a PDF."""
    try:
        content = storage.get_extracted_content(pdf_name)
        
        if not content:
            raise HTTPException(status_code=404, detail=f"No content found for PDF '{pdf_name}'")
        
        return {
            "pdf_name": pdf_name,
            "extracted_content": content,
            "total_entries": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blueprints/list")
async def list_processed_pdfs():
    """List all processed PDFs and their status."""
    try:
        pdfs = storage.list_processed_pdfs()
        
        return {
            "processed_pdfs": pdfs,
            "total": len(pdfs)
        }
        
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )