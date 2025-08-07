from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from backend.worker import process_blueprint_task
import os
import shutil
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional

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
DB_PATH = os.getenv("DB_PATH", "emergency_lighting.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Database setup
def init_database():
    """Initialize SQLite database for storing PDF processing results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_processing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            task_id TEXT,
            result TEXT,  -- JSON string
            rulebook TEXT,  -- JSON string  
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT NOT NULL,
            content_type TEXT NOT NULL,  -- 'note' or 'table_row'
            symbol TEXT,
            description TEXT,
            content TEXT,  -- Full extracted text
            source_sheet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pdf_name) REFERENCES pdf_processing (pdf_name)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if os.path.exists(DB_PATH) else "not_found"
    }

@app.post("/blueprints/upload")
async def upload_blueprint(file: UploadFile = File(...)):
    """
    Upload a PDF blueprint and initiate background processing.
    Matches competition requirement exactly.
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
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO pdf_processing (pdf_name, status, task_id, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (file.filename, "processing", task.id, datetime.now()))
        conn.commit()
        conn.close()
        
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
    Matches competition requirement exactly.
    """
    try:
        # Query database for result
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT status, result, task_id FROM pdf_processing 
            WHERE pdf_name = ?
        ''', (pdf_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"PDF '{pdf_name}' not found")
        
        status, result_json, task_id = row
        
        if status == "complete" and result_json:
            # Return completed result
            result_data = json.loads(result_json)
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
                        # Update database with completed result
                        result_data = task_result.get()
                        
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE pdf_processing 
                            SET status = ?, result = ?, updated_at = ?
                            WHERE pdf_name = ?
                        ''', ("complete", json.dumps(result_data), datetime.now(), pdf_name))
                        conn.commit()
                        conn.close()
                        
                        # Also store extracted content in database
                        store_extracted_content(pdf_name, result_data)
                        
                        return {
                            "pdf_name": pdf_name,
                            "status": "complete", 
                            "result": result_data.get("grouped_results", {})
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
    """Store extracted content in database as required by competition."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Clear existing content for this PDF
        cursor.execute('DELETE FROM extracted_content WHERE pdf_name = ?', (pdf_name,))
        
        # Store rulebook content
        rulebook = result_data.get("rulebook", {})
        for entry in rulebook.get("rulebook", {}).get("rulebook", []):
            cursor.execute('''
                INSERT INTO extracted_content 
                (pdf_name, content_type, symbol, description, content, source_sheet)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pdf_name,
                entry.get("type"),
                entry.get("symbol", ""),
                entry.get("description", ""),
                entry.get("text", entry.get("description", "")),
                entry.get("source_sheet", "")
            ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error storing extracted content: {e}")

@app.get("/blueprints/content/{pdf_name}")
async def get_extracted_content(pdf_name: str):
    """Get all extracted content for a PDF from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT content_type, symbol, description, content, source_sheet, created_at
            FROM extracted_content 
            WHERE pdf_name = ?
            ORDER BY created_at
        ''', (pdf_name,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            raise HTTPException(status_code=404, detail=f"No content found for PDF '{pdf_name}'")
        
        content = []
        for row in rows:
            content.append({
                "type": row[0],
                "symbol": row[1],
                "description": row[2], 
                "content": row[3],
                "source_sheet": row[4],
                "created_at": row[5]
            })
        
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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pdf_name, status, created_at, updated_at
            FROM pdf_processing 
            ORDER BY updated_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        pdfs = []
        for row in rows:
            pdfs.append({
                "pdf_name": row[0],
                "status": row[1],
                "created_at": row[2],
                "updated_at": row[3]
            })
        
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