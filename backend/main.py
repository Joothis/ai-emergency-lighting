from fastapi import FastAPI, File, UploadFile
from celery.result import AsyncResult
from backend.worker import process_blueprint_task
import os
import shutil

app = FastAPI()

# Define a directory to store uploaded files
UPLOAD_DIR = "C:\Users\jooth\Desktop\Projects\New folder\ai-emergency-lighting\output\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/blueprints/upload")
async def upload_blueprint(file: UploadFile = File(...)):
    """
    Uploads a PDF and initiates background processing.
    """
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded PDF
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Start the background processing task
    task = process_blueprint_task.delay(file.filename)

    return {"status": "uploaded", "pdf_name": file.filename, "task_id": task.id}

@app.get("/blueprints/result")
async def get_blueprint_result(task_id: str):
    """
    Retrieves the result of a processing task.
    """
    task_result = AsyncResult(task_id)

    if task_result.ready():
        if task_result.successful():
            return {"status": "complete", "result": task_result.get()}
        else:
            return {"status": "failed", "error": str(task_result.info)}
    else:
        return {"status": "in_progress"}