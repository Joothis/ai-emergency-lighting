# AI Emergency Lighting Detection

This project detects emergency lighting fixtures in PDF blueprints.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Redis:**
   - [Redis Installation Guide](https://redis.io/topics/installation)

## Usage

1. **Start the Celery worker:**
   ```bash
   celery -A backend.worker worker --loglevel=info
   ```

2. **Run the FastAPI server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

3. **Use the API:**
   - **`POST /blueprints/upload`**: Upload a PDF to start processing.
   - **`GET /blueprints/result?task_id=<task_id>`**: Check the status and get the results."# ai-emergency-lighting" 
