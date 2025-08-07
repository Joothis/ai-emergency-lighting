# AI Emergency Lighting Detection System

This project is an AI-powered vision system for detecting emergency lighting fixtures from electrical construction blueprints.

## Features

- **Advanced Detection**: Trained to identify various emergency lighting fixtures.
- **LLM Integration**: Utilizes Large Language Models for intelligent analysis of blueprints.
- **Data Storage**: Stores extracted data for further analysis, using MongoDB by default or local file system as a fallback.
- **Real-time Processing**: Employs background workers for efficient processing of blueprint files.

## Local Development Setup

### Prerequisites

- Python 3.10+
- MongoDB (optional, project will use local file storage if not available)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-emergency-lighting.git
   cd ai-emergency-lighting
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**

   Create a `.env` file in the project root and add the following variables:

   ```
   MONGO_URL=mongodb://localhost:27017/emergency_lighting # Optional: If not set or MongoDB is unavailable, local file storage will be used.
   GOOGLE_API_KEY=your_google_api_key
   UPLOAD_DIR=output/uploads
   OUTPUT_DIR=output
   ```

### Running the Application

1. **Start the FastAPI server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Start the Celery worker:**
   ```bash
   celery -A backend.worker worker --loglevel=info --concurrency=1
   ```

The API will be available at `http://localhost:8000`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/blueprints/upload` | POST | Upload a blueprint PDF for processing. |
| `/blueprints/result` | GET | Get the processing results for a given PDF. |
| `/blueprints/list` | GET | List all processed PDFs. |
| `/health` | GET | Health check endpoint. |

## Project Structure

```
.
├── backend/         # FastAPI application, Celery worker, and storage logic
├── models/          # Trained AI models
├── output/          # Output files (uploads, images, results, processing status, extracted content)
├── samples/         # Sample blueprint files
├── app.py           # Main application startup script
├── requirements.txt # Project dependencies
└── README.md        # This file
```
