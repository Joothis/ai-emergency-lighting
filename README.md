# Emergency Lighting Detection System - Competition Entry

AI Vision system for detecting emergency lighting fixtures from electrical construction blueprints.

## 🎯 Competition Features

- **Advanced Detection**: Specifically trained for 2'X4' recessed LED luminaires and wallpacks with photocells
- **LLM Integration**: Uses Google Gemini for intelligent grouping and rulebook extraction  
- **Database Storage**: Stores extracted content against PDF names as required
- **Production Ready**: Deployed on Render.com with Redis backend
- **Real-time Processing**: Background processing with status tracking

## 🚀 Quick Demo

**Live API**: https://your-app.onrender.com
**API Docs**: https://your-app.onrender.com/docs

### Upload & Process
```bash
curl -X POST "https://your-app.onrender.com/blueprints/upload" \
     -F "file=@blueprint.pdf"
# Response: {"status": "uploaded", "pdf_name": "blueprint.pdf", "message": "Processing started in background."}
```

### Get Results  
```bash
curl "https://your-app.onrender.com/blueprints/result?pdf_name=blueprint.pdf"
# Response: {"pdf_name": "blueprint.pdf", "status": "complete", "result": {...}}
```

## 🔧 Local Development

1. **Setup Environment**
   ```bash
   python scripts/setup_competition.py
   ```

2. **Google API Key Set**
   Your Google API key has been set in the `.env` file.

3. **Start Services**
   ```bash
   ./start_competition.sh
   ```

## 📊 Detection Capabilities

- **Emergency Lights**: Shaded rectangular areas with symbols (A1, A1E, etc.)
- **2'X4' LED Fixtures**: Recessed luminaires with battery backup  
- **Wallpack Fixtures**: Outdoor emergency lighting with photocells
- **Exit Signs**: Combination exit/emergency units
- **Symbol Association**: Links symbols with nearby text and specifications

## 🗄️ Database Schema

**PDF Processing Table**:
- pdf_name, status, task_id, result, created_at, updated_at

**Extracted Content Table**: 
- pdf_name, content_type, symbol, description, content, source_sheet

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/blueprints/upload` | POST | Upload PDF and start processing |
| `/blueprints/result?pdf_name=X` | GET | Get processing results |
| `/blueprints/list` | GET | List all processed PDFs |
| `/blueprints/content/{pdf_name}` | GET | Get extracted database content |
| `/health` | GET | System health check |

## 🎨 Example Output

```json
{
  "pdf_name": "E2.4.pdf",
  "status": "complete", 
  "result": {
    "A1": {"count": 12, "description": "2x4 LED Emergency Fixture"},
    "A1E": {"count": 5, "description": "Exit/Emergency Combo Unit"}, 
    "W": {"count": 9, "description": "Wall-Mounted Emergency LED"}
  }
}
```

## 🏗️ Architecture

```
PDF Upload → FastAPI → Celery Worker → AI Processing → Database Storage
                ↓           ↓              ↓
            Redis Queue → CV Detection → Results API
                        → OCR Extract
                        → LLM Grouping
```

## 📦 Deployment (Render.com)

1. **Push to GitHub**
2. **Connect to Render**  
3. **Use render.yaml config**
4. **Set environment variables**
5. **Deploy!**

## 📸 Competition Deliverables

1. ✅ **Screenshot**: `output/annotations/annotated_page_1.png`
2. ✅ **Hosted API**: https://your-app.onrender.com
3. ✅ **Postman Collection**: `postman/competition_collection.json`
4. ✅ **GitHub Repo**: This repository
5. ✅ **Demo Video**: [Link to 2-minute demo]

## 🏆 Technical Highlights

- **Multi-method Detection**: Combines shape detection, OCR, and pattern matching
- **LLM Enhancement**: GPT-powered text extraction and intelligent grouping
- **Production Scale**: Redis queuing, database persistence, error handling
- **Competition Compliance**: Exact API specification matching
- **Visual Validation**: Annotated images for verification

## 🔍 Detection Process

1. **PDF → Images**: High-resolution conversion (300 DPI)
2. **Shape Detection**: Find shaded rectangles using multiple CV methods
3. **OCR Analysis**: Extract text and associate with shapes
4. **Symbol Matching**: Regex patterns for emergency lighting symbols
5. **LLM Processing**: Intelligent grouping and description matching
6. **Database Storage**: Persist all extracted content
7. **Result Compilation**: Format for competition requirements

## 📞 Support

For questions about this competition entry:
- **Email**: your.email@domain.com  
- **Demo**: [YouTube/Loom link]
- **Repository**: [GitHub link]

Built with ❤️ for the AI Vision Competition
