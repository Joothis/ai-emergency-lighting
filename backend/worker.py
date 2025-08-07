from celery import Celery
import os
import logging
import fitz
from backend.enhanced_competition_processor import CompetitionProcessor

logger = logging.getLogger(__name__)

# Configure Celery
celery_app = Celery(
    "emergency_lighting_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

# Production-ready Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,
    task_time_limit=30 * 60,  # 30 minute timeout
    task_soft_time_limit=25 * 60,  # 25 minute soft timeout
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Directories
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "output/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

@celery_app.task(bind=True)
def process_blueprint_task(self, pdf_name: str):
    """
    Enhanced processing task for competition requirements.
    """
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Starting PDF processing', 'progress': 10}
        )
        
        pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing {pdf_name}")
        
        # Initialize processor
        processor = CompetitionProcessor()
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Converting PDF to images', 'progress': 20}
        )
        
        # Convert PDF to images
        image_paths = processor.process_pdf_to_images(pdf_path, os.path.join(OUTPUT_DIR, "images"))
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Detecting emergency lights', 'progress': 40}
        )
        
        # Detect emergency lights on all pages
        all_detections = []
        for i, image_path in enumerate(image_paths):
            page_detections = processor.detect_emergency_lights_enhanced(image_path, i + 1)
            all_detections.extend(page_detections)
        
        # Update progress  
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Extracting rulebook information', 'progress': 60}
        )
        
        # Extract rulebook with LLM
        rulebook = processor.extract_rulebook_with_llm(pdf_path)
        
        # Update progress
        self.update_state(
            state='PROCESSING', 
            meta={'status': 'Grouping detections', 'progress': 80}
        )
        
        # Group detections using LLM
        grouped_results = processor.group_detections_with_llm(all_detections, rulebook)
        
        # Create visualizations
        processor.create_annotated_visualization(image_paths, all_detections, OUTPUT_DIR)
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Finalizing results', 'progress': 95}
        )
        
        # Prepare final result in competition format
        result = {
            "grouped_results": grouped_results,
            "rulebook": {
                "rulebook": [
                    {
                        "type": "note",
                        "text": note["text"],
                        "source_sheet": note.get("source", "Unknown")
                    }
                    for note in rulebook.get("notes", [])
                ] + [
                    {
                        "type": "table_row",
                        "symbol": entry["symbol"],
                        "description": entry["description"],
                        "source_sheet": "Lighting Schedule"
                    }
                    for entry in rulebook.get("lighting_schedule", [])
                ]
            },
            "raw_detections": [
                {
                    "symbol": det.symbol,
                    "bounding_box": det.bounding_box,
                    "text_nearby": det.text_nearby,
                    "source_sheet": det.source_sheet,
                    "light_type": det.light_type,
                    "confidence": det.confidence
                }
                for det in all_detections
            ],
            "summary_stats": {
                "total_detections": len(all_detections),
                "unique_symbols": len(grouped_results),
                "2x4_led_count": sum(1 for det in all_detections if det.light_type == "2x4_led"),
                "wallpack_count": sum(1 for det in all_detections if det.light_type == "wallpack")
            }
        }
        
        logger.info(f"Processing complete for {pdf_name}: {len(all_detections)} detections found")
        return result
        
    except Exception as e:
        logger.error(f"Processing failed for {pdf_name}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'Processing failed'}
        )
        raise e

