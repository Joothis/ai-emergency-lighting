from celery import Celery
import os
import shutil
from backend.processor import (
    process_pdf_to_images,
    detect_emergency_lights,
    extract_text_with_easyocr,
    extract_tables_with_easyocr,
    extract_general_notes,
    group_detections,
)

# Configure Celery
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Define directories
UPLOAD_DIR = r"C:\Users\jooth\Desktop\Projects\New folder\ai-emergency-lighting\output\uploads"
IMAGE_DIR = r"C:\Users\jooth\Desktop\Projects\New folder\ai-emergency-lighting\output\images"

@celery_app.task
def process_blueprint_task(pdf_name: str):
    """
    Celery task to process a PDF blueprint in the background.
    """
    pdf_path = os.path.join(UPLOAD_DIR, pdf_name)

    # 1. Convert PDF to images
    image_paths = process_pdf_to_images(pdf_path, os.path.join(IMAGE_DIR, pdf_name))

    # 2. Dynamically build the rulebook
    rulebook = {"rulebook": []}
    for i, image_path in enumerate(image_paths):
        # Extract tables
        tables = extract_tables_with_easyocr(image_path)
        for table in tables:
            # Simple heuristic to identify lighting schedule
            if "SYMBOL" in [cell.upper() for cell in table]:
                header = table
                for row in tables:
                    if len(row) == len(header):
                        rulebook["rulebook"].append({
                            "type": "table_row",
                            "symbol": row[0],
                            "description": row[1],
                            "source_sheet": f"{pdf_name} - Page {i + 1}"
                        })

        # Extract notes
        notes = extract_general_notes(image_path)
        for note in notes:
            rulebook["rulebook"].append({
                "type": "note",
                "text": note,
                "source_sheet": f"{pdf_name} - Page {i + 1}"
            })

    # 3. Process each image for detections
    all_detections = []
    for i, image_path in enumerate(image_paths):
        detected_lights = detect_emergency_lights(image_path)
        extracted_text = extract_text_with_easyocr(image_path)

        for light_bbox in detected_lights:
            nearby_text = []
            symbol = ""
            for text_bbox, text, _ in extracted_text:
                text_center_x = (text_bbox[0][0] + text_bbox[1][0]) / 2
                text_center_y = (text_bbox[0][1] + text_bbox[2][1]) / 2

                if (light_bbox[0] < text_center_x < light_bbox[2] and
                    light_bbox[1] < text_center_y < light_bbox[3]):
                    nearby_text.append(text)
                    if len(text) <= 3 and any(char.isdigit() for char in text):
                        symbol = text

            all_detections.append({
                "symbol": symbol,
                "bounding_box": light_bbox,
                "text_nearby": nearby_text,
                "source_sheet": f"{pdf_name} - Page {i + 1}"
            })

    # 4. Group the detections
    grouped_results = group_detections(all_detections, rulebook)

    return {"grouped_results": grouped_results, "rulebook": rulebook, "raw_detections": all_detections}
