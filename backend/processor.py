import fitz  # PyMuPDF
import os
from PIL import Image
import cv2
import numpy as np
import easyocr
import json

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

def process_pdf_to_images(pdf_path: str, output_dir: str):
    """
    Converts each page of a PDF document to a high-resolution PNG image.

    Args:
        pdf_path (str): The absolute path to the PDF file.
        output_dir (str): The absolute path to the directory where images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render page to a pixmap (image) at a high resolution
        pix = page.get_pixmap(dpi=300)
        image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths

def extract_text_with_easyocr(image_path: str):
    """
    Extracts text from an image using easyocr.

    Args:
        image_path (str): The absolute path to the image file.

    Returns:
        list: A list of tuples, where each tuple contains the bounding box and the extracted text.
    """
    return reader.readtext(image_path)

def detect_emergency_lights(image_path: str):
    """
    Detects shaded rectangles in an image, representing emergency lights.

    Args:
        image_path (str): The absolute path to the image file.

    Returns:
        list: A list of bounding boxes for detected emergency lights.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a simple threshold to find dark areas
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_lights = []
    for contour in contours:
        # Filter contours based on area to remove noise
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detected_lights.append([x, y, x + w, y + h])

    return detected_lights

def load_rulebook(rulebook_path: str):
    """
    Loads the rulebook from a JSON file.

    Args:
        rulebook_path (str): The absolute path to the rulebook.json file.

    Returns:
        dict: The loaded rulebook.
    """
    with open(rulebook_path, 'r') as f:
        return json.load(f)

def group_detections(detections: list, rulebook: dict):
    """
    Groups the detections by symbol and maps them to the rulebook descriptions.

    Args:
        detections (list): A list of detections, where each detection is a dictionary.
        rulebook (dict): The rulebook containing symbol descriptions.

    Returns:
        dict: A dictionary containing the grouped results.
    """
    grouped_results = {}
    for detection in detections:
        symbol = detection.get("symbol")
        if not symbol:
            continue

        if symbol not in grouped_results:
            grouped_results[symbol] = {"count": 0, "description": ""}

        grouped_results[symbol]["count"] += 1

        # Find the description in the rulebook
        for rule in rulebook.get("rulebook", []):
            if rule.get("type") == "table_row" and rule.get("symbol") == symbol:
                grouped_results[symbol]["description"] = rule.get("description")
                break

    return grouped_results

def extract_tables_with_easyocr(image_path: str):
    """
    Extracts tables from an image using easyocr.

    Args:
        image_path (str): The absolute path to the image file.

    Returns:
        list: A list of tables, where each table is a list of rows.
    """
    # This is a simplified implementation and can be improved
    results = reader.readtext(image_path)
    
    # Group results by y-coordinate to identify rows
    rows = {}
    for (bbox, text, prob) in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        # Group by a tolerance of 20 pixels
        row_key = int(y_center // 20)
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append((bbox, text))

    # Sort rows by y-coordinate
    sorted_rows = [rows[key] for key in sorted(rows.keys())]

    # Sort cells within each row by x-coordinate
    tables = []
    for row in sorted_rows:
        sorted_row = sorted(row, key=lambda x: x[0][0][0])
        tables.append([text for bbox, text in sorted_row])

    return tables

def extract_general_notes(image_path: str):
    """
    Extracts general notes from an image using easyocr.

    Args:
        image_path (str): The absolute path to the image file.

    Returns:
        list: A list of extracted notes.
    """
    results = reader.readtext(image_path, detail=0, paragraph=True)
    notes = []
    for text in results:
        if "NOTE" in text.upper() or "GENERAL" in text.upper():
            notes.append(text)
    return notes
