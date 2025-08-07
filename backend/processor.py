import fitz
import os
import cv2
import numpy as np
import easyocr
import json
import re
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import camelot
import pandas as pd
import google.generativeai as genai

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Configure Google Gemini (you'll need to set API key)
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=gemini_api_key)


@dataclass
class EmergencyLight:
    """Dataclass for emergency light detections."""
    symbol: str
    bounding_box: List[int]  # [x1, y1, x2, y2]
    text_nearby: List[str]
    source_sheet: str
    light_type: str  # "2x4_led", "wallpack", "exit_combo", "unknown"
    confidence: float

class BlueprintProcessor:
    """Processes blueprints to detect emergency lighting fixtures and extract relevant information."""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        
        # Patterns for emergency lighting
        self.emergency_patterns = {
            r'A\d+E?': 'emergency_light',
            r'E\d+': 'exit_sign', 
            r'EM\d*': 'emergency_light',
            r'EXIT\d*': 'exit_sign',
            r'W\d+': 'wallpack'
        }
        
        # Light type detection keywords
        self.light_type_keywords = {
            "2x4_led": ["2X4", "2'X4'", "RECESSED", "LED", "LUMINAIRE"],
            "wallpack": ["WALLPACK", "WALL PACK", "PHOTOCELL", "OUTDOOR"],
            "exit_combo": ["EXIT", "COMBO", "EMERGENCY", "COMBINATION"]
        }

    def process_pdf_to_images(self, pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
        """
        Convert PDF to high-resolution images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        doc = fitz.open(pdf_path)
        image_paths = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        
        doc.close()
        return image_paths

    def detect_emergency_lights_enhanced(self, image_path: str, page_num: int) -> List[EmergencyLight]:
        """
        Detects emergency lighting fixtures, including:
        - 2' X 4' RECESSED LED LUMINAIRE
        - WALLPACK WITH BUILT IN PHOTOCELL
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        detections = []
        
        # Multiple detection methods
        shaded_rectangles = self._detect_shaded_rectangles(image)
        outlined_shapes = self._detect_outlined_shapes(image)
        
        # Get OCR results
        ocr_results = self.reader.readtext(image_path)
        
        # Process all potential light locations
        all_candidates = shaded_rectangles + outlined_shapes
        
        for bbox in all_candidates:
            nearby_text, symbol = self._find_nearby_text_enhanced(bbox, ocr_results)
            
            if symbol or self._has_emergency_indicators(nearby_text):
                light_type = self._classify_light_type(nearby_text)
                confidence = self._calculate_confidence(bbox, symbol, nearby_text, light_type)
                
                detection = EmergencyLight(
                    symbol=symbol or "UNIDENTIFIED",
                    bounding_box=bbox,
                    text_nearby=nearby_text,
                    source_sheet=f"Page {page_num}",
                    light_type=light_type,
                    confidence=confidence
                )
                detections.append(detection)
        
        # Filter and deduplicate
        detections = self._filter_detections(detections)
        
        logger.info(f"Found {len(detections)} emergency lights on page {page_num}")
        return detections

    def _detect_shaded_rectangles(self, image: np.ndarray) -> List[List[int]]:
        """Detect shaded rectangular areas (emergency lights) using adaptive thresholding and morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up noise and connect broken parts
        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (emergency lights have reasonable size)
            if 100 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's rectangular-ish
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0 and w > 15 and h > 15:
                    
                    # Further check for "solidity" to ensure it's a filled rectangle
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.8:
                        candidates.append([x, y, x + w, y + h])
        
        return candidates

    def _detect_outlined_shapes(self, image: np.ndarray) -> List[List[int]]:
        """Detect outlined emergency lights using Canny edge detection and contour analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection with adjusted thresholds
        edges = cv2.Canny(blurred, 75, 200)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 15000:
                # Approximate polygon to find corners
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Check if contour approximates a rectangle or circle
                if len(approx) >= 4 and len(approx) <= 8:  # Rectangles have 4, circles can have more
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20: # Filter for reasonable aspect ratio and size
                        candidates.append([x, y, x + w, y + h])
        
        return candidates

    def _find_nearby_text_enhanced(self, bbox: List[int], ocr_results: List) -> Tuple[List[str], str]:
        """Associates nearby text with a given bounding box."""
        nearby_text = []
        symbol = ""
        
        # Larger search area for better text association
        margin = 75
        search_bbox = [
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            bbox[2] + margin,
            bbox[3] + margin
        ]
        
        for text_bbox, text, confidence in ocr_results:
            if confidence < 0.3:  # Very low confidence OCR
                continue
            
            # Calculate text center
            text_center_x = sum([point[0] for point in text_bbox]) / 4
            text_center_y = sum([point[1] for point in text_bbox]) / 4
            
            # Check if text is within search area
            if (search_bbox[0] <= text_center_x <= search_bbox[2] and
                search_bbox[1] <= text_center_y <= search_bbox[3]):
                
                clean_text = text.strip().upper()
                nearby_text.append(clean_text)
                
                # Check for emergency lighting symbols
                for pattern, light_type in self.emergency_patterns.items():
                    if re.match(pattern, clean_text):
                        symbol = clean_text
                        break
        
        return nearby_text, symbol

    def _has_emergency_indicators(self, nearby_text: List[str]) -> bool:
        """Check if nearby text indicates emergency lighting."""
        text_combined = " ".join(nearby_text).upper()
        
        indicators = [
            "EMERGENCY", "EXIT", "EM", "UNSWITCHED", 
            "BATTERY", "LED", "LUMINAIRE", "WALLPACK", "PHOTOCELL"
        ]
        
        return any(indicator in text_combined for indicator in indicators)

    def _classify_light_type(self, nearby_text: List[str]) -> str:
        """Classify the type of emergency light based on nearby text."""
        text_combined = " ".join(nearby_text).upper()
        
        for light_type, keywords in self.light_type_keywords.items():
            if any(keyword in text_combined for keyword in keywords):
                return light_type
        
        # Fallback for symbols
        if any(re.match(p, t) for p in [r'A\d+E?', r'E\d+', r'EM\d*'] for t in nearby_text):
            return "2x4_led"
        if any(re.match(r'W\d+', t) for t in nearby_text):
            return "wallpack"
        if any(re.match(r'EXIT\d*', t) for t in nearby_text):
            return "exit_combo"

        return "unknown"

    def _calculate_confidence(self, bbox: List[int], symbol: str, nearby_text: List[str], light_type: str) -> float:
        """Calculate confidence score for detection."""
        confidence = 0.2  # Base confidence
        
        # Symbol found
        if symbol and symbol != "UNIDENTIFIED":
            confidence += 0.4
        
        # Reasonable size
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if 20 < width < 400 and 20 < height < 400:
            confidence += 0.2
        
        # Emergency indicators found
        if self._has_emergency_indicators(nearby_text):
            confidence += 0.2
        
        # Specific light type identified
        if light_type != "unknown":
            confidence += 0.2
        
        return min(confidence, 1.0)

    def _filter_detections(self, detections: List[EmergencyLight]) -> List[EmergencyLight]:
        """Filter and remove duplicate detections."""
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for detection in detections:
            # Skip low confidence
            if detection.confidence < 0.5: # Increased threshold for better precision
                continue
            
            # Check for overlaps
            is_duplicate = False
            for existing in filtered:
                if self._boxes_overlap(detection.bounding_box, existing.bounding_box): # Use the overlap function
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered

    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> bool:
        """Check if bounding boxes overlap significantly."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        intersection_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(box1_area + box2_area - intersection_area)
        
        # Calculate IoU
        if union_area == 0:
            return False
        iou = intersection_area / union_area
        
        return iou > 0.3

    def extract_rulebook_with_llm(self, pdf_path: str) -> Dict[str, Any]:
        """Extract rulebook information using LLM for better accuracy."""
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            
            # Extract text from all pages
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            
            # Use LLM to extract structured information
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
Extract the rulebook from the following text. The rulebook should include the lighting schedule, notes, and any special fixtures.
If a lighting schedule is present, extract the 'SYMBOL' and 'DESCRIPTION' columns.
If general notes are present, extract them.
If special fixtures are mentioned, extract their details.

The output should be in JSON format with keys: "notes", "lighting_schedule", "special_fixtures".
For "lighting_schedule", each item should have "symbol" and "description".
For "notes", each item should have "text" and "source_sheet".
For "special_fixtures", provide a dictionary of fixture names and their descriptions.

{all_text}
"""
            response = model.generate_content(prompt)
            
            # Post-process to clean up JSON
            cleaned_json = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(cleaned_json)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._extract_rulebook_ocr(pdf_path)

    def _extract_rulebook_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback OCR-based rulebook extraction, with Camelot for tables."""
        notes = []
        lighting_schedule = []
        special_fixtures = {}
        
        try:
            # Use Camelot to extract tables from the PDF
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', edge_tol=500)
            for table in tables:
                df = table.df
                # Find header row and column indices for symbol and description
                header_row = df.iloc[0].astype(str).str.upper()
                symbol_col = -1
                desc_col = -1
                if 'SYMBOL' in header_row.values:
                    symbol_col = header_row[header_row == 'SYMBOL'].index[0]
                if 'DESCRIPTION' in header_row.values:
                    desc_col = header_row[header_row == 'DESCRIPTION'].index[0]

                if symbol_col != -1 and desc_col != -1:
                    for idx, row in df.iloc[1:].iterrows():
                        symbol = str(row[symbol_col]).strip()
                        description = str(row[desc_col]).strip()
                        if symbol and description and symbol != 'nan' and description != 'nan':
                            lighting_schedule.append({
                                "symbol": symbol,
                                "description": description,
                                "source_sheet": f"Page {table.page}"
                            })
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")

        # Use EasyOCR to extract notes and special fixtures
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1: # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            ocr_results = self.reader.readtext(img_array)
            
            current_notes_section = False
            for (bbox, text, conf) in ocr_results:
                clean_text = text.strip()
                if "GENERAL NOTES" in clean_text.upper() or "NOTES" in clean_text.upper():
                    current_notes_section = True
                    continue
                
                if current_notes_section and clean_text and conf > 0.5:
                    if re.match(r'^[A-Z0-9\s]+', clean_text) and len(clean_text) < 30:
                        current_notes_section = False
                    else:
                        notes.append({"text": clean_text, "source_sheet": f"Page {page_num + 1}"})
                
                # Simple special fixture detection (can be enhanced)
                if "FIXTURE" in clean_text.upper() and conf > 0.5:
                    match = re.search(r'([A-Z0-9]+)\s*-\s*(.*)', clean_text)
                    if match:
                        fixture_name = match.group(1).strip()
                        fixture_desc = match.group(2).strip()
                        special_fixtures[fixture_name] = fixture_desc

        doc.close()
        
        return {
            "notes": notes,
            "lighting_schedule": lighting_schedule,
            "special_fixtures": special_fixtures
        }

    def group_detections_with_llm(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Use LLM to intelligently group emergency lighting detections."""
        
        # Prepare detection data for LLM
        detection_data = []
        for det in detections:
            detection_data.append({
                "symbol": det.symbol,
                "nearby_text": det.text_nearby,
                "light_type": det.light_type,
                "confidence": det.confidence,
                "source_sheet": det.source_sheet # Changed from "source" to "source_sheet"
            })
        
        prompt = f"""
        Group these emergency lighting detections by symbol type and provide descriptions.
        Use the rulebook information to match symbols with their descriptions.
        
        Detections: {json.dumps(detection_data, indent=2)}
        
        Rulebook: {json.dumps(rulebook, indent=2)}
        
        Return JSON in this exact format:
        {{
            "A1": {{"count": 12, "description": "2x4 LED Emergency Fixture"}},
            "A1E": {{"count": 5, "description": "Exit/Emergency Combo Unit"}},
            "W": {{"count": 9, "description": "Wall-Mounted Emergency LED"}}
        }}
        """
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            # Post-process to clean up JSON
            cleaned_json = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(cleaned_json)
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        # Create a mapping from rulebook for descriptions
        rulebook_descriptions = {
            item["symbol"]: item["description"]
            for item in rulebook.get("lighting_schedule", [])
        }
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                description = rulebook_descriptions.get(symbol, f"{detection.light_type.replace('_', ' ').title()} Emergency Light")
                grouped[symbol] = {
                    "count": 1,
                    "description": description
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Creates annotated images showing detections."""
        viz_dir = os.path.join(output_dir, "annotations")
        os.makedirs(viz_dir, exist_ok=True)
        
        page_detections = {}
        for det in detections:
            page_num = int(det.source_sheet.split()[-1])
            if page_num not in page_detections:
                page_detections[page_num] = []
            page_detections[page_num].append(det)
        
        for page_num, image_path in enumerate(image_paths, 1):
            if page_num not in page_detections:
                continue
                
            image = Image.open(image_path).convert("RGB") # Ensure RGB for drawing
            draw = ImageDraw.Draw(image)
            
            for detection in page_detections[page_num]:
                bbox = detection.bounding_box
                
                # Color based on light type
                colors = {
                    "2x4_led": "red",
                    "wallpack": "blue", 
                    "exit_combo": "green",
                    "unknown": "orange"
                }
                color = colors.get(detection.light_type, "orange")
                
                # Draw bounding box
                draw.rectangle(bbox, outline=color, width=3)
                
                # Draw label
                label = f"{detection.symbol} ({detection.confidence:.2f})"
                draw.text((bbox[0], bbox[1] - 25), label, fill=color)
                
                # Draw light type
                type_label = detection.light_type.replace('_', ' ').title()
                draw.text((bbox[0], bbox[3] + 5), type_label, fill=color)
            
            # Save annotated image
            output_path = os.path.join(viz_dir, f"annotated_page_{page_num}.png")
            image.save(output_path)
            logger.info(f"Saved annotation: {output_path}")

    def run_full_competition_pipeline(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Main pipeline for the competition.
        """
        logger.info("Starting competition pipeline...")
        
        # 1. Convert PDF to images
        image_paths = self.process_pdf_to_images(pdf_path, os.path.join(output_dir, "images"))
        
        # 2. Detect emergency lights on all pages
        all_detections = []
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            detections = self.detect_emergency_lights_enhanced(image_path, page_num)
            all_detections.extend(detections)
        
        # 3. Extract rulebook
        rulebook = self.extract_rulebook_with_llm(pdf_path)
        
        # 4. Group detections
        grouped_results = self.group_detections_with_llm(all_detections, rulebook)
        
        # 5. Create annotated visualizations
        self.create_annotated_visualization(image_paths, all_detections, output_dir)
        
        # 6. Final result format
        final_result = {
            "pdf_name": os.path.basename(pdf_path),
            "status": "complete",
            "result": grouped_results,
            "rulebook": rulebook, # Include the rulebook in the final result
            "raw_detections": [det.__dict__ for det in all_detections] # Include raw detections for debugging/analysis
        }
        
        # Save results to JSON
        result_path = os.path.join(output_dir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(final_result, f, indent=4)
        
        # Save rulebook to JSON
        rulebook_path = os.path.join(output_dir, "rulebook.json")
        with open(rulebook_path, 'w') as f:
            json.dump(rulebook, f, indent=4)
        
        logger.info(f"Competition pipeline finished. Results saved to {result_path}")
        return final_result