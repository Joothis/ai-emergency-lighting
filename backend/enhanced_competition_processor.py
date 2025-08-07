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

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Configure Google Gemini (you'll need to set API key)


@dataclass
class EmergencyLight:
    """Dataclass for emergency light detections."""
    symbol: str
    bounding_box: List[int]  # [x1, y1, x2, y2]
    text_nearby: List[str]
    source_sheet: str
    light_type: str  # "2x4_led", "wallpack", "exit_combo", "unknown"
    confidence: float

class CompetitionProcessor:
    """Enhanced processor specifically for competition requirements."""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        
        # Competition-specific patterns for emergency lighting
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
        Enhanced detection specifically for competition requirements:
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
                if len(approx) >= 4 and len(approx) <= 6:  # Rectangles have 4, circles can have more
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20: # Filter for reasonable aspect ratio and size
                        candidates.append([x, y, x + w, y + h])
        
        return candidates

    def _find_nearby_text_enhanced(self, bbox: List[int], ocr_results: List) -> Tuple[List[str], str]:
        """Enhanced text association for competition requirements."""
        nearby_text = []
        symbol = ""
        
        # Larger search area for better text association
        margin = 50
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
            "BATTERY", "LED", "LUMINAIRE", "WALLPACK"
        ]
        
        return any(indicator in text_combined for indicator in indicators)

    def _classify_light_type(self, nearby_text: List[str]) -> str:
        """Classify the type of emergency light based on nearby text."""
        text_combined = " ".join(nearby_text).upper()
        
        for light_type, keywords in self.light_type_keywords.items():
            if any(keyword in text_combined for keyword in keywords):
                return light_type
        
        return "unknown"

    def _calculate_confidence(self, bbox: List[int], symbol: str, nearby_text: List[str], light_type: str) -> float:
        """Calculate confidence score for detection."""
        confidence = 0.3  # Base confidence
        
        # Symbol found
        if symbol and symbol != "UNIDENTIFIED":
            confidence += 0.3
        
        # Reasonable size
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if 20 < width < 300 and 20 < height < 300:
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
            if detection.confidence < 0.4:
                continue
            
            # Check for overlaps
            is_duplicate = False
            for existing in filtered:
                if self._boxes_overlap(detection.bounding_box, existing.bounding_box):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered

    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> bool:
        """Check if bounding boxes overlap significantly."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        overlap_ratio = intersection / min(area1, area2)
        return overlap_ratio > 0.3

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
            prompt = f"""
            Extract emergency lighting information from this electrical blueprint text.
            Find and structure:
            1. General notes about emergency lighting
            2. Lighting schedule table entries with symbols and descriptions
            3. Any specifications for 2'X4' recessed LED luminaires
            4. Any specifications for wallpacks with photocells
            
            Text content:
            {all_text[:4000]}  # Limit for API
            
            Return as JSON with this structure:
            {{
                "notes": [
                    {{"text": "note content", "source": "page info"}}
                ],
                "lighting_schedule": [
                    {{"symbol": "A1", "description": "fixture description", "specs": "technical specs"}}
                ],
                "special_fixtures": {{
                    "2x4_led": ["specifications"],
                    "wallpack": ["specifications"]
                }}
            }}
            """
            
            # Mock LLM response for rulebook extraction
            except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
        
        # Fallback to traditional OCR method
        return self._extract_rulebook_ocr(pdf_path)

    def _extract_rulebook_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback OCR-based rulebook extraction, with Camelot for tables."""
        notes = []
        lighting_schedule = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            # Try Camelot for table extraction first
            try:
                # Camelot works best on the PDF directly
                tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream', edge_tol=500)
                for table in tables:
                    df = table.df
                    # Assuming the first row is header, and looking for 'SYMBOL' and 'DESCRIPTION'
                    # This part might need further refinement based on actual table structures
                    header_row = df.iloc[0].astype(str).str.upper()
                    symbol_col = -1
                    desc_col = -1
                    
                    if 'SYMBOL' in header_row.values:
                        symbol_col = header_row[header_row == 'SYMBOL'].index[0]
                    if 'DESCRIPTION' in header_row.values:
                        desc_col = header_row[header_row == 'DESCRIPTION'].index[0]

                    if symbol_col != -1 and desc_col != -1:
                        for idx, row in df.iloc[1:].iterrows(): # Skip header
                            symbol = str(row[symbol_col]).strip()
                            description = str(row[desc_col]).strip()
                            if symbol and description and symbol != 'nan' and description != 'nan':
                                lighting_schedule.append({
                                    "symbol": symbol,
                                    "description": description,
                                    "source_sheet": f"Page {page_num + 1}"
                                })
            except Exception as e:
                logger.warning(f"Camelot table extraction failed on page {page_num + 1}: {e}")

            # Fallback to EasyOCR for notes and if Camelot fails for tables
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
                    if re.match(r'^[A-Z0-9\s]+

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
                "source": det.source_sheet
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
            # Mock LLM response for grouping
            mock_grouped_results = self._simple_grouping(detections, rulebook)
            return mock_grouped_results
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                grouped[symbol] = {
                    "count": 1,
                    "description": f"{detection.light_type.replace('_', ' ').title()} Emergency Light"
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Create annotated images showing detections for competition submission."""
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
                
            image = Image.open(image_path)
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
, clean_text) and len(clean_text) < 30: # Likely a new heading
                        current_notes_section = False
                    else:
                        notes.append({"text": clean_text, "source": f"Page {page_num + 1}"})

            # Heuristic for lighting schedule table: look for headers and then parse rows
            # This is a simplified approach; a more robust solution would involve table detection libraries
            header_found = False
            symbol_col_x = -1
            desc_col_x = -1
            
            # Collect all text on the page, sorted by y-coordinate
            page_texts = sorted([(b[0][1], b[0][0], t.strip(), c) for (b, t, c) in ocr_results if c > 0.5])

            for y, x, text, conf in page_texts:
                clean_text = text.upper()
                if not header_found:
                    if "SYMBOL" in clean_text and "DESCRIPTION" in clean_text:
                        header_found = True
                        # Attempt to find column x-coordinates
                        for (bbox, t, c) in ocr_results:
                            if "SYMBOL" in t.upper():
                                symbol_col_x = bbox[0][0]
                            if "DESCRIPTION" in t.upper():
                                desc_col_x = bbox[0][0]
                        continue
                
                if header_found:
                    # Try to parse rows based on column positions
                    # This is a very basic approach and assumes consistent column alignment
                    if symbol_col_x != -1 and desc_col_x != -1:
                        # Find text that aligns with symbol column
                        symbol_text = ""
                        description_text = ""
                        
                        for (bbox, t, c) in ocr_results:
                            if abs(bbox[0][0] - symbol_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                symbol_text = t.strip()
                            if abs(bbox[0][0] - desc_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                description_text = t.strip()
                        
                        if symbol_text and description_text:
                            lighting_schedule.append({
                                "symbol": symbol_text,
                                "description": description_text,
                                "source_sheet": f"Page {page_num + 1}"
                            })
                            
        doc.close()
        
        return {
            "notes": notes,
            "lighting_schedule": lighting_schedule,
            "special_fixtures": {} # OCR alone might not easily extract this
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
                "source": det.source_sheet
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
            # Mock LLM response for grouping
            mock_grouped_results = self._simple_grouping(detections, rulebook)
            return mock_grouped_results
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                grouped[symbol] = {
                    "count": 1,
                    "description": f"{detection.light_type.replace('_', ' ').title()} Emergency Light"
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Create annotated images showing detections for competition submission."""
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
                
            image = Image.open(image_path)
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
, clean_text) and len(clean_text) < 30: # Likely a new heading
                        current_notes_section = False
                    else:
                        notes.append({"text": clean_text, "source": f"Page {page_num + 1}"})

            # Heuristic for lighting schedule table: look for headers and then parse rows
            # This is a simplified approach; a more robust solution would involve table detection libraries
            header_found = False
            symbol_col_x = -1
            desc_col_x = -1
            
            # Collect all text on the page, sorted by y-coordinate
            page_texts = sorted([(b[0][1], b[0][0], t.strip(), c) for (b, t, c) in ocr_results if c > 0.5])

            for y, x, text, conf in page_texts:
                clean_text = text.upper()
                if not header_found:
                    if "SYMBOL" in clean_text and "DESCRIPTION" in clean_text:
                        header_found = True
                        # Attempt to find column x-coordinates
                        for (bbox, t, c) in ocr_results:
                            if "SYMBOL" in t.upper():
                                symbol_col_x = bbox[0][0]
                            if "DESCRIPTION" in t.upper():
                                desc_col_x = bbox[0][0]
                        continue
                
                if header_found:
                    # Try to parse rows based on column positions
                    # This is a very basic approach and assumes consistent column alignment
                    if symbol_col_x != -1 and desc_col_x != -1:
                        # Find text that aligns with symbol column
                        symbol_text = ""
                        description_text = ""
                        
                        for (bbox, t, c) in ocr_results:
                            if abs(bbox[0][0] - symbol_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                symbol_text = t.strip()
                            if abs(bbox[0][0] - desc_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                description_text = t.strip()
                        
                        if symbol_text and description_text:
                            lighting_schedule.append({
                                "symbol": symbol_text,
                                "description": description_text,
                                "source_sheet": f"Page {page_num + 1}"
                            })
                            
        doc.close()
        
        return {
            "notes": notes,
            "lighting_schedule": lighting_schedule,
            "special_fixtures": {} # OCR alone might not easily extract this
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
                "source": det.source_sheet
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
            # Mock LLM response for grouping
            mock_grouped_results = self._simple_grouping(detections, rulebook)
            return mock_grouped_results
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                grouped[symbol] = {
                    "count": 1,
                    "description": f"{detection.light_type.replace('_', ' ').title()} Emergency Light"
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Create annotated images showing detections for competition submission."""
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
                
            image = Image.open(image_path)
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
, clean_text) and len(clean_text) < 30: 
                        current_notes_section = False
                    else:
                        notes.append({"text": clean_text, "source": f"Page {page_num + 1}"})
                            
        doc.close()
        
        return {
            "notes": notes,
            "lighting_schedule": lighting_schedule,
            "special_fixtures": {} # OCR alone might not easily extract this
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
                "source": det.source_sheet
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
            # Mock LLM response for grouping
            mock_grouped_results = self._simple_grouping(detections, rulebook)
            return mock_grouped_results
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                grouped[symbol] = {
                    "count": 1,
                    "description": f"{detection.light_type.replace('_', ' ').title()} Emergency Light"
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Create annotated images showing detections for competition submission."""
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
                
            image = Image.open(image_path)
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
, clean_text) and len(clean_text) < 30: # Likely a new heading
                        current_notes_section = False
                    else:
                        notes.append({"text": clean_text, "source": f"Page {page_num + 1}"})

            # Heuristic for lighting schedule table: look for headers and then parse rows
            # This is a simplified approach; a more robust solution would involve table detection libraries
            header_found = False
            symbol_col_x = -1
            desc_col_x = -1
            
            # Collect all text on the page, sorted by y-coordinate
            page_texts = sorted([(b[0][1], b[0][0], t.strip(), c) for (b, t, c) in ocr_results if c > 0.5])

            for y, x, text, conf in page_texts:
                clean_text = text.upper()
                if not header_found:
                    if "SYMBOL" in clean_text and "DESCRIPTION" in clean_text:
                        header_found = True
                        # Attempt to find column x-coordinates
                        for (bbox, t, c) in ocr_results:
                            if "SYMBOL" in t.upper():
                                symbol_col_x = bbox[0][0]
                            if "DESCRIPTION" in t.upper():
                                desc_col_x = bbox[0][0]
                        continue
                
                if header_found:
                    # Try to parse rows based on column positions
                    # This is a very basic approach and assumes consistent column alignment
                    if symbol_col_x != -1 and desc_col_x != -1:
                        # Find text that aligns with symbol column
                        symbol_text = ""
                        description_text = ""
                        
                        for (bbox, t, c) in ocr_results:
                            if abs(bbox[0][0] - symbol_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                symbol_text = t.strip()
                            if abs(bbox[0][0] - desc_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                description_text = t.strip()
                        
                        if symbol_text and description_text:
                            lighting_schedule.append({
                                "symbol": symbol_text,
                                "description": description_text,
                                "source_sheet": f"Page {page_num + 1}"
                            })
                            
        doc.close()
        
        return {
            "notes": notes,
            "lighting_schedule": lighting_schedule,
            "special_fixtures": {} # OCR alone might not easily extract this
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
                "source": det.source_sheet
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
            # Mock LLM response for grouping
            mock_grouped_results = self._simple_grouping(detections, rulebook)
            return mock_grouped_results
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                grouped[symbol] = {
                    "count": 1,
                    "description": f"{detection.light_type.replace('_', ' ').title()} Emergency Light"
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Create annotated images showing detections for competition submission."""
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
                
            image = Image.open(image_path)
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
, clean_text) and len(clean_text) < 30: # Likely a new heading
                        current_notes_section = False
                    else:
                        notes.append({"text": clean_text, "source": f"Page {page_num + 1}"})

            # Heuristic for lighting schedule table: look for headers and then parse rows
            # This is a simplified approach; a more robust solution would involve table detection libraries
            header_found = False
            symbol_col_x = -1
            desc_col_x = -1
            
            # Collect all text on the page, sorted by y-coordinate
            page_texts = sorted([(b[0][1], b[0][0], t.strip(), c) for (b, t, c) in ocr_results if c > 0.5])

            for y, x, text, conf in page_texts:
                clean_text = text.upper()
                if not header_found:
                    if "SYMBOL" in clean_text and "DESCRIPTION" in clean_text:
                        header_found = True
                        # Attempt to find column x-coordinates
                        for (bbox, t, c) in ocr_results:
                            if "SYMBOL" in t.upper():
                                symbol_col_x = bbox[0][0]
                            if "DESCRIPTION" in t.upper():
                                desc_col_x = bbox[0][0]
                        continue
                
                if header_found:
                    # Try to parse rows based on column positions
                    # This is a very basic approach and assumes consistent column alignment
                    if symbol_col_x != -1 and desc_col_x != -1:
                        # Find text that aligns with symbol column
                        symbol_text = ""
                        description_text = ""
                        
                        for (bbox, t, c) in ocr_results:
                            if abs(bbox[0][0] - symbol_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                symbol_text = t.strip()
                            if abs(bbox[0][0] - desc_col_x) < 20 and abs(bbox[0][1] - y) < 10:
                                description_text = t.strip()
                        
                        if symbol_text and description_text:
                            lighting_schedule.append({
                                "symbol": symbol_text,
                                "description": description_text,
                                "source_sheet": f"Page {page_num + 1}"
                            })
                            
        doc.close()
        
        return {
            "notes": notes,
            "lighting_schedule": lighting_schedule,
            "special_fixtures": {} # OCR alone might not easily extract this
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
                "source": det.source_sheet
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
            # Mock LLM response for grouping
            mock_grouped_results = self._simple_grouping(detections, rulebook)
            return mock_grouped_results
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}")
        
        # Fallback to simple grouping
        return self._simple_grouping(detections, rulebook)

    def _simple_grouping(self, detections: List[EmergencyLight], rulebook: Dict) -> Dict[str, Any]:
        """Fallback simple grouping method."""
        grouped = {}
        
        for detection in detections:
            symbol = detection.symbol
            if symbol in grouped:
                grouped[symbol]["count"] += 1
            else:
                grouped[symbol] = {
                    "count": 1,
                    "description": f"{detection.light_type.replace('_', ' ').title()} Emergency Light"
                }
        
        return grouped

    def create_annotated_visualization(self, image_paths: List[str], detections: List[EmergencyLight], output_dir: str):
        """Create annotated images showing detections for competition submission."""
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
                
            image = Image.open(image_path)
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
