import os
import base64
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import math

from agentpress.tool import ToolResult, openapi_schema
from sandbox.tool_base import SandboxToolsBase
from agentpress.thread_manager import ThreadManager

@dataclass
class DentDetection:
    """Represents a detected dent with its properties."""
    x: int
    y: int
    width: int
    height: int
    severity: str  # 'minor', 'moderate', 'severe'
    confidence: float
    estimated_cost: float
    panel: str  # 'hood', 'roof', 'door', 'fender', etc.

@dataclass
class HailDamageAssessment:
    """Complete hail damage assessment result."""
    total_dents: int
    dents_by_severity: Dict[str, int]
    total_estimated_cost: float
    repair_method: str  # 'PDR', 'Traditional', 'Panel Replacement'
    is_total_loss: bool
    confidence_score: float
    detected_dents: List[DentDetection]
    processed_image_base64: str

class HailDamageEstimationTool(SandboxToolsBase):
    """Advanced computer vision tool for estimating hail damage in vehicles for PDR industry."""
    
    def __init__(self, project_id: str, thread_id: str, thread_manager: ThreadManager):
        super().__init__(project_id, thread_manager)
        self.thread_id = thread_id
        self.thread_manager = thread_manager
        
        # Pricing structure based on industry standards
        self.pricing = {
            'minor': {'min': 30, 'max': 45},  # Small dents (dime size)
            'moderate': {'min': 45, 'max': 55},  # Medium dents (nickel/quarter size)
            'severe': {'min': 75, 'max': 150}  # Large dents (half dollar+)
        }
        
        # Panel accessibility factors for PDR
        self.panel_factors = {
            'hood': 1.0,
            'roof': 1.2,
            'door': 1.1,
            'fender': 1.0,
            'trunk': 1.0,
            'quarter_panel': 1.3,
            'pillar': 1.5
        }

    @openapi_schema({
        "type": "function",
        "function": {
            "name": "analyze_hail_damage",
            "description": "Analyze vehicle images for hail damage using computer vision. Detects dents, estimates repair costs, and provides comprehensive damage assessment for paintless dent repair industry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the vehicle image file to analyze for hail damage"
                    },
                    "vehicle_year": {
                        "type": "integer",
                        "description": "Year of the vehicle (affects repair complexity and costs)"
                    },
                    "vehicle_make": {
                        "type": "string",
                        "description": "Make of the vehicle (e.g., Toyota, BMW, Mercedes)"
                    },
                    "vehicle_model": {
                        "type": "string",
                        "description": "Model of the vehicle"
                    },
                    "vehicle_value": {
                        "type": "number",
                        "description": "Estimated value of the vehicle (for total loss calculation)"
                    },
                    "analysis_mode": {
                        "type": "string",
                        "enum": ["quick", "detailed", "insurance"],
                        "description": "Analysis mode: quick (basic detection), detailed (comprehensive analysis), insurance (full report)",
                        "default": "detailed"
                    }
                },
                "required": ["image_path"]
            }
        }
    })
    async def analyze_hail_damage(
        self,
        image_path: str,
        vehicle_year: Optional[int] = None,
        vehicle_make: Optional[str] = None,
        vehicle_model: Optional[str] = None,
        vehicle_value: Optional[float] = None,
        analysis_mode: str = "detailed"
    ) -> ToolResult:
        """Analyze vehicle image for hail damage and provide cost estimation."""
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                return ToolResult(
                    success=False,
                    error=f"Image file not found: {image_path}"
                )
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return ToolResult(
                    success=False,
                    error="Failed to load image. Please ensure it's a valid image file."
                )
            
            # Perform hail damage analysis
            assessment = await self._analyze_image(image, analysis_mode)
            
            # Apply vehicle-specific adjustments
            if vehicle_year and vehicle_make:
                assessment = self._apply_vehicle_adjustments(assessment, vehicle_year, vehicle_make, vehicle_model)
            
            # Check for total loss
            if vehicle_value:
                assessment.is_total_loss = assessment.total_estimated_cost > (vehicle_value * 0.75)
            
            # Generate detailed report
            report = self._generate_report(assessment, vehicle_year, vehicle_make, vehicle_model, vehicle_value)
            
            return ToolResult(
                success=True,
                result={
                    "assessment": {
                        "total_dents": assessment.total_dents,
                        "dents_by_severity": assessment.dents_by_severity,
                        "total_estimated_cost": assessment.total_estimated_cost,
                        "repair_method": assessment.repair_method,
                        "is_total_loss": assessment.is_total_loss,
                        "confidence_score": assessment.confidence_score
                    },
                    "detailed_report": report,
                    "processed_image": assessment.processed_image_base64,
                    "detected_dents": [
                        {
                            "location": f"({dent.x}, {dent.y})",
                            "size": f"{dent.width}x{dent.height}",
                            "severity": dent.severity,
                            "panel": dent.panel,
                            "estimated_cost": dent.estimated_cost,
                            "confidence": dent.confidence
                        }
                        for dent in assessment.detected_dents
                    ]
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error analyzing hail damage: {str(e)}"
            )

    async def _analyze_image(self, image: np.ndarray, mode: str) -> HailDamageAssessment:
        """Perform computer vision analysis on the vehicle image."""
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect dents using multiple techniques
        dents = []
        
        # Method 1: Circular Hough Transform for round dents
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Analyze the circular region for dent characteristics
                severity, confidence = self._analyze_dent_region(gray, x, y, r)
                if confidence > 0.3:  # Confidence threshold
                    panel = self._identify_panel(x, y, image.shape)
                    cost = self._calculate_dent_cost(severity, panel, r)
                    
                    dents.append(DentDetection(
                        x=x, y=y, width=r*2, height=r*2,
                        severity=severity, confidence=confidence,
                        estimated_cost=cost, panel=panel
                    ))
        
        # Method 2: Contour detection for irregular dents
        if mode in ["detailed", "insurance"]:
            contour_dents = self._detect_contour_dents(gray, image)
            dents.extend(contour_dents)
        
        # Method 3: Edge detection and morphological operations
        if mode == "insurance":
            edge_dents = self._detect_edge_dents(gray, image)
            dents.extend(edge_dents)
        
        # Remove duplicate detections
        dents = self._remove_duplicates(dents)
        
        # Calculate overall assessment
        total_dents = len(dents)
        dents_by_severity = {
            'minor': len([d for d in dents if d.severity == 'minor']),
            'moderate': len([d for d in dents if d.severity == 'moderate']),
            'severe': len([d for d in dents if d.severity == 'severe'])
        }
        
        total_cost = sum(dent.estimated_cost for dent in dents)
        
        # Determine repair method
        severe_ratio = dents_by_severity['severe'] / max(total_dents, 1)
        if severe_ratio > 0.3 or total_dents > 100:
            repair_method = "Panel Replacement"
        elif severe_ratio > 0.1 or any(d.estimated_cost > 100 for d in dents):
            repair_method = "Traditional Repair"
        else:
            repair_method = "PDR (Paintless Dent Repair)"
        
        # Calculate confidence score
        avg_confidence = np.mean([d.confidence for d in dents]) if dents else 0.0
        
        # Generate processed image with annotations
        processed_image = self._annotate_image(image.copy(), dents)
        processed_image_base64 = self._image_to_base64(processed_image)
        
        return HailDamageAssessment(
            total_dents=total_dents,
            dents_by_severity=dents_by_severity,
            total_estimated_cost=total_cost,
            repair_method=repair_method,
            is_total_loss=False,  # Will be calculated later if vehicle value provided
            confidence_score=avg_confidence,
            detected_dents=dents,
            processed_image_base64=processed_image_base64
        )

    def _analyze_dent_region(self, gray: np.ndarray, x: int, y: int, radius: int) -> Tuple[str, float]:
        """Analyze a circular region to determine if it's a dent and its severity."""
        
        # Extract region of interest
        roi = gray[max(0, y-radius):min(gray.shape[0], y+radius),
                  max(0, x-radius):min(gray.shape[1], x+radius)]
        
        if roi.size == 0:
            return "minor", 0.0
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate statistics
        mean_gradient = np.mean(gradient_magnitude)
        std_gradient = np.std(gradient_magnitude)
        
        # Calculate intensity variation
        intensity_std = np.std(roi)
        
        # Determine severity based on gradient and intensity patterns
        confidence = min(1.0, (mean_gradient + intensity_std) / 100.0)
        
        if mean_gradient > 30 and intensity_std > 20:
            severity = "severe"
        elif mean_gradient > 15 and intensity_std > 10:
            severity = "moderate"
        else:
            severity = "minor"
        
        return severity, confidence

    def _detect_contour_dents(self, gray: np.ndarray, image: np.ndarray) -> List[DentDetection]:
        """Detect dents using contour analysis."""
        
        dents = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Filter by area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h
                
                # Filter by aspect ratio (dents are roughly circular)
                if 0.5 < aspect_ratio < 2.0:
                    # Analyze the contour region
                    roi = gray[y:y+h, x:x+w]
                    severity, confidence = self._analyze_contour_region(roi, contour, area)
                    
                    if confidence > 0.4:
                        panel = self._identify_panel(x + w//2, y + h//2, image.shape)
                        cost = self._calculate_dent_cost(severity, panel, max(w, h)//2)
                        
                        dents.append(DentDetection(
                            x=x + w//2, y=y + h//2, width=w, height=h,
                            severity=severity, confidence=confidence,
                            estimated_cost=cost, panel=panel
                        ))
        
        return dents

    def _detect_edge_dents(self, gray: np.ndarray, image: np.ndarray) -> List[DentDetection]:
        """Detect dents using advanced edge detection techniques."""
        
        dents = []
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Find local minima (potential dent centers)
        local_minima = cv2.morphologyEx(opened, cv2.MORPH_ERODE, kernel)
        
        # Find difference
        diff = cv2.absdiff(opened, local_minima)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional validation
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    intensity_var = np.var(roi)
                    if intensity_var > 50:  # Sufficient variation indicates a dent
                        severity = "minor" if area < 200 else "moderate" if area < 800 else "severe"
                        confidence = min(0.8, intensity_var / 200.0)
                        
                        panel = self._identify_panel(x + w//2, y + h//2, image.shape)
                        cost = self._calculate_dent_cost(severity, panel, max(w, h)//2)
                        
                        dents.append(DentDetection(
                            x=x + w//2, y=y + h//2, width=w, height=h,
                            severity=severity, confidence=confidence,
                            estimated_cost=cost, panel=panel
                        ))
        
        return dents

    def _analyze_contour_region(self, roi: np.ndarray, contour: np.ndarray, area: float) -> Tuple[str, float]:
        """Analyze a contour region for dent characteristics."""
        
        if roi.size == 0:
            return "minor", 0.0
        
        # Calculate contour properties
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate intensity statistics
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        
        # Determine severity
        if area > 1000 and std_intensity > 25:
            severity = "severe"
        elif area > 300 and std_intensity > 15:
            severity = "moderate"
        else:
            severity = "minor"
        
        # Calculate confidence based on circularity and intensity variation
        confidence = min(1.0, (circularity + std_intensity / 50.0) / 2.0)
        
        return severity, confidence

    def _identify_panel(self, x: int, y: int, image_shape: Tuple[int, int, int]) -> str:
        """Identify which vehicle panel the dent is located on based on position."""
        
        height, width = image_shape[:2]
        
        # Normalize coordinates
        norm_x = x / width
        norm_y = y / height
        
        # Simple heuristic for panel identification
        if norm_y < 0.3:  # Top third
            return "roof"
        elif norm_y > 0.7:  # Bottom third
            if norm_x < 0.5:
                return "door"
            else:
                return "quarter_panel"
        else:  # Middle third
            if norm_x < 0.2:
                return "fender"
            elif norm_x > 0.8:
                return "fender"
            elif norm_x < 0.6:
                return "hood"
            else:
                return "door"

    def _calculate_dent_cost(self, severity: str, panel: str, radius: int) -> float:
        """Calculate repair cost for a single dent."""
        
        base_cost = self.pricing[severity]['min']
        
        # Size adjustment
        if radius > 30:  # Large dent
            base_cost = self.pricing[severity]['max']
        elif radius > 20:  # Medium dent
            base_cost = (self.pricing[severity]['min'] + self.pricing[severity]['max']) / 2
        
        # Panel accessibility factor
        panel_factor = self.panel_factors.get(panel, 1.0)
        
        return base_cost * panel_factor

    def _remove_duplicates(self, dents: List[DentDetection], threshold: int = 30) -> List[DentDetection]:
        """Remove duplicate dent detections based on proximity."""
        
        if not dents:
            return dents
        
        # Sort by confidence (highest first)
        dents.sort(key=lambda d: d.confidence, reverse=True)
        
        filtered_dents = []
        
        for dent in dents:
            is_duplicate = False
            for existing_dent in filtered_dents:
                distance = math.sqrt((dent.x - existing_dent.x)**2 + (dent.y - existing_dent.y)**2)
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_dents.append(dent)
        
        return filtered_dents

    def _apply_vehicle_adjustments(self, assessment: HailDamageAssessment, year: int, make: str, model: Optional[str]) -> HailDamageAssessment:
        """Apply vehicle-specific cost adjustments."""
        
        # Luxury vehicle multiplier
        luxury_brands = ['BMW', 'Mercedes', 'Audi', 'Lexus', 'Acura', 'Infiniti', 'Cadillac', 'Lincoln']
        luxury_multiplier = 1.5 if make.upper() in [b.upper() for b in luxury_brands] else 1.0
        
        # Age adjustment
        current_year = 2024
        age = current_year - year
        age_multiplier = max(0.8, 1.0 - (age * 0.02))  # Slight reduction for older vehicles
        
        # Apply adjustments
        total_multiplier = luxury_multiplier * age_multiplier
        assessment.total_estimated_cost *= total_multiplier
        
        for dent in assessment.detected_dents:
            dent.estimated_cost *= total_multiplier
        
        return assessment

    def _annotate_image(self, image: np.ndarray, dents: List[DentDetection]) -> np.ndarray:
        """Annotate the image with detected dents and information."""
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Color mapping for severity
        colors = {
            'minor': (0, 255, 0),    # Green
            'moderate': (255, 165, 0), # Orange
            'severe': (255, 0, 0)     # Red
        }
        
        for i, dent in enumerate(dents):
            color = colors[dent.severity]
            
            # Draw circle around dent
            radius = max(dent.width, dent.height) // 2
            draw.ellipse(
                [dent.x - radius, dent.y - radius, dent.x + radius, dent.y + radius],
                outline=color,
                width=3
            )
            
            # Add label
            label = f"{i+1}: ${dent.estimated_cost:.0f}"
            draw.text((dent.x + radius + 5, dent.y - 10), label, fill=color)
        
        # Add summary information
        summary_text = f"Total Dents: {len(dents)} | Est. Cost: ${sum(d.estimated_cost for d in dents):.0f}"
        draw.text((10, 10), summary_text, fill=(255, 255, 255))
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string."""
        
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"

    def _generate_report(self, assessment: HailDamageAssessment, year: Optional[int], make: Optional[str], model: Optional[str], value: Optional[float]) -> str:
        """Generate a detailed damage assessment report."""
        
        report = "\n=== HAIL DAMAGE ASSESSMENT REPORT ===\n\n"
        
        if year and make:
            report += f"Vehicle: {year} {make} {model or ''}\n"
        if value:
            report += f"Estimated Value: ${value:,.2f}\n"
        
        report += f"\nDAMAGE SUMMARY:\n"
        report += f"- Total Dents Detected: {assessment.total_dents}\n"
        report += f"- Minor Dents: {assessment.dents_by_severity['minor']}\n"
        report += f"- Moderate Dents: {assessment.dents_by_severity['moderate']}\n"
        report += f"- Severe Dents: {assessment.dents_by_severity['severe']}\n"
        
        report += f"\nCOST ANALYSIS:\n"
        report += f"- Total Estimated Repair Cost: ${assessment.total_estimated_cost:,.2f}\n"
        report += f"- Recommended Repair Method: {assessment.repair_method}\n"
        
        if assessment.is_total_loss:
            report += f"- TOTAL LOSS: Repair cost exceeds 75% of vehicle value\n"
        
        report += f"\nCONFIDENCE METRICS:\n"
        report += f"- Analysis Confidence: {assessment.confidence_score:.1%}\n"
        
        # Cost breakdown by severity
        minor_cost = sum(d.estimated_cost for d in assessment.detected_dents if d.severity == 'minor')
        moderate_cost = sum(d.estimated_cost for d in assessment.detected_dents if d.severity == 'moderate')
        severe_cost = sum(d.estimated_cost for d in assessment.detected_dents if d.severity == 'severe')
        
        report += f"\nCOST BREAKDOWN:\n"
        report += f"- Minor Dents: ${minor_cost:,.2f}\n"
        report += f"- Moderate Dents: ${moderate_cost:,.2f}\n"
        report += f"- Severe Dents: ${severe_cost:,.2f}\n"
        
        # Panel distribution
        panel_counts = {}
        for dent in assessment.detected_dents:
            panel_counts[dent.panel] = panel_counts.get(dent.panel, 0) + 1
        
        if panel_counts:
            report += f"\nDAMAGE BY PANEL:\n"
            for panel, count in sorted(panel_counts.items()):
                report += f"- {panel.replace('_', ' ').title()}: {count} dents\n"
        
        # Recommendations
        report += f"\nRECOMMENDations:\n"
        if assessment.repair_method == "PDR (Paintless Dent Repair)":
            report += "- Paintless Dent Repair is recommended for cost-effective restoration\n"
            report += "- Original paint will be preserved\n"
            report += "- Typical repair time: 1-3 days\n"
        elif assessment.repair_method == "Traditional Repair":
            report += "- Traditional bodywork required due to paint damage or dent severity\n"
            report += "- Some panels may need repainting\n"
            report += "- Typical repair time: 3-7 days\n"
        else:
            report += "- Panel replacement recommended due to extensive damage\n"
            report += "- Complete refinishing may be required\n"
            report += "- Typical repair time: 1-2 weeks\n"
        
        report += "\n=== END REPORT ===\n"
        
        return report

    @openapi_schema({
        "type": "function",
        "function": {
            "name": "estimate_repair_time",
            "description": "Estimate repair time based on damage assessment and repair method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_dents": {
                        "type": "integer",
                        "description": "Total number of dents detected"
                    },
                    "repair_method": {
                        "type": "string",
                        "enum": ["PDR", "Traditional", "Panel Replacement"],
                        "description": "Recommended repair method"
                    },
                    "severity_breakdown": {
                        "type": "object",
                        "properties": {
                            "minor": {"type": "integer"},
                            "moderate": {"type": "integer"},
                            "severe": {"type": "integer"}
                        },
                        "description": "Breakdown of dents by severity"
                    }
                },
                "required": ["total_dents", "repair_method"]
            }
        }
    })
    async def estimate_repair_time(
        self,
        total_dents: int,
        repair_method: str,
        severity_breakdown: Optional[Dict[str, int]] = None
    ) -> ToolResult:
        """Estimate repair time based on damage assessment."""
        
        try:
            # Base time estimates (in hours)
            base_times = {
                "PDR": {
                    "minor": 0.5,
                    "moderate": 1.0,
                    "severe": 2.0
                },
                "Traditional": {
                    "minor": 2.0,
                    "moderate": 4.0,
                    "severe": 8.0
                },
                "Panel Replacement": {
                    "minor": 4.0,
                    "moderate": 6.0,
                    "severe": 12.0
                }
            }
            
            if severity_breakdown:
                total_hours = 0
                method_key = repair_method.split(" ")[0]  # Extract PDR, Traditional, etc.
                
                for severity, count in severity_breakdown.items():
                    if severity in base_times.get(method_key, {}):
                        total_hours += base_times[method_key][severity] * count
            else:
                # Fallback calculation
                avg_time_per_dent = {
                    "PDR": 1.0,
                    "Traditional": 4.0,
                    "Panel Replacement": 8.0
                }
                method_key = repair_method.split(" ")[0]
                total_hours = total_dents * avg_time_per_dent.get(method_key, 2.0)
            
            # Convert to business days (8 hours per day)
            business_days = math.ceil(total_hours / 8)
            
            # Add buffer time
            if repair_method.startswith("PDR"):
                buffer_days = max(1, business_days * 0.2)
            else:
                buffer_days = max(2, business_days * 0.3)
            
            total_days = business_days + buffer_days
            
            return ToolResult(
                success=True,
                result={
                    "estimated_hours": round(total_hours, 1),
                    "business_days": business_days,
                    "total_days_with_buffer": round(total_days),
                    "repair_method": repair_method,
                    "breakdown": {
                        "labor_hours": round(total_hours, 1),
                        "buffer_time": round(buffer_days, 1),
                        "total_calendar_days": round(total_days)
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error estimating repair time: {str(e)}"
            )

    @openapi_schema({
        "type": "function",
        "function": {
            "name": "generate_insurance_report",
            "description": "Generate a comprehensive insurance report for hail damage claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assessment_data": {
                        "type": "object",
                        "description": "Complete damage assessment data from analyze_hail_damage"
                    },
                    "policy_holder": {
                        "type": "string",
                        "description": "Name of the policy holder"
                    },
                    "claim_number": {
                        "type": "string",
                        "description": "Insurance claim number"
                    },
                    "adjuster_name": {
                        "type": "string",
                        "description": "Name of the insurance adjuster"
                    }
                },
                "required": ["assessment_data"]
            }
        }
    })
    async def generate_insurance_report(
        self,
        assessment_data: Dict[str, Any],
        policy_holder: Optional[str] = None,
        claim_number: Optional[str] = None,
        adjuster_name: Optional[str] = None
    ) -> ToolResult:
        """Generate a comprehensive insurance report for hail damage claims."""
        
        try:
            from datetime import datetime
            
            report = "\n" + "="*60 + "\n"
            report += "INSURANCE HAIL DAMAGE ASSESSMENT REPORT\n"
            report += "="*60 + "\n\n"
            
            # Header information
            report += f"Report Date: {datetime.now().strftime('%B %d, %Y')}\n"
            if claim_number:
                report += f"Claim Number: {claim_number}\n"
            if policy_holder:
                report += f"Policy Holder: {policy_holder}\n"
            if adjuster_name:
                report += f"Adjuster: {adjuster_name}\n"
            
            report += "\nASSESSMENT METHOD: Computer Vision Analysis\n"
            report += "ANALYSIS SOFTWARE: Suna AI Hail Damage Estimation Tool\n\n"
            
            # Extract assessment data
            assessment = assessment_data.get('assessment', {})
            
            report += "DAMAGE SUMMARY:\n"
            report += "-" * 20 + "\n"
            report += f"Total Dents Detected: {assessment.get('total_dents', 0)}\n"
            
            severity_breakdown = assessment.get('dents_by_severity', {})
            report += f"Minor Dents (≤1 inch): {severity_breakdown.get('minor', 0)}\n"
            report += f"Moderate Dents (1-2 inches): {severity_breakdown.get('moderate', 0)}\n"
            report += f"Severe Dents (>2 inches): {severity_breakdown.get('severe', 0)}\n\n"
            
            report += "COST ASSESSMENT:\n"
            report += "-" * 20 + "\n"
            total_cost = assessment.get('total_estimated_cost', 0)
            report += f"Total Estimated Repair Cost: ${total_cost:,.2f}\n"
            report += f"Recommended Repair Method: {assessment.get('repair_method', 'N/A')}\n"
            
            if assessment.get('is_total_loss', False):
                report += "\n*** TOTAL LOSS RECOMMENDATION ***\n"
                report += "Repair costs exceed 75% of vehicle value\n"
            
            report += f"\nCONFIDENCE LEVEL: {assessment.get('confidence_score', 0):.1%}\n\n"
            
            # Detailed dent analysis
            detected_dents = assessment_data.get('detected_dents', [])
            if detected_dents:
                report += "DETAILED DENT ANALYSIS:\n"
                report += "-" * 30 + "\n"
                report += f"{'#':<3} {'Location':<12} {'Size':<10} {'Severity':<10} {'Panel':<12} {'Cost':<8}\n"
                report += "-" * 60 + "\n"
                
                for i, dent in enumerate(detected_dents[:20], 1):  # Limit to first 20 for readability
                    report += f"{i:<3} {dent.get('location', ''):<12} {dent.get('size', ''):<10} "
                    report += f"{dent.get('severity', ''):<10} {dent.get('panel', ''):<12} "
                    report += f"${dent.get('estimated_cost', 0):<7.0f}\n"
                
                if len(detected_dents) > 20:
                    report += f"... and {len(detected_dents) - 20} more dents\n"
            
            report += "\nREPAIR RECOMMENDATIONS:\n"
            report += "-" * 25 + "\n"
            
            repair_method = assessment.get('repair_method', '')
            if 'PDR' in repair_method:
                report += "• Paintless Dent Repair (PDR) is recommended\n"
                report += "• Original factory paint will be preserved\n"
                report += "• Environmentally friendly repair method\n"
                report += "• Maintains vehicle resale value\n"
                report += "• Estimated completion: 1-3 business days\n"
            elif 'Traditional' in repair_method:
                report += "• Traditional bodywork and refinishing required\n"
                report += "• Some panels may require repainting\n"
                report += "• Body filler and primer application needed\n"
                report += "• Estimated completion: 3-7 business days\n"
            else:
                report += "• Panel replacement recommended\n"
                report += "• Extensive damage requires new panels\n"
                report += "• Complete refinishing necessary\n"
                report += "• Estimated completion: 1-2 weeks\n"
            
            report += "\nNOTES:\n"
            report += "-" * 10 + "\n"
            report += "• Assessment performed using advanced computer vision technology\n"
            report += "• Costs based on current industry PDR pricing standards\n"
            report += "• Final repair costs may vary based on shop rates and accessibility\n"
            report += "• Recommend physical inspection to confirm computer analysis\n\n"
            
            report += "ADJUSTER SIGNATURE: _________________________ DATE: _________\n\n"
            report += "="*60 + "\n"
            
            return ToolResult(
                success=True,
                result={
                    "insurance_report": report,
                    "summary": {
                        "total_dents": assessment.get('total_dents', 0),
                        "estimated_cost": total_cost,
                        "repair_method": repair_method,
                        "is_total_loss": assessment.get('is_total_loss', False),
                        "confidence": assessment.get('confidence_score', 0)
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error generating insurance report: {str(e)}"
            )