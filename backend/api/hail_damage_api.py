from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from typing import Optional
import logging
from pathlib import Path

from agent.tools.hail_damage_tool import HailDamageEstimationTool
from agentpress.thread_manager import ThreadManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api", tags=["hail-damage"])

# Initialize thread manager (you may need to adjust this based on your setup)
thread_manager = ThreadManager()

@router.post("/analyze-hail-damage")
async def analyze_hail_damage(
    image: UploadFile = File(...),
    vehicle_year: Optional[int] = Form(None),
    vehicle_make: Optional[str] = Form(None),
    vehicle_model: Optional[str] = Form(None),
    vehicle_value: Optional[float] = Form(None),
    analysis_mode: str = Form("detailed")
):
    """
    Analyze uploaded vehicle image for hail damage.
    
    Args:
        image: Uploaded image file
        vehicle_year: Year of the vehicle (optional)
        vehicle_make: Make of the vehicle (optional)
        vehicle_model: Model of the vehicle (optional)
        vehicle_value: Estimated value of the vehicle (optional)
        analysis_mode: Analysis mode (quick, detailed, insurance)
    
    Returns:
        JSON response with damage assessment results
    """
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate analysis mode
    valid_modes = ["quick", "detailed", "insurance"]
    if analysis_mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Analysis mode must be one of: {valid_modes}")
    
    temp_file_path = None
    
    try:
        # Create temporary file to store uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(image.file, temp_file)
        
        logger.info(f"Processing image: {image.filename}, Mode: {analysis_mode}")
        
        # Initialize hail damage tool
        project_id = "hail_damage_analysis"
        thread_id = "analysis_thread"
        hail_tool = HailDamageEstimationTool(project_id, thread_id, thread_manager)
        
        # Perform analysis
        result = await hail_tool.analyze_hail_damage(
            image_path=temp_file_path,
            vehicle_year=vehicle_year,
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            vehicle_value=vehicle_value,
            analysis_mode=analysis_mode
        )
        
        if not result.success:
            logger.error(f"Analysis failed: {result.error}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {result.error}")
        
        logger.info(f"Analysis completed successfully. Total dents: {result.result['assessment']['total_dents']}")
        
        return JSONResponse(content=result.result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

@router.post("/estimate-repair-time")
async def estimate_repair_time(
    total_dents: int = Form(...),
    repair_method: str = Form(...),
    minor_dents: int = Form(0),
    moderate_dents: int = Form(0),
    severe_dents: int = Form(0)
):
    """
    Estimate repair time based on damage assessment.
    
    Args:
        total_dents: Total number of dents
        repair_method: Recommended repair method
        minor_dents: Number of minor dents
        moderate_dents: Number of moderate dents
        severe_dents: Number of severe dents
    
    Returns:
        JSON response with time estimates
    """
    
    try:
        # Initialize hail damage tool
        project_id = "hail_damage_analysis"
        thread_id = "time_estimation_thread"
        hail_tool = HailDamageEstimationTool(project_id, thread_id, thread_manager)
        
        # Prepare severity breakdown
        severity_breakdown = {
            "minor": minor_dents,
            "moderate": moderate_dents,
            "severe": severe_dents
        }
        
        # Estimate repair time
        result = await hail_tool.estimate_repair_time(
            total_dents=total_dents,
            repair_method=repair_method,
            severity_breakdown=severity_breakdown
        )
        
        if not result.success:
            logger.error(f"Time estimation failed: {result.error}")
            raise HTTPException(status_code=500, detail=f"Time estimation failed: {result.error}")
        
        return JSONResponse(content=result.result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during time estimation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/generate-insurance-report")
async def generate_insurance_report(
    assessment_data: dict,
    policy_holder: Optional[str] = Form(None),
    claim_number: Optional[str] = Form(None),
    adjuster_name: Optional[str] = Form(None)
):
    """
    Generate comprehensive insurance report for hail damage claims.
    
    Args:
        assessment_data: Complete damage assessment data
        policy_holder: Name of the policy holder (optional)
        claim_number: Insurance claim number (optional)
        adjuster_name: Name of the insurance adjuster (optional)
    
    Returns:
        JSON response with insurance report
    """
    
    try:
        # Initialize hail damage tool
        project_id = "hail_damage_analysis"
        thread_id = "insurance_report_thread"
        hail_tool = HailDamageEstimationTool(project_id, thread_id, thread_manager)
        
        # Generate insurance report
        result = await hail_tool.generate_insurance_report(
            assessment_data=assessment_data,
            policy_holder=policy_holder,
            claim_number=claim_number,
            adjuster_name=adjuster_name
        )
        
        if not result.success:
            logger.error(f"Insurance report generation failed: {result.error}")
            raise HTTPException(status_code=500, detail=f"Report generation failed: {result.error}")
        
        return JSONResponse(content=result.result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during report generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the hail damage API.
    
    Returns:
        JSON response indicating service status
    """
    return JSONResponse(content={
        "status": "healthy",
        "service": "Hail Damage Analysis API",
        "version": "1.0.0"
    })

@router.get("/pricing-info")
async def get_pricing_info():
    """
    Get current PDR pricing information.
    
    Returns:
        JSON response with pricing structure
    """
    
    pricing_info = {
        "base_pricing": {
            "minor": {"min": 30, "max": 45, "description": "Small dents (dime size)"},
            "moderate": {"min": 45, "max": 55, "description": "Medium dents (nickel/quarter size)"},
            "severe": {"min": 75, "max": 150, "description": "Large dents (half dollar+)"}
        },
        "panel_factors": {
            "hood": {"factor": 1.0, "description": "Easy access"},
            "roof": {"factor": 1.2, "description": "Moderate difficulty"},
            "door": {"factor": 1.1, "description": "Standard access"},
            "fender": {"factor": 1.0, "description": "Easy access"},
            "trunk": {"factor": 1.0, "description": "Easy access"},
            "quarter_panel": {"factor": 1.3, "description": "Difficult access"},
            "pillar": {"factor": 1.5, "description": "Very difficult access"}
        },
        "vehicle_adjustments": {
            "luxury_multiplier": 1.5,
            "luxury_brands": ["BMW", "Mercedes", "Audi", "Lexus", "Acura", "Infiniti", "Cadillac", "Lincoln"],
            "age_reduction": "2% per year (minimum 80% of base price)"
        },
        "notes": [
            "Prices are estimates based on industry standards",
            "Final costs may vary based on shop rates and accessibility",
            "Complex damage may require traditional bodywork",
            "Total loss threshold is typically 75% of vehicle value"
        ]
    }
    
    return JSONResponse(content=pricing_info)

@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get information about supported image formats and requirements.
    
    Returns:
        JSON response with format specifications
    """
    
    format_info = {
        "supported_formats": ["JPEG", "JPG", "PNG", "WebP", "BMP", "TIFF"],
        "max_file_size": "10MB",
        "recommended_resolution": "1920x1080 or higher",
        "image_requirements": [
            "Clear, well-lit images",
            "Vehicle should fill most of the frame",
            "Avoid extreme angles or shadows",
            "Multiple angles recommended for comprehensive analysis"
        ],
        "analysis_modes": {
            "quick": "Basic dent detection using circular Hough transform",
            "detailed": "Comprehensive analysis with multiple detection methods",
            "insurance": "Full analysis with detailed reporting for insurance claims"
        }
    }
    
    return JSONResponse(content=format_info)