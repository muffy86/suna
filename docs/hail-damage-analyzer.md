# AI Hail Damage Analyzer

A comprehensive computer vision system for estimating hail damage in vehicles, specifically designed for the Paintless Dent Repair (PDR) industry.

## Overview

The AI Hail Damage Analyzer uses advanced computer vision techniques to automatically detect, classify, and estimate repair costs for hail damage on vehicles. This system provides accurate assessments that help PDR professionals, insurance adjusters, and vehicle owners make informed decisions about repair strategies.

## Features

### 🔍 Advanced Detection
- **Multi-Method Analysis**: Combines circular Hough transforms, contour detection, and edge analysis
- **Severity Classification**: Automatically categorizes dents as minor, moderate, or severe
- **Panel Identification**: Recognizes different vehicle panels and applies appropriate cost factors
- **Confidence Scoring**: Provides reliability metrics for each detection

### 💰 Cost Estimation
- **Industry-Standard Pricing**: Based on current PDR market rates
- **Vehicle-Specific Adjustments**: Accounts for luxury brands and vehicle age
- **Panel Accessibility Factors**: Considers repair difficulty by panel location
- **Total Loss Assessment**: Automatically flags vehicles exceeding repair thresholds

### 📊 Comprehensive Reporting
- **Visual Annotations**: Processed images with highlighted damage areas
- **Detailed Breakdowns**: Cost analysis by severity and panel
- **Insurance Reports**: Professional documentation for claims processing
- **Time Estimates**: Repair duration calculations

### 🎯 Analysis Modes
- **Quick Scan**: Basic detection for rapid assessments
- **Detailed Analysis**: Comprehensive evaluation with multiple detection methods
- **Insurance Mode**: Full analysis with detailed reporting for claims

## Technical Architecture

### Backend Components

#### Core Tool: `HailDamageEstimationTool`
Location: `backend/agent/tools/hail_damage_tool.py`

**Key Methods:**
- `analyze_hail_damage()`: Main analysis function
- `estimate_repair_time()`: Time estimation calculations
- `generate_insurance_report()`: Professional report generation

**Detection Algorithms:**
1. **Circular Hough Transform**: Detects round dents typical of hail damage
2. **Contour Analysis**: Identifies irregular damage patterns
3. **Edge Detection**: Advanced morphological operations for subtle damage

#### API Endpoints
Location: `backend/api/hail_damage_api.py`

- `POST /api/analyze-hail-damage`: Main analysis endpoint
- `POST /api/estimate-repair-time`: Time estimation service
- `POST /api/generate-insurance-report`: Report generation
- `GET /api/pricing-info`: Current pricing structure
- `GET /api/supported-formats`: Image format specifications

### Frontend Components

#### Main Interface: `HailDamageAnalyzer`
Location: `frontend/src/components/HailDamageAnalyzer.tsx`

**Features:**
- Drag-and-drop image upload
- Vehicle information input
- Real-time analysis progress
- Interactive results display
- Report export functionality

## Usage Guide

### 1. Image Upload
```typescript
// Supported formats
const supportedFormats = ['JPEG', 'JPG', 'PNG', 'WebP', 'BMP', 'TIFF'];
const maxFileSize = '10MB';
const recommendedResolution = '1920x1080 or higher';
```

### 2. Vehicle Information (Optional)
- **Year**: Affects age-based pricing adjustments
- **Make**: Luxury brand detection for premium pricing
- **Model**: Additional context for specialized vehicles
- **Value**: Total loss threshold calculation (75% rule)

### 3. Analysis Process
```python
# Example API call
result = await hail_tool.analyze_hail_damage(
    image_path="/path/to/vehicle/image.jpg",
    vehicle_year=2022,
    vehicle_make="BMW",
    vehicle_model="X5",
    vehicle_value=65000,
    analysis_mode="detailed"
)
```

### 4. Results Interpretation

#### Damage Assessment
```json
{
  "total_dents": 45,
  "dents_by_severity": {
    "minor": 30,
    "moderate": 12,
    "severe": 3
  },
  "total_estimated_cost": 2850.00,
  "repair_method": "PDR (Paintless Dent Repair)",
  "is_total_loss": false,
  "confidence_score": 0.87
}
```

#### Individual Dent Details
```json
{
  "location": "(450, 320)",
  "size": "24x22",
  "severity": "moderate",
  "panel": "hood",
  "estimated_cost": 50.00,
  "confidence": 0.92
}
```

## Pricing Structure

### Base Pricing (Per Dent)
- **Minor Dents** (≤1 inch): $30 - $45
- **Moderate Dents** (1-2 inches): $45 - $55
- **Severe Dents** (>2 inches): $75 - $150

### Panel Accessibility Factors
- **Hood/Fender/Trunk**: 1.0x (standard rate)
- **Doors**: 1.1x
- **Roof**: 1.2x
- **Quarter Panels**: 1.3x
- **Pillars**: 1.5x

### Vehicle Adjustments
- **Luxury Brands**: 1.5x multiplier
- **Age Factor**: 2% reduction per year (minimum 80%)
- **Total Loss**: Repair cost > 75% of vehicle value

## Installation & Setup

### Backend Dependencies
```bash
# Install required packages
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install scipy>=1.11.0
```

### Frontend Dependencies
```bash
# UI components (already included in Suna)
npm install lucide-react
npm install @radix-ui/react-*
```

### Environment Setup
```bash
# No additional environment variables required
# Uses existing Suna configuration
```

## API Reference

### Analyze Hail Damage
```http
POST /api/analyze-hail-damage
Content-Type: multipart/form-data

Parameters:
- image: File (required)
- vehicle_year: Integer (optional)
- vehicle_make: String (optional)
- vehicle_model: String (optional)
- vehicle_value: Float (optional)
- analysis_mode: String (quick|detailed|insurance)
```

### Response Format
```json
{
  "assessment": {
    "total_dents": 45,
    "dents_by_severity": {...},
    "total_estimated_cost": 2850.00,
    "repair_method": "PDR (Paintless Dent Repair)",
    "is_total_loss": false,
    "confidence_score": 0.87
  },
  "detailed_report": "=== HAIL DAMAGE ASSESSMENT REPORT ===\n...",
  "processed_image": "data:image/jpeg;base64,...",
  "detected_dents": [...]
}
```

## Best Practices

### Image Quality Guidelines
1. **Lighting**: Use natural daylight or bright, even artificial lighting
2. **Angles**: Capture multiple angles for comprehensive coverage
3. **Distance**: Fill frame with vehicle while maintaining clarity
4. **Focus**: Ensure sharp focus on damaged areas
5. **Shadows**: Minimize harsh shadows that can obscure damage

### Analysis Tips
1. **Multiple Images**: Analyze different panels separately for accuracy
2. **Vehicle Info**: Provide complete vehicle details for precise estimates
3. **Mode Selection**: Use "insurance" mode for formal assessments
4. **Verification**: Cross-reference with manual inspection when possible

## Limitations & Considerations

### Technical Limitations
- **Paint Damage**: Cannot detect paint scratches or chips
- **Interior Damage**: Limited to exterior panel analysis
- **Lighting Dependency**: Poor lighting affects detection accuracy
- **Angle Sensitivity**: Extreme angles may reduce detection quality

### Business Considerations
- **Regional Pricing**: Costs may vary by geographic location
- **Shop Rates**: Individual shop pricing may differ from estimates
- **Insurance Policies**: Coverage terms affect actual repair costs
- **Vehicle Condition**: Pre-existing damage not accounted for

## Integration Examples

### React Component Usage
```tsx
import HailDamageAnalyzer from '@/components/HailDamageAnalyzer';

function App() {
  return (
    <div className="container">
      <HailDamageAnalyzer />
    </div>
  );
}
```

### API Integration
```javascript
const analyzeHailDamage = async (imageFile, vehicleInfo) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('analysis_mode', 'detailed');
  
  if (vehicleInfo.year) formData.append('vehicle_year', vehicleInfo.year);
  if (vehicleInfo.make) formData.append('vehicle_make', vehicleInfo.make);
  
  const response = await fetch('/api/analyze-hail-damage', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

## Support & Maintenance

### Monitoring
- API endpoint health checks
- Analysis accuracy metrics
- Performance monitoring
- Error rate tracking

### Updates
- Pricing structure adjustments
- Algorithm improvements
- New vehicle model support
- Enhanced detection methods

### Troubleshooting

#### Common Issues
1. **Low Confidence Scores**: Improve image quality or lighting
2. **Missing Detections**: Try different analysis modes
3. **Incorrect Classifications**: Verify vehicle information accuracy
4. **API Errors**: Check file format and size requirements

#### Error Codes
- `400`: Invalid file format or missing parameters
- `413`: File size exceeds limit
- `500`: Internal processing error
- `503`: Service temporarily unavailable

## Future Enhancements

### Planned Features
- **3D Damage Modeling**: Depth estimation for more accurate sizing
- **Machine Learning Training**: Continuous improvement from user feedback
- **Mobile App Integration**: Native mobile applications
- **Real-time Processing**: Live camera analysis
- **Historical Tracking**: Damage progression monitoring

### Research Areas
- **AI Model Optimization**: Improved detection algorithms
- **Multi-spectral Analysis**: UV and infrared damage detection
- **Automated Reporting**: Enhanced report generation
- **Integration APIs**: Third-party system connections

---

*This documentation is part of the Suna AI agent platform. For technical support or feature requests, please contact the development team.*