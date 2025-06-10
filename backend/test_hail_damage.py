#!/usr/bin/env python3
"""
Test script for the Hail Damage Estimation Tool

This script demonstrates how to use the hail damage analysis functionality
and provides examples for testing the system with sample images.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Optional

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.tools.hail_damage_tool import HailDamageEstimationTool
from agentpress.thread_manager import ThreadManager

class HailDamageTestSuite:
    """Test suite for hail damage analysis functionality."""
    
    def __init__(self):
        self.thread_manager = ThreadManager()
        self.project_id = "test_hail_damage"
        self.thread_id = "test_thread"
        self.tool = HailDamageEstimationTool(
            self.project_id, 
            self.thread_id, 
            self.thread_manager
        )
    
    async def test_basic_analysis(self, image_path: str) -> dict:
        """Test basic hail damage analysis."""
        print(f"\n🔍 Testing basic analysis on: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return {}
        
        try:
            result = await self.tool.analyze_hail_damage(
                image_path=image_path,
                analysis_mode="detailed"
            )
            
            if result.success:
                assessment = result.result['assessment']
                print(f"✅ Analysis completed successfully!")
                print(f"   Total dents detected: {assessment['total_dents']}")
                print(f"   Estimated cost: ${assessment['total_estimated_cost']:,.2f}")
                print(f"   Repair method: {assessment['repair_method']}")
                print(f"   Confidence: {assessment['confidence_score']:.1%}")
                return result.result
            else:
                print(f"❌ Analysis failed: {result.error}")
                return {}
                
        except Exception as e:
            print(f"❌ Exception during analysis: {str(e)}")
            return {}
    
    async def test_vehicle_specific_analysis(self, image_path: str) -> dict:
        """Test analysis with vehicle-specific information."""
        print(f"\n🚗 Testing vehicle-specific analysis on: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return {}
        
        try:
            result = await self.tool.analyze_hail_damage(
                image_path=image_path,
                vehicle_year=2022,
                vehicle_make="BMW",
                vehicle_model="X5",
                vehicle_value=65000,
                analysis_mode="insurance"
            )
            
            if result.success:
                assessment = result.result['assessment']
                print(f"✅ Vehicle-specific analysis completed!")
                print(f"   Vehicle: 2022 BMW X5 (${65000:,})")
                print(f"   Total dents: {assessment['total_dents']}")
                print(f"   Estimated cost: ${assessment['total_estimated_cost']:,.2f}")
                print(f"   Total loss: {'Yes' if assessment['is_total_loss'] else 'No'}")
                print(f"   Repair method: {assessment['repair_method']}")
                return result.result
            else:
                print(f"❌ Analysis failed: {result.error}")
                return {}
                
        except Exception as e:
            print(f"❌ Exception during analysis: {str(e)}")
            return {}
    
    async def test_repair_time_estimation(self, assessment_data: dict) -> dict:
        """Test repair time estimation."""
        print(f"\n⏱️ Testing repair time estimation")
        
        if not assessment_data:
            print(f"❌ No assessment data provided")
            return {}
        
        try:
            assessment = assessment_data.get('assessment', {})
            severity_breakdown = assessment.get('dents_by_severity', {})
            
            result = await self.tool.estimate_repair_time(
                total_dents=assessment.get('total_dents', 0),
                repair_method=assessment.get('repair_method', 'PDR'),
                severity_breakdown=severity_breakdown
            )
            
            if result.success:
                time_data = result.result
                print(f"✅ Time estimation completed!")
                print(f"   Labor hours: {time_data['estimated_hours']}")
                print(f"   Business days: {time_data['business_days']}")
                print(f"   Total days (with buffer): {time_data['total_days_with_buffer']}")
                return result.result
            else:
                print(f"❌ Time estimation failed: {result.error}")
                return {}
                
        except Exception as e:
            print(f"❌ Exception during time estimation: {str(e)}")
            return {}
    
    async def test_insurance_report(self, assessment_data: dict) -> dict:
        """Test insurance report generation."""
        print(f"\n📋 Testing insurance report generation")
        
        if not assessment_data:
            print(f"❌ No assessment data provided")
            return {}
        
        try:
            result = await self.tool.generate_insurance_report(
                assessment_data=assessment_data,
                policy_holder="John Doe",
                claim_number="CLM-2024-001234",
                adjuster_name="Jane Smith"
            )
            
            if result.success:
                print(f"✅ Insurance report generated successfully!")
                report_length = len(result.result.get('insurance_report', ''))
                print(f"   Report length: {report_length} characters")
                print(f"   Summary included: {'Yes' if 'summary' in result.result else 'No'}")
                return result.result
            else:
                print(f"❌ Report generation failed: {result.error}")
                return {}
                
        except Exception as e:
            print(f"❌ Exception during report generation: {str(e)}")
            return {}
    
    def create_sample_test_image(self, output_path: str) -> bool:
        """Create a sample test image for demonstration purposes."""
        try:
            import cv2
            import numpy as np
            
            # Create a sample car-like image with simulated dents
            img = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Light gray background
            
            # Draw a simple car outline
            cv2.rectangle(img, (100, 200), (700, 500), (150, 150, 150), -1)  # Car body
            cv2.rectangle(img, (150, 150), (650, 200), (120, 120, 120), -1)  # Roof
            
            # Add some simulated "dents" (darker circles)
            dent_positions = [
                (250, 300, 15),  # (x, y, radius)
                (350, 280, 12),
                (450, 320, 18),
                (550, 290, 10),
                (300, 400, 14),
                (500, 380, 16)
            ]
            
            for x, y, r in dent_positions:
                cv2.circle(img, (x, y), r, (100, 100, 100), -1)  # Dark circles as "dents"
                cv2.circle(img, (x, y), r+2, (80, 80, 80), 2)    # Darker outline
            
            # Save the image
            cv2.imwrite(output_path, img)
            print(f"✅ Sample test image created: {output_path}")
            return True
            
        except ImportError:
            print(f"❌ OpenCV not available for creating sample image")
            return False
        except Exception as e:
            print(f"❌ Error creating sample image: {str(e)}")
            return False
    
    async def run_comprehensive_test(self, image_path: Optional[str] = None):
        """Run a comprehensive test of all functionality."""
        print("🚀 Starting Hail Damage Analysis Test Suite")
        print("=" * 50)
        
        # Create sample image if none provided
        if not image_path:
            sample_path = "sample_hail_damage.jpg"
            if self.create_sample_test_image(sample_path):
                image_path = sample_path
            else:
                print("❌ No test image available. Please provide an image path.")
                return
        
        # Test 1: Basic Analysis
        basic_result = await self.test_basic_analysis(image_path)
        
        # Test 2: Vehicle-Specific Analysis
        vehicle_result = await self.test_vehicle_specific_analysis(image_path)
        
        # Test 3: Repair Time Estimation
        if basic_result:
            await self.test_repair_time_estimation(basic_result)
        
        # Test 4: Insurance Report
        if vehicle_result:
            insurance_result = await self.test_insurance_report(vehicle_result)
            
            # Save sample report
            if insurance_result and 'insurance_report' in insurance_result:
                with open('sample_insurance_report.txt', 'w') as f:
                    f.write(insurance_result['insurance_report'])
                print(f"📄 Sample insurance report saved to: sample_insurance_report.txt")
        
        # Test 5: Save sample results
        if vehicle_result:
            with open('sample_analysis_results.json', 'w') as f:
                json.dump(vehicle_result, f, indent=2, default=str)
            print(f"💾 Sample analysis results saved to: sample_analysis_results.json")
        
        print("\n" + "=" * 50)
        print("🎉 Test suite completed!")
        
        # Clean up sample image if we created it
        if image_path == "sample_hail_damage.jpg" and os.path.exists(image_path):
            os.remove(image_path)
            print(f"🧹 Cleaned up sample image: {image_path}")

def main():
    """Main function to run the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Hail Damage Analysis Tool')
    parser.add_argument('--image', '-i', type=str, help='Path to test image')
    parser.add_argument('--create-sample', '-c', action='store_true', 
                       help='Create a sample test image')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = HailDamageTestSuite()
    
    if args.create_sample:
        # Just create a sample image and exit
        sample_path = "sample_hail_damage.jpg"
        if test_suite.create_sample_test_image(sample_path):
            print(f"Sample image created: {sample_path}")
        return
    
    # Run comprehensive test
    try:
        asyncio.run(test_suite.run_comprehensive_test(args.image))
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()