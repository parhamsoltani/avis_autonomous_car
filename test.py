#!/usr/bin/env python3
"""
Test script to verify all components are working
"""
import sys
import os
import cv2
import numpy as np

print("Testing Enhanced AVIS System with ONNX Models...")
print("-" * 50)

# Test 1: Check ONNX Runtime
print("1. Testing ONNX Runtime...")
try:
    import onnxruntime as ort
    print(f"   ONNX Runtime version: {ort.__version__}")
except ImportError:
    print("   Error: onnxruntime not installed. Install with: pip install onnxruntime")

# Test 2: Check ONNX models
print("\n2. Checking ONNX models...")
models_dir = "models"
required_models = [
    "obstacle_segmentation.onnx",
    "sign_detection.onnx"
]

for model_name in required_models:
    model_path = os.path.join(models_dir, model_name)
    if os.path.exists(model_path):
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(f"   {model_name}: loaded successfully")
            print(f"     Input: {input_name}, Shape: {input_shape}")
        except Exception as e:
            print(f"   {model_name}: failed to load - {e}")
    else:
        print(f"   {model_name}: file not found")
        print(f"     Please copy the model to {model_path}")

# Test 3: Test Race Mode components
print("\n3. Testing Race Mode components...")
try:
    from race_mode.race_main import RaceMode
    print("   Race mode imports successfully")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Test Urban Mode components
print("\n4. Testing Urban Mode components...")
try:
    from urban_mode.urban_main import UrbanMode
    from urban_mode.sign_detector_onnx import SignDetectorONNX
    print("   Urban mode imports successfully")
    print("   ONNX sign detector available")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: Test lane detection
print("\n5. Testing lane detection...")
try:
    from common.lane_detection import LaneDetector
    detector = LaneDetector()
    print("   Lane detector initialized")
except Exception as e:
    print(f"   Error: {e}")

# Test 6: Verify crosswalk detector is available but not used
print("\n6. Checking legacy components (kept for compatibility)...")
try:
    from urban_mode.crosswalk_detector import CrosswalkDetector
    print("   Crosswalk detector available (not used in main)")
    from urban_mode.apriltag_detector import AprilTagDetector
    print("   AprilTag detector available (not used in main)")
except Exception as e:
    print(f"   Note: Legacy components not available - {e}")

print("\n" + "-" * 50)
print("System check complete!")
print("\nUsage:")
print("  python run.py race    # For race mode with obstacle detection")
print("  python run.py urban   # For urban mode with sign detection")
print("\nNote: Make sure to place the ONNX models in the 'models' directory:")
print("  - models/obstacle_segmentation.onnx")
print("  - models/sign_detection.onnx")