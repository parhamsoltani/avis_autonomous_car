#!/usr/bin/env python3
"""
Setup script for AVIS Autonomous Car System
"""
import os
import sys
import subprocess

def setup():
    print("Setting up AVIS Autonomous Car System...")
    
    # Install requirements
    print("\n1. Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check for model files
    print("\n2. Checking for model files...")
    model_path = os.path.join("urban_mode", "best_model.h5")
    if not os.path.exists(model_path):
        print(f"WARNING: {model_path} not found!")
        print("Please copy best_model.h5 from the reference project to urban_mode/")
    else:
        print("Model file found ✓")
    
    # Check for car mask
    mask_path = os.path.join("urban_mode", "car_mask.npy")
    if not os.path.exists(mask_path):
        print(f"\n3. Car mask not found. Generating...")
        response = input("Do you want to generate car mask now? (y/n): ")
        if response.lower() == 'y':
            os.chdir("urban_mode")
            subprocess.run([sys.executable, "generate_car_mask.py"])
            os.chdir("..")
    else:
        print("3. Car mask found ✓")
    
    print("\nSetup complete!")
    print("\nUsage:")
    print("  Race mode:  python main.py --mode race")
    print("  Urban mode: python main.py --mode urban")
    print("\nOptional arguments:")
    print("  --ip IP     Simulator IP (default: 127.0.0.1)")
    print("  --port PORT Simulator port (default: 25001)")

if __name__ == "__main__":
    setup()