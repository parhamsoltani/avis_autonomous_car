#!/usr/bin/env python3
"""
AVIS Autonomous Car System - Launcher
Usage:
    python run.py race    # For race mode
    python run.py urban   # For urban mode
"""
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [race|urban]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "race":
        print("Starting Race Mode...")
        os.chdir("race_mode")
        os.system("python race_main.py")
    elif mode == "urban":
        print("Starting Urban Mode...")
        os.chdir("urban_mode")
        os.system("python urban_main.py")
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'race' or 'urban'")
        sys.exit(1)