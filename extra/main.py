#!/usr/bin/env python3
"""
AVIS Autonomous Car System - Main Launcher
"""
import sys
import argparse
from race_mode.race_main import RaceMode
from urban_mode.urban_main import UrbanMode

def main():
    parser = argparse.ArgumentParser(description='AVIS Autonomous Car System')
    parser.add_argument('--mode', type=str, choices=['race', 'urban'], 
                       required=True, help='Select driving mode')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                       help='Simulator IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=25001,
                       help='Simulator port (default: 25001)')
    
    args = parser.parse_args()
    
    print(f"Starting AVIS Autonomous Car in {args.mode.upper()} mode")
    print(f"Connecting to simulator at {args.ip}:{args.port}")
    
    try:
        if args.mode == 'race':
            driver = RaceMode(ip=args.ip, port=args.port)
        else:  # urban
            driver = UrbanMode(ip=args.ip, port=args.port)
        
        driver.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()