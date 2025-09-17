# AVIS Autonomous Car System

A comprehensive autonomous driving system for the AVIS Engine simulator, featuring both high-speed race track navigation and complex urban environment handling with traffic sign recognition.

## Features

### Race Mode
- **High-speed lane following** with advanced color detection (yellow lane markers)
- **Real-time obstacle detection** using YOLOv11 segmentation model (ONNX)
- **Dynamic obstacle avoidance** with lane switching capabilities
- **Sensor fusion** combining vision and ultrasonic sensor data
- **Adaptive speed control** based on steering angle
- **Exponential smoothing** for stable control

### Urban Mode
- **Traffic sign detection** supporting 5 classes:
  - Traffic lights
  - Proceed Forward
  - Proceed Left
  - Proceed Right
  - Stop signs
- **White dashed lane following** for urban roads
- **State machine-based navigation** for complex intersection handling
- **Real-time decision making** at traffic lights and stop signs
- **Optimized inference** at 384×384 resolution for low latency


### Installation

1. **Clone the repository**
```bash
git clone https://github.com/parhamsoltani/avis-autonomous-car.git
cd avis-autonomous-car
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```


3. **Verify installation**
```bash
python test.py
```

## Usage

### Starting the Simulator
1. Launch the AVIS Engine simulator
2. Select the appropriate track (Race or Urban)
3. Note the IP address and port (default: `127.0.0.3:25004`)

### Running Autonomous Modes

**Race Mode:**
```bash
python run.py race
```

**Urban Mode:**
```bash
python run.py urban
```

**Direct module execution:**
```bash
# Race mode
cd race_mode
python race_main.py

# Urban mode  
cd urban_mode
python urban_main.py
```


## Configuration

### Race Mode Configuration (`race_mode/race_config.py`)
```python
BASE_SPEED = 15         # Base driving speed
MAX_SPEED = 20          # Maximum speed limit
MIN_SPEED = 7           # Minimum speed (turns)
STEERING_SMOOTH_FACTOR = 0.85  # Steering smoothing
SENSOR_SMOOTH_FACTOR = 0.3     # Sensor data smoothing
```

### Urban Mode Configuration (`urban_mode/urban_config.py`)
```python
BASE_SPEED = 10         # Urban driving speed
SIGN_CONFIDENCE = 0.8   # Sign detection threshold
TRAFFIC_LIGHT_WAIT = 3.0  # Wait time at red lights
TURN_DURATION = 5.0     # Turn maneuver duration
```

## Control System

### Lane Detection Pipeline
1. **Preprocessing**: Median blur (7x7) for noise reduction
2. **Color Segmentation**: HSV-based detection for lane markers
3. **Morphological Operations**: Erosion and dilation for clean masks
4. **Contour Analysis**: Find lane boundaries using distance-based selection
5. **Perspective Transform**: Warped view for accurate lane positioning
6. **Exponential Smoothing**: Stable steering output

### Obstacle Avoidance Strategy
- **Detection**: YOLOv11 segmentation at 640×640 resolution
- **Decision**: Lane switching based on sensor data and free space
- **Execution**: Smooth steering transitions with speed adaptation
- **Recovery**: Return to original lane after obstacle clearance

### Traffic Sign Recognition Flow
1. **Detection**: ONNX model inference at 384×384 resolution
2. **Classification**: 5-class detection with confidence scoring
3. **State Management**: Finite state machine for traffic scenarios
4. **Action Execution**: Controlled maneuvers based on sign type

## Performance

### System Requirements
- **Minimum**: Intel i5, 8GB RAM, GTX 1050
- **Recommended**: Intel i7, 16GB RAM, RTX 2060
- **Optimal**: Intel i9, 32GB RAM, RTX 3070+

### Benchmarks
| Component | CPU (fps) | GPU (fps) | Latency (ms) |
|-----------|-----------|-----------|--------------|
| Lane Detection | 45 | 60+ | 15-20 |
| Obstacle Detection | 15 | 30+ | 30-40 |
| Sign Detection | 20 | 40+ | 20-25 |
| Full Pipeline | 12 | 25+ | 40-50 |

## Troubleshooting

### Common Issues

**Connection Failed**
```bash
# Check simulator is running
# Verify IP and port in config
# Default: 127.0.0.3:25004
```

**Model Not Found**
```bash
# Ensure ONNX models are in models/ directory
# Check file names match exactly:
# - obstacle_segmentation.onnx
# - sign_detection.onnx
```

**Low FPS Performance**
```bash
# Reduce resolution in config
# Enable GPU acceleration (CUDA)
# Increase frame skipping for inference
```

**Import Errors**
```bash
# Reinstall dependencies:
pip install --upgrade -r requirements.txt
```

## Testing

Run the comprehensive test suite:
```bash
python test.py
```

This will verify:
- ONNX Runtime installation
- Model file presence and loading
- Module imports
- Lane detector initialization
- Legacy component availability

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Technical Details

### Color Space Configuration (HSV)
```python
# Optimized color ranges for detection
YELLOW_LINE = [22, 102, 122] to [30, 255, 255]
WHITE_LINE = [0, 11, 148] to [41, 19, 255]
BLUE_LANE = [100, 10, 25] to [120, 50, 60]
```

### Perspective Transform Points
```python
# Bird's-eye view transformation
top_left = (160, 230)
top_right = (352, 230)
bottom_right = (482, 392)
bottom_left = (60, 392)
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **AVIS Engine Team** for the simulation platform
- **Ultralytics** for YOLOv11 architecture
- **ONNX Runtime Team** for efficient inference
- **OpenCV Community** for computer vision tools

## Contact

For questions, issues, or collaborations:
- Create an [Issue](https://github.com/parhamsoltani/avis-autonomous-car/issues)
- Email: parham.soltany@gmail.com
- AVIS Engine Support: amir@avisengine.com


---

**Last Updated**: September 2025
**Version**: 2.0.0
**Maintainer**: Parham Soltani
