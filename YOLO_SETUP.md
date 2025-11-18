# YOLOv8 Model Setup Guide

## Quick Setup for YOLOv8n.pt

### Step 1: Download YOLOv8n Model
1. Download `yolov8n.pt` from:
   - **Official Ultralytics**: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   - **Direct link**: https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
   - Or use: `wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt`

### Step 2: Place Model File
1. Create folder: `ai_models` (if it doesn't exist)
2. Place `yolov8n.pt` inside: `ai_models/yolov8n.pt`

### Step 3: Install Dependencies (Optional for Better Performance)
```bash
pip install ultralytics
```

## File Structure
```
MPT-Karan-5/
├── ai_models/
│   └── yolov8n.pt          ← Place your downloaded model here
├── complete_dashboard.py
├── local_ai_engine.py
└── ...
```

## What Happens:
- **With yolov8n.pt + ultralytics**: Full YOLOv8 object detection (80 classes)
- **With yolov8n.pt only**: OpenCV DNN YOLOv8 detection 
- **Without yolov8n.pt**: Falls back to OpenCV cascade detection

## Model Performance:
- **YOLOv8n**: Fast, accurate, 80 COCO classes
- **File size**: ~6MB
- **Speed**: Real-time on CPU, faster on GPU
- **Accuracy**: Excellent for restaurant monitoring

## Test Your Setup:
```bash
python test_ai_engine.py
```

This will show you which detection method is being used and verify everything works!