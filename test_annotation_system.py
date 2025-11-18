#!/usr/bin/env python3
"""
Test the enhanced annotation system with KPI tracking
"""

from local_ai_engine import LocalAIEngine
import numpy as np
import cv2
import os

def create_test_video(output_path: str, width: int = 640, height: int = 480, fps: int = 30, duration: int = 5):
    """Create a simple test video with moving objects"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for frame_num in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some moving rectangles to simulate people
        for i in range(3):
            x = int((frame_num * 2 + i * 50) % (width - 50))
            y = int(200 + i * 100)
            cv2.rectangle(frame, (x, y), (x + 40, y + 80), (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"Test Video Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {output_path}")

def test_kitchen_annotations():
    """Test kitchen sink area annotations"""
    print("🧪 Testing Kitchen Sink Annotations...")
    
    # Create test video
    video_path = "test_kitchen_video.mp4"
    create_test_video(video_path)
    
    # Initialize AI engine
    ai_engine = LocalAIEngine()
    
    # Define sink annotation (center of frame with radius)
    sink_annotation = {
        'sink_location': {
            'x': 320,  # Center X
            'y': 240,  # Center Y  
            'radius': 80
        }
    }
    
    # Process video with annotations
    results = ai_engine.process_video_with_annotations(video_path, 'kitchen', sink_annotation)
    
    print(f"📊 Kitchen processing results: {results}")
    
    # Clean up
    if os.path.exists(video_path):
        os.remove(video_path)
    
    return results

def test_dining_annotations():
    """Test dining queue area annotations"""
    print("🧪 Testing Dining Queue Annotations...")
    
    # Create test video
    video_path = "test_dining_video.mp4"
    create_test_video(video_path)
    
    # Initialize AI engine
    ai_engine = LocalAIEngine()
    
    # Define queue area annotation (rectangular region)
    queue_annotation = {
        'queue_area': {
            'x1': 100,  # Top-left X
            'y1': 150,  # Top-left Y
            'x2': 300,  # Bottom-right X
            'y2': 350   # Bottom-right Y
        }
    }
    
    # Process video with annotations
    results = ai_engine.process_video_with_annotations(video_path, 'dining', queue_annotation)
    
    print(f"📊 Dining processing results: {results}")
    
    # Clean up
    if os.path.exists(video_path):
        os.remove(video_path)
    
    return results

def test_parking_kpis():
    """Test parking KPI tracking"""
    print("🧪 Testing Parking KPI Tracking...")
    
    # Create test video
    video_path = "test_parking_video.mp4"
    create_test_video(video_path)
    
    # Initialize AI engine
    ai_engine = LocalAIEngine()
    
    # Process video (no annotations needed for parking)
    results = ai_engine.process_video_with_annotations(video_path, 'parking', {})
    
    print(f"📊 Parking processing results: {results}")
    
    # Clean up
    if os.path.exists(video_path):
        os.remove(video_path)
    
    return results

def check_output_files():
    """Check if KPI output files were created"""
    print("📁 Checking KPI output files...")
    
    for region in ['kitchen', 'dining', 'parking']:
        file_path = f'data/output_{region}.txt'
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            # Show first few lines
            with open(file_path, 'r') as f:
                lines = f.readlines()[:10]
                print("📄 First 10 lines:")
                for line in lines:
                    print(f"   {line.strip()}")
                print("   ...")
        else:
            print(f"❌ Missing: {file_path}")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Annotation System with KPI Tracking")
    print("=" * 60)
    
    try:
        # Test kitchen annotations
        kitchen_results = test_kitchen_annotations()
        print()
        
        # Test dining annotations  
        dining_results = test_dining_annotations()
        print()
        
        # Test parking KPIs
        parking_results = test_parking_kpis()
        print()
        
        # Check output files
        check_output_files()
        
        print("=" * 60)
        print("✅ All annotation tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()