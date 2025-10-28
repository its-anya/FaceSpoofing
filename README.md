# Face Spoofing Detection System

This application detects face spoofing attempts during video calls using computer vision and machine learning techniques. It can identify various spoofing methods including printed photos, digital screen displays, and video replays.

## Features

- Real-time face detection
- Anti-spoofing detection using texture and color analysis
- Web interface for easy monitoring
- Confidence score for detection results

## Installation

make sure latest version Python 3.14.0 

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
python app.py
```

3. Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

## How It Works

The system uses a combination of techniques to detect spoofing:

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in the video stream.
2. **Texture Analysis**: Extracts Local Binary Pattern (LBP) features to analyze texture patterns.
3. **Color Analysis**: Examines color distributions across different color spaces to detect printed photos.
4. **Reflection Analysis**: Looks for natural reflection patterns that are present in real faces but not in spoofed ones.

## Testing the System

To test the anti-spoofing capabilities:

1. Run the application and ensure your webcam is working.
2. Try different spoofing methods:
   - Hold up a printed photo of a face
   - Display a face on a digital screen (phone, tablet, etc.)
   - Play a video of a face

The system will display "Real Face" for genuine faces and "Fake Face Detected!" for spoofing attempts, along with a confidence score.

## Limitations

- The current implementation may have difficulty with high-quality 3D masks.
- Performance depends on lighting conditions; good lighting improves detection accuracy.
- The system works best with frontal faces.

## Future Improvements

- Implement deep learning-based spoofing detection for higher accuracy
- Add support for multiple face detection and tracking
- Improve performance in challenging lighting conditions
- Add liveness detection through eye blink or head movement recognition
