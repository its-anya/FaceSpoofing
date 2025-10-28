import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Global variables
detection_results = {
    "face_detected": "false",
    "is_real": "true",
    "confidence": 0.0,
    "timestamp": time.time()
}

def detect_faces(frame):
    """Detect faces in the input frame using Haar Cascade."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade
    cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces

def detect_spoofing(face_region):
    """Simple spoofing detection based on image quality."""
    if face_region.size == 0:
        return True, 0.0
    
    # Convert to grayscale
    if len(face_region.shape) == 3:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_region
    
    # Calculate image quality metrics
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate reflection patterns (specular highlights)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    reflection_ratio = np.sum(thresh) / (gray.shape[0] * gray.shape[1] * 255)
    
    # Higher values indicate more likely to be real
    score = (blur / 1000) - (reflection_ratio * 10)
    
    # Normalize to 0-1 range
    confidence = 1 / (1 + np.exp(-score))  # Sigmoid function
    
    # Classify based on confidence
    is_real = confidence > 0.5
    
    return is_real, confidence

def generate_frames():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame for face detection and spoofing
        frame, results = process_frame(frame)
        
        # Update global detection results
        global detection_results
        detection_results = results
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

def process_frame(frame):
    """Process a frame for face detection and spoofing detection."""
    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    
    # Detect faces
    faces = detect_faces(frame)
    
    results = {
        "face_detected": "true" if len(faces) > 0 else "false",
        "is_real": "true",
        "confidence": 0.0,
        "timestamp": time.time()
    }
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face region for spoofing detection
        face_region = frame[y:y+h, x:x+w]
        
        # Check for spoofing
        is_real, confidence = detect_spoofing(face_region)
        
        results["is_real"] = "true" if is_real else "false"
        results["confidence"] = confidence
        
        # Display spoofing detection result
        status = "Real" if is_real else "Fake"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        
        cv2.putText(display_frame, f"{status} ({confidence:.2f})", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return display_frame, results

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Return the current detection status as JSON."""
    return jsonify(detection_results)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)