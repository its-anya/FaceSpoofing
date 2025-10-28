import cv2
import numpy as np
import os

class FaceDetector:
    """Class for detecting faces in images using OpenCV."""
    
    def __init__(self, confidence_threshold=0.5):
        """Initialize the face detector.
        
        Args:
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained face detection model
        # Using Haar Cascade for simplicity and speed
        try:
            model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(model_path)
            
            # Check if the cascade classifier was loaded successfully
            if self.detector.empty():
                print("Warning: Haar cascade model not loaded properly. Check OpenCV installation.")
                # Fallback to a local copy if available
                if os.path.exists('haarcascade_frontalface_default.xml'):
                    print("Using local copy of Haar cascade model.")
                    self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            print("Using default parameters for face detection.")
            self.detector = cv2.CascadeClassifier()
    
    def detect(self, frame):
        """Detect faces in the input frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces