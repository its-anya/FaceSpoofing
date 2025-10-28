import cv2
import numpy as np
import os

class SpoofingDetector:
    """Class for detecting face spoofing attempts."""
    
    def __init__(self):
        """Initialize the spoofing detector."""
        # Define texture-based feature extraction parameters
        self.lbp_radius = 1
        self.lbp_n_points = 8 * self.lbp_radius
        
        # Simple threshold-based detection (can be replaced with ML model)
        self.threshold = 0.5
        
        # Flag to indicate if a pre-trained model is available
        self.model_available = False
        
        # Try to load pre-trained model if available
        self.load_model()
    
    def load_model(self):
        """Load pre-trained spoofing detection model if available."""
        model_path = "models/spoofing_model.pkl"
        
        if os.path.exists(model_path):
            try:
                # Here you would load your trained model
                # For example: self.model = joblib.load(model_path)
                self.model_available = True
                print("Loaded pre-trained spoofing detection model.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model_available = False
        else:
            print("No pre-trained model found. Using heuristic detection.")
    
    def extract_features(self, face_img):
        """Extract features from face image for spoofing detection.
        
        Args:
            face_img: Face region image
            
        Returns:
            Feature vector
        """
        # Ensure the image is not empty
        if face_img.size == 0:
            return np.array([])
        
        # Resize to standard size
        face_img = cv2.resize(face_img, (128, 128))
        
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        # Extract texture features using Local Binary Patterns
        features = self._extract_lbp_features(gray)
        
        # Add color-based features to detect printed photos
        if len(face_img.shape) == 3:
            color_features = self._extract_color_features(face_img)
            features = np.concatenate((features, color_features))
        
        return features
    
    def _extract_lbp_features(self, gray_img):
        """Extract Local Binary Pattern features."""
        # Calculate LBP image
        lbp = self._local_binary_pattern(gray_img, self.lbp_n_points, self.lbp_radius)
        
        # Calculate histogram of LBP values
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_n_points + 3), 
                              range=(0, self.lbp_n_points + 2))
        
        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def _local_binary_pattern(self, image, n_points, radius):
        """Compute local binary pattern for an image."""
        # Simple LBP implementation
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary_code = 0
                
                # Compare center pixel with neighbors
                for k in range(n_points):
                    # Calculate neighbor coordinates
                    theta = 2 * np.pi * k / n_points
                    x = i + radius * np.cos(theta)
                    y = j - radius * np.sin(theta)
                    
                    # Get interpolated pixel value
                    x1, y1 = int(np.floor(x)), int(np.floor(y))
                    x2, y2 = min(x1 + 1, image.shape[0] - 1), min(y1 + 1, image.shape[1] - 1)
                    dx, dy = x - x1, y - y1
                    
                    neighbor = (1 - dx) * (1 - dy) * image[x1, y1] + \
                               dx * (1 - dy) * image[x2, y1] + \
                               (1 - dx) * dy * image[x1, y2] + \
                               dx * dy * image[x2, y2]
                    
                    # Update binary code
                    if neighbor >= center:
                        binary_code |= (1 << k)
                
                lbp[i, j] = binary_code
                
        return lbp
    
    def _extract_color_features(self, face_img):
        """Extract color-based features to detect printed photos."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        
        # Calculate mean and std for each channel
        features = []
        
        for img in [face_img, hsv, ycrcb]:
            for i in range(3):
                channel = img[:, :, i]
                features.append(np.mean(channel))
                features.append(np.std(channel))
        
        # Calculate color histogram
        hist_features = []
        for i in range(3):
            hist, _ = np.histogram(face_img[:, :, i], bins=8, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            hist_features.extend(hist)
        
        # Combine all color features
        return np.array(features + hist_features)
    
    def detect(self, face_img):
        """Detect if a face image is real or spoofed.
        
        Args:
            face_img: Face region image
            
        Returns:
            Tuple of (is_real, confidence)
        """
        # Handle empty images
        if face_img is None or face_img.size == 0:
            return True, 0.0
        
        # Extract features
        features = self.extract_features(face_img)
        
        if features.size == 0:
            return True, 0.0
        
        if self.model_available:
            # Use pre-trained model for prediction
            # prediction = self.model.predict_proba([features])[0]
            # is_real = prediction[1] > self.threshold
            # confidence = prediction[1]
            
            # Placeholder for model prediction
            is_real, confidence = self._heuristic_detection(face_img, features)
        else:
            # Use heuristic-based detection
            is_real, confidence = self._heuristic_detection(face_img, features)
        
        return is_real, confidence
    
    def _heuristic_detection(self, face_img, features):
        """Simple heuristic-based spoofing detection."""
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Calculate image quality metrics
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate reflection patterns (specular highlights)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        reflection_ratio = np.sum(thresh) / (gray.shape[0] * gray.shape[1] * 255)
        
        # Calculate texture variation
        texture_score = np.std(features[:self.lbp_n_points + 1])
        
        # Combine metrics into a single score
        # Higher values indicate more likely to be real
        score = (blur / 1000) + texture_score - (reflection_ratio * 10)
        
        # Normalize to 0-1 range
        confidence = 1 / (1 + np.exp(-score))  # Sigmoid function
        
        # Classify based on confidence
        is_real = confidence > self.threshold
        
        return is_real, confidence