# face_recognizer.py
import numpy as np
import pickle
import os
import cv2
import time
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self):
        self.known_faces = {}
        self.encodings_file = "data/known_faces.pkl"
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_faces = {}
        else:
            self.known_faces = {}
            print("No known faces file found. Starting fresh.")
    
    def save_known_faces(self):
        """Save known faces to file"""
        try:
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"Saved {len(self.known_faces)} known faces")
        except Exception as e:
            print(f"Error saving known faces: {e}")
    
    def extract_face_features(self, face_image):
        """Extract basic face features using OpenCV"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            resized = cv2.resize(gray, (64, 64))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Flatten and return
            return normalized.flatten()
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return None
    
    def register_face(self, image, person_id, name, metadata=None):
        """Register a new face to the whitelist"""
        try:
            # Extract face features
            features = self.extract_face_features(image)
            
            if features is not None:
                self.known_faces[person_id] = {
                    'features': features,
                    'name': name,
                    'metadata': metadata or {},
                    'registration_date': time.time(),
                    'last_seen': time.time()
                }
                self.save_known_faces()
                return True
            return False
            
        except Exception as e:
            print(f"Face registration error: {e}")
            return False
    
    def recognize_face(self, image):
        """Recognize face from known faces using cosine similarity"""
        try:
            # Extract features from input image
            unknown_features = self.extract_face_features(image)
            
            if unknown_features is None:
                return None
            
            best_match = None
            best_similarity = 0.7  # Minimum similarity threshold
            
            for person_id, data in self.known_faces.items():
                known_features = data['features']
                
                # Calculate cosine similarity
                similarity = cosine_similarity([unknown_features], [known_features])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (person_id, data['name'])
            
            if best_match:
                # Update last seen time
                self.known_faces[best_match[0]]['last_seen'] = time.time()
                return best_match
            
            return None
            
        except Exception as e:
            print(f"Face recognition error: {e}")
            return None
    
    def get_similarity_score(self, image1, image2):
        """Get similarity score between two face images"""
        try:
            features1 = self.extract_face_features(image1)
            features2 = self.extract_face_features(image2)
            
            if features1 is None or features2 is None:
                return 0.0
            
            # Calculate cosine similarity
            similarity = cosine_similarity([features1], [features2])[0][0]
            
            # Convert to percentage (0-100)
            return max(0, similarity * 100)
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0
    
    def get_known_faces_list(self):
        """Get list of all known faces"""
        return [
            {
                'person_id': pid,
                'name': data['name'],
                'registration_date': data['registration_date'],
                'last_seen': data.get('last_seen', 0)
            }
            for pid, data in self.known_faces.items()
        ]