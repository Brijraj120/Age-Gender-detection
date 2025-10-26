# crowd_analyzer.py
import cv2
import numpy as np
from collections import deque

class CrowdAnalyzer:
    def __init__(self):
        self.heatmap = None
        self.density_history = deque(maxlen=100)
        self.frame_shape = None
        
    def update_heatmap(self, detections, frame_shape):
        """Update crowd heatmap with new detections"""
        if self.heatmap is None or self.frame_shape != frame_shape:
            self.heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
            self.frame_shape = frame_shape
        
        # Create temporary heatmap for current frame
        temp_heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            center_x, center_y = x + w//2, y + h//2
            
            # Create Gaussian blob at detection center
            cv2.circle(temp_heatmap, (center_x, center_y), 30, 1, -1)
        
        # Update main heatmap with decay
        self.heatmap = 0.95 * self.heatmap + 0.05 * temp_heatmap
        
        # Normalize heatmap for visualization
        heatmap_viz = self.heatmap / (self.heatmap.max() + 1e-8) * 255
        heatmap_viz = heatmap_viz.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def get_density_zones(self):
        """Identify high, medium, low density zones"""
        if self.heatmap is None or self.heatmap.max() == 0:
            return {
                'high_density': 0,
                'medium_density': 0,
                'low_density': self.frame_shape[0] * self.frame_shape[1] if self.frame_shape else 0
            }
        
        # Calculate density thresholds
        high_thresh = np.percentile(self.heatmap, 80)
        medium_thresh = np.percentile(self.heatmap, 50)
        
        high_mask = self.heatmap > high_thresh
        medium_mask = (self.heatmap > medium_thresh) & (self.heatmap <= high_thresh)
        low_mask = self.heatmap <= medium_thresh
        
        return {
            'high_density': np.sum(high_mask),
            'medium_density': np.sum(medium_mask),
            'low_density': np.sum(low_mask),
            'total_area': self.heatmap.size
        }
    
    def calculate_crowd_density(self, detections, frame_area):
        """Calculate crowd density percentage"""
        if frame_area == 0:
            return 0
        
        # Estimate area occupied by people
        person_area = sum(detection['bbox'][2] * detection['bbox'][3] for detection in detections)
        density_percentage = (person_area / frame_area) * 100
        
        return min(density_percentage, 100)  # Cap at 100%