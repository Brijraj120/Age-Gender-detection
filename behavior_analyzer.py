# behavior_analyzer.py
import numpy as np
from collections import defaultdict, deque
import time

class BehaviorAnalyzer:
    def __init__(self):
        self.person_tracks = defaultdict(lambda: deque(maxlen=50))
        self.group_clusters = []
        self.suspicious_behaviors = []
        self.loitering_threshold = 30  # seconds
        
    def update_tracks(self, person_id, bbox, timestamp, stream_id):
        """Update tracking data for a person"""
        x, y, w, h = bbox
        center = (x + w/2, y + h/2)
        
        track_data = {
            'center': center,
            'timestamp': timestamp,
            'stream_id': stream_id,
            'bbox': bbox,
            'speed': self._calculate_speed(person_id, center, timestamp),
            'direction': self._calculate_direction(person_id, center)
        }
        
        self.person_tracks[person_id].append(track_data)
        
        # Analyze behavior patterns
        self._analyze_behavior(person_id)
    
    def _calculate_speed(self, person_id, new_center, new_time):
        """Calculate movement speed"""
        if len(self.person_tracks[person_id]) < 2:
            return 0
        
        prev_data = self.person_tracks[person_id][-2]
        prev_center, prev_time = prev_data['center'], prev_data['timestamp']
        
        # Calculate Euclidean distance
        distance = np.sqrt((new_center[0]-prev_center[0])**2 + (new_center[1]-prev_center[1])**2)
        time_diff = new_time - prev_time
        
        return distance / time_diff if time_diff > 0 else 0
    
    def _calculate_direction(self, person_id, new_center):
        """Calculate movement direction"""
        if len(self.person_tracks[person_id]) < 2:
            return (0, 0)
        
        prev_center = self.person_tracks[person_id][-2]['center']
        direction = (new_center[0] - prev_center[0], new_center[1] - prev_center[1])
        
        # Normalize direction vector
        magnitude = np.sqrt(direction[0]**2 + direction[1]**2)
        if magnitude > 0:
            direction = (direction[0]/magnitude, direction[1]/magnitude)
        
        return direction
    
    def _analyze_behavior(self, person_id):
        """Analyze behavior patterns for suspicious activities"""
        track = self.person_tracks[person_id]
        
        if len(track) < 10:
            return
        
        # Check for loitering
        if self.detect_loitering(person_id):
            behavior = {
                'type': 'loitering',
                'person_id': person_id,
                'duration': self.get_track_duration(person_id),
                'timestamp': time.time(),
                'severity': 'medium'
            }
            if behavior not in self.suspicious_behaviors:
                self.suspicious_behaviors.append(behavior)
        
        # Check for rapid movement
        if self.detect_rapid_movement(person_id):
            behavior = {
                'type': 'rapid_movement',
                'person_id': person_id,
                'speed': track[-1]['speed'],
                'timestamp': time.time(),
                'severity': 'low'
            }
            if behavior not in self.suspicious_behaviors:
                self.suspicious_behaviors.append(behavior)
    
    def detect_loitering(self, person_id, time_window=30):
        """Detect if a person is loitering in one area"""
        track = self.person_tracks[person_id]
        if len(track) < 5:
            return False
        
        # Check time spent in area
        first_time = track[0]['timestamp']
        last_time = track[-1]['timestamp']
        time_spent = last_time - first_time
        
        if time_spent < time_window:
            return False
        
        # Calculate total movement distance
        total_movement = 0
        for i in range(1, len(track)):
            prev_center = track[i-1]['center']
            curr_center = track[i]['center']
            distance = np.sqrt((curr_center[0]-prev_center[0])**2 + (curr_center[1]-prev_center[1])**2)
            total_movement += distance
        
        # If total movement is small, person is loitering
        return total_movement < 200  # pixels threshold
    
    def detect_rapid_movement(self, person_id, speed_threshold=50):
        """Detect unusually rapid movement"""
        track = self.person_tracks[person_id]
        if not track:
            return False
        
        recent_speeds = [data['speed'] for data in list(track)[-5:] if data['speed'] > 0]
        if not recent_speeds:
            return False
        
        avg_speed = np.mean(recent_speeds)
        return avg_speed > speed_threshold
    
    def get_track_duration(self, person_id):
        """Get duration person has been tracked"""
        track = self.person_tracks[person_id]
        if len(track) < 2:
            return 0
        return track[-1]['timestamp'] - track[0]['timestamp']
    
    def detect_group_formation(self, current_detections, proximity_threshold=100):
        """Detect groups of people forming"""
        groups = []
        used_ids = set()
        
        for i, det1 in enumerate(current_detections):
            if i in used_ids:
                continue
                
            group = [i]
            used_ids.add(i)
            x1, y1, w1, h1 = det1['bbox']
            center1 = (x1 + w1/2, y1 + h1/2)
            
            for j, det2 in enumerate(current_detections):
                if j in used_ids:
                    continue
                    
                x2, y2, w2, h2 = det2['bbox']
                center2 = (x2 + w2/2, y2 + h2/2)
                distance = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
                
                if distance < proximity_threshold:
                    group.append(j)
                    used_ids.add(j)
            
            if len(group) >= 2:  # Only consider groups of 2+ people
                groups.append({
                    'size': len(group),
                    'members': group,
                    'center': center1
                })
        
        return groups
    
    def get_suspicious_behaviors(self, clear_old=True):
        """Get recent suspicious behaviors"""
        current_time = time.time()
        
        if clear_old:
            # Remove behaviors older than 5 minutes
            self.suspicious_behaviors = [
                behavior for behavior in self.suspicious_behaviors
                if current_time - behavior['timestamp'] < 300
            ]
        
        return self.suspicious_behaviors