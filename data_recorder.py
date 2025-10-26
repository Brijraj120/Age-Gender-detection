# data_recorder.py - Enhanced Data Recording System
import json
import csv
import os
import time
import logging
from datetime import datetime
from collections import defaultdict

class DataRecorder:
    def __init__(self, data_dir='data'):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.sessions = {}
        self.current_session = None
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'sessions'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'exports'), exist_ok=True)
        
        self.logger.info("✅ Enhanced Data Recorder initialized")
    
    def start_session(self, session_name=None):
        """Start a new recording session"""
        session_id = f"session_{int(time.time())}"
        
        if session_name:
            session_id = f"{session_name}_{session_id}"
        
        session_data = {
            'session_id': session_id,
            'start_time': time.time(),
            'start_time_iso': datetime.now().isoformat(),
            'end_time': None,
            'total_detections': 0,
            'detection_history': [],
            'statistics': {
                'male_count': 0,
                'female_count': 0,
                'age_distribution': defaultdict(int),
                'emotion_distribution': defaultdict(int),
                'peak_occupancy': 0,
                'total_unique_faces': 0
            },
            'status': 'recording'
        }
        
        self.sessions[session_id] = session_data
        self.current_session = session_id
        
        self.logger.info(f"🎥 Started recording session: {session_id}")
        return session_id
    
    def stop_session(self, session_id):
        """Stop a recording session"""
        if session_id not in self.sessions:
            self.logger.warning(f"Session {session_id} not found")
            return False
        
        self.sessions[session_id]['end_time'] = time.time()
        self.sessions[session_id]['end_time_iso'] = datetime.now().isoformat()
        self.sessions[session_id]['status'] = 'completed'
        
        # Calculate session duration
        duration = self.sessions[session_id]['end_time'] - self.sessions[session_id]['start_time']
        self.sessions[session_id]['duration_seconds'] = duration
        
        # Save session data
        self._save_session_data(session_id)
        
        self.logger.info(f"⏹️ Stopped recording session: {session_id} (Duration: {duration:.2f}s)")
        
        if self.current_session == session_id:
            self.current_session = None
        
        return True
    
    def record_detection(self, detections, session_id=None):
        """Record detection data"""
        if session_id is None:
            session_id = self.current_session
        
        if session_id not in self.sessions:
            self.logger.warning(f"Cannot record: Session {session_id} not found")
            return False
        
        timestamp = time.time()
        detection_entry = {
            'timestamp': timestamp,
            'timestamp_iso': datetime.now().isoformat(),
            'detections': detections,
            'face_count': len(detections)
        }
        
        # Update session data
        self.sessions[session_id]['detection_history'].append(detection_entry)
        self.sessions[session_id]['total_detections'] += len(detections)
        
        # Update statistics
        self._update_statistics(session_id, detections)
        
        return True
    
    def _update_statistics(self, session_id, detections):
        """Update session statistics"""
        stats = self.sessions[session_id]['statistics']
        
        # Update peak occupancy
        stats['peak_occupancy'] = max(stats['peak_occupancy'], len(detections))
        
        for detection in detections:
            # Gender count
            if detection.get('gender') == 'Male':
                stats['male_count'] += 1
            elif detection.get('gender') == 'Female':
                stats['female_count'] += 1
            
            # Age distribution
            age = detection.get('custom_age', 'Unknown')
            stats['age_distribution'][age] += 1
            
            # Emotion distribution
            emotion = detection.get('emotion', 'Neutral')
            stats['emotion_distribution'][emotion] += 1
        
        # Estimate unique faces (simplified - in real system use face recognition)
        stats['total_unique_faces'] = int(stats['peak_occupancy'] * 1.5)
    
    def _save_session_data(self, session_id):
        """Save session data to file"""
        try:
            filename = os.path.join(self.data_dir, 'sessions', f"{session_id}.json")
            
            with open(filename, 'w') as f:
                json.dump(self.sessions[session_id], f, indent=2)
            
            self.logger.info(f"💾 Saved session data: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving session data: {e}")
            return False
    
    def get_session_data(self, session_id):
        """Get detailed session data"""
        if session_id not in self.sessions:
            # Try to load from file
            filename = os.path.join(self.data_dir, 'sessions', f"{session_id}.json")
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.error(f"❌ Error loading session data: {e}")
        
        return self.sessions.get(session_id)
    
    def get_all_sessions(self):
        """Get all recorded sessions"""
        sessions_list = []
        
        # Add active sessions
        for session_id, session_data in self.sessions.items():
            sessions_list.append({
                'session_id': session_id,
                'start_time': session_data['start_time_iso'],
                'end_time': session_data.get('end_time_iso', 'Active'),
                'duration': session_data.get('duration_seconds', 0),
                'total_detections': session_data['total_detections'],
                'status': session_data['status'],
                'peak_occupancy': session_data['statistics']['peak_occupancy']
            })
        
        # Load completed sessions from files
        sessions_dir = os.path.join(self.data_dir, 'sessions')
        if os.path.exists(sessions_dir):
            for filename in os.listdir(sessions_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json
                    if session_id not in self.sessions:
                        try:
                            with open(os.path.join(sessions_dir, filename), 'r') as f:
                                session_data = json.load(f)
                            
                            sessions_list.append({
                                'session_id': session_id,
                                'start_time': session_data['start_time_iso'],
                                'end_time': session_data.get('end_time_iso', 'Unknown'),
                                'duration': session_data.get('duration_seconds', 0),
                                'total_detections': session_data['total_detections'],
                                'status': 'completed',
                                'peak_occupancy': session_data['statistics']['peak_occupancy']
                            })
                        except Exception as e:
                            self.logger.error(f"❌ Error loading session file {filename}: {e}")
        
        # Sort by start time (newest first)
        sessions_list.sort(key=lambda x: x['start_time'], reverse=True)
        
        return sessions_list
    
    def export_session_csv(self, session_id, export_dir=None):
        """Export session data as CSV"""
        if export_dir is None:
            export_dir = os.path.join(self.data_dir, 'exports')
        
        os.makedirs(export_dir, exist_ok=True)
        
        session_data = self.get_session_data(session_id)
        if not session_data:
            return None
        
        filename = os.path.join(export_dir, f"{session_id}_analysis.csv")
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    'Timestamp', 'Face Count', 'Gender', 'Age', 'Age Group', 
                    'Emotion', 'Confidence', 'Bounding Box'
                ])
                
                # Write data
                for detection_entry in session_data['detection_history']:
                    timestamp = detection_entry['timestamp_iso']
                    
                    for detection in detection_entry['detections']:
                        writer.writerow([
                            timestamp,
                            len(detection_entry['detections']),
                            detection.get('gender', 'Unknown'),
                            detection.get('age', 'Unknown'),
                            detection.get('custom_age', 'Unknown'),
                            detection.get('emotion', 'Neutral'),
                            detection.get('face_confidence', 0),
                            str(detection.get('bbox', []))
                        ])
            
            self.logger.info(f"📊 Exported session data to CSV: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ CSV export error: {e}")
            return None
    
    def get_system_statistics(self):
        """Get overall system statistics"""
        total_sessions = len(self.get_all_sessions())
        total_detections = 0
        total_duration = 0
        
        for session in self.get_all_sessions():
            total_detections += session['total_detections']
            total_duration += session.get('duration', 0)
        
        return {
            'total_sessions': total_sessions,
            'total_detections': total_detections,
            'total_duration_hours': total_duration / 3600,
            'average_detections_per_session': total_detections / max(1, total_sessions),
            'data_directory': self.data_dir
        }