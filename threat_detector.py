# threat_detector.py
import time
from datetime import datetime

class ThreatDetector:
    def __init__(self):
        self.alert_levels = {
            'low': {'color': (0, 255, 255), 'message': 'Low Threat', 'priority': 1},
            'medium': {'color': (0, 165, 255), 'message': 'Medium Threat', 'priority': 2},
            'high': {'color': (0, 0, 255), 'message': 'High Threat', 'priority': 3}
        }
        self.threat_history = []
    
    def analyze_potential_threats(self, detections, context):
        """Analyze detections for potential security threats"""
        threats = []
        
        # Unauthorized access during restricted hours
        if context.get('restricted_hours', False) and detections:
            threats.append({
                'level': 'high',
                'type': 'unauthorized_hours_access',
                'message': 'Access during restricted hours',
                'details': f'{len(detections)} people detected during restricted hours',
                'timestamp': datetime.now().isoformat(),
                'location': context.get('location', 'unknown')
            })
        
        # Crowd density threat
        max_capacity = context.get('max_capacity', 20)
        if len(detections) > max_capacity:
            threats.append({
                'level': 'medium',
                'type': 'overcrowding',
                'message': 'Area over capacity',
                'details': f'{len(detections)} people in area (max: {max_capacity})',
                'timestamp': datetime.now().isoformat(),
                'location': context.get('location', 'unknown')
            })
        
        # Age-based threats (minors in restricted areas)
        min_age = context.get('min_age_restricted', 18)
        for detection in detections:
            age = self._parse_age(detection.get('custom_age', ''))
            if age and age < min_age:
                threats.append({
                    'level': 'high',
                    'type': 'underage_restricted_area',
                    'message': 'Underage person in restricted area',
                    'details': f'Age: {age}, Gender: {detection.get("gender", "Unknown")}',
                    'timestamp': datetime.now().isoformat(),
                    'location': context.get('location', 'unknown')
                })
        
        # Unknown persons threat (if face recognition is available)
        if context.get('require_authorization', False):
            unauthorized_count = sum(1 for d in detections if not d.get('recognized_person'))
            if unauthorized_count > 0:
                threats.append({
                    'level': 'medium',
                    'type': 'unauthorized_persons',
                    'message': 'Unauthorized persons detected',
                    'details': f'{unauthorized_count} unrecognized persons',
                    'timestamp': datetime.now().isoformat(),
                    'location': context.get('location', 'unknown')
                })
        
        # Log threats
        for threat in threats:
            self._log_threat(threat)
        
        return threats
    
    def _parse_age(self, age_str):
        """Parse age string to numeric value"""
        try:
            if '-' in age_str:
                return int(age_str.split('-')[0])
            elif age_str.endswith('+'):
                return int(age_str.replace('+', ''))
            return int(age_str)
        except:
            return None
    
    def _log_threat(self, threat):
        """Log threat to history"""
        threat['id'] = f"threat_{len(self.threat_history) + 1}"
        self.threat_history.append(threat)
        
        # Keep only recent threats (last 1000)
        if len(self.threat_history) > 1000:
            self.threat_history = self.threat_history[-1000:]
    
    def get_current_alert_level(self, threats):
        """Determine overall alert level based on threats"""
        if not threats:
            return 'low'
        
        threat_levels = [self.alert_levels[t['level']]['priority'] for t in threats]
        max_priority = max(threat_levels)
        
        if max_priority >= 3:
            return 'high'
        elif max_priority >= 2:
            return 'medium'
        else:
            return 'low'
    
    def get_recent_threats(self, hours=24):
        """Get threats from the last specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            threat for threat in self.threat_history
            if datetime.fromisoformat(threat['timestamp']).timestamp() > cutoff_time
        ]
    
    def generate_threat_report(self, hours=24):
        """Generate comprehensive threat report"""
        recent_threats = self.get_recent_threats(hours)
        
        threat_types = {}
        threat_locations = {}
        
        for threat in recent_threats:
            # Count by type
            threat_type = threat['type']
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
            
            # Count by location
            location = threat.get('location', 'unknown')
            threat_locations[location] = threat_locations.get(location, 0) + 1
        
        return {
            'period_hours': hours,
            'total_threats': len(recent_threats),
            'threat_types': threat_types,
            'threat_locations': threat_locations,
            'high_severity': sum(1 for t in recent_threats if t['level'] == 'high'),
            'medium_severity': sum(1 for t in recent_threats if t['level'] == 'medium'),
            'low_severity': sum(1 for t in recent_threats if t['level'] == 'low'),
            'recent_threats': recent_threats[-20:]  # Last 20 threats
        }