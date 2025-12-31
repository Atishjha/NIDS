"""
NIDS Windows Service
Run as: python nids_service.py install/start/stop/remove
"""

import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import time
import sys
import os
import threading
import logging
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class NIDSService(win32serviceutil.ServiceFramework):
    """NIDS Windows Service"""
    
    _svc_name_ = "NetworkIntrusionDetectionService"
    _svc_display_name_ = "Network Intrusion Detection System"
    _svc_description_ = "Monitors network traffic for intrusion attempts and generates alerts"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_running = False
        self.monitor_thread = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the service"""
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            filename=os.path.join(log_dir, 'nids_service.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.is_running = False
        win32event.SetEvent(self.hWaitStop)
        logging.info("Service stopping...")
        
    def SvcDoRun(self):
        """Main service loop"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        self.is_running = True
        logging.info(f"{self._svc_display_name_} started")
        
        # Start monitoring in a separate thread
        self.monitor_thread = threading.Thread(target=self.run_monitoring)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Main service loop
        while self.is_running:
            time.sleep(1)
            
        logging.info("Service stopped")
        
    def run_monitoring(self):
        """Run the NIDS monitoring"""
        try:
            from nids_complete import RealTimeNIDS
            import numpy as np
            
            logging.info("Initializing NIDS...")
            
            # Initialize NIDS
            nids = RealTimeNIDS()
            
            # Create results directory
            results_dir = os.path.join(os.path.dirname(__file__), 'service_results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Statistics file
            stats_file = os.path.join(results_dir, 'service_stats.json')
            
            # Main monitoring loop
            packet_count = 0
            stats_interval = 60  # Save stats every 60 seconds
            last_stats_save = time.time()
            
            logging.info("NIDS monitoring started")
            
            while self.is_running:
                try:
                    # Simulate packet processing (replace with real packet capture)
                    packet_data = self.generate_simulated_packet()
                    
                    # Predict
                    result = nids.predict_packet(packet_data)
                    
                    packet_count += 1
                    
                    # Log attacks
                    if result.get('is_attack', False):
                        alert_data = {
                            'timestamp': datetime.now().isoformat(),
                            'message': result.get('message', 'Attack detected'),
                            'confidence': result.get('confidence', 0),
                            'packet_count': packet_count
                        }
                        
                        # Log alert
                        logging.warning(f"ALERT: {alert_data['message']} (Confidence: {alert_data['confidence']:.2%})")
                        
                        # Save alert
                        alert_file = os.path.join(results_dir, 'alerts.json')
                        self.append_to_json(alert_file, alert_data)
                    
                    # Save statistics periodically
                    current_time = time.time()
                    if current_time - last_stats_save > stats_interval:
                        self.save_service_stats(nids, stats_file, packet_count)
                        last_stats_save = current_time
                        packet_count = 0
                    
                    # Control processing rate
                    time.sleep(0.1)  # 10 packets per second
                    
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logging.error(f"Failed to initialize NIDS: {e}")
            
    def generate_simulated_packet(self):
        """Generate simulated packet data"""
        import numpy as np
        
        packet = {}
        
        # Simulate different types of traffic
        is_attack = np.random.random() < 0.05  # 5% attack rate
        
        if is_attack:
            # Attack pattern
            packet['src_bytes'] = np.random.randint(10000, 1000000)
            packet['dst_bytes'] = np.random.randint(10000, 1000000)
            packet['duration'] = np.random.uniform(100, 1000)
            packet['count'] = np.random.randint(100, 1000)
        else:
            # Normal pattern
            packet['src_bytes'] = np.random.randint(100, 1000)
            packet['dst_bytes'] = np.random.randint(100, 1000)
            packet['duration'] = np.random.uniform(0.1, 10)
            packet['count'] = np.random.randint(1, 10)
        
        # Add random features
        for i in range(20):
            packet[f'feature_{i}'] = np.random.uniform(0, 1)
        
        return packet
    
    def save_service_stats(self, nids, stats_file, packet_count):
        """Save service statistics"""
        try:
            stats = nids.get_statistics()
            stats['service_uptime'] = time.time() - nids.stats['start_time']
            stats['packets_processed'] = packet_count
            stats['timestamp'] = datetime.now().isoformat()
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
            logging.info(f"Statistics saved: {packet_count} packets processed")
            
        except Exception as e:
            logging.error(f"Error saving stats: {e}")
    
    def append_to_json(self, filename, data):
        """Append data to JSON file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.append(data)
            
            # Keep only last 1000 alerts
            if len(existing_data) > 1000:
                existing_data = existing_data[-1000:]
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error appending to JSON: {e}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(NIDSService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(NIDSService)