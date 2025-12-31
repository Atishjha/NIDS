#!/usr/bin/env python3
"""
NIDS Linux Daemon
Usage: python nids_daemon.py start|stop|restart|status
"""

import sys
import os
import time
import signal
import logging
import atexit
from datetime import datetime
import json
import daemon
from daemon import pidfile
import lockfile

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class NIDSDaemon:
    """Linux Daemon for NIDS"""
    
    def __init__(self, pidfile_path='/var/run/nids_daemon.pid'):
        self.pidfile = pidfile_path
        self.is_running = False
        self.monitor_thread = None
        
        # Setup directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['/var/log/nids', '/var/lib/nids', '/tmp/nids']
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
    
    def daemonize(self):
        """Daemonize the process"""
        try:
            # First fork
            pid = os.fork()
            if pid > 0:
                sys.exit(0)  # Exit parent
                
        except OSError as e:
            sys.stderr.write(f"Fork failed: {e}\n")
            sys.exit(1)
        
        # Decouple from parent environment
        os.chdir('/')
        os.setsid()
        os.umask(0)
        
        try:
            # Second fork
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
                
        except OSError as e:
            sys.stderr.write(f"Second fork failed: {e}\n")
            sys.exit(1)
        
        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Write PID file
        with open(self.pidfile, 'w') as f:
            f.write(str(os.getpid()))
        
        # Register cleanup
        atexit.register(self.cleanup)
        
    def cleanup(self):
        """Cleanup before exit"""
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)
    
    def start(self):
        """Start the daemon"""
        if self.is_running:
            print("Daemon is already running")
            return
        
        print("Starting NIDS Daemon...")
        
        # Daemonize
        self.daemonize()
        
        # Setup logging
        self.setup_logging()
        
        # Start monitoring
        self.is_running = True
        self.run_monitoring()
    
    def stop(self):
        """Stop the daemon"""
        if not os.path.exists(self.pidfile):
            print("Daemon is not running")
            return
        
        with open(self.pidfile, 'r') as f:
            pid = int(f.read().strip())
        
        try:
            os.kill(pid, signal.SIGTERM)
            print("Daemon stopped")
        except ProcessLookupError:
            print("Daemon process not found")
            os.remove(self.pidfile)
    
    def restart(self):
        """Restart the daemon"""
        self.stop()
        time.sleep(2)
        self.start()
    
    def status(self):
        """Check daemon status"""
        if os.path.exists(self.pidfile):
            with open(self.pidfile, 'r') as f:
                pid = int(f.read().strip())
            
            try:
                os.kill(pid, 0)  # Check if process exists
                print(f"NIDS Daemon is running (PID: {pid})")
                return True
            except ProcessLookupError:
                print("PID file exists but process is not running")
                os.remove(self.pidfile)
                return False
        else:
            print("NIDS Daemon is not running")
            return False
    
    def setup_logging(self):
        """Setup logging for daemon"""
        log_file = '/var/log/nids/nids_daemon.log'
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Also log to console if in foreground
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        logging.info("NIDS Daemon started")
    
    def run_monitoring(self):
        """Run NIDS monitoring"""
        try:
            from nids_complete import RealTimeNIDS
            import numpy as np
            
            logging.info("Initializing NIDS monitoring...")
            
            # Initialize NIDS
            nids = RealTimeNIDS()
            
            # Statistics
            stats = {
                'start_time': time.time(),
                'total_packets': 0,
                'attack_packets': 0,
                'alerts': []
            }
            
            # Signal handler
            def signal_handler(signum, frame):
                logging.info("Received shutdown signal")
                self.is_running = False
            
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            logging.info("NIDS monitoring started")
            
            while self.is_running:
                try:
                    # Generate simulated packet (replace with real capture)
                    packet_data = self.generate_packet()
                    
                    # Predict
                    result = nids.predict_packet(packet_data)
                    
                    stats['total_packets'] += 1
                    
                    if result.get('is_attack', False):
                        stats['attack_packets'] += 1
                        
                        alert = {
                            'timestamp': datetime.now().isoformat(),
                            'message': result.get('message', 'Attack detected'),
                            'confidence': result.get('confidence', 0),
                            'features': {k: packet_data.get(k, 0) for k in list(packet_data.keys())[:5]}
                        }
                        
                        stats['alerts'].append(alert)
                        logging.warning(f"ALERT: {alert['message']}")
                        
                        # Save alert
                        self.save_alert(alert)
                    
                    # Save statistics every 100 packets
                    if stats['total_packets'] % 100 == 0:
                        self.save_statistics(stats)
                    
                    # Sleep to control rate
                    time.sleep(0.05)  # ~20 packets/second
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    time.sleep(1)
            
            # Save final statistics
            self.save_statistics(stats, final=True)
            logging.info("NIDS monitoring stopped")
            
        except Exception as e:
            logging.error(f"Failed to start monitoring: {e}")
    
    def generate_packet(self):
        """Generate simulated packet"""
        import numpy as np
        
        packet = {}
        
        # 10% chance of attack
        is_attack = np.random.random() < 0.1
        
        if is_attack:
            # Attack characteristics
            packet['src_bytes'] = np.random.randint(50000, 1000000)
            packet['dst_bytes'] = np.random.randint(50000, 1000000)
            packet['duration'] = np.random.uniform(50, 500)
            packet['count'] = np.random.randint(100, 1000)
        else:
            # Normal traffic
            packet['src_bytes'] = np.random.randint(100, 5000)
            packet['dst_bytes'] = np.random.randint(100, 5000)
            packet['duration'] = np.random.uniform(0.1, 10)
            packet['count'] = np.random.randint(1, 20)
        
        # Add additional features
        for i in range(15):
            packet[f'feat_{i}'] = np.random.uniform(0, 1)
        
        return packet
    
    def save_alert(self, alert):
        """Save alert to file"""
        alert_file = '/var/lib/nids/alerts.json'
        
        try:
            alerts = []
            if os.path.exists(alert_file):
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            
            alerts.append(alert)
            
            # Keep only last 1000 alerts
            if len(alerts) > 1000:
                alerts = alerts[-1000:]
            
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving alert: {e}")
    
    def save_statistics(self, stats, final=False):
        """Save statistics to file"""
        stats_file = '/var/lib/nids/statistics.json'
        
        try:
            stats_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - stats['start_time'],
                'total_packets': stats['total_packets'],
                'attack_packets': stats['attack_packets'],
                'normal_packets': stats['total_packets'] - stats['attack_packets'],
                'attack_rate': (stats['attack_packets'] / stats['total_packets'] * 100) 
                              if stats['total_packets'] > 0 else 0,
                'recent_alerts': stats['alerts'][-10:] if stats['alerts'] else []
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            if final:
                logging.info(f"Final statistics: {stats['total_packets']} total packets, "
                           f"{stats['attack_packets']} attacks detected")
            else:
                logging.debug(f"Statistics saved: {stats['total_packets']} packets processed")
                
        except Exception as e:
            logging.error(f"Error saving statistics: {e}")

def main():
    """Main function"""
    daemon = NIDSDaemon()
    
    if len(sys.argv) == 2:
        command = sys.argv[1].lower()
        
        if command == 'start':
            daemon.start()
        elif command == 'stop':
            daemon.stop()
        elif command == 'restart':
            daemon.restart()
        elif command == 'status':
            daemon.status()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python nids_daemon.py start|stop|restart|status")
            sys.exit(1)
    else:
        print("Usage: python nids_daemon.py start|stop|restart|status")
        sys.exit(1)

if __name__ == '__main__':
    main()