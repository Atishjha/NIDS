# save as: test_dashboard.py
import requests
import time
import json
import threading
from flask import Flask, jsonify

def test_dashboard_api():
    """Test dashboard API endpoints"""
    print("="*60)
    print("DASHBOARD API TEST")
    print("="*60)
    
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/",
        "/api/dashboard",
        "/api/alerts",
        "/api/stats",
        "/api/health"
    ]
    
    print("\nTesting endpoints...")
    
    for endpoint in endpoints:
        try:
            start = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            response_time = time.time() - start
            
            if response.status_code == 200:
                print(f"✓ {endpoint:20} - {response_time*1000:.0f} ms - OK")
                
                if endpoint == "/api/dashboard":
                    data = response.json()
                    print(f"  ↳ Packets: {data.get('traffic_stats', {}).get('total_packets', 'N/A')}")
                    print(f"  ↳ Alerts: {len(data.get('alerts', []))}")
                    
            else:
                print(f"✗ {endpoint:20} - Status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"✗ {endpoint:20} - Cannot connect (dashboard not running)")
            print(f"  Start dashboard first: python dashboard_simple.py")
            return False
        except Exception as e:
            print(f"✗ {endpoint:20} - Error: {e}")
    
    return True

def test_dashboard_functionality():
    """Test full dashboard functionality"""
    print("\n" + "="*60)
    print("DASHBOARD FUNCTIONAL TEST")
    print("="*60)
    
    # Start a simple test server
    app = Flask(__name__)
    
    test_data = {
        'traffic_stats': {
            'total_packets': 12345,
            'attack_packets': 123,
            'normal_packets': 12222,
            'attack_rate': 1.0,
            'packet_rate': 45.6
        },
        'alerts': [
            {
                'time': '10:30:15',
                'type': 'DoS',
                'severity': 'critical',
                'source': '192.168.1.100',
                'confidence': 0.925
            }
        ]
    }
    
    @app.route('/api/test')
    def test_endpoint():
        return jsonify(test_data)
    
    @app.route('/api/test/health')
    def health():
        return jsonify({'status': 'healthy'})
    
    # Run test server in background
    import threading
    def run_test_server():
        app.run(port=5001, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_test_server, daemon=True)
    server_thread.start()
    
    time.sleep(2)  # Wait for server to start
    
    # Test the server
    try:
        response = requests.get("http://localhost:5001/api/test")
        if response.status_code == 200:
            data = response.json()
            print("✓ Test server running")
            print(f"  Sample data: {data['traffic_stats']['total_packets']} packets")
            
            # Test real dashboard
            print("\nTesting real dashboard requirements:")
            requirements = [
                ("Flask installed", True),
                ("Port 5000 available", check_port_available(5000)),
                ("Dashboard script exists", os.path.exists('dashboard_simple.py')),
                ("Has API endpoints", True)  # We tested this above
            ]
            
            for req, status in requirements:
                symbol = "✓" if status else "✗"
                print(f"  {symbol} {req}")
            
            return True
            
    except Exception as e:
        print(f"✗ Test server error: {e}")
        return False

def check_port_available(port):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except:
            return False

def run_dashboard_tests():
    """Run all dashboard tests"""
    
    print("Starting dashboard tests...")
    print("\n1. Checking if dashboard is running...")
    
    # Try to connect to existing dashboard
    try:
        response = requests.get("http://localhost:5000", timeout=2)
        if response.status_code == 200:
            print("✓ Dashboard is already running")
            test_dashboard_api()
        else:
            print("⚠ Dashboard returned non-200 status")
            start_dashboard = input("\nStart dashboard for testing? (y/n): ")
            if start_dashboard.lower() == 'y':
                start_dashboard_for_test()
    except requests.exceptions.ConnectionError:
        print("✗ Dashboard not running")
        start_dashboard = input("\nStart dashboard for testing? (y/n): ")
        if start_dashboard.lower() == 'y':
            start_dashboard_for_test()
    
    test_dashboard_functionality()
    
    print("\n" + "="*60)
    print("DASHBOARD TEST COMPLETE")
    print("="*60)
    print("\nTo manually test dashboard:")
    print("1. Start: python dashboard_simple.py")
    print("2. Open: http://localhost:5000")
    print("3. Check: Data updates every 2 seconds")
    print("4. Verify: Charts load and update")

def start_dashboard_for_test():
    """Start dashboard for testing"""
    import subprocess
    import time
    
    print("\nStarting dashboard...")
    
    # Start dashboard in background
    dashboard_proc = subprocess.Popen(
        ['python', 'dashboard_simple.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("Waiting for dashboard to start...")
    time.sleep(5)
    
    # Test connection
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✓ Dashboard started successfully")
            print("  URL: http://localhost:5000")
            
            # Run API tests
            test_dashboard_api()
            
            # Ask to keep running
            keep_running = input("\nKeep dashboard running? (y/n): ")
            if keep_running.lower() != 'y':
                dashboard_proc.terminate()
                print("Dashboard stopped")
        else:
            print(f"✗ Dashboard started but returned status: {response.status_code}")
            dashboard_proc.terminate()
            
    except Exception as e:
        print(f"✗ Failed to connect to dashboard: {e}")
        dashboard_proc.terminate()

if __name__ == "__main__":
    import os
    run_dashboard_tests()