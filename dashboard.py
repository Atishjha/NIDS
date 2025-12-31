"""
NIDS Web Dashboard
Run: python dashboard.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, send_file
import json
import os
import time
from datetime import datetime, timedelta
import threading
import numpy as np
from collections import deque
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables for real-time updates
alerts_history = deque(maxlen=1000)
traffic_stats = {
    'total_packets': 0,
    'attack_packets': 0,
    'normal_packets': 0,
    'packet_rate': 0,
    'start_time': time.time()
}
system_status = {
    'status': 'running',
    'model_accuracy': 0.9546,
    'last_update': datetime.now().isoformat()
}

def create_directories():
    """Create necessary directories"""
    dirs = ['static', 'templates', 'data', 'static/css', 'static/js', 'static/images']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

def generate_sample_data():
    """Generate sample data for dashboard"""
    # Load existing data or generate sample
    data_file = 'data/dashboard_data.json'
    
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    
    # Generate sample data
    sample_data = {
        'hourly_traffic': [],
        'attack_types': [],
        'top_alerts': [],
        'performance_metrics': {}
    }
    
    # Generate hourly traffic for last 24 hours
    now = datetime.now()
    for i in range(24):
        hour = (now - timedelta(hours=i)).strftime('%H:00')
        total = np.random.randint(1000, 10000)
        attacks = np.random.randint(10, 500)
        
        sample_data['hourly_traffic'].append({
            'hour': hour,
            'total': total,
            'attacks': attacks,
            'normal': total - attacks
        })
    
    # Generate attack type distribution
    attack_types = ['DoS', 'Probe', 'R2L', 'U2R', 'Malware', 'DDoS', 'Brute Force']
    for attack_type in attack_types:
        sample_data['attack_types'].append({
            'type': attack_type,
            'count': np.random.randint(10, 1000),
            'percentage': np.random.uniform(1, 30)
        })
    
    # Generate top alerts
    for i in range(10):
        sample_data['top_alerts'].append({
            'id': i + 1,
            'time': (now - timedelta(minutes=np.random.randint(1, 60))).strftime('%H:%M:%S'),
            'type': np.random.choice(attack_types),
            'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical']),
            'source_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'destination_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'confidence': np.random.uniform(0.5, 0.99)
        })
    
    # Performance metrics
    sample_data['performance_metrics'] = {
        'accuracy': 0.9546,
        'precision': 0.98,
        'recall': 0.94,
        'f1_score': 0.96,
        'false_positive_rate': 0.02,
        'processing_speed': 45.2
    }
    
    # Save sample data
    with open(data_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    return sample_data

def background_monitoring():
    """Background thread for monitoring updates"""
    while True:
        try:
            # Update traffic stats
            traffic_stats['packet_rate'] = traffic_stats['total_packets'] / (
                time.time() - traffic_stats['start_time']
            ) if time.time() > traffic_stats['start_time'] else 0
            
            # Simulate new packet
            is_attack = np.random.random() < 0.05  # 5% attack rate
            
            traffic_stats['total_packets'] += 1
            
            if is_attack:
                traffic_stats['attack_packets'] += 1
                
                # Create alert
                alert = {
                    'id': len(alerts_history) + 1,
                    'timestamp': datetime.now().isoformat(),
                    'type': np.random.choice(['DoS', 'Probe', 'R2L', 'DDoS']),
                    'severity': np.random.choice(['Medium', 'High', 'Critical']),
                    'source': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    'destination': f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    'confidence': np.round(np.random.uniform(0.6, 0.95), 2),
                    'description': f"{np.random.choice(['SYN flood', 'Port scan', 'Brute force', 'Malware'])} detected"
                }
                
                alerts_history.append(alert)
            
            traffic_stats['normal_packets'] = traffic_stats['total_packets'] - traffic_stats['attack_packets']
            system_status['last_update'] = datetime.now().isoformat()
            
            # Save to file periodically
            if traffic_stats['total_packets'] % 100 == 0:
                save_dashboard_data()
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            print(f"Background monitoring error: {e}")
            time.sleep(5)

def save_dashboard_data():
    """Save dashboard data to file"""
    data = {
        'traffic_stats': dict(traffic_stats),
        'system_status': dict(system_status),
        'recent_alerts': list(alerts_history)[-100:],  # Last 100 alerts
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/current_status.json', 'w') as f:
        json.dump(data, f, indent=2)

# HTML Templates
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIDS Dashboard - Network Intrusion Detection System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
        }
        
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .stat-card i {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .alert-critical {
            border-left: 5px solid var(--danger-color);
        }
        
        .alert-high {
            border-left: 5px solid var(--warning-color);
        }
        
        .alert-medium {
            border-left: 5px solid var(--secondary-color);
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .last-update {
            font-size: 0.8rem;
            color: #6c757d;
            text-align: right;
        }
        
        .navbar-brand {
            font-weight: bold;
            font-size: 1.5rem;
        }
        
        .navbar-brand i {
            margin-right: 10px;
        }
        
        .attack-badge {
            font-size: 0.7rem;
            padding: 3px 8px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark dashboard-header">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-shield-alt"></i> NIDS Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <span class="nav-item nav-link">
                    <i class="fas fa-circle text-success"></i> System Online
                </span>
                <span class="nav-item nav-link last-update" id="lastUpdate">
                    Last update: Just now
                </span>
            </div>
        </div>
    </nav>

    <div class="container">
        <!-- System Stats -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-network-wired text-primary"></i>
                    <div class="stat-value" id="totalPackets">0</div>
                    <div class="stat-label">Total Packets</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-exclamation-triangle text-warning"></i>
                    <div class="stat-value" id="attackPackets">0</div>
                    <div class="stat-label">Attack Packets</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-tachometer-alt text-success"></i>
                    <div class="stat-value" id="attackRate">0%</div>
                    <div class="stat-label">Attack Rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-bolt text-info"></i>
                    <div class="stat-value" id="packetRate">0</div>
                    <div class="stat-label">Packets/sec</div>
                </div>
            </div>
        </div>

        <!-- Charts -->
        <div class="row mb-4">
            <div class="col-md-8">
                <div class="chart-container">
                    <h5><i class="fas fa-chart-line"></i> Traffic Overview (Last 24 Hours)</h5>
                    <div id="trafficChart" style="height: 300px;"></div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="chart-container">
                    <h5><i class="fas fa-chart-pie"></i> Attack Type Distribution</h5>
                    <div id="attackTypeChart" style="height: 300px;"></div>
                </div>
            </div>
        </div>

        <!-- Recent Alerts -->
        <div class="row">
            <div class="col-md-12">
                <div class="chart-container">
                    <h5><i class="fas fa-bell"></i> Recent Alerts</h5>
                    <div class="table-responsive">
                        <table class="table table-hover" id="alertsTable">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Type</th>
                                    <th>Severity</th>
                                    <th>Source</th>
                                    <th>Destination</th>
                                    <th>Confidence</th>
                                    <th>Description</th>
                                </tr>
                            </thead>
                            <tbody id="alertsBody">
                                <!-- Alerts will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Performance -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="chart-container">
                    <h5><i class="fas fa-chart-bar"></i> Model Performance Metrics</h5>
                    <div class="row">
                        <div class="col-md-2 text-center">
                            <div class="stat-card">
                                <div class="stat-value text-success" id="accuracy">95.46%</div>
                                <div class="stat-label">Accuracy</div>
                            </div>
                        </div>
                        <div class="col-md-2 text-center">
                            <div class="stat-card">
                                <div class="stat-value text-primary" id="precision">98.0%</div>
                                <div class="stat-label">Precision</div>
                            </div>
                        </div>
                        <div class="col-md-2 text-center">
                            <div class="stat-card">
                                <div class="stat-value text-info" id="recall">94.0%</div>
                                <div class="stat-label">Recall</div>
                            </div>
                        </div>
                        <div class="col-md-2 text-center">
                            <div class="stat-card">
                                <div class="stat-value text-warning" id="f1Score">96.0%</div>
                                <div class="stat-label">F1 Score</div>
                            </div>
                        </div>
                        <div class="col-md-2 text-center">
                            <div class="stat-card">
                                <div class="stat-value text-danger" id="falsePositive">2.0%</div>
                                <div class="stat-label">False Positive</div>
                            </div>
                        </div>
                        <div class="col-md-2 text-center">
                            <div class="stat-card">
                                <div class="stat-value text-secondary" id="processingSpeed">45.2</div>
                                <div class="stat-label">Packets/sec</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="mt-5 py-3 text-center text-muted">
        <div class="container">
            <p>Network Intrusion Detection System Dashboard | Real-time Monitoring</p>
            <p class="mb-0">Last model update: <span id="modelUpdate">2024-01-01</span></p>
        </div>
    </footer>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initial data load
        $(document).ready(function() {
            updateDashboard();
            setInterval(updateDashboard, 2000); // Update every 2 seconds
        });
        
        function updateDashboard() {
            $.getJSON('/api/dashboard', function(data) {
                // Update stats
                $('#totalPackets').text(data.traffic_stats.total_packets.toLocaleString());
                $('#attackPackets').text(data.traffic_stats.attack_packets.toLocaleString());
                
                let attackRate = data.traffic_stats.attack_rate || 0;
                $('#attackRate').text(attackRate.toFixed(2) + '%');
                
                $('#packetRate').text(data.traffic_stats.packet_rate.toFixed(1));
                $('#lastUpdate').text('Last update: ' + formatTime(new Date()));
                
                // Update alerts table
                updateAlertsTable(data.recent_alerts);
                
                // Update charts if data available
                if (data.hourly_traffic) {
                    updateTrafficChart(data.hourly_traffic);
                }
                if (data.attack_types) {
                    updateAttackTypeChart(data.attack_types);
                }
                
                // Update performance metrics
                if (data.performance_metrics) {
                    $('#accuracy').text((data.performance_metrics.accuracy * 100).toFixed(2) + '%');
                    $('#precision').text((data.performance_metrics.precision * 100).toFixed(1) + '%');
                    $('#recall').text((data.performance_metrics.recall * 100).toFixed(1) + '%');
                    $('#f1Score').text((data.performance_metrics.f1_score * 100).toFixed(1) + '%');
                    $('#falsePositive').text((data.performance_metrics.false_positive_rate * 100).toFixed(1) + '%');
                    $('#processingSpeed').text(data.performance_metrics.processing_speed.toFixed(1));
                }
            });
        }
        
        function updateAlertsTable(alerts) {
            let tbody = $('#alertsBody');
            tbody.empty();
            
            alerts.slice(0, 10).forEach(alert => {
                let severityClass = 'alert-' + alert.severity.toLowerCase();
                let row = `
                    <tr class="${severityClass}">
                        <td>${formatTime(new Date(alert.timestamp))}</td>
                        <td><span class="badge bg-secondary">${alert.type}</span></td>
                        <td><span class="badge ${getSeverityBadge(alert.severity)}">${alert.severity}</span></td>
                        <td><code>${alert.source}</code></td>
                        <td><code>${alert.destination}</code></td>
                        <td><span class="badge ${getConfidenceBadge(alert.confidence)}">${(alert.confidence * 100).toFixed(1)}%</span></td>
                        <td>${alert.description}</td>
                    </tr>
                `;
                tbody.append(row);
            });
        }
        
        function getSeverityBadge(severity) {
            switch(severity.toLowerCase()) {
                case 'critical': return 'bg-danger';
                case 'high': return 'bg-warning';
                case 'medium': return 'bg-primary';
                case 'low': return 'bg-secondary';
                default: return 'bg-secondary';
            }
        }
        
        function getConfidenceBadge(confidence) {
            if (confidence >= 0.9) return 'bg-success';
            if (confidence >= 0.7) return 'bg-warning';
            return 'bg-danger';
        }
        
        function formatTime(date) {
            return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
        }
        
        function updateTrafficChart(data) {
            let hours = data.map(d => d.hour).reverse();
            let totals = data.map(d => d.total).reverse();
            let attacks = data.map(d => d.attacks).reverse();
            
            let trace1 = {
                x: hours,
                y: totals,
                name: 'Total Traffic',
                type: 'scatter',
                line: {color: '#3498db'}
            };
            
            let trace2 = {
                x: hours,
                y: attacks,
                name: 'Attack Traffic',
                type: 'scatter',
                line: {color: '#e74c3c'}
            };
            
            let layout = {
                showlegend: true,
                legend: {x: 0, y: 1},
                margin: {l: 40, r: 20, t: 20, b: 40},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('trafficChart', [trace1, trace2], layout);
        }
        
        function updateAttackTypeChart(data) {
            let types = data.map(d => d.type);
            let counts = data.map(d => d.count);
            let colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71', '#1abc9c', '#34495e'];
            
            let trace = {
                labels: types,
                values: counts,
                type: 'pie',
                marker: {colors: colors},
                textinfo: 'label+percent',
                insidetextorientation: 'radial'
            };
            
            let layout = {
                showlegend: false,
                margin: {l: 20, r: 20, t: 20, b: 20},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('attackTypeChart', [trace], layout);
        }
    </script>
</body>
</html>
'''

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard')
def get_dashboard_data():
    """API endpoint for dashboard data"""
    data = {
        'traffic_stats': dict(traffic_stats),
        'system_status': dict(system_status),
        'recent_alerts': list(alerts_history)[-10:],  # Last 10 alerts
        'hourly_traffic': sample_data['hourly_traffic'][-24:],  # Last 24 hours
        'attack_types': sample_data['attack_types'],
        'performance_metrics': sample_data['performance_metrics'],
        'timestamp': datetime.now().isoformat()
    }
    
    # Calculate attack rate
    if traffic_stats['total_packets'] > 0:
        data['traffic_stats']['attack_rate'] = (
            traffic_stats['attack_packets'] / traffic_stats['total_packets'] * 100
        )
    else:
        data['traffic_stats']['attack_rate'] = 0
    
    return jsonify(data)

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    alerts = list(alerts_history)[-50:]  # Last 50 alerts
    return jsonify(alerts)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    return jsonify({
        'traffic_stats': dict(traffic_stats),
        'system_status': dict(system_status),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """API endpoint to retrain model"""
    # This would trigger model retraining
    return jsonify({
        'status': 'success',
        'message': 'Model retraining initiated',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/export/alerts')
def export_alerts():
    """Export alerts as CSV"""
    alerts = list(alerts_history)
    df = pd.DataFrame(alerts)
    csv = df.to_csv(index=False)
    
    # Save to file
    filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    
    return send_file(filepath, as_attachment=True)

@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    alerts_history.clear()
    return jsonify({'status': 'success', 'message': 'Alerts cleared'})

if __name__ == '__main__':
    # Create directories
    create_directories()
    
    # Create HTML template
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Start background monitoring thread
    monitor_thread = threading.Thread(target=background_monitoring, daemon=True)
    monitor_thread.start()
    
    print("="*60)
    print("NIDS Web Dashboard")
    print("="*60)
    print(f"Access dashboard at: http://localhost:5000")
    print(f"API endpoints:")
    print(f"  - /api/dashboard   - Dashboard data")
    print(f"  - /api/alerts      - Recent alerts")
    print(f"  - /api/stats       - System statistics")
    print("="*60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)