#!/bin/bash
# NIDS Complete Setup Script

echo "========================================="
echo "NIDS Complete Setup"
echo "========================================="

# Create directory structure
mkdir -p {data/raw,data/processed,models,results,logs,static,templates}

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Additional dependencies for production
pip install gunicorn gevent  # For production web server
pip install psutil           # For system monitoring
pip install pyshark scapy    # For packet capture (optional)

# Install system dependencies (Ubuntu/Debian)
if [ -f /etc/debian_version ]; then
    sudo apt-get update
    sudo apt-get install -y python3-dev build-essential
    sudo apt-get install -y libpcap-dev tshark  # For packet capture
fi

# Train initial model
echo "Training initial model..."
python nids_complete.py

# Create service files
echo "Creating service files..."
sudo cp nids.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To start NIDS as service:"
echo "  sudo systemctl start nids"
echo ""
echo "To start web dashboard:"
echo "  python dashboard.py"
echo ""
echo "To monitor logs:"
echo "  tail -f /var/log/nids/nids_daemon.log"