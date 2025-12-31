@echo off
echo Installing NIDS Windows Service...

REM Install Python packages
pip install pywin32

REM Install the service
python nids_service.py install

echo Service installed. To start: net start NetworkIntrusionDetectionService
echo To stop: net stop NetworkIntrusionDetectionService
echo To remove: python nids_service.py remove
pause