#!/bin/bash
# NOMA AI Startup Script - Uses virtual environment

echo "=== NOMA AI Startup ==="
date

# Navigate to app directory
cd /home/havil/noma_ai || {
    echo "ERROR: Cannot cd to /home/havil/noma_ai"
    exit 1
}

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
    echo "Python: $(python --version)"
    echo "Python path: $(which python)"
else
    echo "ERROR: Virtual environment not found at venv/bin/activate"
    exit 1
fi

# Clean up Qt plugin paths that might conflict
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset QT_PLUGIN_PATH

# Kill any existing Python processes
echo "Cleaning up existing processes..."
pkill -f "python.*noma_app" || true
sleep 1

# Reset GPIO pins safely
echo "Resetting GPIO pins..."
if [ -d /sys/class/gpio ]; then
    for pin in 17 27 22; do
        if [ -d "/sys/class/gpio/gpio${pin}" ]; then
            echo "in" > /sys/class/gpio/gpio${pin}/direction 2>/dev/null || true
            echo "${pin}" > /sys/class/gpio/unexport 2>/dev/null || true
        fi
    done
fi

# Wait for X11/display to be ready
echo "Waiting for display..."
sleep 5

# Set display environment - CRITICAL FOR QT
export DISPLAY=:0
export XAUTHORITY=/home/havil/.Xauthority
export XDG_RUNTIME_DIR=/run/user/$(id -u)

# Configure Qt for kiosk/fullscreen mode - EGLFS for Raspberry Pi
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_HIDECURSOR=1
export QT_QPA_EGLFS_WIDTH=800
export QT_QPA_EGLFS_HEIGHT=480
export QT_QPA_EGLFS_PHYSICAL_WIDTH=154   # 7" screen width in mm
export QT_QPA_EGLFS_PHYSICAL_HEIGHT=86   # 7" screen height in mm
export QT_QPA_EGLFS_FORCE888=1
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1
export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false"

# Disable screen blanking and power management
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

# Run the app
echo "Starting NOMA AI application..."
python noma_app.py

# Log exit
APP_EXIT_CODE=$?
echo "NOMA AI application exited with code: $APP_EXIT_CODE"
date

# Clean up
echo "Cleaning up..."
deactivate 2>/dev/null || true
exit $APP_EXIT_CODE
