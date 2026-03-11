#!/bin/bash
# start_wechat.sh - WeChat auto-reply service with auto-reconnect

cd /opt/wechat_auto

LAST_DEVICE=""

while true; do
    echo "[$(date)] Checking for ADB device..."
    
    # Wait for ADB device
    DEVICE=$(adb devices | grep -w "device" | head -1 | awk '{print $1}')
    
    if [ -n "$DEVICE" ]; then
        # Check if device changed
        if [ "$DEVICE" != "$LAST_DEVICE" ]; then
            echo "[$(date)] Device changed: $LAST_DEVICE -> $DEVICE"
            LAST_DEVICE="$DEVICE"
        fi
        
        echo "[$(date)] Found device: $DEVICE"
        echo "[$(date)] Starting wechat.py..."
        
        # Run wechat.py (it will exit if device disconnects)
        python3 wechat.py
        
        EXIT_CODE=$?
        echo "[$(date)] wechat.py exited with code: $EXIT_CODE"
        
        # Small delay before restart
        sleep 2
    else
        echo "[$(date)] No ADB device found, waiting 5 seconds..."
        LAST_DEVICE=""
        sleep 5
    fi
done
