#!/bin/bash
# =========================================================
# Script: record-video.sh
# Description: records a 5-second high-resolution video.
# Output: Saved to ~/pest-monitoring/PEST-VIDEO-<DATE>.mp4
# =========================================================

# Create output folder if not exists
OUTPUT_DIR="$HOME/pest-monitoring"
mkdir -p "$OUTPUT_DIR"

# Get timestamp for filename
TIMESTAMP=$(date +"%d.%m.%Y-%I.%M%p")
OUTPUT_FILE="$OUTPUT_DIR/PEST-VIDEO-$TIMESTAMP.mp4"

# Camera device
DEVICE=/dev/video0

# Resolution and format
RESOLUTION="1920x1080"
FRAMERATE=30
FORMAT="mjpeg"

echo " ^=^n Capturing 5-second 1080p video..."
ffmpeg -f v4l2 -input_format $FORMAT -video_size $RESOLUTION -framerate $FRAMERATE -t 5 -i $DEVICE "$OUTPUT_FILE" -y
echo " ^|^e Video saved as: $OUTPUT_FILE"
