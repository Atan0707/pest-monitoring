#!/bin/bash
# =========================================================
# Script: capture_highres_video.sh
# Description: Warm up camera for 10 seconds, then record
#              a 5-second high-resolution video.
# Output: Saved to ~/pest-monitoring/PEST-VIDEO-<DATE>.mp4
# =========================================================

# Create output folder if not exists
OUTPUT_DIR=~/pest-monitoring
mkdir -p "$OUTPUT_DIR"

# Get timestamp for filename
TIMESTAMP=$(date +"%d.%m.%Y-%I.%M%p")
#OUTPUT_FILE="$OUTPUT_DIR/PEST-VIDEO-$TIMESTAMP.mp4"
OUTPUT_FILE="OUTPUT_DIR/recording"

# Camera device
DEVICE=/dev/video0

# Resolution and format
RESOLUTION="4656x3496"
FRAMERATE=10
FORMAT="mjpeg"

echo "📸 Warming up camera for 10 seconds..."
ffmpeg -f v4l2 -input_format $FORMAT -video_size $RESOLUTION -framerate $FRAMERATE -t 10 -i $DEVICE -f null - > /dev/null 2>&1

echo "🎥 Capturing 5-second high-resolution video..."
ffmpeg -f v4l2 -input_format $FORMAT -video_size $RESOLUTION -framerate $FRAMERATE -t 5 -i $DEVICE "$OUTPUT_FILE" -y

echo "✅ Video saved as: $OUTPUT_FILE"
