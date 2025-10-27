#!/bin/bash
# =========================================================
# Script: record-and-slice.sh
# Description: Records a 5-second high-resolution video and extracts the last frame
# Output: Video saved to ~/pest-monitoring/PEST-VIDEO-<DATE>.mp4
#         Last frame saved to ~/pest-monitoring/PEST-FRAME-<DATE>.jpg
# =========================================================

# Create output folder if not exists
OUTPUT_DIR="$HOME/pest-monitoring"
mkdir -p "$OUTPUT_DIR"

# Get timestamp for filename
TIMESTAMP=$(date +"%d.%m.%Y-%I.%M%p")
VIDEO_FILE="$OUTPUT_DIR/PEST-VIDEO-$TIMESTAMP.mp4"
FRAME_FILE="$OUTPUT_DIR/PEST-FRAME-$TIMESTAMP.jpg"

# Camera device
DEVICE=/dev/video0

# Resolution and format
RESOLUTION="1920x1080"
FRAMERATE=30
FORMAT="mjpeg"

echo "🎥 Starting video recording and frame extraction process..."
echo "📹 Recording 5-second 1080p video..."

# Record video using ffmpeg
ffmpeg -f v4l2 -input_format $FORMAT -video_size $RESOLUTION -framerate $FRAMERATE -t 5 -i $DEVICE "$VIDEO_FILE" -y

# Check if video recording was successful
if [ $? -eq 0 ] && [ -f "$VIDEO_FILE" ]; then
    echo "✅ Video saved as: $VIDEO_FILE"
    
    echo "🖼️  Extracting last frame from video using Python script..."
    
    # Use the Python slice-frame.py script to extract the last frame
    python3 "$OUTPUT_DIR/slice-frame.py" --input "$VIDEO_FILE" --output "$FRAME_FILE"
    PYTHON_EXIT_CODE=$?
    
    # Check if frame extraction was successful
    if [ $PYTHON_EXIT_CODE -eq 0 ] && [ -f "$FRAME_FILE" ]; then
        echo "✅ Last frame saved as: $FRAME_FILE"
        
        # Display file sizes
        VIDEO_SIZE=$(du -h "$VIDEO_FILE" | cut -f1)
        FRAME_SIZE=$(du -h "$FRAME_FILE" | cut -f1)
        
        echo ""
        echo "📊 Summary:"
        echo "   Video: $VIDEO_FILE ($VIDEO_SIZE)"
        echo "   Frame: $FRAME_FILE ($FRAME_SIZE)"
        echo ""
        echo "🎉 Recording and frame extraction completed successfully!"
    else
        echo "❌ Failed to extract last frame from video (Python script exit code: $PYTHON_EXIT_CODE)"
        exit 1
    fi
else
    echo "❌ Failed to record video"
    exit 1
fi
