#!/bin/bash
# Smart Focus Sweep + Sharpness Analysis (Python OpenCV)

DEVICE="/dev/video0"
VIDEO_SIZE="1920x1080"
OUTDIR="$HOME/pest-monitoring"
mkdir -p "$OUTDIR"

DATE=$(date +"%d.%m.%Y-%-I.%M%p")
TEMP_DIR="$OUTDIR/temp"
mkdir -p "$TEMP_DIR"

echo "🎥 Starting pest monitoring capture..."
echo "🔥 Warming up camera for 10 seconds..."
sleep 10

# Disable autofocus and autoexposure
sudo v4l2-ctl -d $DEVICE -c focus_automatic_continuous=0 2>/dev/null
sudo v4l2-ctl -d $DEVICE -c exposure_auto=1 2>/dev/null

# Sweep through multiple focus levels
echo "🔍 Capturing focus sweep..."
for FOCUS in $(seq 100 50 1000); do
    echo "📸 Capturing at focus=$FOCUS"
    sudo v4l2-ctl -d $DEVICE -c focus_absolute=$FOCUS
    sleep 0.5
    ffmpeg -f v4l2 -input_format mjpeg -video_size $VIDEO_SIZE -i $DEVICE -frames:v 1 -y "$TEMP_DIR/focus_${FOCUS}.jpg" -loglevel error
done

# Save one image at focus=350 as baseline
if [ -f "$TEMP_DIR/focus_350.jpg" ]; then
    cp "$TEMP_DIR/focus_350.jpg" "$OUTDIR/PEST-350-${DATE}.jpg"
    echo "✅ Saved baseline focus=350 image."
fi

# Analyze all for sharpness using Python Laplacian variance
python3 - <<'EOF'
import cv2, os, numpy as np, datetime

folder = os.path.expanduser("~/pest-monitoring/temp")
outdir = os.path.expanduser("~/pest-monitoring")

# Get timestamp for naming
timestamp = datetime.datetime.now().strftime("%d.%m.%Y-%-I.%M%p")

images = [f for f in os.listdir(folder) if f.startswith("focus_") and f.endswith(".jpg")]
sharpness = []

for img in images:
    path = os.path.join(folder, img)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None: continue
    sharp_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness.append((sharp_val, img))

if sharpness:
    best_score, best_img = max(sharpness)
    src = os.path.join(folder, best_img)
    dst = os.path.join(outdir, f"PEST-SHARP-{timestamp}.jpg")
    os.rename(src, dst)
    print(f"🏆 Sharpest frame saved as {dst} (sharpness={best_score:.2f})")
else:
    print("❌ No frames found.")
EOF

# Cleanup temporary images
rm -rf "$TEMP_DIR"
echo "🧹 Cleanup done. All results stored in: $OUTDIR"
