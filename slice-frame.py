#!/usr/bin/env python3
"""
Script: slice-frame.py
Description: Extract the last frame from a video file and save it as an image
Usage: python3 slice-frame.py --input video.mp4 --output frame.jpg --format jpg
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def extract_last_frame(video_path: str, output_path: str = None, format: str = "jpg") -> str:
    """
    Extract the last frame from a video file
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the extracted frame (optional)
        format: Output image format ('jpg' or 'png', default: 'jpg')
    
    Returns:
        Path to the saved frame image
    
    Raises:
        ValueError: If video cannot be opened or has no frames
        FileNotFoundError: If input video file doesn't exist
    """
    
    # Validate format
    format = format.lower()
    if format not in ['jpg', 'jpeg', 'png']:
        raise ValueError(f"Unsupported format: {format}. Supported formats: jpg, jpeg, png")
    
    # Normalize jpeg to jpg for consistency
    if format == 'jpeg':
        format = 'jpg'
    
    # Check if input file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output path if not provided
    if output_path is None:
        video_file = Path(video_path)
        output_path = video_file.parent / f"{video_file.stem}_lastframe.{format}"
    
    print(f"📹 Processing video: {video_path}")
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    print(f"📊 Video info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f} seconds")
    
    # Set position to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    # Read the last frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise ValueError(f"Could not read last frame from video: {video_path}")
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    print(f"🖼️  Frame dimensions: {width}x{height}")
    
    # Save the frame
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(output_file), frame)
    
    if not success:
        raise ValueError(f"Failed to save frame to: {output_path}")
    
    print(f"✅ Last frame saved to: {output_path}")
    
    # Display file size
    file_size = output_file.stat().st_size
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    elif file_size > 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size} bytes"
    
    print(f"📁 File size: {size_str}")
    
    return str(output_file)


def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Extract the last frame from a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 slice-frame.py --input video.mp4
  python3 slice-frame.py --input video.mp4 --format png
  python3 slice-frame.py --input video.mp4 --output frame.jpg --format jpg
  python3 slice-frame.py -i /path/to/video.mp4 -o /path/to/frame.png -f png
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to output image file (optional, auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format (default: jpg)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Extract the last frame
        output_path = extract_last_frame(args.input, args.output, args.format)
        
        if args.verbose:
            print(f"\n🎉 Frame extraction completed successfully!")
            print(f"   Input:  {args.input}")
            print(f"   Output: {output_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
        
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
