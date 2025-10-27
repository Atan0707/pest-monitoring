import cv2
import numpy as np
import json
import argparse
from typing import List, Tuple, Dict, Any
import torch
from pathlib import Path
from ultralytics import YOLO


class SAHIDetector:
    def __init__(
        self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45
    ):
        """
        Initialize SAHI detector with YOLOv8 model

        Args:
            model_path: Path to YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def create_overlapping_grid(
        self,
        image_height: int,
        image_width: int,
        grid_size: Tuple[int, int] = (3, 3),
        overlap_ratio: float = 0.2,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Create overlapping grid coordinates for SAHI

        Args:
            image_height: Height of the original image
            image_width: Width of the original image
            grid_size: Grid dimensions (rows, cols)
            overlap_ratio: Overlap ratio between adjacent tiles

        Returns:
            List of (x1, y1, x2, y2) coordinates for each tile
        """
        rows, cols = grid_size
        tiles = []

        # Calculate tile dimensions with overlap
        tile_height = image_height // rows
        tile_width = image_width // cols

        overlap_h = int(tile_height * overlap_ratio)
        overlap_w = int(tile_width * overlap_ratio)

        for row in range(rows):
            for col in range(cols):
                # Calculate base coordinates
                x1 = col * tile_width
                y1 = row * tile_height
                x2 = x1 + tile_width
                y2 = y1 + tile_height

                # Add overlap
                if col > 0:  # Not first column
                    x1 -= overlap_w
                if row > 0:  # Not first row
                    y1 -= overlap_h
                if col < cols - 1:  # Not last column
                    x2 += overlap_w
                if row < rows - 1:  # Not last row
                    y2 += overlap_h

                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_width, x2)
                y2 = min(image_height, y2)

                tiles.append((x1, y1, x2, y2))

        return tiles

    def run_inference_on_tile(
        self, tile_image: np.ndarray, tile_coords: Tuple[int, int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 inference on a single tile

        Args:
            tile_image: Image tile as numpy array
            tile_coords: Tile coordinates (x1, y1, x2, y2) in original image

        Returns:
            List of detection dictionaries with global coordinates
        """
        x1, y1, x2, y2 = tile_coords

        # Run inference
        results = self.model(tile_image, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    # Ensure box coordinates are valid
                    if len(box) >= 4:
                        # Convert tile coordinates to global coordinates
                        global_box = [
                            float(box[0]) + x1,  # x1
                            float(box[1]) + y1,  # y1
                            float(box[2]) + x1,  # x2
                            float(box[3]) + y1,  # y2
                        ]

                        # Validate bounding box coordinates
                        if (
                            global_box[0] < global_box[2]
                            and global_box[1] < global_box[3]
                            and global_box[0] >= 0
                            and global_box[1] >= 0
                        ):

                            detection = {
                                "bbox": global_box,
                                "confidence": float(conf),
                                "class_id": int(cls_id),
                                "class_name": self.model.names[cls_id],
                            }
                            detections.append(detection)

        return detections

    def apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections

        Args:
            detections: List of detection dictionaries

        Returns:
            Filtered list of detections after NMS
        """
        if not detections:
            return []

        # Convert to tensors for NMS (ensure float32 type)
        boxes = torch.tensor([det["bbox"] for det in detections], dtype=torch.float32)
        scores = torch.tensor(
            [det["confidence"] for det in detections], dtype=torch.float32
        )

        # Apply NMS using torchvision
        from torchvision.ops import nms

        keep_indices = nms(boxes, scores, self.iou_threshold)

        # Return filtered detections
        return [detections[i] for i in keep_indices.tolist()]

    def draw_detections(
        self, image: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image

        Args:
            image: Original image
            detections: List of detection dictionaries

        Returns:
            Image with drawn bounding boxes
        """
        output_image = image.copy()
        height, width = image.shape[:2]

        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class_name"]

            # if class_name == "unknown":
            #     continue

            if class_name == "planthoppers":
                class_name = "nilaparvata lugens"

            # Convert to integers and ensure coordinates are valid
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1), width))
            y1 = max(0, min(int(y1), height))
            x2 = max(0, min(int(x2), width))
            y2 = max(0, min(int(y2), height))

            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                continue

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Ensure label fits within image bounds
            label_y = max(y1, label_size[1] + 10)
            label_x2 = min(x1 + label_size[0], width)

            cv2.rectangle(
                output_image,
                (x1, label_y - label_size[1] - 10),
                (label_x2, label_y),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                output_image,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return output_image

    def extract_last_frame(self, video_path: str) -> np.ndarray:
        """
        Extract the last frame from a video file

        Args:
            video_path: Path to the video file

        Returns:
            Last frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")

        print(f"Total frames in video: {total_frames}")

        # Set position to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # Read the last frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise ValueError(f"Could not read last frame from video: {video_path}")

        print(f"Extracted last frame (shape: {frame.shape})")
        return frame

    def process_video(
        self,
        video_path: str,
        output_path: str,
        grid_size: Tuple[int, int] = (3, 3),
        overlap_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Process the last frame of a video with SAHI detection

        Args:
            video_path: Path to input video file
            output_path: Path to save output JSON file
            grid_size: Grid dimensions for slicing
            overlap_ratio: Overlap ratio between tiles

        Returns:
            Dictionary containing detection results
        """
        print(f"Processing video: {video_path}")

        # Extract last frame
        image = self.extract_last_frame(video_path)
        height, width = image.shape[:2]
        print(f"Frame dimensions: {width}x{height}")

        # Create overlapping grid
        tiles = self.create_overlapping_grid(height, width, grid_size, overlap_ratio)
        print(f"Created {len(tiles)} overlapping tiles")

        # Run inference on each tile
        all_detections = []
        for i, tile_coords in enumerate(tiles):
            x1, y1, x2, y2 = tile_coords
            tile_image = image[y1:y2, x1:x2]

            print(f"Processing tile {i+1}/{len(tiles)}: ({x1},{y1}) to ({x2},{y2})")

            tile_detections = self.run_inference_on_tile(tile_image, tile_coords)

            if tile_detections:
                print(f"Found {len(tile_detections)} detections in tile {i+1}")

            all_detections.extend(tile_detections)

        print(f"Total detections before NMS: {len(all_detections)}")

        # Apply NMS to remove duplicates
        final_detections = self.apply_nms(all_detections)
        print(f"Final detections after NMS: {len(final_detections)}")

        # Prepare output data
        output_data = {
            "video_path": video_path,
            "frame_size": {"width": width, "height": height},
            "grid_size": grid_size,
            "overlap_ratio": overlap_ratio,
            "total_detections": len(final_detections),
            "detections": [],
        }

        # Format detections
        for detection in final_detections:

            # if detection["class_name"] == "unknown":
            #     continue

            class_name = detection["class_name"]
            if class_name == "planthoppers":
                class_name = "nilaparvata lugens"
            print(f"class name: {class_name}")

            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Skip invalid bounding boxes
            if x1 >= x2 or y1 >= y2:
                continue

            # Normalize coordinates
            x_min = x1 / width
            y_min = y1 / height
            x_max = x2 / width
            y_max = y2 / height

            output_data["detections"].append(
                {
                    "label": class_name,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "probability": detection["confidence"],
                }
            )

        # Save output JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        print(f"Total detections: {len(output_data['detections'])}")

        # Optional: Save annotated image
        if len(final_detections) > 0:
            annotated_image = self.draw_detections(image, final_detections)
            image_output_path = output_file.with_suffix(".jpg")
            cv2.imwrite(str(image_output_path), annotated_image)
            print(f"Annotated image saved to: {image_output_path}")

        return output_data


def main():
    parser = argparse.ArgumentParser(
        description="SAHI YOLOv8 Detection on Last Frame of Video"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input video file (.mp4)"
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLOv8 model (.pt file)"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--grid-rows", type=int, default=3, help="Number of grid rows")
    parser.add_argument(
        "--grid-cols", type=int, default=3, help="Number of grid columns"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.2, help="Overlap ratio between tiles"
    )

    args = parser.parse_args()

    # Initialize detector
    detector = SAHIDetector(
        model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou
    )

    # Process video
    try:
        detector.process_video(
            video_path=args.input,
            output_path=args.output,
            grid_size=(args.grid_rows, args.grid_cols),
            overlap_ratio=args.overlap,
        )

        print(f"\nProcessing complete!")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
