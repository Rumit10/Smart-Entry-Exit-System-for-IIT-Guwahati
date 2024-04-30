import argparse
import torch
import cv2
from pathlib import Path

def detect_objects(weights, img_size, conf_threshold, source, frame_delay):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set inference size
    imgsz = img_size

    # Open video file
    cap = cv2.VideoCapture(source)

    # Initialize flag to indicate if object has been detected
    object_detected = False
    detection_frame_number = -1

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame, size=imgsz)

        # Process detection results
        for result in results.xyxy[0]:
            label = int(result[5])
            confidence = float(result[4])

            # Check if object is detected with confidence above threshold
            if confidence >= conf_threshold:
                object_detected = True
                detection_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                break

        # Stop processing if object is detected
        if object_detected:
            break

    # Release video capture object
    cap.release()

    if object_detected:
        # Calculate the frame number to capture after the detection frame
        capture_frame_number = detection_frame_number + frame_delay

        # Re-open video file to capture the desired frame
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, capture_frame_number)

        # Read and capture the frame
        ret, captured_frame = cap.read()
        if ret:
            # Save the captured frame as PNG with higher quality
            output_path = Path(source).parent / f"captured_frame_.png"
            # Specify higher quality (0-9), default is 3
            cv2.imwrite(str(output_path), captured_frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"Captured frame {capture_frame_number} after object detection. Saved as {output_path}")
        else:
            print(f"Failed to capture frame {capture_frame_number} after object detection.")
        
        # Release video capture object
        cap.release()
    else:
        print("Object not detected in the video.")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Object Detection in Video')
    parser.add_argument('--weights', type=str, default='last.pt', help='path to YOLOv5 model weights file (default: last.pt)')
    parser.add_argument('--img', type=int, default=640, help='inference size (height and width) of the input images (default: 640)')
    parser.add_argument('--conf', type=float, default=0.70, help='confidence threshold for detections (default: 0.70)')
    parser.add_argument('--source', type=str, default='IMG_1813.mp4', help='path to the input video file (default: IMG_1813.mp4)')
    parser.add_argument('--frame-delay', type=int, default=50, help='number of frames to capture after object detection (default: 50)')
    args = parser.parse_args()

    # Perform object detection and frame capture
    detect_objects(args.weights, args.img, args.conf, args.source, args.frame_delay)
