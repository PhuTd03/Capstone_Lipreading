import argparse
import pandas as pd
import cv2
from collections import deque
from utils import *

def track_faces(video_path, result_csv_path, result_video_path, change_threshold=65, smooth_factor=3, min_duration=3, min_detection_confidence=0.6):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(result_video_path, fourcc, frame_rate, (frame_width, frame_height))

    tracking_data = []
    current_id = 1
    start_time = None
    last_bbox = None
    colors = {}
    bbox_history = deque(maxlen=smooth_factor)

    face_detector = initialize_face_detector(min_detection_confidence)

    with face_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / frame_rate
            results = process_frame(face_detector, frame)

            if results.detections:
                if len(results.detections) == 1:
                    detection = results.detections[0]
                    center_x, center_y, width, height = extract_bbox(detection, frame.shape)

                    bbox_history.append((center_x, center_y, width, height))
                    if len(bbox_history) >= smooth_factor:
                        smoothed_coords = smooth_coordinates(bbox_history, smooth_factor)
                        center_x, center_y, width, height = next(smoothed_coords)

                    if start_time is None:
                        start_time = current_time

                    if last_bbox is not None:
                        last_center_x, last_center_y, last_width, last_height = last_bbox
                        if abs(center_x - last_center_x) > change_threshold or abs(center_y - last_center_y) > change_threshold:
                            current_id += 1
                            start_time = current_time

                    last_bbox = (center_x, center_y, width, height)

                    if current_id not in colors:
                        colors[current_id] = (int(current_id * 50 % 256), int(current_id * 80 % 256), int(current_id * 110 % 256))

                    color = colors[current_id]

                    cv2.rectangle(frame, (int(center_x - width / 2), int(center_y - height / 2)),
                                  (int(center_x + width / 2), int(center_y + (height+40) / 2)), color, 2)
                    cv2.putText(frame, f'ID: {current_id}', (int(center_x - width / 2), int(center_y - height / 2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, f'Frame: {frame_count}', (int(center_x - width / 2), int(center_y - height / 2) - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    tracking_data.append({
                        'ID': current_id,
                        'frame': frame_count,
                        'start_time': start_time,
                        'end_time': current_time,
                        'bbox': [center_x, center_y, width, height]
                    })
                else:
                    if len(tracking_data) > 0 and tracking_data[-1]['ID'] == current_id:
                        current_id += 1
                        start_time = None
                        last_bbox = None
                        bbox_history.clear()
            else:
                if len(tracking_data) > 0 and tracking_data[-1]['ID'] == current_id:
                    current_id += 1
                    start_time = None
                    last_bbox = None
                    bbox_history.clear()

            out.write(frame)

    cap.release()
    out.release()

    # Filter out IDs with a duration less than min_duration
    filtered_data = []
    for data in tracking_data:
        duration = data['end_time'] - data['start_time']
        if duration >= min_duration:
            filtered_data.append(data)

    df = pd.DataFrame(filtered_data)
    df.to_csv(result_csv_path, index=False)

    print(f"Tracking data saved to {result_csv_path}")
    print(f"Output video saved to {result_video_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Track faces in a video.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file') # must input
    parser.add_argument('--result_csv_path', type=str, required=True, help='Path to the output CSV file') # must input
    parser.add_argument('--result_video_path', type=str, required=True, help='Path to the output video file') # must input
    parser.add_argument('--change_threshold', type=int, default=65, help='Threshold for detecting significant changes in face position') # default = 65
    parser.add_argument('--smooth_factor', type=int, default=3, help='Number of frames for smoothing face coordinates') # default = 3
    parser.add_argument('--min_duration', type=int, default=3, help='Minimum duration (in seconds) to consider a face tracking valid') # default = 3
    parser.add_argument('--min_detection_confidence', type=float, default=0.6, help='Minimum confidence for face detection') # default = 0.6
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    track_faces(
        video_path=args.video_path,
        result_csv_path=args.result_csv_path,
        result_video_path=args.result_video_path,
        change_threshold=args.change_threshold,
        smooth_factor=args.smooth_factor,
        min_duration=args.min_duration,
        min_detection_confidence=args.min_detection_confidence
    )
