import cv2
import mediapipe as mp
from argparse import ArgumentParser as parse
import numpy as np

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def extract_mouth_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    mouth_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi màu BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lấy tọa độ của vùng miệng
                mouth_points = [face_landmarks.landmark[i] for i in range(48, 68)]  # Các điểm mốc 48-67 cho miệng

                # Chuyển đổi tọa độ từ tỷ lệ sang pixel
                h, w, _ = frame.shape
                mouth_points = [(int(p.x * w), int(p.y * h)) for p in mouth_points]

                # Tạo vùng giới hạn cho miệng
                x_min = min([p[0] for p in mouth_points])
                x_max = max([p[0] for p in mouth_points])
                y_min = min([p[1] for p in mouth_points])
                y_max = max([p[1] for p in mouth_points])

                roi = frame[y_min:y_max, x_min:x_max]
                roi = cv2.resize(roi, (88, 88))  # Kích thước chuẩn
                mouth_frames.append(roi)

    cap.release()
    mouth_frames = np.array(mouth_frames)
    return mouth_frames

def save_mouth_roi_as_npz(video_path, output_path):
    mouth_frames = extract_mouth_frames(video_path)
    np.savez_compressed(output_path, data=mouth_frames)

def main():
    args = parse("Extract mouth ROIs from videos")
    args.add_argument("--video_path", type=str, help="Path to the video file")
    args.add_argument("--output_path", type=str, help="Path to save the mouth ROIs")
    args = args.parse_args()

    save_mouth_roi_as_npz(args.video_path, args.output_path)

if __name__ == "__main__":
    main()