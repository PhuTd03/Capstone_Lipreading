import cv2
import numpy as np
from collections import deque
import mediapipe as mp

def initialize_face_detector(min_detection_confidence=0.6):
    return mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)

def process_frame(face_detection, frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    return results

def extract_bbox(detection, frame_shape):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame_shape
    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    x, y, w = bbox[:3]
    h = bbox[3] if len(bbox) == 4 else 0
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y, w, h

def smooth_coordinates(coords, smooth_factor=5):
    smoothed_coords = []
    for coord in zip(*coords):
        smoothed_coords.append(np.convolve(coord, np.ones(smooth_factor) / smooth_factor, mode='valid'))
    return zip(*smoothed_coords)