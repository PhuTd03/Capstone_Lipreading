# Preprocess video, used to detect face and save the video with all the frame that \
# contain only one face only

import cv2
import mediapipe as mp
import argparse
import configparser

def parse_args():
    parser = argparse.ArgumentParser(description='Process some videos.')
    parser.add_argument('--input_path', type=str, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, help='Path to the output video file')
    parser.add_argument('--config', type=str, help='Path to the config file', default=None)
    args = parser.parse_args()

    if args.config:
        config = configparser.ConfigParser()
        config.read(args.config)
        if 'video' in config:
            if not args.input_path:
                args.input_path = config['video'].get('input_path')
            if not args.output_path:
                args.output_path = config['video'].get('output_path')

    return args

def main():
    args = parse_args()

    if not args.input_path or not args.output_path:
        raise ValueError('Both input_path and output_path must be specified.')

    # initialize face detection model
    detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)

    # read frame
    cap = cv2.VideoCapture(args.input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # RGB to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # face detecting
        results = detector.process(frame)
        faces = results.detections
        
        if len(faces) == 1:
            if out is None:
                # write video
                frame_height, frame_width = frame.shape[:2]
                out = cv2.VideoWriter(args.output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))
            out.write(frame)

    cap.release()
    if out:
        out.release()

    print('Preprocessed video saved at', args.output_path)

if __name__ == "__main__":
    main()
