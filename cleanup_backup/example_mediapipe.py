
# MEDIAPIPE EXAMPLE
import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        cv2.imshow('MediaPipe Face Detection', frame)
