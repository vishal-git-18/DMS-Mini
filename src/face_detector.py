import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

def detect_face(image, size=48):
    """
    Detect the largest face in the image and return a cropped size x size image.
    If no face detected, returns a black image.
    """
    results = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, _ = image.shape
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
        x, y = max(0,x), max(0,y)
        cropped = image[y:y+bh, x:x+bw]
        return cv2.resize(cropped, (size, size))
    else:
        return np.zeros((size, size, 3), dtype=np.uint8)
