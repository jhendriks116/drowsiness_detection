"""Annotation: Draw the detection onto a frame so a human can see it work.

This is what turns the pipeline's numbers into a visible demo. Eye points, the
live EAR value, and an AWAKE/DROWSY status drawn onto each frame.

OpenCV drawing notes:
    - All draw calls modify the frame array in place.
    - Colours are BGR, not RGB.
    - Coordinates are (x, y) integers in pixel space.
"""
import cv2
import numpy as np

from src.detector import FrameResult
from src.landmarks import EyeLandmarks

#BGR colours
GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 220)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_eye_points(frame: np.ndarray, eyes: EyeLandmarks, color = YELLOW, radius: int = 2) -> None:
    #Draw as small dot on each of the six points per eye
    for eye in (eyes.left, eyes.right):
        for (x, y) in eye:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)

def annotate(frame: np.ndarray, result: FrameResult, eyes: EyeLandmarks | None = None) -> np.ndarray:
    #Draw landmarks, EAR value, and AWAKE/DROWSY status onto the frame.
    if eyes is not None:
        draw_eye_points(frame, eyes)

    #EAR resad-out
    cv2.putText(frame, f"EAR: {result.ear:.3f}", (15,30), FONT, 0.7, WHITE, 2, cv2.LINE_AA)

    #Closed-frame counter, so you can watch the run build toward the threshold
    cv2.putText(frame, f"Closed: {result.closed_frames}", (15, 58), FONT, 0.6, WHITE, 2, cv2.LINE_AA)

    #Status: green AWAKE, or red DROWSY alert with a banner
    if result.drowsy:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), RED, 8)
        cv2.putText(frame, "DROWSY!", (15, 95), FONT, 1.1, RED, 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "AWAKE", (15, 95), FONT, 0.9, GREEN, 2, cv2.LINE_AA)

    return frame