"""Facial landmarks.

We run MediaPipe Face Mesh on each frame. It returns 468 landmark points mapped
onto the detected face, as normalised coordinates (x, y each in 0..1). Normalised
coordinates make the downstream EAR math resolution independent.

Of the 468 points, a fixed, known set rings each eye. We extract the six points
per eye that the Eye Aspect Ratio needs: the two corners pluse two points on the
upper lid and two on the lower lid.

Not every frame has a detectable face (turned away, motion blur, poor light).
MediaPipe returns nothing for those, and get_eye_landmarks returns None and the
caller skips the frame rather than crashing.
"""
import logging
from dataclasses import dataclass

import mediapipe as mp
import numpy as np

logging.basicConfig(level = logging.INFO, format = "%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#MediaPipe Face Mesh landmark indices for the six EAR points of each eye
LEFT_EYE = [33, 133, 160, 144, 158, 153]
RIGHT_EYE = [362, 263, 385, 380, 387, 373]

@dataclass
class EyeLandmarks:
    #The six (x, y) points for each eye, in pixel coordinates
    left: np.ndarray
    right: np.ndarray

class LandmarkDetector:
    #Wraps MediaPipe Face Mesh
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode = False,      #Video mode: track across frames
            max_num_faces = 1,              #One driver/face
            refine_landmarks = True,        #Sharper eye/iris points
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )

    def get_eye_landmarks(self, frame: np.ndarray) -> EyeLandmarks | None:
        #Return eye landmarks for the face in 'frame', or None if no face
        h, w = frame.shape[:2]

        #OpenCV frames are BGR; MediaPipe wants RGB
        rgb = frame[:, :, ::-1]
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None         #No face detected in this frame

        face = results.multi_face_landmarks[0]
        pts = face.landmark     #468 normalised landmarks

        def to_pixels(indices):
            out = np.empty((len(indices), 2), dtype = np.float64)
            for row, idx in enumerate(indices):
                lm = pts[idx]
                out[row, 0] = lm.x * w      #Normalised -> pixel x
                out[row, 1] = lm.y * h      #Normalised -> pixel y
            return out

        return EyeLandmarks(left = to_pixels(LEFT_EYE), right = to_pixels(RIGHT_EYE))

    def close(self):
        #Release MediaPipe resources
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()