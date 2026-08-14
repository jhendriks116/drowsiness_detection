"""Eye Aspect Ratio (EAR): Six eye points -> one openness number.

The EAR is the ratio of an eye's vertical opening to its horizontal width.

Open eye -> lids far apart -> large numerator -> EAR ~ 0.25-0.35
Closed eye -> lids meet -> small numerator -> EAR ~ 0.1 or below

Dividing by the horizontal width makes EAR scale-independent. Near or far from
the camera, the ratio stays consistent, because height and width scale together.
"""
import numpy as np

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    #Euclidean distance between two points
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(eye: np.ndarray) -> float:
    #Compute the EAR for one eye
    if eye.shape != (6, 2):
        raise ValueError(f"Expected a (6, 2) array. Got {eye.shape}")

    left_corner, right_corner, top1, bottom1, top2, bottom2 = eye

    horizontal = _dist(left_corner, right_corner)
    if horizontal == 0.0:
        return 0.0              #Avoid zero division

    vertical1 = _dist(top1, bottom1)
    vertical2 = _dist(top2, bottom2)

    return (vertical1 + vertical2) / (2.0 * horizontal)

def average_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    #Average the EAR of both eyes
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0