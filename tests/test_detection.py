"""Tests for the drowsiness detector.

Run:  python -m pytest -q      (from the project root)

Design note: the two pieces the system's correctness rests on — the EAR
geometry and the blink-vs-drowsiness state machine — are pure functions with
no camera, no MediaPipe, no video. So this whole suite runs in well under a
second with no hardware and no model, and it covers the logic that actually
matters. (The landmark detector and video I/O are thin wrappers over libraries;
the judgment lives in ear.py and detector.py.)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ear import average_ear, eye_aspect_ratio      # noqa: E402
from src.detector import DrowsinessDetector             # noqa: E402


# --- helpers: build synthetic eyes of known openness ----------------------
def make_eye(width: float = 40.0, opening: float = 14.0) -> np.ndarray:
    """A 6-point eye of given width and vertical opening.

    Order: [left_corner, right_corner, top1, bottom1, top2, bottom2].
    """
    cx_gap = width / 3.0
    half = opening / 2.0
    return np.array([
        [0.0, 10.0],                 # left corner
        [width, 10.0],               # right corner
        [cx_gap, 10.0 - half],       # top1
        [cx_gap, 10.0 + half],       # bottom1
        [2 * cx_gap, 10.0 - half],   # top2
        [2 * cx_gap, 10.0 + half],   # bottom2
    ], dtype=float)


OPEN_EYE = make_eye(opening=14.0)     # EAR ~0.35
CLOSED_EYE = make_eye(opening=2.0)    # EAR ~0.05


# --- EAR geometry ---------------------------------------------------------
class TestEyeAspectRatio:
    def test_open_eye_has_high_ear(self):
        assert eye_aspect_ratio(OPEN_EYE) > 0.30

    def test_closed_eye_has_low_ear(self):
        assert eye_aspect_ratio(CLOSED_EYE) < 0.10

    def test_open_ear_exceeds_closed_ear(self):
        assert eye_aspect_ratio(OPEN_EYE) > eye_aspect_ratio(CLOSED_EYE)

    def test_ear_is_scale_independent(self):
        """A face near or far from the camera must give the same EAR."""
        small = eye_aspect_ratio(OPEN_EYE)
        big = eye_aspect_ratio(OPEN_EYE * 3.0)      # 3x bigger, same shape
        assert abs(small - big) < 1e-9

    def test_zero_width_returns_zero_not_crash(self):
        degenerate = np.array([[5, 10], [5, 10], [5, 3],
                               [5, 17], [5, 3], [5, 17]], dtype=float)
        assert eye_aspect_ratio(degenerate) == 0.0

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            eye_aspect_ratio(np.zeros((4, 2)))

    def test_average_of_both_eyes(self):
        avg = average_ear(OPEN_EYE, CLOSED_EYE)
        assert (eye_aspect_ratio(CLOSED_EYE) < avg < eye_aspect_ratio(OPEN_EYE))


# --- the blink-vs-drowsiness state machine --------------------------------
class TestDrowsinessDetector:
    def _detector(self):
        return DrowsinessDetector(ear_threshold=0.15, consecutive_frames=15)

    def test_blink_does_not_trigger(self):
        """A short dip (a few closed frames) is a blink, not drowsiness."""
        det = self._detector()
        triggered = any(det.update(e).drowsy
                        for e in [0.26] * 5 + [0.05] * 4 + [0.26] * 5)
        assert not triggered
        assert det.drowsy_events == 0

    def test_sustained_closure_triggers(self):
        det = self._detector()
        for e in [0.26] * 5 + [0.05] * 30:
            det.update(e)
        assert det.drowsy_events == 1

    def test_triggers_at_exact_threshold(self):
        """14 consecutive closed frames = awake; the 15th = drowsy."""
        det = self._detector()
        result = None
        for _ in range(14):
            result = det.update(0.05)
        assert not result.drowsy
        result = det.update(0.05)          # 15th
        assert result.drowsy

    def test_open_frame_resets_the_run(self):
        """A single open frame breaks the run — closures don't accumulate."""
        det = self._detector()
        result = None
        for e in [0.05] * 10 + [0.26] * 1 + [0.05] * 10:
            result = det.update(e)
        assert not result.drowsy           # neither run reached 15
        assert det.drowsy_events == 0

    def test_two_separate_closures_are_two_events(self):
        det = self._detector()
        for e in [0.05] * 20 + [0.26] * 10 + [0.05] * 20:
            det.update(e)
        assert det.drowsy_events == 2

    def test_one_long_closure_is_one_event(self):
        """A sustained closure counts once, not once per frame."""
        det = self._detector()
        for e in [0.05] * 100:
            det.update(e)
        assert det.drowsy_events == 1

    def test_closed_frame_count_reported(self):
        det = self._detector()
        r = None
        for _ in range(5):
            r = det.update(0.05)
        assert r.closed_frames == 5

    def test_reset_clears_state(self):
        det = self._detector()
        for e in [0.05] * 30:
            det.update(e)
        det.reset()
        assert det.drowsy_events == 0
        r = det.update(0.26)
        assert not r.drowsy and r.closed_frames == 0

    def test_threshold_boundary_value(self):
        """EAR exactly at threshold is 'open' (strictly-below counts as closed)."""
        det = self._detector()
        r = det.update(0.15)               # exactly the threshold
        assert not r.eyes_closed           # 0.15 < 0.15 is False
