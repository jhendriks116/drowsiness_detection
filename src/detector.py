"""Drowsiness logic.

Every frame gives us one EAR value. A value below the threshold means the eyes
are closed right now, but that alone can't distinguish a blink from falling
asleep, because both look identical in a single frame.

The distinction is duration. We count consecutive frames below the threshold.
So the rule is: closed for >= CONSECUTIVE_FRAMES in a row -> DROWSY.
A single frame back above the threshold resets the counter.

This is a tiny state machine. It holds no frames and no images, just a running
count, which makes it pure, fast, and easy to unit test.
"""
from dataclasses import dataclass, field

import config

@dataclass
class FrameResult:
    #What the detector reports for a single frame.
    ear: float
    eyes_closed: bool       #Was EAR below the threshold this frame?
    drowsy: bool            #Has the closed-run reached the alarm length?
    closed_frames: int      #Current consecutive-closed count

@dataclass
class DrowsinessDetector:
    #Stateful detector: feed it one EAR per frame, it tracks the closed-run
    ear_threshold: float = config.EAR_THRESHOLD
    consecutive_frames: int = config.CONSECUTIVE_FRAMES

    _closed_run: int = field(default = 0, init = False)         #Consecutive closed frames
    _drowsy_events: int = field(default = 0, init = False)      #Distint drowsy episodes
    _was_drowsy: bool = field(default = False, init = False)    #Were we drowsy last frame?

    def update(self, ear: float) -> FrameResult:
        #Process one frame's EAR value and return the current state
        eyes_closed = ear < self.ear_threshold

        if eyes_closed:
            self._closed_run += 1
        else:
            self._closed_run = 0

        drowsy = self._closed_run >= self.consecutive_frames

        #Count a new drowsy event only on the transition into drowsiness
        if drowsy and not self._was_drowsy:
            self._drowsy_events += 1
        self._was_drowsy = drowsy

        return FrameResult(
            ear = ear,
            eyes_closed = eyes_closed,
            drowsy = drowsy,
            closed_frames = self._closed_run
        )

    @property
    def drowsy_events(self) -> int:
        #How many distinct drowsy episodes have occured so far
        return self._drowsy_events

    def reset(self) -> None:
        #Clear all state
        self._closed_run = 0
        self._drowsy_events = 0
        self._was_drowsy = False