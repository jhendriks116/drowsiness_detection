"""Frame source.

Yields frames one at a time from wherever they come from. The detection
pipeline shouldn't care whether frames come from a video file, a folder
of images, or a live webcam. This module hides that behind a single
interface. Iterate a FrameSource and you get the frames, one per loop,
until they run out. Swap the source, nothing downstream changes.
"""
import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import config

logging.basicConfig(level = logging.INFO, format = "%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#Image extensions we treat as frames when reading a folder.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class FrameSource:
    #Yields frames from a video file, an image folder, or a webcam
    def __init__(self, source):
        self.source = source
        self._kind = self._classify(source)
        self.fps = 30.0
        logger.info("Frame Source: %s (%s)", source, self._kind)

    @staticmethod
    def _classify(source) -> str:
        #Decide whether source is a webcam, a video file, or an image folder.
        if isinstance(source, int):
            return "webcam"
        path = Path(source)
        if path.is_dir():
            return "image_folder"
        if path.is_file():
            return "video_file"
        raise FileNotFoundError(f"Source not Found: {source}. Pass a Video File, a Folder of Images, or an int Webcam Index (e.g. 0).")

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._kind == "image_folder":
            yield from self._iter_images()
        else:
            yield from self._iter_capture()

    def _iter_images(self) -> Iterator[np.ndarray]:
        #Yield images from a folder, in sorted filename order.
        paths = []
        for p in Path(self.source).iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"No Images ({', '.join(sorted(IMAGE_EXTS))}) in {self.source}")
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                logger.warning("Could not Read Image: %s", p)
                continue
            yield frame

    def _iter_capture(self) -> Iterator[np.ndarray]:
        #Yield frames from a video file or webcam via cv2.VideoCapture
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not Open Source: {self.source}. (For a webcam, is the index right and the camera free?)")

        #A video file reports its real fps, a webcam often reports 0
        reported = cap.get(cv2.CAP_PROP_FPS)
        if reported and reported > 0:
            self.fps = reported

        try:
            while True:
                ok, frame = cap.read()
                if not ok:              #End of file, or webcam dropped
                    break
                yield frame
        finally:
            cap.release()               #Always free the device/file handle

def frame_count_hint(source) -> int | None:
    #Best effort total frame count, or None if it can't be known cheaply
    if isinstance(source, int):
        return None
    path = Path(source)
    if path.is_dir():
        count = 0
        for p in path.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                count += 1
        return count
    cap = cv2.VideoCapture(str(source))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            return n
        else:
            return None
    finally:
        cap.release()