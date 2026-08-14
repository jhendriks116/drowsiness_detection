"""Runner script.

Ties the six modules together. For each frame:
    video -> landmarkes -> EAR -> detector -> annotate -> write out

Produces an annotated video (eye points, live EAR, AWAKE/DROWSY status) and
prints a summary of the drowsy events found and when they occurred.
"""
import argparse
import logging
import time
from pathlib import Path

import cv2

import config
from src.annotate import annotate
from src.detector import DrowsinessDetector
from src.ear import average_ear
from src.landmarks import LandmarkDetector
from src.video import FrameSource, frame_count_hint

logging.basicConfig(level = logging.INFO, format = "%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def _parse_source(raw: str):
    #A bare integer means a webcam index, otherwise it's a file/folder path
    if raw.isdigit():
        return int(raw)
    else:
        return raw

def run(source, output_path: str | None = None, show: bool = False) -> dict:
    #PRocess 'source', write an annotated video, return a run summary
    src = ""
    if isinstance(source, str):
        src = FrameSource(_parse_source(source))
    else:
        src = source
    landmarks = LandmarkDetector()
    detector = DrowsinessDetector()

    #Set up the output video writer lazily
    writer = None
    if output_path is None:
        config.OUTPUT_DIR.mkdir(exist_ok = True)
        output_path = str(config.OUTPUT_DIR / "annotated.mp4")

    total = frame_count_hint(src.source)
    frame_num = 0
    faces_missed = 0
    drowsy_frame_nums = []          #Frame indices flagged drowsy
    event_start_frames = []         #Frame index where each event began
    was_drowsy = False
    start = time.time()

    for frame in src:
        frame_num += 1
        eyes = landmarks.get_eye_landmarks(frame)

        if eyes is None:
            faces_missed += 1
            #Still write the frame (unannotated) so the output stays in sync
            annotated = frame
        else:
            ear = average_ear(eyes.left, eyes.right)
            result = detector.update(ear)
            annotated = annotate(frame, result, eyes)

            if result.drowsy:
                drowsy_frame_nums.append(frame_num)
                if not was_drowsy:
                    event_start_frames.append(frame_num)
                was_drowsy = True
            else:
                was_drowsy = False

        #Lazily open the writer once we know the frame size
        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, src.fps, (w, h))
        writer.write(annotated)

        if show:
            cv2.imshow("Drowsiness Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):   #Press q to quit live view
                break

        if total and frame_num % 50 == 0:
            logger.info("   %d / %d frames", frame_num, total)

    #Cleanup
    landmarks.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    fps_processed = ""
    if elapsed > 0:
        fps_processed = frame_num / elapsed
    else:
        fps_processed = 0.0

    #Convert event start frames to timestamps (seconds) using source fps
    event_times = []
    for f in event_start_frames:
        event_times.append(round(f / src.fps, 1))

    summary = {
        "frames": frame_num,
        "faces_missed": faces_missed,
        "drowsy_events": detector.drowsy_events,
        "drowsy_frames": len(drowsy_frame_nums),
        "event_times_sec": event_times,
        "output": output_path,
        "process_fps": round(fps_processed, 1)
    }
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description = "Detect drowsiness in a video.")
    parser.add_argument("source", help = "video file, image folder, or webcam index (e.g. 0)")
    parser.add_argument("-o", "--output", default = None, help = "output video path")
    parser.add_argument("--show", action = "store_true", help = "show live window (press q to quit)")
    args = parser.parse_args()

    summary = run(args.source, output_path = args.output, show = args.show)

    print("\n=== Drowsiness Detection Summary ===")
    print(f"Frames Processed:   {summary['frames']}")
    print(f"Faces Missed:       {summary['faces_missed']}")
    print(f"Drowsy Events:      {summary['drowsy_events']}")
    print(f"Frames Flagged:     {summary['drowsy_frames']}")
    if summary["event_times_sec"]:
        times = ", ".join(f"{t}s" for t in summary["event_times_sec"])
        print(f"Events began at:    {times}")
    print(f"Output Video:       {summary['output']}")
    print(f"Processing Speed:   {summary['process_fps']} fps")

if __name__ == "__main__":
    main()