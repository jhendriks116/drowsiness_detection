"""Central configuration for the drowsiness detector.

The two values that actually get tuned live here. The EAR threshold (below
which an eye counts as "closed") and how many consecutive closed frames signal
drowsiness rather than a blink. Keeping them in one place means tuning is a 
config edit, not a code edit.
"""
from pathlib import Path

#Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"        #Sample videos/images folder
OUTPUT_DIR = BASE_DIR / "output"    #Annotated result videos

#Detection thresholds. Eye Aspect Ratio (EAR) below this counts as "eye closed".
EAR_THRESHOLD = 0.20

#How many consecutive closed frames before we call it drowsiness.
CONSECUTIVE_FRAMES = 15

#Video. When reading a webcam, which camera index
DEFAULT_WEBCAM_INDEX = 0