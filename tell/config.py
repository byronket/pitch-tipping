"""Paths and tuning constants shared by every stage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIPS = DATA / "clips"            # one broadcast mp4 per pitch
LABELS = DATA / "labels"          # pitch type and count for every pitch
TRACKS = DATA / "tracks"          # cached pitcher detections per clip
SET_FRAMES = DATA / "set_frames"  # which frame of each clip is the set position
FRAMES = DATA / "frames"          # that frame at full resolution
CROPS = DATA / "crops"            # registered crops per game
POSE = DATA / "pose"              # body keypoints per set frame
MODELS = ROOT / "models"

FRAME_W, FRAME_H = 1280, 720      # Baseball Savant clip resolution

# --- set-frame picker -------------------------------------------------------
# Thresholds are fractions of the clip's motion range, capped by absolute
# ceilings in gray levels. A true set hold measures about 1.5-3.5; the pause
# at the top of a leg lift measures about 6. Without the ceilings a violent
# delivery stretches the range until that pause looks still.
MIN_SEGMENT_S = 2.0        # the tracked segment must cover the set and the delivery
THR_STILL = 0.12
THR_MOVE = 0.35
THR_HOLD = 0.25
ABS_STILL = 4.0
ABS_HOLD = 5.0
MIN_CONTRAST = 3.5         # below this the clip never shows a delivery
SET_H_REL = (0.80, 1.30)   # pitcher height at the set vs the game median
MIN_HOLD_S = 0.20
MIN_MOVE_S = 0.12          # a one-frame fidget is not the start of the delivery
SIGMA_FRAMES = 5

# Stretch or windup. With runners on he is in the stretch, so those pitches
# give the reference body shape; bases-empty pitches are compared against it.
MIN_STRETCH_REFS = 5
STANCE_Z = 3.0
MIN_STANCE_SPREAD = 0.01

# --- registered crops -------------------------------------------------------
PATCH_W, PATCH_H = 128, 192

# name: (crop width, crop height, mask pad x, mask pad y, per-pitch scale).
# Sizes are in units of the pitcher's height; pads are fractions of his box.
#   wide   sizes every crop from the game-median height and pads the mask
#          generously, which leaves room for the outfield wall.
#   tight  zooms in on the body and sizes each crop from the pitcher's height
#          in that pitch, so a mid-game camera zoom no longer smears the
#          averages.
CROP_GEOMS = {
    "wide": (1.00, 1.50, 0.30, 0.10, False),
    "tight": (0.62, 1.15, 0.03, 0.03, True),
}
DEFAULT_GEOM = "tight"

# --- frame quality gates ----------------------------------------------------
ASPECT_GATE = 1.35         # box wider than this x the game median: mid-delivery
APPEARANCE_Z = 3.5         # robust z of distance from the game's median frame
REG_H_TOL = 0.08           # on-screen height off the game's set median by this
KNEE_CONF_MIN = 0.5
SET_KNEE_MIN = 0.75        # knees this far below the hips, in torso lengths
SET_KNEE_ASYM_MAX = 0.09   # one knee higher than the other: a leg lift

# --- statistics -------------------------------------------------------------
N_PERM = 1000
N_BOOT = 2000
RNG_SEED = 20260826
