"""Body keypoints at the set, as a second way of looking at the same frames.

Pixels see everything, including things that are not the pitcher. A pose
model reduces him to seventeen joints, which cannot be fooled by an ad board
but also cannot see a glove turned a few degrees. Here it runs once per set
frame and the elbows and wrists are scored exactly like the pixels are.

The joints are expressed in the pitcher's own frame: origin at the middle of
the hips, one axis running shoulders to hips, lengths in torso units. Camera
position and zoom drop out, which matters because home and road broadcasts
shoot from slightly different angles.
"""

import json
import urllib.request

import cv2
import numpy as np
import polars as pl

from . import tracking
from .config import FRAMES, MODELS, POSE, SET_FRAMES, TRACKS

MODEL_PATH = MODELS / "yolo11n-pose.pt"
MODEL_URL = ("https://github.com/ultralytics/assets/releases/download/"
             "v8.3.0/yolo11n-pose.pt")

# COCO keypoint indices
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
L_KNE, R_KNE = 13, 14
TORSO_AND_ARMS = (L_SHO, R_SHO, L_HIP, R_HIP, L_ELB, R_ELB, L_WRI, R_WRI)
SCORED = (L_ELB, R_ELB, L_WRI, R_WRI)
CONF_MIN = 0.30
MIN_BOX_OVERLAP = 0.25


def load_model():
    from ultralytics import YOLO
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(str(MODEL_PATH))


def extract_game(game_pk: int, model, force: bool = False) -> dict:
    """Keypoints for every set frame of one game, cached to
    data/pose/{game_pk}.json as play_id -> {"kp": 17 x [x, y, confidence]}.

    The pose model finds everyone in the frame. The pitcher is the person
    whose box best covers the box the tracker already has for him; with no
    good match the pitch is stored with kp = None.
    """
    out_path = POSE / f"{game_pk}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    setf = pl.read_csv(SET_FRAMES / f"set_frames_{game_pk}.csv")
    tracks = {p.stem.split("_", 1)[1]: json.loads(p.read_text())
              for p in TRACKS.glob(f"{game_pk}_*.json")}
    med = tracking.game_median_geometry(tracks)

    out = {}
    for r in setf.filter(pl.col("status") == "ok").iter_rows(named=True):
        pid = r["play_id"]
        idxs, boxes = tracking.valid_segment(tracks[pid], med)
        j = int(np.argmin(np.abs(idxs - r["set_frame"])))
        x1, y1, x2, y2 = boxes[j]
        area = max(1.0, (x2 - x1) * (y2 - y1))

        frame = cv2.imread(str(FRAMES / f"{game_pk}_{pid}.jpg"))
        assert frame is not None, f"{game_pk}/{pid}: set-frame JPEG missing"
        pred = model.predict(frame, imgsz=960, conf=0.25, verbose=False)[0]
        best, best_overlap = None, 0.0
        if pred.keypoints is not None and pred.boxes is not None:
            people = pred.keypoints.data.cpu().numpy()
            for kp, (bx1, by1, bx2, by2) in zip(
                    people, pred.boxes.xyxy.cpu().numpy()):
                iw = max(0.0, min(x2, bx2) - max(x1, bx1))
                ih = max(0.0, min(y2, by2) - max(y1, by1))
                if iw * ih / area > best_overlap:
                    best, best_overlap = kp, iw * ih / area
        found = best is not None and best_overlap >= MIN_BOX_OVERLAP
        out[str(pid)] = {"kp": np.round(best, 2).tolist() if found else None}

    POSE.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    n_found = sum(v["kp"] is not None for v in out.values())
    print(f"  pose {game_pk}: pitcher found in {n_found}/{len(out)} frames")
    return out


def load_pose(game_pk: int) -> dict:
    return json.loads((POSE / f"{game_pk}.json").read_text())


def torso_frame(kp) -> np.ndarray | None:
    """All seventeen joints in the pitcher's own frame, or None when the
    shoulders, hips, elbows or wrists are not confidently located.

    Returns (17, 2): x across the body, y from the hips (0) toward the
    shoulders (-1).
    """
    if kp is None:
        return None
    a = np.asarray(kp, dtype=np.float64)
    if any(a[i, 2] < CONF_MIN for i in TORSO_AND_ARMS):
        return None
    mid_hip = (a[L_HIP, :2] + a[R_HIP, :2]) / 2
    mid_sho = (a[L_SHO, :2] + a[R_SHO, :2]) / 2
    down = mid_hip - mid_sho
    scale = float(np.linalg.norm(down))
    if scale < 1e-6:
        return None
    yhat = down / scale
    xhat = np.array([yhat[1], -yhat[0]])
    if xhat[0] < 0:
        xhat = -xhat
    out = np.empty((len(a), 2))
    for i in range(len(a)):
        rel = (a[i, :2] - mid_hip) / scale
        out[i] = float(rel @ xhat), float(rel @ yhat)
    return out


def pose_features(skeleton: np.ndarray) -> np.ndarray:
    """The eight numbers that get scored: x and y of both elbows and both
    wrists. At the set those are the joints that hold the ball and glove."""
    return skeleton[list(SCORED)].reshape(-1)
