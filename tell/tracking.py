"""Follow the pitcher through a clip and keep the crop steady on him.

A broadcast clip cuts between cameras. Everything downstream needs the
stretch of the clip shot from center field with the pitcher at his usual
size and place on screen, so each detection is checked against the game's
typical framing and the longest run that passes is kept.
"""

import json
from pathlib import Path

import numpy as np

from .config import FRAME_H, FRAME_W

STRIDE = 8             # detect on every 8th frame (about 7.5 per second)
TRACK_FRAC = 0.75      # the pitch is over well before the clip ends
MIN_CONF_STORE = 0.25  # detections are cached down to this confidence
MIN_CONF = 0.30        # and used from this confidence up
MAX_GAP = 2            # missed samples tolerated inside one segment

# Framing limits relative to the game median, in fractions of the frame.
H_REL_TOL = (0.60, 1.45)
CX_TOL = 0.20
CY_TOL = 0.12

CROP_WX, CROP_HX = 1.6, 1.30   # motion crop: the median box, expanded
SMOOTH_S = 0.5                 # smoothing window for the crop center

_DEVICE = None


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def track_clip(model, pitcher_cls: int, path: Path) -> dict:
    """Detect the pitcher on every STRIDE-th frame. Returns fps, frame count,
    stride and samples of [frame, x1, y1, x2, y2, confidence]."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
        samples = []
        last = int(n * TRACK_FRAC)
        idx = 0
        while idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % STRIDE == 0:
                res = model.predict(frame, device=_device(), verbose=False)[0]
                best = None
                for box in res.boxes:
                    if (int(box.cls) == pitcher_cls
                            and float(box.conf) >= MIN_CONF_STORE):
                        conf = float(box.conf)
                        if best is None or conf > best[4]:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            best = [x1, y1, x2, y2, conf]
                if best is not None:
                    samples.append([idx] + best)
            idx += 1
        return {"fps": fps, "n_frames": n, "stride": STRIDE, "samples": samples}
    finally:
        cap.release()


def load_or_track(model_ref: list, path: Path, cache: Path,
                  force: bool = False) -> dict:
    """Cached track for one clip. `model_ref` is a one-item list so the
    detector loads once, and only if some clip actually needs tracking."""
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    if model_ref[0] is None:
        from baseballcv.functions import LoadTools
        from ultralytics import YOLO
        model = YOLO(LoadTools().load_model("phc_detector"))
        pitcher_cls = next(k for k, v in model.names.items()
                           if v.lower() == "pitcher")
        model_ref[0] = (model, pitcher_cls)
    model, pitcher_cls = model_ref[0]
    track = track_clip(model, pitcher_cls, path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(track))
    return track


def game_median_geometry(tracks: dict) -> dict:
    """Median pitcher height and position over every detection in the game.
    The center-field shot dominates, so the median describes it."""
    hs, cxs, cys = [], [], []
    for t in tracks.values():
        for _, x1, y1, x2, y2, _ in t["samples"]:
            hs.append((y2 - y1) / FRAME_H)
            cxs.append((x1 + x2) / 2 / FRAME_W)
            cys.append((y1 + y2) / 2 / FRAME_H)
    return {"h": float(np.median(hs)), "cx": float(np.median(cxs)),
            "cy": float(np.median(cys))}


def valid_segment(track: dict, med: dict):
    """Longest run of detections that match the game's usual framing.
    Returns (frame indices, boxes), or None when the clip has no usable run."""
    ok_samples = []
    for s in track["samples"]:
        _, x1, y1, x2, y2, conf = s
        if conf < MIN_CONF:
            continue
        h = (y2 - y1) / FRAME_H
        cx = (x1 + x2) / 2 / FRAME_W
        cy = (y1 + y2) / 2 / FRAME_H
        if not (H_REL_TOL[0] * med["h"] <= h <= H_REL_TOL[1] * med["h"]):
            continue
        if abs(cx - med["cx"]) > CX_TOL or abs(cy - med["cy"]) > CY_TOL:
            continue
        ok_samples.append(s)
    if len(ok_samples) < 2:
        return None
    runs, cur = [], [ok_samples[0]]
    for s in ok_samples[1:]:
        if (s[0] - cur[-1][0]) // track["stride"] <= MAX_GAP + 1:
            cur.append(s)
        else:
            runs.append(cur)
            cur = [s]
    runs.append(cur)
    best = max(runs, key=lambda r: r[-1][0] - r[0][0])
    idxs = np.array([s[0] for s in best], dtype=float)
    boxes = np.array([s[1:5] for s in best], dtype=float)
    return idxs, boxes


def crop_track(idxs, boxes, fps):
    """Crop rectangle for every frame of the segment: the box center,
    interpolated and smoothed, at one fixed size for the whole clip."""
    f0, f1 = int(idxs[0]), int(idxs[-1])
    frames = np.arange(f0, f1 + 1)
    cx = np.interp(frames, idxs, (boxes[:, 0] + boxes[:, 2]) / 2)
    cy = np.interp(frames, idxs, (boxes[:, 1] + boxes[:, 3]) / 2)
    win = max(3, int(SMOOTH_S * fps) | 1)
    kernel = np.ones(win) / win
    pad = win // 2
    cx = np.convolve(np.pad(cx, pad, mode="edge"), kernel, mode="valid")
    cy = np.convolve(np.pad(cy, pad, mode="edge"), kernel, mode="valid")
    w = float(np.median(boxes[:, 2] - boxes[:, 0])) * CROP_WX
    h = float(np.median(boxes[:, 3] - boxes[:, 1])) * CROP_HX
    return frames, cx, cy, w, h


def crop_rect(cx, cy, w, h, W, H):
    x0 = int(round(min(max(cx - w / 2, 0), W - w)))
    y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return x0, y0, int(round(w)), int(round(h))
