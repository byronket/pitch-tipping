"""Line every set frame up on the pitcher.

Each set frame becomes a small grayscale crop centered on the pitcher and
scaled so he is the same size in all of them. Only then does averaging mean
anything: pixel (40, 70) is his glove in every crop, so the average of the
fastball crops minus the average of the off-speed crops shows where he
differs.

Each crop also carries a mask of which pixels are the pitcher. The mask is
what lets the scoring ask whether a difference is on him or on the wall
behind him.
"""

import json

import cv2
import numpy as np
import polars as pl

from . import tracking
from .config import (CROP_GEOMS, CROPS, DEFAULT_GEOM, FRAME_H, FRAMES, LABELS,
                     PATCH_H, PATCH_W, SET_FRAMES, TRACKS)

PER_PITCH = ("X", "mask", "play_id", "pitch_type", "balls", "strikes",
             "source_center", "source_size")


def _ok_set_frames(game_pk: int) -> pl.DataFrame:
    return pl.read_csv(SET_FRAMES / f"set_frames_{game_pk}.csv").filter(
        pl.col("status") == "ok")


def _tracks(game_pk: int, play_ids) -> dict:
    return {pid: json.loads((TRACKS / f"{game_pk}_{pid}.json").read_text())
            for pid in play_ids}


def build_game_crops(game_pk: int, geom: str = DEFAULT_GEOM,
                     force: bool = False) -> dict:
    """Registered crops and pitcher masks for every extracted pitch of one
    game, cached to data/crops/{game_pk}_{geom}.npz.

    X is (n, PATCH_H, PATCH_W) float32 gray levels and mask the matching
    booleans. source_center and source_size record where each crop came from
    in the full frame, so pictures can be re-cut at higher resolution.
    """
    crop_w, crop_h, pad_x, pad_y, per_pitch_scale = CROP_GEOMS[geom]
    cache = CROPS / f"{game_pk}_{geom}.npz"
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    labels = pl.read_csv(LABELS / f"labels_{game_pk}.csv")
    ok = _ok_set_frames(game_pk).join(
        labels.select(["play_id", "balls", "strikes"]), on="play_id")
    tracks = _tracks(game_pk, [p for p in labels["play_id"]
                               if (TRACKS / f"{game_pk}_{p}.json").exists()])
    med = tracking.game_median_geometry(tracks)

    out = {k: [] for k in PER_PITCH}
    for r in ok.iter_rows(named=True):
        pid = r["play_id"]
        track = tracks[pid]
        idxs, boxes = tracking.valid_segment(track, med)
        frames, cx, cy, _, _ = tracking.crop_track(idxs, boxes, track["fps"])
        k = int(np.where(frames == r["set_frame"])[0][0])

        frame = cv2.imread(str(FRAMES / f"{game_pk}_{pid}.jpg"))
        assert frame is not None, f"{game_pk}/{pid}: set-frame JPEG missing"

        if per_pitch_scale:
            # His height in this pitch: the median box height within half a
            # second of the set. He is standing still, so the median removes
            # detector jitter without borrowing another camera framing.
            near = np.abs(idxs - r["set_frame"]) <= 0.5 * track["fps"]
            height = (float(np.median(boxes[near, 3] - boxes[near, 1]))
                      if near.any() else med["h"] * frame.shape[0])
        else:
            height = med["h"] * frame.shape[0]
        w_px, h_px = crop_w * height, crop_h * height
        size = (int(round(w_px)), int(round(h_px)))
        center = (float(cx[k]), float(cy[k]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Subpixel center with replicated borders, so the crop never shifts
        # by a rounding step from one pitch to the next.
        patch = cv2.getRectSubPix(gray, size, center)
        patch = cv2.resize(patch, (PATCH_W, PATCH_H),
                           interpolation=cv2.INTER_AREA).astype(np.float32)

        # Pitcher mask: the detector box nearest the set frame, padded.
        j = int(np.argmin(np.abs(idxs - r["set_frame"])))
        x1, y1, x2, y2 = boxes[j]
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = x1 - pad_x * bw, x2 + pad_x * bw
        y1, y2 = y1 - pad_y * bh, y2 + pad_y * bh
        ox, oy = cx[k] - w_px / 2, cy[k] - h_px / 2
        u1 = int(np.clip((x1 - ox) / w_px * PATCH_W, 0, PATCH_W))
        u2 = int(np.clip((x2 - ox) / w_px * PATCH_W, 0, PATCH_W))
        v1 = int(np.clip((y1 - oy) / h_px * PATCH_H, 0, PATCH_H))
        v2 = int(np.clip((y2 - oy) / h_px * PATCH_H, 0, PATCH_H))
        mask = np.zeros((PATCH_H, PATCH_W), dtype=bool)
        mask[v1:v2, u1:u2] = True

        out["X"].append(patch)
        out["mask"].append(mask)
        out["play_id"].append(pid)
        out["pitch_type"].append(r["pitch_type"])
        out["balls"].append(r["balls"])
        out["strikes"].append(r["strikes"])
        out["source_center"].append(center)
        out["source_size"].append(size)

    out = {
        "X": np.stack(out["X"]),
        "mask": np.stack(out["mask"]),
        "play_id": np.array(out["play_id"]),
        "pitch_type": np.array(out["pitch_type"]),
        "balls": np.array(out["balls"], dtype=np.int64),
        "strikes": np.array(out["strikes"], dtype=np.int64),
        "source_center": np.asarray(out["source_center"], dtype=np.float32),
        "source_size": np.asarray(out["source_size"], dtype=np.int32),
    }
    CROPS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"  crops {game_pk} ({geom}): {len(out['play_id'])} pitches")
    return out


def display_crops(game_pk: int, g: dict, scale: int = 2) -> np.ndarray:
    """The same crops re-cut from the full-resolution frames at `scale`
    times the scoring size. For pictures only; scoring always uses X."""
    target = (PATCH_W * scale, PATCH_H * scale)
    out = []
    for pid, center, size in zip(g["play_id"], g["source_center"],
                                 g["source_size"]):
        frame = cv2.imread(str(FRAMES / f"{game_pk}_{pid}.jpg"),
                           cv2.IMREAD_GRAYSCALE)
        patch = cv2.getRectSubPix(frame, (int(size[0]), int(size[1])),
                                  (float(center[0]), float(center[1])))
        upscaling = patch.shape[1] < target[0] or patch.shape[0] < target[1]
        out.append(cv2.resize(
            patch, target,
            interpolation=cv2.INTER_CUBIC if upscaling else cv2.INTER_AREA,
        ).astype(np.float32))
    return np.stack(out)


def set_height_rel(game_pk: int) -> dict:
    """play_id -> pitcher box height at the set over the game-median height.
    1.0 is his usual size on screen; a broadcast zoom shows up as a cluster
    well above or below it."""
    setf = _ok_set_frames(game_pk)
    tracks = _tracks(game_pk, setf["play_id"])
    med = tracking.game_median_geometry(tracks)
    out = {}
    for pid, sf in zip(setf["play_id"], setf["set_frame"]):
        idxs, boxes = tracking.valid_segment(tracks[pid], med)
        j = int(np.argmin(np.abs(idxs - sf)))
        out[str(pid)] = float((boxes[j][3] - boxes[j][1])
                              / (med["h"] * FRAME_H))
    return out


def subset_crops(g: dict, keep: np.ndarray) -> dict:
    """The crop dict restricted to a boolean mask over its pitches."""
    keep = np.asarray(keep, dtype=bool)
    assert keep.shape == (len(g["play_id"]),), keep.shape
    return {k: g[k][keep] for k in PER_PITCH}
