"""Find the one frame per pitch where the pitcher is set.

Comparing pitches only works if every picture shows the same moment. The
set position is the natural one: he comes to a stop with the ball in his
glove, and whatever he does differently by pitch type is there to see.

Per clip:
  1. Track the pitcher and keep the center-field segment (tracking.py).
  2. Measure motion inside a crop that follows him. The full frame is no
     use for this: ad boards and score graphics move more than he does.
  3. Find the first sustained burst of motion, which is the delivery, and
     walk back to the last still frame before it.

A clip the camera never shows him set in is dropped with a reason, never
guessed at.
"""

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from .config import (ABS_HOLD, ABS_STILL, CLIPS, FRAME_H, FRAMES, LABELS,
                     MIN_CONTRAST, MIN_HOLD_S, MIN_MOVE_S, MIN_SEGMENT_S,
                     MIN_STANCE_SPREAD, MIN_STRETCH_REFS, PATCH_H, PATCH_W,
                     SET_FRAMES, SET_H_REL, SIGMA_FRAMES, STANCE_Z, THR_HOLD,
                     THR_MOVE, THR_STILL, TRACKS)
from .tracking import (crop_rect, crop_track, game_median_geometry,
                       load_or_track, valid_segment)

# Drop reasons that mean the broadcast never showed the set, as opposed to
# the picker failing on footage that did.
CAMERA_REASONS = {"camera_or_track", "set_not_in_segment", "set_pose_geometry"}


def motion_curve(path: Path, frames, cx, cy, w, h) -> np.ndarray:
    """Mean absolute gray-level change between consecutive frames, inside
    the crop that follows the pitcher."""
    cap = cv2.VideoCapture(str(path))
    try:
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
        m = np.full(len(frames), np.nan)
        prev = None
        for i in range(len(frames)):
            ok, frame = cap.read()
            if not ok:
                break
            x0, y0, cw, ch = crop_rect(cx[i], cy[i], w, h, W, H)
            patch = cv2.cvtColor(frame[y0:y0 + ch, x0:x0 + cw],
                                 cv2.COLOR_BGR2GRAY)
            patch = cv2.resize(patch, (PATCH_W, PATCH_H)).astype(np.float32)
            if prev is not None:
                m[i] = float(np.mean(np.abs(patch - prev)))
            prev = patch
        return m
    finally:
        cap.release()


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def pick_set_frame(m: np.ndarray, fps: float):
    """Last still frame before the first sustained move.

    Committing to the FIRST sustained move matters. Searching for the
    quietest moment anywhere in the clip finds the pause at the top of the
    leg lift on some pitchers and calls it the set. If the first move has no
    hold before it, the clip is rejected.

    Returns (pick, None) or (None, reason).
    """
    valid = ~np.isnan(m)
    if valid.sum() < int(1.0 * fps):
        return None, "motion_curve_too_short"
    ms = m.copy()
    ms[~valid] = np.nanmedian(m)
    ms = gaussian_smooth(ms, SIGMA_FRAMES)
    base = float(np.percentile(ms, 5))
    peak = float(np.percentile(ms, 98))
    if peak - base < MIN_CONTRAST:
        return None, "no_delivery_motion"
    motion_range = peak - base
    thr_move = base + THR_MOVE * motion_range
    thr_still = min(thr_move, base + THR_STILL * motion_range, ABS_STILL)
    thr_hold = max(thr_still, min(base + THR_HOLD * motion_range, ABS_HOLD))

    move_n = max(2, int(round(MIN_MOVE_S * fps)))
    t_move = next(
        (t for t in range(1, len(ms) - move_n + 1)
         if np.all(ms[t:t + move_n] >= thr_move)),
        None)
    if t_move is None:
        return None, "no_sustained_delivery_motion"

    t_set = t_move - 1
    while t_set > 0 and ms[t_set] > thr_still:
        t_set -= 1
    if t_set == 0:
        # The camera cut to him after he had already started.
        return None, "set_not_in_segment"

    t0 = t_set
    while t0 > 0 and ms[t0 - 1] < thr_hold:
        t0 -= 1
    hold_s = (t_set - t0) / fps
    if hold_s < MIN_HOLD_S:
        return None, "hold_too_short"

    return {"t_set": t_set, "t_hold_start": t0, "t_move": t_move,
            "hold_s": hold_s, "hold_censored": t0 == 0, "curve": ms,
            "thr_still": thr_still, "thr_hold": thr_hold,
            "thr_move": thr_move}, None


def assign_stances(rows: list[dict]) -> None:
    """Label each extracted pitch stretch, windup or unknown.

    From center field the stretch is side-on and narrow, the windup squarer
    and wider. Pitches with runners on are stretch by rule, so their box
    shape sets the upper edge of "narrow" for this game. Without enough of
    them, bases-empty pitches stay unknown.
    """
    stretch_aspects = np.asarray(
        [r["aspect"] for r in rows if r["runners_on"]], dtype=float)
    stretch_max = None
    if len(stretch_aspects) >= MIN_STRETCH_REFS:
        med = float(np.median(stretch_aspects))
        mad = float(np.median(np.abs(stretch_aspects - med)))
        sigma = max(1.4826 * mad, MIN_STANCE_SPREAD * med)
        stretch_max = med + STANCE_Z * sigma
    for r in rows:
        if r["runners_on"]:
            r["stance"] = "stretch"
        elif stretch_max is None:
            r["stance"] = "unknown"
        elif r["aspect"] <= stretch_max:
            r["stance"] = "stretch"
        else:
            r["stance"] = "windup"


def plot_motion_curve(play_id: str, pick: dict, frames, fps, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = (frames - frames[0]) / fps
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, pick["curve"], lw=1)
    for key, color in (("thr_still", "g"), ("thr_hold", "orange"),
                       ("thr_move", "purple")):
        ax.axhline(pick[key], color=color, ls="--", lw=0.8, label=key[4:])
    for key, color in (("t_hold_start", "g"), ("t_set", "r"),
                       ("t_move", "purple")):
        ax.axvline(t[pick[key]], color=color, lw=1, label=key[2:])
    ax.set_xlabel("seconds into the tracked segment")
    ax.set_ylabel("motion")
    ax.set_title(play_id, fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def extract_game(game_pk: int, force_track: bool = False,
                 debug_dir: Path | None = None) -> list[dict]:
    """Pick the set frame for every labeled pitch of one game.

    Writes data/set_frames/set_frames_{game_pk}.csv and one full-resolution
    JPEG per extracted pitch, so later stages never need the video again.
    Returns the per-pitch records, each with the crop at the set under
    "tile" (None for dropped pitches).
    """
    rows = list(pl.read_csv(LABELS / f"labels_{game_pk}.csv")
                .iter_rows(named=True))

    model_ref = [None]
    tracks = {}
    for i, r in enumerate(rows):
        pid = r["play_id"]
        cache = TRACKS / f"{game_pk}_{pid}.json"
        fresh = force_track or not cache.exists()
        tracks[pid] = load_or_track(model_ref, CLIPS / f"{game_pk}_{pid}.mp4",
                                    cache, force_track)
        if fresh:
            print(f"  tracked {i + 1}/{len(rows)}", flush=True)
    med = game_median_geometry(tracks)

    FRAMES.mkdir(parents=True, exist_ok=True)
    results = []
    for r in rows:
        pid = r["play_id"]
        track = tracks[pid]
        fps = track["fps"]
        rec = {
            "play_id": pid, "pitch_type": r["pitch_type"],
            "inning": r["inning"], "pitch_number": r["pitch_number"],
            "runners_on": any(r[k] is not None
                              for k in ("on_1b", "on_2b", "on_3b")),
            "status": "ok", "reason": "", "set_frame": None,
            "t_set_s": None, "hold_s": None, "hold_censored": None,
            "aspect": None, "stance": None, "tile": None,
        }
        results.append(rec)

        def drop(reason: str) -> None:
            rec["status"], rec["reason"] = "dropout", reason

        seg = valid_segment(track, med)
        if seg is None or (seg[0][-1] - seg[0][0]) / fps < MIN_SEGMENT_S:
            drop("camera_or_track")
            continue
        idxs, boxes = seg
        frames, cx, cy, w, h = crop_track(idxs, boxes, fps)
        clip = CLIPS / f"{game_pk}_{pid}.mp4"
        pick, reason = pick_set_frame(motion_curve(clip, frames, cx, cy, w, h),
                                      fps)
        if pick is None:
            drop(reason)
            continue

        t_set = pick["t_set"]
        set_frame = int(frames[t_set])
        j = int(np.argmin(np.abs(idxs - set_frame)))
        bw = boxes[j, 2] - boxes[j, 0]
        bh = boxes[j, 3] - boxes[j, 1]
        # Standing at the set he is about his usual height on screen. A
        # quiet moment mid-delivery is not.
        h_rel = (bh / FRAME_H) / med["h"]
        if not (SET_H_REL[0] <= h_rel <= SET_H_REL[1]):
            drop("set_pose_geometry")
            continue

        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, set_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            drop("decode_fail")
            continue
        cv2.imwrite(str(FRAMES / f"{game_pk}_{pid}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        H, W = frame.shape[:2]
        x0, y0, cw, ch = crop_rect(cx[t_set], cy[t_set], w, h, W, H)
        rec.update({
            "set_frame": set_frame,
            "t_set_s": round(set_frame / fps, 3),
            "hold_s": round(pick["hold_s"], 3),
            "hold_censored": pick["hold_censored"],
            "aspect": round(float(bw / bh), 4),
            "tile": frame[y0:y0 + ch, x0:x0 + cw].copy(),
        })
        if debug_dir is not None:
            plot_motion_curve(pid, pick, frames, fps,
                              debug_dir / f"{pid}_curve.png")

    assign_stances([r for r in results if r["status"] == "ok"])

    SET_FRAMES.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{k: v for k, v in r.items() if k != "tile"} for r in results]
    ).write_csv(SET_FRAMES / f"set_frames_{game_pk}.csv")
    return results
