"""Put one pitcher, a few consecutive starts and two pitch types on the table.

A tell is not a season-long habit. It shows up for two or three starts and
then someone notices and it is gone, so the unit of analysis is a window of
a few starts.

`assemble` produces, for each game in the window, the pixel matrices and the
pose matrix for exactly the same pitches, so the two can be compared.
Order matters: quality checks first, then the stance, then the pitch types.
"""

import numpy as np
import polars as pl
from scipy.stats import fisher_exact

from .config import DEFAULT_GEOM, SET_FRAMES
from .crops import build_game_crops, subset_crops
from .gates import apply_gates
from .pose import load_pose, pose_features, torso_frame
from .scoring import CHANNELS, Contrast, game_matrix

STANCES = ("stretch", "windup")


def _class_balance(removed: dict, kept: dict) -> dict:
    """Did something remove one pitch class more often than the other?"""
    table = [[removed["pos"], kept["pos"]], [removed["neg"], kept["neg"]]]
    return {"pos_removed": removed["pos"], "pos_kept": kept["pos"],
            "neg_removed": removed["neg"], "neg_kept": kept["neg"],
            "fisher_p": float(fisher_exact(table).pvalue)}


def assemble(games: list[int], contrast: Contrast, stance: str,
             geom: str = DEFAULT_GEOM, knee_check: bool = True) -> dict:
    """Everything the scoring and the pictures need for one window.

    Stretch and windup are different body positions and are never mixed:
    averaging the two would show the difference between stances, and pitch
    mix differs with runners on.

    Returns
      X         channel -> game -> matrix; channels are full, body, bg, pose
      y         game -> 1 for `contrast.pos`, 0 for `contrast.neg`
      strata    game -> ball-strike count code, for the shuffles
      crops     game -> the scored pitches' crops, for pictures
      skeleton  game -> (n, 17, 2) joints in the pitcher's frame
      gates     pitches each quality check removed, and the class balance
      pose_dropout  class balance of pitches lost to low-confidence joints
    """
    assert stance in STANCES, stance
    raw = {pk: build_game_crops(pk, geom) for pk in games}
    gated, removed = apply_gates(games, raw, knee_check)

    def cls(pitch_type):
        return ("pos" if pitch_type == contrast.pos
                else "neg" if pitch_type in contrast.neg else None)

    gate_removed, gate_kept = {"pos": 0, "neg": 0}, {"pos": 0, "neg": 0}
    for pk in games:
        setf = pl.read_csv(SET_FRAMES / f"set_frames_{pk}.csv").filter(
            pl.col("status") == "ok")
        stance_of = dict(zip(setf["play_id"].cast(pl.Utf8), setf["stance"]))
        survivors = {str(p) for p in gated[pk]["play_id"]}
        for pid, pt in zip(raw[pk]["play_id"], raw[pk]["pitch_type"]):
            if stance_of.get(str(pid)) == stance and cls(pt):
                side = gate_kept if str(pid) in survivors else gate_removed
                side[cls(pt)] += 1
        gated[pk] = subset_crops(gated[pk], np.array(
            [stance_of.get(str(p)) == stance for p in gated[pk]["play_id"]],
            dtype=bool))

    out = {"games": games, "contrast": contrast, "stance": stance,
           "geom": geom, "X": {ch: {} for ch in CHANNELS + ("pose",)},
           "y": {}, "strata": {}, "crops": {}, "skeleton": {}}
    pose_lost, pose_kept = {"pos": 0, "neg": 0}, {"pos": 0, "neg": 0}
    for pk in games:
        g = gated[pk]
        pose = load_pose(pk)
        sel = np.isin(g["pitch_type"], contrast.classes)
        skeletons = [torso_frame(pose.get(str(p), {}).get("kp"))
                     for p in g["play_id"][sel]]
        keep = np.array([s is not None for s in skeletons], dtype=bool)
        for pt, k in zip(g["pitch_type"][sel], keep):
            (pose_kept if k else pose_lost)[cls(pt)] += 1

        # Pixel channels fill the masked region from the average of ALL of
        # the game's surviving pitches, so build them before narrowing to
        # the pitches that also have a pose.
        for ch in CHANNELS:
            X, y, strata = game_matrix(g, contrast, ch)
            out["X"][ch][pk] = X[keep]
        out["y"][pk], out["strata"][pk] = y[keep], strata[keep]
        out["skeleton"][pk] = np.array([s for s in skeletons if s is not None])
        out["X"]["pose"][pk] = np.array(
            [pose_features(s) for s in out["skeleton"][pk]])
        scored = sel.copy()
        scored[np.where(sel)[0][~keep]] = False
        out["crops"][pk] = subset_crops(g, scored)

    out["gates"] = {"removed": removed,
                    "class_balance": _class_balance(gate_removed, gate_kept)}
    out["pose_dropout"] = _class_balance(pose_lost, pose_kept)
    return out


def pitcher_mask(window: dict) -> np.ndarray:
    """Pixels that are pitcher in most of the window's crops."""
    return np.concatenate([window["crops"][pk]["mask"]
                           for pk in window["games"]]).mean(axis=0) > 0.5
