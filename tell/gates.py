"""Throw out frames that are not really the set position.

The set-frame picker is right most of the time, and the rest of the time it
hands over a frame from the middle of the delivery. A few of those are
enough to fake a result or bury one. Four checks remove them before any
scoring. None of them looks at pitch type.

Every exclusion is counted by pitch class afterwards (window.py): a check
that removes more fastballs than changeups can manufacture a difference on
its own.
"""

import numpy as np
import polars as pl

from .config import (APPEARANCE_Z, ASPECT_GATE, KNEE_CONF_MIN, REG_H_TOL,
                     SET_FRAMES, SET_KNEE_ASYM_MAX, SET_KNEE_MIN)
from .crops import set_height_rel, subset_crops
from .pose import L_HIP, L_KNE, L_SHO, R_HIP, R_KNE, R_SHO, load_pose


def aspect_gate(game_pk: int, g: dict) -> np.ndarray:
    """A box much wider than usual means a leg or arm is already out."""
    setf = pl.read_csv(SET_FRAMES / f"set_frames_{game_pk}.csv").filter(
        pl.col("status") == "ok")
    aspect = dict(zip(setf["play_id"], setf["aspect"]))
    med = float(np.median(list(aspect.values())))
    return np.array([aspect[p] <= ASPECT_GATE * med for p in g["play_id"]])


def appearance_gate(game_pk: int, g: dict) -> np.ndarray:
    """Frames that look nothing like the game's typical set frame.

    Catches the pitcher who pauses at the top of his leg lift: he is still
    for most of a second and his box is narrow, so the motion and shape
    checks both pass him. Distance is measured on the pitcher's pixels only,
    so a runner or batter drifting through the background cannot trigger it.
    """
    X = g["X"].astype(np.float64)
    med = np.median(X, axis=0)
    mask = g["mask"].mean(axis=0) > 0.5
    d = np.array([np.abs(x - med)[mask].mean() for x in X])
    m = np.median(d)
    mad = np.median(np.abs(d - m)) + 1e-9
    return (d - m) / (1.4826 * mad) <= APPEARANCE_Z


def registration_gate(game_pk: int, g: dict) -> np.ndarray:
    """The pitcher is a different size on screen than in the rest of the
    game's set frames, usually because the broadcast zoomed."""
    h = set_height_rel(game_pk)
    med = float(np.median(list(h.values())))
    rel = np.array([h[str(p)] / med for p in g["play_id"]])
    return np.abs(rel - 1.0) <= REG_H_TOL


def set_phase_gate(game_pk: int, g: dict) -> np.ndarray:
    """At the set both knees are low and level. A knee near hip height, or
    one clearly above the other, is a leg lift. Pitches whose knees the pose
    model is unsure about are kept, not guessed at.

    The thresholds describe a typical set. They do not fit everyone: this
    check rejects nearly every 2019 Tyler Glasnow frame, so a window can be
    scored without it."""
    pose = load_pose(game_pk)
    joints = [L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE]
    keep = []
    for pid in g["play_id"]:
        kp = pose.get(str(pid), {}).get("kp")
        lifted = False
        if kp is not None:
            kp = np.asarray(kp, dtype=np.float64)
            if np.min(kp[joints, 2]) >= KNEE_CONF_MIN:
                shoulder_y = float(kp[[L_SHO, R_SHO], 1].mean())
                hip_y = float(kp[[L_HIP, R_HIP], 1].mean())
                torso = max(abs(hip_y - shoulder_y), 1.0)
                knees = (kp[[L_KNE, R_KNE], 1] - hip_y) / torso
                lifted = (float(knees.min()) < SET_KNEE_MIN
                          or float(abs(knees[0] - knees[1])) > SET_KNEE_ASYM_MAX)
        keep.append(not lifted)
    return np.asarray(keep, dtype=bool)


GATES = (("aspect", aspect_gate), ("appearance", appearance_gate),
         ("registration", registration_gate), ("set_phase", set_phase_gate))


def apply_gates(games: list[int], crops: dict,
                knee_check: bool = True) -> tuple[dict, dict]:
    """Run the checks in order, each on the survivors of the last.
    Returns the surviving crops and how many pitches each check removed."""
    gated = dict(crops)
    removed = {}
    for name, gate in GATES:
        if gate is set_phase_gate and not knee_check:
            continue
        removed[name] = 0
        for pk in games:
            keep = gate(pk, gated[pk])
            removed[name] += int((~keep).sum())
            gated[pk] = subset_crops(gated[pk], keep)
    return gated, removed
