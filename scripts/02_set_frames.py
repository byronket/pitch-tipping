"""Pick the set frame for every pitch of one or more games.

    python scripts/02_set_frames.py 565039 566683 [--debug]

Per game it writes the set-frame table, one full-resolution JPEG per pitch,
and a contact sheet of every pick (data/set_frames/{game}_grid.png) with
each tile bordered by pitch type. Look at the sheet: a wrong pick is easy to
spot there and invisible in the numbers. --debug also saves the motion
curve behind each pick.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tell.config import SET_FRAMES                                # noqa: E402
from tell.set_frames import CAMERA_REASONS, extract_game          # noqa: E402

TILE_W, TILE_H, BORDER, COLS = 168, 252, 6, 10
PITCH_COLORS = {   # BGR
    "FF": (60, 76, 231), "SI": (0, 140, 255), "FC": (182, 89, 155),
    "CH": (0, 215, 255), "SL": (113, 204, 46), "CU": (235, 166, 0),
}
OTHER_COLOR = (128, 128, 128)


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    n_camera = sum(r["reason"] in CAMERA_REASONS for r in results)
    shown = len(results) - n_camera
    print(f"  extracted {len(ok)}/{len(results)} pitches; "
          f"{len(ok)}/{shown} of those the camera showed at the set")
    # A pitch type that gets dropped more often than the others would bias
    # everything downstream, so the split is printed every time.
    for pt in sorted({r["pitch_type"] for r in results}, key=str):
        rows = [r for r in results if r["pitch_type"] == pt]
        dropped = [r["reason"] for r in rows if r["status"] != "ok"]
        stances = [r["stance"] for r in rows if r["status"] == "ok"]
        print(f"    {pt}: {len(dropped)}/{len(rows)} dropped; "
              + ", ".join(f"{stances.count(s)} {s}"
                          for s in ("stretch", "windup", "unknown")
                          if stances.count(s)))


def contact_sheet(results: list[dict], path: Path) -> None:
    tile_w, tile_h = TILE_W + 2 * BORDER, TILE_H + 2 * BORDER
    n_rows = (len(results) + COLS - 1) // COLS
    legend_h = 40
    sheet = np.full((legend_h + n_rows * tile_h, COLS * tile_w, 3), 20,
                    dtype=np.uint8)
    x = 10
    for pt in sorted({r["pitch_type"] for r in results}, key=str):
        cv2.rectangle(sheet, (x, 10), (x + 26, 32),
                      PITCH_COLORS.get(pt, OTHER_COLOR), -1)
        cv2.putText(sheet, str(pt), (x + 34, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        x += 110
    for i, r in enumerate(results):
        label = f"{i + 1} {r['pitch_type']}"
        if r["tile"] is not None:
            h, w = r["tile"].shape[:2]
            s = min(TILE_W / w, TILE_H / h)
            small = cv2.resize(r["tile"], (int(w * s), int(h * s)))
            tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            y0 = (TILE_H - small.shape[0]) // 2
            x0 = (TILE_W - small.shape[1]) // 2
            tile[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
            if r["stance"] == "windup":
                label += " windup"
        else:
            tile = np.full((TILE_H, TILE_W, 3), 45, dtype=np.uint8)
            cv2.putText(tile, r["reason"][:22], (4, TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        tile = cv2.copyMakeBorder(
            tile, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT,
            value=PITCH_COLORS.get(r["pitch_type"], OTHER_COLOR))
        cv2.putText(tile, label, (BORDER + 4, tile_h - BORDER - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        row, col = divmod(i, COLS)
        y0, x0 = legend_h + row * tile_h, col * tile_w
        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    cv2.imwrite(str(path), sheet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("games", nargs="+", type=int, help="game ids")
    ap.add_argument("--debug", action="store_true",
                    help="save the motion curve behind every pick")
    ap.add_argument("--force-track", action="store_true",
                    help="re-run the pitcher detector on every clip")
    args = ap.parse_args()

    for game_pk in args.games:
        print(f"game {game_pk}")
        debug_dir = None
        if args.debug:
            debug_dir = SET_FRAMES / f"debug_{game_pk}"
            debug_dir.mkdir(parents=True, exist_ok=True)
        results = extract_game(game_pk, args.force_track, debug_dir)
        summarize(results)
        contact_sheet(results, SET_FRAMES / f"{game_pk}_grid.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
