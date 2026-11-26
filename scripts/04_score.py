"""Score one window: a pitcher, a few starts, two pitch types.

    python scripts/04_score.py 825045 825039 825038 \
        --pos FF --neg CH --stance windup --name rodriguez

Prints the held-out score for four views of the same pitches and writes
the numbers, the pictures and a one-page write-up to reports/{name}/.

    pitcher only       his pixels, background blanked    <- the result
    background only    everything but him                <- should be a coin flip
    whole crop         both
    elbows and wrists  pose keypoints

If the background scores as well as the pitcher, the difference is an ad
board or a shadow that happens to track pitch mix, and the window proves
nothing.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tell.config import CROP_GEOMS, DEFAULT_GEOM, N_PERM, ROOT   # noqa: E402
from tell.report import TooFewPitches, score_window, summary     # noqa: E402
from tell.scoring import Contrast                                # noqa: E402
from tell.window import STANCES                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("games", nargs="+", type=int,
                    help="game ids of consecutive starts (at least two)")
    ap.add_argument("--pos", required=True, help="pitch code, e.g. FF")
    ap.add_argument("--neg", required=True,
                    help="pitch code(s) to compare against; comma-separate "
                         "codes that are one pitch (FS,FO)")
    ap.add_argument("--stance", required=True, choices=STANCES,
                    help="stretch and windup are scored separately")
    ap.add_argument("--name", required=True, help="output folder name")
    ap.add_argument("--heading", help="title of the write-up")
    ap.add_argument("--titles", help="comma-separated picture titles, one "
                                     "per game (default: the game ids)")
    ap.add_argument("--geom", default=DEFAULT_GEOM, choices=tuple(CROP_GEOMS))
    ap.add_argument("--no-knee-check", action="store_true",
                    help="skip the knee-position check, for a pitcher whose "
                         "set stance it wrongly rejects")
    ap.add_argument("--n-perm", type=int, default=N_PERM,
                    help="label shuffles behind each p-value")
    ap.add_argument("--circle", help="x,y in crop pixels to circle in the "
                                     "GIF (default: the strongest spot)")
    args = ap.parse_args()

    if len(args.games) < 2:
        raise SystemExit("one game cannot be held out from itself; "
                         "give at least two")
    titles = (dict(zip(args.games, args.titles.split(",")))
              if args.titles else None)
    try:
        results = score_window(
            args.games, Contrast(args.pos, tuple(args.neg.split(","))),
            args.stance, args.name, heading=args.heading, titles=titles,
            geom=args.geom, knee_check=not args.no_knee_check,
            n_perm=args.n_perm,
            circle=(tuple(float(v) for v in args.circle.split(","))
                    if args.circle else None))
    except TooFewPitches as e:
        raise SystemExit(f"{e}. Leave that game out of the window.")
    print(summary(results))
    print(f"-> {ROOT / 'reports' / args.name / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
