"""Check a pitcher's recent starts for a tell.

    python run.py "Eduardo Rodriguez"
    python run.py "Eduardo Rodriguez" --dates 2026-08-21 2026-08-26 2026-09-01

Downloads the starts (his last three unless told otherwise), finds the set
frame of every pitch, and compares his fastball with his most-used other
pitch, from the stretch and from the windup. Each comparison gets a
one-page write-up in reports/.

Pick the pitches yourself with --pos and --neg (Statcast codes) and one
stance with --stance.
"""

import argparse
import sys
from collections import Counter
from datetime import date

import polars as pl

from tell import pose
from tell.config import LABELS, ROOT
from tell.fetch import fetch_start
from tell.lookup import find_pitcher, find_starts
from tell.render import PITCH_NAMES
from tell.report import TooFewPitches, score_window, summary
from tell.scoring import Contrast
from tell.set_frames import extract_game
from tell.window import STANCES

FASTBALLS = ("FF", "SI", "FC")
SAME_PITCH = {"FO": "FS", "KC": "CU", "CS": "CU"}   # logged under two codes


def default_contrast(games: list[int]) -> Contrast:
    """His most-used fastball against his most-used other pitch."""
    mix = Counter()
    for pk in games:
        mix.update(pl.read_csv(LABELS / f"labels_{pk}.csv")["pitch_type"]
                   .drop_nulls().to_list())
    fastball = max(FASTBALLS, key=lambda c: mix[c])
    other = Counter()
    for code, n in mix.items():
        if code not in FASTBALLS:
            other[SAME_PITCH.get(code, code)] += n
    if not mix[fastball] or not other:
        raise SystemExit(f"cannot pick two pitches from this mix: {dict(mix)}")
    pitch = other.most_common(1)[0][0]
    return Contrast(fastball, tuple(
        c for c in mix if SAME_PITCH.get(c, c) == pitch))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pitcher", help="pitcher's name")
    ap.add_argument("--starts", type=int, default=3,
                    help="how many of his most recent starts (default 3)")
    ap.add_argument("--dates", nargs="+", metavar="YYYY-MM-DD",
                    help="these starts, in place of the most recent ones")
    ap.add_argument("--season", type=int,
                    help="default: the year of --dates, else this year")
    ap.add_argument("--pos", help="pitch code, e.g. FF")
    ap.add_argument("--neg", help="pitch code(s) to compare it with, e.g. CH")
    ap.add_argument("--stance", choices=STANCES,
                    help="default: both, each scored on its own")
    args = ap.parse_args()
    if bool(args.pos) != bool(args.neg):
        ap.error("give both --pos and --neg, or neither")
    n_starts = len(args.dates) if args.dates else args.starts
    if n_starts < 2:
        ap.error("needs at least two starts: each one is scored against "
                 "the others")
    season = args.season or (int(args.dates[0][:4]) if args.dates
                             else date.today().year)

    try:
        pitcher = find_pitcher(args.pitcher)
        starts = find_starts(pitcher, season, args.starts, args.dates)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"{pitcher['name']}: " + ", ".join(s["date"] for s in starts))

    games, titles = [], {}
    for s in starts:
        for pk in fetch_start(pitcher["id"], s["date"], s["team"]):
            games.append(pk)
            titles[pk] = date.fromisoformat(s["date"]).strftime("%b %d").replace(" 0", " ")

    model = pose.load_model()
    for pk in games:
        print(f"finding set frames, game {pk}")
        extract_game(pk)
        pose.extract_game(pk, model)

    contrast = (Contrast(args.pos, tuple(args.neg.split(",")))
                if args.pos else default_contrast(games))
    pos = PITCH_NAMES.get(contrast.pos, contrast.pos)
    neg = PITCH_NAMES.get(contrast.neg[0], contrast.neg[0])
    slug = pitcher["name"].lower().replace(" ", "_")
    for stance in ([args.stance] if args.stance else STANCES):
        name = f"{slug}_{contrast.pos}_{contrast.neg[0]}_{stance}"
        try:
            results = score_window(games, contrast, stance, name,
                                   heading=pitcher["name"], titles=titles)
        except TooFewPitches as e:
            print(f"{stance}: skipped ({e})")
            continue
        print(summary(results))
        print(f"-> {ROOT / 'reports' / name / 'index.html'}")
    print(f"compared {pos} with {neg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
