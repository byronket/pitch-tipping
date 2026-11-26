"""Download every pitch clip from a pitcher's starts and label them.

    python scripts/01_fetch.py 607192 2019-04-28 2019-05-10

Takes the pitcher's MLBAM id and one or more game dates, and prints the
game ids the later scripts take.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tell.fetch import fetch_start          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pitcher", type=int, help="MLBAM player id")
    ap.add_argument("dates", nargs="+", help="game dates, YYYY-MM-DD")
    ap.add_argument("--team", help="team abbreviation of either side; needed "
                                   "for starts made before a trade")
    args = ap.parse_args()

    games = [g for date in args.dates
             for g in fetch_start(args.pitcher, date, args.team)]
    print("game ids:", " ".join(str(g) for g in games))
    return 0


if __name__ == "__main__":
    sys.exit(main())
