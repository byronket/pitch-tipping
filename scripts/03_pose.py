"""Run the pose model on every set frame of one or more games.

    python scripts/03_pose.py 565039 566683 [--force]

One inference per pitch, on the JPEGs the set-frame step saved. The model
weights (about 6 MB) download on first use.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tell import pose          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("games", nargs="+", type=int, help="game ids")
    ap.add_argument("--force", action="store_true", help="redo cached games")
    args = ap.parse_args()

    model = pose.load_model()
    for game_pk in args.games:
        pose.extract_game(game_pk, model, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
