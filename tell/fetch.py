"""Download a start's pitch clips from Baseball Savant and label them."""

import concurrent.futures
from pathlib import Path

import polars as pl

from .config import CLIPS, LABELS
from .labels import build_labels

MAX_WORKERS = 4
DOWNLOAD_PASSES = 3


def fetch_start(pitcher: int, date: str, team: str | None = None) -> list[int]:
    """Download and label every pitch `pitcher` threw on `date`. Clips already
    on disk are skipped. Returns the game ids (two for a doubleheader in
    which he pitched both games).

    `team` is the abbreviation of either club in the game. The scraper
    resolves a player to his CURRENT team and rejects any other, so a start
    made before a trade has to be queried by team.
    """
    from baseballcv.functions import BaseballSavVideoScraper

    CLIPS.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    scraper = BaseballSavVideoScraper(
        date, date,
        player=None if team else pitcher,
        team_abbr=team,
        download_folder=str(CLIPS),
        max_return_videos=None,   # the default returns a random sample of 10
    )
    # The play list also holds pickoffs and the like, and a player filter
    # matches him as a batter too.
    df = scraper.play_ids_df.filter(
        (pl.col("pitcher") == pitcher) & pl.col("is_pitch"))

    def clip(game_pk, play_id) -> Path:
        return CLIPS / f"{game_pk}_{play_id}.mp4"

    def missing() -> list:
        return [(g, p) for g, p in zip(df["game_pk"], df["play_id"])
                if not (clip(g, p).exists() and clip(g, p).stat().st_size)]

    todo = missing()
    print(f"{date}: {df.height} pitches, {len(todo)} to download")
    for _ in range(DOWNLOAD_PASSES):
        if not todo:
            break
        with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as pool:
            list(pool.map(lambda pair: scraper._download_video(*pair), todo))
        todo = missing()
    if todo:
        raise RuntimeError(f"{date}: {len(todo)} clips failed to download")

    games = []
    for (game_pk,), gdf in df.group_by("game_pk"):
        play_ids = LABELS / f"play_ids_{game_pk}.csv"
        gdf.write_csv(play_ids)
        build_labels(play_ids_csv=play_ids, start=date, end=date,
                     pitcher_id=pitcher,
                     out_csv=LABELS / f"labels_{game_pk}.csv",
                     quarantine_csv=LABELS / f"quarantine_{game_pk}.csv")
        games.append(int(game_pk))
    return games
