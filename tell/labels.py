"""Join each downloaded clip to its Statcast row.

Every label downstream comes through this join, and a silent mismatch would
poison all of it, so the join refuses to guess:

  - the key (game, pitcher, batter, inning, pitch number in the at-bat) must
    be unique on both sides;
  - clip count, Statcast count and joined count must agree for the game;
  - unmatched rows on either side are written to a quarantine file and fail
    the run;
  - the pitch type must agree between the two sources on every joined row.
"""

from pathlib import Path

import polars as pl

JOIN_KEYS = ["game_pk", "pitcher", "batter", "inning", "pitch_number"]

STATCAST_KEEP = [
    "pitch_type", "pitch_name", "release_speed",
    "balls", "strikes", "outs_when_up",
    "stand", "p_throws",
    "on_1b", "on_2b", "on_3b",
    "inning_topbot", "at_bat_number",
    "description", "events",
]


def _statcast(start: str, end: str, pitcher_id: int) -> pl.DataFrame:
    from pybaseball import statcast_pitcher   # slow import

    sc = pl.from_pandas(statcast_pitcher(start, end, pitcher_id))
    keep = JOIN_KEYS + [c for c in STATCAST_KEEP if c in sc.columns]
    return sc.select(keep).with_columns(
        [pl.col(k).cast(pl.Int64) for k in JOIN_KEYS])


def build_labels(play_ids_csv: Path, start: str, end: str, pitcher_id: int,
                 out_csv: Path, quarantine_csv: Path) -> pl.DataFrame:
    clips = pl.read_csv(play_ids_csv)
    clips = clips.filter(
        (pl.col("pitcher") == pitcher_id) & pl.col("is_pitch")
    ).rename({"pitch_number_ab": "pitch_number"})
    clips = clips.with_columns([pl.col(k).cast(pl.Int64) for k in JOIN_KEYS])

    sc = _statcast(start, end, pitcher_id)

    problems: list[str] = []
    for name, df in (("clips", clips), ("statcast", sc)):
        dupes = df.height - df.unique(subset=JOIN_KEYS).height
        if dupes:
            problems.append(f"{name}: {dupes} duplicate join keys")

    clip_side = clips.select(
        JOIN_KEYS + ["play_id", "pitch_type"]
    ).rename({"pitch_type": "pitch_type_gameday"})
    joined = clip_side.join(sc, on=JOIN_KEYS, how="inner")

    unmatched_clips = clip_side.join(sc, on=JOIN_KEYS, how="anti")
    unmatched_sc = sc.join(clip_side, on=JOIN_KEYS, how="anti")
    if unmatched_clips.height or unmatched_sc.height:
        quarantine = pl.concat([
            unmatched_clips.select(JOIN_KEYS).with_columns(
                pl.lit("clip_without_statcast").alias("side")),
            unmatched_sc.select(JOIN_KEYS).with_columns(
                pl.lit("statcast_without_clip").alias("side")),
        ])
        quarantine_csv.parent.mkdir(parents=True, exist_ok=True)
        quarantine.write_csv(quarantine_csv)
        problems.append(
            f"unmatched rows: {unmatched_clips.height} clips, "
            f"{unmatched_sc.height} statcast -> {quarantine_csv}")

    per_game = (
        clips.group_by("game_pk").len().rename({"len": "n_clips"})
        .join(sc.group_by("game_pk").len().rename({"len": "n_statcast"}),
              on="game_pk", how="full", coalesce=True)
        .join(joined.group_by("game_pk").len().rename({"len": "n_joined"}),
              on="game_pk", how="full", coalesce=True)
        .fill_null(0)
        .sort("game_pk")
    )
    bad = per_game.filter(
        (pl.col("n_clips") != pl.col("n_statcast"))
        | (pl.col("n_clips") != pl.col("n_joined")))
    if bad.height:
        problems.append(f"pitch counts do not reconcile:\n{bad}")

    disagree = joined.filter(
        pl.col("pitch_type_gameday").fill_null("")
        != pl.col("pitch_type").fill_null(""))
    if disagree.height:
        problems.append(
            f"pitch type disagrees on {disagree.height} joined rows:\n"
            f"{disagree.select(JOIN_KEYS + ['pitch_type_gameday', 'pitch_type'])}")

    if problems:
        raise AssertionError("label join failed:\n- " + "\n- ".join(problems))

    labels = joined.drop("pitch_type_gameday").sort(
        ["game_pk", "inning", "at_bat_number", "pitch_number"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.write_csv(out_csv)
    print(f"{labels.height} labeled pitches -> {out_csv}")
    return labels
