"""Score a pitch-type difference on games the template has never seen.

The whole method is one subtraction. Within a game, average the pictures
before one pitch type, average the pictures before the other, and subtract:
that difference map is what a tell would look like. To test it, build the
map from every game but one, then ask how well it sorts the pitches of the
game left out. Rotate through the games and combine.

Three rules keep the answer honest:

  - Pictures are never pooled across games. Lighting, uniform and ballpark
    differ far more between games than any tell does, so every difference
    is taken inside one game and only the differences are averaged.
  - The held-out split is by game, never by pitch. Pitches from one game
    share a camera and an evening, and would leak into each other.
  - Significance comes from shuffling the labels and re-running everything.
    Shuffles stay inside a game and inside a ball-strike count, because the
    count itself predicts pitch type.

Scores are AUCs: 0.5 is a coin flip, 1.0 is a perfect read. The same code
scores pixels and pose keypoints; it only ever sees a matrix per game.
"""

from dataclasses import dataclass

import numpy as np

from .config import N_BOOT, N_PERM, RNG_SEED

CHANNELS = ("full", "body", "bg")


@dataclass(frozen=True)
class Contrast:
    """Two pitch classes to tell apart. `neg` may pool several Statcast
    codes that are one pitch in the hand (a splitter logged as FS or FO)."""
    pos: str
    neg: tuple[str, ...]

    @property
    def classes(self) -> tuple[str, ...]:
        return (self.pos,) + self.neg


def game_matrix(g: dict, contrast: Contrast, channel: str):
    """(X, y, strata) for one game's contrast pitches on one pixel channel.

    full  the crop as it is
    body  background replaced by the game's average frame: pitcher only
    bg    pitcher replaced by the game's average frame: background only

    The replacement image averages every pitch regardless of type, so it
    cannot carry a label. y is 1 for `pos`; strata encode the count.
    """
    assert channel in CHANNELS, channel
    sel = np.isin(g["pitch_type"], contrast.classes)
    X = g["X"][sel].astype(np.float64)
    if channel != "full" and X.shape[0]:
        mean_img = g["X"].astype(np.float64).mean(axis=0)
        for i, m in enumerate(g["mask"][sel]):
            region = m if channel == "bg" else ~m
            X[i][region] = mean_img[region]
    y = (g["pitch_type"][sel] == contrast.pos).astype(np.int64)
    strata = g["balls"][sel] * 10 + g["strikes"][sel]
    D = int(np.prod(X.shape[1:]))
    return X.reshape(X.shape[0], D), y, strata


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Probability that a random `pos` pitch outscores a random `neg` one,
    ties counting half."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sv = s[order]
    r = np.arange(1, len(s) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def diff_map(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def logo_scores(mats: dict, ys: dict) -> dict:
    """Score every game against the average difference map of the others.
    Each game is centered on its own mean first, so only within-game
    variation is scored."""
    d = {pk: diff_map(mats[pk], ys[pk]) for pk in mats}
    out = {}
    for pk in mats:
        template = np.mean([d[q] for q in mats if q != pk], axis=0)
        out[pk] = (mats[pk] - mats[pk].mean(axis=0)) @ template
    return out


def logo_aucs(mats: dict, ys: dict) -> dict:
    scores = logo_scores(mats, ys)
    return {pk: auc_score(ys[pk], scores[pk]) for pk in mats}


def weighted_auc(aucs: dict, ns: dict) -> float:
    return float(sum(aucs[pk] * ns[pk] for pk in aucs) / sum(ns.values()))


def permute_within_strata(y: np.ndarray, strata: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    yp = y.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        yp[idx] = y[rng.permutation(idx)]
    return yp


def evaluate(mats: dict, ys: dict, strata: dict, n_perm: int = N_PERM,
             n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """Held-out AUC per game with bootstrap intervals, the pitch-weighted
    aggregate, and its permutation p-value.

    The mean of the shuffled runs is returned next to the p-value because it
    is not 0.5. With a few small games it often sits several points higher,
    and an AUC means little without it.
    """
    games = list(mats)
    ns = {pk: len(ys[pk]) for pk in games}
    scores = logo_scores(mats, ys)
    aucs = {pk: auc_score(ys[pk], scores[pk]) for pk in games}
    agg = weighted_auc(aucs, ns)

    rng = np.random.default_rng(seed)
    cis = {}
    for pk in games:
        y, s = ys[pk], scores[pk]
        i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
        boot = []
        for _ in range(n_boot):
            b = np.concatenate([rng.choice(i1, len(i1)),
                                rng.choice(i0, len(i0))])
            boot.append(auc_score(y[b], s[b]))
        cis[pk] = [float(np.percentile(boot, 2.5)),
                   float(np.percentile(boot, 97.5))]

    null = np.empty(n_perm)
    for p in range(n_perm):
        yp = {pk: permute_within_strata(ys[pk], strata[pk], rng)
              for pk in games}
        null[p] = weighted_auc(logo_aucs(mats, yp), ns)

    return {
        "auc": agg,
        "null_mean": float(null.mean()),
        "p": float((1 + np.sum(null >= agg)) / (n_perm + 1)),
        "auc_per_game": {str(pk): aucs[pk] for pk in games},
        "ci95_per_game": {str(pk): cis[pk] for pk in games},
        "n_per_game": {str(pk): ns[pk] for pk in games},
    }


def template(mats: dict, ys: dict) -> np.ndarray:
    """The window's difference map: per-game maps, averaged."""
    return np.mean([diff_map(mats[pk], ys[pk]) for pk in mats], axis=0)


def localization(T: np.ndarray, mask: np.ndarray, top_frac: float = 0.05) -> dict:
    """How much of a pixel template sits on the pitcher. `mask_area` is what
    a template with no preference would score."""
    a = np.abs(T)
    k = max(1, int(top_frac * a.size))
    top = np.zeros_like(mask)
    top.reshape(-1)[np.argsort(a.reshape(-1))[-k:]] = True
    return {
        "mask_area": float(mask.mean()),
        "energy_in_mask": float(a[mask].sum() / a.sum()),
        "top5pct_in_mask": float((top & mask).sum() / k),
    }
