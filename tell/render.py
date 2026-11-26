"""Pictures of a scored window.

The numbers say whether something is there. These show what it is, and they
are also how the mistakes get caught: a frame from the middle of a delivery
that slipped past every check is obvious the moment it is on screen.
"""

import cv2
import numpy as np

from .crops import display_crops
from .pose import L_ELB, L_HIP, L_SHO, L_WRI, R_ELB, R_HIP, R_SHO, R_WRI
from .scoring import diff_map, logo_scores
from .window import pitcher_mask

PITCH_NAMES = {"FF": "fastball", "SI": "sinker", "FC": "cutter",
               "CH": "changeup", "FS": "splitter", "FO": "forkball",
               "SL": "slider", "ST": "sweeper", "SV": "slurve",
               "CU": "curveball", "KC": "curveball", "CS": "curveball"}


def class_names(contrast) -> tuple[str, str]:
    neg = sorted({PITCH_NAMES.get(c, c) for c in contrast.neg})
    return PITCH_NAMES.get(contrast.pos, contrast.pos), "/".join(neg)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _display(window: dict) -> dict:
    """Full-resolution versions of the window's crops, cut once."""
    if "display" not in window:
        window["display"] = {pk: display_crops(pk, window["crops"][pk])
                             for pk in window["games"]}
    return window["display"]


def _titles(window: dict, titles: dict | None) -> dict:
    return {pk: str((titles or {}).get(pk, pk)) for pk in window["games"]}


def render_heatmap(T: np.ndarray, window: dict, path) -> None:
    """A typical set frame, the difference map, and the two overlaid. The
    green box is the pitcher mask."""
    plt = _plt()
    pos, neg = class_names(window["contrast"])
    first = window["games"][0]
    X = window["crops"][first]["X"]
    typical = int(np.argmin(np.mean(np.abs(X - np.median(X, axis=0)),
                                    axis=(1, 2))))
    base = _display(window)[first][typical]
    sy, sx = base.shape[0] / T.shape[0], base.shape[1] / T.shape[1]
    shown = cv2.resize(T, (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    vmax = np.abs(T).max()
    alpha = np.clip(np.abs(shown) / vmax, 0, 1) ** 0.7

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 5.5))
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("a typical set frame", fontsize=9)
    axes[1].imshow(shown, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"where they differ\n(red = brighter before the {pos},"
                      f" blue = before the {neg})", fontsize=9)
    axes[2].imshow(base, cmap="gray")
    axes[2].imshow(shown, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=alpha)
    axes[2].set_title("overlay", fontsize=9)
    ys, xs = np.where(pitcher_mask(window))
    for ax in (axes[0], axes[2]):
        ax.add_patch(plt.Rectangle((xs.min() * sx, ys.min() * sy),
                                   (xs.max() - xs.min()) * sx,
                                   (ys.max() - ys.min()) * sy,
                                   fill=False, edgecolor="lime", lw=1))
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_classmeans(window: dict, path, titles: dict | None = None) -> None:
    """The average frame before each pitch type, one column per game."""
    plt = _plt()
    games, titles = window["games"], _titles(window, titles)
    pos, neg = class_names(window["contrast"])
    show = _display(window)
    fig, axes = plt.subplots(2, len(games), figsize=(3.0 * len(games), 6.5),
                             squeeze=False)
    for c, pk in enumerate(games):
        y = window["y"][pk]
        axes[0][c].imshow(show[pk][y == 1].mean(axis=0), cmap="gray")
        axes[0][c].set_title(f"{titles[pk]}\naverage {pos} (n={(y == 1).sum()})",
                             fontsize=9, color="tab:red")
        axes[1][c].imshow(show[pk][y == 0].mean(axis=0), cmap="gray")
        axes[1][c].set_title(f"average {neg} (n={(y == 0).sum()})",
                             fontsize=9, color="tab:blue")
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_flip(T: np.ndarray, window: dict, path, titles: dict | None = None,
                circle: tuple | None = None) -> None:
    """Animated GIF fading between the two average frames, games side by
    side, with the hottest part of the difference map circled.

    `circle` is an (x, y) in crop pixels to circle instead, for a tell that
    was located by eye.
    """
    from PIL import Image

    games, titles = window["games"], _titles(window, titles)
    pos, neg = class_names(window["contrast"])
    show = _display(window)
    h, w = show[games[0]].shape[1:3]
    # Circle the peak of the smoothed on-body template. The raw top pixels
    # are scattered along limb edges and their centroid lands off the body.
    if circle is None:
        energy = cv2.GaussianBlur(
            np.abs(T).astype(np.float32) * pitcher_mask(window), (0, 0), 3)
        py, px = np.unravel_index(int(np.argmax(energy)), energy.shape)
    else:
        px, py = circle
    center = (int(px * w / T.shape[1]), int(py * h / T.shape[0]))
    header, gap = 34, 6

    panels = []
    for pk in games:
        y = window["y"][pk]
        a, b = show[pk][y == 1].mean(axis=0), show[pk][y == 0].mean(axis=0)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())

        def to8(m):
            return np.clip((m - lo) / max(hi - lo, 1e-6) * 255, 0, 255
                           ).astype(np.uint8)
        panels.append((titles[pk], to8(a), to8(b),
                       int((y == 1).sum()), int((y == 0).sum())))

    def compose(t: float) -> np.ndarray:
        canvas = np.full((header + h, len(games) * (w + gap) - gap, 3), 24,
                         dtype=np.uint8)
        for i, (title, a, b, na, nb) in enumerate(panels):
            x0 = i * (w + gap)
            tile = cv2.cvtColor(cv2.addWeighted(a, 1 - t, b, t, 0),
                                cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas, title, (x0 + 4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.circle(tile, center, int(0.22 * w), (0, 220, 255), 2,
                       cv2.LINE_AA)
            text, color = ((f"avg {pos} (n={na})", (80, 80, 255))
                           if t < 0.5 else
                           (f"avg {neg} (n={nb})", (255, 160, 60)))
            cv2.putText(canvas, text, (x0 + 4, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
            canvas[header:, x0:x0 + w] = tile
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    steps = ([0.0] * 8 + [i / 6 for i in range(1, 6)]
             + [1.0] * 8 + [1 - i / 6 for i in range(1, 6)])
    frames = [Image.fromarray(compose(t)) for t in steps]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=False)


def render_exemplars(window: dict, path, titles: dict | None = None,
                     n_each: int = 3) -> None:
    """The pitches the held-out template is most sure about, per game and
    class, with the template's hottest 3% outlined.

    This sheet is worth looking at every time. If the top scorers are frames
    from the middle of a delivery, the template has learned the picker's
    mistakes, not the pitcher.
    """
    plt = _plt()
    games, titles = window["games"], _titles(window, titles)
    pos, neg = class_names(window["contrast"])
    mats, ys = window["X"]["body"], window["y"]
    d = {pk: diff_map(mats[pk], ys[pk]) for pk in games}
    scores = logo_scores(mats, ys)
    show = _display(window)
    shape = window["crops"][games[0]]["X"].shape[1:]

    fig, axes = plt.subplots(2 * len(games), n_each,
                             figsize=(2.6 * n_each, 7.8 * len(games)),
                             squeeze=False)
    for gi, pk in enumerate(games):
        held_out = np.abs(np.mean([d[q] for q in games if q != pk], axis=0)
                          ).reshape(shape)
        outline = cv2.resize(held_out, show[pk].shape[2:0:-1],
                             interpolation=cv2.INTER_NEAREST)
        level = np.quantile(held_out, 0.97)
        s, y = scores[pk], ys[pk]
        for row, (name, order, want) in enumerate(
                ((pos, np.argsort(-s), 1), (neg, np.argsort(s), 0))):
            top = [i for i in order if y[i] == want][:n_each]
            for col in range(n_each):
                ax = axes[gi * 2 + row][col]
                ax.set_xticks([]), ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{titles[pk]}\n{name}", fontsize=8)
                if col >= len(top):
                    ax.set_visible(False)
                    continue
                ax.imshow(show[pk][top[col]], cmap="gray")
                ax.contour(outline, levels=[level], colors="lime",
                           linewidths=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_pose_means(window: dict, path) -> None:
    """The average upper-body skeleton before each pitch type, pooled over
    the window, in the pitcher's own frame."""
    plt = _plt()
    pos, neg = class_names(window["contrast"])
    skeleton = np.concatenate([window["skeleton"][pk]
                               for pk in window["games"]])
    y = np.concatenate([window["y"][pk] for pk in window["games"]])
    links = [(L_SHO, R_SHO), (L_HIP, R_HIP), (L_SHO, L_HIP), (R_SHO, R_HIP),
             (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI)]
    joints = sorted({j for link in links for j in link})

    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    for want, name, color in ((1, pos, "tab:red"), (0, neg, "tab:blue")):
        mean = skeleton[y == want].mean(axis=0)
        for a, b in links:
            ax.plot(mean[[a, b], 0], mean[[a, b], 1], "-", color=color,
                    lw=2, alpha=0.8)
        ax.plot(mean[joints, 0], mean[joints, 1], "o", color=color, ms=5)
        ax.plot([], [], "-o", color=color,
                label=f"average {name} (n={(y == want).sum()})")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("across the body (torso lengths)")
    ax.set_ylabel("hips (0) to shoulders (-1)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
