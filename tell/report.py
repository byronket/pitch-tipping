"""Score a window and write it up as one page.

The page leads with the pictures and one plain sentence. It reports what
the scores show and never declares a tell: a single window is one
uncorrected test, and a real claim also needs the frames looked at by eye.
"""

import html
import json

from . import render
from .config import DEFAULT_GEOM, N_PERM, PATCH_H, PATCH_W, ROOT
from .scoring import Contrast, evaluate, localization, template
from .window import assemble, pitcher_mask

MIN_PER_CLASS = 6    # pitches of each type a game needs to be scored
ALPHA = 0.05
VIEWS = (("body", "pitcher only"), ("bg", "background only"),
         ("full", "whole crop"), ("pose", "elbows and wrists"))


class TooFewPitches(ValueError):
    pass


def score_window(games: list[int], contrast: Contrast, stance: str,
                 name: str, *, heading: str | None = None,
                 titles: dict | None = None, geom: str = DEFAULT_GEOM,
                 knee_check: bool = True, n_perm: int = N_PERM,
                 circle: tuple | None = None) -> dict:
    """Score one window and write reports/{name}/: results.json, the
    pictures, and index.html. Returns the results."""
    window = assemble(games, contrast, stance, geom, knee_check)
    for pk in games:
        y = window["y"][pk]
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if min(n_pos, n_neg) < MIN_PER_CLASS:
            raise TooFewPitches(
                f"game {pk} has {n_pos} {contrast.pos} and {n_neg} "
                f"{'/'.join(contrast.neg)} usable {stance} pitches; each "
                f"game needs {MIN_PER_CLASS} of both")

    results = {
        "games": games, "pos": contrast.pos, "neg": list(contrast.neg),
        "stance": stance, "geom": geom, "knee_check": knee_check,
        "n_perm": n_perm, "circle": circle,
        "n_pitches": sum(len(window["y"][pk]) for pk in games),
        "gates": window["gates"], "pose_dropout": window["pose_dropout"],
        "views": {key: evaluate(window["X"][key], window["y"],
                                window["strata"], n_perm=n_perm)
                  for key, _ in VIEWS},
    }
    T = template(window["X"]["body"], window["y"]).reshape(PATCH_H, PATCH_W)
    results["localization"] = localization(T, pitcher_mask(window))
    results["reading"] = reading(results)

    out = ROOT / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    render.render_heatmap(T, window, out / "heatmap.png")
    render.render_classmeans(window, out / "classmeans.png", titles)
    render.render_flip(T, window, out / "flip.gif", titles, circle=circle)
    render.render_exemplars(window, out / "exemplars.png", titles)
    render.render_pose_means(window, out / "pose_means.png")
    (out / "index.html").write_text(
        page(results, heading or name, titles), encoding="utf-8")
    return results


def reading(results: dict) -> str:
    """signal, background suspect, or inconclusive.

    The background is checked first: when it predicts the pitch, a high
    pitcher-only score cannot be trusted, whatever its p-value.
    """
    body, bg = results["views"]["body"], results["views"]["bg"]
    if bg["p"] <= ALPHA:
        return "background suspect"
    return "signal" if body["p"] <= ALPHA else "inconclusive"


def summary(results: dict) -> str:
    """The terminal version of the page."""
    lines = [f"{results['n_pitches']} pitches over {len(results['games'])} "
             f"games ({results['pos']} vs {'/'.join(results['neg'])}, "
             f"{results['stance']}): {results['reading']}",
             f"  {'view':<19}{'score':>6}{'shuffled':>10}{'p':>8}   per game"]
    for key, label in VIEWS:
        r = results["views"][key]
        lines.append(
            f"  {label:<19}{r['auc']:>6.3f}{r['null_mean']:>10.3f}"
            f"{r['p']:>8.4f}   "
            + "  ".join(f"{v:.2f}" for v in r["auc_per_game"].values()))
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def page(results: dict, heading: str, titles: dict | None) -> str:
    pos, neg = render.class_names(
        Contrast(results["pos"], tuple(results["neg"])))
    views = results["views"]
    body, bg = views["body"], views["bg"]
    span = ""
    if titles:
        shown = [str(titles[pk]) for pk in results["games"]]
        span = f", {shown[0]} to {shown[-1]}"

    lead = (f"Take one {pos} and one {neg} from a start it had not seen: "
            f"going only on his set position, it ranks them the right way "
            f"round {_pct(body['auc'])} of the time. With the labels "
            f"shuffled that number is {_pct(body['null_mean'])}.")
    explain = {
        "signal": "The background check did not flag it.",
        "background suspect":
            f"The background alone scores {_pct(bg['auc'])}, so something "
            f"behind him tracks the pitch and this score cannot be trusted.",
        "inconclusive": "That is no better than shuffled labels.",
    }[results["reading"]]

    rows = "".join(
        f"<tr><td>{label}</td><td>{_pct(views[key]['auc'])}</td>"
        f"<td>{_pct(views[key]['null_mean'])}</td>"
        f"<td>{views[key]['p']:.3f}</td></tr>"
        for key, label in VIEWS)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(heading)}</title>
<style>
 body {{ font: 17px/1.55 Georgia, serif; color: #1c1c1c; background: #fff;
        max-width: 720px; margin: 3rem auto; padding: 0 1.2rem; }}
 h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
 .note {{ color: #666; font-size: .9rem; }}
 img {{ max-width: 100%; display: block; margin: 1.2rem 0; }}
 table {{ border-collapse: collapse; font-size: .9rem; margin-top: 2rem; }}
 td, th {{ padding: .3rem 1.2rem .3rem 0; text-align: left;
          border-bottom: 1px solid #ddd; }}
</style></head><body>
<h1>{html.escape(heading)}</h1>
<p class="note">{pos} vs {neg}, from the {results['stance']}{span}.
{len(results['games'])} starts, {results['n_pitches']} pitches.</p>

<img src="flip.gif" alt="Average frame before each pitch, fading back and forth">
<p><b>{results['reading'].capitalize()}.</b> {lead} {explain}</p>

<img src="heatmap.png" alt="Where the two averages differ">

<table>
<tr><th>Looking at</th><th>Score</th><th>Shuffled</th><th>p</th></tr>
{rows}
</table>
<p class="note">One comparison on {len(results['games'])} starts, not
corrected for how many others were tried. Check the frames in
exemplars.png before believing it.</p>
</body></html>
"""
