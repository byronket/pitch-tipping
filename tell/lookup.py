"""Turn a pitcher's name into his id and recent starts (MLB Stats API)."""

from datetime import date

import requests

API = "https://statsapi.mlb.com/api/v1"
PITCHER_POSITIONS = {"1", "Y"}    # pitcher, two-way player


def find_pitcher(name: str) -> dict:
    """{'id', 'name', 'team_id'} for the one pitcher matching `name`.
    Active players win over retired namesakes; anything still ambiguous
    raises with the candidates listed."""
    people = requests.get(
        f"{API}/people/search", timeout=30,
        params={"names": name, "hydrate": "currentTeam"},
    ).json().get("people", [])
    hits = [p for p in people
            if p.get("primaryPosition", {}).get("code") in PITCHER_POSITIONS]
    exact = [p for p in hits if p["fullName"].lower() == name.strip().lower()]
    hits = exact or hits
    hits = [p for p in hits if p.get("active")] or hits
    if not hits:
        raise ValueError(f"no pitcher found for {name!r}")
    if len(hits) > 1:
        raise ValueError(f"{name!r} matches several pitchers: "
                         + ", ".join(p["fullName"] for p in hits))
    p = hits[0]
    return {"id": p["id"], "name": p["fullName"],
            "team_id": p.get("currentTeam", {}).get("id")}


def find_starts(pitcher: dict, season: int, last: int = 3,
                dates: list[str] | None = None) -> list:
    """His starts in `season`, oldest first, as {'date', 'game_pk', 'team'}:
    the ones on `dates` (YYYY-MM-DD) if given, else his `last` few. `team`
    is the club's abbreviation when he made the start for a team other than
    his current one, else None."""
    log = requests.get(
        f"{API}/people/{pitcher['id']}/stats", timeout=30,
        params={"stats": "gameLog", "group": "pitching",
                "season": season}).json()
    splits = log["stats"][0]["splits"] if log.get("stats") else []
    starts = [s for s in splits if s["stat"].get("gamesStarted")]
    if dates:
        missing = sorted(set(dates) - {s["date"] for s in starts})
        if missing:
            raise ValueError(f"{pitcher['name']} did not start on "
                             + ", ".join(missing))
        starts = [s for s in starts if s["date"] in dates]
    else:
        starts = starts[-last:]
    if not starts:
        raise ValueError(f"{pitcher['name']} has no starts in {season}")

    def abbreviation(team_id: int) -> str:
        return requests.get(f"{API}/teams/{team_id}", timeout=30
                            ).json()["teams"][0]["abbreviation"]

    return [{"date": s["date"], "game_pk": s["game"]["gamePk"],
             "team": (abbreviation(s["team"]["id"])
                      if s["team"]["id"] != pitcher["team_id"] else None)}
            for s in starts]
