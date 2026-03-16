from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def _to_float(s: str) -> float | None:
    s = s.strip()
    if s == "" or s.upper() in {"NA", "N/A", "NULL", "NONE"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_team_name(name: str) -> str:
    return " ".join(name.strip().split())


def _looks_like_leak_feature(col: str) -> bool:
    """
    "Use every feature" is risky because some files contain tournament outcomes or simulation outputs.
    We still include a *lot*, but avoid the most direct leakage columns.
    """
    c = col.strip().upper()
    return c in {
        "SCORE",
        "ROUND",
        "CURRENT ROUND",
        "CHAMP",
        "CHAMP%",
        "F2",
        "F4",
        "F4%",
        "E8",
        "S16",
        "R32",
        "R64",
        "TOP2",
        "SIM",
        "SIMS",
        "PICK",
        "PUBLIC",
    }


def _list_feature_csvs(dataset_dir: Path) -> list[Path]:
    """
    Include "team-level" feature tables; exclude obvious outcome/bracket tables.
    """
    exclude = {
        "Tournament Matchups.csv",
        "Tournament Simulation.csv",
        "Tournament Locations.csv",
        "Seed Results.csv",
        "Upset Count.csv",
        "Upset Seed Info.csv",
        "Public Picks.csv",
        "Coach Results.csv",
        "Conference Results.csv",
        "Team Results.csv",
    }
    paths = []
    for p in sorted(dataset_dir.glob("*.csv")):
        if p.name in exclude:
            continue
        paths.append(p)
    return paths


def load_team_feature_store(
    dataset_dir: Path, *, years: list[int]
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, np.ndarray], list[str]]:
    """
    Builds a wide feature matrix by taking *all numeric columns* from the selected dataset CSVs.

    Returns:
    - by_year_team: (year, team_name) -> feature vector
    - by_team_hist_mean: team_name -> historical mean feature vector (over provided years)
    - feature_names: list of feature names aligned to the vectors
    """
    files = _list_feature_csvs(dataset_dir)

    # 1) Collect union of numeric feature columns per file
    # We create stable feature names as "<file>::<col>"
    feature_names: list[str] = []
    feature_idx: dict[str, int] = {}

    # Cache rows per file to avoid reread
    file_rows: dict[Path, list[dict[str, str]]] = {p: _read_csv_rows(p) for p in files}

    for p, rows in file_rows.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        for col in cols:
            col_u = col.strip().upper()
            if col_u in {"YEAR", "TEAM", "TEAM NO", "TEAM ID", "CONF", "CONF ID"}:
                continue
            if _looks_like_leak_feature(col):
                continue
            # consider numeric if at least one row parses
            any_numeric = False
            for r in rows[:200]:
                v = _to_float(r.get(col, ""))
                if v is not None and math.isfinite(v):
                    any_numeric = True
                    break
            if not any_numeric:
                continue
            key = f"{p.stem}::{col}"
            if key not in feature_idx:
                feature_idx[key] = len(feature_names)
                feature_names.append(key)

    d = len(feature_names)
    if d == 0:
        raise SystemExit("No numeric features discovered in dataset CSVs.")

    # 2) Build vectors: (year, team) -> aggregated (mean) per feature
    acc: dict[tuple[int, str], list[np.ndarray]] = {}

    for p, rows in file_rows.items():
        for r in rows:
            y_raw = r.get("YEAR", "")
            team_raw = r.get("TEAM", "")
            if y_raw is None or team_raw is None:
                continue
            try:
                y = int(y_raw)
            except ValueError:
                continue
            if y not in years:
                continue
            team = _normalize_team_name(team_raw)
            vec = np.full(d, np.nan, dtype=np.float64)
            for col, val in r.items():
                col_u = col.strip().upper()
                if col_u in {"YEAR", "TEAM", "TEAM NO", "TEAM ID", "CONF", "CONF ID"}:
                    continue
                if _looks_like_leak_feature(col):
                    continue
                fkey = f"{p.stem}::{col}"
                j = feature_idx.get(fkey)
                if j is None:
                    continue
                fv = _to_float(val or "")
                if fv is None or not math.isfinite(fv):
                    continue
                vec[j] = fv
            acc.setdefault((y, team), []).append(vec)

    by_year_team: dict[tuple[int, str], np.ndarray] = {}
    for k, mats in acc.items():
        M = np.stack(mats, axis=0)
        by_year_team[k] = np.nanmean(M, axis=0)

    # 3) Historical mean per team
    team_to_year_vecs: dict[str, list[np.ndarray]] = {}
    for (y, team), v in by_year_team.items():
        if y in years:
            team_to_year_vecs.setdefault(team, []).append(v)
    by_team_hist_mean: dict[str, np.ndarray] = {}
    for team, vs in team_to_year_vecs.items():
        by_team_hist_mean[team] = np.nanmean(np.stack(vs, axis=0), axis=0)

    return by_year_team, by_team_hist_mean, feature_names


@dataclass(frozen=True)
class TeamFeatures:
    seed: float
    z_rating: float


@dataclass(frozen=True)
class Game:
    year: int
    round_num: int
    team_a: str
    team_b: str
    team_no_a: int
    team_no_b: int
    score_a: float | None
    score_b: float | None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_kenpom_barttorvik_adj_em(path: Path) -> dict[tuple[int, str], tuple[float, float]]:
    """
    Returns (KADJ EM, BADJ EM) keyed by (YEAR, TEAM).
    """
    rows = _read_csv_rows(path)
    out: dict[tuple[int, str], tuple[float, float]] = {}
    for r in rows:
        try:
            year = int(r["YEAR"])
            team = r["TEAM"].strip()
            kadj_em = float(r["KADJ EM"])
            badj_em = float(r["BADJ EM"])
        except (KeyError, ValueError):
            continue
        if not (math.isfinite(kadj_em) and math.isfinite(badj_em)):
            continue
        out[(year, team)] = (kadj_em, badj_em)
    return out


def make_program_strength(
    kenpom: dict[tuple[int, str], tuple[float, float]],
    *,
    years: list[int],
    use: str = "kadj_em",
) -> dict[str, float]:
    idx = 0 if use == "kadj_em" else 1
    acc: dict[str, list[float]] = {}
    for (y, team), vals in kenpom.items():
        if y not in years:
            continue
        acc.setdefault(team, []).append(vals[idx])
    return {team: float(np.mean(v)) for team, v in acc.items() if len(v) > 0}


@dataclass(frozen=True)
class BracketMatchup:
    region: str
    seed_a: int
    team_a: str
    seed_b: int
    team_b: str


def bracket_2026_round_of_64() -> list[BracketMatchup]:
    """
    Matchups transcribed from the provided 2026 bracket image.
    First Four play-in winners are kept as "X/Y" placeholders.
    """
    return [
        # EAST
        BracketMatchup("EAST", 1, "Duke", 16, "Siena"),
        BracketMatchup("EAST", 8, "Ohio St.", 9, "TCU"),
        BracketMatchup("EAST", 5, "St. John's", 12, "North Texas"),
        BracketMatchup("EAST", 4, "Kansas", 13, "Cal Baptist"),
        BracketMatchup("EAST", 6, "Louisville", 11, "South Florida"),
        BracketMatchup("EAST", 3, "Michigan St.", 14, "North Dakota St."),
        BracketMatchup("EAST", 7, "UCLA", 10, "UCF"),
        BracketMatchup("EAST", 2, "UConn", 15, "Furman"),
        # SOUTH
        BracketMatchup("SOUTH", 1, "Florida", 16, "Lehigh/PVAMU"),
        BracketMatchup("SOUTH", 8, "Clemson", 9, "Iowa"),
        BracketMatchup("SOUTH", 5, "Vanderbilt", 12, "McNeese St."),
        BracketMatchup("SOUTH", 4, "Nebraska", 13, "Troy"),
        BracketMatchup("SOUTH", 6, "North Carolina", 11, "VCU"),
        BracketMatchup("SOUTH", 3, "Illinois", 14, "Penn"),
        BracketMatchup("SOUTH", 7, "Saint Mary's", 10, "Texas A&M"),
        BracketMatchup("SOUTH", 2, "Houston", 15, "Idaho"),
        # WEST
        BracketMatchup("WEST", 1, "Arizona", 16, "Long Island"),
        BracketMatchup("WEST", 8, "Villanova", 9, "Utah St."),
        BracketMatchup("WEST", 5, "Wisconsin", 12, "High Point"),
        BracketMatchup("WEST", 4, "Arkansas", 13, "Hawaii"),
        BracketMatchup("WEST", 6, "BYU", 11, "NC St./Texas"),
        BracketMatchup("WEST", 3, "Gonzaga", 14, "Kennesaw St."),
        BracketMatchup("WEST", 7, "Miami (FL)", 10, "Missouri"),
        BracketMatchup("WEST", 2, "Purdue", 15, "Queens (N.C.)"),
        # MIDWEST
        BracketMatchup("MIDWEST", 1, "Michigan", 16, "Howard/UMBC"),
        BracketMatchup("MIDWEST", 8, "Georgia", 9, "Saint Louis"),
        BracketMatchup("MIDWEST", 5, "Texas Tech", 12, "Akron"),
        BracketMatchup("MIDWEST", 4, "Alabama", 13, "Hofstra"),
        BracketMatchup("MIDWEST", 6, "Tennessee", 11, "SMU/Miami (Ohio)"),
        BracketMatchup("MIDWEST", 3, "Virginia", 14, "Wright St."),
        BracketMatchup("MIDWEST", 7, "Kentucky", 10, "Santa Clara"),
        BracketMatchup("MIDWEST", 2, "Iowa St.", 15, "Tennessee St."),
    ]


def load_z_ratings(path: Path) -> dict[tuple[int, int], TeamFeatures]:
    rows = _read_csv_rows(path)
    out: dict[tuple[int, int], TeamFeatures] = {}
    for r in rows:
        try:
            year = int(r["YEAR"])
            team_no = int(r["TEAM NO"])
            seed = float(r["SEED"]) if r.get("SEED", "").strip() != "" else float("nan")
            z = float(r["Z RATING"]) if r.get("Z RATING", "").strip() != "" else float("nan")
        except (KeyError, ValueError):
            continue
        if not (math.isfinite(seed) and math.isfinite(z)):
            continue
        out[(year, team_no)] = TeamFeatures(seed=seed, z_rating=z)
    return out


def load_games_from_tournament_matchups(path: Path) -> list[Game]:
    """
    `Tournament Matchups.csv` is structured as one row per team per game (not a single row per matchup).
    Within a (YEAR, CURRENT ROUND) block, rows are ordered so that each matchup is two consecutive rows.
    """
    rows = _read_csv_rows(path)

    parsed: list[tuple[int, int, int, str, int, float | None]] = []
    # (year, current_round, by_year_no, team, team_no, score)
    for r in rows:
        try:
            year = int(r["YEAR"])
            current_round = int(r["CURRENT ROUND"])
            by_year_no = int(r["BY YEAR NO"])
            team_no = int(r["TEAM NO"])
            team = r["TEAM"].strip()
            score_raw = r.get("SCORE", "").strip()
            score = float(score_raw) if score_raw != "" else None
        except (KeyError, ValueError):
            continue
        parsed.append((year, current_round, by_year_no, team, team_no, score))

    # sort to ensure pairing is consistent
    parsed.sort(key=lambda t: (t[0], t[1], -t[2]))

    games: list[Game] = []
    i = 0
    while i + 1 < len(parsed):
        year, rnd, _, team_a, team_no_a, score_a = parsed[i]
        year2, rnd2, _, team_b, team_no_b, score_b = parsed[i + 1]
        if year != year2 or rnd != rnd2:
            i += 1
            continue
        games.append(
            Game(
                year=year,
                round_num=rnd,
                team_a=team_a,
                team_b=team_b,
                team_no_a=team_no_a,
                team_no_b=team_no_b,
                score_a=score_a,
                score_b=score_b,
            )
        )
        i += 2
    return games


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def make_game_features(
    game: Game, feats: dict[tuple[int, int], TeamFeatures]
) -> tuple[np.ndarray, int] | None:
    fa = feats.get((game.year, game.team_no_a))
    fb = feats.get((game.year, game.team_no_b))
    if fa is None or fb is None:
        return None

    # Features are from A perspective.
    x = np.array(
        [
            fa.z_rating - fb.z_rating,
            fb.seed - fa.seed,  # positive if A has better (lower) seed
        ],
        dtype=np.float64,
    )

    if game.score_a is None or game.score_b is None:
        return x, -1
    if game.score_a == game.score_b:
        return None
    y = 1 if game.score_a > game.score_b else 0
    return x, y


def train_logreg_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    steps: int = 5000,
    l2: float = 1e-2,
) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        # gradients
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def impute_nan_with_col_means(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    X2 = np.where(np.isnan(X), col_means[None, :], X)
    return X2, col_means


def _vec_for_team_name(
    team: str,
    year: int,
    by_year_team: dict[tuple[int, str], np.ndarray],
    by_team_mean: dict[str, np.ndarray],
) -> np.ndarray | None:
    team_n = _normalize_team_name(team)
    v = by_year_team.get((year, team_n))
    if v is not None:
        return v
    if "/" in team_n:
        parts = [p.strip() for p in team_n.split("/")]
        vals = [by_team_mean[p] for p in parts if p in by_team_mean]
        if vals:
            return np.nanmean(np.stack(vals, axis=0), axis=0)
    return by_team_mean.get(team_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy win-probability model for March Madness.")
    parser.add_argument("--target_year", type=int, default=2026, help="Year to predict (default: 2026).")
    parser.add_argument(
        "--predict_round",
        type=int,
        default=64,
        help="Round to predict (64=Round of 64, 32=Round of 32, ...). Default: 64",
    )
    parser.add_argument("--train_start_year", type=int, default=2021, help="First training year (inclusive).")
    parser.add_argument("--train_end_year", type=int, default=2025, help="Last training year (inclusive).")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--steps", type=int, default=6000, help="Gradient steps.")
    parser.add_argument("--l2", type=float, default=1e-2, help="L2 regularization.")
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable matplotlib plotting (still prints probabilities).",
    )
    args = parser.parse_args()

    z_path = DATASET_DIR / "Z Rating Teams.csv"
    tm_path = DATASET_DIR / "Tournament Matchups.csv"
    kp_path = DATASET_DIR / "KenPom Barttorvik.csv"

    # (Legacy) KenPom AdjEM loader; kept for reference.
    _ = load_kenpom_barttorvik_adj_em(kp_path)
    games = load_games_from_tournament_matchups(tm_path)

    train_years = list(range(args.train_start_year, args.train_end_year + 1))

    by_year_team, by_team_mean, feature_names = load_team_feature_store(DATASET_DIR, years=train_years)
    z_seeds = load_z_ratings(z_path)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for g in games:
        if g.year not in train_years:
            continue
        if g.score_a is None or g.score_b is None or g.score_a == g.score_b:
            continue

        va = _vec_for_team_name(g.team_a, g.year, by_year_team, by_team_mean)
        vb = _vec_for_team_name(g.team_b, g.year, by_year_team, by_team_mean)
        if va is None or vb is None:
            continue

        fa = z_seeds.get((g.year, g.team_no_a))
        fb = z_seeds.get((g.year, g.team_no_b))
        seed_adv = float(fb.seed - fa.seed) if (fa is not None and fb is not None) else 0.0

        x = np.concatenate([va - vb, np.array([seed_adv], dtype=np.float64)], axis=0)
        y = 1 if g.score_a > g.score_b else 0
        X_list.append(x)
        y_list.append(y)

    if len(X_list) < 200:
        raise SystemExit(f"Not enough training games found ({len(X_list)}).")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float64)

    X, _ = impute_nan_with_col_means(X)
    mu, sigma = standardize_fit(X)
    Xs = standardize_apply(X, mu, sigma)

    w, b = train_logreg_numpy(Xs, y, lr=args.lr, steps=args.steps, l2=args.l2)
    p_train = _sigmoid(Xs @ w + b)
    print(f"Trained on {len(y)} games (years {args.train_start_year}-{args.train_end_year}).")
    print(f"Train Brier score: {brier_score(p_train, y):.4f}")
    print(f"Weights (standardized): {w}, bias: {b:.3f}")

    rows_out: list[tuple[str, str, float, str]] = []
    for m in bracket_2026_round_of_64():
        va = _vec_for_team_name(m.team_a, args.target_year, by_year_team, by_team_mean)
        vb = _vec_for_team_name(m.team_b, args.target_year, by_year_team, by_team_mean)
        if va is None:
            va = np.zeros(len(feature_names), dtype=np.float64)
        if vb is None:
            vb = np.zeros(len(feature_names), dtype=np.float64)

        x = np.concatenate([va - vb, np.array([float(m.seed_b - m.seed_a)], dtype=np.float64)], axis=0)
        x = np.where(np.isnan(x), 0.0, x)
        xs = standardize_apply(x[None, :], mu, sigma)
        p_a = float(_sigmoid(xs @ w + b)[0])
        rows_out.append((m.team_a, m.team_b, p_a, m.region))

    rows_out.sort(key=lambda t: (t[3], -abs(t[2] - 0.5), t[0]))

    print("\n2026 Round of 64 — predicted win% (Team A vs Team B):")
    cur_region = None
    for a, b_name, p_a, region in rows_out:
        if region != cur_region:
            cur_region = region
            print(f"\n[{region}]")
        print(f"- {a} vs {b_name}: {100.0*p_a:5.1f}% / {100.0*(1.0-p_a):5.1f}%")

    if args.no_plot:
        return

    # Simple visualization: strongest edges first
    top = rows_out[:32]
    labels = [f"{region}: {a} vs {b_name}" for a, b_name, _, region in top]
    probs = np.array([p for _, _, p, _ in top], dtype=np.float64)

    fig_h = max(8.0, 0.28 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probs, color="#1f77b4")
    ax.axvline(0.5, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(Team A wins)")
    ax.set_title(f"Predicted win probabilities — {args.target_year} Round {args.predict_round} (top {len(labels)})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
