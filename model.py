from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

DATASET_DIR = Path(__file__).resolve().parent / "dataset"


# ==========================================
# FEATURE ANALYZER CLASSES
# ==========================================
@dataclass
class FeatureImportance:
    name: str
    relative_importance: float


class FeatureAnalyzer:
    def __init__(self, feature_names: list[str], importances: np.ndarray):
        self.feature_names = list(feature_names)
        self.importances = np.asarray(importances, dtype=np.float64)

    def get_significant_features(
        self, min_relative_importance: float = 1.0
    ) -> list[FeatureImportance]:
        total_importance = np.sum(np.abs(self.importances))

        importances_list = []
        for name, imp in zip(self.feature_names, self.importances):
            rel_imp = (abs(imp) / total_importance) * 100 if total_importance > 0 else 0.0
            importances_list.append(FeatureImportance(name, rel_imp))

        importances_list.sort(key=lambda x: x.relative_importance, reverse=True)
        return [f for f in importances_list if f.relative_importance >= min_relative_importance]

    def print_report(self, min_relative_importance: float = 1.0) -> None:
        significant = self.get_significant_features(min_relative_importance)

        print("\n=======================================================")
        print(f"   SIGNIFICANT FEATURES (Threshold: >= {min_relative_importance}%)")
        print("=======================================================")
        print(f"{'Feature Name':<55} | {'Importance':<10}")
        print("-" * 70)

        cumulative = 0.0
        for f in significant:
            print(f"{f.name:<55} | {f.relative_importance:>5.2f}%")
            cumulative += f.relative_importance

        print("-" * 70)
        print(f"Total features kept: {len(significant)} out of {len(self.feature_names)}")
        print(f"These account for {cumulative:.1f}% of the model's coefficient magnitude.\n")


# ==========================================
# HELPER FUNCTIONS
# ==========================================
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
        "WINS",
        "HEAT CHECK TOURNAMENT INDEX::WINS",
        "HEAT CHECK TOURNAMENT INDEX::POOL VALUE",
        "HEAT CHECK TOURNAMENT INDEX::VAL Z-SCORE",
        "HEAT CHECK TOURNAMENT INDEX::POWER-PATH",
    }


def _list_feature_csvs(dataset_dir: Path) -> list[Path]:
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_team_feature_store(
    dataset_dir: Path, *, years: list[int]
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, np.ndarray], list[str]]:
    files = _list_feature_csvs(dataset_dir)
    feature_names: list[str] = []
    feature_idx: dict[str, int] = {}
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

    def _safe_nanmean(M: np.ndarray, axis: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(M, axis=axis)
        return np.where(np.isfinite(m), m, np.nan)

    by_year_team: dict[tuple[int, str], np.ndarray] = {}
    for k, mats in acc.items():
        M = np.stack(mats, axis=0)
        by_year_team[k] = _safe_nanmean(M, axis=0)

    team_to_year_vecs: dict[str, list[np.ndarray]] = {}
    for (y, team), v in by_year_team.items():
        if y in years:
            team_to_year_vecs.setdefault(team, []).append(v)

    by_team_hist_mean: dict[str, np.ndarray] = {}
    for team, vs in team_to_year_vecs.items():
        by_team_hist_mean[team] = _safe_nanmean(np.stack(vs, axis=0), axis=0)

    return by_year_team, by_team_hist_mean, feature_names


def drop_high_missing_features(
    by_year_team: dict[tuple[int, str], np.ndarray],
    by_team_hist_mean: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    max_missing_frac: float = 0.60,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, np.ndarray], list[str]]:
    if not by_year_team:
        return by_year_team, by_team_hist_mean, feature_names

    M = np.stack(list(by_year_team.values()), axis=0)
    missing_frac = np.mean(np.isnan(M), axis=0)
    keep = missing_frac <= max_missing_frac
    keep_idx = np.where(keep)[0]

    if keep_idx.size == 0:
        return by_year_team, by_team_hist_mean, feature_names

    new_names = [feature_names[i] for i in keep_idx.tolist()]
    new_by_year = {k: v[keep_idx] for k, v in by_year_team.items()}
    new_by_team = {k: v[keep_idx] for k, v in by_team_hist_mean.items()}
    return new_by_year, new_by_team, new_names


def keep_only_selected_base_features(
    by_year_team: dict[tuple[int, str], np.ndarray],
    by_team_mean: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    allowed_substrings: list[str],
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, np.ndarray], list[str]]:
    keep_idx = []
    for i, name in enumerate(feature_names):
        name_u = name.upper()
        if any(s.upper() in name_u for s in allowed_substrings):
            keep_idx.append(i)

    if not keep_idx:
        return by_year_team, by_team_mean, feature_names

    new_names = [feature_names[i] for i in keep_idx]
    new_by_year = {k: v[keep_idx] for k, v in by_year_team.items()}
    new_by_team = {k: v[keep_idx] for k, v in by_team_mean.items()}
    return new_by_year, new_by_team, new_names


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


@dataclass(frozen=True)
class BracketMatchup:
    region: str
    seed_a: int
    team_a: str
    seed_b: int
    team_b: str


def bracket_2026_round_of_64() -> list[BracketMatchup]:
    return [
        BracketMatchup("EAST", 1, "Duke", 16, "Siena"),
        BracketMatchup("EAST", 8, "Ohio St.", 9, "TCU"),
        BracketMatchup("EAST", 5, "St. John's", 12, "North Texas"),
        BracketMatchup("EAST", 4, "Kansas", 13, "Cal Baptist"),
        BracketMatchup("EAST", 6, "Louisville", 11, "South Florida"),
        BracketMatchup("EAST", 3, "Michigan St.", 14, "North Dakota St."),
        BracketMatchup("EAST", 7, "UCLA", 10, "UCF"),
        BracketMatchup("EAST", 2, "Connecticut", 15, "Furman"),
        BracketMatchup("SOUTH", 1, "Florida", 16, "Lehigh"),
        BracketMatchup("SOUTH", 8, "Clemson", 9, "Iowa"),
        BracketMatchup("SOUTH", 5, "Vanderbilt", 12, "McNeese St."),
        BracketMatchup("SOUTH", 4, "Nebraska", 13, "Troy"),
        BracketMatchup("SOUTH", 6, "North Carolina", 11, "VCU"),
        BracketMatchup("SOUTH", 3, "Illinois", 14, "Penn"),
        BracketMatchup("SOUTH", 7, "Saint Mary's", 10, "Texas A&M"),
        BracketMatchup("SOUTH", 2, "Houston", 15, "Idaho"),
        BracketMatchup("WEST", 1, "Arizona", 16, "LIU Brooklyn"),
        BracketMatchup("WEST", 8, "Villanova", 9, "Utah St."),
        BracketMatchup("WEST", 5, "Wisconsin", 12, "High Point"),
        BracketMatchup("WEST", 4, "Arkansas", 13, "Hawaii"),
        BracketMatchup("WEST", 6, "BYU", 11, "North Carolina St."),
        BracketMatchup("WEST", 3, "Gonzaga", 14, "Kennesaw St."),
        BracketMatchup("WEST", 7, "Miami FL", 10, "Missouri"),
        BracketMatchup("WEST", 2, "Purdue", 15, "Queens"),
        BracketMatchup("MIDWEST", 1, "Michigan", 16, "Howard"),
        BracketMatchup("MIDWEST", 8, "Georgia", 9, "Saint Louis"),
        BracketMatchup("MIDWEST", 5, "Texas Tech", 12, "Akron"),
        BracketMatchup("MIDWEST", 4, "Alabama", 13, "Hofstra"),
        BracketMatchup("MIDWEST", 6, "Tennessee", 11, "SMU"),
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
    rows = _read_csv_rows(path)
    parsed: list[tuple[int, int, int, str, int, float | None]] = []

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


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy_from_probs(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(int) == y))


def log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def impute_nan_with_col_means(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                m = np.nanmean(np.stack(vals, axis=0), axis=0)
            return np.where(np.isfinite(m), m, np.nan)

    return by_team_mean.get(team_n)


def prune_correlated_features_train_only(
    X_train: np.ndarray,
    feature_names: list[str],
    *,
    corr_threshold: float = 0.985,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if X_train.shape[1] <= 1:
        keep_idx = np.arange(X_train.shape[1])
        return X_train, keep_idx, feature_names, np.eye(X_train.shape[1], dtype=bool)

    X = np.asarray(X_train, dtype=np.float64)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    keep = np.ones(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        if not keep[i]:
            continue
        for j in range(i + 1, X.shape[1]):
            if keep[j] and abs(corr[i, j]) >= corr_threshold:
                keep[j] = False

    keep_idx = np.where(keep)[0]
    new_names = [feature_names[i] for i in keep_idx.tolist()]
    return X_train[:, keep_idx], keep_idx, new_names, corr


def _safe_value(x: np.ndarray, idx: int | None) -> float:
    if idx is None or idx < 0 or idx >= len(x):
        return 0.0
    v = x[idx]
    return float(v) if np.isfinite(v) else 0.0


def _make_feature_index(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _first_present(feature_map: dict[str, int], candidates: list[str]) -> int | None:
    for c in candidates:
        if c in feature_map:
            return feature_map[c]
    return None


def source_weight_for_feature(
    feature_name: str,
    *,
    kenpom_weight: float,
    zrating_weight: float,
    teamrankings_weight: float,
    other_source_weight: float,
) -> float:
    name_u = feature_name.upper()

    if "KENPOM BARTTORVIK::" in name_u:
        return kenpom_weight
    if "Z RATING TEAMS::" in name_u:
        return zrating_weight
    if "TEAMRANKINGS::" in name_u:
        return teamrankings_weight
    return other_source_weight

def build_matchup_feature_vector(
    va: np.ndarray,
    vb: np.ndarray,
    *,
    seed_a: float,
    seed_b: float,
    round_num: int,
    base_feature_names: list[str],
    base_feature_map: dict[str, int],
    kenpom_weight: float,
    zrating_weight: float,
    teamrankings_weight: float,
    other_source_weight: float,
) -> tuple[np.ndarray, list[str]]:
    raw_diff = va - vb
    raw_sum = va + vb
    seed_diff = float(seed_b - seed_a)

    source_weights = np.array(
        [
            source_weight_for_feature(
                name,
                kenpom_weight=kenpom_weight,
                zrating_weight=zrating_weight,
                teamrankings_weight=teamrankings_weight,
                other_source_weight=other_source_weight,
            )
            for name in base_feature_names
        ],
        dtype=np.float64,
    )

    diff = raw_diff * source_weights
    summ = raw_sum * source_weights

    diff_names = [f"{name} (DIFF)" for name in base_feature_names]
    sum_names = [f"{name} (SUM)" for name in base_feature_names]

    base_x = np.concatenate([diff, summ, np.array([seed_diff], dtype=np.float64)], axis=0)
    base_names = diff_names + sum_names + ["Calculated Seed Advantage"]

    idx_badj_em = _first_present(
        base_feature_map,
        ["KenPom Barttorvik::BADJ EM", "KenPom Barttorvik::KADJ EM"],
    )
    idx_wab = _first_present(base_feature_map, ["KenPom Barttorvik::WAB"])
    idx_z = _first_present(base_feature_map, ["Z Rating Teams::Z RATING"])
    idx_badj_o = _first_present(
        base_feature_map,
        ["KenPom Barttorvik::BADJ O", "KenPom Barttorvik::KADJ O"],
    )
    idx_badj_d = _first_present(
        base_feature_map,
        ["KenPom Barttorvik::BADJ D", "KenPom Barttorvik::KADJ D"],
    )
    idx_tr_rating = _first_present(base_feature_map, ["TeamRankings::TR RATING"])
    idx_sos = _first_present(
        base_feature_map,
        ["TeamRankings::SOS", "TeamRankings::SOS RATING", "KenPom Barttorvik::SOS"],
    )
    idx_win_pct = _first_present(
        base_feature_map,
        ["KenPom Barttorvik::WIN%", "TeamRankings::WIN %"],
    )

    badj_em_diff = _safe_value(diff, idx_badj_em)
    wab_diff = _safe_value(diff, idx_wab)
    z_diff = _safe_value(diff, idx_z)
    badj_o_diff = _safe_value(diff, idx_badj_o)
    badj_d_diff = _safe_value(diff, idx_badj_d)
    tr_rating_diff = _safe_value(diff, idx_tr_rating)
    sos_diff = _safe_value(diff, idx_sos)
    win_pct_diff = _safe_value(diff, idx_win_pct)

    # Make the hand-built interaction features also lean toward KenPom/Z data
    extra_x = np.array(
        [
            abs(seed_diff),
            abs(badj_em_diff),
            abs(wab_diff),
            abs(z_diff),
            seed_diff * seed_diff,
            badj_em_diff * seed_diff,
            wab_diff * seed_diff,
            z_diff * seed_diff,
            badj_o_diff - badj_d_diff,
            badj_em_diff + z_diff,
            badj_em_diff + wab_diff,
            tr_rating_diff * seed_diff,
            sos_diff * seed_diff,
            win_pct_diff * seed_diff,
            badj_em_diff * float(round_num),
            wab_diff * (1.0 if round_num == 64 else 0.0),
            z_diff * (1.0 if round_num <= 16 else 0.0),
            abs(badj_em_diff) * (1.0 if abs(seed_diff) <= 3 else 0.0),
            float(round_num),
            1.0 if round_num == 64 else 0.0,
            1.0 if round_num == 32 else 0.0,
            1.0 if round_num == 16 else 0.0,
            1.0 if round_num == 8 else 0.0,
            1.0 if round_num <= 16 else 0.0,
            1.0 if seed_a <= 4 else 0.0,
            1.0 if seed_b <= 4 else 0.0,
            1.0 if abs(seed_diff) <= 3 else 0.0,
            1.0 if seed_diff > 0 else 0.0,
        ],
        dtype=np.float64,
    )

    extra_names = [
        "ABS Seed Diff",
        "ABS BADJ EM Diff",
        "ABS WAB Diff",
        "ABS Z Rating Diff",
        "Seed Diff Squared",
        "BADJ EM Diff x Seed Diff",
        "WAB Diff x Seed Diff",
        "Z Rating Diff x Seed Diff",
        "OffenseMinusDefense Diff",
        "BADJ EM + Z Diff",
        "BADJ EM + WAB Diff",
        "TR Rating Diff x Seed Diff",
        "SOS Diff x Seed Diff",
        "WinPct Diff x Seed Diff",
        "BADJ EM Diff x Round Number",
        "WAB Diff x Is Round of 64",
        "Z Diff x Is Sweet16OrLater",
        "ABS BADJ EM Diff x Is Close Seed Matchup",
        "Round Number",
        "Is Round of 64",
        "Is Round of 32",
        "Is Sweet 16",
        "Is Elite 8",
        "Is Sweet16OrLater",
        "Team A Is Top4 Seed",
        "Team B Is Top4 Seed",
        "Is Close Seed Matchup",
        "Team A Better Seed",
    ]

    return np.concatenate([base_x, extra_x], axis=0), base_names + extra_names


def _predict_matchup_prob(
    team_a: str,
    seed_a: int,
    team_b: str,
    seed_b: int,
    year: int,
    round_num: int,
    by_year_team: dict[tuple[int, str], np.ndarray],
    by_team_mean: dict[str, np.ndarray],
    base_feature_names: list[str],
    base_feature_map: dict[str, int],
    mu: np.ndarray,
    sigma: np.ndarray,
    lr_model: LogisticRegressionCV,
    keep_idx: np.ndarray,
    kenpom_weight: float,
    zrating_weight: float,
    teamrankings_weight: float,
    other_source_weight: float,
) -> float:
    va = _vec_for_team_name(team_a, year, by_year_team, by_team_mean)
    vb = _vec_for_team_name(team_b, year, by_year_team, by_team_mean)

    base_len = len(base_feature_names)
    if va is None:
        va = np.zeros(base_len, dtype=np.float64)
    if vb is None:
        vb = np.zeros(base_len, dtype=np.float64)

    x, _ = build_matchup_feature_vector(
        va,
        vb,
        seed_a=seed_a,
        seed_b=seed_b,
        round_num=round_num,
        base_feature_names=base_feature_names,
        base_feature_map=base_feature_map,
        kenpom_weight=kenpom_weight,
        zrating_weight=zrating_weight,
        teamrankings_weight=teamrankings_weight,
        other_source_weight=other_source_weight,
    )
    x = np.where(np.isnan(x), 0.0, x)
    x = x[keep_idx]
    xs = standardize_apply(x[None, :], mu, sigma)
    return float(lr_model.predict_proba(xs)[0, 1])


# ==========================================
# MAIN EXECUTION
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure logistic-regression March Madness model.")
    parser.add_argument("--target_year", type=int, default=2026, help="Year to predict (default: 2026).")
    parser.add_argument("--predict_round", type=int, default=64, help="Round to predict (default: 64).")
    parser.add_argument("--train_start_year", type=int, default=2008, help="First training year (inclusive).")
    parser.add_argument("--train_end_year", type=int, default=2024, help="Last pure-training year (inclusive).")
    parser.add_argument("--val_year", type=int, default=2025, help="Validation year for tuning.")
    parser.add_argument("--test_year", type=int, default=0, help="True holdout year. Set to 0 to skip holdout evaluation.")
    parser.add_argument("--max_missing_frac", type=float, default=0.60, help="Drop features missing above this fraction.")
    parser.add_argument("--corr_prune_threshold", type=float, default=0.985, help="Drop train-only highly correlated features above this absolute correlation.")
    parser.add_argument("--use_feature_whitelist", action="store_true", help="Keep only selected metric families before matchup construction.")
    parser.add_argument("--use_weights", action="store_true", help="Use custom sample weights.")
    parser.add_argument("--penalty", choices=["l1", "l2"], default="l2", help="Logistic penalty.")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting.")
    parser.add_argument("--kenpom_weight", type=float, default=1.75, help="Weight multiplier for KenPom/Barttorvik features.")
    parser.add_argument("--zrating_weight", type=float, default=1.20, help="Weight multiplier for Z Rating features.")
    parser.add_argument("--teamrankings_weight", type=float, default=0.60, help="Weight multiplier for TeamRankings features.")
    parser.add_argument("--other_source_weight", type=float, default=0.35, help="Weight multiplier for other feature sources.")
    args = parser.parse_args()

    if args.val_year != args.train_end_year + 1:
        raise SystemExit("--val_year must be exactly one year after --train_end_year.")
    if args.test_year != 0 and args.test_year <= args.val_year:
        raise SystemExit("--test_year must be after --val_year for a clean split.")

    train_years = list(range(args.train_start_year, args.train_end_year + 1))
    val_years = [args.val_year]
    test_years = [args.test_year] if args.test_year != 0 else []

    z_path = DATASET_DIR / "Z Rating Teams.csv"
    tm_path = DATASET_DIR / "Tournament Matchups.csv"

    games = load_games_from_tournament_matchups(tm_path)

    last_feature_year = max(
        args.train_end_year,
        args.val_year,
        args.target_year,
        args.test_year if args.test_year != 0 else args.target_year,
    )
    all_years = list(range(args.train_start_year, last_feature_year + 1))

    print("Loading features...")
    print(f"Train years:      {train_years}")
    print(f"Validation year:  {val_years}")
    print(f"Test year:        {test_years if test_years else 'SKIPPED'}")
    print(f"Target year:      {args.target_year}")

    by_year_team, by_team_mean, base_feature_names = load_team_feature_store(
        DATASET_DIR, years=all_years
    )
    by_year_team, by_team_mean, base_feature_names = drop_high_missing_features(
        by_year_team,
        by_team_mean,
        base_feature_names,
        max_missing_frac=args.max_missing_frac,
    )

    if args.use_feature_whitelist:
        allowed_metric_substrings = [
            "Z RATING",
            "SEED",
            "BADJ EM",
            "KADJ EM",
            "BADJ O",
            "KADJ O",
            "BADJ D",
            "KADJ D",
            "WAB",
            "TR RATING",
            "SOS",
            "WIN%",
            "V 1-25 WINS",
            "TOP 25",
            "Q1",
            "Q2",
            "3PT",
            "3PTR",
            "3PTD",
            "TOV",
            "OREB",
            "DREB",
            "REB",
            "FTR",
            "PPPO",
            "TALENT",
        ]
        by_year_team, by_team_mean, base_feature_names = keep_only_selected_base_features(
            by_year_team,
            by_team_mean,
            base_feature_names,
            allowed_substrings=allowed_metric_substrings,
        )
        print(f"Whitelist enabled. Base feature count: {len(base_feature_names)}")

    z_seeds = load_z_ratings(z_path)
    base_feature_map = _make_feature_index(base_feature_names)

    def build_matrix(target_years: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        year_list: list[int] = []
        w_list: list[float] = []
        feature_names_out: list[str] | None = None

        min_year = min(train_years) if train_years else 2008

        for g in games:
            if g.year not in target_years:
                continue
            if g.score_a is None or g.score_b is None or g.score_a == g.score_b:
                continue

            va = _vec_for_team_name(g.team_a, g.year, by_year_team, by_team_mean)
            vb = _vec_for_team_name(g.team_b, g.year, by_year_team, by_team_mean)
            if va is None or vb is None:
                continue

            fa = z_seeds.get((g.year, g.team_no_a))
            fb = z_seeds.get((g.year, g.team_no_b))
            seed_a = fa.seed if fa is not None else 8.0
            seed_b = fb.seed if fb is not None else 8.0

            x_ab, names = build_matchup_feature_vector(
                va,
                vb,
                seed_a=seed_a,
                seed_b=seed_b,
                round_num=g.round_num,
                base_feature_names=base_feature_names,
                base_feature_map=base_feature_map,
                kenpom_weight=args.kenpom_weight,
                zrating_weight=args.zrating_weight,
                teamrankings_weight=args.teamrankings_weight,
                other_source_weight=args.other_source_weight,
            )
            y_ab = 1 if g.score_a > g.score_b else 0

            x_ba, _ = build_matchup_feature_vector(
                vb,
                va,
                seed_a=seed_b,
                seed_b=seed_a,
                round_num=g.round_num,
                base_feature_names=base_feature_names,
                base_feature_map=base_feature_map,
                kenpom_weight=args.kenpom_weight,
                zrating_weight=args.zrating_weight,
                teamrankings_weight=args.teamrankings_weight,
                other_source_weight=args.other_source_weight,
            )
            y_ba = 1 - y_ab

            if feature_names_out is None:
                feature_names_out = names

            if args.use_weights:
                weight = 1.0
                weight += (g.year - min_year) * 0.05
                if abs(seed_b - seed_a) <= 3:
                    weight += 0.25
                if g.round_num <= 16:
                    weight += 0.15
            else:
                weight = 1.0

            X_list.append(x_ab)
            y_list.append(y_ab)
            year_list.append(g.year)
            w_list.append(weight)

            X_list.append(x_ba)
            y_list.append(y_ba)
            year_list.append(g.year)
            w_list.append(weight)

        if not X_list:
            return (
                np.empty((0, 0)),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                [],
            )

        return (
            np.stack(X_list, axis=0),
            np.array(y_list, dtype=np.int32),
            np.array(year_list, dtype=np.int32),
            np.array(w_list, dtype=np.float32),
            feature_names_out if feature_names_out is not None else [],
        )

    X_train, y_train, groups_train, w_train, full_feature_names = build_matrix(train_years)
    X_val, y_val, groups_val, w_val, _ = build_matrix(val_years)

    if args.test_year != 0:
        X_test, y_test, groups_test, w_test, _ = build_matrix(test_years)
    else:
        X_test = np.empty((0, X_train.shape[1]))
        y_test = np.empty(0, dtype=np.int32)

    if len(X_train) < 100:
        raise SystemExit(f"Not enough training examples found ({len(X_train)}).")
    if len(X_val) == 0:
        raise SystemExit(f"No validation games found for {args.val_year}.")
    if args.test_year != 0 and len(X_test) == 0:
        raise SystemExit(f"No test games found for {args.test_year}.")

    X_train, col_means = impute_nan_with_col_means(X_train)
    X_val = np.where(np.isnan(X_val), col_means[None, :], X_val)
    if args.test_year != 0:
        X_test = np.where(np.isnan(X_test), col_means[None, :], X_test)

    print("\n--- Correlation Pruning ---")
    X_train_pruned, keep_idx, pruned_feature_names, _ = prune_correlated_features_train_only(
        X_train,
        full_feature_names,
        corr_threshold=args.corr_prune_threshold,
    )
    X_val_pruned = X_val[:, keep_idx]
    if args.test_year != 0:
        X_test_pruned = X_test[:, keep_idx]
    else:
        X_test_pruned = np.empty((0, len(keep_idx)))

    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Pruned feature count:   {X_train_pruned.shape[1]}")

    mu, sigma = standardize_fit(X_train_pruned)
    Xs_train = standardize_apply(X_train_pruned, mu, sigma)
    Xs_val = standardize_apply(X_val_pruned, mu, sigma)
    if args.test_year != 0:
        Xs_test = standardize_apply(X_test_pruned, mu, sigma)
    else:
        Xs_test = np.empty((0, Xs_train.shape[1]))

    print("\n--- Training Pure Logistic Regression (Group-by-Year CV) ---")
    unique_train_years = np.unique(groups_train)
    if len(unique_train_years) < 3:
        raise SystemExit("Need at least 3 distinct training years for GroupKFold tuning.")

    gkf = GroupKFold(n_splits=len(unique_train_years))
    lr_cv_splits = list(gkf.split(Xs_train, y_train, groups_train))

    if args.penalty == "l1":
        solver = "saga"
        max_iter = 5000
        n_jobs = -1
    else:
        solver = "liblinear"
        max_iter = 3000
        n_jobs = None

    lr_model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 30),
        cv=lr_cv_splits,
        penalty=args.penalty,
        solver=solver,
        scoring="neg_brier_score",
        max_iter=max_iter,
        random_state=42,
        refit=True,
        n_jobs=n_jobs,
    )
    lr_model.fit(Xs_train, y_train, sample_weight=w_train)

    p_train = lr_model.predict_proba(Xs_train)[:, 1]
    p_val = lr_model.predict_proba(Xs_val)[:, 1]
    if args.test_year != 0:
        p_test = lr_model.predict_proba(Xs_test)[:, 1]

    print(f"\nTrained on {len(y_train)} examples.")
    print(f"Validation examples: {len(y_val)}")
    if args.test_year != 0:
        print(f"Test examples:       {len(y_test)}")

    chosen_c = float(lr_model.C_[0]) if hasattr(lr_model, "C_") else float("nan")
    print(f"Chosen regularization C: {chosen_c:.6f}")
    print(f"Penalty: {args.penalty}")
    print(f"Whitelist enabled: {args.use_feature_whitelist}")
    print(f"Custom weights enabled: {args.use_weights}")

    print("\n--- TRAIN METRICS ---")
    print(f"Train Accuracy: {accuracy_from_probs(p_train, y_train)*100:.2f}%")
    print(f"Train Brier:    {brier_score(p_train, y_train):.4f}")
    print(f"Train LogLoss:  {log_loss_binary(p_train, y_train):.4f}")

    print("\n--- VALIDATION METRICS ---")
    print(f"Val Accuracy: {accuracy_from_probs(p_val, y_val)*100:.2f}%")
    print(f"Val Brier:    {brier_score(p_val, y_val):.4f}")
    print(f"Val LogLoss:  {log_loss_binary(p_val, y_val):.4f}")

    if args.test_year != 0:
        print("\n--- TEST METRICS (TRUE HOLDOUT) ---")
        print(f"Test Accuracy: {accuracy_from_probs(p_test, y_test)*100:.2f}%")
        print(f"Test Brier:    {brier_score(p_test, y_test):.4f}")
        print(f"Test LogLoss:  {log_loss_binary(p_test, y_test):.4f}")

    coefs = lr_model.coef_.ravel()
    analyzer = FeatureAnalyzer(pruned_feature_names, coefs)
    analyzer.print_report(min_relative_importance=0.5)

    initial = bracket_2026_round_of_64()

    def _run_region(region_name: str, matchups: list[BracketMatchup]) -> None:
        nonlocal all_games, region_champions

        cur_games: list[tuple[str, int, str, int]] = [
            (m.team_a, m.seed_a, m.team_b, m.seed_b) for m in matchups
        ]
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
        round_idx = 0
        round_to_num = {"Round of 64": 64, "Round of 32": 32, "Sweet 16": 16, "Elite 8": 8}

        while True:
            label = round_labels[round_idx] if round_idx < len(round_labels) else f"Round {round_idx + 1}"
            round_num = round_to_num.get(label, 4)
            winners: list[tuple[str, int]] = []

            for team_a, seed_a, team_b, seed_b in cur_games:
                p_a = _predict_matchup_prob(
                    team_a,
                    seed_a,
                    team_b,
                    seed_b,
                    args.target_year,
                    round_num,
                    by_year_team,
                    by_team_mean,
                    base_feature_names,
                    base_feature_map,
                    mu,
                    sigma,
                    lr_model,
                    keep_idx,
                    kenpom_weight=args.kenpom_weight,
                    zrating_weight=args.zrating_weight,
                    teamrankings_weight=args.teamrankings_weight,
                    other_source_weight=args.other_source_weight,
                )
                winner, w_seed = (team_a, seed_a) if p_a >= 0.5 else (team_b, seed_b)
                all_games.append((label, region_name, team_a, team_b, p_a, winner))
                winners.append((winner, w_seed))

            if len(winners) == 1:
                region_champions[region_name] = winners[0]
                break

            next_games: list[tuple[str, int, str, int]] = []
            for i in range(0, len(winners), 2):
                ta, sa = winners[i]
                tb, sb = winners[i + 1]
                next_games.append((ta, sa, tb, sb))

            cur_games = next_games
            round_idx += 1

    all_games: list[tuple[str, str, str, str, float, str]] = []
    region_champions: dict[str, tuple[str, int]] = {}
    by_region: dict[str, list[BracketMatchup]] = {}

    for m in initial:
        by_region.setdefault(m.region, []).append(m)

    for region_name, matchups in sorted(by_region.items()):
        _run_region(region_name, matchups)

    def _ff_game(region1: str, region2: str, slot_name: str) -> str:
        nonlocal all_games
        team_a, seed_a = region_champions[region1]
        team_b, seed_b = region_champions[region2]
        p_a = _predict_matchup_prob(
            team_a,
            seed_a,
            team_b,
            seed_b,
            args.target_year,
            4,
            by_year_team,
            by_team_mean,
            base_feature_names,
            base_feature_map,
            mu,
            sigma,
            lr_model,
            keep_idx,
            kenpom_weight=args.kenpom_weight,
            zrating_weight=args.zrating_weight,
            teamrankings_weight=args.teamrankings_weight,
            other_source_weight=args.other_source_weight,
        )
        winner, _ = (team_a, seed_a) if p_a >= 0.5 else (team_b, seed_b)
        all_games.append(("Final Four", slot_name, team_a, team_b, p_a, winner))
        return winner

    east_south_winner = _ff_game("EAST", "SOUTH", "EAST/SOUTH")
    west_midwest_winner = _ff_game("WEST", "MIDWEST", "WEST/MIDWEST")

    def _find_seed(team_name: str) -> int:
        for _, (t, s) in region_champions.items():
            if t == team_name:
                return s
        return 1

    seed_a = _find_seed(east_south_winner)
    seed_b = _find_seed(west_midwest_winner)

    p_a_final = _predict_matchup_prob(
        east_south_winner,
        seed_a,
        west_midwest_winner,
        seed_b,
        args.target_year,
        2,
        by_year_team,
        by_team_mean,
        base_feature_names,
        base_feature_map,
        mu,
        sigma,
        lr_model,
        keep_idx,
        kenpom_weight=args.kenpom_weight,
        zrating_weight=args.zrating_weight,
        teamrankings_weight=args.teamrankings_weight,
        other_source_weight=args.other_source_weight,
    )

    champ_winner = east_south_winner if p_a_final >= 0.5 else west_midwest_winner
    all_games.append(
        ("Championship", "NATIONAL", east_south_winner, west_midwest_winner, p_a_final, champ_winner)
    )

    round_order = {
        "Round of 64": 1,
        "Round of 32": 2,
        "Sweet 16": 3,
        "Elite 8": 4,
        "Final Four": 5,
        "Championship": 6,
    }
    all_games.sort(key=lambda t: (round_order.get(t[0], 99), t[1], -abs(t[4] - 0.5), t[2]))

    print("\n=======================================================")
    print(f"   PREDICTING THE FULL {args.target_year} TOURNAMENT")
    print("=======================================================")

    cur_round = None
    cur_region = None
    for round_label, region, team_a, team_b, p_a, winner in all_games:
        if round_label != cur_round:
            cur_round = round_label
            cur_region = None
            print(f"\n=== {round_label} ===")
        if region != cur_region:
            cur_region = region
            print(f"\n[{region}]")
        print(f"- {team_a} vs {team_b}: {100.0*p_a:5.1f}% / {100.0*(1.0-p_a):5.1f}% (winner: {winner})")

    if not args.no_plot:
        coef_abs = np.abs(coefs)
        top_k = min(25, len(coef_abs))
        top_idx = np.argsort(-coef_abs)[:top_k][::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_k), coef_abs[top_idx])
        plt.yticks(range(top_k), [pruned_feature_names[i] for i in top_idx])
        plt.xlabel("Absolute Coefficient Magnitude")
        plt.title("Top Logistic Regression Features")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
