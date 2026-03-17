from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

# ==========================================
# FEATURE ANALYZER CLASSES
# ==========================================
@dataclass
class FeatureImportance:
    name: str
    relative_importance: float  # Percentage of total importance

class FeatureAnalyzer:
    """
    Analyzes the learned feature importances of a tree-based model.
    """
    def __init__(self, feature_names: list[str], importances: np.ndarray):
        self.feature_names = list(feature_names)
        if len(importances) == len(self.feature_names) + 1:
            self.feature_names.append("Calculated Seed Advantage")
            
        self.importances = importances

    def get_significant_features(self, min_relative_importance: float = 1.0) -> list[FeatureImportance]:
        total_importance = np.sum(self.importances)
        
        importances_list = []
        for name, imp in zip(self.feature_names, self.importances):
            rel_imp = (imp / total_importance) * 100 if total_importance > 0 else 0.0
            importances_list.append(FeatureImportance(name, rel_imp))
            
        importances_list.sort(key=lambda x: x.relative_importance, reverse=True)
        return [f for f in importances_list if f.relative_importance >= min_relative_importance]

    def print_report(self, min_relative_importance: float = 1.0) -> None:
        significant = self.get_significant_features(min_relative_importance)
        
        print("\n=======================================================")
        print(f"   SIGNIFICANT FEATURES (Threshold: >= {min_relative_importance}%)")
        print("=======================================================")
        print(f"{'Feature Name':<50} | {'Importance':<10}")
        print("-" * 65)
        
        cumulative = 0.0
        for f in significant:
            print(f"{f.name:<50} | {f.relative_importance:>5.2f}%")
            cumulative += f.relative_importance
            
        print("-" * 65)
        print(f"Total features kept: {len(significant)} out of {len(self.feature_names)}")
        print(f"These account for {cumulative:.1f}% of the model's decision-making power.\n")


# ==========================================
# EXISTING HELPER FUNCTIONS
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

    # Explicit leak/outcome columns
    if c in {
        "SCORE", "ROUND", "CURRENT ROUND", "CHAMP", "CHAMP%",
        "F2", "F4", "F4%", "E8", "S16", "R32", "R64", "TOP2",
        "SIM", "SIMS", "PICK", "PUBLIC", "WINS",
        "SEED",
        # Season record — directly encodes team strength in a target-leaky way
        "W", "L", "WIN%", "GAMES",
        # ID/lookup columns, not predictive features
        "QUAD NO", "QUAD ID", "TEAM ID",
        "HEAT CHECK TOURNAMENT INDEX::WINS",
        "HEAT CHECK TOURNAMENT INDEX::POOL VALUE",
        "HEAT CHECK TOURNAMENT INDEX::VAL Z-SCORE",
        "HEAT CHECK TOURNAMENT INDEX::POWER-PATH",
    }:
        return True

    # Block ALL rank columns regardless of prefix format
    # Catches: "KADJ EM RANK", "KO RANK", "KD RANK", "BADJT RANK", etc.
    if "RANK" in c:
        return True

    return False

def _list_feature_csvs(dataset_dir: Path) -> list[Path]:
    exclude = {
        "Tournament Matchups.csv", "Tournament Simulation.csv", "Tournament Locations.csv",
        "Seed Results.csv", "Upset Count.csv", "Upset Seed Info.csv", "Public Picks.csv",
        "Coach Results.csv", "Conference Results.csv", "Team Results.csv",
    }
    paths = []
    for p in sorted(dataset_dir.glob("*.csv")):
        if p.name in exclude:
            continue
        paths.append(p)
    return paths

def load_team_feature_store(
    dataset_dir: Path, *, years: list[int], hist_years: list[int]
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

    by_year_team: dict[tuple[int, str], np.ndarray] = {}
    def _safe_nanmean(M: np.ndarray, axis: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(M, axis=axis)
        return np.where(np.isfinite(m), m, np.nan)

    for k, mats in acc.items():
        M = np.stack(mats, axis=0)
        by_year_team[k] = _safe_nanmean(M, axis=0)

    # Build historical means using ONLY the specified hist_years to avoid
    # leaking future seasons into earlier training examples.
    team_to_year_vecs: dict[str, list[np.ndarray]] = {}
    for (y, team), v in by_year_team.items():
        if y in hist_years:
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
        BracketMatchup("EAST", 2, "UConn", 15, "Furman"),
        BracketMatchup("SOUTH", 1, "Florida", 16, "Lehigh/PVAMU"),
        BracketMatchup("SOUTH", 8, "Clemson", 9, "Iowa"),
        BracketMatchup("SOUTH", 5, "Vanderbilt", 12, "McNeese St."),
        BracketMatchup("SOUTH", 4, "Nebraska", 13, "Troy"),
        BracketMatchup("SOUTH", 6, "North Carolina", 11, "VCU"),
        BracketMatchup("SOUTH", 3, "Illinois", 14, "Penn"),
        BracketMatchup("SOUTH", 7, "Saint Mary's", 10, "Texas A&M"),
        BracketMatchup("SOUTH", 2, "Houston", 15, "Idaho"),
        BracketMatchup("WEST", 1, "Arizona", 16, "Long Island"),
        BracketMatchup("WEST", 8, "Villanova", 9, "Utah St."),
        BracketMatchup("WEST", 5, "Wisconsin", 12, "High Point"),
        BracketMatchup("WEST", 4, "Arkansas", 13, "Hawaii"),
        BracketMatchup("WEST", 6, "BYU", 11, "NC St./Texas"),
        BracketMatchup("WEST", 3, "Gonzaga", 14, "Kennesaw St."),
        BracketMatchup("WEST", 7, "Miami (FL)", 10, "Missouri"),
        BracketMatchup("WEST", 2, "Purdue", 15, "Queens (N.C.)"),
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
                year=year, round_num=rnd, team_a=team_a, team_b=team_b,
                team_no_a=team_no_a, team_no_b=team_no_b, score_a=score_a, score_b=score_b,
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

def impute_nan_with_col_means(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    X2 = np.where(np.isnan(X), col_means[None, :], X)
    return X2, col_means

def _vec_for_team_name(
    team: str, year: int, by_year_team: dict[tuple[int, str], np.ndarray], by_team_mean: dict[str, np.ndarray],
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

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def _predict_matchup_prob(
    team_a: str, seed_a: int, team_b: str, seed_b: int, year: int,
    by_year_team: dict[tuple[int, str], np.ndarray], by_team_mean: dict[str, np.ndarray],
    feature_names: list[str], mu: np.ndarray, sigma: np.ndarray, model: RandomForestClassifier,
) -> float:
    va = _vec_for_team_name(team_a, year, by_year_team, by_team_mean)
    vb = _vec_for_team_name(team_b, year, by_year_team, by_team_mean)
    if va is None:
        va = np.zeros(len(feature_names), dtype=np.float64)
    if vb is None:
        vb = np.zeros(len(feature_names), dtype=np.float64)

    x = np.concatenate([va - vb, np.array([float(seed_b - seed_a)], dtype=np.float64)], axis=0)
    x = np.where(np.isnan(x), 0.0, x)
    xs = standardize_apply(x[None, :], mu, sigma)
    
    return float(model.predict_proba(xs)[0, 1])

def boost_priority_features(
    by_year_team: dict[tuple[int, str], np.ndarray],
    by_team_hist_mean: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    priority_keywords: list[str],
    multiplier: int = 3,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, np.ndarray], list[str]]:
    """
    Duplicates high-signal features `multiplier` times so tree splits
    favour them proportionally more during random feature sampling.
    """
    boost_idx = [
        i for i, name in enumerate(feature_names)
        if any(kw.lower() in name.lower() for kw in priority_keywords)
    ]
    if not boost_idx:
        print("⚠️  No priority features found to boost — check keywords.")
        return by_year_team, by_team_hist_mean, feature_names

    extra_names = []
    for _ in range(multiplier - 1):
        extra_names += [f"{feature_names[i]}__boosted" for i in boost_idx]

    new_names = feature_names + extra_names

    def _extend(vec: np.ndarray) -> np.ndarray:
        extras = np.concatenate([vec[boost_idx]] * (multiplier - 1))
        return np.concatenate([vec, extras])

    new_by_year  = {k: _extend(v) for k, v in by_year_team.items()}
    new_by_team  = {k: _extend(v) for k, v in by_team_hist_mean.items()}

    print(f"✅ Boosted {len(boost_idx)} features x{multiplier}: "
          f"{[feature_names[i] for i in boost_idx[:5]]}{'...' if len(boost_idx) > 5 else ''}")

    return new_by_year, new_by_team, new_names

# ==========================================
# MAIN EXECUTION
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Random Forest win-probability model for March Madness.")
    parser.add_argument("--target_year", type=int, default=2026)
    parser.add_argument("--predict_round", type=int, default=64)
    parser.add_argument("--train_start_year", type=int, default=2018)
    parser.add_argument("--train_end_year", type=int, default=2025)
    parser.add_argument("--test_year", type=int, default=2025)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    if args.train_start_year < 2018:
        print("⚠️ Override: Forcing train_start_year to 2018.")
        args.train_start_year = 2018

    z_path = DATASET_DIR / "Z Rating Teams.csv"
    tm_path = DATASET_DIR / "Tournament Matchups.csv"
    games = load_games_from_tournament_matchups(tm_path)

    all_years = list(range(args.train_start_year, args.train_end_year + 1))
    train_years = [y for y in all_years if y != args.test_year]
    test_years = [args.test_year]

    print(f"Loading features...")
    print(f"Training on years: {train_years}")
    print(f"Testing on year:   {test_years}")

    by_year_team, by_team_mean, feature_names = load_team_feature_store(
        DATASET_DIR, years=all_years, hist_years=train_years
    )
    by_year_team, by_team_mean, feature_names = drop_high_missing_features(
        by_year_team, by_team_mean, feature_names, max_missing_frac=0.60
    )
    by_year_team, by_team_mean, feature_names = boost_priority_features(
        by_year_team, by_team_mean, feature_names,
        priority_keywords=["KenPom", "Barttorvik", "KADJ", "BADJ"],
        multiplier=3,
    )
    z_seeds = load_z_ratings(z_path)

    # ---- build_matrix with mirror augmentation ----
    def build_matrix(target_years, augment: bool = False):
        X_list, y_list = [], []
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
            seed_adv = float(fb.seed - fa.seed) if (fa and fb) else 0.0
            diff = va - vb
            x  = np.concatenate([diff,  np.array([ seed_adv], dtype=np.float64)])
            y  = 1 if g.score_a > g.score_b else 0
            X_list.append(x)
            y_list.append(y)
            if augment:
                # Mirror: swap team order — negate diff & seed_adv, flip label
                x_mir = np.concatenate([-diff, np.array([-seed_adv], dtype=np.float64)])
                X_list.append(x_mir)
                y_list.append(1 - y)
        if not X_list:
            return np.empty((0, len(feature_names) + 1)), np.empty(0)
        return np.stack(X_list), np.array(y_list, dtype=np.int32)

    # ---- 1. Tune hyperparams on augmented training data ----
    X_train, y_train = build_matrix(train_years, augment=True)
    if len(X_train) < 50:
        raise SystemExit(f"Not enough training games ({len(X_train)}).")

    X_train, col_means = impute_nan_with_col_means(X_train)
    mu, sigma = standardize_fit(X_train)
    Xs_train = standardize_apply(X_train, mu, sigma)

    print("\nStarting Hyperparameter Tuning...")
    param_grid = {
        'n_estimators':      [100, 150, 200],
        'max_depth':         [4, 5, 6],
        'min_samples_split': [8, 12, 16],
        'min_samples_leaf':  [4, 6, 8],
        'max_features':      ['sqrt', 0.3, 0.4],
        'max_samples':       [0.6, 0.7, 0.8],
        'bootstrap':         [True],
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=150, scoring='neg_brier_score',
        cv=5, verbose=1, random_state=42, n_jobs=-1
    )
    random_search.fit(Xs_train, y_train)
    best_params = random_search.best_params_
    print(f"\n--- TUNING COMPLETE ---")
    print(f"Best Parameters: {best_params}")

    model = random_search.best_estimator_
    p_train = model.predict_proba(Xs_train)[:, 1]
    train_acc = np.mean((p_train >= 0.5).astype(int) == y_train)
    print(f"\nTrained on {len(y_train)} games (with mirroring).")
    print(f"Training Accuracy: {train_acc*100:.2f}% | Brier: {brier_score(p_train, y_train):.4f}")

    # ---- 2. Feature importance report ----
    analyzer = FeatureAnalyzer(feature_names, model.feature_importances_)
    analyzer.print_report(min_relative_importance=0.5)

    # ---- 3. Single held-out year test ----
    X_test, y_test = build_matrix(test_years, augment=False)  # no augment on test
    if len(y_test) > 0:
        X_test_imp = np.where(np.isnan(X_test), col_means[None, :], X_test)
        Xs_test = standardize_apply(X_test_imp, mu, sigma)
        p_test = model.predict_proba(Xs_test)[:, 1]
        test_acc = np.mean((p_test >= 0.5).astype(int) == y_test)
        print(f"Held-out {args.test_year}: {len(y_test)} games | "
              f"Accuracy: {test_acc*100:.2f}% | Brier: {brier_score(p_test, y_test):.4f}\n")

    # ---- 4. Leave-One-Year-Out CV for reliable accuracy estimate ----
    print("=" * 55)
    print("   LEAVE-ONE-YEAR-OUT CROSS VALIDATION")
    print("=" * 55)
    loyo_accs = []
    for test_yr in sorted(all_years):
        tr_yrs = [y for y in all_years if y != test_yr]
        X_tr, y_tr = build_matrix(tr_yrs, augment=True)
        X_te, y_te = build_matrix([test_yr], augment=False)
        if len(y_te) == 0:
            continue
        X_tr, cm = impute_nan_with_col_means(X_tr)
        mu_l, sigma_l = standardize_fit(X_tr)
        Xs_tr = standardize_apply(X_tr, mu_l, sigma_l)
        X_te = np.where(np.isnan(X_te), cm[None, :], X_te)
        Xs_te = standardize_apply(X_te, mu_l, sigma_l)
        m_l = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        m_l.fit(Xs_tr, y_tr)
        acc = np.mean((m_l.predict_proba(Xs_te)[:, 1] >= 0.5).astype(int) == y_te)
        loyo_accs.append(acc)
        print(f"  LOYO {test_yr}: {acc*100:.1f}%  ({len(y_te)} games)")
    print(f"\n  LOYO Mean Accuracy: {np.mean(loyo_accs)*100:.1f}%")
    print("=" * 55 + "\n")

    # ---- 5. Learning curve plot ----
    X_test_lc_raw, y_test_lc = build_matrix(test_years, augment=False)
    X_test_lc = np.where(np.isnan(X_test_lc_raw), col_means[None, :], X_test_lc_raw)
    Xs_test_lc = standardize_apply(X_test_lc, mu, sigma)
    estimator_range = list(range(10, best_params.get('n_estimators', 200) + 1, 10))
    train_accs_lc, test_accs_lc = [], []
    lc_model = RandomForestClassifier(
        **{k: v for k, v in best_params.items() if k != 'n_estimators'},
        warm_start=True, random_state=42, n_jobs=-1
    )
    for n in estimator_range:
        lc_model.n_estimators = n
        lc_model.fit(Xs_train, y_train)
        train_accs_lc.append(np.mean((lc_model.predict_proba(Xs_train)[:, 1] >= 0.5).astype(int) == y_train))
        if len(y_test_lc) > 0:
            test_accs_lc.append(np.mean((lc_model.predict_proba(Xs_test_lc)[:, 1] >= 0.5).astype(int) == y_test_lc))

    if not args.no_plot:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(estimator_range, [a * 100 for a in train_accs_lc], label="Train Accuracy", color="#1f77b4", linewidth=2)
        if test_accs_lc:
            ax2.plot(estimator_range, [a * 100 for a in test_accs_lc], label=f"Test Accuracy ({args.test_year})", color="#d62728", linewidth=2)
        ax2.axvline(best_params.get('n_estimators', 200), color="gray", linestyle="--",
                    label=f"Best n_estimators={best_params.get('n_estimators', 200)}")
        ax2.set_xlabel("Number of Trees (n_estimators)")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Random Forest: n_estimators vs Train/Test Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ---- 6. Retrain on ALL years including test_year for final 2026 predictions ----
    print(f"Retraining on ALL years {all_years} for final {args.target_year} predictions...")
    X_full, y_full = build_matrix(all_years, augment=True)
    X_full, col_means_full = impute_nan_with_col_means(X_full)
    mu_full, sigma_full = standardize_fit(X_full)
    Xs_full = standardize_apply(X_full, mu_full, sigma_full)
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(Xs_full, y_full)
    p_full = final_model.predict_proba(Xs_full)[:, 1]
    print(f"Final model trained on {len(y_full)} games (with mirroring).")
    print(f"Full-data Train Accuracy: {np.mean((p_full >= 0.5).astype(int) == y_full)*100:.2f}%\n")

    # ---- 7. Predict the 2026 Tournament using final_model ----
    initial = bracket_2026_round_of_64()

    def _predict_final(team_a, seed_a, team_b, seed_b):
        return _predict_matchup_prob(
            team_a, seed_a, team_b, seed_b, args.target_year,
            by_year_team, by_team_mean, feature_names,
            mu_full, sigma_full, final_model,
        )

    def _run_region(region_name: str, matchups: list[BracketMatchup]):
        nonlocal all_games, region_champions
        cur_games = [(m.team_a, m.seed_a, m.team_b, m.seed_b) for m in matchups]
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
        round_idx = 0
        while True:
            label = round_labels[round_idx] if round_idx < len(round_labels) else f"Round {round_idx+1}"
            winners = []
            for team_a, seed_a, team_b, seed_b in cur_games:
                p_a = _predict_final(team_a, seed_a, team_b, seed_b)
                winner, w_seed = (team_a, seed_a) if p_a >= 0.5 else (team_b, seed_b)
                all_games.append((label, region_name, team_a, team_b, p_a, winner))
                winners.append((winner, w_seed))
            if len(winners) == 1:
                region_champions[region_name] = winners[0]
                break
            cur_games = [(winners[i][0], winners[i][1], winners[i+1][0], winners[i+1][1])
                         for i in range(0, len(winners), 2)]
            round_idx += 1

    all_games: list[tuple[str, str, str, str, float, str]] = []
    region_champions: dict[str, tuple[str, int]] = {}
    by_region: dict[str, list[BracketMatchup]] = {}
    for m in initial:
        by_region.setdefault(m.region, []).append(m)
    for region_name, matchups in sorted(by_region.items()):
        _run_region(region_name, matchups)

    def _ff_game(region1, region2, slot_name):
        team_a, seed_a = region_champions[region1]
        team_b, seed_b = region_champions[region2]
        p_a = _predict_final(team_a, seed_a, team_b, seed_b)
        winner = team_a if p_a >= 0.5 else team_b
        all_games.append(("Final Four", slot_name, team_a, team_b, p_a, winner))
        return winner

    ew = _ff_game("EAST", "WEST", "EAST/WEST")
    sm = _ff_game("SOUTH", "MIDWEST", "SOUTH/MIDWEST")

    def _find_seed(name):
        for _, (t, s) in region_champions.items():
            if t == name:
                return s
        return 1

    p_final = _predict_final(ew, _find_seed(ew), sm, _find_seed(sm))
    champ = ew if p_final >= 0.5 else sm
    all_games.append(("Championship", "NATIONAL", ew, sm, p_final, champ))

    round_order = {"Round of 64": 1, "Round of 32": 2, "Sweet 16": 3,
                   "Elite 8": 4, "Final Four": 5, "Championship": 6}
    all_games.sort(key=lambda t: (round_order.get(t[0], 99), t[1], -abs(t[4] - 0.5), t[2]))

    print(f"=======================================================")
    print(f"   PREDICTING THE FULL {args.target_year} TOURNAMENT")
    print(f"=======================================================")
    cur_round = cur_region = None
    for round_label, region, team_a, team_b, p_a, winner in all_games:
        if round_label != cur_round:
            cur_round = round_label
            cur_region = None
            print(f"\n=== {round_label} ===")
        if region != cur_region:
            cur_region = region
            print(f"\n[{region}]")
        print(f"- {team_a} vs {team_b}: {100*p_a:5.1f}% / {100*(1-p_a):5.1f}% (winner: {winner})")

    if args.no_plot:
        return

    top_games = sorted(all_games, key=lambda t: -abs(t[4] - 0.5))[:32]
    labels = [f"{rl} / {rg}: {ta} vs {tb}" for (rl, rg, ta, tb, _, _) in top_games]
    probs  = np.array([p for (_, _, _, _, p, _) in top_games])
    fig_h  = max(8.0, 0.28 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(np.arange(len(labels)), probs, color="#1f77b4")
    ax.axvline(0.5, color="black", linewidth=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(Team A wins)")
    ax.set_title(f"Predicted win probabilities — {args.target_year} (top {len(labels)})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()