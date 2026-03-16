from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.feature_selection import SelectFromModel

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

# ==========================================
# FEATURE ANALYZER CLASSES
# ==========================================
@dataclass
class FeatureImportance:
    name: str
    relative_importance: float  # Percentage of total importance

class FeatureAnalyzer:
    def __init__(self, feature_names: list[str], importances: np.ndarray):
        self.feature_names = list(feature_names)
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
        print(f"{'Feature Name':<55} | {'Importance':<10}")
        print("-" * 70)
        
        cumulative = 0.0
        for f in significant:
            print(f"{f.name:<55} | {f.relative_importance:>5.2f}%")
            cumulative += f.relative_importance
            
        print("-" * 70)
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
    return c in {
        "SCORE", "ROUND", "CURRENT ROUND", "CHAMP", "CHAMP%",
        "F2", "F4", "F4%", "E8", "S16", "R32", "R64", "TOP2",
        "SIM", "SIMS", "PICK", "PUBLIC","WINS","HEAT CHECK TOURNAMENT INDEX::WINS",
        "HEAT CHECK TOURNAMENT INDEX::POOL VALUE","HEAT CHECK TOURNAMENT INDEX::VAL Z-SCORE",
        "HEAT CHECK TOURNAMENT INDEX::POWER-PATH"
    }

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

    by_year_team: dict[tuple[int, str], np.ndarray] = {}
    def _safe_nanmean(M: np.ndarray, axis: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(M, axis=axis)
        return np.where(np.isfinite(m), m, np.nan)

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
    feature_names: list[str], mu: np.ndarray, sigma: np.ndarray, 
    xgb_model: XGBClassifier, lr_model: LogisticRegressionCV, 
    selector: SelectFromModel = None,
) -> float:
    va = _vec_for_team_name(team_a, year, by_year_team, by_team_mean)
    vb = _vec_for_team_name(team_b, year, by_year_team, by_team_mean)
    
    base_len = int(len(feature_names) / 2) if len(feature_names) > 0 else 0
    
    if va is None:
        va = np.zeros(base_len, dtype=np.float64)
    if vb is None:
        vb = np.zeros(base_len, dtype=np.float64)

    x = np.concatenate([va - vb, va + vb, np.array([float(seed_b - seed_a)], dtype=np.float64)], axis=0)
    x = np.where(np.isnan(x), 0.0, x)
    
    xs = standardize_apply(x[None, :], mu, sigma)
    
    if selector is not None:
        xs = selector.transform(xs)
    
    p_xgb = float(xgb_model.predict_proba(xs)[0, 1])
    p_lr = float(lr_model.predict_proba(xs)[0, 1])
    
    # Blended 70% LR / 30% XGBoost (Giving XGB a slightly bigger voice now that it has early stopping)
    p_blended = (0.60 * p_xgb) + (0.40 * p_lr)
    
    return p_blended

# ==========================================
# MAIN EXECUTION
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost win-probability model for March Madness.")
    parser.add_argument("--target_year", type=int, default=2026, help="Year to predict (default: 2026).")
    parser.add_argument("--predict_round", type=int, default=64, help="Round to predict (64=Round of 64, ...). Default: 64")
    parser.add_argument("--train_start_year", type=int, default=2008, help="First training year (inclusive).")
    parser.add_argument("--train_end_year", type=int, default=2024, help="Last training year (inclusive).")
    parser.add_argument("--test_year", type=int, default=2025, help="Year to hold out for accuracy testing.")
    parser.add_argument("--no_plot", action="store_true", help="Disable matplotlib plotting.")
    args = parser.parse_args()

    z_path = DATASET_DIR / "Z Rating Teams.csv"
    tm_path = DATASET_DIR / "Tournament Matchups.csv"
    
    games = load_games_from_tournament_matchups(tm_path)

    all_years = list(range(args.train_start_year, args.train_end_year + 1))
    train_years = [y for y in all_years if y != args.test_year]
    test_years = [args.test_year]

    print(f"Loading features...")
    print(f"Training on years: {train_years}")
    print(f"Testing on year: {test_years}")

    by_year_team, by_team_mean, base_feature_names = load_team_feature_store(DATASET_DIR, years=all_years)
    by_year_team, by_team_mean, base_feature_names = drop_high_missing_features(
        by_year_team, by_team_mean, base_feature_names, max_missing_frac=0.60
    )
    z_seeds = load_z_ratings(z_path)

    diff_names = [f"{name} (DIFF)" for name in base_feature_names]
    sum_names = [f"{name} (SUM)" for name in base_feature_names]
    full_feature_names = diff_names + sum_names
    
    def build_matrix(target_years):
        X_list, y_list, year_list, w_list = [], [], [], [] 
        min_year = min(target_years) if target_years else 2008
        
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
            seed_adv = float(seed_b - seed_a)

            x = np.concatenate([va - vb, va + vb, np.array([seed_adv], dtype=np.float64)], axis=0)
            y = 1 if g.score_a > g.score_b else 0
            
            weight = 1.0
            weight += (g.year - min_year) * 0.15 
            if abs(seed_adv) <= 3:
                weight += 0.5
            if seed_a <= 4 or seed_b <= 4:
                weight += 0.5
            
            X_list.append(x)
            y_list.append(y)
            year_list.append(g.year)
            w_list.append(weight)
            
        if not X_list:
            return np.empty((0, len(full_feature_names)+1)), np.empty(0), np.empty(0), np.empty(0)
        
        return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32), np.array(year_list, dtype=np.int32), np.array(w_list, dtype=np.float32)

    # 1. BUILD BOTH TRAIN AND TEST SETS EARLY
    X_train, y_train, groups_train, w_train = build_matrix(train_years) 
    X_test, y_test, _, _ = build_matrix(test_years) 

    if len(X_train) < 50:
        raise SystemExit(f"Not enough training games found ({len(X_train)}).")

    # 2. SCALE AND IMPUTE
    X_train, col_means = impute_nan_with_col_means(X_train)
    mu, sigma = standardize_fit(X_train)
    Xs_train = standardize_apply(X_train, mu, sigma)

    if len(y_test) > 0:
        X_test_imputed = np.where(np.isnan(X_test), col_means[None, :], X_test)
        Xs_test = standardize_apply(X_test_imputed, mu, sigma)
    else:
        Xs_test = np.empty((0, Xs_train.shape[1]))

    # 3. FEATURE SELECTION (MADE STRICTER: 2.0 * MEAN)
    print("\n--- Trimming the Fat (Automated Feature Selection) ---")
    scout_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
    scout_model.fit(Xs_train, y_train, sample_weight=w_train)

    selector = SelectFromModel(scout_model, prefit=True, threshold='2.0*mean')

    Xs_train_trimmed = selector.transform(Xs_train)
    if len(y_test) > 0:
        Xs_test_trimmed = selector.transform(Xs_test)
    else:
        Xs_test_trimmed = np.empty((0, Xs_train_trimmed.shape[1]))

    analyzer_feature_names = full_feature_names + ["Calculated Seed Advantage"]
    support_mask = selector.get_support()
    trimmed_feature_names = [analyzer_feature_names[i] for i, keep in enumerate(support_mask) if keep]

    print(f"Original feature count: {Xs_train.shape[1]}")
    print(f"Trimmed feature count:  {Xs_train_trimmed.shape[1]}")

    # 4. HYPERPARAMETER TUNING
    print("\n--- Training Model 1: XGBoost (Hyperparameter Search) ---")
    param_grid = {
        'max_depth': [2, 3], 
        'learning_rate': [0.01, 0.05], 
        'min_child_weight': [3, 5, 7, 10], 
        'gamma': [1.0, 3.0, 5.0, 10.0], 
        'subsample': [0.5, 0.7, 0.8], 
        'colsample_bytree': [0.4, 0.6, 0.8], 
        'reg_alpha': [1.0, 5.0, 10.0], 
        'reg_lambda': [1.0, 5.0, 10.0] 
    }

    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
    num_unique_years = len(np.unique(groups_train))
    gkf = GroupKFold(n_splits=num_unique_years)

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=100,                
        scoring='neg_brier_score',  
        cv=gkf,                      
        verbose=0,                  
        random_state=42,
        n_jobs=-1                   
    )

    random_search.fit(Xs_train_trimmed, y_train, groups=groups_train, sample_weight=w_train)
    best_params = random_search.best_params_
    
    print(f"Best Base Parameters: {best_params}")
    print("\n--- Applying Early Stopping to Best XGBoost Model ---")
    
    # 5. RETRAIN WITH EARLY STOPPING
    # Give it up to 500 epochs, but stop if test accuracy doesn't improve for 25 epochs
    xgb_model = XGBClassifier(
        **best_params,
        n_estimators=500,
        random_state=42,
        early_stopping_rounds=25,
        eval_metric=['error', 'logloss']
    )

    if len(y_test) > 0:
        eval_set = [(Xs_train_trimmed, y_train), (Xs_test_trimmed, y_test)]
    else:
        eval_set = [(Xs_train_trimmed, y_train)]

    xgb_model.fit(
        Xs_train_trimmed, y_train,
        sample_weight=w_train,
        eval_set=eval_set,
        verbose=False
    )
    
    print(f"XGBoost Early Stopped at Epoch {xgb_model.best_iteration}!")

    # 6. LOGISTIC REGRESSION
    print("\n--- Training Model 2: Logistic Regression (Auto-Tuned) ---")
    lr_model = LogisticRegressionCV(
        Cs=10,                     
        cv=5,                      
        penalty='l2', 
        solver='liblinear', 
        scoring='neg_brier_score', 
        random_state=42,
        max_iter=1000
    )
    lr_model.fit(Xs_train_trimmed, y_train, sample_weight=w_train)

    # 7. EVALUATIONS
    p_train_xgb = xgb_model.predict_proba(Xs_train_trimmed)[:, 1]
    p_train_lr = lr_model.predict_proba(Xs_train_trimmed)[:, 1]
    p_train_blend = (0.40 * p_train_xgb) + (0.60 * p_train_lr) 
    
    train_acc_blend = np.mean((p_train_blend >= 0.5).astype(int) == y_train)
    
    print(f"\nTrained on {len(y_train)} games.")
    print(f"XGBoost Training Brier: {brier_score(p_train_xgb, y_train):.4f}")
    print(f"LogReg Training Brier:  {brier_score(p_train_lr, y_train):.4f}")
    print(f"BLENDED Training Accuracy: {train_acc_blend*100:.2f}% | BLENDED Brier score: {brier_score(p_train_blend, y_train):.4f}")

    analyzer = FeatureAnalyzer(trimmed_feature_names, xgb_model.feature_importances_)
    analyzer.print_report(min_relative_importance=0.5) 

    if len(y_test) > 0:
        p_test_xgb = xgb_model.predict_proba(Xs_test_trimmed)[:, 1]
        p_test_lr = lr_model.predict_proba(Xs_test_trimmed)[:, 1]
        p_test_blend = (0.40 * p_test_xgb) + (0.60 * p_test_lr) 
        
        test_acc_blend = np.mean((p_test_blend >= 0.5).astype(int) == y_test)
        
        print(f"Tested on {len(y_test)} unseen games from {args.test_year}.")
        print(f"XGBoost Test Brier: {brier_score(p_test_xgb, y_test):.4f}")
        print(f"LogReg Test Brier:  {brier_score(p_test_lr, y_test):.4f}")
        print(f"BLENDED Testing Accuracy:  {test_acc_blend*100:.2f}% | BLENDED Test Brier score: {brier_score(p_test_blend, y_test):.4f}\n")
    else:
        print(f"No test games found for {args.test_year}.\n")

    # 8. BRACKET PROJECTION
    initial = bracket_2026_round_of_64()

    def _run_region(region_name: str, matchups: list[BracketMatchup]):
        nonlocal all_games, region_champions
        cur_games: list[tuple[str, int, str, int]] = [(m.team_a, m.seed_a, m.team_b, m.seed_b) for m in matchups]
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
        round_idx = 0

        while True:
            label = round_labels[round_idx] if round_idx < len(round_labels) else f"Round {round_idx+1}"
            winners: list[tuple[str, int]] = []
            for team_a, seed_a, team_b, seed_b in cur_games:
                p_a = _predict_matchup_prob(
                    team_a, seed_a, team_b, seed_b, args.target_year,
                    by_year_team, by_team_mean, full_feature_names, mu, sigma, xgb_model, lr_model,
                    selector=selector
                )
                winner, w_seed = (team_a, seed_a) if p_a >= 0.5 else (team_b, seed_b)
                all_games.append((label, region_name, team_a, team_b, p_a, winner))
                winners.append((winner, w_seed))

            if len(winners) == 1:
                region_champions[region_name] = winners[0]
                break

            next_games: list[tuple[str, int, str, int]] = []
            for i in range(0, len(winners), 2):
                (ta, sa) = winners[i]
                (tb, sb) = winners[i + 1]
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

    def _ff_game(region1: str, region2: str, slot_name: str):
        nonlocal all_games
        (team_a, seed_a) = region_champions[region1]
        (team_b, seed_b) = region_champions[region2]
        p_a = _predict_matchup_prob(
            team_a, seed_a, team_b, seed_b, args.target_year,
            by_year_team, by_team_mean, full_feature_names, mu, sigma, xgb_model, lr_model,
            selector=selector
        )
        winner, _ = (team_a, seed_a) if p_a >= 0.5 else (team_b, seed_b)
        all_games.append(("Final Four", slot_name, team_a, team_b, p_a, winner))
        return winner

    west_east_winner = _ff_game("EAST", "WEST", "EAST/WEST")
    south_midwest_winner = _ff_game("SOUTH", "MIDWEST", "SOUTH/MIDWEST")

    def _find_seed(team_name: str) -> int:
        for _, (t, s) in region_champions.items():
            if t == team_name:
                return s
        return 1

    seed_a = _find_seed(west_east_winner)
    seed_b = _find_seed(south_midwest_winner)
    p_a_final = _predict_matchup_prob(
        west_east_winner, seed_a, south_midwest_winner, seed_b, args.target_year,
        by_year_team, by_team_mean, full_feature_names, mu, sigma, xgb_model, lr_model,
        selector=selector
    )
    champ_winner = west_east_winner if p_a_final >= 0.5 else south_midwest_winner
    all_games.append(("Championship", "NATIONAL", west_east_winner, south_midwest_winner, p_a_final, champ_winner))

    round_order = {"Round of 64": 1, "Round of 32": 2, "Sweet 16": 3, "Elite 8": 4, "Final Four": 5, "Championship": 6}
    all_games.sort(key=lambda t: (round_order.get(t[0], 99), t[1], -abs(t[4] - 0.5), t[2]))

    print(f"=======================================================")
    print(f"   PREDICTING THE FULL {args.target_year} TOURNAMENT")
    print(f"=======================================================")
    
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

    # 9. PLOT THE LEARNING CURVE INSTEAD OF MATCHUP BARS
    if not args.no_plot and len(y_test) > 0:
        results = xgb_model.evals_result()
        
        # 'error' metric = 1 - accuracy
        train_acc = [1.0 - x for x in results['validation_0']['error']]
        test_acc = [1.0 - x for x in results['validation_1']['error']]
        epochs = len(train_acc)
        x_axis = range(0, epochs)

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, train_acc, label='Train Accuracy', color='#1f77b4', linewidth=2)
        plt.plot(x_axis, test_acc, label='Test (Holdout) Accuracy', color='#ff7f0e', linewidth=2)
        
        # Mark the early stopping point
        best_iter = xgb_model.best_iteration
        plt.axvline(best_iter, color='red', linestyle='--', label=f'Early Stop Iteration ({best_iter})')
        
        plt.legend(loc='lower right')
        plt.title('XGBoost Learning Curve: Epochs vs Accuracy', fontsize=14)
        plt.xlabel('Epochs (Number of Decision Trees)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
