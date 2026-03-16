from dataclasses import dataclass
import numpy as np

@dataclass
class FeatureImportance:
    name: str
    weight: float
    abs_weight: float
    relative_importance: float  # Percentage of total importance

class FeatureAnalyzer:
    """
    Analyzes the learned weights of a linear model to determine true feature importance.
    """
    def __init__(self, feature_names: list[str], weights: np.ndarray):
        # The script manually appends the seed advantage during training, 
        # so we need to add it to our names list to match the weights array.
        self.feature_names = list(feature_names)
        if len(weights) == len(self.feature_names) + 1:
            self.feature_names.append("Calculated Seed Advantage")
            
        self.weights = weights

    def analyze_all(self) -> list[FeatureImportance]:
        """Calculates and sorts the importance of every single feature."""
        total_abs_weight = np.sum(np.abs(self.weights))
        
        importances = []
        for name, w in zip(self.feature_names, self.weights):
            abs_w = float(np.abs(w))
            # Calculate what percentage this feature contributes to the overall model
            rel_imp = (abs_w / total_abs_weight) * 100 if total_abs_weight > 0 else 0.0
            importances.append(FeatureImportance(name, float(w), abs_w, rel_imp))
            
        # Sort from most important to least important
        importances.sort(key=lambda x: x.abs_weight, reverse=True)
        return importances

    def get_significant_features(self, min_relative_importance: float = 1.0) -> list[FeatureImportance]:
        """
        Returns only the features that actually matter, filtering out the noise.
        min_relative_importance: The minimum percentage of overall weight a feature 
                                 needs to be considered "significant".
        """
        all_features = self.analyze_all()
        # Keep only features that cross our dynamic threshold
        return [f for f in all_features if f.relative_importance >= min_relative_importance]

    def print_report(self, min_relative_importance: float = 1.0) -> None:
        """Prints a clean, formatted report of the significant features."""
        significant = self.get_significant_features(min_relative_importance)
        
        print("\n=======================================================")
        print(f"   SIGNIFICANT FEATURES (Threshold: >= {min_relative_importance}%)")
        print("=======================================================")
        print(f"{'Feature Name':<40} | {'Weight':<8} | {'Importance':<10}")
        print("-" * 65)
        
        cumulative = 0.0
        for f in significant:
            direction = "+" if f.weight > 0 else "-"
            print(f"{f.name:<40} | {direction}{f.abs_weight:.4f}  | {f.relative_importance:>5.2f}%")
            cumulative += f.relative_importance
            
        print("-" * 65)
        print(f"Total features kept: {len(significant)} out of {len(self.feature_names)}")
        print(f"These account for {cumulative:.1f}% of the model's decision-making power.\n")
