import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score

class Evaluator:
    def __init__(self, real_data, synthetic_data, target_col=None, minority_class=None):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.target_col = target_col
        self.minority_class = minority_class

    def evaluate_fidelity(self):
        """Computes statistical similarity between real and synthetic data."""
        print("Evaluating Fidelity...")
        ks_scores = []
        for col in self.real_data.columns:
            if np.issubdtype(self.real_data[col].dtype, np.number):
                statistic, p_value = ks_2samp(self.real_data[col], self.synthetic_data[col])
                ks_scores.append(statistic)
        
        avg_ks = np.mean(ks_scores) if ks_scores else 0.0
        print(f"Average KS Statistic: {avg_ks:.4f}")
        return {'avg_ks': avg_ks}

    def evaluate_privacy(self):
        """Computes Distance to Closest Record (DCR) as a privacy metric."""
        print("Evaluating Privacy (DCR)...")
        # Sample for efficiency if data is large
        n_samples = min(1000, len(self.real_data), len(self.synthetic_data))
        real_sample = self.real_data.sample(n=n_samples, random_state=42).copy()
        syn_sample = self.synthetic_data.sample(n=n_samples, random_state=42).copy()

        # Preprocess for distance calculation: Label Encode Categoricals
        # We need to encode both real and syn together to ensure consistent mapping
        # or just use independent encoding if the overlap is small, but consistent is better.
        # For DCR, we need numeric representation.
        from sklearn.preprocessing import LabelEncoder
        
        # Identify categorical columns (object type)
        cat_cols = real_sample.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            le = LabelEncoder()
            # Fit on combined unique values to handle unseen labels safely
            all_vals = pd.concat([real_sample[col], syn_sample[col]]).unique()
            le.fit(all_vals.astype(str))
            real_sample[col] = le.transform(real_sample[col].astype(str))
            syn_sample[col] = le.transform(syn_sample[col].astype(str))
            
        # Fit NN on real data
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(real_sample)
        distances, indices = nbrs.kneighbors(syn_sample)
        
        min_dcr = np.min(distances)
        avg_dcr = np.mean(distances)
        
        print(f"Min DCR: {min_dcr:.4f}, Avg DCR: {avg_dcr:.4f}")
        return {'min_dcr': min_dcr, 'avg_dcr': avg_dcr}

    def evaluate_rare_events(self):
        """Computes utility metrics specifically for the minority class."""
        if not self.target_col or self.minority_class is None:
            print("Skipping Rare Event Evaluation (Target/Minority class not defined).")
            return {}

        print("Evaluating Rare Event Preservation...")
        
        # Calculate ratio in synthetic data
        syn_counts = self.synthetic_data[self.target_col].value_counts(normalize=True)
        syn_ratio = syn_counts.get(self.minority_class, 0.0)
        
        real_counts = self.real_data[self.target_col].value_counts(normalize=True)
        real_ratio = real_counts.get(self.minority_class, 0.0)

        print(f"Minority Ratio - Real: {real_ratio:.4f}, Synthetic: {syn_ratio:.4f}")
        
        return {
            'real_minority_ratio': real_ratio,
            'syn_minority_ratio': syn_ratio,
            'ratio_diff': abs(real_ratio - syn_ratio)
        }

    def run_all(self):
        results = {}
        results.update(self.evaluate_fidelity())
        results.update(self.evaluate_privacy())
        results.update(self.evaluate_rare_events())
        return results
