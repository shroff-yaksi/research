import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, real_data, synthetic_data, train_data=None, target_col=None, minority_class=None):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.train_data = train_data
        self.target_col = target_col
        self.minority_class = minority_class

    def evaluate_fidelity(self):
        """Computes statistical similarity between real and synthetic data."""
        print("Evaluating Fidelity...")
        ks_scores = []
        for col in tqdm(self.real_data.columns, desc="Calculating KS Stats"):
            if np.issubdtype(self.real_data[col].dtype, np.number):
                statistic, p_value = ks_2samp(self.real_data[col], self.synthetic_data[col])
                ks_scores.append(statistic)
        
        avg_ks = np.mean(ks_scores) if ks_scores else 0.0
        print(f"Average KS Statistic: {avg_ks:.4f}")
        return {'avg_ks': avg_ks}

    def evaluate_privacy(self):
        """Computes Distance to Closest Record (DCR) as a privacy metric."""
        print("Evaluating Privacy (DCR)...")
        
        # Use training data for privacy check (Did model memorize training data?)
        # Fallback to real_data (test) if train not provided
        reference_data = self.train_data if self.train_data is not None else self.real_data
        
        if self.train_data is None:
            print("Warning: Training data not provided. DCR calculated against Test data (Less accurate for privacy).")
            
        # Sample for efficiency if data is large
        n_samples = min(1000, len(reference_data), len(self.synthetic_data))
        real_sample = reference_data.sample(n=n_samples, random_state=42).copy()
        syn_sample = self.synthetic_data.sample(n=n_samples, random_state=42).copy()

        # Preprocess for distance calculation: One-Hot Encode Categoricals
        # This provides a valid Euclidean distance in the expanded space (Hamming-like)
        
        # Combine to ensure consistent feature space
        real_sample['__type__'] = 0
        syn_sample['__type__'] = 1
        combined = pd.concat([real_sample, syn_sample], ignore_index=True)
        
        # Identify columns
        cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        # Remove helper column from num_cols if it got picked up
        if '__type__' in num_cols: num_cols.remove('__type__')
        
        # Simple preprocessing pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ],
            verbose_feature_names_out=False
        )
        
        # Transform
        combined_encoded = preprocessor.fit_transform(combined.drop(columns=['__type__']))
        
        # Split back
        real_encoded = combined_encoded[combined['__type__'] == 0]
        syn_encoded = combined_encoded[combined['__type__'] == 1]
            
        # Fit NN on real data
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(real_encoded)
        distances, indices = nbrs.kneighbors(syn_encoded)
        
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

    def evaluate_utility(self):
        """
        Computes Machine Learning Utility (Train on Synthetic, Test on Real - TSTR).
        Trains a Random Forest on synthetic data and evaluates on real test data.
        """
        if not self.target_col:
            print("Skipping Utility Evaluation (Target column not defined).")
            return {}

        print("Evaluating Utility (TSTR)...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report

        # Prepare Data
        X_syn = self.synthetic_data.drop(columns=[self.target_col])
        y_syn = self.synthetic_data[self.target_col]
        
        X_real = self.real_data.drop(columns=[self.target_col])
        y_real = self.real_data[self.target_col]

        # Identify types
        cat_cols = X_syn.select_dtypes(include=['object', 'category']).columns
        num_cols = X_syn.select_dtypes(include=[np.number]).columns

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
        ])

        # Train on Synthetic
        try:
            pipeline.fit(X_syn, y_syn)
            
            # Test on Real
            y_pred = pipeline.predict(X_real)
            
            # Calculate Metrics (Macro average to balance importance of classes)
            f1 = f1_score(y_real, y_pred, average='macro')
            prec = precision_score(y_real, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_real, y_pred, average='macro')
            
            print(f"Utility Scores - F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
            
            return {
                'utility_f1': f1,
                'utility_precision': prec,
                'utility_recall': rec
            }
        except Exception as e:
            print(f"Utility evaluation failed: {e}")
            return {'utility_f1': 0.0, 'utility_precision': 0.0, 'utility_recall': 0.0}

    def run_all(self):
        results = {}
        results.update(self.evaluate_fidelity())
        results.update(self.evaluate_privacy())
        results.update(self.evaluate_rare_events())
        results.update(self.evaluate_utility())
        return results
