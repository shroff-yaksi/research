import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import argparse
import os
import glob

def get_latest_results(results_dir='results/multi_benchmark'):
    """Find the most recent synthetic data file in the results directory."""
    # Look for synthetic data files
    files = glob.glob(os.path.join(results_dir, "synthetic_*.csv"))
    if not files:
        return None, None
    
    # Sort by modification time
    latest_file = max(files, key=os.path.getmtime)
    
    # Try to deduce dataset name from filename: synthetic_{dataset}_{model}_seed{seed}.csv
    filename = os.path.basename(latest_file)
    parts = filename.split('_')
    
    # This is a bit heuristics-based, assuming "synthetic" is first.
    # We need to find the real file. 
    # Usually real data is not saved in results unless we save it explicitly.
    # But wait, run_multi_benchmark doesn't save real data...
    # It only saves synthetic data.
    # We might need to load the real data using DataLoader again!
    
    dataset_name = None
    # Heuristic: synthetic_adult_... -> adult
    if len(parts) >= 2:
        dataset_name = parts[1]
        # Handle multi-word datasets if needed (e.g. credit_default)
        # But run_multi_benchmark uses dataset names from its list.
        # Let's assume the naming convention holds.
        
    return latest_file, dataset_name

def visualize_results(real_data_path=None, synthetic_data_path=None, dataset_name=None, output_dir='results/multi_benchmark'):
    print("Visualizing results...")
    
    if synthetic_data_path is None:
        synthetic_data_path, deduced_dataset = get_latest_results(output_dir)
        if synthetic_data_path is None:
            print(f"No synthetic data found in {output_dir}")
            return
        if dataset_name is None:
            dataset_name = deduced_dataset
            print(f"Auto-detected dataset name: {dataset_name}")

    if not os.path.exists(synthetic_data_path):
        print(f"Error: Synthetic data file not found: {synthetic_data_path}")
        return

    # Load Synthetic Data
    syn_df = pd.read_csv(synthetic_data_path)
    syn_df['Type'] = 'Synthetic'

    # Load Real Data
    # Since we might not have a CSV of real data saved, let's use DataLoader to get it.
    real_df = None
    if real_data_path and os.path.exists(real_data_path):
        real_df = pd.read_csv(real_data_path)
    elif dataset_name:
        print(f"Loading real data for {dataset_name} using DataLoader...")
        from data_loader import DataLoader
        try:
            loader = DataLoader(dataset_name)
            loader.load_data()
            real_df = loader.data
        except Exception as e:
            print(f"Failed to load real data: {e}")
            return
    else:
        print("Error: No real data path provided and could not deduce dataset name.")
        return

    real_df['Type'] = 'Real'
    
    # Align columns (Synthetic might have different columns or order?)
    # Usually they should match.
    common_cols = [c for c in real_df.columns if c in syn_df.columns and c != 'Type']
    
    if len(common_cols) < len(real_df.columns) - 1:
        print("Warning: Columns mismatch between real and synthetic data.")
        print(f"Real: {real_df.columns.tolist()}")
        print(f"Syn: {syn_df.columns.tolist()}")
    
    combined = pd.concat([real_df[common_cols + ['Type']], syn_df[common_cols + ['Type']]], axis=0)
    
    # Sample for performance
    if len(combined) > 5000:
        combined = combined.sample(n=5000, random_state=42)

    # Preprocess for PCA/t-SNE
    features = combined.drop('Type', axis=1)
    cat_cols = features.select_dtypes(include=['object', 'category']).columns
    num_cols = features.select_dtypes(include=[np.number]).columns
    
    # Handle NaNs
    features = features.fillna(0) # Simple fill
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    print("Preprocessing data for visualization...")
    X = preprocessor.fit_transform(features)
    
    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=combined['Type'], alpha=0.5, s=15)
    plt.title(f'PCA: Real vs Synthetic ({dataset_name})')
    pca_path = os.path.join(output_dir, f'pca_{dataset_name}.png')
    plt.savefig(pca_path)
    print(f"Saved {pca_path}")
    
    # t-SNE
    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=combined['Type'], alpha=0.5, s=15)
    plt.title(f't-SNE: Real vs Synthetic ({dataset_name})')
    tsne_path = os.path.join(output_dir, f'tsne_{dataset_name}.png')
    plt.savefig(tsne_path)
    print(f"Saved {tsne_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Real vs Synthetic Data')
    parser.add_argument('--real', type=str, help='Path to real data CSV')
    parser.add_argument('--syn', type=str, help='Path to synthetic data CSV')
    parser.add_argument('--dataset', type=str, help='Dataset name (to load real data via DataLoader)')
    parser.add_argument('--output-dir', type=str, default='results/multi_benchmark', help='Output directory')
    
    args = parser.parse_args()
    
    visualize_results(args.real, args.syn, args.dataset, args.output_dir)