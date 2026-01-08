import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def visualize_results(real_data_path, synthetic_data_path, output_dir='results'):
    print("Visualizing results...")
    real_df = pd.read_csv(real_data_path)
    syn_df = pd.read_csv(synthetic_data_path)
    
    # Add labels
    real_df['Type'] = 'Real'
    syn_df['Type'] = 'Synthetic'
    
    combined = pd.concat([real_df, syn_df], axis=0)
    
    # Preprocess for PCA/t-SNE
    # Exclude 'Type' from feature columns
    features = combined.drop('Type', axis=1)
    cat_cols = features.select_dtypes(include=['object', 'category']).columns
    num_cols = features.select_dtypes(include=[np.number]).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    X = preprocessor.fit_transform(combined.drop('Type', axis=1))
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=combined['Type'], alpha=0.5, s=10)
    plt.title('PCA: Real vs Synthetic Data')
    plt.savefig(f'{output_dir}/pca_plot.png')
    print(f"Saved {output_dir}/pca_plot.png")
    
    # t-SNE (Sample subset for speed)
    if X.shape[0] > 2000:
        indices = np.random.choice(X.shape[0], 2000, replace=False)
        X_sub = X[indices]
        types_sub = combined['Type'].iloc[indices]
    else:
        X_sub = X
        types_sub = combined['Type']
        
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_sub)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=types_sub, alpha=0.5, s=10)
    plt.title('t-SNE: Real vs Synthetic Data')
    plt.savefig(f'{output_dir}/tsne_plot.png')
    print(f"Saved {output_dir}/tsne_plot.png")

if __name__ == "__main__":
    import os
    # Paths
    real_path = "results/real_adult.csv"
    syn_path = "results/synthetic_adult_RE-TabSyn.csv"
    
    if os.path.exists(real_path) and os.path.exists(syn_path):
        visualize_results(real_path, syn_path)
    else:
        print(f"Error: Data files not found. Expected {real_path} and {syn_path}")
