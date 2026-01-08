"""
Enhanced Visualization with Confidence Intervals
Generates benchmark result plots with error bars based on multi-seed results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_benchmark_results(csv_path):
    """Load benchmark results CSV."""
    return pd.read_csv(csv_path)


def calculate_ci(data, confidence=0.95):
    """Calculate mean and 95% confidence interval."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    ci = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    return mean, ci


def plot_ks_with_ci(df, output_path='results/ks_with_ci.png'):
    """Plot KS statistic by dataset with confidence intervals."""
    # Group by dataset and calculate mean/CI
    datasets = df['dataset'].unique()
    means = []
    cis = []
    
    for dataset in datasets:
        data = df[df['dataset'] == dataset]['avg_ks']
        mean, ci = calculate_ci(data)
        means.append(mean)
        cis.append(ci)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    
    bars = ax.bar(x, means, yerr=cis, capsize=5, color='steelblue', alpha=0.8,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('KS Statistic (↓ lower is better)', fontsize=12)
    ax.set_title('RE-TabSyn: Statistical Fidelity with 95% CI\n(3 seeds per dataset)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.01,
                f'{mean:.3f}±{ci:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='CTGAN baseline (~0.15)')
    ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='TabSyn baseline (~0.10)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_minority_boost_with_ci(df, output_path='results/minority_boost_with_ci.png'):
    """Plot minority class boost with confidence intervals."""
    datasets = df['dataset'].unique()
    real_ratios = []
    syn_means = []
    syn_cis = []
    
    for dataset in datasets:
        data = df[df['dataset'] == dataset]
        real_ratios.append(data['real_minority_ratio'].iloc[0])
        syn_data = data['syn_minority_ratio']
        mean, ci = calculate_ci(syn_data)
        syn_means.append(mean)
        syn_cis.append(ci)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, np.array(real_ratios) * 100, width, label='Original', color='lightcoral', alpha=0.8)
    bars = ax.bar(x + width/2, np.array(syn_means) * 100, width, 
                  yerr=np.array(syn_cis) * 100, capsize=5,
                  label='RE-TabSyn', color='steelblue', alpha=0.8,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='Target (50%)')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Minority Class Ratio (%)', fontsize=12)
    ax.set_title('RE-TabSyn: Minority Class Control with 95% CI\nCFG enables boosting rare events to ~50%', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 65)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_with_ci(df, output_path='results/comparison_with_ci.png'):
    """Create comparison plot showing KS vs Minority Control with error bars."""
    datasets = df['dataset'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: KS Statistic
    ax1 = axes[0]
    ks_means = []
    ks_cis = []
    for dataset in datasets:
        data = df[df['dataset'] == dataset]['avg_ks']
        mean, ci = calculate_ci(data)
        ks_means.append(mean)
        ks_cis.append(ci)
    
    x = np.arange(len(datasets))
    ax1.barh(x, ks_means, xerr=ks_cis, capsize=4, color='steelblue', alpha=0.8)
    ax1.set_yticks(x)
    ax1.set_yticklabels(datasets)
    ax1.set_xlabel('KS Statistic (↓ lower is better)')
    ax1.set_title('Fidelity (with 95% CI)')
    ax1.axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='CTGAN')
    ax1.legend()
    
    # Plot 2: Minority Boost
    ax2 = axes[1]
    boost_means = []
    boost_cis = []
    for dataset in datasets:
        data = df[df['dataset'] == dataset]
        real = data['real_minority_ratio'].iloc[0]
        syn_data = data['syn_minority_ratio']
        boost_data = syn_data - real
        mean, ci = calculate_ci(boost_data)
        boost_means.append(mean * 100)
        boost_cis.append(ci * 100)
    
    ax2.barh(x, boost_means, xerr=boost_cis, capsize=4, color='forestgreen', alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(datasets)
    ax2.set_xlabel('Minority Boost (%)')
    ax2.set_title('Rare Event Control (with 95% CI)')
    
    plt.suptitle('RE-TabSyn Performance Summary (3 seeds per dataset)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(df):
    """Generate summary table with CI for the paper."""
    print("\n" + "="*80)
    print("SUMMARY TABLE WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    datasets = df['dataset'].unique()
    print(f"\n{'Dataset':<20} {'KS (mean±CI)':<18} {'Minority % (mean±CI)':<22} {'Boost':<10}")
    print("-"*70)
    
    for dataset in datasets:
        data = df[df['dataset'] == dataset]
        
        ks_mean, ks_ci = calculate_ci(data['avg_ks'])
        syn_mean, syn_ci = calculate_ci(data['syn_minority_ratio'])
        real_ratio = data['real_minority_ratio'].iloc[0]
        boost = (syn_mean - real_ratio) * 100
        
        print(f"{dataset:<20} {ks_mean:.3f} ± {ks_ci:.3f}      "
              f"{syn_mean*100:.1f}% ± {syn_ci*100:.1f}%         +{boost:.1f}%")
    
    print("="*80)


def main():
    """Generate all visualizations with confidence intervals."""
    csv_path = 'results/full_benchmark/benchmark_results_FINAL_20251210_011902.csv'
    
    try:
        df = load_benchmark_results(csv_path)
        print(f"Loaded {len(df)} results from {len(df['dataset'].unique())} datasets")
        
        # Generate plots
        plot_ks_with_ci(df)
        plot_minority_boost_with_ci(df)
        plot_comparison_with_ci(df)
        
        # Generate summary table
        generate_summary_table(df)
        
        print("\n✅ All visualizations with confidence intervals generated!")
        
    except FileNotFoundError:
        print(f"Error: {csv_path} not found")
        print("Run the benchmark first to generate results.")


if __name__ == "__main__":
    main()
