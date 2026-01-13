#!/usr/bin/env python3
"""
Multi-Dataset Benchmark Script for RE-TabSyn

Runs RE-TabSyn and baseline models across multiple financial datasets
to validate generalization for publication.

Usage:
    python run_multi_benchmark.py                    # Full benchmark
    python run_multi_benchmark.py --quick-test       # Quick smoke test
    python run_multi_benchmark.py --dataset adult    # Single dataset
"""

import pandas as pd
import os
import argparse
import time
import sys
from datetime import datetime, timedelta
from data_loader import DataLoader
from models import LatentDiffusionWrapper, GANWrapper, TVAEWrapper
from evaluator import Evaluator

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Configuration - Financial datasets only (9 total)
ALL_DATASETS = [
    # Core Financial (5 from UCI/standard sources)
    'adult',              # Census Income (proxy for financial - income prediction)
    'credit_default',     # Taiwan Credit Card Default
    'german_credit',      # German Credit Risk
    'bank_marketing',     # Bank Term Deposit Subscription
    'australian_credit',  # Australian Credit Approval
    # Extended Financial (4 additional)
    'credit_approval',    # UCI Credit Approval (anonymized)
    # 'lending_club',       # P2P Lending Loan Default (Requires manual download)
    # 'give_me_credit',     # Kaggle Credit Delinquency (Requires manual download)
    'polish_bankruptcy',  # Company Bankruptcy Prediction
]

# FULL SCALE CONFIGURATION
MODELS = ['RE-TabSyn', 'CTGAN', 'TVAE']  # RE-TabSyn + baselines
SEEDS = [42, 123, 456]  # 3 seeds for statistical significance

# Epoch configuration
QUICK_TEST_EPOCHS = 10
FULL_EPOCHS = 100


def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_single_benchmark(dataset_name, model_name, seed=42, epochs=100, output_dir='results/multi_benchmark'):
    """Run a single benchmark configuration."""
    log(f"Starting: {dataset_name} | {model_name} | Seed: {seed}")
    print(f"{'='*60}", flush=True)
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    log(f"Loading dataset: {dataset_name}")
    loader = DataLoader(dataset_name, random_state=seed)
    try:
        loader.load_data()
    except Exception as e:
        log(f"Failed to load {dataset_name}: {e}", "ERROR")
        return None
    
    log(f"Dataset loaded: {loader.data.shape[0]:,} rows, {loader.data.shape[1]} columns")
    
    # 2. Get rare event stats
    rare_stats = loader.get_rare_event_stats()
    
    # 3. Split data
    train_data, test_data = loader.split_data()
    
    # 4. Initialize model
    if model_name == 'RE-TabSyn':
        model = LatentDiffusionWrapper(
            epochs=epochs, 
            vae_epochs=min(epochs, 100),
            guidance_scale=2.0, 
            backbone='transformer'
        )
    elif model_name == 'CTGAN':
        model = GANWrapper(epochs=epochs)
    elif model_name == 'TVAE':
        model = TVAEWrapper(epochs=epochs)
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    # 5. Train
    print(f"Training {model_name}...")
    model.train(train_data, target_col=loader.target_col)
    
    # 6. Generate
    if model_name == 'RE-TabSyn':
        # Explicitly boost minority class to 50% for controllable synthesis
        synthetic_data = model.generate(num_samples=len(test_data), minority_ratio=0.5)
    else:
        synthetic_data = model.generate(num_samples=len(test_data))
    
    # Save synthetic data
    syn_path = f'{output_dir}/synthetic_{dataset_name}_{model_name}_seed{seed}.csv'
    synthetic_data.to_csv(syn_path, index=False)
    print(f"Saved synthetic data to {syn_path}")
    
    # 7. Evaluate
    evaluator = Evaluator(
        real_data=test_data,
        synthetic_data=synthetic_data,
        train_data=train_data, # Pass training data for Privacy (DCR) check
        target_col=loader.target_col,
        minority_class=rare_stats['minority_class'] if rare_stats else None
    )
    metrics = evaluator.run_all()
    
    # Add metadata
    metrics['dataset'] = dataset_name
    metrics['model'] = model_name
    metrics['seed'] = seed
    metrics['epochs'] = epochs
    metrics['timestamp'] = datetime.now().isoformat()
    
    elapsed = time.time() - start_time
    log(f"Single run completed in {format_time(elapsed)}")
    
    return metrics


def _save_results(results, output_dir, prefix=""):
    """Save results to CSV."""
    if not results:
        return
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/benchmark_results_{prefix}_{timestamp}.csv' if prefix else f'{output_dir}/benchmark_results_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    log(f"Results saved to: {filename}")

def run_multi_benchmark(datasets=None, models=None, seeds=None, quick_test=False, output_dir='results/multi_benchmark'):
    """Run benchmarks across multiple datasets, models, and seeds."""
    
    datasets = datasets or ALL_DATASETS
    models = models or MODELS
    seeds = seeds or SEEDS
    epochs = QUICK_TEST_EPOCHS if quick_test else FULL_EPOCHS
    
    total_runs = len(datasets) * len(models) * len(seeds)
    
    print(f"\n{'#'*70}", flush=True)
    print(f"#  RE-TabSyn Multi-Dataset Benchmark", flush=True)
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#'*70}", flush=True)
    print(f"#  Datasets: {len(datasets)} datasets", flush=True)
    print(f"#  Models:   {models}", flush=True)
    print(f"#  Seeds:    {seeds}", flush=True)
    print(f"#  Epochs:   {epochs}", flush=True)
    print(f"#  Total:    {total_runs} runs", flush=True)
    print(f"{'#'*70}\n", flush=True)
    
    all_results = []
    completed = 0
    failed = 0
    start_time = time.time()
    run_times = []
    
    # Create list of all runs
    runs = [(d, m, s) for d in datasets for m in models for s in seeds]
    
    # Wrap runs with tqdm for overall progress
    pbar = tqdm(enumerate(runs), total=total_runs, desc="Total Benchmark Progress")
    
    for i, (dataset, model_name, seed) in pbar:
        run_start = time.time()
        
        # Update progress bar description
        pbar.set_description(f"Running: {dataset} | {model_name} | seed={seed}")
        
        # Progress header (Keep for log file readability)
        print(f"\n{'='*70}", flush=True)
        print(f"  RUN {i+1}/{total_runs}: {dataset} | {model_name} | seed={seed}", flush=True)
        if run_times:
            avg_time = sum(run_times) / len(run_times)
            remaining = (total_runs - i) * avg_time
            print(f"  Avg time per run: {format_time(avg_time)} | Est. remaining: {format_time(remaining)}", flush=True)
        print(f"{'='*70}", flush=True)
        
        try:
            metrics = run_single_benchmark(
                dataset_name=dataset,
                model_name=model_name,
                seed=seed,
                epochs=epochs,
                output_dir=output_dir
            )
            if metrics:
                all_results.append(metrics)
                completed += 1
                log(f"✅ Completed: {dataset}/{model_name}/seed{seed}")
            else:
                failed += 1
                log(f"⚠️  No metrics returned: {dataset}/{model_name}/seed{seed}", "WARN")
        except Exception as e:
            failed += 1
            log(f"❌ FAILED: {dataset}/{model_name}/seed{seed} - {e}", "ERROR")
            continue
        
        run_time = time.time() - run_start
        run_times.append(run_time)
        log(f"Run time: {format_time(run_time)}")
        
        # Save intermediate results every 5 runs
        if len(all_results) > 0 and len(all_results) % 5 == 0:
            _save_results(all_results, output_dir, "intermediate")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'#'*70}", flush=True)
    print(f"#  BENCHMARK COMPLETE", flush=True)
    print(f"{'#'*70}", flush=True)
    print(f"#  Total time:    {format_time(total_time)}", flush=True)
    print(f"#  Completed:     {completed}/{total_runs} runs", flush=True)
    print(f"#  Failed:        {failed}/{total_runs} runs", flush=True)
    print(f"{'#'*70}\n", flush=True)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'{output_dir}/benchmark_results_FINAL_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        log(f"Final results saved to: {results_path}")
        
        # Print summary table
        print(f"\n{'='*70}", flush=True)
        print("RESULTS SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        summary_cols = ['dataset', 'model', 'seed', 'avg_ks', 'syn_minority_ratio', 'min_dcr', 'utility_f1']
        available_cols = [c for c in summary_cols if c in results_df.columns]
        print(results_df[available_cols].to_string(index=False), flush=True)
        
        # Aggregated stats by model
        print(f"\n{'='*70}", flush=True)
        print("AGGREGATED BY MODEL (Mean ± Std)", flush=True)
        print(f"{'='*70}", flush=True)
        if 'avg_ks' in results_df.columns:
            agg_dict = {
                'avg_ks': ['mean', 'std'],
                'syn_minority_ratio': ['mean', 'std'] if 'syn_minority_ratio' in results_df.columns else lambda x: 0,
            }
            if 'utility_f1' in results_df.columns:
                agg_dict['utility_f1'] = ['mean', 'std']
                
            agg = results_df.groupby('model').agg(agg_dict).round(4)
            print(agg.to_string(), flush=True)
        
        return results_df
    
    return None


def main():
    parser = argparse.ArgumentParser(description='RE-TabSyn Multi-Dataset Benchmark')
    parser.add_argument('--quick-test', action='store_true', help='Run quick smoke test (10 epochs)')
    parser.add_argument('--dataset', type=str, help='Run on specific dataset only')
    parser.add_argument('--model', type=str, help='Run specific model only (default: all configured models)')
    parser.add_argument('--output-dir', type=str, default='results/multi_benchmark', help='Output directory')
    
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else None
    models = [args.model] if args.model else None  # None means use MODELS from config
    
    run_multi_benchmark(
        datasets=datasets,
        models=models,
        quick_test=args.quick_test,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
