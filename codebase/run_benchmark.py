import pandas as pd
import os
from data_loader import DataLoader
from models import GANWrapper, LLMWrapper
from evaluator import Evaluator

def run_benchmark(dataset_name='adult', # models_to_run = ['CTGAN', 'TVAE', 'TabDDPM']
    models_to_run = ['RE-TabSyn'], output_dir='results'):
    print(f"--- Starting Benchmark on {dataset_name} ---")
    
    results = [] # Initialize results list
    
    # 1. Load Data
    loader = DataLoader(dataset_name)
    loader.load_data()
    
    # 2. Analyze Rare Events
    rare_stats = loader.get_rare_event_stats()
    
    train_data, test_data = loader.split_data()
    
    # Save Real Data for Visualization
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    test_data.to_csv(f'{output_dir}/real_{dataset_name}.csv', index=False)
    print(f"Saved real data to {output_dir}/real_{dataset_name}.csv")
    
    # 2. Analyze Rare Events = []

    # 2. Iterate Models
    for model_name in models_to_run:
        print(f"\nProcessing {model_name}...")
        
        if model_name == 'CTGAN':
            model = GANWrapper(epochs=10)
        elif model_name == 'TVAE':
            from models import TVAEWrapper
            model = TVAEWrapper(epochs=10)
        elif model_name == 'TabDDPM':
            model = DiffusionWrapper(epochs=500) # Increased epochs for convergence
        elif model_name == 'TabularVAE':
            from models import VAEWrapper
            model = VAEWrapper(epochs=200, latent_dim=64)
        elif model_name == 'RE-TabSyn':
            from models import LatentDiffusionWrapper
            model = LatentDiffusionWrapper(epochs=100, vae_epochs=200, guidance_scale=2.0, backbone='transformer', target_epsilon=10.0)
        elif model_name == 'TabLLM':
            model = LLMWrapper()
        else:
            print(f"Unknown model {model_name}, skipping.")
            continue

        # 3. Train
        model.train(train_data)
        
        # 4. Generate
        # Generate same size as test set for fair comparison
        synthetic_data = model.generate(num_samples=len(test_data))
        
        # Save Synthetic Data for Visualization
        syn_path = f'{output_dir}/synthetic_{dataset_name}_{model_name}.csv'
        synthetic_data.to_csv(syn_path, index=False)
        print(f"Saved synthetic data to {syn_path}")
        
        # 4. Evaluate
        print(f"Evaluating {model_name}...")
        evaluator = Evaluator(
            real_data=test_data, 
            synthetic_data=synthetic_data,
            target_col=loader.target_col,
            minority_class=rare_stats['minority_class'] if rare_stats else None
        )
        metrics = evaluator.run_all()
        metrics['model'] = model_name
        metrics['dataset'] = dataset_name
        results.append(metrics)

    # 6. Save Results
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'benchmark_{dataset_name}.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Benchmark Complete. Results saved to {output_path} ---")
    print(results_df)

if __name__ == "__main__":
    # Run a quick test
    run_benchmark(dataset_name='adult')
