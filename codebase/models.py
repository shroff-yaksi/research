from diffusion_model import MLPDiffusion, GaussianDiffusion
from vae import TabularVAE
from latent_diffusion import LatentDiffusion, LatentMLP
from transformer import TabularTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from ctgan import CTGAN
from sdv.single_table import TVAESynthesizer
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

class BaseModel:
    def __init__(self, name):
        self.name = name
        self.is_trained = False

    def train(self, train_data, target_col=None):
        raise NotImplementedError

    def generate(self, num_samples):
        raise NotImplementedError

# ... (VAEWrapper code remains same, just update train signature if needed, but it ignores target_col)

class LatentDiffusionWrapper(BaseModel):
    def __init__(self, epochs=500, vae_epochs=200, latent_dim=64, guidance_scale=2.0, backbone='mlp', target_epsilon=None, target_delta=1e-5):
        super().__init__("RE-TabSyn")
        self.epochs = epochs
        self.vae_epochs = vae_epochs
        self.latent_dim = latent_dim
        self.guidance_scale = guidance_scale
        self.backbone = backbone
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vae = None
        self.diffusion = None
        self.preprocessor = None
        self.label_encoder = None
        self.output_dim = 0
        self.original_columns = None
        self.num_cols_idx = None
        self.cat_cols_idx_list = []
        self.cat_cols = []
        self.num_cols = []
        self.target_col = None
        self.minority_class_label = None
        self.can_guide = False

    def train(self, train_data, target_col='salary'):
        print(f"Training {self.name} on {self.device}...")
        self.original_columns = train_data.columns
        self.target_col = target_col
        
        # --- Step 1: Train VAE ---
        print(f"--- Phase 1: Training VAE ({self.vae_epochs} epochs) ---")
        
        # Preprocessing
        self.cat_cols = train_data.select_dtypes(include=['object', 'category']).columns
        self.num_cols = train_data.select_dtypes(include=[np.number]).columns
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', QuantileTransformer(output_distribution='normal'), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols)
            ],
            verbose_feature_names_out=False
        )
        
        X_train = self.preprocessor.fit_transform(train_data)
        self.output_dim = X_train.shape[1]
        
        # Prepare Labels for Guidance
        if target_col in train_data.columns:
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(train_data[target_col])
            y_tensor = torch.LongTensor(y_train).to(self.device)
            self.can_guide = True
            
            # Identify minority class
            counts = np.bincount(y_train)
            self.minority_class_label = np.argmin(counts)
            print(f"Guidance enabled. Minority Class Label: {self.minority_class_label} (Count: {counts[self.minority_class_label]})")
            
        else:
            print(f"Warning: Target column '{target_col}' not found. Guidance disabled.")
            y_tensor = torch.zeros(X_train.shape[0], dtype=torch.long).to(self.device) # Dummy labels
            self.can_guide = False
        
        # Indices for VAE Loss
        num_count = len(self.num_cols)
        self.num_cols_idx = list(range(num_count))
        self.cat_cols_idx_list = []
        current_idx = num_count
        
        if len(self.cat_cols) > 0:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_encoder, 'categories_'):
                for categories in cat_encoder.categories_:
                    n_cats = len(categories)
                    self.cat_cols_idx_list.append((current_idx, current_idx + n_cats))
                    current_idx += n_cats
            
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        
        # Initialize VAE
        self.vae = TabularVAE(input_dim=self.output_dim, hidden_dim=256, latent_dim=self.latent_dim).to(self.device)
        vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        
        # Train VAE
        batch_size = 64
        n_samples = X_tensor.shape[0]
        
        # tqdm for VAE
        from tqdm import tqdm
        vae_pbar = tqdm(range(self.vae_epochs), desc="VAE Training")
        
        for epoch in vae_pbar:
            epoch_loss = 0
            permutation = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = X_tensor[indices]
                
                vae_optimizer.zero_grad()
                recon_x, mu, logvar = self.vae(batch_x)
                loss, _, _, _ = self.vae.loss_function(recon_x, batch_x, mu, logvar, self.num_cols_idx, self.cat_cols_idx_list)
                loss.backward()
                vae_optimizer.step()
                epoch_loss += loss.item()
            
            # Update progress bar
            vae_pbar.set_postfix({'loss': f"{epoch_loss / n_samples:.4f}"})
        
        print("VAE Trained. Freezing VAE.")
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # --- Step 2: Train Latent Diffusion ---
        print(f"--- Phase 2: Training Latent Diffusion ({self.epochs} epochs) ---")
        
        # Encode Data to Latent Space
        with torch.no_grad():
            mu, logvar = self.vae.encoder(X_tensor)
            # We train diffusion on the posterior mean (or sample? usually mean is more stable for simple setups, but sample is more correct. Let's use mu for stability first, or reparameterize.)
            # TabSyn uses latent features. Let's use reparameterize to capture variance.
            z_train = self.vae.reparameterize(mu, logvar)
            
        # Initialize Diffusion
        if self.backbone == 'transformer':
            print("Using Transformer Backbone (Tabular DiT)...")
            model_backbone = TabularTransformer(d_in=self.latent_dim, d_model=256, nhead=4, num_layers=4, d_cond=64).to(self.device)
        else:
            print("Using MLP Backbone...")
            model_backbone = LatentMLP(latent_dim=self.latent_dim, hidden_dim=256).to(self.device)
            model_backbone.class_emb = torch.nn.Embedding(3, 256).to(self.device)
        
        self.diffusion = LatentDiffusion(model_backbone, timesteps=1000, device=self.device)
        diff_optimizer = torch.optim.Adam(model_backbone.parameters(), lr=1e-3)
        
        # Differential Privacy Setup
        privacy_engine = None
        if self.target_epsilon is not None:
            if not OPACUS_AVAILABLE:
                raise ImportError("Opacus is required for Differential Privacy but not installed.")
            
            print(f"Enabling Differential Privacy (Target Epsilon: {self.target_epsilon})...")
            
            # Create DataLoader for Opacus (it needs a DataLoader, not raw tensors)
            from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
            
            # Combine z and y for DataLoader
            train_dataset = TensorDataset(z_train, y_tensor)
            train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            privacy_engine = PrivacyEngine()
            
            # Wrap model, optimizer, and data loader
            # Note: Opacus wraps the *module* that has parameters. 
            # Here self.diffusion.model is the one with parameters (model_backbone).
            # But self.diffusion itself is just a wrapper class, not an nn.Module.
            # So we should wrap model_backbone.
            
            # Fix incompatible modules (e.g., MultiheadAttention -> DPMultiheadAttention)
            from opacus.validators import ModuleValidator
            model_backbone = ModuleValidator.fix(model_backbone)
            
            # Re-initialize optimizer because model parameters have changed!
            diff_optimizer = torch.optim.Adam(model_backbone.parameters(), lr=1e-3)
            
            model_backbone, diff_optimizer, train_loader = privacy_engine.make_private(
                module=model_backbone,
                optimizer=diff_optimizer,
                data_loader=train_loader,
                noise_multiplier=1.0, 
                max_grad_norm=1.0,
            )
            
            # CRITICAL: Update the diffusion wrapper to use the DP-wrapped model!
            self.diffusion.model = model_backbone
            
            print("Opacus PrivacyEngine attached.")
            
        else:
            # Standard Scheduler
            scheduler = CosineAnnealingLR(diff_optimizer, T_max=self.epochs)
            train_loader = None # We use manual loop below for non-DP
        
        # Train Diffusion
        diff_pbar = tqdm(range(self.epochs), desc="Diffusion Training")
        
        for epoch in diff_pbar:
            epoch_loss = 0
            
            if self.target_epsilon is not None:
                # DP Training Loop using DataLoader
                # Opacus requires using the wrapped DataLoader
                with torch.enable_grad(): # Ensure grad is enabled
                    for batch_z, batch_y in train_loader:
                        # Opacus handles device moving usually, but let's be safe
                        batch_z = batch_z.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        # We need to call train_step but train_step does optimizer.step().
                        # Opacus optimizer.step() does the noise addition.
                        # But our train_step does: optimizer.zero_grad(), loss.backward(), optimizer.step().
                        # This should be fine if we pass the wrapped optimizer.
                        # BUT: train_step generates 't' and 'noise' internally.
                        # Opacus needs the per-sample gradients.
                        # 'loss.backward()' on the loss from 'model(x)' works.
                        # So we can reuse train_step?
                        # Yes, as long as self.diffusion.model is the wrapped module.
                        
                        loss = self.diffusion.train_step(batch_z, batch_y, diff_optimizer, guidance_prob=0.1)
                        epoch_loss += loss
                
                # Calculate current epsilon
                epsilon = privacy_engine.get_epsilon(self.target_delta)
                avg_loss = epoch_loss / len(train_loader)
                diff_pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'eps': f"{epsilon:.2f}"})
                     
                if epsilon > self.target_epsilon:
                    print(f"Target epsilon {self.target_epsilon} reached at epoch {epoch+1}. Stopping.")
                    break
                    
            else:
                # Standard Training Loop
                permutation = torch.randperm(n_samples)
                for i in range(0, n_samples, batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_z = z_train[indices]
                    batch_y = y_tensor[indices]
                    
                    loss = self.diffusion.train_step(batch_z, batch_y, diff_optimizer, guidance_prob=0.1)
                    epoch_loss += loss
                
                scheduler.step()
                avg_loss = epoch_loss / (n_samples/batch_size)
                diff_pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
        
        self.is_trained = True
        print(f"{self.name} trained.")

    def generate(self, num_samples, minority_ratio=None):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        print(f"Generating {num_samples} samples with {self.name} (Guidance Scale: {self.guidance_scale})...")
        
        y_gen = None
        current_guidance_scale = self.guidance_scale
        
        if self.can_guide and self.minority_class_label is not None:
            # Determine target ratio (default to 50/50 if not specified)
            ratio = minority_ratio if minority_ratio is not None else 0.5
            
            num_minority = int(num_samples * ratio)
            num_majority = num_samples - num_minority
            
            # Identify majority label (assuming binary: 0 or 1)
            majority_label = 1 - self.minority_class_label
            
            # Create label tensor
            labels = np.array([self.minority_class_label] * num_minority + [majority_label] * num_majority)
            np.random.shuffle(labels)
            y_gen = torch.LongTensor(labels).to(self.device)
            
            print(f"   Target Ratio: {ratio:.2f} (Minority: {num_minority}, Majority: {num_majority})")
        else:
            print("   Guidance disabled (Unconditional generation).")
            current_guidance_scale = 0.0
            y_gen = None
        
        # Sample z from Diffusion
        z_gen = self.diffusion.sample(num_samples, self.latent_dim, y_cond=y_gen, guidance_scale=current_guidance_scale)
        
        # Decode z with VAE
        with torch.no_grad():
            recon_x = self.vae.decoder(z_gen)
            
        # Inverse Transform
        synthetic_data = self._inverse_transform(recon_x.cpu().numpy())
        
        # We need to overwrite the target column with the labels we conditioned on!
        if self.target_col and self.label_encoder and y_gen is not None:
            synthetic_data[self.target_col] = self.label_encoder.inverse_transform(y_gen.cpu().numpy())
            
        return synthetic_data

    def _inverse_transform(self, vectors_np):
        # (Same as VAEWrapper._inverse_transform)
        # ... copy logic ...
        num_count = len(self.num_cols)
        num_part = vectors_np[:, :num_count]
        cat_part = vectors_np[:, num_count:]
        
        num_restored = self.preprocessor.named_transformers_['num'].inverse_transform(num_part)
        df_num = pd.DataFrame(num_restored, columns=self.num_cols)
        
        cat_restored_dict = {}
        
        if len(self.cat_cols) > 0:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_encoder, 'categories_'):
                cat_categories = cat_encoder.categories_
                current_idx = 0
                for i, col_name in enumerate(self.cat_cols):
                    n_cats = len(cat_categories[i])
                    col_slice = cat_part[:, current_idx : current_idx + n_cats]
                    class_indices = np.argmax(col_slice, axis=1)
                    cat_restored_dict[col_name] = cat_categories[i][class_indices]
                    current_idx += n_cats
                    
        df_cat = pd.DataFrame(cat_restored_dict)
        
        synthetic_data = pd.concat([df_num, df_cat], axis=1)
        synthetic_data = synthetic_data[self.original_columns]
        synthetic_data = synthetic_data.infer_objects()
        return synthetic_data




class GANWrapper(BaseModel):
    def __init__(self, epochs=10):
        super().__init__("CTGAN")
        self.epochs = epochs
        self.model = CTGAN(epochs=epochs, verbose=True)

    def train(self, train_data, target_col=None):
        print(f"Training {self.name} for {self.epochs} epochs...")
        # Detect discrete columns
        discrete_columns = train_data.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Discrete columns: {discrete_columns}")
        
        self.model.fit(train_data, discrete_columns=discrete_columns)
        self.is_trained = True
        print(f"{self.name} trained.")

    def generate(self, num_samples):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        print(f"Generating {num_samples} samples with {self.name}...")
        return self.model.sample(num_samples)

class TVAEWrapper(BaseModel):
    def __init__(self, epochs=10):
        super().__init__("TVAE")
        self.epochs = epochs
        # TVAE from SDV/CTGAN ecosystem
        self.model = None # Initialize in train to use metadata

    def train(self, train_data, target_col=None):
        print(f"Training {self.name} for {self.epochs} epochs...")
        
        # Use SDV SingleTableSynthesizer interface which handles metadata
        from sdv.single_table import TVAESynthesizer
        from sdv.metadata import SingleTableMetadata
        
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data=train_data)
        
        self.model = TVAESynthesizer(metadata=self.metadata, epochs=self.epochs)
        self.model.fit(train_data)
        self.is_trained = True
        print(f"{self.name} trained.")

    def generate(self, num_samples):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        print(f"Generating {num_samples} samples with {self.name}...")
        return self.model.sample(num_samples)

class LLMWrapper(BaseModel):
    def __init__(self):
        super().__init__("TabLLM_Placeholder")
    
    def train(self, train_data):
        print(f"Skipping {self.name} training (requires heavy GPU resources).")
        self.train_data_stats = {
            'mean': train_data.select_dtypes(include=np.number).mean(),
            'std': train_data.select_dtypes(include=np.number).std(),
            'columns': train_data.columns
        }
        self.is_trained = True

    def generate(self, num_samples):
        print(f"Generating {num_samples} samples with {self.name} (Gaussian Approx)...")
        # Simple Gaussian approximation for the placeholder
        synthetic_data = pd.DataFrame()
        for col in self.train_data_stats['columns']:
            if col in self.train_data_stats['mean']:
                mean = self.train_data_stats['mean'][col]
                std = self.train_data_stats['std'][col]
                synthetic_data[col] = np.random.normal(loc=mean, scale=std, size=num_samples)
            else:
                # Fill categorical with random choice? No, just leave empty or fill with mode?
                # For now, just skip or fill with 0 to avoid breaking evaluator
                synthetic_data[col] = 0 
        return synthetic_data
