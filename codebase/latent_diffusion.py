import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class LatentMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Class embedding for Guidance (0=Majority, 1=Minority, 2=Null/Unconditional)
        self.class_emb = nn.Embedding(3, hidden_dim)

        # Main network
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, t, y=None):
        # x: [batch, latent_dim]
        # t: [batch, 1]
        # y: [batch] (class labels)
        
        h = self.input_proj(x)
        t_emb = self.time_mlp(t)
        
        # Add time embedding
        h = h + t_emb
        
        # Add class embedding if provided (Guidance)
        if y is not None:
            c_emb = self.class_emb(y)
            h = h + c_emb
            
        h = self.layers(h)
        return self.output_proj(h)

class LatentDiffusion:
    def __init__(self, model, timesteps=1000, device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t):
        """Forward diffusion: Add noise to x_0 at timestep t."""
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def train_step(self, x_0, y, optimizer, guidance_prob=0.1):
        """
        Train with Classifier-Free Guidance.
        y: Class labels (0 or 1).
        guidance_prob: Probability of dropping the label (setting to null/unconditional).
        """
        self.model.train()
        optimizer.zero_grad()
        
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t)
        
        # Classifier-Free Guidance: Randomly drop labels
        # We'll use a special label (e.g., 0 for majority, 1 for minority).
        # But for "unconditional", we need a way to signal it.
        # A common trick is to use a mask or a specific index if using embeddings.
        # Let's say: 0=Majority, 1=Minority. 
        # But wait, if we drop the label, we shouldn't pass it to the model?
        # Or we pass a "null" token. Let's assume the embedding has size 3: [Majority, Minority, Null].
        # So we need to resize the embedding in LatentMLP to 3.
        
        # Masking
        mask = torch.rand(batch_size, device=self.device) < guidance_prob
        
        # We need to handle the y input carefully.
        # If mask is True, we pass y=None (or a null token index).
        # Let's adjust LatentMLP to handle this. 
        # Actually, let's pass the raw y, and handle masking inside or before.
        # Let's assume y is a tensor of 0s and 1s.
        # We'll create a new y_in where masked entries are replaced by 2 (Null).
        
        y_in = y.clone()
        y_in[mask] = 2 
        
        t_tensor = t.view(-1, 1).float() / self.timesteps
        
        predicted_noise = self.model(x_t, t_tensor, y_in)
        
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples, latent_dim, y_cond=None, guidance_scale=0.0):
        """
        Sample with Classifier-Free Guidance.
        y_cond: Tensor of class labels to condition on. If None, unconditional.
        guidance_scale: w. Formula: noise = noise_uncond + w * (noise_cond - noise_uncond).
        """
        self.model.eval()
        x_t = torch.randn(num_samples, latent_dim, device=self.device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.tensor([i] * num_samples, device=self.device)
            t_input = t.view(-1, 1).float() / self.timesteps
            
            if guidance_scale > 0 and y_cond is not None:
                # Conditional forward
                noise_cond = self.model(x_t, t_input, y_cond)
                
                # Unconditional forward (y=2 for Null)
                y_null = torch.full((num_samples,), 2, device=self.device, dtype=torch.long)
                noise_uncond = self.model(x_t, t_input, y_null)
                
                # Combine
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # Standard unconditional (or just conditional if w=0? No, usually w=0 means standard sampling)
                # If y_cond is provided but w=0, we just use conditional?
                # Usually guidance_scale=1.0 means "use conditional". 
                # Wait, the formula is: eps = eps_uncond + w * (eps_cond - eps_uncond).
                # If w=1, eps = eps_cond.
                # If w=0, eps = eps_uncond.
                # Usually we want w > 1 for "guidance".
                # So if y_cond is None, we use y_null.
                
                if y_cond is None:
                    y_in = torch.full((num_samples,), 2, device=self.device, dtype=torch.long)
                else:
                    y_in = y_cond
                
                predicted_noise = self.model(x_t, t_input, y_in)

            # Reverse step
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            
            mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(beta_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean
                
        return x_t
