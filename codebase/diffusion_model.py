h/cxz
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPDiffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # Simple MLP backbone
        # Input: [batch, input_dim] concatenated with time embedding [batch, hidden_dim]? 
        # Actually, let's project time to hidden_dim and add it.
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # x: [batch, input_dim]
        # t: [batch, 1] (normalized 0-1 or raw steps)
        
        h = self.input_proj(x)
        t_emb = self.time_emb(t.float())
        
        h = h + t_emb # Add time embedding
        h = self.layers(h)
        return self.output_proj(h)

class GaussianDiffusion:
    def __init__(self, model, timesteps=100, device='cpu'):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t):
        """Forward diffusion: Add noise to data x_0 at timestep t."""
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise, noise

    def p_sample(self, x_t, t):
        """Reverse diffusion: Predict x_{t-1} from x_t."""
        # Predict noise
        t_tensor = torch.tensor([t] * x_t.shape[0], device=self.device).view(-1, 1)
        predicted_noise = self.model(x_t, t_tensor)
        
        # Coefficients
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        
        # Mean
        # mu = (1 / sqrt(alpha)) * (x_t - (beta / sqrt(1 - alpha_bar)) * noise)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            # sigma = sqrt(beta)
            sigma = torch.sqrt(beta_t)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
            
        # Clip to prevent explosion (data is N(0,1) so [-4, 4] is safe)
        return torch.clamp(x_prev, -4, 4)

    def train_step(self, x_0, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        batch_size = x_0.shape[0]
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t)
        
        # Predict noise
        t_input = t.view(-1, 1).float() / self.timesteps # Normalize time 0-1 for embedding
        predicted_noise = self.model(x_t, t_input)
        
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples, input_dim):
        self.model.eval()
        # Start from pure noise
        x = torch.randn((num_samples, input_dim), device=self.device)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
            
        return x
