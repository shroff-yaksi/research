import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class TabularVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, num_cols_idx, cat_cols_idx_list):
        """
        Custom loss function for mixed tabular data.
        num_cols_idx: slice or list of indices for numerical columns
        cat_cols_idx_list: list of (start, end) tuples for each categorical column (one-hot groups)
        """
        # 1. Numerical Loss (MSE)
        if num_cols_idx is not None:
            recon_num = recon_x[:, num_cols_idx]
            x_num = x[:, num_cols_idx]
            mse_loss = F.mse_loss(recon_num, x_num, reduction='sum')
        else:
            mse_loss = 0

        # 2. Categorical Loss (Cross Entropy)
        ce_loss = 0
        for start, end in cat_cols_idx_list:
            recon_cat_logits = recon_x[:, start:end]
            x_cat_onehot = x[:, start:end]
            # Target for CrossEntropy should be class indices
            x_cat_idx = torch.argmax(x_cat_onehot, dim=1)
            ce_loss += F.cross_entropy(recon_cat_logits, x_cat_idx, reduction='sum')

        # 3. KL Divergence
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return mse_loss + ce_loss + kld_loss, mse_loss, ce_loss, kld_loss
