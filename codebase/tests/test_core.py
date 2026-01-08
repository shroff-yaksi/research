"""
Unit Tests for RE-TabSyn Codebase
Tests for core modules: VAE, Evaluator, and LatentDiffusion

Run with: python -m pytest tests/test_core.py -v
"""

import pytest
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vae import TabularVAE, Encoder, Decoder
from latent_diffusion import LatentMLP, LatentDiffusion
from evaluator import Evaluator


# ============================================================================
# VAE Tests
# ============================================================================

class TestEncoder:
    """Tests for Encoder class."""
    
    def test_encoder_init(self):
        """Test Encoder initialization."""
        encoder = Encoder(input_dim=100, hidden_dim=256, latent_dim=64)
        assert encoder is not None
        assert encoder.fc_mu.out_features == 64
        assert encoder.fc_logvar.out_features == 64
    
    def test_encoder_forward(self):
        """Test Encoder forward pass."""
        encoder = Encoder(input_dim=100, hidden_dim=256, latent_dim=64)
        x = torch.randn(32, 100)
        mu, logvar = encoder(x)
        
        assert mu.shape == (32, 64)
        assert logvar.shape == (32, 64)


class TestDecoder:
    """Tests for Decoder class."""
    
    def test_decoder_init(self):
        """Test Decoder initialization."""
        decoder = Decoder(latent_dim=64, hidden_dim=256, output_dim=100)
        assert decoder is not None
    
    def test_decoder_forward(self):
        """Test Decoder forward pass."""
        decoder = Decoder(latent_dim=64, hidden_dim=256, output_dim=100)
        z = torch.randn(32, 64)
        output = decoder(z)
        
        assert output.shape == (32, 100)


class TestTabularVAE:
    """Tests for TabularVAE class."""
    
    def test_vae_init(self):
        """Test VAE initialization."""
        vae = TabularVAE(input_dim=100, hidden_dim=256, latent_dim=64)
        assert vae is not None
        assert vae.encoder is not None
        assert vae.decoder is not None
    
    def test_vae_forward(self):
        """Test VAE forward pass."""
        vae = TabularVAE(input_dim=100, hidden_dim=256, latent_dim=64)
        x = torch.randn(32, 100)
        recon_x, mu, logvar = vae(x)
        
        assert recon_x.shape == x.shape
        assert mu.shape == (32, 64)
        assert logvar.shape == (32, 64)
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        vae = TabularVAE(input_dim=100)
        mu = torch.zeros(32, 64)
        logvar = torch.zeros(32, 64)
        z = vae.reparameterize(mu, logvar)
        
        assert z.shape == (32, 64)
        # With logvar=0, std=1, so z should be close to mu + gaussian noise
    
    def test_loss_function_numerical_only(self):
        """Test loss function with numerical columns only."""
        vae = TabularVAE(input_dim=10)
        x = torch.randn(32, 10)
        recon_x = torch.randn(32, 10)
        mu = torch.randn(32, 64)
        logvar = torch.randn(32, 64)
        
        num_cols_idx = list(range(10))
        cat_cols_idx_list = []
        
        total_loss, mse, ce, kld = vae.loss_function(
            recon_x, x, mu, logvar, num_cols_idx, cat_cols_idx_list
        )
        
        assert total_loss > 0
        assert mse > 0
        assert ce == 0  # No categorical columns
        assert kld != 0  # KL divergence should be non-zero
    
    def test_loss_function_mixed(self):
        """Test loss function with mixed numerical and categorical columns."""
        vae = TabularVAE(input_dim=15)
        x = torch.randn(32, 15)
        # Make last 5 columns one-hot encoded (1 categorical with 5 classes)
        x[:, 10:] = torch.nn.functional.one_hot(
            torch.randint(0, 5, (32,)), num_classes=5
        ).float()
        
        recon_x = torch.randn(32, 15)
        mu = torch.randn(32, 64)
        logvar = torch.randn(32, 64)
        
        num_cols_idx = list(range(10))
        cat_cols_idx_list = [(10, 15)]
        
        total_loss, mse, ce, kld = vae.loss_function(
            recon_x, x, mu, logvar, num_cols_idx, cat_cols_idx_list
        )
        
        assert total_loss > 0
        assert mse > 0
        assert ce > 0  # Should have categorical loss
        assert kld != 0


# ============================================================================
# Latent Diffusion Tests
# ============================================================================

class TestLatentMLP:
    """Tests for LatentMLP class."""
    
    def test_latent_mlp_init(self):
        """Test LatentMLP initialization."""
        model = LatentMLP(latent_dim=64, hidden_dim=256)
        assert model is not None
    
    def test_latent_mlp_forward_conditional(self):
        """Test LatentMLP forward pass with conditioning."""
        model = LatentMLP(latent_dim=64, hidden_dim=256)
        x = torch.randn(32, 64)
        t = torch.rand(32, 1)
        y = torch.randint(0, 2, (32,))
        
        output = model(x, t, y)
        assert output.shape == (32, 64)
    
    def test_latent_mlp_forward_unconditional(self):
        """Test LatentMLP forward pass without conditioning."""
        model = LatentMLP(latent_dim=64, hidden_dim=256)
        x = torch.randn(32, 64)
        t = torch.rand(32, 1)
        
        output = model(x, t, y=None)
        assert output.shape == (32, 64)


class TestLatentDiffusion:
    """Tests for LatentDiffusion class."""
    
    @pytest.fixture
    def diffusion(self):
        """Create a LatentDiffusion instance for testing."""
        model = LatentMLP(latent_dim=64, hidden_dim=128)
        model.class_emb = torch.nn.Embedding(3, 128)  # 3 classes for CFG
        return LatentDiffusion(model, timesteps=100, device='cpu')
    
    def test_latent_diffusion_init(self, diffusion):
        """Test LatentDiffusion initialization."""
        assert diffusion is not None
        assert diffusion.timesteps == 100
        assert len(diffusion.betas) == 100
    
    def test_q_sample(self, diffusion):
        """Test forward diffusion process."""
        x_0 = torch.randn(32, 64)
        t = torch.randint(0, 100, (32,))
        
        x_t, noise = diffusion.q_sample(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
    
    def test_train_step(self, diffusion):
        """Test training step."""
        x_0 = torch.randn(16, 64)
        y = torch.randint(0, 2, (16,))
        optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=1e-3)
        
        loss = diffusion.train_step(x_0, y, optimizer, guidance_prob=0.1)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_sample_unconditional(self, diffusion):
        """Test unconditional sampling."""
        samples = diffusion.sample(
            num_samples=8,
            latent_dim=64,
            y_cond=None,
            guidance_scale=0.0
        )
        
        assert samples.shape == (8, 64)
    
    def test_sample_conditional(self, diffusion):
        """Test conditional sampling with CFG."""
        y_cond = torch.ones(8, dtype=torch.long)  # All minority class
        
        samples = diffusion.sample(
            num_samples=8,
            latent_dim=64,
            y_cond=y_cond,
            guidance_scale=2.0
        )
        
        assert samples.shape == (8, 64)


# ============================================================================
# Evaluator Tests
# ============================================================================

class TestEvaluator:
    """Tests for Evaluator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample real and synthetic data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        real_data = pd.DataFrame({
            'num1': np.random.randn(n_samples),
            'num2': np.random.randn(n_samples) * 2 + 5,
            'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        synthetic_data = pd.DataFrame({
            'num1': np.random.randn(n_samples) + 0.1,  # Slight shift
            'num2': np.random.randn(n_samples) * 2.1 + 5.1,
            'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])  # Balanced
        })
        
        return real_data, synthetic_data
    
    def test_evaluator_init(self, sample_data):
        """Test Evaluator initialization."""
        real_data, synthetic_data = sample_data
        evaluator = Evaluator(real_data, synthetic_data, target_col='target', minority_class=1)
        
        assert evaluator is not None
        assert evaluator.target_col == 'target'
        assert evaluator.minority_class == 1
    
    def test_evaluate_fidelity(self, sample_data):
        """Test fidelity evaluation."""
        real_data, synthetic_data = sample_data
        evaluator = Evaluator(real_data, synthetic_data)
        
        results = evaluator.evaluate_fidelity()
        
        assert 'avg_ks' in results
        assert 0 <= results['avg_ks'] <= 1  # KS statistic is between 0 and 1
    
    def test_evaluate_privacy(self, sample_data):
        """Test privacy evaluation (DCR)."""
        real_data, synthetic_data = sample_data
        evaluator = Evaluator(real_data, synthetic_data)
        
        results = evaluator.evaluate_privacy()
        
        assert 'min_dcr' in results
        assert 'avg_dcr' in results
        assert results['min_dcr'] >= 0
        assert results['avg_dcr'] >= 0
    
    def test_evaluate_rare_events(self, sample_data):
        """Test rare event evaluation."""
        real_data, synthetic_data = sample_data
        evaluator = Evaluator(real_data, synthetic_data, target_col='target', minority_class=1)
        
        results = evaluator.evaluate_rare_events()
        
        assert 'real_minority_ratio' in results
        assert 'syn_minority_ratio' in results
        assert 'ratio_diff' in results
        # Synthetic data should have higher minority ratio (we generated it balanced)
        assert results['syn_minority_ratio'] > results['real_minority_ratio']
    
    def test_run_all(self, sample_data):
        """Test running all evaluations."""
        real_data, synthetic_data = sample_data
        evaluator = Evaluator(real_data, synthetic_data, target_col='target', minority_class=1)
        
        results = evaluator.run_all()
        
        # Should contain all metric keys
        assert 'avg_ks' in results
        assert 'min_dcr' in results
        assert 'avg_dcr' in results
        assert 'real_minority_ratio' in results
        assert 'syn_minority_ratio' in results


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_vae_encode_decode_roundtrip(self):
        """Test that VAE can encode and decode without dimension errors."""
        vae = TabularVAE(input_dim=50, hidden_dim=128, latent_dim=32)
        x = torch.randn(16, 50)
        
        # Forward pass
        recon_x, mu, logvar = vae(x)
        
        # Check shapes
        assert recon_x.shape == x.shape
        
        # Encode separately
        mu2, logvar2 = vae.encoder(x)
        z = vae.reparameterize(mu2, logvar2)
        
        # Decode
        recon_x2 = vae.decoder(z)
        assert recon_x2.shape == x.shape
    
    def test_diffusion_training_loop(self):
        """Test a mini training loop for diffusion."""
        model = LatentMLP(latent_dim=32, hidden_dim=64)
        model.class_emb = torch.nn.Embedding(3, 64)
        diffusion = LatentDiffusion(model, timesteps=50, device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x_0 = torch.randn(8, 32)
        y = torch.randint(0, 2, (8,))
        
        initial_loss = None
        for i in range(5):
            loss = diffusion.train_step(x_0, y, optimizer)
            if i == 0:
                initial_loss = loss
        
        # Loss should decrease (or at least not explode)
        assert loss < initial_loss * 10  # Sanity check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
