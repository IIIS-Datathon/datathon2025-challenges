import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x
    
class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax for differentiable categorical sampling"""
    def __init__(self, tau=1.0, hard=False):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)
    
class TransformerVAE(nn.Module):
    """Transformer-based VAE for tabular data"""
    def __init__(self, input_dim, categorical_indices, numerical_indices,
                 categorical_dims, latent_dim=128, d_model=256, num_heads=4,
                 num_layers=3, d_ff=512, dropout=0.1, tau=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices
        self.categorical_dims = categorical_dims
        self.tau = tau

        # Feature embedding
        self.feature_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.position_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))

        # Encoder - Transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Latent space projection
        self.fc_mu = nn.Linear(d_model * input_dim, latent_dim)
        self.fc_logvar = nn.Linear(d_model * input_dim, latent_dim)

        # Decoder - from latent to features
        self.decoder_input = nn.Linear(latent_dim, d_model * input_dim)

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output heads for numerical features
        self.numerical_head = nn.Linear(d_model, 1)

        # Output heads for categorical features (Gumbel-Softmax)
        self.categorical_heads = nn.ModuleList([
            nn.Linear(d_model, num_classes) for num_classes in categorical_dims
        ])
        self.gumbel_softmax = GumbelSoftmax(tau=tau, hard=False)

    def encode(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)

        # Embed each feature
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        x = self.feature_embedding(x)  # (batch_size, input_dim, d_model)

        # Add positional encoding
        x = x + self.position_embedding

        # Pass through transformer blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Flatten and project to latent space
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z shape: (batch_size, latent_dim)
        batch_size = z.size(0)

        # Project from latent space
        x = self.decoder_input(z)
        x = x.view(batch_size, self.input_dim, -1)

        # Pass through decoder transformer blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Reconstruct features
        reconstructed = torch.zeros(batch_size, self.input_dim).to(z.device)

        # Numerical features
        for idx in self.numerical_indices:
            reconstructed[:, idx] = self.numerical_head(x[:, idx]).squeeze(-1)

        # Categorical features with Gumbel-Softmax
        for cat_idx, head in enumerate(self.categorical_heads):
            feat_idx = self.categorical_indices[cat_idx]
            logits = head(x[:, feat_idx])
            # Sample from Gumbel-Softmax
            probs = self.gumbel_softmax(logits)
            # Convert back to class index (soft argmax)
            class_indices = torch.sum(probs * torch.arange(probs.size(-1)).float().to(z.device), dim=-1)
            reconstructed[:, feat_idx] = class_indices

        return reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def encode_to_latent(self, x):
        """Encode input to latent representation (deterministic)"""
        mu, _ = self.encode(x)
        return mu