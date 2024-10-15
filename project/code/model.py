# code/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) architecture for encoding and decoding piano roll sequences.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initializes the VAE with encoder and decoder networks.

        Args:
            input_dim (int): Dimensionality of the input data (flattened piano roll).
            hidden_dims (list): List of hidden layer sizes for the encoder.
            latent_dim (int): Dimensionality of the latent space (z).
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Build Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
        Encodes the input into latent space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (mu, logvar) of the latent distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log-variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent vector back to the original input space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        h = self.decoder(z)
        x_recon = self.output_layer(h)
        x_recon = self.sigmoid(x_recon)
        return x_recon

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (reconstructed input, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
