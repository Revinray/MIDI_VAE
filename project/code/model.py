# code/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Multistream(nn.Module):
    """
    Variational Autoencoder (VAE) architecture for encoding and decoding pitch, instrument, and velocity rolls with style embedding.
    """

    def __init__(self, input_dims, hidden_dims, latent_dim, num_composers, style_embed_dim):
        """
        Initializes the VAE with encoder and decoder networks.

        Args:
            input_dims (dict): Dictionary containing input dimensions for 'pitch', 'instrument', and 'velocity'.
            hidden_dims (dict): Dictionary containing hidden dimensions for 'pitch', 'instrument', 'velocity', and 'combined'.
            latent_dim (int): Dimensionality of the latent space (z).
            num_composers (int): Number of unique composers (styles).
            style_embed_dim (int): Dimension of the style (composer) embedding.
        """
        super(VAE_Multistream, self).__init__()
        self.latent_dim = latent_dim

        # Style Embedding
        self.style_embedding = nn.Embedding(num_composers, style_embed_dim)

        # Encoder GRUs
        self.pitch_encoder_gru = nn.GRU(input_size=input_dims['pitch'], hidden_size=hidden_dims['pitch'], num_layers=2, batch_first=True)
        self.instr_encoder_gru = nn.GRU(input_size=input_dims['instrument'], hidden_size=hidden_dims['instrument'], num_layers=1, batch_first=True)
        self.vel_encoder_gru = nn.GRU(input_size=input_dims['velocity'], hidden_size=hidden_dims['velocity'], num_layers=1, batch_first=True)

        # Combine GRU outputs
        combined_hidden = hidden_dims['pitch'] + hidden_dims['instrument'] + hidden_dims['velocity']
        self.fc = nn.Linear(combined_hidden, hidden_dims['combined'])

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims['combined'], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims['combined'], latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim + style_embed_dim, hidden_dims['combined'])

        # Decoder GRUs
        self.pitch_decoder_gru = nn.GRU(input_size=hidden_dims['combined'], hidden_size=input_dims['pitch'], num_layers=2, batch_first=True)
        self.instr_decoder_gru = nn.GRU(input_size=hidden_dims['combined'], hidden_size=input_dims['instrument'], num_layers=1, batch_first=True)
        self.vel_decoder_gru = nn.GRU(input_size=hidden_dims['combined'], hidden_size=input_dims['velocity'], num_layers=1, batch_first=True)

    def encode(self, pitch, instrument, velocity):
        """
        Encodes the input into latent space.

        Args:
            pitch (torch.Tensor): Pitch roll tensor of shape (batch, seq, 128).
            instrument (torch.Tensor): Instrument roll tensor of shape (batch, seq, num_instruments).
            velocity (torch.Tensor): Velocity roll tensor of shape (batch, seq, 128).

        Returns:
            tuple: (mu, logvar) of the latent distribution.
        """
        _, h_pitch = self.pitch_encoder_gru(pitch)
        h_pitch = h_pitch[-1]  # Shape: (batch, hidden_dim_pitch)

        _, h_instr = self.instr_encoder_gru(instrument)
        h_instr = h_instr[-1]  # Shape: (batch, hidden_dim_instr)

        _, h_vel = self.vel_encoder_gru(velocity)
        h_vel = h_vel[-1]      # Shape: (batch, hidden_dim_vel)

        # Concatenate all hidden states
        h = torch.cat([h_pitch, h_instr, h_vel], dim=1)  # Shape: (batch, combined_hidden)

        h = F.relu(self.fc(h))  # Shape: (batch, combined_hidden)

        mu = self.fc_mu(h)       # Shape: (batch, latent_dim)
        logvar = self.fc_logvar(h)  # Shape: (batch, latent_dim)

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
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, composer_ids):
        """
        Decodes the latent vector back to the original input space.

        Args:
            z (torch.Tensor): Latent vector of shape (batch, latent_dim).
            composer_ids (torch.Tensor): Composer IDs of shape (batch,).

        Returns:
            tuple: Reconstructed pitch, instrument, and velocity rolls.
        """
        style_embed = self.style_embedding(composer_ids)  # Shape: (batch, style_embed_dim)
        z = torch.cat([z, style_embed], dim=1)           # Shape: (batch, latent_dim + style_embed_dim)

        h_dec = F.relu(self.decoder_fc(z))              # Shape: (batch, combined_hidden)

        # Repeat for sequence length
        seq_length = self.sequence_length  # Ensure this is accessible or passed appropriately
        h_dec = h_dec.unsqueeze(1).repeat(1, seq_length, 1)  # Shape: (batch, seq_length, combined_hidden)

        # Decode pitch
        pitch_recon, _ = self.pitch_decoder_gru(h_dec)    # Shape: (batch, seq_length, 128)
        pitch_recon = torch.sigmoid(pitch_recon)

        # Decode instrument
        instr_recon, _ = self.instr_decoder_gru(h_dec)    # Shape: (batch, seq_length, num_instruments)
        instr_recon = torch.sigmoid(instr_recon)

        # Decode velocity
        vel_recon, _ = self.vel_decoder_gru(h_dec)        # Shape: (batch, seq_length, 128)
        vel_recon = torch.sigmoid(vel_recon)

        return pitch_recon, instr_recon, vel_recon

    def forward(self, pitch, instrument, velocity, composer_ids):
        """
        Forward pass through the VAE.

        Args:
            pitch (torch.Tensor): Pitch roll tensor of shape (batch, seq, 128).
            instrument (torch.Tensor): Instrument roll tensor of shape (batch, seq, num_instruments).
            velocity (torch.Tensor): Velocity roll tensor of shape (batch, seq, 128).
            composer_ids (torch.Tensor): Composer IDs of shape (batch,).

        Returns:
            tuple: (pitch_recon, instr_recon, vel_recon, mu, logvar)
        """
        mu, logvar = self.encode(pitch, instrument, velocity)
        z = self.reparameterize(mu, logvar)
        pitch_recon, instr_recon, vel_recon = self.decode(z, composer_ids)
        return pitch_recon, instr_recon, vel_recon, mu, logvar
