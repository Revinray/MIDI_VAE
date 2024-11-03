# code/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def loss_function(pitch_recon, instr_recon, vel_recon, pitch, instrument, velocity, mu, logvar):
    """
    Computes the VAE loss function as the sum of reconstruction loss and KL divergence.

    Parameters:
    - pitch_recon (Tensor): Reconstructed pitch roll.
    - instr_recon (Tensor): Reconstructed instrument roll.
    - vel_recon (Tensor): Reconstructed velocity roll.
    - pitch (Tensor): Original pitch roll.
    - instrument (Tensor): Original instrument roll.
    - velocity (Tensor): Original velocity roll.
    - mu (Tensor): Mean from the encoder's latent space.
    - logvar (Tensor): Log variance from the encoder's latent space.

    Returns:
    - total_loss (Tensor): The combined loss.
    """
    # Reconstruction loss
    pitch_loss = nn.functional.binary_cross_entropy(pitch_recon, pitch, reduction='sum')
    instr_loss = nn.functional.binary_cross_entropy(instr_recon, instrument, reduction='sum')
    vel_loss = nn.functional.mse_loss(vel_recon, velocity, reduction='sum')  # Using MSE for velocity

    recon_loss = pitch_loss + instr_loss + vel_loss

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + KLD

def train_epoch(model, dataloader, optimizer, device):
    """
    Performs one epoch of training.

    Parameters:
    - model (nn.Module): The VAE model.
    - dataloader (DataLoader): DataLoader for the training data.
    - optimizer (Optimizer): Optimizer for updating model parameters.
    - device (torch.device): Device to perform computations on.

    Returns:
    - average_loss (float): Average loss over the epoch.
    """
    model.train()
    train_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        pitch = batch['pitch'].to(device)
        instrument = batch['instrument'].to(device)
        velocity = batch['velocity'].to(device)
        composer = batch['composer'].to(device)

        optimizer.zero_grad()
        pitch_recon, instr_recon, vel_recon, mu, logvar = model(pitch, instrument, velocity, composer)
        loss = loss_function(pitch_recon, instr_recon, vel_recon, pitch, instrument, velocity, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    average_loss = train_loss / len(dataloader.dataset)
    return average_loss

def test_epoch(model, dataloader, device):
    """
    Evaluates the model on the test dataset.

    Parameters:
    - model (nn.Module): The VAE model.
    - dataloader (DataLoader): DataLoader for the test data.
    - device (torch.device): Device to perform computations on.

    Returns:
    - average_loss (float): Average loss over the test dataset.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            pitch = batch['pitch'].to(device)
            instrument = batch['instrument'].to(device)
            velocity = batch['velocity'].to(device)
            composer = batch['composer'].to(device)

            pitch_recon, instr_recon, vel_recon, mu, logvar = model(pitch, instrument, velocity, composer)
            loss = loss_function(pitch_recon, instr_recon, vel_recon, pitch, instrument, velocity, mu, logvar)
            test_loss += loss.item()
    average_loss = test_loss / len(dataloader.dataset)
    return average_loss

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint.

    Parameters:
    - model (nn.Module): The VAE model.
    - optimizer (Optimizer): Optimizer with its state.
    - epoch (int): Current epoch number.
    - loss (float): Loss value at the checkpoint.
    - path (str): File path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path, device):
    """
    Loads the model checkpoint.

    Parameters:
    - model (nn.Module): The VAE model.
    - optimizer (Optimizer): Optimizer to load its state.
    - path (str): File path from where to load the checkpoint.
    - device (torch.device): Device to map the model parameters.

    Returns:
    - epoch (int): The epoch number at which the checkpoint was saved.
    - loss (float): The loss value at the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path} at epoch {epoch} with loss {loss}")
    return epoch, loss
