# code/utils.py

import os
import matplotlib.pyplot as plt
import torch

def plot_losses(train_losses, test_losses, save_path=None):
    """
    Plots the training and test loss curves.

    Parameters:
    - train_losses (list of float): List of training loss values per epoch.
    - test_losses (list of float): List of test loss values per epoch.
    - save_path (str, optional): Path to save the plot image. If None, the plot is shown.
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()

def create_dir(directory):
    """
    Creates a directory if it does not already exist.

    Parameters:
    - directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def save_model(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer states to a file.

    Parameters:
    - model (nn.Module): The VAE model to save.
    - optimizer (Optimizer): The optimizer to save.
    - epoch (int): The current epoch number.
    - loss (float): The loss at the time of saving.
    - path (str): The file path to save the model.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path, device):
    """
    Loads the model and optimizer states from a file.

    Parameters:
    - model (nn.Module): The VAE model to load.
    - optimizer (Optimizer): The optimizer to load.
    - path (str): The file path from where to load the model.
    - device (torch.device): The device to map the model parameters.

    Returns:
    - epoch (int): The epoch number at which the model was saved.
    - loss (float): The loss value at the time of saving.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {path} at epoch {epoch} with loss {loss}")
    return epoch, loss
