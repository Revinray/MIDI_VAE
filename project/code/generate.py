# code/generate.py

import torch
import numpy as np
import pretty_midi
import os
from tqdm import tqdm

def piano_roll_to_midi(piano_roll, fs=100, program=0, filename='output.mid'):
    """
    Converts a piano roll to a MIDI file.

    Parameters:
    - piano_roll (np.ndarray): Binary piano roll array of shape (128, time_steps).
    - fs (int): Frames per second (sampling frequency).
    - program (int): MIDI program number (instrument). 0 corresponds to Acoustic Grand Piano.
    - filename (str): Output filename for the MIDI file.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    piano_roll = piano_roll * 127  # Scale velocities to MIDI range

    # Iterate through each note (MIDI pitch)
    for note_num in range(piano_roll.shape[0]):
        # Find where the note is active
        indices = np.where(piano_roll[note_num, :] > 0)[0]
        if len(indices) == 0:
            continue

        # Group consecutive indices
        groups = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
        for group in groups:
            start = group[0] / fs
            end = (group[-1] + 1) / fs  # Add 1 to include the last frame
            velocity = int(piano_roll[note_num, group[0]])
            note = pretty_midi.Note(velocity=velocity,
                                    pitch=note_num,
                                    start=start,
                                    end=end)
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(filename)
    print(f"MIDI file saved to {filename}")

def reconstruct_and_save_midi(model, dataloader, device, output_dir, num_samples=10, latent_dim=20, layer_dims=[512]):
    """
    Reconstructs MIDI files from the test dataset using the trained VAE model.

    Parameters:
    - model (nn.Module): The trained VAE model.
    - dataloader (DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform computations on.
    - output_dir (str): Directory where reconstructed MIDI files will be saved.
    - num_samples (int): Number of samples to reconstruct.
    - latent_dim (int): Dimension of the latent space.
    - layer_dims (list of int): List of layer dimensions used in the model.

    Returns:
    - None
    """
    model.eval()
    samples_reconstructed = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Reconstructing MIDI", leave=False)):
            data = data.to(device)
            recon_batch, mu, log_sigma = model(data)
            recon_batch = recon_batch.cpu().numpy()

            for i in range(recon_batch.shape[0]):
                if samples_reconstructed >= num_samples:
                    return
                reconstructed = recon_batch[i]
                # Reshape to original piano roll shape
                # Assuming input_dim = 128 * sequence_length
                # Adjust sequence_length accordingly
                sequence_length = int(reconstructed.shape[0] / 128)
                piano_roll = reconstructed.reshape(128, sequence_length)
                # Binarize the piano roll
                piano_roll = (piano_roll > 0.5).astype(np.float32)
                # Save as MIDI
                filename = os.path.join(output_dir, f'reconstructed_{batch_idx}_{i}.mid')
                piano_roll_to_midi(piano_roll, filename=filename)
                samples_reconstructed += 1

def sample_latent_space(model, device, output_dir, num_samples=10, latent_dim=20):
    """
    Generates new MIDI files by sampling from the latent space.

    Parameters:
    - model (nn.Module): The trained VAE model.
    - device (torch.device): Device to perform computations on.
    - output_dir (str): Directory where generated MIDI files will be saved.
    - num_samples (int): Number of MIDI files to generate.
    - latent_dim (int): Dimension of the latent space.

    Returns:
    - None
    """
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Sampling MIDI"):
            # Sample from standard normal distribution
            z = torch.randn(1, latent_dim).to(device)
            # Decode the latent vector
            recon = model.decoder(z)
            recon = recon.cpu().numpy().flatten()
            # Reshape to piano roll
            sequence_length = int(recon.shape[0] / 128)
            piano_roll = recon.reshape(128, sequence_length)
            # Binarize the piano roll
            piano_roll = (piano_roll > 0.5).astype(np.float32)
            # Save as MIDI
            filename = os.path.join(output_dir, f'sampled_{i}.mid')
            piano_roll_to_midi(piano_roll, filename=filename)
