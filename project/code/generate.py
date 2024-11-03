# code/generate.py

import torch
import numpy as np
import pretty_midi
import os
from tqdm import tqdm

def piano_roll_to_midi(pitch_roll, instrument_roll, velocity_roll, fs=100, filename='output.mid'):
    """
    Converts pitch, instrument, and velocity rolls to a MIDI file.

    Parameters:
    - pitch_roll (np.ndarray): Binary pitch roll of shape (128, time_steps).
    - instrument_roll (np.ndarray): Binary instrument roll of shape (num_instruments, time_steps).
    - velocity_roll (np.ndarray): Velocity roll of shape (128, time_steps).
    - fs (int): Frames per second.
    - filename (str): Output filename.
    """
    pm = pretty_midi.PrettyMIDI()
    num_instruments = instrument_roll.shape[0]

    for instr_id in range(num_instruments):
        if np.sum(instrument_roll[instr_id]) == 0:
            continue
        instrument = pretty_midi.Instrument(program=instr_id)  # Adjust mapping as needed

        active_pitches = np.where(pitch_roll[:, :] > 0)
        for pitch, time_step in zip(*active_pitches):
            velocity = int(velocity_roll[pitch, time_step] * 127)
            start_time = time_step / fs
            end_time = (time_step + 1) / fs
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
            instrument.notes.append(note)

        pm.instruments.append(instrument)

    pm.write(filename)
    print(f"MIDI file saved to {filename}")

def reconstruct_and_save_midi(model, dataloader, device, output_dir, num_samples=10):
    """
    Reconstructs MIDI files from the test dataset using the trained VAE model.

    Parameters:
    - model (nn.Module): The trained VAE model.
    - dataloader (DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform computations on.
    - output_dir (str): Directory where reconstructed MIDI files will be saved.
    - num_samples (int): Number of samples to reconstruct.

    Returns:
    - None
    """
    model.eval()
    samples_reconstructed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Reconstructing MIDI", leave=False)):
            pitch = batch['pitch'].to(device)
            instrument = batch['instrument'].to(device)
            velocity = batch['velocity'].to(device)
            composer = batch['composer'].to(device)

            pitch_recon, instr_recon, vel_recon, mu, logvar = model(pitch, instrument, velocity, composer)

            pitch_recon = pitch_recon.cpu().numpy()
            instr_recon = instr_recon.cpu().numpy()
            vel_recon = vel_recon.cpu().numpy()

            for i in range(pitch_recon.shape[0]):
                if samples_reconstructed >= num_samples:
                    return
                reconstructed_pitch = (pitch_recon[i] > 0.5).astype(np.float32)
                reconstructed_instrument = (instr_recon[i] > 0.5).astype(np.float32)
                reconstructed_velocity = vel_recon[i]

                filename = os.path.join(output_dir, f'reconstructed_{batch_idx}_{i}.mid')
                piano_roll_to_midi(reconstructed_pitch, reconstructed_instrument, reconstructed_velocity, filename=filename)
                samples_reconstructed += 1

def sample_latent_space(model, device, output_dir, num_samples=10, latent_dim=20, composer_id=0):
    """
    Generates new MIDI files by sampling from the latent space.

    Parameters:
    - model (nn.Module): The trained VAE model.
    - device (torch.device): Device to perform computations on.
    - output_dir (str): Directory where generated MIDI files will be saved.
    - num_samples (int): Number of MIDI files to generate.
    - latent_dim (int): Dimension of the latent space.
    - composer_id (int): ID of the composer style to embed.

    Returns:
    - None
    """
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Sampling MIDI"):
            # Sample from standard normal distribution
            z = torch.randn(1, latent_dim).to(device)
            composer = torch.tensor([composer_id], dtype=torch.long).to(device)

            pitch_recon, instr_recon, vel_recon, mu, logvar = model(z, composer)

            reconstructed_pitch = (pitch_recon.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
            reconstructed_instrument = (instr_recon.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
            reconstructed_velocity = vel_recon.squeeze(0).cpu().numpy()

            filename = os.path.join(output_dir, f'sampled_{i}.mid')
            piano_roll_to_midi(reconstructed_pitch, reconstructed_instrument, reconstructed_velocity, filename=filename)
