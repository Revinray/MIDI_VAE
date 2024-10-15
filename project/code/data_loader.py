# code/data_loader.py

import pretty_midi
import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MidiDataset(Dataset):
    """
    Custom Dataset for loading MIDI files and converting them into piano roll sequences.
    """

    def __init__(self, midi_files, sequence_length=500, fs=100):
        """
        Initializes the dataset by loading and processing MIDI files.

        Args:
            midi_files (list): List of paths to MIDI files.
            sequence_length (int): Number of time steps per sequence.
            fs (int): Sampling frequency (frames per second) for piano roll.
        """
        self.midi_files = midi_files
        self.sequence_length = sequence_length
        self.fs = fs
        self.data = []
        self.load_data()

    def load_data(self):
        """
        Loads MIDI files, converts them to piano rolls, splits into sequences, and stores them.
        """
        for midi_file in self.midi_files:
            try:
                piano_roll = self.midi_to_pianoroll(midi_file, fs=self.fs)
                piano_roll = (piano_roll > 0).astype(np.float32)  # Binarize to 0 or 1
                num_timesteps = piano_roll.shape[1]
                num_sequences = num_timesteps // self.sequence_length

                for i in range(num_sequences):
                    start = i * self.sequence_length
                    end = start + self.sequence_length
                    seq = piano_roll[:, start:end]
                    if seq.shape[1] == self.sequence_length:
                        self.data.append(seq.flatten())  # Flatten to 1D vector
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")

    def midi_to_pianoroll(self, midi_file, fs=100):
        """
        Converts a MIDI file to a piano roll.

        Args:
            midi_file (str): Path to the MIDI file.
            fs (int): Sampling frequency (frames per second).

        Returns:
            np.ndarray: Piano roll matrix (128 pitches x time steps).
        """
        pm = pretty_midi.PrettyMIDI(midi_file)
        piano_roll = pm.get_piano_roll(fs=fs)
        return piano_roll

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: Number of data samples.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a data sample at the specified index.

        Args:
            index (int): Index of the data sample.

        Returns:
            torch.Tensor: 1D tensor representing a flattened piano roll sequence.
        """
        return torch.tensor(self.data[index])

def prepare_dataloaders(raw_data_dir, sequence_length=500, fs=100, batch_size=64, test_size=0.2, random_state=42):
    """
    Prepares training and testing dataloaders from raw MIDI files.

    Args:
        raw_data_dir (str): Path to the directory containing raw MIDI files.
        sequence_length (int): Number of time steps per sequence.
        fs (int): Sampling frequency (frames per second).
        batch_size (int): Batch size for DataLoader.
        test_size (float): Proportion of data to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Collect all MIDI file paths
    midi_files = []
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                midi_files.append(os.path.join(root, file))

    print(f"Found {len(midi_files)} MIDI files.")

    # Initialize the dataset
    full_dataset = MidiDataset(midi_files, sequence_length=sequence_length, fs=fs)

    # Split into training and testing sets
    train_data, test_data = train_test_split(full_dataset.data, test_size=test_size, random_state=random_state)

    # Create Dataset objects
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    return train_loader, test_loader
