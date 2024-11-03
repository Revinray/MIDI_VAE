# code/data_loader.py

import pretty_midi
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

class MidiDataset(Dataset):
    """
    Custom Dataset for loading MIDI files and converting them into pitch, instrument, and velocity roll sequences.
    """

    def __init__(self, midi_files, sequence_length=12, fs=100):
        """
        Initializes the dataset by loading and processing MIDI files.

        Args:
            midi_files (list): List of paths to MIDI files.
            sequence_length (int): Number of crotchet notes per sequence.
            fs (int): Sampling frequency (frames per second) for piano roll.
        """
        self.midi_files = midi_files
        self.sequence_length = sequence_length
        self.fs = fs
        self.data = []
        self.composer_map = self.create_composer_map()
        self.instrument_map = self.create_instrument_map()
        self.load_data()

    def create_composer_map(self):
        """
        Creates a mapping from composer names to unique IDs.

        Returns:
            dict: Composer name to ID mapping.
        """
        composers = set()
        for midi_file in self.midi_files:
            composer = self.get_composer_from_path(midi_file)
            composers.add(composer)
        return {composer: idx for idx, composer in enumerate(sorted(composers))}

    def create_instrument_map(self):
        """
        Creates a mapping from instrument programs to unique IDs.

        Returns:
            dict: Instrument program number to ID mapping.
        """
        instruments = set()
        for midi_file in self.midi_files:
            try:
                pm = pretty_midi.PrettyMIDI(midi_file)
                for instrument in pm.instruments:
                    instruments.add(instrument.program)
            except:
                continue
        return {instr: idx for idx, instr in enumerate(sorted(instruments))}

    def load_data(self):
        """
        Loads MIDI files, extracts rolls, splits into sequences, and stores them.
        """
        for midi_file in self.midi_files:
            try:
                composer = self.get_composer_from_path(midi_file)
                composer_id = self.composer_map[composer]

                pm = pretty_midi.PrettyMIDI(midi_file)

                pitch_roll, instrument_roll, velocity_roll = self.extract_rolls(pm, fs=self.fs)

                # Calculate number of sequences
                sequence_length_seconds = self.sequence_length * 0.5  # Assuming crotchet = 0.5s
                num_timesteps = int(sequence_length_seconds * self.fs)
                num_sequences = pitch_roll.shape[1] // num_timesteps

                for i in range(num_sequences):
                    start = i * num_timesteps
                    end = start + num_timesteps

                    pitch_seq = pitch_roll[:, start:end]
                    instr_seq = instrument_roll[:, start:end]
                    vel_seq = velocity_roll[:, start:end]

                    if pitch_seq.shape[1] == num_timesteps:
                        self.data.append({
                            'pitch': pitch_seq,
                            'instrument': instr_seq,
                            'velocity': vel_seq,
                            'composer': composer_id
                        })
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")

    def extract_rolls(self, pm, fs):
        """
        Extracts pitch, instrument, and velocity rolls from a PrettyMIDI object.

        Args:
            pm (pretty_midi.PrettyMIDI): Parsed MIDI object.
            fs (int): Sampling frequency.

        Returns:
            tuple: (pitch_roll, instrument_roll, velocity_roll)
        """
        num_pitches = 128
        num_instruments = len(self.instrument_map)

        pitch_roll = np.zeros((num_pitches, pm.get_end_time() * fs))
        instrument_roll = np.zeros((num_instruments, pm.get_end_time() * fs))
        velocity_roll = np.zeros((num_pitches, pm.get_end_time() * fs))

        for instrument in pm.instruments:
            instr_id = self.instrument_map.get(instrument.program, 0)
            for note in instrument.notes:
                start_idx = int(note.start * fs)
                end_idx = int(note.end * fs)
                pitch = note.pitch
                velocity = note.velocity / 127.0  # Normalize

                pitch_roll[pitch, start_idx:end_idx] = 1
                instrument_roll[instr_id, start_idx:end_idx] = 1
                velocity_roll[pitch, start_idx:end_idx] = velocity

        return pitch_roll, instrument_roll, velocity_roll

    def get_composer_from_path(self, midi_file):
        """
        Extracts composer name from the MIDI file path.

        Args:
            midi_file (str): Path to the MIDI file.

        Returns:
            str: Composer name.
        """
        return os.path.basename(os.path.dirname(midi_file))

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
            dict: Dictionary containing pitch, instrument, velocity tensors and composer ID.
        """
        sample = self.data[index]
        return {
            'pitch': torch.tensor(sample['pitch'], dtype=torch.float32),
            'instrument': torch.tensor(sample['instrument'], dtype=torch.float32),
            'velocity': torch.tensor(sample['velocity'], dtype=torch.float32),
            'composer': torch.tensor(sample['composer'], dtype=torch.long)
        }
