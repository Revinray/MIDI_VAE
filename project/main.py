# main.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
from code.data_loader import MidiDataset
from code.model import VAE_Multistream
from code.train import train_epoch, test_epoch, save_checkpoint, load_checkpoint
from code.utils import plot_losses, create_dir
from code.generate import reconstruct_and_save_midi, sample_latent_space

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VAE for MIDI Encoding and Decoding with Style Embedding")
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Directory containing raw MIDI files')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Directory to store processed data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output MIDI files')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=12, help='Number of crotchet notes per sequence')
    parser.add_argument('--fs', type=int, default=100, help='Sampling frequency for piano rolls')
    parser.add_argument('--latent_dim', type=int, default=20, help='Dimension of the latent space')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128], help='List of hidden layer dimensions for GRUs')
    parser.add_argument('--style_embed_dim', type=int, default=16, help='Dimension of the style (composer) embedding')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to resume training')
    parser.add_argument('--generate', action='store_true', help='Flag to generate MIDI files from the model')
    parser.add_argument('--reconstruct', action='store_true', help='Flag to reconstruct MIDI files from the test set')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of MIDI files to generate or reconstruct')
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Create necessary directories
    create_dir(args.processed_dir)
    create_dir(args.output_dir)
    create_dir(args.models_dir)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading and Preprocessing
    print("Loading and preprocessing data...")
    midi_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_files.append(os.path.join(root, file))
    print(f"Total MIDI files found: {len(midi_files)}")

    # Initialize Dataset
    dataset = MidiDataset(
        midi_files=midi_files,
        sequence_length=args.sequence_length,
        fs=args.fs
    )

    if len(dataset) == 0:
        print("No valid data found. Exiting.")
        return

    # Split into training and testing sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model Initialization
    input_dims = {
        'pitch': 128,         # Number of MIDI pitches
        'instrument': len(dataset.instrument_map),     # Number of instruments dynamically determined
        'velocity': 128       # Number of MIDI pitches for velocity
    }
    num_composers = len(dataset.composer_map)

    model = VAE_Multistream(
        input_dims=input_dims,
        hidden_dims={'pitch': args.hidden_dims[0], 'instrument': args.hidden_dims[1], 'velocity': args.hidden_dims[1], 'combined': 128},
        latent_dim=args.latent_dim,
        num_composers=num_composers,
        style_embed_dim=args.style_embed_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 1
    if args.checkpoint:
        # Load checkpoint if provided
        start_epoch, _ = load_checkpoint(model, optimizer, args.checkpoint, device)
        start_epoch += 1  # Start from the next epoch

    # Lists to store loss values
    train_losses = []
    test_losses = []

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        # Testing
        test_loss = test_epoch(model, test_loader, device)
        test_losses.append(test_loss)
        print(f"Test Loss: {test_loss:.4f}")

        # Save checkpoint every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.models_dir, f'vae_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_path)

        # Plot losses every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_path = os.path.join(args.output_dir, f'loss_epoch_{epoch}.png')
            plot_losses(train_losses, test_losses, save_path=plot_path)

    # Generation or Reconstruction
    if args.generate or args.reconstruct:
        print("\nLoading the latest model for generation/reconstruction...")
        latest_checkpoint = os.path.join(args.models_dir, f'vae_epoch_{epoch}.pth')
        load_checkpoint(model, optimizer, latest_checkpoint, device)

        if args.reconstruct:
            print("Reconstructing MIDI files from the test set...")
            reconstruct_and_save_midi(
                model=model,
                dataloader=test_loader,
                device=device,
                output_dir=args.output_dir,
                num_samples=args.num_samples
            )

        if args.generate:
            print("Generating new MIDI files by sampling the latent space...")
            # Example: Assigning a random composer for generation
            for _ in range(args.num_samples):
                composer_id = torch.randint(0, num_composers, (1,)).item()
                sample_latent_space(
                    model=model,
                    device=device,
                    output_dir=args.output_dir,
                    num_samples=1,
                    latent_dim=args.latent_dim,
                    composer_id=composer_id
                )

if __name__ == "__main__":
    main()
