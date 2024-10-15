# MIDI VAE: Variational Autoencoder for Music Encoding and Decoding

![MIDI VAE Logo](https://example.com/logo.png) <!-- Replace with your project's logo if available -->

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Resuming Training from a Checkpoint](#resuming-training-from-a-checkpoint)
   - [Generating New MIDI Files](#generating-new-midi-files)
   - [Reconstructing MIDI Files from the Test Set](#reconstructing-midi-files-from-the-test-set)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [Evaluation](#evaluation)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

---

## Project Overview

**MIDI VAE** is a Variational Autoencoder (VAE) designed to encode and decode MIDI music files. The primary goal is to compress MIDI data into a lower-dimensional latent space (`z`) and accurately reconstruct the original music, enabling tasks such as music generation, style transfer, and analysis.

By progressively adding encoding and decoding layers, MIDI VAE aims to identify the minimal latent dimensionality that maintains high-quality reconstruction while minimizing computational complexity.

---

## Features

- **Modular VAE Architecture:** Easily add or remove encoding and decoding layers.
- **Dimensionality Reduction:** Explore the optimal size of the latent space for music encoding.
- **MIDI Generation:** Generate new MIDI files by sampling from the latent space.
- **MIDI Reconstruction:** Reconstruct existing MIDI files to assess encoding quality.
- **Checkpointing:** Save and resume training from specific epochs.
- **Visualization:** Plot training and test loss curves for performance monitoring.

---

## Directory Structure

```
/project
├── /data
│   ├── /raw
│   ├── /processed
│   ├── /train
│   └── /test
├── /models
├── /outputs
├── /code
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

- **`/data/raw`**: Contains all original MIDI files, organized by composer.
- **`/data/processed`**: Stores processed data such as piano roll representations.
- **`/data/train` & `/data/test`**: Hold the training and testing subsets of the processed data.
- **`/models`**: Stores saved model checkpoints after training.
- **`/outputs`**: Contains reconstructed and generated MIDI files.
- **`/code`**: Includes all Python scripts necessary for data loading, model definition, training, and generation.
- **`main.py`**: The main script to execute training and generation processes.
- **`requirements.txt`**: Lists all Python dependencies required for the project.
- **`README.md`**: Provides an overview and instructions for the project.

---

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **pip** package manager
- **CUDA** (optional, for GPU acceleration)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/midi-vae.git
   cd midi-vae
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**

   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

   Ensure that PyTorch is installed correctly and can detect your GPU (if available):

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

   Should return `True` if CUDA is available and PyTorch is correctly configured to use it.

---

## Data Preparation

### 1. Organize MIDI Files

Place all your MIDI files within the `/data/raw` directory, maintaining subdirectories by composer for better organization. For example:

```
/data/raw
├── /Bach
│   ├── 2186_vs6_1.mid
│   ├── 2191_vs6_5.mid
│   └── ...
├── /Beethoven
│   ├── 2313_qt15_1.mid
│   ├── 2314_qt15_2.mid
│   └── ...
└── ...
```

### 2. Preprocess MIDI Files

The preprocessing step converts MIDI files into piano roll representations, splits them into fixed-length sequences, and prepares them for training and testing.

This is handled automatically when you run the `main.py` script. Ensure that all MIDI files are correctly placed in the `/data/raw` directory before proceeding.

---

## Usage

The primary interface for interacting with MIDI VAE is the `main.py` script. It supports various functionalities, including training, generating new MIDI files, and reconstructing existing ones.

### Running the Main Script

```bash
python main.py [OPTIONS]
```

### Available Command-Line Arguments

| Argument             | Description                                                      | Default        |
|----------------------|------------------------------------------------------------------|----------------|
| `--data_dir`         | Directory containing raw MIDI files                              | `data/raw`     |
| `--processed_dir`    | Directory to store processed data                               | `data/processed`|
| `--output_dir`       | Directory to save output MIDI files                             | `outputs`      |
| `--models_dir`       | Directory to save model checkpoints                             | `models`       |
| `--epochs`           | Number of training epochs                                       | `50`           |
| `--batch_size`       | Batch size for training                                         | `64`           |
| `--sequence_length`  | Number of time steps per sequence                               | `500`          |
| `--fs`               | Sampling frequency for piano rolls (frames per second)          | `100`          |
| `--latent_dim`       | Dimension of the latent space                                   | `20`           |
| `--layer_dims`       | List of encoder layer dimensions (e.g., `512 256`)             | `[512]`        |
| `--learning_rate`    | Learning rate for the optimizer                                 | `0.001`        |
| `--checkpoint`       | Path to a checkpoint to resume training                        | `None`         |
| `--generate`         | Flag to generate new MIDI files from the model                  | `False`        |
| `--reconstruct`      | Flag to reconstruct MIDI files from the test set                | `False`        |
| `--num_samples`      | Number of MIDI files to generate or reconstruct                 | `10`           |

### Functionalities

1. **Training the Model**
2. **Resuming Training from a Checkpoint**
3. **Generating New MIDI Files**
4. **Reconstructing MIDI Files from the Test Set**

#### 1. Training the Model

To train the VAE model with default parameters:

```bash
python main.py
```

**Customizing Training Parameters:**

```bash
python main.py --epochs 100 --batch_size 128 --sequence_length 600 --latent_dim 30 --layer_dims 512 256 --learning_rate 0.0005
```

**Explanation:**

- **`--epochs 100`**: Train for 100 epochs.
- **`--batch_size 128`**: Use a batch size of 128.
- **`--sequence_length 600`**: Each sequence contains 600 time steps.
- **`--latent_dim 30`**: The latent space has 30 dimensions.
- **`--layer_dims 512 256`**: The encoder has two hidden layers with 512 and 256 neurons, respectively.
- **`--learning_rate 0.0005`**: Set the learning rate to 0.0005.

#### 2. Resuming Training from a Checkpoint

If training was previously interrupted or you wish to continue training from a saved state, use the `--checkpoint` argument:

```bash
python main.py --checkpoint models/vae_epoch_50.pth --epochs 100
```

**Explanation:**

- **`--checkpoint models/vae_epoch_50.pth`**: Load the model and optimizer states from epoch 50.
- **`--epochs 100`**: Continue training until epoch 100.

#### 3. Generating New MIDI Files

After training, you can generate new MIDI compositions by sampling from the latent space:

```bash
python main.py --generate --num_samples 20
```

**Explanation:**

- **`--generate`**: Activates the MIDI generation mode.
- **`--num_samples 20`**: Generate 20 new MIDI files.

**Customizing Latent Space Dimensions:**

Ensure that `--latent_dim` matches the dimension used during training.

```bash
python main.py --generate --latent_dim 30 --num_samples 15
```

#### 4. Reconstructing MIDI Files from the Test Set

To reconstruct existing MIDI files and assess the model's encoding quality:

```bash
python main.py --reconstruct --num_samples 10
```

**Explanation:**

- **`--reconstruct`**: Activates the MIDI reconstruction mode.
- **`--num_samples 10`**: Reconstruct 10 MIDI files from the test set.

**Combining Generation and Reconstruction:**

You can perform both generation and reconstruction in a single run:

```bash
python main.py --generate --reconstruct --num_samples 10
```

---

## Configuration

### Command-Line Arguments Detailed Description

- **`--data_dir`**: Specify the path to your raw MIDI files. Ensure that the directory exists and contains subdirectories organized by composer.

- **`--processed_dir`**: Path where processed piano roll data will be stored. The script will automatically create this directory if it doesn't exist.

- **`--output_dir`**: Directory to save generated and reconstructed MIDI files.

- **`--models_dir`**: Directory to store model checkpoints after each epoch.

- **`--epochs`**: Total number of epochs to train the model.

- **`--batch_size`**: Number of samples per batch during training.

- **`--sequence_length`**: Number of time steps per sequence in the piano roll.

- **`--fs`**: Sampling frequency for the piano roll conversion (frames per second).

- **`--latent_dim`**: Dimensionality of the latent space `z`. A higher value allows for more complex representations but increases computational load.

- **`--layer_dims`**: Space-separated list of integers specifying the number of neurons in each hidden layer of the encoder. The decoder mirrors this architecture in reverse.

- **`--learning_rate`**: Learning rate for the Adam optimizer. Lower values may lead to more stable training but require more epochs.

- **`--checkpoint`**: Path to a saved checkpoint file to resume training.

- **`--generate`**: If set, the script will generate new MIDI files after training.

- **`--reconstruct`**: If set, the script will reconstruct MIDI files from the test set after training.

- **`--num_samples`**: Number of MIDI files to generate or reconstruct.

### Example Configuration Command

```bash
python main.py \
    --data_dir data/raw \
    --processed_dir data/processed \
    --output_dir outputs \
    --models_dir models \
    --epochs 100 \
    --batch_size 64 \
    --sequence_length 500 \
    --fs 100 \
    --latent_dim 20 \
    --layer_dims 512 256 \
    --learning_rate 0.001 \
    --generate \
    --reconstruct \
    --num_samples 10
```

---

## Examples

### 1. **Training the Model with Default Parameters**

```bash
python main.py
```

This command will:

- Load MIDI files from `data/raw`.
- Preprocess them into piano rolls with a sequence length of 500 and sampling frequency of 100 Hz.
- Train the VAE for 50 epochs with a batch size of 64.
- Use a latent space dimension of 20 and a single encoder hidden layer with 512 neurons.
- Save model checkpoints in the `models` directory.
- Save loss plots in the `outputs` directory every 10 epochs.

### 2. **Training with Custom Parameters**

```bash
python main.py --epochs 100 --batch_size 128 --latent_dim 30 --layer_dims 1024 512 256 --learning_rate 0.0005
```

This command configures the training to:

- Run for 100 epochs.
- Use a larger batch size of 128.
- Increase the latent space to 30 dimensions.
- Add two additional encoder layers with 512 and 256 neurons, respectively.
- Use a lower learning rate of 0.0005 for finer updates.

### 3. **Resuming Training from a Checkpoint**

```bash
python main.py --checkpoint models/vae_epoch_50.pth --epochs 100
```

This command will:

- Load the model and optimizer states from `vae_epoch_50.pth`.
- Resume training from epoch 51 up to epoch 100.

### 4. **Generating New MIDI Files**

```bash
python main.py --generate --num_samples 20
```

After training, this command will:

- Generate 20 new MIDI files by sampling from the latent space.
- Save the generated MIDI files in the `outputs` directory with filenames like `sampled_0.mid`, `sampled_1.mid`, etc.

### 5. **Reconstructing MIDI Files from the Test Set**

```bash
python main.py --reconstruct --num_samples 10
```

This command will:

- Reconstruct 10 MIDI files from the test dataset.
- Save the reconstructed MIDI files in the `outputs` directory with filenames like `reconstructed_0_0.mid`, `reconstructed_0_1.mid`, etc.

### 6. **Combining Generation and Reconstruction**

```bash
python main.py --generate --reconstruct --num_samples 10
```

This command will perform both generation and reconstruction, creating a total of 10 MIDI files each.

---

## Evaluation

### Monitoring Training Progress

During training, the script outputs the training and test loss for each epoch. Additionally, it saves loss plots (`loss_epoch_X.png`) in the `outputs` directory every 10 epochs. These plots help in visualizing the model's convergence and detecting potential overfitting.

### Assessing Reconstruction Quality

After training, reconstruct MIDI files from the test set using the `--reconstruct` flag. Listen to these files to subjectively evaluate the quality of reconstruction. High-quality reconstructions should closely resemble the original MIDI compositions.

### Generating New Music

Use the `--generate` flag to create new MIDI compositions. These generated files represent the model's learned understanding of the musical patterns in the training data. Listening to these can provide insights into the creativity and coherence of the model.

### Quantitative Metrics

While the primary evaluation metric is the VAE's loss (combining reconstruction loss and KL divergence), additional metrics such as BLEU scores or other similarity measures can be implemented to quantitatively assess reconstruction quality.

---

## Contributing

Contributions are welcome! Whether it's improving the model architecture, adding new features, or enhancing the documentation, your input can help make MIDI VAE better.

### Steps to Contribute

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/midi-vae.git
   cd midi-vae
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Implement your feature or fix bugs.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

   Navigate to the original repository and create a pull request from your fork's branch.

### Code of Conduct

Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) when contributing to this project.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- **[PrettyMIDI](https://github.com/craffel/pretty-midi):** For MIDI file parsing and piano roll conversion.
- **[PyTorch](https://pytorch.org/):** For providing the deep learning framework.
- **[tqdm](https://github.com/tqdm/tqdm):** For progress bars during training and generation.
- **OpenAI:** For foundational research in VAEs and generative models.
- **Community Contributors:** Special thanks to all the contributors who have helped improve this project.

---

## Contact

For any questions or feedback, please reach out to us via the issues page

---

*Happy Music Encoding and Decoding! 🎶*