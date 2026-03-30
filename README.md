# Automatic Music Generation with PyTorch & LSTMs 🎵

An end-to-end deep learning project that automatically composes and generates Polyphonic Piano MIDI music. 

By training a **Multi-Layer LSTM (Long Short-Term Memory)** neural network on classical Piano MIDI files (e.g., Schubert), this repository parses musical notes and chords, learns their sequential structure, and generates entirely new MIDI compositions from scratch.

---

## 🏗️ Architecture & How It Works (The Pipeline)

This project treats music generation as a **Sequence-to-Sequence (Time-Series) Prediction** problem. The pipeline is split into distinct technical phases:

### 1. Data Extraction (`music21`)
MIDI files are parsed using the `music21` library. 
- The script isolates specifically the **Piano** instruments.
- Musical events are extracted sequentially. Single notes are extracted via their pitch (e.g., `C4`), while **Chords** (multiple notes played simultaneously) are parsed into strings of their normal order integers (e.g., `4.7.11`).

### 2. Vocabulary & Thresholding
To prevent the model from struggling to learn very rare chords:
- The parser counts the frequency of all unique notes/chords.
- It applies a **Threshold Filter** (e.g., keeping only notes that appear 50+ times in the dataset). 
- The filtered sequence is then mapped to integer indices creating two dictionaries: `note2ind` and `ind2note`.

### 3. Sequence Modeling
The data is structured into sequences of `timesteps = 50`. The model receives 50 consecutive notes/chords as input $X$ and is tasked with predicting the 51st note as target $y$.

### 4. Deep Learning Model (`MusicLSTM`)
The neural network is built with PyTorch and consists of:
- **LSTM Layer 1**: 256 hidden units.
- **Dropout (0.3)**: To prevent overfitting and encourage generalized learning.
- **LSTM Layer 2**: 256 hidden units.
- **Dropout (0.3)**: Additional regularization.
- **Dense / Linear Layers**: Compresses the 256 dimensions down to the number of unique classes (vocabulary size) using a ReLU activation in between.

### 5. Training & Regularization
The model is trained using **CrossEntropyLoss** and the **Adam Optimizer**. To prevent severe overfitting:
- **Weight Decay (L2 Penalty)** is applied to the optimizer.
- **Model Checkpointing**: The script tracks Validation Loss. It *only* saves the model weights (`s2s/model.pth`) when a new lowest validation loss is achieved.
- **Early Stopping**: If the validation loss fails to improve for 50 consecutive epochs, training dynamically halts.

### 6. Generation & Decoding
During inference, a random 50-note sequence from the test set is fed to the trained model.
- The model outputs probability scores for the next note.
- `argmax` is used to pick the most likely prediction.
- The newly predicted note is appended to the pattern, the oldest note is dropped, and the window slides forward to predict the next note (Autoregressive generation).
- Finally, the integers are mapped back to `music21` Note/Chord objects and saved directly to a `.mid` file.

---

## 📂 Repository Structure

```text
music-gen/
├── automatic-music-generation/
│   ├── auto_music_gen.py    # Main script: Extracts data, trains model, and runs generation
│   ├── inference.py         # Standalone generation script using a pre-trained model
│   ├── generate.py          # Quick wrapper script to execute inference
│   ├── All Midi Files/      # Dataset directory containing subdirectories of MIDI files (e.g., /schubert)
│   ├── s2s/                 # Automatically generated directory for artifacts
│   │   ├── model.pth        # Trained PyTorch model weights (Best Checkpoint)
│   │   ├── mappings.pkl     # Pickled note2ind / ind2note dictionaries
│   │   └── x_test.pkl       # Pickled test sequence for random generation seeding
│   ├── requirements.txt     # Python dependencies
│   └── pred_music.mid       # Resulting generated music output (created after generation)
```

---

## 🛠️ Installation & Setup

1. **Clone the repository and set up environment:**
   Ensure you have Python 3.11+ installed.
   ```bash
   # (Optional) Create a virtual environment
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch torchvision torchaudio numpy tqdm scikit-learn music21
   ```

3. **Prepare Dataset:**
   Place your `.mid` training files inside the directory:
   `automatic-music-generation/All Midi Files/schubert/` (or modify `file_path` in `auto_music_gen.py`).

---

## 🚀 Usage

### 1. Training the Model (End-to-End)
To parse the MIDI dataset, train the network from scratch, and instantly generate a sample track:
```bash
cd automatic-music-generation
python auto_music_gen.py
```
*Depending on the dataset size and your system's GPU (`cuda` supported), this may take a few minutes. Checkpoints will be aggressively saved to `s2s/model.pth`.*

### 2. Generating Music (Inference Only)
If you have already trained the model and simply want to generate new tracks infinitely without retraining, use the generation scripts:
```bash
cd automatic-music-generation

# Option 1: Quick helper script
python generate.py

# Option 2: Direct inference script execution
python inference.py
```

The resulting AI-composed track will be saved directly into the folder as `pred_music.mid`. You can open this file in any DAW (Ableton, FL Studio, Logic), standard Media Player, or an online MIDI visualizer.
