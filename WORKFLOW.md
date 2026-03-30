# 🎵 Music Generation - Complete Workflow

## Overview
This project trains an LSTM neural network on classical piano pieces and generates new original music.

## Workflow

### Step 1: Train the Model
**File:** `auto_music_gen.py`

```bash
python auto_music_gen.py
```

**What it does:**
- Loads MIDI files from `All Midi Files/schubert/` (configurable)
- Extracts piano notes and creates sequences
- Trains an LSTM model for 80 epochs
- Saves model weights to `s2s/model.pth`
- Saves note mappings to `s2s/mappings.pkl`
- Saves test data to `s2s/x_test.pkl`

**Output files created:**
```
s2s/
├── model.pth          # Neural network weights
├── mappings.pkl       # Note names <-> indices mapping
└── x_test.pkl         # Test patterns for inference
```

### Step 2: Generate Music (Inference)
**File:** `inference.py`

```bash
python inference.py
```

**What it does:**
- Loads the trained model and note mappings
- Generates 200 new notes using the model
- Converts predictions to MIDI format
- Saves output as `pred_music.mid`

## Features

### Training Script (`auto_music_gen.py`)
✅ PyTorch-based LSTM with 2 stacked layers
✅ Dropout (0.2) for regularization
✅ Adam optimizer with learning rate 0.001
✅ Cross-entropy loss function
✅ 80 training epochs with validation monitoring
✅ GPU support (auto-detects CUDA)
✅ Saves model and data for inference
✅ Full type hints and modern Python 3.11+ code

**Hyperparameters** (customize as needed):
```python
batch_size = 128      # Training batch size
epochs = 80           # Number of training epochs
hidden_size = 256     # LSTM hidden dimension
timesteps = 50        # Sequence length
threshold = 50        # Min note frequency
```

### Inference Script (`inference.py`)
✅ Loads pre-trained model
✅ Generates 200-note sequences
✅ Converts to playable MIDI
✅ Handles both single notes and chords
✅ Fallback to random patterns if test data missing
✅ No_grad() mode for memory efficiency
✅ Detailed progress tracking with tqdm

**Features:**
- Can use random starting pattern or test data
- Customizable number of notes to generate
- Error handling for invalid notes
- Clear console output with emojis

## Directory Structure

```
automatic-music-generation-codes/
├── auto_music_gen.py              # Training script
├── inference.py                   # Inference/generation script
├── setup.py                       # Setup helper
├── requirements.txt               # Dependencies (Python 3.11+)
├── MODERNIZATION.md              # Migration guide
├── BEFORE_AFTER.md               # Code comparison
├── WORKFLOW.md                   # This file
├── All Midi Files/               # Training data
│   ├── albeniz/
│   ├── bach/
│   ├── beeth/
│   └── ... (other composers)
└── s2s/                          # Saved models & data
    ├── model.pth                 # Trained weights (PyTorch)
    ├── mappings.pkl              # Note mappings
    └── x_test.pkl                # Test data
```

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# First time - full training
python auto_music_gen.py
```

Training output:
```
Using device: cuda (or cpu)
Unique Notes: 127
Frequency notes
30: 450
50: 280
70: 180

Training for 80 epochs...
Epoch [10/80], Train Loss: 4.2341, Train Acc: 15.23%, Val Loss: 4.1234, Val Acc: 16.45%
Epoch [20/80], Train Loss: 3.8234, Train Acc: 22.31%, Val Loss: 3.7123, Val Acc: 23.56%
...
Model saved to s2s/model.pth
Mappings saved to s2s/mappings.pkl
Test data saved to s2s/x_test.pkl
```

### 3. Generate Music
```bash
python inference.py
```

Inference output:
```
============================================================
🎵 Music Generation - Inference Phase
============================================================

📍 Using device: cuda
✅ Loaded 127 unique notes
✅ Model loaded from s2s/model.pth
✅ Loaded test data (using random pattern #42)

🎵 Generating 200 notes...
Generating: 100%|██████████| 200/200 [00:15<00:00, 13.33it/s]

🎹 Converting notes to MIDI...
✅ MIDI file saved to pred_music.mid
   Total notes/chords: 200

============================================================
✨ Music generation complete!
📁 Output: pred_music.mid
============================================================
```

### 4. Listen to Generated Music
Open `pred_music.mid` with any MIDI player:
- **Windows**: Windows Media Player, MuseScore, VLC
- **Mac**: GarageBand, MuseScore
- **Linux**: MuseScore, Hydrogen, Timidity

## Customization

### Change Training Data
Edit `auto_music_gen.py`:
```python
file_path = ["schubert"]  # Change this to another composer

# Available:
# - albeniz, bach, balakir, beeth, borodin, brahms, burgm, chopin, 
# - debussy, granados, grieg, haydn, liszt, mendelssohn, mozart, 
# - muss, schubert, schumann, tschai
```

### Adjust Generated Length
Edit `inference.py`:
```python
generated_notes = generate_music(
    model,
    ind2note,
    music_pattern,
    num_notes_to_generate=500,  # Change from 200
    device=device,
)
```

### Use Custom Seed Pattern
In `inference.py`, use the `generate_with_seed()` function:
```python
# Generate with specific starting notes
seed = ["C4", "E4", "G4"]  # C major chord
notes = generate_with_seed(
    model, note2ind, ind2note, 
    seed_pattern=seed, 
    num_notes=300
)
```

## Troubleshooting

**Error: `FileNotFoundError: Model file not found`**
- Solution: Run `python auto_music_gen.py` first to train the model

**Error: `FileNotFoundError: All Midi Files not found`**
- Solution: Make sure MIDI files are in the correct directory structure

**Inference is slow**
- Solution: Uses CPU if CUDA not available. Install GPU drivers if you want faster inference
- Or reduce `num_notes_to_generate` for faster testing

**MIDI file sounds wrong**
- Solution: Different MIDI players have different soundfonts. Try a different player
- Or try adjusting note offset values in the code

## Performance Tips

1. **GPU Training**: Uses CUDA automatically if available (10-100x faster)
2. **Reduce Training Data**: Use only 1-2 composers for faster prototyping
3. **Adjust Hyperparameters**:
   - `batch_size`: Increase for faster training (if GPU memory allows)
   - `epochs`: Reduce to 20-30 for quick testing
   - `threshold`: Increase to 100+ for fewer, more common notes

## Advanced: Training with Different Data

To train on multiple composers:
```python
# In auto_music_gen.py
file_path = ["mozart", "bach", "chopin"]
all_files = []
for composer in file_path:
    all_files.extend(glob.glob(f'All Midi Files/{composer}/*.mid', recursive=True))
```

## Model Architecture

```
Input: (batch_size, 50, 1)
  ↓
LSTM1(256) + Dropout(0.2)
  ↓
LSTM2(256) + Dropout(0.2)
  ↓
Dense(256, ReLU)
  ↓
Dense(num_notes, Softmax)
  ↓
Output: (batch_size, num_notes)
```

## References

- **PyTorch**: https://pytorch.org
- **music21**: http://web.mit.edu/music21/
- **LSTM Overview**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

**Version**: 2.0 (PyTorch modernized)
**Last Updated**: March 2026
**Python**: 3.11+
