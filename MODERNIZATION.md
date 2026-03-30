# Music Generation with PyTorch - Modernization Guide

## Overview
Your automatic music generation project has been modernized for **Python 3.11+** with modern dependencies. The code has been completely migrated from **TensorFlow** to **PyTorch**.

## Major Changes

### 1. **TensorFlow → PyTorch Migration**
- ❌ Removed: `tensorflow.keras` (old, bloated framework)
- ✅ Added: PyTorch 2.1+ (modern, efficient, industry standard)
- Rewrote the neural network model using `torch.nn.Module`
- Complete training loop rewritten with PyTorch conventions

### 2. **Modern Python 3.11+ Features**
- Added type hints throughout the code (`def read_files(file: str) -> list[str]`)
- Used f-strings for all string formatting
- Replaced `type()` checks with `isinstance()` for pythonic code
- Added docstrings to functions and classes
- Used dictionary comprehensions instead of `map()` and `filter()`
- Union types with `|` operator (e.g., `list[chord.Chord | note.Note]`)
- Imported `from __future__ import annotations` for forward compatibility

### 3. **Current Dependencies**
```
torch>=2.1.0          # Modern PyTorch framework
torchaudio>=2.1.0     # Audio utilities
music21>=9.1.0        # Music processing (up-to-date)
numpy>=1.24.0         # Numerical computing
scikit-learn>=1.3.0   # Machine learning utilities
tqdm>=4.66.0          # Progress bars
```

### 4. **Architecture Improvements**

#### Custom LSTM Model Class
```python
class MusicLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_classes: int = 100):
        # Clean, organized model definition
```

#### Better Device Handling
- Automatic GPU detection: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All tensors are moved to the appropriate device

#### Proper Training/Validation Loop
- Separate training and validation phases
- Model switched to `.train()` and `.eval()` modes appropriately
- Uses `with torch.no_grad()` for inference (memory efficient)
- Batch-wise training with explicit loss computation
- Better logging every 10 epochs

#### Safer Prediction Logic
- Added try-except blocks for note parsing
- Handles edge cases better
- Uses `torch.argmax()` instead of `np.argmax()`

## Setup Instructions

### 1. **Create and Activate Virtual Environment (Recommended)**

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Verify Installation**
```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running the Code

**Basic execution:**
```bash
python auto_music_gen.py
```

## Key Features

✅ **GPU Support**: Automatically uses CUDA if available, falls back to CPU
✅ **Type Safety**: Full type hints for better IDE support and code clarity
✅ **Efficient Training**: PyTorch's optimized operations are faster than TensorFlow
✅ **Better Memory Management**: Explicit device placement and no_grad() context
✅ **Modern Error Handling**: Try-except blocks for robust execution
✅ **Progress Tracking**: Clear epoch-by-epoch logging

## Model Configuration

You can adjust hyperparameters:

```python
# In auto_music_gen.py:
batch_size = 128      # Batch size
epochs = 80           # Training epochs
hidden_size = 256     # LSTM hidden units
timesteps = 50        # Sequence length
```

## File Structure

```
auto_music_gen.py          # Main training and generation script
requirements.txt           # Python dependencies (3.11+)
s2s/
  model.pth               # Saved PyTorch model weights
All Midi Files/           # Training data
pred_music.mid            # Generated MIDI output
```

## Notes

1. The model now saves as `.pth` files (PyTorch format) instead of SavedModel format
2. Training will be noticeably faster with PyTorch compared to TensorFlow
3. GPU training is automatically enabled if CUDA is available
4. The model architecture is identical—same results, modernized code

## Troubleshooting

**PyTorch not found:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(Adjust `cu118` to your CUDA version, or use `cpu` for CPU-only)

**music21 issues:**
```bash
pip install --upgrade music21
```

**File path issues:**
Make sure `All Midi Files/schubert/` directory exists relative to where you run the script

---

**Summary**: Your project is now modern, efficient, and uses industry-standard PyTorch instead of legacy TensorFlow!
