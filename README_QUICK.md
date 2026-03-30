# 🎵 Music Generation - Quick Reference

## Two-Step Process

### 👉 Step 1: Train (Optional - Only Run Once)
```bash
python auto_music_gen.py
```
⏱️ Takes ~5-30 minutes depending on GPU
📊 Trains LSTM model on 80 epochs
💾 Saves model to `s2s/model.pth`

### 👉 Step 2: Generate Music
```bash
python inference.py
```
⚡ Takes ~15-30 seconds
🎵 Generates 200 piano notes
📁 Saves to `pred_music.mid`

---

## File Descriptions

| File | Purpose | Run When |
|------|---------|----------|
| `auto_music_gen.py` | Train the LSTM model | First time OR when you want to retrain |
| `inference.py` | Generate new music | Anytime after training to create fresh pieces |
| `generate.py` | Helper script | Alternative way to run inference |
| `setup.py` | Check dependencies | Verify everything is installed |

---

## Playback

Open `pred_music.mid` with:
- **Windows**: Windows Media Player, MuseScore, Finale
- **Mac**: GarageBand, MuseScore
- **Linux**: MuseScore, Timidity, Hydrogen
- **Web**: https://www.midijs.net/ (online player)

---

## Customization Quick Tips

### Change training data:
```python
# In auto_music_gen.py, line ~80
file_path = ["bach"]  # instead of "schubert"
```

### Generate more notes:
```python
# In inference.py, line ~190
num_notes_to_generate=500,  # instead of 200
```

### Use GPU (if available):
- Automatic! Check: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: model.pth` | Run `python auto_music_gen.py` first |
| `No module named torch` | Run `pip install -r requirements.txt` |
| Inference is slow | Check if GPU is available with `torch.cuda.is_available()` |
| MIDI sounds bad | Try a different MIDI player (different default instruments) |

---

## File Sizes (Approximate)

| File | Size | Notes |
|------|------|-------|
| `model.pth` | 10-20 MB | Neural network weights |
| `mappings.pkl` | 1-5 KB | Note mappings (very small) |
| `x_test.pkl` | 5-50 MB | Test data for inference |
| `pred_music.mid` | 100-500 KB | Output MIDI file |

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Train model: `python auto_music_gen.py` (~15 min)
3. ✅ Generate music: `python inference.py` (~15 sec)
4. ✅ Listen to `pred_music.mid`
5. 🔄 Repeat step 3 to generate new variations
6. 📝 Fork & modify hyperparameters for different results

---

**Keys to Different Outputs:**
- Different starting patterns → Different melodies
- Different training data (composers) → Different styles
- Different LSTM sizes → Different complexity
- Different epochs → Different quality/convergence

Experiment and have fun! 🎼
