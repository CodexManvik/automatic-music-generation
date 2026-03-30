"""
Automatic Music Generation using LSTM with PyTorch.
Modernized for Python 3.11+ with current dependencies.
"""

from __future__ import annotations

import glob
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from music21 import chord, converter, instrument, note, stream
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Create s2s directory if it doesn't exist
Path("s2s").mkdir(exist_ok=True)

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def read_files(file: str) -> list[str]:
    """Parse MIDI file and extract piano notes."""
    notes = []
    notes_to_parse = None
    
    # Parse the midi file
    midi = converter.parse(file)
    
    # Separate all instruments from the file
    instrmt = instrument.partitionByInstrument(midi)

    if instrmt is None:
        return notes
    
    for part in instrmt.parts:
        # Fetch data only of Piano instrument
        if "Piano" in str(part):
            notes_to_parse = part.recurse()

            # Iterate over all the parts of sub stream elements
            # Check if element's type is Note or chord
            # If it is chord, split them into notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


class MusicLSTM(nn.Module):
    """LSTM-based neural network for music generation."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_classes: int = 100):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        # Take only the last output
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


# Retrieve paths recursively from inside the directories/files
file_path = ["schubert"]
all_files = glob.glob(f"All Midi Files/{file_path[0]}/*.mid", recursive=True)

# Reading each midi file
notes_array = np.array([read_files(i) for i in tqdm(all_files, position=0, leave=True)], dtype=object)

# Unique notes
notess = sum(notes_array.tolist(), [])
unique_notes = list(set(notess))
print(f"Unique Notes: {len(unique_notes)}")

# Notes with their frequency
freq = {note: notess.count(note) for note in unique_notes}

# Get the threshold frequency
print("\nFrequency notes")
for i in range(30, 100, 20):
    count = len([note_freq for note_freq in freq.values() if note_freq >= i])
    print(f"{i}: {count}")

# Filter notes greater than threshold (i.e., 50)
freq_notes = {note: freq_val for note, freq_val in freq.items() if freq_val >= 50}

# Create new notes using the frequent notes
new_notes = [[n for n in sequence if n in freq_notes] for sequence in notes_array]

# Dictionary having key as note index and value as note
ind2note = dict(enumerate(freq_notes))

# Dictionary having key as note and value as note index
note2ind = {note: idx for idx, note in ind2note.items()}

# Timestep
timesteps = 50

# Store values of input and output
x: list[list[int]] = []
y: list[int] = []

for sequence in new_notes:
    for j in range(len(sequence) - timesteps):
        # Input will be the current index + timestep
        # Output will be the next index after timestep
        inp = sequence[j : j + timesteps]
        out = sequence[j + timesteps]

        # Append the index value of respective notes
        x.append([note2ind[note] for note in inp])
        y.append(note2ind[out])

x_new = np.array(x, dtype=np.int64)
y_new = np.array(y, dtype=np.int64)

# Reshape input and output for the model
x_new = np.reshape(x_new, (len(x_new), timesteps, 1))

# Split the input and value into training and testing sets
# 80% for training and 20% for testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
x_test_tensor = torch.FloatTensor(x_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Create the model
model = MusicLSTM(input_size=1, hidden_size=256, num_classes=len(note2ind)).to(device)
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
batch_size = 768
epochs = 500
num_batches = max(1, len(x_train_tensor) // batch_size)
best_val_loss = float("inf")
patience = 50
patience_counter = 0

print(f"\nTraining for {epochs} epochs...")
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for i in range(0, len(x_train_tensor), batch_size):
        batch_x = x_train_tensor[i : i + batch_size]
        batch_y = y_train_tensor[i : i + batch_size]
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for i in range(0, len(x_test_tensor), batch_size):
            batch_x = x_test_tensor[i : i + batch_size]
            batch_y = y_test_tensor[i : i + batch_size]
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    train_acc = 100 * correct / max(1, total)
    val_acc = 100 * val_correct / max(1, val_total)
    avg_train_loss = train_loss / num_batches
    val_batches = max(1, len(x_test_tensor) // batch_size)
    avg_val_loss = val_loss / val_batches

    # Model checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "s2s/model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
        break

print("\nTraining completed! Best model was saved to s2s/model.pth")

# Save mappings and test data for inference
mappings = {
    "note2ind": note2ind,
    "ind2note": ind2note,
}
with open("s2s/mappings.pkl", "wb") as f:
    pickle.dump(mappings, f)
print("Mappings saved to s2s/mappings.pkl")

# Save test data for inference
with open("s2s/x_test.pkl", "wb") as f:
    pickle.dump(x_test, f)
print("Test data saved to s2s/x_test.pkl")
model.load_state_dict(torch.load("s2s/model.pth", map_location=device))
model.eval()

# Generate random index
index = np.random.randint(0, len(x_test) - 1)

# Get the data of generated index from x_test
music_pattern = x_test[index].copy()

out_pred: list[str] = []  # Store predicted notes

print("\nGenerating music...")
# Generate 200 notes
with torch.no_grad():
    for _ in range(200):
        # Reshape the music pattern
        music_pattern_tensor = torch.FloatTensor(music_pattern.reshape(1, len(music_pattern), 1)).to(device)

        # Get the predicted output
        pred_output = model(music_pattern_tensor)
        
        # Get the maximum probability value from the predicted output
        pred_index = torch.argmax(pred_output, dim=1).item()
        
        # Get the note using predicted index and append to output prediction list
        out_pred.append(ind2note[pred_index])
        
        # Update music pattern with new prediction
        music_pattern = np.append(music_pattern, pred_index)
        music_pattern = music_pattern[1:]

output_notes: list[chord.Chord | note.Note] = []

for offset, pattern in enumerate(out_pred):
    # If pattern is a chord instance
    if "." in pattern or pattern.isdigit():
        # Split notes from the chord
        notes_in_chord = pattern.split(".")
        notes = []
        for current_note in notes_in_chord:
            try:
                i_curr_note = int(current_note)
                # Cast the current note to Note object and append
                new_note = note.Note(i_curr_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            except (ValueError, IndexError):
                continue
        
        if notes:
            # Cast to Chord object
            # offset will be 1 step ahead from the previous note
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
    else:
        # Cast the pattern to Note object, apply the offset
        try:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        except Exception:
            continue

# Save the midi file
midi_stream = stream.Stream(output_notes)
midi_stream.write("midi", fp="pred_music.mid")
print("Generated music saved to pred_music.mid")
