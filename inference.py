"""
Music Generation - Inference Phase
Generate new piano music using a trained PyTorch model.
Modernized for Python 3.11+ with PyTorch.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from music21 import chord, instrument, note, stream
from tqdm import tqdm


class MusicLSTM(nn.Module):
    """LSTM-based neural network for music generation."""

    def __init__(
        self, input_size: int, hidden_size: int = 256, num_classes: int = 100
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
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


def load_mappings(mappings_file: str = "s2s/mappings.pkl") -> tuple[dict, dict, int]:
    """
    Load note-to-index and index-to-note mappings.

    Args:
        mappings_file: Path to the mappings pickle file

    Returns:
        Tuple of (note2ind, ind2note, num_notes)
    """
    try:
        with open(mappings_file, "rb") as f:
            mappings = pickle.load(f)
        note2ind = mappings["note2ind"]
        ind2note = mappings["ind2note"]
        return note2ind, ind2note, len(ind2note)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mappings file not found at {mappings_file}.\n"
            "Make sure to run auto_music_gen.py first to train the model."
        )


def load_model(
    model_path: str = "s2s/model.pth", num_notes: int = 100, device: str = "cpu"
) -> MusicLSTM:
    """
    Load the trained PyTorch model.

    Args:
        model_path: Path to the model weights
        num_notes: Number of unique notes in the vocabulary
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in eval mode
    """
    try:
        model = MusicLSTM(input_size=1, hidden_size=256, num_classes=num_notes)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(device))
        )
        model.eval()
        model.to(device)
        print(f"✅ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at {model_path}.\n"
            "Make sure to run auto_music_gen.py first to train the model."
        )
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model: {e}")


def generate_music(
    model: MusicLSTM,
    ind2note: dict[int, str],
    music_pattern: np.ndarray,
    num_notes_to_generate: int = 200,
    device: str = "cpu",
) -> list[str]:
    """
    Generate music notes using the trained model.

    Args:
        model: Trained PyTorch model
        ind2note: Dictionary mapping indices to note names
        music_pattern: Initial pattern (shape: (timesteps,))
        num_notes_to_generate: Number of notes to generate
        device: Device to run inference on

    Returns:
        List of generated note names
    """
    out_pred: list[str] = []

    print(f"\n🎵 Generating {num_notes_to_generate} notes...")

    with torch.no_grad():
        for _ in tqdm(range(num_notes_to_generate), desc="Generating"):
            # Reshape the music pattern for model input: (1, timesteps, 1)
            music_pattern_tensor = torch.FloatTensor(
                music_pattern.reshape(1, len(music_pattern), 1)
            ).to(device)

            # Get prediction from model
            pred_output = model(music_pattern_tensor)

            # Get the note index with highest probability
            pred_index = torch.argmax(pred_output, dim=1).item()

            # Convert index to note name and store
            out_pred.append(ind2note[pred_index])

            # Update pattern: append new prediction and remove oldest
            music_pattern = np.append(music_pattern, pred_index)
            music_pattern = music_pattern[1:]

    return out_pred


def create_midi_from_notes(
    out_pred: list[str], output_file: str = "pred_music.mid"
) -> None:
    """
    Convert predicted notes to a MIDI file.

    Args:
        out_pred: List of predicted note names (including chords)
        output_file: Output MIDI file path
    """
    output_notes: list[chord.Chord | note.Note] = []

    print(f"\n🎹 Converting notes to MIDI...")

    for offset, pattern in enumerate(out_pred):
        try:
            # Check if pattern is a chord (contains '.' or is numeric)
            if "." in pattern or pattern.isdigit():
                # Parse chord
                notes_in_chord = pattern.split(".")
                notes = []

                for current_note_str in notes_in_chord:
                    try:
                        note_value = int(current_note_str)
                        new_note = note.Note(note_value)
                        new_note.storedInstrument = instrument.Piano()
                        notes.append(new_note)
                    except (ValueError, IndexError):
                        continue

                if notes:
                    # Create chord with notes
                    new_chord = chord.Chord(notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)

            else:
                # Parse single note
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

        except Exception as e:
            print(f"⚠️  Warning: Could not parse note '{pattern}': {e}")
            continue

    # Create stream and save
    if output_notes:
        midi_stream = stream.Stream(output_notes)
        midi_stream.write("midi", fp=output_file)
        print(f"✅ MIDI file saved to {output_file}")
        print(f"   Total notes/chords: {len(output_notes)}")
    else:
        print("❌ No valid notes to save!")


def generate_with_seed(
    model: MusicLSTM,
    note2ind: dict[str, int],
    ind2note: dict[int, str],
    seed_pattern: list[str] | None = None,
    num_notes: int = 200,
    device: str = "cpu",
) -> list[str]:
    """
    Generate music with a custom seed pattern.

    Args:
        model: Trained model
        note2ind: Note to index mapping
        ind2note: Index to note mapping
        seed_pattern: Starting pattern (list of note names)
        num_notes: Number of notes to generate
        device: Device to use

    Returns:
        List of generated notes
    """
    if seed_pattern is None:
        # Use random starting pattern
        timesteps = 50
        music_pattern = np.random.randint(0, len(ind2note), timesteps)
    else:
        # Convert seed pattern to indices
        music_pattern = np.array(
            [
                note2ind.get(n, 0) for n in seed_pattern
            ]
        )

    return generate_music(model, ind2note, music_pattern, num_notes, device)


def main():
    """Main inference pipeline."""
    print("=" * 60)
    print("🎵 Music Generation - Inference Phase")
    print("=" * 60)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Using device: {device}")

    # Load mappings and model
    try:
        note2ind, ind2note, num_notes = load_mappings()
        print(f"✅ Loaded {num_notes} unique notes")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    try:
        model = load_model(num_notes=num_notes, device=device)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        return

    # Load test data for random starting pattern
    try:
        with open("s2s/x_test.pkl", "rb") as f:
            x_test = pickle.load(f)
        # Generate random starting index
        index = np.random.randint(0, len(x_test) - 1)
        music_pattern = x_test[index].copy()
        print(f"✅ Loaded test data (using random pattern #{index})")
    except FileNotFoundError:
        print(
            "⚠️  Test data not found, using random starting pattern"
        )
        timesteps = 50
        music_pattern = np.random.randint(0, num_notes, timesteps)

    # Generate music
    generated_notes = generate_music(
        model,
        ind2note,
        music_pattern,
        num_notes_to_generate=200,
        device=device,
    )

    # Create and save MIDI file
    create_midi_from_notes(generated_notes, "pred_music.mid")

    print("\n" + "=" * 60)
    print("✨ Music generation complete!")
    print(f"📁 Output: pred_music.mid")
    print("=" * 60)


if __name__ == "__main__":
    main()
