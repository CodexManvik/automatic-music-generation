# 📊 System & Model Flowcharts

This document details the architectural pipelines for the Automatic Music Generation project using Mermaid diagrams.

---

## 1. Overall System Architecture
The high-level pipeline from raw MIDI data to AI-generated musical compositions.

```mermaid
graph TD
    subgraph "Data Preparation"
        A[MIDI Dataset] --> B[music21 Parser]
        B --> C[Piano Track Isolation]
        C --> D[Note & Chord Tokenization]
        D --> E[Frequency Threshold Filter]
        E --> F[Integer Encoding]
    end

    subgraph "Dataset Engineering"
        F --> G[Sliding Window: 50 Timesteps]
        G --> H[Input X: Sequence of 50]
        G --> I[Target y: 51st Note]
        H & I --> J[PyTorch Tensors]
    end

    subgraph "Training Phase"
        J --> K[MusicLSTM Model]
        K --> L[CrossEntropyLoss]
        L --> M[Adam Optimizer]
        M --> N{Lowest Val Loss?}
        N -- Yes --> O[Save Checkpoint: model.pth]
        N -- No --> P[Early Stopping Counter]
    end

    subgraph "Inference & Generation"
        O --> Q[Load Best Weights]
        Q --> R[Random Seed Sequence]
        R --> S[Autoregressive Prediction Loop]
        S --> T[Map Indices to music21 Objects]
        T --> U[Final: pred_music.mid]
    end

    style O fill:#2ecc71,stroke:#27ae60,color:#fff
    style A fill:#3498db,stroke:#2980b9,color:#fff
    style U fill:#e67e22,stroke:#d35400,color:#fff
```

---

## 2. MusicLSTM Neural Network Architecture
The internal layer structure of the deep learning model defined in `auto_music_gen.py`.

```mermaid
graph TD
    Input([Input Sequence Shape: Batch, 50, 1]) --> LSTM1[LSTM Layer 1: 256 Units]
    
    subgraph "Recurrent Block 1"
        LSTM1 --> DO1[Dropout: 0.3]
    end
    
    DO1 --> LSTM2[LSTM Layer 2: 256 Units]
    
    subgraph "Recurrent Block 2"
        LSTM2 --> DO2[Dropout: 0.3]
    end
    
    DO2 --> Slice[Select Last Timestep Output]
    
    subgraph "Dense Head"
        Slice --> Dense1[Linear Layer: 256]
        Dense1 --> ReLU[ReLU Activation]
        ReLU --> Dense2[Output Layer: Vocab Size]
    end
    
    Dense2 --> Softmax[Softmax Probability Dist]

    style Input fill:#7f8c8d,stroke:#34495e,color:#fff
    style LSTM1 fill:#9b59b6,stroke:#8e44ad,color:#fff
    style LSTM2 fill:#9b59b6,stroke:#8e44ad,color:#fff
    style Dense2 fill:#e74c3c,stroke:#c0392b,color:#fff
```

---

## 🎵 Model Summary Table

| Layer Type | Configuration | Purpose |
| :--- | :--- | :--- |
| **LSTM 1** | 256 Units, Batch First | Captures low-level temporal features. |
| **Dropout** | 0.3 Probability | Prevents overfitting to training sequences. |
| **LSTM 2** | 256 Units, Batch First | Learns complex musical dependencies. |
| **Linear 1** | 256 Nodes | Compresses temporal features. |
| **Activation** | ReLU | Introduces non-linearity. |
| **Linear 2** | Vocab Size | Maps features to note/chord probabilities. |
