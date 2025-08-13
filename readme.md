

## 🛠️ Tools Overview

| Tool                                                                   | Description                        | Key Features                                                                     |
| ---------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------- |
| [`1DCNNTrain.py`](#1d-cnn-classifier-1dcnntrainpy)                      | 1D CNN for waveform classification | - K-fold cross-validation `<br>`- Early stopping `<br>`- Performance metrics |
| [`HoldTimeCalculator.py`](#breath-hold-calculator-holdtimecalculatorpy) | Breath-hold duration detection     | - Plateau detection `<br>`- Time calculation `<br>`- Visualization           |
| [`PCA_KMeans.py`](#waveform-clustering-pca_kmeanspy)                    | Waveform pattern clustering        | - Two-stage clustering `<br>`- PCA reduction `<br>`- Dynamic expansion       |

## 1️⃣ 1D CNN Classifier (`1DCNNTrain.py`)

### 🚀 Features

- Multi-class classification (3 default classes)
- Stratified 5-fold cross-validation
- Model checkpointing and early stopping
- Comprehensive performance metrics

### 🏃‍♂️ Usage

```bash
python 1DCNNTrain.py
```

### 📊 Outputs

```
/trained_model/
    ├── model_weights.h5
    ├── training_history.png
    ├── confusion_matrix.png
    └── performance_metrics.json
```

## 2️⃣ Hold Time Calculator (`HoldTimeCalculator.py`)

### 🔍 Detection Algorithm

1. Edge artifact removal
2. Sliding window analysis
3. Plateau start/end detection
4. Time duration calculation

### 🛠️ Configuration

```python
# Modify these parameters in the script:
WINDOW_SIZE = 5       # Detection window size
THRESHOLD = 0.01      # Relative change threshold
EDGE_TRIM = 10        # Samples to trim from edges
```

### 📈 Example Output

```
Hold time: 3.42 seconds
```

## 3️⃣ Waveform Clustering (`PCA_KMeans.py`)

### 🔧 Pipeline Stages

1. **Data Preparation**

   - Waveform flattening (300×3 → 900D)
   - Standardization
2. **Dimensionality Reduction**

   - PCA with 99% variance retention
   - Automatic component selection
3. **Dynamic Clustering**

   - Stage 1: MiniBatchKMeans (5,337 initial clusters)
   - Stage 2: Distance-based expansion (99th percentile threshold)

### 📂 Data Requirements

- Input shape: `(n_samples, 300, 3)`
- Required files:
  - `all_waveforms.npy`
  - `metadata.csv`

## 📂 Project Structure

```
.
├── Data/                   # Sample data directory
├── results/                # Auto-generated output folder
├── 1DCNNTrain.py           # CNN classifier
├── HoldTimeCalculator.py   # Hold time Calculator  
├── PCA_KMeans.py           # Clustering pipeline
└── README.md               # This document
```

```

```
