HEAD
# model-labeling-soundscape-ecology

## Features
- MFCC + Spectral + Temporal features
- Feature aggregation (mean, std, max, min)
- PCA dimensionality reduction
- KMeans clustering (k=2)
- Isolation Forest outlier detection

## Usage

1. Put your .wav files in `data/`
2. Install dependencies:
   pip install -r requirements.txt

3. Run:
   python main.py

## Output
- features.npy → PCA features
- labels.npy → cluster labels
- outliers.npy → anomaly detection
 1831f84 (first commit)
