# XAI Guided MRI

Binary classification of brain MRI scans using ResNet50 with Grad-CAM attention alignment.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download BraTS2020 dataset and update `DATA_DIR` in `main.py`:

```python
DATA_DIR = '/path/to/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
```

## Run

```bash
python main.py
```

## Output

Results saved to `guided_binary_classification/`:
- `models/` - trained model
- `visualizations/` - plots
- `heatmaps/` - Grad-CAM maps
- `logs/` - metrics
