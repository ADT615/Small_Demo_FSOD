# Few-Shot Object Detection (FSOD) Project

Implementation of Few-Shot Object Detection using TFA (Two-stage Fine-tuning Approach) with synthetic aerial data.

## Features

- Two-stage training pipeline (base + few-shot fine-tuning)
- Synthetic aerial dataset generation
- Support for both base (car) and novel (drone) classes
- Cosine Classifier option for better few-shot learning
- Simple mAP@0.5 evaluation
- Visualization tools

## Project Structure

- `src/`: Source code directory
  - `config.py`: Configuration parameters
  - `dataset_synth.py`: Synthetic dataset generation
  - `model_tfa.py`: Model architecture and TFA implementation
  - `train_base.py`: Base training (Stage-1)
  - `finetune_tfa.py`: Few-shot fine-tuning (Stage-2)
  - `eval_map50.py`: Evaluation metrics
  - `utils.py`: Utility functions

## Usage

1. Base Training (Stage-1):
```bash
python src/train_base.py
```

2. Few-shot Fine-tuning (Stage-2):
```bash
python src/finetune_tfa.py
```

3. Evaluation:
```bash
python src/eval_map50.py
```

## Configuration

- `K_SHOT_NOVEL`: Number of shots for novel class (default: 5)
- `USE_COSINE_CLASSIFIER`: Enable/disable cosine classifier
- Image size, batch size, and other parameters can be modified in `config.py`

## Notes

- The project uses synthetic data for demonstration
- Can be extended to real datasets like VisDrone
- TFA approach helps preserve base knowledge while learning novel classes

