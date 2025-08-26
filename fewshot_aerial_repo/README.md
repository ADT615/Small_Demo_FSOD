# Few-Shot Aerial Object Detection (TFA Demo)

This repo is a **clean, minimal demo** of **Few-Shot Object Detection** in the **TFA** style using **PyTorch + torchvision**.
It is **self-contained**: a synthetic aerial-like dataset is generated on the fly, so **no external dataset** is required.
If you later switch to **VisDrone**, keep the same two-stage pipeline.

## Method
- **Stage 1**: Train Faster R-CNN (MobileNetV3 FPN, pretrained) on **base classes only**.
- **Stage 2**: **Few-shot fine-tune** with base + novel (K-shot for novel). We use the **ft-last** freezing scheme
  (freeze `backbone + RPN + box_head`, train **only** `roi_heads.box_predictor`). Optionally replace the linear classification
  head with a **CosineClassifier** (metric learning).

## Structure
```
fewshot_aerial_repo/
├─ checkpoints/                 # saved checkpoints
├─ src/
│  ├─ config.py                 # constants (classes, epochs, img size, flags)
│  ├─ dataset_synth.py          # synthetic aerial-like dataset
│  ├─ model_tfa.py              # model, cosine classifier, freeze utils
│  ├─ train_base.py             # Stage-1 training
│  ├─ finetune_tfa.py           # Stage-2 few-shot fine-tuning (TFA ft-last)
│  ├─ eval_map50.py             # simple mAP@0.5 evaluation
│  └─ viz.py                    # visualize a few predictions
└─ requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt

# Stage-1 (base only)
python -m src.train_base

# Stage-2 (few-shot fine-tuning, cosine head optional via config flag)
python -m src.finetune_tfa

# Evaluate mAP@0.5 on synthetic val
python -m src.eval_map50

# Visualize some predictions
python -m src.viz
```

## Notes
- Classes: `1=car` (base), `2=drone` (novel). `0` is background (implicit).
- Synthetic dataset is used for demo. For real aerial data (e.g., **VisDrone**), keep the same two-stage pipeline and replace the dataset.
- For a stronger demo: switch to VisDrone subset, compute **COCO AP**, report **mAP@0.5** and add screenshots.

## Requirements
See `requirements.txt`.
