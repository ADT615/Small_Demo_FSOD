# src/config.py
# Global config/constants shared across scripts

import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping: 1=car (base), 2=drone (novel). 0=background (implicit).
CLASS_ID_TO_NAME = {1: "car", 2: "drone"}
NUM_CLASSES = 1 + len(CLASS_ID_TO_NAME)  # background + 2

# Few-shot settings
K_SHOT_NOVEL = 5  # number of first images in Stage-2 forced to contain at least one novel object

# Training hyper-params
STAGE1_EPOCHS = 6
STAGE2_EPOCHS = 4
BATCH_SIZE = 4
IMG_SIZE = 256

# Cosine classifier (metric learning) toggle for Stage-2
USE_COSINE_CLASSIFIER = True

# Paths
CKPT_DIR = "checkpoints"
STAGE1_CKPT = f"{CKPT_DIR}/stage1_base.pth"
STAGE2_CKPT = f"{CKPT_DIR}/stage2_tfa.pth"
