# src/train_base.py
import torch
from torch.utils.data import DataLoader
from src.config import (SEED, DEVICE, CLASS_ID_TO_NAME, NUM_CLASSES,
                        STAGE1_EPOCHS, BATCH_SIZE, IMG_SIZE, STAGE1_CKPT, CKPT_DIR)
from src.dataset_synth import SyntheticAerialDataset
from src.model_tfa import get_model
from src.utils import collate_fn

import os, random, numpy as np
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    # Stage-1 dataset: base only (class=1)
    ds = SyntheticAerialDataset(num_images=600, include_classes=(1,), img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

    print("[Stage-1] Training on base class only...")
    for ep in range(STAGE1_EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, DEVICE)
        print(f"  Epoch {ep+1}/{STAGE1_EPOCHS} - loss={loss:.4f}")

    torch.save(model.state_dict(), STAGE1_CKPT)
    print(f"Saved Stage-1 checkpoint to {STAGE1_CKPT}")

if __name__ == "__main__":
    main()
