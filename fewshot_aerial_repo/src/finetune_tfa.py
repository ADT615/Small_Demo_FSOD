import torch, os, random, numpy as np
from torch.utils.data import DataLoader
from src.config import (SEED, DEVICE, CLASS_ID_TO_NAME, NUM_CLASSES, K_SHOT_NOVEL,
                        STAGE2_EPOCHS, BATCH_SIZE, IMG_SIZE, STAGE1_CKPT, STAGE2_CKPT,
                        USE_COSINE_CLASSIFIER, CKPT_DIR)
from src.dataset_synth import SyntheticAerialDataset
from src.model_tfa import get_model, freeze_ft_last_tfa, replace_cls_with_cosine
from src.utils import collate_fn

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
    # Build model & load Stage-1 weights
    model = get_model(NUM_CLASSES).to(DEVICE)
    assert os.path.exists(STAGE1_CKPT), f"Missing Stage-1 checkpoint: {STAGE1_CKPT}"
    model.load_state_dict(torch.load(STAGE1_CKPT, map_location=DEVICE))

    # Freeze ft-last
    freeze_ft_last_tfa(model)

    # Optional: replace classification head with cosine classifier
    if USE_COSINE_CLASSIFIER:
        replace_cls_with_cosine(model)

    # Few-shot dataset: include base + novel; force first K images to contain novel
    ds = SyntheticAerialDataset(num_images=60, include_classes=(1,2), img_size=IMG_SIZE,
                                force_novel_first_k=K_SHOT_NOVEL)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Only train roi_heads.box_predictor params
    params = list(model.roi_heads.box_predictor.parameters())
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)

    print("[Stage-2] Few-shot fine-tuning (ft-last)...")
    for ep in range(STAGE2_EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, DEVICE)
        print(f"  Epoch {ep+1}/{STAGE2_EPOCHS} - loss={loss:.4f}")

    torch.save(model.state_dict(), STAGE2_CKPT)
    print(f"Saved Stage-2 checkpoint to {STAGE2_CKPT}")

if __name__ == "__main__":
    main()
