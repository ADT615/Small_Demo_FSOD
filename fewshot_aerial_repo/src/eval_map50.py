import torch, random, numpy as np
from torch.utils.data import DataLoader
from src.config import (SEED, DEVICE, CLASS_ID_TO_NAME, NUM_CLASSES, IMG_SIZE, BATCH_SIZE, STAGE2_CKPT)
from src.dataset_synth import SyntheticAerialDataset
from src.model_tfa import get_model
from src.utils import collate_fn, evaluate_map50

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def main():
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(STAGE2_CKPT, map_location=DEVICE))

    val_ds = SyntheticAerialDataset(num_images=80, include_classes=(1,2), img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("[Eval] computing mAP@0.5 on synthetic val...")
    map50, per_cls = evaluate_map50(model, val_loader, CLASS_ID_TO_NAME, DEVICE, iou_thr=0.5)
    print(f"mAP@0.5 (macro): {map50:.3f}")
    # Print per-class AP in fixed order
    for cid in sorted(CLASS_ID_TO_NAME.keys()):
        print(f"  AP@0.5[{CLASS_ID_TO_NAME[cid]}]: {per_cls[cid-1]:.3f}")

if __name__ == "__main__":
    main()
