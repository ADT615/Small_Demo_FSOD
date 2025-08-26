import torch, random, numpy as np
from src.config import (SEED, DEVICE, CLASS_ID_TO_NAME, NUM_CLASSES, IMG_SIZE, STAGE2_CKPT)
from src.dataset_synth import SyntheticAerialDataset
from src.model_tfa import get_model
from src.utils import visualize_samples

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def main():
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(STAGE2_CKPT, map_location=DEVICE))

    ds = SyntheticAerialDataset(num_images=12, include_classes=(1,2), img_size=IMG_SIZE)
    visualize_samples(model, ds, CLASS_ID_TO_NAME, DEVICE, n=3, score_thr=0.5, title="Predictions (thr=0.5)")

if __name__ == "__main__":
    main()
