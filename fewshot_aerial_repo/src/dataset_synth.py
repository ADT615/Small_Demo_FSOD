import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class SyntheticAerialDataset(Dataset):
    """
    Synthetic aerial-like dataset:
      - include_classes: tuple of class IDs that may appear in an image (e.g., Stage-1: (1,), Stage-2/val: (1,2))
      - force_novel_first_k: ensure that the first K images contain at least one novel (class=2) object (few-shot)
    """
    def __init__(self, num_images, include_classes=(1,), img_size=256, force_novel_first_k=0):
        self.num_images = num_images
        self.include_classes = include_classes
        self.img_size = img_size
        self.force_novel_first_k = force_novel_first_k

    def __len__(self):
        return self.num_images

    def _add_object_rect(self, img, class_id):
        H, W, _ = img.shape
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)
        x1 = np.random.randint(0, W - w)
        y1 = np.random.randint(0, H - h)
        x2, y2 = x1 + w, y1 + h
        color = (255, 0, 0) if class_id == 1 else (0, 255, 0)
        img[y1:y2, x1:x2] = color
        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        img = np.random.randint(100, 150, (self.img_size, self.img_size, 3), dtype=np.uint8)
        boxes, labels = [], []

        # Ensure 1..3 objects per image
        num_objs = np.random.randint(1, 4)

        must_have_novel = (2 in self.include_classes) and (idx < self.force_novel_first_k)

        for j in range(num_objs):
            if must_have_novel and j == 0:
                class_id = 2
            else:
                class_id = int(np.random.choice(self.include_classes))
            boxes.append(self._add_object_rect(img, class_id))
            labels.append(class_id)

        img_tensor = TF.to_tensor(Image.fromarray(img))
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
        return img_tensor, target
