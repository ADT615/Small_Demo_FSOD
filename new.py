# fewshot_tfa_demo.py
# ---------------------------------------------------------
# Demo Few-Shot Object Detection (FSOD) kiểu TFA trên torchvision Faster R-CNN
# - Stage-1: Train chỉ lớp "base"
# - Stage-2: Few-shot fine-tune (base + novel), freeze ft-last, optionally Cosine Classifier
# - Synthetic aerial-like data (không cần dataset ngoài)
# - Đánh giá mAP@0.5 đơn giản + visualize
# ---------------------------------------------------------

import math
import random
import numpy as np
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Corrected import

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# -----------------------------
# 0) Cấu hình
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lớp: 1=car (base), 2=drone (novel). 0 là background (ngầm định, KHÔNG xuất hiện trong targets)
CLASS_ID_TO_NAME = {1: "car", 2: "drone"}
NUM_CLASSES = 1 + len(CLASS_ID_TO_NAME)   # background + 2

# K-shot cho novel ở Stage-2 (số ảnh đầu tiên sẽ ép có ít nhất 1 novel)
K_SHOT_NOVEL = 5

# Bật/tắt Cosine Classifier (metric learning) ở Stage-2
USE_COSINE_CLASSIFIER = True  # đặt False nếu muốn giữ Linear head mặc định


# -----------------------------
# 1) Dataset synthetic aerial
# -----------------------------
class SyntheticAerialDataset(Dataset):
    """
    Sinh ảnh nền xám + hộp màu đại diện object.
    - include_classes: tuple các class_id sẽ xuất hiện (vd Stage-1: (1,), Stage-2/val: (1,2))
    - force_novel_first_k: ép các index < K phải có ít nhất một object novel (class=2)
    """
    def __init__(self,
                 num_images: int,
                 include_classes: Tuple[int, ...],
                 img_size: int = 256,
                 force_novel_first_k: int = 0):
        self.num_images = num_images
        self.include_classes = include_classes
        self.img_size = img_size
        self.force_novel_first_k = force_novel_first_k

    def __len__(self):
        return self.num_images

    def _add_object_rect(self, img: np.ndarray, class_id: int) -> List[int]:
        """Vẽ 1 rectangle, trả bbox [x1,y1,x2,y2]."""
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

        # Số object ngẫu nhiên
        num_objs = np.random.randint(1, 4)

        # Nếu ép novel ở các ảnh đầu (few-shot K ảnh)
        must_have_novel = (2 in self.include_classes) and (idx < self.force_novel_first_k)

        for j in range(num_objs):
            if must_have_novel and j == 0:
                class_id = 2
            else:
                class_id = int(np.random.choice(self.include_classes))
            boxes.append(self._add_object_rect(img, class_id))
            labels.append(class_id)

        # Convert sang tensor; Ảnh rỗng -> trả tensor rỗng (KHÔNG gán label=0)
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


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# 2) Cosine Classifier (metric)
# -----------------------------
class CosineClassifier(nn.Module):
    """
    Thay Linear cls_score bằng Cosine classifier (chuẩn hoá feature & weight)
    Output shape: [N, NUM_CLASSES]
    """
    def __init__(self, in_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * x @ w.t()
        return logits


# -----------------------------
# 3) Model & TFA utils
# -----------------------------
def get_model(num_classes: int = NUM_CLASSES):
    """
    Faster R-CNN + MobileNetV3 FPN (pretrained) -> thay head đúng num_classes.
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def replace_cls_with_cosine(model):
    """
    Thay Linear cls_score bằng CosineClassifier nhưng GIỮ NGUYÊN num_classes tổng.
    (bbox_pred vẫn giữ nguyên; không đụng vào)
    """
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    num_cls = model.roi_heads.box_predictor.cls_score.out_features
    model.roi_heads.box_predictor.cls_score = CosineClassifier(in_feat, num_cls, scale=20.0)
    return model


def freeze_ft_last_tfa(model):
    """
    TFA ft-last: freeze backbone + RPN + box_head; chỉ train box_predictor (cls & bbox).
    """
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.rpn.parameters(): p.requires_grad = False
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = False


# -----------------------------
# 4) Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate_map50(model, loader, iou_thr=0.5):
    """
    mAP@0.5 đơn giản (không COCOeval): per-class AP sau đó macro-average.
    """
    model.eval()
    all_preds: Dict[int, List[Tuple[float, int, np.ndarray]]] = {c: [] for c in CLASS_ID_TO_NAME}
    # all_preds[c]: list of (score, img_id, box)
    gts: Dict[int, Dict[int, List[np.ndarray]]] = {c: {} for c in CLASS_ID_TO_NAME}
    # gts[c][img_id]: list of GT boxes

    img_id = 0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)
        batch = len(images)
        for b in range(batch):
            # GT
            gt_boxes = targets[b]["boxes"].numpy()
            gt_labels = targets[b]["labels"].numpy()
            for c in CLASS_ID_TO_NAME:
                gts[c].setdefault(img_id, [])
                for bb, ll in zip(gt_boxes, gt_labels):
                    if ll == c:
                        gts[c][img_id].append(bb.astype(np.float32))
            # Pred
            pred = outputs[b]
            p_boxes = pred["boxes"].detach().cpu().numpy()
            p_scores = pred["scores"].detach().cpu().numpy()
            p_labels = pred["labels"].detach().cpu().numpy()
            for bb, sc, ll in zip(p_boxes, p_scores, p_labels):
                if ll in CLASS_ID_TO_NAME:
                    all_preds[ll].append((float(sc), img_id, bb.astype(np.float32)))
            img_id += 1

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter + 1e-6
        return inter/union

    aps = []
    for c in CLASS_ID_TO_NAME:
        preds = sorted(all_preds[c], key=lambda x: x[0], reverse=True)  # sort by score
        # Build GT match flags
        gt_used = {img: np.zeros(len(gts[c].get(img, [])), dtype=bool) for img in gts[c]}
        tp = []; fp = []
        for sc, img_idx, box in preds:
            gtb = gts[c].get(img_idx, [])
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gtb):
                i = iou(box, g)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= iou_thr and best_j >= 0 and not gt_used[img_idx][best_j]:
                tp.append(1); fp.append(0); gt_used[img_idx][best_j] = True
            else:
                tp.append(0); fp.append(1)
        if len(tp) == 0:
            aps.append(0.0)
            continue
        tp = np.cumsum(tp); fp = np.cumsum(fp)
        recall = tp / (sum(len(v) for v in gts[c].values()) + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        # AP bằng cách lấy bao phủ precision-envelope rồi tích phân theo recall
        # (xấp xỉ đơn giản)
        ap = 0.0
        prev_r = 0.0
        for r in np.linspace(0, 1, 101):
            p_at_r = np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0
            ap += p_at_r
        ap /= 101.0
        aps.append(ap)
    map50 = float(np.mean(aps)) if len(aps) else 0.0
    return map50, aps


def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


# -----------------------------
# 5) Viz
# -----------------------------
@torch.no_grad()
def visualize_samples(model, dataset, n=3, score_thr=0.5, title="Predictions"):
    model.eval()
    idxs = np.linspace(0, len(dataset)-1, num=min(n, len(dataset)), dtype=int)
    for i in idxs:
        img, _ = dataset[i]
        out = model([img.to(DEVICE)])[0]
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img_np)
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        for b, s, l in zip(boxes, scores, labels):
            if s < score_thr:
                continue
            x1, y1, x2, y2 = b
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           fill=False, lw=2, edgecolor='r'))
            name = CLASS_ID_TO_NAME.get(int(l), f"id{int(l)}")
            ax.text(x1, y1, f"{name} {s:.2f}", bbox=dict(facecolor='yellow', alpha=0.6))
        ax.set_title(title); ax.axis("off")
        plt.show()


# -----------------------------
# 6) Main
# -----------------------------
def main():
    print("=== Few-Shot OD (TFA) – Synthetic Aerial ===")
    print("Device:", DEVICE)

    # ----- Stage-1: Train base (class=1 only)
    base_ds = SyntheticAerialDataset(
        num_images=600,
        include_classes=(1,),      # chỉ car
        img_size=256,
        force_novel_first_k=0
    )
    base_loader = DataLoader(base_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

    print("\n[Stage-1] Training on base class only...")
    for epoch in range(6):
        loss = train_one_epoch(model, base_loader, opt)
        print(f"  Epoch {epoch+1}/6 - loss={loss:.4f}")

    # ----- Stage-2: Few-shot fine-tune (base + novel), ép K ảnh đầu có novel
    ft_ds = SyntheticAerialDataset(
        num_images=60,
        include_classes=(1, 2),    # car + drone
        img_size=256,
        force_novel_first_k=K_SHOT_NOVEL  # K-shot novel
    )
    ft_loader = DataLoader(ft_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Freeze ft-last
    freeze_ft_last_tfa(model)

    # (Optional) thay Linear thành CosineClassifier cho cls_score
    if USE_COSINE_CLASSIFIER:
        model = replace_cls_with_cosine(model)

    # Chỉ train box_predictor (cls & bbox)
    params = list(model.roi_heads.box_predictor.parameters())
    opt2 = torch.optim.SGD(params, lr=1e-3, momentum=0.9)

    print("\n[Stage-2] Few-shot fine-tuning (ft-last)...")
    for epoch in range(4):
        loss = train_one_epoch(model, ft_loader, opt2)
        print(f"  Epoch {epoch+1}/4 - loss={loss:.4f}")

    # ----- Validation (synthetic val)
    val_ds = SyntheticAerialDataset(
        num_images=80,
        include_classes=(1, 2),
        img_size=256,
        force_novel_first_k=0
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print("\n[Eval] computing mAP@0.5 on synthetic val...")
    map50, per_cls = evaluate_map50(model, val_loader, iou_thr=0.5)
    print(f"mAP@0.5 (macro): {map50:.3f}")
    for cid, ap in zip(CLASS_ID_TO_NAME, per_cls):
        print(f"  AP@0.5[{CLASS_ID_TO_NAME[cid]}]: {ap:.3f}")

    # ----- Viz
    print("\n[Viz] show a few predictions (thr=0.5)")
    visualize_samples(model, val_ds, n=3, score_thr=0.5, title="Predictions (thr=0.5)")

    print("\nDone. Tips:")
    print("- Bạn có thể tăng epoch Stage-1/2 để mAP cao hơn.")
    print("- Đặt USE_COSINE_CLASSIFIER=False nếu muốn dùng Linear head thuần.")
    print("- Khi chuyển qua dataset thật (VisDrone), giữ nguyên pipeline (TFA).")


if __name__ == "__main__":
    main()