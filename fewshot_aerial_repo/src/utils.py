import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def evaluate_map50(model, loader, class_id_to_name, device, iou_thr=0.5):
    """
    Simple mAP@0.5 evaluator (per-class AP then macro-average).
    """
    model.eval()
    all_preds = {c: [] for c in class_id_to_name}   # {class_id: [(score, img_id, box), ...]}
    gts = {c: {} for c in class_id_to_name}         # {class_id: {img_id: [boxes...] }}

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter + 1e-6
        return inter/union

    img_id = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for b, out in enumerate(outputs):
            # GT
            gt_boxes = targets[b]["boxes"].numpy()
            gt_labels = targets[b]["labels"].numpy()
            for c in class_id_to_name:
                gts[c].setdefault(img_id, [])
                for bb, ll in zip(gt_boxes, gt_labels):
                    if ll == c:
                        gts[c][img_id].append(bb.astype(np.float32))

            # Pred
            p_boxes = out["boxes"].detach().cpu().numpy()
            p_scores = out["scores"].detach().cpu().numpy()
            p_labels = out["labels"].detach().cpu().numpy()
            for bb, sc, ll in zip(p_boxes, p_scores, p_labels):
                if ll in class_id_to_name:
                    all_preds[ll].append((float(sc), img_id, bb.astype(np.float32)))

            img_id += 1

    aps = []
    for c in class_id_to_name:
        preds = sorted(all_preds[c], key=lambda x: x[0], reverse=True)
        gt_used = {img: np.zeros(len(gts[c].get(img, [])), dtype=bool) for img in gts[c]}
        tp, fp = [], []
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
            aps.append(0.0); continue
        tp = np.cumsum(tp); fp = np.cumsum(fp)
        recall = tp / (sum(len(v) for v in gts[c].values()) + 1e-6)
        precision = tp / (tp + fp + 1e-6)

        # 101-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            p_at_r = np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0
            ap += p_at_r
        ap /= 101.0
        aps.append(ap)

    map50 = float(np.mean(aps)) if len(aps) else 0.0
    return map50, aps

@torch.no_grad()
def visualize_samples(model, dataset, class_id_to_name, device, n=3, score_thr=0.5, title="Predictions"):
    model.eval()
    import numpy as np
    idxs = np.linspace(0, len(dataset)-1, num=min(n, len(dataset)), dtype=int)
    for i in idxs:
        img, _ = dataset[i]
        out = model([img.to(device)])[0]
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
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, lw=2))
            name = class_id_to_name.get(int(l), f"id{int(l)}")
            ax.text(x1, y1, f"{name} {s:.2f}", bbox=dict(facecolor='yellow', alpha=0.6))
        ax.set_title(title); ax.axis("off")
        plt.show()
