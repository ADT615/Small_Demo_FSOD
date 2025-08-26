# src/model_tfa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FastRCNNPredictor

def get_model(num_classes: int):
    """
    Faster R-CNN + MobileNetV3 FPN (pretrained) with correct num_classes.
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model

class CosineClassifier(nn.Module):
    """
    Cosine similarity-based classification head (replaces linear cls_score).
    Keeps the same number of output classes as the linear head.
    """
    def __init__(self, in_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return self.scale * (x @ w.t())

def replace_cls_with_cosine(model):
    """
    Replace the linear classification layer with CosineClassifier while keeping the
    total number of classes unchanged. bbox_pred remains unchanged.
    """
    in_dim = model.roi_heads.box_predictor.cls_score.in_features
    out_dim = model.roi_heads.box_predictor.cls_score.out_features
    model.roi_heads.box_predictor.cls_score = CosineClassifier(in_dim, out_dim, scale=20.0)
    return model

def freeze_ft_last_tfa(model):
    """
    TFA ft-last freezing scheme: freeze backbone + RPN + box_head;
    only train roi_heads.box_predictor (cls & bbox).
    """
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.rpn.parameters(): p.requires_grad = False
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = False
    return model
