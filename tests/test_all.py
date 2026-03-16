"""
tests/test_all.py — Unit tests for each module.

Run from project root:
    python -m pytest tests/ -v
"""

import torch
import numpy as np
import pytest


# ── Model tests ───────────────────────────────────────────────────────────

class TestUNet:
    def test_output_shape(self):
        """Model must output [B, 1, H, W] matching input spatial size."""
        from src.model import UNet
        from src.config import CFG
        model  = UNet()
        dummy  = torch.randn(2, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
        output = model(dummy)
        assert output.shape == (2, 1, CFG.IMG_SIZE, CFG.IMG_SIZE), \
            f"Expected (2,1,{CFG.IMG_SIZE},{CFG.IMG_SIZE}), got {output.shape}"

    def test_output_is_logits(self):
        """Raw output should be logits — model must NOT apply sigmoid internally."""
        from src.model import UNet
        from src.config import CFG
        model  = UNet()
        dummy  = torch.ones(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE) * 5.0
        output = model(dummy)
        assert (output < 0).any(), "Model should output raw logits (with negatives), not probabilities"

    def test_sigmoid_in_range(self):
        """After sigmoid, all values must be in [0, 1]."""
        from src.model import UNet
        from src.config import CFG
        model  = UNet()
        dummy  = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
        probs  = torch.sigmoid(model(dummy))
        assert probs.min() >= 0.0 and probs.max() <= 1.0


# ── Loss tests ────────────────────────────────────────────────────────────

class TestLoss:
    def test_perfect_prediction(self):
        """Loss should be near 0 when prediction matches ground truth exactly."""
        from src.losses import BCEDiceLoss
        criterion = BCEDiceLoss()
        # Perfect prediction: large positive logit → prob ≈ 1 where mask=1
        logits  = torch.full((1, 1, 16, 16), 10.0)
        targets = torch.ones(1, 1, 16, 16)
        loss    = criterion(logits, targets)
        assert loss.item() < 0.1, f"Loss too high for perfect prediction: {loss.item()}"

    def test_all_wrong(self):
        """Loss should be high when prediction is completely wrong."""
        from src.losses import BCEDiceLoss
        criterion = BCEDiceLoss()
        logits  = torch.full((1, 1, 16, 16), -10.0)   # predicts all background
        targets = torch.ones(1, 1, 16, 16)             # all wound
        loss    = criterion(logits, targets)
        assert loss.item() > 0.8, f"Loss too low for wrong prediction: {loss.item()}"

    def test_output_is_scalar(self):
        """Loss must return a scalar tensor."""
        from src.losses import BCEDiceLoss
        criterion = BCEDiceLoss()
        logits  = torch.randn(4, 1, 32, 32)
        targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
        loss    = criterion(logits, targets)
        assert loss.shape == torch.Size([]), f"Loss not scalar: {loss.shape}"


# ── Metrics tests ─────────────────────────────────────────────────────────

class TestMetrics:
    def _make_tensors(self, tp, tn, fp, fn):
        """Helper: create y_true and y_prob tensors with exact TP/TN/FP/FN counts."""
        y_true = torch.tensor([1]*tp + [0]*tn + [0]*fp + [1]*fn, dtype=torch.float32)
        y_prob = torch.tensor(
            [0.9]*tp +    # true positives → high prob
            [0.1]*tn +    # true negatives → low prob
            [0.9]*fp +    # false positives → high prob (wrong)
            [0.1]*fn,     # false negatives → low prob (wrong)
            dtype=torch.float32
        )
        return y_true.unsqueeze(0).unsqueeze(0), y_prob.unsqueeze(0).unsqueeze(0)

    def test_perfect_metrics(self):
        """All metrics should be 1.0 for perfect prediction."""
        from src.metrics import compute_metrics
        y_true = torch.ones(1, 1, 10, 10)
        y_prob = torch.full((1, 1, 10, 10), 0.99)
        m = compute_metrics(y_true, y_prob)
        assert m["IoU"]    >= 0.99
        assert m["Dice"]   >= 0.99
        assert m["Recall"] >= 0.99

    def test_all_keys_present(self):
        """Must return all expected metric keys."""
        from src.metrics import compute_metrics
        y_true = torch.randint(0, 2, (1, 1, 32, 32)).float()
        y_prob = torch.rand(1, 1, 32, 32)
        m = compute_metrics(y_true, y_prob)
        required = ["TP","TN","FP","FN","Accuracy","Precision","Recall",
                    "Specificity","F1","Kappa","MCC","ROC_AUC","PR_AUC",
                    "FPR","FNR","IoU","Dice"]
        for key in required:
            assert key in m, f"Missing metric key: {key}"

    def test_iou_range(self):
        """IoU must always be in [0, 1]."""
        from src.metrics import compute_metrics
        y_true = torch.randint(0, 2, (2, 1, 64, 64)).float()
        y_prob = torch.rand(2, 1, 64, 64)
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["IoU"] <= 1.0

    def test_recall_over_precision_for_wounds(self):
        """
        In clinical setting: missing a wound (FN) is worse than false alarm (FP).
        Recall should be prioritised — verify it's tracked separately from Precision.
        """
        from src.metrics import compute_metrics
        # High FN scenario (many missed wounds)
        y_true = torch.ones(1, 1, 10, 10)
        y_prob = torch.full((1, 1, 10, 10), 0.3)   # all below threshold → all FN
        m = compute_metrics(y_true, y_prob)
        assert m["Recall"] < 0.1, "Recall should be low when all wounds are missed"
        assert m["FNR"]    > 0.9, "FNR should be high when all wounds are missed"
