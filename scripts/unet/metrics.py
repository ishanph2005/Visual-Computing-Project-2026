import numpy as np
import torch


class ConfusionMeter:
    """Accumulates a confusion matrix (rows = ground truth, cols = prediction)."""

    def __init__(self, n_class=3):
        self.n = n_class
        self.mat = torch.zeros(n_class, n_class, dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1).flatten().cpu()
        t = target.flatten().cpu()
        k = (t >= 0) & (t < self.n)
        self.mat += torch.bincount(self.n * t[k] + pred[k], minlength=self.n ** 2).view(self.n, self.n)

    def results(self, names=("background", "negative", "positive")):
        m = self.mat.double().numpy()
        tp = np.diag(m)
        gt = m.sum(1)
        pr = m.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (gt + pr - tp)
            recall = tp / gt
            precision = tp / pr
        out = {"mIoU": float(np.nanmean(iou)),          # all 3 classes, as in LuSeg
               "hazard_mIoU": float(np.nanmean(iou[1:]))}  # negative + positive only
        for i, n in enumerate(names):
            out[f"IoU_{n}"] = float(iou[i])
            out[f"recall_{n}"] = float(recall[i])
            out[f"precision_{n}"] = float(precision[i])
        return out
