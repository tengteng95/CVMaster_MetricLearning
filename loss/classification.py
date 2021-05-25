import megengine
import numpy as np
from megengine import functional as F
from megengine import module as M


class CELoss(M.Module):
    def __init__(self, feat_dim, nr_class):
        super().__init__()
        w_shape = (nr_class, feat_dim)
        self.weight = megengine.Parameter(np.zeros(w_shape, dtype=np.float32))
        megengine.module.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        logits = F.linear(x, self.weight)
        loss = F.loss.cross_entropy(logits, labels)
        return loss


class CosfaceLoss(M.Module):
    def __init__(self, feat_dim, nr_class, scale=30, margin=0.25):
        super().__init__()
        w_shape = (nr_class, feat_dim)
        self.weight = megengine.Parameter(np.zeros(w_shape, dtype=np.float32))
        megengine.module.init.xavier_uniform_(self.weight)

        self.scale = scale
        self.margin = margin

    def forward(self, x, labels):
        """
            Waitng for Your Implementation
        """
        pass


class CircleLoss(M.Module):
    def __init__(self, feat_dim, nr_class, scale=96, margin=0.25):
        super().__init__()
        w_shape = (nr_class, feat_dim)
        self.weight = megengine.Parameter(np.zeros(w_shape, dtype=np.float32))
        megengine.module.init.xavier_uniform_(self.weight)

        self.scale = scale
        self.margin = margin

    def forward(self, x, labels):
        """
            Waitng for Your Implementation
        """
        pass
