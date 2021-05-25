from megengine import module as M


class PairCosface(M.Module):
    def __init__(self, scale, margin):
        self.scale = scale
        self.margin = margin

    def forward(self, feats, labels):
        """
            Waitng for Your Implementation.
        """
        pass


class PairCircle(M.Module):
    def __init__(self, scale, margin):
        self.scale = scale
        self.margin = margin

    def forward(self, feats, labels):
        """
            Waiting for Your Implementation.
        """
        pass
