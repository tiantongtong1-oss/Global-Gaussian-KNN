"""Fixed source prototypes with per-class mean-of-farthest-seven radii."""
import torch
from torch import nn
import torch.nn.functional as F


class PrototypeRadiusBank(nn.Module):
    def __init__(self, prototypes, initialized):
        super().__init__()
        self.register_buffer('prototypes', F.normalize(prototypes.detach().clone(), dim=1))
        self.register_buffer('initialized', initialized.detach().clone().bool())
        self.register_buffer('radii', prototypes.new_zeros(prototypes.size(0)))
        self.register_buffer('counts', torch.zeros(prototypes.size(0), dtype=torch.long,
                                                 device=prototypes.device))
        self.register_buffer('ready', torch.zeros_like(self.initialized))
        self.register_buffer('_farthest', prototypes.new_full((prototypes.size(0), 7),
                                                            -float('inf')), persistent=False)

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        z = F.normalize(torch.nan_to_num(features), dim=1)
        for c in range(self.prototypes.size(0)):
            if not self.initialized[c]:
                continue
            rows = valid & labels.eq(c)
            distances = (z[rows] - self.prototypes[c]).norm(dim=1)
            self.counts[c] += distances.numel()
            self._farthest[c] = torch.cat((self._farthest[c], distances)).topk(7).values

    @torch.no_grad()
    def finalize_source(self):
        self.ready.copy_(self.initialized & (self.counts >= 7))
        self.radii.zero_()
        self.radii[self.ready] = self._farthest[self.ready].mean(dim=1)

    @torch.no_grad()
    def accepts(self, features, labels):
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = (labels >= 0) & (labels < self.prototypes.size(0))
        valid &= torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        safe = labels.clamp(0, self.prototypes.size(0) - 1)
        z = F.normalize(torch.nan_to_num(features), dim=1)
        distances = (z - self.prototypes[safe]).norm(dim=1)
        return valid & self.ready[safe] & (distances < 0.7 * self.radii[safe])
