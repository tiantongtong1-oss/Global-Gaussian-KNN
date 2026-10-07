"""Class-conditional diagonal Gaussian membership and target kNN OR rescue.

Source means and per-coordinate variances define each class distribution.
Query and neighbor membership use the same source-class Mahalanobis ellipsoid.
Gaussian model coverage is nominal; the final support score is not a posterior.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class KNNReliabilityBank(nn.Module):
    def __init__(self, num_classes, feature_dim, k=20, distribution_mass=0.95,
                 sigma_momentum=0.9, bandwidth_multiplier=1.0,
                 score_threshold=0.5, density_threshold=0.5,
                 query_chunk_size=128, variance_floor=1e-4, eps=1e-8):
        super().__init__()
        if num_classes < 1 or feature_dim < 1 or k < 1 or query_chunk_size < 1:
            raise ValueError('class count, feature dimension, k and chunk size must be positive')
        for name, value in [('variance_floor', variance_floor),
                            ('bandwidth_multiplier', bandwidth_multiplier), ('eps', eps)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError('%s must be finite and positive' % name)
        if not 0 <= sigma_momentum < 1:
            raise ValueError('sigma_momentum must be in [0, 1)')
        if not 0 < score_threshold <= 1 or not 0 < density_threshold <= 1:
            raise ValueError('score and density thresholds must be in (0, 1]')
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.k = int(k)
        if not 0 < distribution_mass < 1:
            raise ValueError('distribution_mass must be in (0, 1)')
        self.distribution_mass = float(distribution_mass)
        self.variance_floor = float(variance_floor)
        self.sigma_momentum = float(sigma_momentum)
        self.bandwidth_multiplier = float(bandwidth_multiplier)
        self.score_threshold = float(score_threshold)
        self.density_threshold = float(density_threshold)
        self.query_chunk_size = int(query_chunk_size)
        self.eps = float(eps)

        self.register_buffer('class_means', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_variances', torch.ones(num_classes, feature_dim))
        self.register_buffer('distribution_initialized', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('class_source_counts', torch.zeros(num_classes, dtype=torch.long))
        # A chi-square quantile gives a nominal Gaussian probability-content region.
        # Compute on CPU in float64 once, without adding a SciPy dependency.
        shape = torch.tensor(feature_dim / 2.0, dtype=torch.float64)
        low, high = 0.0, float(max(feature_dim, 1))
        def cdf(value):
            return torch.special.gammainc(shape, shape.new_tensor(value / 2.0)).item()
        while cdf(high) < distribution_mass:
            high *= 2.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if cdf(mid) < distribution_mass:
                low = mid
            else:
                high = mid
        self.register_buffer('mahalanobis_threshold', torch.tensor((low + high) / 2.0))
        # Pooled trace of class variances controls only the local distance kernel.
        self.register_buffer('global_var', torch.zeros(()))
        self.register_buffer('sigma_initialized', torch.tensor(False))
        self.register_buffer('source_count', torch.zeros((), dtype=torch.long))
        # Source accumulators and target memory are rebuilt, never restored as stale features.
        self.register_buffer('_source_sum', torch.zeros(num_classes, feature_dim, dtype=torch.float64), persistent=False)
        self.register_buffer('_source_sq_sum', torch.zeros(num_classes, feature_dim, dtype=torch.float64), persistent=False)
        self.register_buffer('_source_count', torch.zeros(num_classes, dtype=torch.long), persistent=False)
        self.register_buffer('memory_features', torch.empty(0, feature_dim), persistent=False)
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('memory_confidences', torch.empty(0), persistent=False)
        self.register_buffer('memory_ids', torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def sigma(self):
        return self.global_var.clamp_min(self.eps).sqrt()

    @torch.no_grad()
    def begin_source_refresh(self):
        """Reset source accumulators independently of the training prototypes."""
        self._source_sum.zero_()
        self._source_sq_sum.zero_()
        self._source_count.zero_()
        # The refreshed centers must not be scored until the new scan is finalized.
        self.clear_target_memory()

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > self.eps)
        valid &= (labels >= 0) & (labels < self.num_classes)
        features, labels = features[valid], labels[valid]
        if features.size(0) == 0:
            return
        z = F.normalize(features, dim=1).double()
        self._source_sum.index_add_(0, labels, z)
        self._source_sq_sum.index_add_(0, labels, z.square())
        self._source_count.add_(torch.bincount(labels, minlength=self.num_classes))

    @torch.no_grad()
    def finalize_source(self):
        """Fit each Gaussian; EMA uses moment matching, including mean drift."""
        self.class_source_counts.copy_(self._source_count)
        self.source_count.copy_(self._source_count.sum())
        ready = self._source_count >= 2
        counts = self._source_count.clamp_min(1).double().unsqueeze(1)
        means = self._source_sum / counts
        # Gaussian MLE diagonal variance, accumulated in float64 for stability.
        variances = (self._source_sq_sum / counts - means.square()).clamp_min(0)
        ready &= torch.isfinite(means).all(dim=1) & torch.isfinite(variances).all(dim=1)
        for c in ready.nonzero(as_tuple=False).flatten().tolist():
            mean = means[c].to(self.class_means)
            var = variances[c].to(self.class_variances)
            if self.distribution_initialized[c]:
                beta = self.sigma_momentum
                # Variance of a mixture: within-component variance + mean shift.
                delta = self.class_means[c] - mean
                var = (beta * self.class_variances[c] + (1 - beta) * var
                       + beta * (1 - beta) * delta.square())
                mean = beta * self.class_means[c] + (1 - beta) * mean
            self.class_means[c].copy_(mean)
            self.class_variances[c].copy_(var.clamp_min(self.variance_floor))
        # A class missing in the current scan cannot rescue using stale statistics.
        self.distribution_initialized.copy_(ready)
        self.sigma_initialized.copy_(ready.any())
        if ready.any():
            weights = self._source_count[ready].to(self.global_var)
            self.global_var.copy_((self.class_variances[ready].sum(dim=1) * weights).sum()
                                  / weights.sum())
        else:
            self.global_var.zero_()
    @torch.no_grad()
    def distribution_distance(self, features, labels):
        """Squared diagonal Mahalanobis distance for normalized features.

        Accepts [B,D] or [B,K,D] with labels [B] or [B,K]. Missing classes
        and invalid vectors return infinity, so all callers fail closed.
        """
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid_labels = (labels >= 0) & (labels < self.num_classes)
        safe_labels = labels.clamp(0, self.num_classes - 1)
        valid = valid_labels & self.distribution_initialized[safe_labels]
        valid &= torch.isfinite(features).all(dim=-1) & (features.norm(dim=-1) > self.eps)
        z = F.normalize(torch.nan_to_num(features), dim=-1)
        distance = ((z - self.class_means[safe_labels]).square()
                    / self.class_variances[safe_labels].clamp_min(self.variance_floor)).sum(dim=-1)
        return distance.masked_fill(~valid, float('inf'))

    @torch.no_grad()
    def in_distribution(self, features, labels):
        return self.distribution_distance(features, labels) <= self.mahalanobis_threshold

    @torch.no_grad()
    def clear_target_memory(self):
        self.memory_features = self.class_means.new_empty((0, self.feature_dim))
        self.memory_labels = self.distribution_initialized.new_empty((0,), dtype=torch.long)
        self.memory_confidences = self.global_var.new_empty((0,))
        self.memory_ids = self.memory_labels.clone()

    @torch.no_grad()
    def set_target_memory(self, features, labels, confidences, sample_ids):
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError('target memory must have shape [N, feature_dim]')
        n = features.size(0)
        if any(t.shape != (n,) for t in (labels, confidences, sample_ids)):
            raise ValueError('target memory vectors must have shape [N]')
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        confidences = confidences.detach().to(self.global_var)
        sample_ids = sample_ids.detach().to(device=features.device, dtype=torch.long)
        if torch.unique(sample_ids).numel() != n:
            raise ValueError('target memory sample IDs must be unique')
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > self.eps)
        valid &= torch.isfinite(confidences) & (confidences >= 0) & (confidences <= 1)
        valid &= (labels >= 0) & (labels < self.num_classes)
        self.memory_features = F.normalize(features[valid], dim=1).contiguous()
        self.memory_labels = labels[valid].clone()
        self.memory_confidences = confidences[valid].clone()
        self.memory_ids = sample_ids[valid].clone()

    @torch.no_grad()
    def gate(self, features, pseudo_targets, sample_ids, candidate_mask, thresholds):
        """Score agreed candidates. Missing statistics/neighbors cannot rescue.

        density = mean(exp(-distance_sq / (2 * h^2)))
        score = self_in_distribution * mean(kernel * same_class * confident * in_distribution)
        h = bandwidth_multiplier * sigma; density is a support index, not a PDF.
        """
        features = features.detach().to(self.class_means)
        n = features.size(0)
        if features.shape != (n, self.feature_dim):
            raise ValueError('query features must have shape [B, feature_dim]')
        device = features.device
        pseudo_targets = pseudo_targets.detach().to(device=device, dtype=torch.long)
        sample_ids = sample_ids.detach().to(device=device, dtype=torch.long)
        candidate_mask = candidate_mask.detach().to(device=device, dtype=torch.bool)
        if any(t.shape != (n,) for t in (pseudo_targets, sample_ids, candidate_mask)):
            raise ValueError('query vectors must have shape [B]')
        thresholds = thresholds.detach().to(self.global_var)
        if thresholds.shape != (self.num_classes,) or not torch.isfinite(thresholds).all():
            raise ValueError('thresholds must be a finite vector with one value per class')
        result = {
            'score': features.new_zeros(n),
            'mahalanobis_sq': features.new_full((n,), float('inf')),
            'in_distribution': torch.zeros(n, dtype=torch.bool, device=device),
            'density': features.new_zeros(n),
            'support_fraction': features.new_zeros(n),
            'checked_mask': torch.zeros(n, dtype=torch.bool, device=device),
            'dense_mask': torch.zeros(n, dtype=torch.bool, device=device),
            'pass_mask': torch.zeros(n, dtype=torch.bool, device=device),
        }
        if not self.sigma_initialized.item() or self.memory_features.size(0) < self.k:
            return result
        if not torch.isfinite(self.global_var) or self.global_var.item() <= 0:
            return result
        valid = candidate_mask & torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > self.eps
        valid &= (pseudo_targets >= 0) & (pseudo_targets < self.num_classes)
        rows = valid.nonzero(as_tuple=False).flatten()
        rows = rows[self.distribution_initialized[pseudo_targets[rows]]]
        h_sq = (self.bandwidth_multiplier ** 2 * self.global_var).clamp_min(self.eps)
        for start in range(0, rows.numel(), self.query_chunk_size):
            idx = rows[start:start + self.query_chunk_size]
            z = F.normalize(features[idx], dim=1)
            labels = pseudo_targets[idx]
            # Allocate [chunk, N], never the full [N, N] target distance matrix.
            distances = (2.0 - 2.0 * z.mm(self.memory_features.t())).clamp_min(0)
            distances.masked_fill_(sample_ids[idx, None].eq(self.memory_ids[None, :]), float('inf'))
            nn_dist, nn_idx = distances.topk(self.k, dim=1, largest=False)
            enough_neighbors = torch.isfinite(nn_dist).all(dim=1)
            nn_labels = self.memory_labels[nn_idx]
            nn_features = self.memory_features[nn_idx]
            neighbor_in_region = self.in_distribution(
                nn_features, labels[:, None].expand_as(nn_labels)
            )
            neighbor_confident = self.memory_confidences[nn_idx] >= thresholds[nn_labels]
            support = nn_labels.eq(labels[:, None]) & neighbor_confident & neighbor_in_region
            mahalanobis_sq = self.distribution_distance(z, labels)
            self_in_region = mahalanobis_sq <= self.mahalanobis_threshold
            result['mahalanobis_sq'][idx] = mahalanobis_sq
            result['in_distribution'][idx] = self_in_region
            weights = torch.exp(-nn_dist / (2.0 * h_sq))
            # Dividing by sum(weights) would remove the sparse-neighborhood penalty.
            score = (weights * support.float()).mean(dim=1) * self_in_region.float()
            density = weights.mean(dim=1)
            result['score'][idx] = score * enough_neighbors.float()
            result['density'][idx] = density * enough_neighbors.float()
            result['support_fraction'][idx] = support.float().mean(dim=1) * enough_neighbors.float()
            result['checked_mask'][idx] = enough_neighbors
        result['dense_mask'] = result['checked_mask'] & (result['density'] >= self.density_threshold)
        result['pass_mask'] = result['checked_mask'] & (result['score'] >= self.score_threshold)
        return result


def combine_rescue_masks(confidence_mask, agreement_mask, reliability_mask, enabled):
    """Keep v3's OR semantics and hard 0/1 masks used by CAST's affinity loss."""
    confidence_mask = confidence_mask.bool() & agreement_mask.bool()
    rescue_mask = torch.zeros_like(confidence_mask)
    if enabled:
        rescue_mask = agreement_mask.bool() & reliability_mask.bool() & ~confidence_mask
    return confidence_mask | rescue_mask, rescue_mask

