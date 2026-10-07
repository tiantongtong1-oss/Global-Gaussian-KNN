"""Source-labeled class Gaussian reliability with global variance EMA.

Each class has a source-only center mu_c. All classes share one isotropic within-class
variance estimated from correctly labeled source features. The scalar variance is
updated by a momentum schedule that increases across source refreshes.

A pseudo-label candidate must fall inside the chi-square confidence region of its
predicted class. Its k nearest target-memory neighbors are then checked for:
1) same predicted class,
2) sufficient teacher confidence,
3) membership in the candidate class Gaussian region.

Neighborhood density is estimated with a Gaussian distance kernel. Sparse
neighborhoods receive an additional configurable penalty in the final reliability
score. Target ground-truth labels are never used.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class GlobalGaussianKNNReliabilityBank(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim,
        k=20,
        distribution_mass=0.95,
        sigma_momentum=0.70,
        bandwidth_multiplier=1.0,
        score_threshold=0.5,
        density_threshold=0.5,
        query_chunk_size=128,
        variance_floor=1e-4,
        eps=1e-8,
        global_momentum_end=0.95,
        global_momentum_ramp_refreshes=30,
        sparse_penalty=0.5,
    ):
        super().__init__()
        if num_classes < 1 or feature_dim < 1 or k < 1 or query_chunk_size < 1:
            raise ValueError("class count, feature dimension, k and chunk size must be positive")
        if not 0 < distribution_mass < 1:
            raise ValueError("distribution_mass must be in (0, 1)")
        if not 0 <= sigma_momentum < 1:
            raise ValueError("sigma_momentum must be in [0, 1)")
        if not 0 <= global_momentum_end < 1:
            raise ValueError("global_momentum_end must be in [0, 1)")
        if global_momentum_end < sigma_momentum:
            raise ValueError("global_momentum_end must be >= sigma_momentum")
        if global_momentum_ramp_refreshes < 1:
            raise ValueError("global_momentum_ramp_refreshes must be positive")
        if not 0 < sparse_penalty <= 1:
            raise ValueError("sparse_penalty must be in (0, 1]")
        if not 0 < score_threshold <= 1 or not 0 < density_threshold <= 1:
            raise ValueError("score and density thresholds must be in (0, 1]")
        for name, value in (
            ("variance_floor", variance_floor),
            ("bandwidth_multiplier", bandwidth_multiplier),
            ("eps", eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.k = int(k)
        self.distribution_mass = float(distribution_mass)
        self.variance_floor = float(variance_floor)
        self.sigma_momentum = float(sigma_momentum)
        self.global_momentum_end = float(global_momentum_end)
        self.global_momentum_ramp_refreshes = int(global_momentum_ramp_refreshes)
        self.bandwidth_multiplier = float(bandwidth_multiplier)
        self.score_threshold = float(score_threshold)
        self.density_threshold = float(density_threshold)
        self.sparse_penalty = float(sparse_penalty)
        self.query_chunk_size = int(query_chunk_size)
        self.eps = float(eps)

        self.register_buffer("class_means", torch.zeros(num_classes, feature_dim))
        # Kept for checkpoint diagnostics/compatibility. Every ready class receives
        # the same global isotropic variance after each refresh.
        self.register_buffer("class_variances", torch.ones(num_classes, feature_dim))
        self.register_buffer(
            "distribution_initialized", torch.zeros(num_classes, dtype=torch.bool)
        )
        self.register_buffer(
            "class_source_counts", torch.zeros(num_classes, dtype=torch.long)
        )

        shape = torch.tensor(feature_dim / 2.0, dtype=torch.float64)
        low, high = 0.0, float(max(feature_dim, 1))

        def cdf(value):
            return torch.special.gammainc(
                shape, shape.new_tensor(value / 2.0)
            ).item()

        while cdf(high) < distribution_mass:
            high *= 2.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if cdf(mid) < distribution_mass:
                low = mid
            else:
                high = mid
        self.register_buffer(
            "mahalanobis_threshold", torch.tensor((low + high) / 2.0)
        )

        self.register_buffer("global_var", torch.zeros(()))
        self.register_buffer("current_observed_global_var", torch.zeros(()))
        self.register_buffer("current_global_momentum", torch.tensor(self.sigma_momentum))
        self.register_buffer("sigma_initialized", torch.tensor(False))
        self.register_buffer("source_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("refresh_count", torch.zeros((), dtype=torch.long))

        self.register_buffer(
            "_source_sum",
            torch.zeros(num_classes, feature_dim, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_source_sq_sum",
            torch.zeros(num_classes, feature_dim, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_source_count",
            torch.zeros(num_classes, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "memory_features", torch.empty(0, feature_dim), persistent=False
        )
        self.register_buffer(
            "memory_labels", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "memory_confidences", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "memory_ids", torch.empty(0, dtype=torch.long), persistent=False
        )

    @property
    def sigma(self):
        return self.global_var.clamp_min(self.eps).sqrt()

    def momentum_for_refresh(self):
        if self.global_momentum_ramp_refreshes <= 1:
            return self.global_momentum_end
        progress = min(
            float(self.refresh_count.item())
            / float(self.global_momentum_ramp_refreshes - 1),
            1.0,
        )
        return self.sigma_momentum + (
            self.global_momentum_end - self.sigma_momentum
        ) * progress

    @torch.no_grad()
    def begin_source_refresh(self):
        self._source_sum.zero_()
        self._source_sq_sum.zero_()
        self._source_count.zero_()
        self.clear_target_memory()

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (
            features.norm(dim=1) > self.eps
        )
        valid &= (labels >= 0) & (labels < self.num_classes)
        features, labels = features[valid], labels[valid]
        if features.size(0) == 0:
            return

        z = F.normalize(features, dim=1).double()
        self._source_sum.index_add_(0, labels, z)
        self._source_sq_sum.index_add_(0, labels, z.square())
        self._source_count.add_(
            torch.bincount(labels, minlength=self.num_classes)
        )

    @torch.no_grad()
    def finalize_source(self):
        """Fit class means and update one pooled within-class variance by EMA."""
        self.class_source_counts.copy_(self._source_count)
        self.source_count.copy_(self._source_count.sum())

        ready = self._source_count >= 2
        counts = self._source_count.clamp_min(1).double().unsqueeze(1)
        means = self._source_sum / counts

        # Per-coordinate within-class MLE variance only for estimating the pooled
        # source variance. The final Gaussian uses one shared scalar variance.
        raw_class_variances = (
            self._source_sq_sum / counts - means.square()
        ).clamp_min(0)
        ready &= torch.isfinite(means).all(dim=1)
        ready &= torch.isfinite(raw_class_variances).all(dim=1)

        self.distribution_initialized.copy_(ready)
        if not ready.any():
            self.sigma_initialized.fill_(False)
            self.global_var.zero_()
            self.current_observed_global_var.zero_()
            return

        for c in ready.nonzero(as_tuple=False).flatten().tolist():
            self.class_means[c].copy_(means[c].to(self.class_means))

        # Isotropic Gaussian MLE variance:
        # sum_{c,d} N_c * var[c,d] / (sum_c N_c * D)
        # This is the pooled mean squared within-class residual per coordinate.
        weights = self._source_count[ready].double().unsqueeze(1)
        pooled_var = (
            (raw_class_variances[ready] * weights).sum()
            / (weights.sum() * self.feature_dim)
        )
        pooled_var = pooled_var.clamp_min(self.variance_floor)
        observed = pooled_var.to(self.global_var)
        self.current_observed_global_var.copy_(observed)

        beta = float(self.momentum_for_refresh())
        self.current_global_momentum.fill_(beta)
        if not self.sigma_initialized.item():
            self.global_var.copy_(observed)
            self.sigma_initialized.fill_(True)
        else:
            self.global_var.mul_(beta).add_(observed, alpha=1.0 - beta)

        self.global_var.clamp_(min=self.variance_floor)
        self.class_variances[ready].fill_(float(self.global_var.item()))
        self.refresh_count.add_(1)

    @torch.no_grad()
    def distribution_distance(self, features, labels):
        """Squared Mahalanobis distance under N(mu_c, global_var * I)."""
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid_labels = (labels >= 0) & (labels < self.num_classes)
        safe_labels = labels.clamp(0, self.num_classes - 1)
        valid = valid_labels & self.distribution_initialized[safe_labels]
        valid &= torch.isfinite(features).all(dim=-1)
        valid &= features.norm(dim=-1) > self.eps
        valid &= self.sigma_initialized

        z = F.normalize(torch.nan_to_num(features), dim=-1)
        sq = (z - self.class_means[safe_labels]).square().sum(dim=-1)
        distance = sq / self.global_var.clamp_min(self.variance_floor)
        return distance.masked_fill(~valid, float("inf"))

    @torch.no_grad()
    def in_distribution(self, features, labels):
        return (
            self.distribution_distance(features, labels)
            <= self.mahalanobis_threshold
        )

    @torch.no_grad()
    def clear_target_memory(self):
        self.memory_features = self.class_means.new_empty(
            (0, self.feature_dim)
        )
        self.memory_labels = self.distribution_initialized.new_empty(
            (0,), dtype=torch.long
        )
        self.memory_confidences = self.global_var.new_empty((0,))
        self.memory_ids = self.memory_labels.clone()

    @torch.no_grad()
    def set_target_memory(
        self, features, labels, confidences, sample_ids
    ):
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                "target memory must have shape [N, feature_dim]"
            )
        n = features.size(0)
        if any(
            t.shape != (n,)
            for t in (labels, confidences, sample_ids)
        ):
            raise ValueError("target memory vectors must have shape [N]")

        features = features.detach().to(self.class_means)
        labels = labels.detach().to(
            device=features.device, dtype=torch.long
        )
        confidences = confidences.detach().to(self.global_var)
        sample_ids = sample_ids.detach().to(
            device=features.device, dtype=torch.long
        )
        if torch.unique(sample_ids).numel() != n:
            raise ValueError("target memory sample IDs must be unique")

        valid = torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > self.eps
        valid &= torch.isfinite(confidences)
        valid &= (confidences >= 0) & (confidences <= 1)
        valid &= (labels >= 0) & (labels < self.num_classes)

        self.memory_features = F.normalize(
            features[valid], dim=1
        ).contiguous()
        self.memory_labels = labels[valid].clone()
        self.memory_confidences = confidences[valid].clone()
        self.memory_ids = sample_ids[valid].clone()

    @torch.no_grad()
    def gate(
        self,
        features,
        pseudo_targets,
        sample_ids,
        candidate_mask,
        thresholds,
    ):
        """Score pseudo-label reliability using Gaussian membership + kNN support.

        weighted_support = sum_j(w_j * support_j) / sum_j(w_j)
        density = mean_j(w_j)
        density_factor = density if dense else density * sparse_penalty
        score = query_in_region * weighted_support * density_factor

        support_j requires same pseudo class, confidence, and membership in the
        candidate class source Gaussian.
        """
        features = features.detach().to(self.class_means)
        n = features.size(0)
        if features.shape != (n, self.feature_dim):
            raise ValueError(
                "query features must have shape [B, feature_dim]"
            )

        device = features.device
        pseudo_targets = pseudo_targets.detach().to(
            device=device, dtype=torch.long
        )
        sample_ids = sample_ids.detach().to(
            device=device, dtype=torch.long
        )
        candidate_mask = candidate_mask.detach().to(
            device=device, dtype=torch.bool
        )
        if any(
            t.shape != (n,)
            for t in (pseudo_targets, sample_ids, candidate_mask)
        ):
            raise ValueError("query vectors must have shape [B]")

        thresholds = thresholds.detach().to(self.global_var)
        if (
            thresholds.shape != (self.num_classes,)
            or not torch.isfinite(thresholds).all()
        ):
            raise ValueError(
                "thresholds must be a finite vector with one value per class"
            )

        result = {
            "score": features.new_zeros(n),
            "mahalanobis_sq": features.new_full((n,), float("inf")),
            "in_distribution": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "density": features.new_zeros(n),
            "weighted_support": features.new_zeros(n),
            "support_fraction": features.new_zeros(n),
            "checked_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "dense_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "sparse_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "pass_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
        }

        if (
            not self.sigma_initialized.item()
            or self.memory_features.size(0) < self.k
        ):
            return result
        if (
            not torch.isfinite(self.global_var)
            or self.global_var.item() <= 0
        ):
            return result

        valid = candidate_mask & torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > self.eps
        valid &= (pseudo_targets >= 0) & (
            pseudo_targets < self.num_classes
        )
        rows = valid.nonzero(as_tuple=False).flatten()
        rows = rows[
            self.distribution_initialized[pseudo_targets[rows]]
        ]

        h_sq = (
            self.bandwidth_multiplier ** 2 * self.global_var
        ).clamp_min(self.eps)

        for start in range(0, rows.numel(), self.query_chunk_size):
            idx = rows[start : start + self.query_chunk_size]
            z = F.normalize(features[idx], dim=1)
            labels = pseudo_targets[idx]

            distances = (
                2.0 - 2.0 * z.mm(self.memory_features.t())
            ).clamp_min(0)
            distances.masked_fill_(
                sample_ids[idx, None].eq(
                    self.memory_ids[None, :]
                ),
                float("inf"),
            )
            nn_dist, nn_idx = distances.topk(
                self.k, dim=1, largest=False
            )
            enough_neighbors = torch.isfinite(nn_dist).all(dim=1)

            nn_labels = self.memory_labels[nn_idx]
            nn_features = self.memory_features[nn_idx]

            # Every neighbor is tested against the candidate pseudo-label class
            # Gaussian, not its own predicted-class Gaussian.
            candidate_labels = labels[:, None].expand_as(nn_labels)
            neighbor_in_region = self.in_distribution(
                nn_features, candidate_labels
            )
            neighbor_confident = (
                self.memory_confidences[nn_idx]
                >= thresholds[nn_labels]
            )
            support = (
                nn_labels.eq(candidate_labels)
                & neighbor_confident
                & neighbor_in_region
            )

            mahalanobis_sq = self.distribution_distance(
                z, labels
            )
            self_in_region = (
                mahalanobis_sq <= self.mahalanobis_threshold
            )

            weights = torch.exp(
                -nn_dist / (2.0 * h_sq)
            )
            finite_weights = torch.where(
                torch.isfinite(weights),
                weights,
                torch.zeros_like(weights),
            )
            density = finite_weights.mean(dim=1)
            weighted_support = (
                (finite_weights * support.float()).sum(dim=1)
                / finite_weights.sum(dim=1).clamp_min(self.eps)
            )
            dense = density >= self.density_threshold
            density_factor = torch.where(
                dense,
                density,
                density * self.sparse_penalty,
            )
            score = (
                self_in_region.float()
                * weighted_support
                * density_factor
                * enough_neighbors.float()
            )

            result["mahalanobis_sq"][idx] = mahalanobis_sq
            result["in_distribution"][idx] = self_in_region
            result["score"][idx] = score
            result["density"][idx] = (
                density * enough_neighbors.float()
            )
            result["weighted_support"][idx] = (
                weighted_support * enough_neighbors.float()
            )
            result["support_fraction"][idx] = (
                support.float().mean(dim=1)
                * enough_neighbors.float()
            )
            result["checked_mask"][idx] = enough_neighbors
            result["dense_mask"][idx] = (
                enough_neighbors & dense
            )
            result["sparse_mask"][idx] = (
                enough_neighbors & ~dense
            )

        result["pass_mask"] = (
            result["checked_mask"]
            & result["in_distribution"]
            & (result["score"] >= self.score_threshold)
        )
        return result


def combine_rescue_masks(
    confidence_mask,
    agreement_mask,
    reliability_mask,
    enabled,
):
    confidence_mask = confidence_mask.bool() & agreement_mask.bool()
    rescue_mask = torch.zeros_like(confidence_mask)
    if enabled:
        rescue_mask = (
            agreement_mask.bool()
            & reliability_mask.bool()
            & ~confidence_mask
        )
    return confidence_mask | rescue_mask, rescue_mask
