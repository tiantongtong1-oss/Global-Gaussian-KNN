import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureHook:
    """Capture the exact feature tensor produced by a module during forward."""

    def __init__(self, module):
        self.output = None
        self._handle = module.register_forward_hook(self._save_output)

    def _save_output(self, module, inputs, output):
        del module, inputs
        self.output = output

    def close(self):
        self._handle.remove()
        self.output = None


class PrototypeBank(nn.Module):
    # 用于目标域一致性训练的源锚定EMA类原型
    # 源原型由带标签的RAF-DB特征一次性计算得到，并作为固定的语义锚点保持不变。目标原型则从可靠的双视角EMA教师特征中按类别逐一更新。损
    # 失函数所使用的原型是固定源锚点与自适应目标EMA原型的归一化混合。

    def __init__(self, num_classes, feature_dim, momentum=0.99,
                 source_anchor=0.5):
        super().__init__()
        if not (0.0 <= momentum < 1.0):
            raise ValueError('prototype momentum must be in [0, 1)')
        if not (0.0 <= source_anchor <= 1.0):
            raise ValueError('source_anchor must be in [0, 1]')

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.source_anchor = float(source_anchor)

        self.register_buffer(
            'source_feature_sums',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'source_counts',
            torch.zeros(self.num_classes, dtype=torch.long),
        )
        self.register_buffer(
            'source_prototypes',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'source_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool),
        )
        self.register_buffer(
            'target_prototypes',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'target_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool),
        )
        self.register_buffer(
            'target_counts',
            torch.zeros(self.num_classes, dtype=torch.long),
        )

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()
        for c in range(self.num_classes):
            class_mask = labels.eq(c)
            if class_mask.any():
                self.source_feature_sums[c].add_(features[class_mask].sum(dim=0))
                self.source_counts[c].add_(int(class_mask.sum().item()))

    @torch.no_grad()
    def finalize_source(self):
        for c in range(self.num_classes):
            if self.source_counts[c].item() > 0:
                proto = self.source_feature_sums[c]
                self.source_prototypes[c].copy_(
                    F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
                )
                self.source_initialized[c] = True

    @torch.no_grad()

    # 目标原型（target prototype）的动量更新（momentum update）函数，
    # 作用是：用当前 batch中“可靠样本”的类别均值，以动量方式去更新每个类别的目标原型（类中心）
    def update_target(self, features, labels, reliable_mask):
        reliable_mask = reliable_mask.bool()
        if not reliable_mask.any():
            return

        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()

        for c in range(self.num_classes):
            class_mask = reliable_mask & labels.eq(c)
            if not class_mask.any():
                continue

            batch_proto = F.normalize(
                features[class_mask].mean(dim=0, keepdim=True), dim=1
            ).squeeze(0)

            if self.target_initialized[c]:
                updated = (
                    self.target_prototypes[c] * self.momentum
                    + batch_proto * (1.0 - self.momentum)
                )
                self.target_prototypes[c].copy_(
                    F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)
                )
            else:
                self.target_prototypes[c].copy_(batch_proto)
                self.target_initialized[c] = True

            self.target_counts[c].add_(int(class_mask.sum().item()))

    @torch.no_grad()
    def blended_prototypes(self):
        prototypes = self.source_prototypes.clone()
        for c in range(self.num_classes):
            source_ready = bool(self.source_initialized[c].item())
            target_ready = bool(self.target_initialized[c].item())

            if source_ready and target_ready:
                blended = (
                    self.source_anchor * self.source_prototypes[c]
                    + (1.0 - self.source_anchor) * self.target_prototypes[c]
                )
                prototypes[c] = F.normalize(
                    blended.unsqueeze(0), dim=1
                ).squeeze(0)
            elif target_ready:
                prototypes[c] = self.target_prototypes[c]

        return F.normalize(prototypes, dim=1)

    # 类平衡的原型对比损失 在可靠的目标域样本上，拉近特征与“正确类原型”的距离，同时推开与其他类原型的距离
    def consistency_loss(self, features, labels, reliable_mask,
                         temperature=0.2):
        reliable_mask = reliable_mask.bool()
        if not reliable_mask.any():
            # 没有可靠样本，返回可求导的 0
            return features.sum() * 0.0
        if temperature <= 0.0:
            raise ValueError('prototype temperature must be positive')
            # 取可靠样本特征并 L2 归一化，点积即余弦相似度
        selected_features = F.normalize(features[reliable_mask], dim=1)
        selected_labels = labels[reliable_mask].long()
        # 混合原型（源锚点 + 目标 EMA），detach 不参与梯度
        prototypes = self.blended_prototypes().detach()
        # 特征与各类原型的相似度   → 分类 logits
        logits = selected_features.mm(prototypes.t()) / float(temperature)
        # 每个样本的交叉熵损失
        per_sample = F.cross_entropy(logits, selected_labels, reduction='none')
        # 类平衡：每个类先求平均，再对类求平均，避免多数类主导
        per_class = []
        for c in selected_labels.unique(sorted=True):
            class_mask = selected_labels.eq(c)
            per_class.append(per_sample[class_mask].mean())
        return torch.stack(per_class).mean()

    @torch.no_grad()
    def agreement_stats(self, features, labels, reliable_mask):
        reliable_mask = reliable_mask.bool()
        if not reliable_mask.any():
            return 0, 0, 0.0

        selected_features = F.normalize(features.detach()[reliable_mask], dim=1)
        selected_labels = labels.detach()[reliable_mask].long()
        prototypes = self.blended_prototypes()
        similarities = selected_features.mm(prototypes.t())
        prototype_predictions = similarities.argmax(dim=1)
        agreement = prototype_predictions.eq(selected_labels)
        assigned_similarity = similarities.gather(
            1, selected_labels.unsqueeze(1)
        ).squeeze(1)
        return (
            int(agreement.sum().item()),
            int(selected_labels.numel()),
            float(assigned_similarity.mean().item()),
        )

# 来控制“目标原型库相关的损失项在训练中从无到有、逐步启用
def prototype_weight_for_epoch(epoch, max_weight, warmup_epochs, ramp_epochs):
    """Warm up the target prototype bank before enabling its gradient loss."""
    if max_weight <= 0.0 or epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_weight)
    progress = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return float(max_weight) * progress
