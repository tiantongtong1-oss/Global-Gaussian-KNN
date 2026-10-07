from torch import nn
import torch
from torchvision import models


def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    #得到一个source 和 target的相似度矩阵

    """Multi-kernel Gaussian matrix used by MK-MMD (paper Eq. 9-12)."""
    n_samples = int(source.size(0)) + int(target.size(0))  #总样本数
    total = torch.cat([source, target], dim=0)
    #增加维度
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    #一次性计算所有样本两两之间的差，并计算欧氏距离的平方
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if n_samples <= 1:
        return torch.zeros_like(l2_distance)
    #作为典型距离，detach()不做反向传播，仅作为参数尺度
    bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
    #限制最小值
    bandwidth = torch.clamp(bandwidth, min=1e-12)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    #两个样本越近，kernel 越接近 1；两个样本越远，kernel 越接近 0。
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source_features, target_features):
    """Unbiased MK-MMD estimator corresponding to paper Eq. (9)-(10)."""
    ns = source_features.size(0)
    nt = target_features.size(0)
    if ns < 2 or nt < 2:
        return source_features.new_tensor(0.0)

    kernels = compute_kernel_matrix(source_features, target_features)
    xx = kernels[:ns, :ns]
    yy = kernels[ns:, ns:]
    xy = kernels[:ns, ns:]

    xx_term = (xx.sum() - torch.diagonal(xx).sum()) / (ns * (ns - 1))
    yy_term = (yy.sum() - torch.diagonal(yy).sum()) / (nt * (nt - 1))
    xy_term = xy.mean()
    return xx_term + yy_term - 2.0 * xy_term


def remove_element(features, index):
    others = [feature for i, feature in enumerate(features)
              if i != index and feature.size(0) > 0]
    if not others:
        return features[index].new_empty((0, features[index].size(1)))
    return torch.cat(others, dim=0)


class Model(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=7, pretrained=True,
                 drop_rate=0.5, density_bandwidth=0.2):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.density_bandwidth = density_bandwidth
        self.bn = nn.BatchNorm1d(num_classes)

        if backbone == 'resnet18':
            self.feature = nn.Sequential(
                *list(models.resnet18(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        elif backbone == 'resnet50':
            self.feature = nn.Sequential(
                *list(models.resnet50(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(2048, 512)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        elif backbone == 'mobilenet_v2':
            self.feature = nn.Sequential(
                *list(models.mobilenet_v2(pretrained=pretrained).children())[:-1],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(1280, 512),
                nn.Dropout(drop_rate)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        else:
            raise ValueError('Backbone Error!')

    def forward(self, x, targets=None, idx=None, mode='train', task='target',
                epoch=0, source_count=None):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))
        # 测试模式
        if mode != 'train':
            return out, fea.cpu()
        # 在源域训练
        if task == 'source':
            features = self.split_feature_makeLD(fea, targets)
            eta = self.volume(features)
            inter_loss = fea.new_tensor(0.0)

            for i in range(self.num_classes):
                class_features = features[i]
                other_features = remove_element(features, i)
                if class_features.size(0) >= 2 and other_features.size(0) >= 2:
                    # mmd_loss衡量的是第 i 类和其他所有类特征分布之间的相似度
                    inter_loss += mmd_loss(class_features, other_features) * eta[i]

            # Paper Eq. (11)-(12): maximize inter-class discrepancy.
            affinity_loss = -inter_loss / self.num_classes
            # 返回分类 logits以及类间分离损失
            return [out, affinity_loss]
        # 目标域训练
        if task == 'target':
            if idx is None or source_count is None:
                raise ValueError('Target training requires confidence mask idx and source_count.')

            # 1. 用置信掩码 idx 筛选样本
            valid_idx = (idx == 1).nonzero(as_tuple=False).squeeze(1)
            fea_selected = torch.index_select(fea, 0, valid_idx)
            targets_selected = torch.index_select(targets, 0, valid_idx)

            # 2. 前 source_count 个是源域样本，后面是目标域样本
            source_count = min(int(source_count), fea_selected.size(0))
            source_fea = fea_selected[:source_count]
            source_targets = targets_selected[:source_count]
            target_fea = fea_selected[source_count:]
            target_targets = targets_selected[source_count:]

            # 3. 分别按类别拆分
            source_features = self.split_feature_makeLD(source_fea, source_targets)
            target_features = self.split_feature_makeLD(target_fea, target_targets)
            all_features = self.split_feature_makeLD(fea_selected, targets_selected)
            eta = self.volume(all_features)

            # 4. 同时算类内对齐 + 类间分离
            intra_loss = fea.new_tensor(0.0)    # 同类源/目标特征的 MMD
            inter_loss = fea.new_tensor(0.0)    # 不同类特征的 MMD

            for i in range(self.num_classes):

                fea_s = source_features[i]
                fea_t = target_features[i]
                if fea_s.size(0) >= 2 and fea_t.size(0) >= 2:
                    # 类内：源域第 i 类 vs 目标域第 i 类，乘以该类的权重 eta[i]
                    intra_loss += mmd_loss(fea_s, fea_t) * eta[i]

                class_features = all_features[i]  # 第 i 类的所有样本（源+目标）
                other_features = remove_element(all_features, i)    # 除第 i 类外的所有样本
                if class_features.size(0) >= 2 and other_features.size(0) >= 2:
                    # 类间：第 i 类 vs 其他类
                    inter_loss += mmd_loss(class_features, other_features) * eta[i]

            # Paper Eq. (12): intra-class domain alignment + inter-class separation.
            # 特征分布对齐损失
            affinity_loss = (intra_loss - inter_loss) / self.num_classes
            return [out, affinity_loss]

        raise ValueError('Unknown task: %s' % task)

    def split_feature_makeLD(self, x, target):
        if x.size(0) == 0:
            feature_dim = self.fc.in_features
            return [x.new_empty((0, feature_dim)) for _ in range(self.num_classes)]

        x_parts = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero(as_tuple=False).squeeze(1)
            x_parts.append(torch.index_select(x, 0, ind))
        return x_parts

    # CCDR类级别表示权重的实际计算
    def volume(self, features):

        if not features:
            return []

        h = float(self.density_bandwidth)
        weights = []
        for feature in features:
            n = feature.size(0)
            if n == 0:
                weights.append(0.0)
                continue

            with torch.no_grad():
                f = feature.detach()
                distance_sq = torch.cdist(f, f, p=2).pow(2)
                gaussian_kernel = torch.exp(-distance_sq / (2.0 * h * h))
                rho = gaussian_kernel.sum(dim=1) / (float(n) * h)
                rho = rho.clamp_min(1e-12)
                volume = torch.sum(1.0 / rho)
                eta = 1.0 / volume.clamp_min(1e-12)
                weights.append(float(eta.item()))

        return weights
