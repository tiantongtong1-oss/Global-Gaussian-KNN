import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import Networks
from dataset import RafDataSet, FER
from ema_utils import create_ema_teacher, select_dual_view_pseudo_labels, update_ema_teacher
from prototype_utils import FeatureHook, PrototypeBank, prototype_weight_for_epoch
from global_gaussian_knn_reliability import GlobalGaussianKNNReliabilityBank, combine_rescue_masks
import image_utils as util
from randaugment import RandAugmentMC
# 总训练入口与调度器。串起数据、Source 预训练、EMA Teacher、
# 双视图伪标签、自适应阈值、KNN Reliability Rescue、Prototype Consistency、验证和测试

#随机种子
seed = 1314
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(seed + worker_id)

# 定义训练全部超参数
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data1', type=str, default='rafdb', help='source data')
    parser.add_argument('--data2', type=str, default='fer', help='target data')
    parser.add_argument('--source_path', type=str,
                        default='/workspace/ttt/code/test-upload-clean/datesets/raf-basic')
    parser.add_argument('--target_path', type=str,
                        default='/workspace/ttt/code/data/fer2013')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='load model')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        help='mobilenet_v2, resnet18 or resnet50')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--pre_epochs', type=int, default=30,
                        help='source-domain pre-training epochs')
    parser.add_argument('--epochs', type=int, default=30,
                        help='target-domain self-training epochs')
    parser.add_argument('--w1', type=float, default=4.0,
                        help='alpha: classification loss weight')
    parser.add_argument('--w2', type=float, default=0.3,
                        help='beta: feature modulation loss weight')
    parser.add_argument('--w3', type=float, default=0.1,
                        help='gamma: classifier modulation loss weight')
    parser.add_argument('--phi', type=float, default=1.4,
                        help='legacy CAST threshold arg retained for CLI compatibility; '
                             'not used by the stable relative threshold')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='fixed EMA decay for teacher parameters and floating buffers')
    parser.add_argument('--threshold_base', type=float, default=0.85,
                        help='minimum global center of the adaptive threshold')
    parser.add_argument('--threshold_beta', type=float, default=0.5,
                        help='strength of per-class confidence deviation')
    parser.add_argument('--threshold_margin', type=float, default=0.02,
                        help='margin added above teacher global mean confidence')
    parser.add_argument('--threshold_min', type=float, default=0.80,
                        help='minimum class-adaptive threshold')
    parser.add_argument('--threshold_max', type=float, default=0.95,
                        help='maximum class-adaptive threshold')

    # Prototype-consistency regularizer retained from v1/v2.
    parser.add_argument('--proto_weight', type=float, default=0.10,
                        help='maximum prototype-consistency loss weight')
    parser.add_argument('--proto_temperature', type=float, default=0.20,
                        help='temperature for prototype cosine logits')
    parser.add_argument('--proto_momentum', type=float, default=0.99,
                        help='EMA momentum of target class prototypes')
    parser.add_argument('--proto_source_anchor', type=float, default=0.50,
                        help='source-prototype fraction in the blended prototype')
    parser.add_argument('--proto_warmup_epochs', type=int, default=3,
                        help='epochs that only update prototype memory, without its loss')
    parser.add_argument('--proto_ramp_epochs', type=int, default=5,
                        help='epochs used to linearly ramp prototype loss to full weight')

    parser.set_defaults(knn_gate=True)
    parser.add_argument('--knn_gate', dest='knn_gate', action='store_true',
                        help='enable class-Gaussian/kNN OR rescue (default)')
    parser.add_argument('--no_knn_gate', dest='knn_gate', action='store_false',
                        help='strict dual-view confidence baseline without rescue')
    parser.add_argument('--knn_k', type=int, default=20)
    parser.add_argument('--knn_distribution_mass', type=float, default=0.95,
                        help='nominal class Gaussian ellipsoid probability content')
    parser.add_argument('--knn_variance_floor', type=float, default=1e-4,
                        help='minimum per-coordinate class variance')
    parser.add_argument('--knn_sigma_momentum', type=float, default=0.70,
                        help='starting EMA momentum for the pooled source global variance')
    parser.add_argument('--knn_sigma_momentum_end', type=float, default=0.95,
                        help='final EMA momentum for the pooled source global variance')
    parser.add_argument('--knn_sigma_momentum_ramp_refreshes', type=int, default=30,
                        help='number of source refreshes used to increase global variance momentum')
    parser.add_argument('--knn_sparse_penalty', type=float, default=0.5,
                        help='multiplicative score penalty for sparse kNN neighborhoods')
    parser.add_argument('--knn_bandwidth_multiplier', type=float, default=1.0,
                        help='local Gaussian bandwidth = multiplier * sigma')
    parser.add_argument('--knn_score_threshold', type=float, default=0.5)
    parser.add_argument('--knn_density_threshold', type=float, default=0.5,
                        help='dense/sparse cutoff used by the reliability score')
    parser.add_argument('--knn_warmup_epochs', type=int, default=3)
    parser.add_argument('--knn_refresh_interval', type=int, default=1)
    parser.add_argument('--knn_query_chunk_size', type=int, default=128,
                        help='bound temporary kNN distance matrix memory')

    args = parser.parse_args(argv)
    if args.knn_k < 1 or args.knn_query_chunk_size < 1 or args.knn_refresh_interval < 1:
        parser.error('knn_k, knn_query_chunk_size and knn_refresh_interval must be positive')
    if args.knn_warmup_epochs < 0 or args.pre_epochs < 0 or args.epochs < 0:
        parser.error('epoch counts must be nonnegative')
    if not 0 <= args.knn_sigma_momentum < 1:
        parser.error('knn_sigma_momentum must be in [0, 1)')
    if not 0 <= args.knn_sigma_momentum_end < 1:
        parser.error('knn_sigma_momentum_end must be in [0, 1)')
    if args.knn_sigma_momentum_end < args.knn_sigma_momentum:
        parser.error('knn_sigma_momentum_end must be >= knn_sigma_momentum')
    if args.knn_sigma_momentum_ramp_refreshes < 1:
        parser.error('knn_sigma_momentum_ramp_refreshes must be positive')
    if not 0 < args.knn_sparse_penalty <= 1:
        parser.error('knn_sparse_penalty must be in (0, 1]')
    if not 0 < args.knn_distribution_mass < 1:
        parser.error('knn_distribution_mass must be in (0, 1)')
    for name in ('knn_variance_floor', 'knn_bandwidth_multiplier'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error('%s must be finite and positive' % name)
    for name in ('knn_score_threshold', 'knn_density_threshold'):
        if not 0 < getattr(args, name) <= 1:
            parser.error('%s must be in (0, 1]' % name)
    if args.pre_epochs == 0 and not args.checkpoint:
        parser.error('--pre_epochs 0 requires --checkpoint with pretrained model weights')
    return args


#图像预处理、数据增强生成 弱增强 weak、强增强 strong、测试预处理 test
def build_transforms():
    # 对图像的 RGB 三个通道做标准化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # 弱增强
    weak = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    strong = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        #强数据增强（随机选两种增强方式，强度为10 ）
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize,
    ])

    test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return weak, strong, test

# 分类器调制损失，对分类器各类别权重计算两两 余弦相似度，约束分类器权重关系
def classifier_modulation_loss(model):
    weight = model.fc.weight
    norm = torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = weight.mm(weight.t()) / norm.mm(norm.t())
    matrix = cosine - torch.eye(weight.size(0), device=weight.device)
    return torch.mean((matrix + 1.0) / 2.0)


def calculate_target_thresholds(model, loader, class_num, threshold_base,
                                threshold_beta, threshold_margin,
                                threshold_min, threshold_max):

    # EMA Teacher 扫描整个target train，根据各预测类别平均confidence 生成类别自适应置信度阈值

    class_sum = torch.zeros(class_num, dtype=torch.float64)
    # 记录每个预测类别包含的样本数量
    class_count = torch.zeros(class_num, dtype=torch.float64)
    # 切换到评估模式，关闭 Dropout、固定 BatchNorm 等
    model.eval()

    # 这里只进行推理，不计算梯度，减少显存和计算开销
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].cuda(non_blocking=True)

            # 使用模型对目标域图像进行预测
            # logits：分类输出
            # _：其他不需要使用的输出
            logits, _ = model(imgs, None, None, mode='test', task='target')

            #转化成类别概率
            probs = F.softmax(logits, dim=1).cpu()
            # 获取每个样本的最大预测概率和对应预测类别
            # max_prob：模型预测的最高置信度
            # pred_label：最高置信度对应的类别
            max_prob, pred_label = torch.max(probs, dim=1)

            class_sum.scatter_add_(0, pred_label, max_prob.double())
            class_count.scatter_add_(
                0, pred_label, torch.ones_like(max_prob, dtype=torch.float64)
            )

    valid = class_count > 0
    class_mean = class_sum / class_count.clamp_min(1.0)
    if torch.any(valid):
        global_mean = class_mean[valid].mean()
    else:
        global_mean = class_mean.new_tensor(float(threshold_base))

    threshold_center = max(
        float(threshold_base),
        float(global_mean.item()) + float(threshold_margin),
    )
    threshold_center = min(threshold_center, float(threshold_max))

    thresholds = threshold_center + threshold_beta * (class_mean - global_mean)
    thresholds = torch.clamp(
        thresholds, min=threshold_min, max=threshold_max
    ).float()
    thresholds[~valid] = float(threshold_max)
    return (
        thresholds,
        class_mean.float(),
        float(global_mean.item()),
        float(threshold_center),
    )


def evaluate(model, loader, criterion, num_samples, epoch, split_name):
    # 在 FER val/test 上算 CE、accuracy 和 confusion matrix
    val_loss = 0.0
    iter_cnt = 0
    bingo_cnt = 0
    preds, labels = [], []

    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            out, _ = model(imgs, targets, None, mode='test')
            loss = torch.mean(criterion(out, targets))
            val_loss += loss.item()
            iter_cnt += 1

            predicts = torch.argmax(out, dim=1)
            bingo_cnt += torch.eq(predicts, targets).sum().item()
            preds.append(predicts.cpu())
            labels.append(targets.cpu())

    avg_loss = val_loss / max(iter_cnt, 1)
    acc = float(bingo_cnt) / float(num_samples)
    print('[Epoch %d] Target %s accuracy: %.4f. Loss: %.3f' %
          (epoch, split_name, acc, avg_loss))
    util.make_confucion_matrix(preds, labels)
    return acc


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args,
                    teacher=None, prototype_bank=None, reliability_bank=None):
    # 保存 Student、optimizer、scheduler、epoch 等；
    # 目标阶段还可保存 EMA Teacher 和 PrototypeBank
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc,
        'args': vars(args),
    }
    if teacher is not None:
        state['ema_teacher'] = teacher.state_dict()
    if prototype_bank is not None:
        state['prototype_bank'] = prototype_bank.state_dict()
    if reliability_bank is not None:
        state['reliability_bank'] = reliability_bank.state_dict()
    torch.save(state, path)


def initialize_source_prototypes(teacher, feature_hook, loader, prototype_bank):

    # 用 Teacher 在确定性的 RAF-DB source view 上抽特征，创建固定 源域原型

    teacher.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            teacher(imgs, None, None, mode='test', task='source')
            prototype_bank.accumulate_source(feature_hook.output, targets)
    prototype_bank.finalize_source()


def log_gaussian_means(reliability_bank):
    """Print complete per-class mu vectors and scale diagnostics once per refresh."""
    means = reliability_bank.class_means.detach().cpu().numpy()
    ready = reliability_bank.distribution_initialized.detach().cpu().tolist()
    counts = reliability_bank.class_source_counts.detach().cpu().tolist()
    for c, mu in enumerate(means):
        if not ready[c]:
            print('[Gaussian mu] class=%d count=%d ready=False (no valid distribution)' %
                  (c, counts[c]), flush=True)
            continue
        norm = float(np.linalg.norm(mu))
        abs_max = float(np.abs(mu).max())
        finite = bool(np.isfinite(mu).all())
        # Means of unit vectors and their convex EMA have norm <= 1.
        bound_ok = finite and norm <= 1.00001 and abs_max <= 1.00001
        print('[Gaussian mu] class=%d count=%d ready=True l2=%.8g abs_max=%.8g '
              'finite=%s bound_ok=%s' %
              (c, counts[c], norm, abs_max, finite, bound_ok), flush=True)
        print('[Gaussian mu values] class=%d mu=%s' %
              (c, np.array2string(mu, precision=8, threshold=mu.size,
                                 max_line_width=1000000, separator=', ')), flush=True)
        if not bound_ok:
            print('[Gaussian mu WARNING] class=%d exceeds normalized-mean bounds; '
                  'check features/statistics/checkpoint.' % c, flush=True)


@torch.no_grad()
def refresh_reliability_bank(teacher, feature_hook, source_loader, target_loader,
                             reliability_bank):
    """Refresh source scale and target-train memory in the same teacher space.

    Target labels from the dataset are intentionally ignored. A deterministic
    view is used for memory; only the two training weak views decide agreement.
    """
    device = next(teacher.parameters()).device
    reliability_bank.begin_source_refresh()
    teacher.eval()
    for imgs, targets in source_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher(imgs, None, None, mode='test', task='source')
        reliability_bank.accumulate_source(feature_hook.output, targets)
    reliability_bank.finalize_source()

    features, labels, confidences, sample_ids = [], [], [], []
    for imgs, _, ids in target_loader:
        logits, _ = teacher(imgs.to(device, non_blocking=True), None, None,
                            mode='test', task='target')
        conf, pred = F.softmax(logits, dim=1).max(dim=1)
        features.append(feature_hook.output.detach().cpu())
        labels.append(pred.cpu())
        confidences.append(conf.cpu())
        sample_ids.append(ids.cpu())
    if features:
        reliability_bank.set_target_memory(
            torch.cat(features), torch.cat(labels), torch.cat(confidences), torch.cat(sample_ids)
        )


def run_training():
    args = parse_args()
    model_path = os.path.join('./models', args.data1 + '_' + args.data2)
    os.makedirs(model_path, exist_ok=True)

    source_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_source_global_gaussian_knn_v7_source_best.pth'
    )
    target_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_source_global_gaussian_knn_v7_target_best.pth'
    )

    print('---------------------------------------------------------------------------------------')
    print('EMA + Dual View + Stable CAT + Prototype KNN v4 + Confidence/KNN OR: '
          '%s with source %s and target %s' %
          (args.backbone, args.data1, args.data2))
    print('alpha(w1):%s beta(w2):%s gamma(w3):%s ema:%s '
          'tau0:%s threshold_beta:%s margin:%s threshold_range:[%s,%s]' %
          (args.w1, args.w2, args.w3, args.ema_decay,
           args.threshold_base, args.threshold_beta, args.threshold_margin,
           args.threshold_min, args.threshold_max))
    print('prototype weight:%s temp:%s momentum:%s source_anchor:%s warmup:%s ramp:%s' %
          (args.proto_weight, args.proto_temperature, args.proto_momentum,
           args.proto_source_anchor, args.proto_warmup_epochs, args.proto_ramp_epochs))
    print('knn rescue:%s k:%s distribution_mass:%s global_momentum:[%s->%s] ramp:%s '
          'bandwidth:%s score_threshold:%s density_threshold:%s sparse_penalty:%s '
          'warmup:%s refresh_interval:%s' %
          (args.knn_gate, args.knn_k, args.knn_distribution_mass,
           args.knn_sigma_momentum, args.knn_sigma_momentum_end,
           args.knn_sigma_momentum_ramp_refreshes, args.knn_bandwidth_multiplier,
           args.knn_score_threshold, args.knn_density_threshold, args.knn_sparse_penalty,
           args.knn_warmup_epochs, args.knn_refresh_interval))
    print('---------------------------------------------------------------------------------------')

    if args.backbone == 'resnet18':
        train_batch, test_batch = 128, 128
    elif args.backbone == 'resnet50':
        train_batch, test_batch = 128, 100
    elif args.backbone == 'mobilenet_v2':
        train_batch, test_batch = 128, 128
    else:
        raise ValueError('Backbone Error!')

    weak_transform, strong_transform, test_transform = build_transforms()

    if args.data1 != 'rafdb' or args.data2 != 'fer':
        raise ValueError('This branch implements RAF-DB -> FER2013.')

    source_train = RafDataSet(
        args.source_path, phase='train', transform=weak_transform,
        strong_transform=None, basic_aug=False
    )
    source_prototype = RafDataSet(
        args.source_path, phase='train', transform=test_transform,
        strong_transform=None, basic_aug=False
    )

    target_train = FER(
        args.target_path, phase='train', transform=weak_transform,
        weak2_transform=weak_transform, strong_transform=strong_transform,
        basic_aug=False, return_index=True
    )
    target_threshold = FER(
        args.target_path, phase='train', transform=weak_transform,
        strong_transform=None, basic_aug=False
    )
    target_memory = None
    if args.knn_gate:
        target_memory = FER(
            args.target_path, phase='train', transform=test_transform,
            basic_aug=False, return_index=True,
        )
        if target_train.file_paths != target_memory.file_paths:
            raise RuntimeError('training and memory views must share identical sample IDs')
    target_val = FER(
        args.target_path, phase='val', transform=test_transform,
        strong_transform=None
    )
    target_test = FER(
        args.target_path, phase='test', transform=test_transform,
        strong_transform=None
    )

    class_num = 7

    train_loader_source = torch.utils.data.DataLoader(
        source_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True, worker_init_fn=_init_fn,
    )
    prototype_loader_source = torch.utils.data.DataLoader(
        source_prototype, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True, worker_init_fn=_init_fn,
    )
    train_loader_target = torch.utils.data.DataLoader(
        target_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True, worker_init_fn=_init_fn,
    )
    threshold_loader_target = torch.utils.data.DataLoader(
        target_threshold, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True, worker_init_fn=_init_fn,
    )
    memory_loader_target = None
    if target_memory is not None:
        memory_loader_target = torch.utils.data.DataLoader(
            target_memory, batch_size=test_batch, num_workers=args.workers,
            shuffle=False, pin_memory=True, worker_init_fn=_init_fn,
        )
    val_loader_target = torch.utils.data.DataLoader(
        target_val, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True,
    )
    test_loader_target = torch.utils.data.DataLoader(
        target_test, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True,
    )
    # 输出分类 logits 和 source affinity loss
    model = Networks.Model(backbone=args.backbone, num_classes=class_num).cuda()
    # 训练组件初始化
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.checkpoint:
        print('Loading pretrained weights...', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=True)

    best_source_val_acc = -1.0
    for i in range(args.pre_epochs):
        model.train()
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        batch_count = 0
        bingo_cnt = 0
        sample_count = 0

        for imgs, targets in train_loader_source:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            # 清空梯度
            optimizer.zero_grad()
            output = model(imgs, targets, None, 'train', 'source')
            cls_loss = torch.mean(criterion(output[0], targets))
            aff_loss = output[1]
            weight_loss = classifier_modulation_loss(model)
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            predicts = torch.argmax(output[0], dim=1)
            bingo_cnt += torch.eq(predicts, targets).sum().item()
            sample_count += targets.numel()
            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            batch_count += 1
        # 更新学习率
        scheduler.step()
        train_acc = float(bingo_cnt) / max(sample_count, 1)
        print('[Source Epoch %d] Training accuracy: %.4f. Classification Loss: %.3f '
              'Affinity Loss: %.3f Weight Loss: %.3f LR: %.6f' %
              (i, train_acc,
               train_loss1 / max(batch_count, 1),
               train_loss2 / max(batch_count, 1),
               train_loss3 / max(batch_count, 1),
               optimizer.param_groups[0]['lr']))
        # 保存最佳的model
        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_source_val_acc:
            best_source_val_acc = val_acc
            save_checkpoint(source_best_path, model, optimizer, scheduler, i,
                            best_source_val_acc, args)
            print('best source-stage validation accuracy %.4f' % best_source_val_acc)
    # --checkpoint is weights initialization, not a full target-stage resume.
    if args.pre_epochs > 0:
        checkpoint = torch.load(source_best_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        # Weights were loaded above; start target training with a fresh optimizer.
        source_best_path = args.checkpoint
        best_source_val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), -1, 'Validation'
        )

    # 构建 EMA 教师模型
    teacher = create_ema_teacher(model).cuda()

    # 一个特征钩子，挂到 teacher.feature 和 model.feature 上
    # 作用：在前向传播时自动截获骨干网络的输出特征 fea，不需要改模型代码。
    teacher_feature_hook = FeatureHook(teacher.feature)
    student_feature_hook = FeatureHook(model.feature)

    # 初始化原型库（存储每个类别的特征原型）
    prototype_bank = PrototypeBank(
        class_num,
        model.fc.in_features,
        momentum=args.proto_momentum,
        source_anchor=args.proto_source_anchor,  #锚点权重
    ).cuda()

    # 源域数据初始化原型
    initialize_source_prototypes(
        teacher, teacher_feature_hook, prototype_loader_source, prototype_bank
    )
    print('source prototype counts: %s' %
          np.array2string(prototype_bank.source_counts.cpu().numpy(), separator=', '))

    # 新的拯救模块：逐类源域高斯分布 + 逐样本归属检查 + 目标域全类别近邻库。
    reliability_bank = GlobalGaussianKNNReliabilityBank(
        class_num,
        model.fc.in_features,
        k=args.knn_k,
        distribution_mass=args.knn_distribution_mass,
        variance_floor=args.knn_variance_floor,
        sigma_momentum=args.knn_sigma_momentum,
        global_momentum_end=args.knn_sigma_momentum_end,
        global_momentum_ramp_refreshes=args.knn_sigma_momentum_ramp_refreshes,
        sparse_penalty=args.knn_sparse_penalty,
        bandwidth_multiplier=args.knn_bandwidth_multiplier,
        score_threshold=args.knn_score_threshold,
        density_threshold=args.knn_density_threshold,
        query_chunk_size=args.knn_query_chunk_size,
    ).cuda()

    best_target_val_acc = -1.0
    global_step = 0

    for i in range(args.epochs):
        thresholds, class_mean, global_mean, threshold_center = calculate_target_thresholds(
            teacher,
            threshold_loader_target,
            class_num,
            args.threshold_base,
            args.threshold_beta,
            args.threshold_margin,
            args.threshold_min,
            args.threshold_max,
        )
        if args.knn_gate and i % args.knn_refresh_interval == 0:
            refresh_reliability_bank(
                teacher, teacher_feature_hook, prototype_loader_source,
                memory_loader_target, reliability_bank,
            )
            print('[Target Epoch %d] KNN Source_Count: %d Target_Memory_Count: %d '
                  'Global_Sigma: %.6f Observed_Global_Var: %.8f EMA_Momentum: %.4f '
                  'Sigma_Ready: %s' %
                  (i, reliability_bank.source_count.item(),
                   reliability_bank.memory_ids.numel(), reliability_bank.sigma.item(),
                   reliability_bank.current_observed_global_var.item(),
                   reliability_bank.current_global_momentum.item(),
                   reliability_bank.sigma_initialized.item()))
            log_gaussian_means(reliability_bank)
        current_proto_weight = prototype_weight_for_epoch(
            i, args.proto_weight, args.proto_warmup_epochs, args.proto_ramp_epochs
        )
        enable_knn_rescue = args.knn_gate and i >= args.knn_warmup_epochs

        print('[Target Epoch %d] class mean confidence: %s global_mean: %.4f '
              'threshold_center: %.4f' %
              (i, np.array2string(class_mean.numpy(), precision=4),
               global_mean, threshold_center))
        print('[Target Epoch %d] class-adaptive thresholds: %s' %
              (i, np.array2string(thresholds.numpy(), precision=4)))
        print('[Target Epoch %d] KNN OR rescue enabled: %s' %
              (i, enable_knn_rescue))

        model.train()
        source_train_iter = iter(train_loader_source)
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_proto_loss = 0.0
        batch_count = 0
        agreement_num = 0
        confidence_accept_num = 0
        knn_pass_num = 0
        knn_rescue_num = 0
        final_accept_num = 0
        score_sum = 0.0
        density_sum = 0.0
        knn_checked_num = 0
        dense_num = 0
        prototype_agree_num = 0
        prototype_checked_num = 0
        prototype_similarity_sum = 0.0
        pseudo_distribution_confidence = np.zeros(class_num, dtype=np.int64)
        pseudo_distribution_rescued = np.zeros(class_num, dtype=np.int64)
        pseudo_distribution_final = np.zeros(class_num, dtype=np.int64)

        for weak1, weak2, strong, _, sample_ids in train_loader_target:
            try:
                source_imgs, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(train_loader_source)
                source_imgs, source_targets = next(source_train_iter)

            weak1 = weak1.cuda(non_blocking=True)
            weak2 = weak2.cuda(non_blocking=True)
            strong = strong.cuda(non_blocking=True)

            teacher.eval()
            # 用EMA Teacher对目标域样本的两个弱增强视图进行预测，生成伪标签，并提取一个融合后的teacher 特征，供后面的
            # kNN可靠性判断和prototype更新使用
            with torch.no_grad():
                logits1, _ = teacher(weak1, None, None, 'test', 'target')
                teacher_features1 = teacher_feature_hook.output.detach()
                logits2, _ = teacher(weak2, None, None, 'test', 'target')
                teacher_features2 = teacher_feature_hook.output.detach()
                pseudo_targets, confidence_idx, agree_count = select_dual_view_pseudo_labels(
                    logits1, logits2, thresholds
                )
                teacher_mean_features = F.normalize(
                    F.normalize(teacher_features1, dim=1)
                    + F.normalize(teacher_features2, dim=1),
                    dim=1,
                )

                # 传统的高置信度伪标签
                confidence_mask = confidence_idx.bool()


                agreement_mask = torch.argmax(logits1, dim=1).eq(
                    torch.argmax(logits2, dim=1)
                )
                knn_result = None
                reliability_mask = torch.zeros_like(agreement_mask)
                if args.knn_gate:
                    knn_result = reliability_bank.gate(
                        teacher_mean_features, pseudo_targets, sample_ids,
                        agreement_mask, thresholds,
                    )
                    reliability_mask = knn_result['pass_mask']

                final_mask, rescue_mask = combine_rescue_masks(
                    confidence_mask, agreement_mask, reliability_mask, enable_knn_rescue
                )
                # Networks.forward selects idx == 1: retain a binary mask here.
                con_idx = final_mask.float()

                # 评估函数，用来统计在“可靠样本”上，模型特征与原型之间的预测一致性和相似度，会只检查 con_idx=1 的最终可靠目标样本
                proto_agree, proto_checked, proto_similarity = (
                    prototype_bank.agreement_stats(
                        teacher_mean_features, pseudo_targets, con_idx
                    )
                )
            # 统计当前epoch中不同阶段一共接纳了多少目标样本
            agreement_num += agree_count
            confidence_accept_num += int(confidence_mask.sum().item())
            knn_pass_num += int(reliability_mask.sum().item())
            knn_rescue_num += int(rescue_mask.sum().item())
            final_accept_num += int(final_mask.sum().item())

            if knn_result is not None:
                checked = knn_result['checked_mask']
                score_sum += float(knn_result['score'][checked].sum().item())
                density_sum += float(knn_result['density'][checked].sum().item())
                knn_checked_num += int(checked.sum().item())
                dense_num += int(knn_result['dense_mask'].sum().item())
            # 累计prototype诊断指标
            prototype_agree_num += proto_agree
            prototype_checked_num += proto_checked
            prototype_similarity_sum += proto_similarity * proto_checked

            # 统计伪标签在不同掩码筛选下的类别分布
            confidence_labels = pseudo_targets[confidence_mask].cpu().numpy()
            rescued_labels = pseudo_targets[rescue_mask].cpu().numpy()
            final_labels = pseudo_targets[final_mask].cpu().numpy()
            if confidence_labels.size > 0:
                pseudo_distribution_confidence += np.bincount(
                    confidence_labels, minlength=class_num
                ).astype(np.int64)
            if rescued_labels.size > 0:
                pseudo_distribution_rescued += np.bincount(
                    rescued_labels, minlength=class_num
                ).astype(np.int64)
            if final_labels.size > 0:
                pseudo_distribution_final += np.bincount(
                    final_labels, minlength=class_num
                ).astype(np.int64)

            model.train()
            # 给所有源域样本一个权重1，默认全可信
            source_con_idx = torch.ones(source_imgs.shape[0])
            # 源域图像+目标域strong augmentation
            train_imgs = torch.cat((source_imgs, strong.cpu()), dim=0).cuda(non_blocking=True)
            train_targets = torch.cat((source_targets, pseudo_targets.cpu()), dim=0).cuda(non_blocking=True)
            train_con_idx = torch.cat((source_con_idx, con_idx.cpu()), dim=0).cuda(non_blocking=True)

            # 清空模型参数的梯度
            optimizer.zero_grad()
            output = model(
                train_imgs, train_targets, train_con_idx, 'train', 'target',
                source_count=source_imgs.shape[0],
            )
            # 分类损失
            per_sample_loss = criterion(output[0], train_targets) * train_con_idx
            # 对真正有效的源域和目标域样本求平均分类损失
            cls_loss = per_sample_loss.sum() / train_con_idx.sum().clamp_min(1.0)
            aff_loss = output[1]
            # 分类器权重之间的正则项
            weight_loss = classifier_modulation_loss(model)
            # 把前面的源域特征切掉，只保留zt（student）
            target_student_features = student_feature_hook.output[source_imgs.shape[0]:]
            #prototype consistency原型一致性，让目标样本的 Student 特征靠近对应伪标签类别的 prototype
            proto_loss = prototype_bank.consistency_loss(
                target_student_features,
                pseudo_targets,
                con_idx,
                temperature=args.proto_temperature,
            )
            # 总损失
            loss = (
                cls_loss * args.w1
                + aff_loss * args.w2
                + weight_loss * args.w3
                + proto_loss * current_proto_weight
            )
            #反向传播
            loss.backward()
            optimizer.step()

            global_step += 1
            # 更新EMA Teacher
            update_ema_teacher(teacher, model, args.ema_decay, global_step)
            # 这是更新目标域prototype
            prototype_bank.update_target(
                teacher_mean_features, pseudo_targets, con_idx
            )

            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            train_proto_loss += proto_loss.item()
            batch_count += 1
        # 更新学习率
        scheduler.step()
        # 求均值
        proto_agreement_rate = (
            float(prototype_agree_num) / float(prototype_checked_num)
            if prototype_checked_num > 0 else 0.0
        )
        proto_mean_similarity = (
            prototype_similarity_sum / float(prototype_checked_num)
            if prototype_checked_num > 0 else 0.0
        )
        mean_score = score_sum / max(knn_checked_num, 1)
        mean_density = density_sum / max(knn_checked_num, 1)

        print('[Target Epoch %d] Agreement_Num: %d Confidence_Accept_Num: %d '
              'KNN_Pass_Num: %d KNN_Rescue_Num: %d Final_Accept_Num: %d '
              'Mean_Reliability: %.4f Mean_Density: %.4f '
              'KNN_Checked_Num: %d Dense_Num: %d Sparse_Num: %d' %
              (i, agreement_num, confidence_accept_num, knn_pass_num,
               knn_rescue_num, final_accept_num, mean_score, mean_density,
               knn_checked_num, dense_num, knn_checked_num - dense_num))
        print('[Target Epoch %d] Pseudo_Distribution_Confidence: %s' %
              (i, np.array2string(pseudo_distribution_confidence, separator=', ')))
        print('[Target Epoch %d] Pseudo_Distribution_Rescued: %s' %
              (i, np.array2string(pseudo_distribution_rescued, separator=', ')))
        print('[Target Epoch %d] Pseudo_Distribution_Final: %s' %
              (i, np.array2string(pseudo_distribution_final, separator=', ')))
        print('[Target Epoch %d] Prototype_Agreement: %d/%d (%.4f) '
              'Mean_Assigned_Cosine: %.4f Target_Prototype_Counts: %s' %
              (i, prototype_agree_num, prototype_checked_num,
               proto_agreement_rate, proto_mean_similarity,
               np.array2string(prototype_bank.target_counts.cpu().numpy(), separator=', ')))
        print('[Target Epoch %d] Classification Loss: %.3f Affinity Loss: %.3f '
              'Weight Loss: %.3f Prototype Loss: %.3f ProtoWeight: %.4f LR: %.6f' %
              (i,
               train_loss1 / max(batch_count, 1),
               train_loss2 / max(batch_count, 1),
               train_loss3 / max(batch_count, 1),
               train_proto_loss / max(batch_count, 1),
               current_proto_weight,
               optimizer.param_groups[0]['lr']))
        # 在每个epoch 后用目标域 validation set做验证
        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_target_val_acc:
            best_target_val_acc = val_acc
            save_checkpoint(
                target_best_path, model, optimizer, scheduler, i,
                best_target_val_acc, args, teacher=teacher,
                prototype_bank=prototype_bank,
                reliability_bank=reliability_bank,
            )
            print('best target-stage validation accuracy %.4f' % best_target_val_acc)

    if args.epochs > 0:
        checkpoint = torch.load(target_best_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        selected_val_acc = best_target_val_acc
    else:
        checkpoint = torch.load(source_best_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        selected_val_acc = best_source_val_acc

    teacher_feature_hook.close()
    student_feature_hook.close()

    test_acc = evaluate(
        model, test_loader_target, criterion, len(target_test), args.epochs, 'Test'
    )
    print('best source-stage validation accuracy %.4f' % best_source_val_acc)
    if args.epochs > 0:
        print('best target-stage validation accuracy %.4f' % best_target_val_acc)
    print('selected validation accuracy %.4f' % selected_val_acc)
    print('final target test accuracy %.4f' % test_acc)
    return test_acc


if __name__ == '__main__':
    run_training()

