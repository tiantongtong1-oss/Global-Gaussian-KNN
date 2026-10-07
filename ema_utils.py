import copy
import torch
import torch.nn.functional as F


def create_ema_teacher(student):
    """Create a frozen EMA teacher initialized from the student."""
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        #关闭教师梯度，不经过反向传播
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, decay=0.999, global_step=None):
    """使用固定的 EMA 衰减率更新 teacher 的参数和浮点型 buffer

    由于 teacher 是由 student 初始化得到的，因此在训练早期没有必要对衰减率进行 warmup；这样做反而会使 teacher 过快地跟随带有较大噪声的 target 更新，。
    浮点型 buffer（尤其是 BatchNorm 中的 running_mean 和 running_var）同样采用 EMA 方式更新。对于整数类型的计数器，例如 num_batches_tracked，则直接复制。
    保留 global_step 参数仅用于兼容现有调用接口。

    """
    del global_step
    ema_decay = float(decay)

    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher_params.items():
        #教师参数更新
        teacher_param.mul_(ema_decay).add_(
            student_params[name].detach(), alpha=1.0 - ema_decay
        )

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        student_buffer = student_buffers[name].detach()
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.mul_(ema_decay).add_(
                student_buffer.to(dtype=teacher_buffer.dtype),
                alpha=1.0 - ema_decay,
            )
        else:
            teacher_buffer.copy_(student_buffer)

    teacher.eval()


@torch.no_grad()
def select_dual_view_pseudo_labels(logits1, logits2, thresholds):
    """严格的双视图伪标签筛选。

    只有当两个弱增强视图预测出相同的类别，并且每个视图的预测置信度都分别超过该类别对应的自适应阈值时，才认为该目标样本是可靠的。
           
    """
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)

    conf1, pred1 = probs1.max(dim=1)
    conf2, pred2 = probs2.max(dim=1)
    agreement = pred1.eq(pred2)

    pseudo_targets = pred1
    thresholds = thresholds.to(device=logits1.device, dtype=conf1.dtype)
    sample_threshold = thresholds.index_select(0, pseudo_targets)
    reliable = (
        agreement
        & (conf1 >= sample_threshold)
        & (conf2 >= sample_threshold)
    )

    return pseudo_targets, reliable.float(), int(agreement.sum().item())
