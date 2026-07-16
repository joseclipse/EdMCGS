import torch
import torch.nn.functional as F


def rgb_to_gray(image):
    # image: (H, W, 3), values in [0, 1]
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def compute_event_loss(rendered_start, rendered_end, event_map,
                       contrast_threshold=0.2, eps=1e-3, use_l1=True):
    # Accept (3, H, W) or (H, W, 3)
    if rendered_start.shape[0] == 3:
        rendered_start = rendered_start.permute(1, 2, 0)
    if rendered_end.shape[0] == 3:
        rendered_end = rendered_end.permute(1, 2, 0)

    gray_start = rgb_to_gray(rendered_start).clamp(eps, 1.0)
    gray_end   = rgb_to_gray(rendered_end).clamp(eps, 1.0)

    predicted_events = (torch.log(gray_end) - torch.log(gray_start)) / contrast_threshold
    # if use_l1:
    #     return F.l1_loss(predicted_events, event_map)
    return F.mse_loss(predicted_events, event_map)


def compute_event_loss_omega(rendered_start, rendered_end, event_map, omega_map,
                             contrast_threshold=0.2, eps=1e-3, delta=1.0):
    """模式②：ω 驱动的能量再平衡事件 loss。

    把逐像素事件误差按 ω 软划分为动态/静态两部分，各自除以自身质量后平权相加。
    这样 4% 的运动臂与 96% 的静态背景各按"平均误差"计票，臂不再被像素数碾压，
    静态仍受监督。omega_map 必须是 detach 的（否则 ω 会塌缩到 0 把 loss 乘没）。
    """
    if rendered_start.shape[0] == 3:
        rendered_start = rendered_start.permute(1, 2, 0)
    if rendered_end.shape[0] == 3:
        rendered_end = rendered_end.permute(1, 2, 0)

    gray_start = rgb_to_gray(rendered_start).clamp(eps, 1.0)
    gray_end   = rgb_to_gray(rendered_end).clamp(eps, 1.0)
    predicted_events = (torch.log(gray_end) - torch.log(gray_start)) / contrast_threshold

    if omega_map.dim() == 3:                       # (C,H,W) -> (H,W)
        omega_map = omega_map[0]
    w = omega_map.detach().clamp(0.0, 1.0)         # detach：权重不可学，防 ω 作弊塌缩

    e = (predicted_events - event_map) ** 2        # (H,W) 逐像素事件误差
    L_dyn  = (w * e).sum()       / (w.sum()       + delta)
    L_stat = ((1.0 - w) * e).sum() / ((1.0 - w).sum() + delta)
    return L_dyn + L_stat
