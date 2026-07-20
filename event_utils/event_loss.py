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

