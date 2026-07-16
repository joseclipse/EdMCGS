import torch


def accumulate_events(events, H, W, device='cuda'):
    event_map = torch.zeros(H, W, device=device, dtype=torch.float32)
    if len(events['x']) == 0:
        return event_map
    x = events['x'].long().to(device).clamp(0, W - 1)
    y = events['y'].long().to(device).clamp(0, H - 1)
    p = events['p'].to(device)
    event_map.view(-1).scatter_add_(0, y * W + x, p)
    return event_map


def get_subsegment_event_maps(t_start_norm, t_end_norm, n_sub,
                               event_loader, H, W, device='cuda'):
    dt = (t_end_norm - t_start_norm) / n_sub
    event_maps = []
    sub_timestamps = []
    for j in range(n_sub):
        tau_s = t_start_norm + j * dt
        tau_e = t_start_norm + (j + 1) * dt
        seg = event_loader.get_events_between(tau_s, tau_e)
        event_maps.append(accumulate_events(seg, H, W, device=device))
        sub_timestamps.append((tau_s, tau_e))
    return event_maps, sub_timestamps
