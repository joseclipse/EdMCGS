import numpy as np
import torch


class EventLoader:
    def __init__(self, npz_path, total_duration_ns):
        data = np.load(npz_path)
        self.x = torch.tensor(data['x'].astype(np.int64))
        self.y = torch.tensor(data['y'].astype(np.int64))
        self.t = torch.tensor(data['t'].astype(np.int64))
        self.p = torch.tensor(data['p'].astype(np.float32))
        self.total_duration_ns = total_duration_ns

        # 区间查询用二分（searchsorted），要求 t 单调；esim 逐帧生成本已有序，仅在异常时排序。
        if not bool(torch.all(self.t[1:] >= self.t[:-1])):
            order = torch.argsort(self.t)
            self.x, self.y, self.t, self.p = self.x[order], self.y[order], self.t[order], self.p[order]

        print(f"[EventLoader] {len(self.t):,} events, t={self.t[0].item()} ~ {self.t[-1].item()} ns")

    def normalized_to_ns(self, t_normalized):
        return int(t_normalized * self.total_duration_ns)

    def get_events_between(self, t_start_norm, t_end_norm):
        # O(log N) 二分定位区间，取代 O(N) 全表布尔扫描（4200 万事件下是数量级加速）。
        t_start_ns = self.normalized_to_ns(t_start_norm)
        t_end_ns = self.normalized_to_ns(t_end_norm)
        lo = int(torch.searchsorted(self.t, torch.tensor(t_start_ns, dtype=self.t.dtype), right=False))
        hi = int(torch.searchsorted(self.t, torch.tensor(t_end_ns, dtype=self.t.dtype), right=False))
        sl = slice(lo, hi)
        return {
            'x': self.x[sl],
            'y': self.y[sl],
            't': self.t[sl],
            'p': self.p[sl],
        }
