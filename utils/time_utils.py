import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, sample_farthest_points
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t, **kwargs):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling


class ControlNodeDeform(nn.Module):

    def __init__(self, node_num=512, K=3, is_blender=False, is_6dof=False,
                 use_markov=True, markov_window_size=5, markov_alpha_pos=0.5,
                 markov_hidden_dim=128, markov_num_heads=4):
        super().__init__()
        self.node_num = node_num
        self.K = K
        self.mlp = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof)
        self.nodes = nn.Parameter(torch.randn(node_num, 3))
        self._node_radius = nn.Parameter(torch.zeros(node_num))
        self.register_buffer('inited', torch.tensor(False))
        self.use_markov = use_markov              
        self.markov_alpha_pos = markov_alpha_pos 
        D_evt = 32
        self.event_encoder = nn.Sequential(       # (1,1,H,W) → (1,D_evt,H/4,W/4)
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, D_evt, kernel_size=3, stride=1, padding=1),
        )
        self.evt_embed_fn, evt_xyz_ch = get_embedder(4, 3)   
        self.event_g = nn.Sequential(             
            nn.Linear(evt_xyz_ch + D_evt, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.event_g[-1].weight)   
        nn.init.zeros_(self.event_g[-1].bias)
        self.event_provider = None

    @property
    def node_radius(self):
        return torch.exp(self._node_radius)

    def _event_residual(self, node_pos_det, event_obs, event_proj):
        ef = event_obs.float()
        if ef.dim() == 2:
            ef = ef[None, None]                        # (H,W) → (1,1,H,W)
        elif ef.dim() == 3:
            ef = ef[None]                              # (1,H,W) → (1,1,H,W)
        feat = self.event_encoder(ef)                  # (1, D, H', W') 
        M = node_pos_det.shape[0]
        hom = torch.cat([node_pos_det, node_pos_det.new_ones(M, 1)], -1)   # (M,4)
        clip = hom @ event_proj                        ）
        ndc = clip[:, :2] / clip[:, 3:4].clamp(min=1e-6)                   # (M,2)∈[-1,1]
        grid = torch.stack([ndc[:, 0], -ndc[:, 1]], dim=-1).view(1, M, 1, 2)
        m_node = F.grid_sample(feat, grid, mode='bilinear',
                               align_corners=True, padding_mode='zeros')  # (1,D,M,1)
        m_node = m_node[0, :, :, 0].transpose(0, 1)    # (M, D) 
        xyz_emb = self.evt_embed_fn(self.nodes)        # (M, evt_xyz_ch)
        return self.event_g(torch.cat([xyz_emb, m_node], dim=-1))         # (M,3)

    def _compute_markov_smooth(self, markov_dx, K_conn=10):
        with torch.no_grad():
            nn_idx = knn_points(self.nodes.detach()[None], self.nodes.detach()[None],
                                K=K_conn + 1)[1][0, :, 1:]      
        neighbor_mean = markov_dx[nn_idx].mean(dim=1)           
        return ((markov_dx - neighbor_mean) ** 2).mean()

    def markov_smooth_loss(self):
        return getattr(self, '_markov_smooth_loss', self.nodes.new_zeros(()))

    @torch.no_grad()
    def _init_nodes(self, x):
        N = x.shape[0]
        if N >= self.node_num:
            idx = sample_farthest_points(x[None], K=self.node_num)[1][0]
            nodes = x[idx]
        else:  
            nodes = x[torch.randint(0, N, (self.node_num,), device=x.device)]
        self.nodes.data = nodes.float()
        scene_scale = (x.max(dim=0).values - x.min(dim=0).values).norm()
        self._node_radius.data = torch.log(0.1 * scene_scale + 1e-7) * torch.ones(
            self.node_num, device=x.device)
        self.inited.data = torch.tensor(True, device=self.inited.device)

    def forward(self, x, t, event_obs=None, event_proj=None):
        if not bool(self.inited):
            self._init_nodes(x.detach())
        t_node = t.reshape(-1)[0].view(1, 1).expand(self.node_num, 1)
        node_dx, node_rot, node_sc = self.mlp(self.nodes, t_node)        # (M,3),(M,4),(M,3)
        if self.use_markov and event_obs is None and self.event_provider is not None:
            event_obs, event_proj = self.event_provider(float(t.reshape(-1)[0].item()))
        if self.use_markov and event_obs is not None and event_proj is not None:
            node_pos_det = (self.nodes + node_dx).detach()
            g = self._event_residual(node_pos_det, event_obs, event_proj)   # (M,3)
            node_dx = node_dx + g
            self._markov_smooth_loss = self._compute_markov_smooth(g)       
        else:
            self._markov_smooth_loss = node_dx.new_zeros(())
        with torch.no_grad():
            nn_idx = knn_points(x.detach()[None], self.nodes.detach()[None],
                                K=self.K)[1][0]                          # (N,K)
        nbr = self.nodes[nn_idx]                                         # (N,K,3)
        d2 = ((x.detach()[:, None, :] - nbr) ** 2).sum(-1)              # (N,K)
        w = torch.exp(-d2 / (2 * self.node_radius[nn_idx] ** 2 + 1e-9)) + 1e-7
        w = (w / w.sum(-1, keepdim=True))[..., None]                     # (N,K,1)
        d_xyz   = (node_dx[nn_idx]  * w).sum(1)                          # (N,3)
        d_rot   = (node_rot[nn_idx] * w).sum(1)                          # (N,4)
        d_scale = (node_sc[nn_idx]  * w).sum(1)                          # (N,3)
        return d_xyz, d_rot, d_scale

    def tli_loss(self, t, delta_t=0.05, K_conn=10):
        if not bool(self.inited):
            return self.nodes.new_zeros(())
        with torch.no_grad():
            nn_idx = knn_points(self.nodes.detach()[None], self.nodes.detach()[None],
                                K=K_conn + 1)[1][0, :, 1:]               
        t_val = float(t.reshape(-1)[0].item())
        t1_val = max(0.0, min(1.0, t_val - 0.5 * delta_t))
        t2_val = max(0.0, min(1.0, t_val + 0.5 * delta_t))
        if abs(t2_val - t1_val) < 1e-6:
            return self.nodes.new_zeros(())                              
        t1 = torch.full((self.node_num, 1), t1_val, device=self.nodes.device)
        t2 = torch.full((self.node_num, 1), t2_val, device=self.nodes.device)
        d1, _, _ = self.mlp(self.nodes, t1)
        d2, _, _ = self.mlp(self.nodes, t2)
        pos1 = self.nodes + d1                                           # (M,3) 
        pos2 = self.nodes + d2                                           # (M,3) 
        edges1 = pos1[nn_idx] - pos1[:, None, :]                         # (M,K,3)
        edges2 = pos2[nn_idx] - pos2[:, None, :]
        len1 = edges1.norm(dim=-1)                                       # (M,K)
        len2 = edges2.norm(dim=-1)
        return ((len1 - len2) ** 2).mean()
        # ── 备用：SC-GS λ 退火表（若 jump 因 TLI 回退 ≥0.5dB，首选改这里）──
        # landmarks = [1e-4, 1e-4, 1e-5, 1e-5, 0]; steps = [0, 5000, 10000, 20000, 20001]
