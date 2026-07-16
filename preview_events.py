"""
只从已有的 events.npz 生成多张 preview，不重新模拟事件。
同时对比：直接用 convert("RGB") vs 正确 alpha 合成的灰度图差异。
"""
import numpy as np
from PIL import Image
import os

EV_PATH  = "../dataset/lego/events/events.npz"
IMG_DIR  = "../dataset/lego/rgb"
OUT_DIR  = "../dataset/lego/events"
N_PREVIEWS = 8
WINDOW_FRAMES = 10  # 每张 preview 覆盖多少帧的事件
FPS = 1000.0

ev = np.load(EV_PATH)
x, y, t, p = ev['x'], ev['y'], ev['t'], ev['p']
H, W = 400, 400
ns_per_frame = int(1e9 / FPS)
window_ns = WINDOW_FRAMES * ns_per_frame
t_min, t_max = t[0], t[-1]

def visualize(x, y, p, H, W, path):
    img = np.ones((H, W, 3), dtype=np.float32) * 0.5
    img[y[p > 0], x[p > 0]] = [1, 0, 0]  # 红 = 正极性
    img[y[p < 0], x[p < 0]] = [0, 0, 1]  # 蓝 = 负极性
    Image.fromarray((img * 255).astype(np.uint8)).save(path)

# ── 多张事件 preview ────────────────────────────────────────
print("生成事件 preview:")
for i in range(N_PREVIEWS):
    t_center = t_min + int((t_max - t_min) * i / max(N_PREVIEWS - 1, 1))
    t_s = max(t_min, t_center - window_ns // 2)
    t_e = t_s + window_ns
    mask = (t >= t_s) & (t < t_e)
    n = mask.sum()
    path = os.path.join(OUT_DIR, f"preview_{i:02d}_t{t_s//1_000_000}ms.png")
    visualize(x[mask], y[mask], p[mask], H, W, path)
    print(f"  [{t_s/1e6:.0f}~{t_e/1e6:.0f}ms] {n:,} 事件 → {os.path.basename(path)}")

# ── 对比：旧版 convert("RGB") vs 正确 alpha 合成 ───────────
print("\n对比灰度图差异（前5帧）:")
img_paths = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))[:5]
for name in img_paths:
    path = os.path.join(IMG_DIR, name)
    img_orig = Image.open(path)

    # 旧版：直接丢弃 alpha
    bad = np.array(img_orig.convert("RGB"), dtype=np.float32) / 255.0
    bad_gray = 0.299*bad[:,:,0] + 0.587*bad[:,:,1] + 0.114*bad[:,:,2]

    # 正确版：合成到白色背景
    bg = Image.new("RGB", img_orig.size, (255, 255, 255))
    bg.paste(img_orig, mask=img_orig.split()[3])
    good = np.array(bg, dtype=np.float32) / 255.0
    good_gray = 0.299*good[:,:,0] + 0.587*good[:,:,1] + 0.114*good[:,:,2]

    diff = np.abs(bad_gray - good_gray)
    print(f"  {name}: 最大差异={diff.max():.4f}, 影响像素={( diff>0.01).sum()}")
