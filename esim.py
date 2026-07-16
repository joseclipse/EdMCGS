import argparse
import glob
import os

import esim_torch
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Generate ESIM events from rendered images.")
    parser.add_argument("--image_dir", default="../dataset/lego/rgb")
    parser.add_argument("--output_dir", default="../dataset/lego/events")
    parser.add_argument("--fps", type=float, default=1000.0)
    parser.add_argument("--contrast_threshold", type=float, default=0.2)
    parser.add_argument("--refractory_period_ns", type=int, default=0)
    parser.add_argument("--preview_frames", type=int, default=10)
    parser.add_argument("--n_previews", type=int, default=5, help="生成几张分布在不同时段的预览图")
    parser.add_argument("--white_background", action="store_true", default=True,
                        help="RGBA 图像合成到白色背景（与 --white_background 训练一致）")
    parser.add_argument("--black_background", dest="white_background", action="store_false",
                        help="RGBA 图像合成到黑色背景")
    return parser.parse_args()


def load_image_as_log_gray(path, device, white_background=True):
    """Read an image and return log grayscale intensity.
    RGBA images are composited onto white (or black) background to match D-3DGS behavior.
    """
    img = Image.open(path)
    if img.mode == "RGBA":
        bg_color = (255, 255, 255) if white_background else (0, 0, 0)
        bg = Image.new("RGB", img.size, bg_color)
        bg.paste(img, mask=img.split()[3])  # 用 alpha 通道合成，而非直接丢弃
        img = bg
    else:
        img = img.convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    log_gray = np.log(gray + 1e-4)
    return torch.from_numpy(log_gray).to(device=device, dtype=torch.float32)


def collect_image_paths(image_dir):
    patterns = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    return sorted(image_paths)


def visualize_events(events, height, width, t_start_ns, t_end_ns, save_path):
    """Save a quick polarity preview for a selected time interval."""
    mask = (events["t"] >= t_start_ns) & (events["t"] < t_end_ns)
    event_img = np.ones((height, width, 3), dtype=np.float32) * 0.5

    x = events["x"][mask]
    y = events["y"][mask]
    p = events["p"][mask]

    event_img[y[p > 0], x[p > 0]] = [1, 0, 0]
    event_img[y[p < 0], x[p < 0]] = [0, 0, 1]

    img = Image.fromarray((event_img * 255).astype(np.uint8))
    img.save(save_path)
    print(f"可视化保存到: {save_path}")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("esim_torch 的 CUDA kernel 要求 CUDA tensor，但当前 PyTorch 没有可用 CUDA。")

    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = collect_image_paths(args.image_dir)
    if len(image_paths) < 2:
        raise RuntimeError(f"至少需要 2 张图片，当前在 {args.image_dir} 找到 {len(image_paths)} 张。")

    ns_per_frame = int(1e9 / args.fps)
    timestamps_ns = torch.arange(len(image_paths), dtype=torch.int64, device=device) * ns_per_frame

    with Image.open(image_paths[0]) as first_image:
        width, height = first_image.size

    print(f"使用设备: {device} ({torch.cuda.get_device_name(device)})")
    print(f"找到 {len(image_paths)} 张图片: {args.image_dir}")
    print(f"图片尺寸: {width}x{height}")
    print(f"每帧时间间隔: {ns_per_frame} ns ({1000 / args.fps:.2f} ms)")
    print(f"总时长: {timestamps_ns[-1].item() / 1e9:.3f} 秒")

    esim = esim_torch.ESIM(
        contrast_threshold_neg=args.contrast_threshold,
        contrast_threshold_pos=args.contrast_threshold,
        refractory_period_ns=args.refractory_period_ns,
    )

    print("生成事件中...")
    all_events = {"x": [], "y": [], "t": [], "p": []}
    total_events = 0

    for i, image_path in enumerate(image_paths):
        log_image = load_image_as_log_gray(image_path, device, args.white_background)
        events = esim.forward(log_image, timestamps_ns[i])

        if events is not None and len(events["t"]) > 0:
            events = {k: v.cpu() for k, v in events.items()}
            for key in all_events:
                all_events[key].append(events[key])
            total_events += len(events["t"])

        if i % 100 == 0 or i == len(image_paths) - 1:
            print(f"  帧 {i}/{len(image_paths) - 1}: 累计 {total_events:,} 个事件")

    if not all_events["t"]:
        print("警告：没有生成任何事件！请检查图片是否有运动或降低 contrast_threshold。")
        return

    final_events = {
        "x": torch.cat(all_events["x"]).numpy().astype(np.int16),
        "y": torch.cat(all_events["y"]).numpy().astype(np.int16),
        "t": torch.cat(all_events["t"]).numpy().astype(np.int64),
        "p": torch.cat(all_events["p"]).numpy().astype(np.int8),
    }

    output_path = os.path.join(args.output_dir, "events.npz")
    np.savez(output_path, **final_events)
    print(f"\n总事件数: {len(final_events['t']):,}")
    print(f"平均每帧事件数: {len(final_events['t']) / len(image_paths):.0f}")
    print(f"事件已保存到: {output_path}")

    t_total = final_events["t"][-1] - final_events["t"][0]
    window_ns = args.preview_frames * ns_per_frame
    for i in range(args.n_previews):
        # 均匀分布在整个时间轴上
        t_center = final_events["t"][0] + int(t_total * i / max(args.n_previews - 1, 1))
        t_s = max(final_events["t"][0], t_center - window_ns // 2)
        t_e = t_s + window_ns
        preview_path = os.path.join(args.output_dir, f"events_preview_{i:02d}.png")
        n_ev = int(((final_events["t"] >= t_s) & (final_events["t"] < t_e)).sum())
        print(f"preview {i}: t=[{t_s/1e6:.1f}, {t_e/1e6:.1f}]ms, {n_ev:,} 个事件")
        visualize_events(final_events, height, width, t_s, t_e, preview_path)


if __name__ == "__main__":
    main()