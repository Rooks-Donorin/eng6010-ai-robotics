"""
Method 1 - Color spaces.

Question: which color space lets a simple threshold find the colored objects reliably,
even when the lighting changes?

Experiment:
  * Convert a synthetic Go2 front-camera image to HSV, Lab and YCrCb and show the channels.
  * For each colored object, calibrate a color threshold on the NORMAL-lit image in three
    spaces (RGB distance, HSV hue+saturation, Lab a*b* chroma distance), choosing the threshold
    that gives the best IoU against the exact ground-truth mask from the simulator.
  * Apply the SAME thresholds to the DIM and WARM images and measure IoU again.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from viz_style import save, log_metrics, image_axes, SERIES, TEXT2
import json

D = "data/lighting"
LABELS = json.load(open("data/ego/poses.json"))["labels"]
CONDS = ["normal", "dim", "warm"]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else np.nan


def features(bgr, space):
    """Per-pixel feature vector used for the distance test in each space."""
    if space == "RGB":
        return bgr[..., ::-1].astype(np.float32)
    if space == "HSV":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)   # H in [0,180)
        return hsv
    if space == "Lab (a*b*)":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab[..., 1:]                                             # drop lightness L*


def distance(feat, ref, space):
    if space == "HSV":
        dh = np.abs(feat[..., 0] - ref[0]); dh = np.minimum(dh, 180 - dh)  # hue wraps around
        # pixels with very low saturation have meaningless hue -> push them far away
        return np.where(feat[..., 1] < 60, 999, dh)
    return np.linalg.norm(feat - ref, axis=-1)


def main():
    imgs = {c: cv2.imread(f"{D}/rgb_{c}.png") for c in CONDS}
    seg = cv2.imread(f"{D}/seg_normal.png", 0)       # geometry is identical for all 3 lightings
    present = {n: l for n, l in LABELS.items() if (seg == l).sum() > 150}

    # --- figure 1: channels of each color space
    bgr = imgs["normal"]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    panels = [(bgr[..., ::-1], "RGB image")] + \
             [(hsv[..., i], f"HSV: {n}") for i, n in enumerate(["H (hue)", "S (saturation)", "V (value)"])] + \
             [(lab[..., i], f"Lab: {n}") for i, n in enumerate(["L* (lightness)", "a* (green-red)", "b* (blue-yellow)"])] + \
             [(ycc[..., 1], "YCrCb: Cr")]
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.6))
    for ax, (im, t) in zip(axs.flat, panels):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        image_axes(ax, t)
    fig.suptitle("Synthetic Go2 front-camera frame in different color spaces", x=0.01, ha="left")
    save(fig, "fig01a_color_space_channels.png")

    # --- threshold calibration on the normal image, then test on dim / warm
    spaces = ["RGB", "HSV", "Lab (a*b*)"]
    grids = {"RGB": np.arange(10, 200, 5), "HSV": np.arange(2, 40, 1), "Lab (a*b*)": np.arange(3, 80, 2)}
    results = {s: {c: {} for c in CONDS} for s in spaces}
    masks_dim = {s: np.zeros(seg.shape, bool) for s in spaces}
    for name, l in present.items():
        gt = seg == l
        for s in spaces:
            f_norm = features(imgs["normal"], s)
            ref = np.median(f_norm[gt], axis=0)
            dist = distance(f_norm, ref, s)
            best_t = max(grids[s], key=lambda t: iou(dist < t, gt))
            for c in CONDS:
                m = distance(features(imgs[c], s), ref, s) < best_t
                results[s][c][name] = float(iou(m, gt))
                if c == "dim":
                    masks_dim[s] |= m

    summary = {s: {c: float(np.mean(list(results[s][c].values()))) for c in CONDS} for s in spaces}
    print("Mean IoU over objects (thresholds tuned on the NORMAL image):")
    print(f"  {'space':12s}" + "".join(f"{c:>9s}" for c in CONDS))
    for s in spaces:
        print(f"  {s:12s}" + "".join(f"{summary[s][c]:9.3f}" for c in CONDS))
    log_metrics("color_space", {"objects": list(present), "mean_iou": summary, "per_object_iou": results})

    # --- figure 2: grouped bars (space x lighting)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(spaces)); w = 0.26
    for i, c in enumerate(CONDS):
        vals = [summary[s][c] for s in spaces]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=SERIES[i], label=f"{c} lighting")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=TEXT2)
    ax.set_xticks(x, spaces); ax.set_ylim(0, 1.08); ax.set_ylabel("mean IoU vs. ground truth")
    ax.set_title("Color threshold tuned under normal light, tested under dim and warm light", loc="left")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.grid(axis="x", visible=False)
    save(fig, "fig01b_color_threshold_iou.png")

    # --- figure 3: what the masks look like under DIM light
    fig, axs = plt.subplots(1, 4, figsize=(14, 2.9))
    axs[0].imshow(imgs["dim"][..., ::-1]); image_axes(axs[0], "dim-light image")
    for ax, s in zip(axs[1:], spaces):
        ax.imshow(masks_dim[s], cmap="gray"); image_axes(ax, f"{s} threshold  (IoU {summary[s]['dim']:.2f})")
    save(fig, "fig01c_masks_dim_light.png")


if __name__ == "__main__":
    main()
