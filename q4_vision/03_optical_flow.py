"""
Method 3 - Optical flow.

Because the data is synthetic, we know the exact camera pose and the depth of every pixel,
so we can compute the TRUE optical flow and grade the algorithms against it.

  ground truth: back-project pixel (u,v) with its depth into 3-D, move it into the next
                camera's frame, project it again -> (u', v').  flow = (u'-u, v'-v)
  sanity check: warping frame t+1 back with the true flow should reproduce frame t.

Algorithms (OpenCV):
  * Farneback dense optical flow - on raw frames, and on contrast-equalized frames
    (lesson learned: Farneback returned ~0 motion on low-contrast texture until I equalized)
  * DIS (Dense Inverse Search) dense optical flow
  * Pyramidal Lucas-Kanade sparse optical flow on Shi-Tomasi corners
Both are tested on consecutive frames (small motion) and on frames 3 apart (larger motion),
and with / without image pyramids, to show why the Gaussian pyramid matters (see 02).
Error metric: end-point error (EPE) = |flow_estimated - flow_true| in pixels.
"""
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from viz_style import save, log_metrics, image_axes, SERIES, TEXT2

E = "data/ego"
META = json.load(open(f"{E}/poses.json"))["frames"]


def load(k):
    rgb = cv2.imread(f"{E}/rgb_{k:03d}.png")
    depth = cv2.imread(f"{E}/depth_{k:03d}.png", -1).astype(np.float64) / 1000.0
    return rgb, depth


def true_flow(k0, k1):
    """Exact flow from frame k0 to k1 using depth(k0) and both camera poses. Returns flow, valid mask."""
    _, d0 = load(k0)
    _, d1 = load(k1)
    c0, c1 = META[k0], META[k1]
    H, W = d0.shape
    f, cx, cy = c0["f"], c0["cx"], c0["cy"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # MuJoCo camera frame: x right, y up, looks along -z
    pc = np.stack([(u - cx) / f * d0, -(v - cy) / f * d0, -d0], -1)
    R0, t0 = np.array(c0["R"]), np.array(c0["pos"])
    R1, t1 = np.array(c1["R"]), np.array(c1["pos"])
    pw = pc @ R0.T + t0
    p1 = (pw - t1) @ R1                               # = R1^T (pw - t1)
    z = -p1[..., 2]
    u1 = cx + f * p1[..., 0] / z
    v1 = cy - f * p1[..., 1] / z
    flow = np.stack([u1 - u, v1 - v], -1)
    valid = (d0 > 0) & (z > 0.05) & (u1 >= 0) & (u1 <= W - 1) & (v1 >= 0) & (v1 <= H - 1)
    # occlusion check: the point must be what frame k1 actually sees at that pixel
    ui, vi = np.clip(np.round(u1).astype(int), 0, W - 1), np.clip(np.round(v1).astype(int), 0, H - 1)
    d1s = d1[vi, ui]
    valid &= (d1s > 0) & (np.abs(d1s - z) < 0.03 * z + 0.01)
    return flow.astype(np.float32), valid


def flow_to_color(flow, max_mag=None):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    max_mag = max_mag or np.percentile(mag, 99)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def warp_back(img1, flow):
    H, W = flow.shape[:2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img1, u + flow[..., 0], v + flow[..., 1], cv2.INTER_LINEAR)


def farneback(g0, g1, levels, equalize=True):
    if equalize:                      # Farneback ignores low-contrast texture; stretch contrast first
        g0, g1 = cv2.equalizeHist(g0), cv2.equalizeHist(g1)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, levels, 21, 5, 7, 1.5, 0)


def dis(g0, g1):
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g0, g1, None)


def lucas_kanade(g0, g1, max_level):
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01, minDistance=7)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=max_level,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    return pts.reshape(-1, 2), nxt.reshape(-1, 2), st.ravel().astype(bool)


def evaluate(step):
    """Average EPE over all frame pairs (k, k+step)."""
    res = {"farneback_raw_L4": [], "farneback_L1": [], "farneback_L4": [], "dis": [], "lk_L0": [], "lk_L3": [], "lk_L0_tracked": [],
           "lk_L3_tracked": [], "zero_flow": [], "true_flow_mag": [], "warp_err_true": [], "warp_err_none": []}
    for k in range(0, len(META) - step):
        r0, _ = load(k)
        r1, _ = load(k + step)
        g0, g1 = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
        gt, valid = true_flow(k, k + step)
        res["true_flow_mag"].append(float(np.linalg.norm(gt[valid], axis=1).mean()))
        res["zero_flow"].append(float(np.linalg.norm(gt[valid], axis=1).mean()))
        # sanity check of the ground truth: photometric error after warping
        wb = warp_back(g1.astype(np.float32), gt)
        res["warp_err_true"].append(float(np.abs(wb - g0)[valid].mean()))
        res["warp_err_none"].append(float(np.abs(g1.astype(np.float32) - g0)[valid].mean()))
        epe = lambda fl: float(np.linalg.norm((fl - gt)[valid], axis=1).mean())
        res["farneback_raw_L4"].append(epe(farneback(g0, g1, 4, equalize=False)))
        for L in (1, 4):
            res[f"farneback_L{L}"].append(epe(farneback(g0, g1, L)))
        res["dis"].append(epe(dis(g0, g1)))
        for L in (0, 3):
            p0, p1, ok = lucas_kanade(g0, g1, L)
            xi, yi = np.round(p0[:, 0]).astype(int), np.round(p0[:, 1]).astype(int)
            good = ok & valid[yi, xi]
            err = np.linalg.norm((p1 - p0)[good] - gt[yi[good], xi[good]], axis=1)
            res[f"lk_L{L}"].append(float(np.median(err)))
            res[f"lk_L{L}_tracked"].append(float(np.mean(err < 1.0) * good.mean()))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    results = {}
    for step in (1, 3):
        results[f"step{step}"] = r = evaluate(step)
        print(f"frames {step} apart: mean true motion {r['true_flow_mag']:.2f} px")
        print(f"  ground-truth check - photometric error: no warp {r['warp_err_none']:.2f}, "
              f"warped with true flow {r['warp_err_true']:.2f} gray levels")
        print(f"  mean EPE (px): zero-flow baseline {r['zero_flow']:.2f} | Farneback raw {r['farneback_raw_L4']:.2f} | "
              f"Farneback+equalize 1 level {r['farneback_L1']:.2f}, 4 levels {r['farneback_L4']:.2f} | DIS {r['dis']:.2f}")
        print(f"  LK median EPE (px): no pyramid {r['lk_L0']:.2f}, 3-level pyramid {r['lk_L3']:.2f}")
        print(f"  LK features tracked to <1 px: no pyramid {r['lk_L0_tracked']:.0%}, pyramid {r['lk_L3_tracked']:.0%}")
    log_metrics("optical_flow", results)

    # ---- figure A: one frame pair, true vs estimated flow
    k, step = 18, 1
    r0, _ = load(k); r1, _ = load(k + step)
    g0, g1 = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
    gt, valid = true_flow(k, k + step)
    fb = dis(g0, g1)
    mm = np.percentile(np.linalg.norm(gt[valid], axis=1), 99)
    epe = np.linalg.norm(fb - gt, axis=-1); epe[~valid] = np.nan
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    axs[0].imshow(r0[..., ::-1]); image_axes(axs[0], f"frame {k}")
    g_vis = flow_to_color(gt, mm); g_vis[~valid] = 255
    axs[1].imshow(g_vis); image_axes(axs[1], "true flow (from depth + pose)")
    f_vis = flow_to_color(fb, mm); f_vis[~valid] = 255
    axs[2].imshow(f_vis); image_axes(axs[2], "DIS estimate")
    im = axs[3].imshow(epe, cmap="Blues", vmin=0, vmax=3)
    image_axes(axs[3], "DIS error (px)")
    fig.colorbar(im, ax=axs[3], fraction=0.035)
    fig.suptitle("Dense optical flow while the Go2 walks forward (color = direction, brightness = speed)",
                 x=0.01, ha="left")
    save(fig, "fig03a_dense_flow.png")

    # ---- figure B: LK tracks with and without pyramid on a large-motion pair
    k, step = 18, 3
    r0, _ = load(k); r1, _ = load(k + step)
    g0, g1 = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
    gt, valid = true_flow(k, k + step)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, L, title in zip(axs, (0, 3), ("Lucas-Kanade, no pyramid", "Lucas-Kanade, 3-level pyramid")):
        p0, p1, ok = lucas_kanade(g0, g1, L)
        xi, yi = np.round(p0[:, 0]).astype(int), np.round(p0[:, 1]).astype(int)
        good = ok & valid[yi, xi]
        err = np.linalg.norm((p1 - p0) - gt[yi, xi], axis=1)
        right = good & (err < 1.0)
        ax.imshow(cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB))
        for (a, b), c in [((p0[right], p1[right]), SERIES[0]), ((p0[good & ~right], p1[good & ~right]), SERIES[1])]:
            for (x0, y0), (x1, y1) in zip(a, b):
                ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-", color=c, lw=1.2))
        image_axes(ax, f"{title}: {right.sum()} of {good.sum()} tracks within 1 px of truth")
    axs[1].plot([], [], color=SERIES[0], label="track error < 1 px")
    axs[1].plot([], [], color=SERIES[1], label="track error >= 1 px")
    axs[1].legend(loc="lower right", fontsize=8, facecolor="white", framealpha=0.8)
    fig.suptitle(f"Sparse tracks between frames {k} and {k + step} (larger motion)", x=0.01, ha="left")
    save(fig, "fig03b_lk_tracks.png")

    # ---- figure C: EPE summary bars
    fig, ax = plt.subplots(figsize=(11, 4))
    names = ["zero-flow\nbaseline", "Farneback\nraw", "Farneback+eq\n1 level", "Farneback+eq\n4 levels", "DIS",
             "LK (median)\nno pyramid", "LK (median)\n3 levels"]
    keys = ["zero_flow", "farneback_raw_L4", "farneback_L1", "farneback_L4", "dis", "lk_L0", "lk_L3"]
    x = np.arange(len(keys)); w = 0.38
    for i, step in enumerate((1, 3)):
        vals = [results[f"step{step}"][k_] for k_ in keys]
        bars = ax.bar(x + (i - 0.5) * w, vals, w - 0.03, color=SERIES[i],
                      label=f"frames {step} apart (~{results[f'step{step}']['true_flow_mag']:.0f} px motion)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color=TEXT2)
    ax.set_xticks(x, names, fontsize=9); ax.set_ylabel("end-point error (px), lower is better")
    ax.set_title("Optical flow error vs. exact ground truth", loc="left")
    ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
    save(fig, "fig03c_flow_error.png")


if __name__ == "__main__":
    main()
