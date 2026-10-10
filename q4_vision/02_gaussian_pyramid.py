"""
Method 2 - Gaussian pyramid (and the Laplacian pyramid built from it).

Experiments on a synthetic Go2 front-camera frame:
  1. Build a 4-level Gaussian pyramid (blur + downsample by 2 each level) and show it.
  2. Why blur before shrinking? Compare level 3 (1/8 size) of the pyramid against naive
     "take every 8th pixel" subsampling. Reference = area-averaged resize (cv2.INTER_AREA).
     Measure PSNR: the pyramid should be much closer to the reference (less aliasing).
  3. Build the Laplacian pyramid (band-pass details) and reconstruct the original image from it.
     Reconstruction error should be ~0 - the Laplacian pyramid is a lossless representation.
  4. Speed benefit: time a template search (find the yellow box) at full size vs. on level 2.
The pyramid is also what lets optical flow handle large motions - see 03_optical_flow.py.
"""
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from viz_style import save, log_metrics, image_axes, SERIES, TEXT2

LEVELS = 4


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def main():
    img = cv2.imread("data/ego/rgb_015.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Gaussian pyramid
    gp = [img]
    for _ in range(LEVELS - 1):
        gp.append(cv2.pyrDown(gp[-1]))
    fig = plt.figure(figsize=(12, 4.2))
    widths = [g.shape[1] for g in gp]
    x0 = 0.0
    for i, g in enumerate(gp):
        w = widths[i] / sum(widths) * 0.98
        ax = fig.add_axes([x0, 0.05 + (0.85 - 0.85 * g.shape[0] / gp[0].shape[0]) / 2, w, 0.85 * g.shape[0] / gp[0].shape[0]])
        ax.imshow(g[..., ::-1]); image_axes(ax, f"level {i}: {g.shape[1]}x{g.shape[0]}")
        x0 += w + 0.005
    fig.suptitle("Gaussian pyramid of a synthetic Go2 front-camera frame", x=0.01, ha="left", y=1.02)
    save(fig, "fig02a_gaussian_pyramid.png")

    # 2) aliasing: pyramid vs. naive subsampling at 1/8 size (on a texture-rich crop: floor + wall)
    f = 2 ** (LEVELS - 1)
    ref = cv2.resize(gray, (gray.shape[1] // f, gray.shape[0] // f), interpolation=cv2.INTER_AREA)
    g = gray.copy()
    for _ in range(LEVELS - 1):
        g = cv2.pyrDown(g)
    naive = gray[::f, ::f][:ref.shape[0], :ref.shape[1]]
    g = g[:ref.shape[0], :ref.shape[1]]
    p_pyr, p_naive = psnr(g, ref), psnr(naive, ref)
    print(f"1/{f}-size image vs. area-averaged reference: pyramid PSNR {p_pyr:.1f} dB, naive subsampling PSNR {p_naive:.1f} dB")

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, im, t in zip(axs, [ref, g, naive], ["reference (area average)", f"Gaussian pyramid level 3 ({p_pyr:.1f} dB)",
                                                f"naive every-{f}th-pixel ({p_naive:.1f} dB)"]):
        ax.imshow(cv2.resize(im, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_NEAREST), cmap="gray")
        image_axes(ax, t)
    fig.suptitle("Blurring before shrinking avoids aliasing (images enlarged so pixels are visible)", x=0.01, ha="left")
    save(fig, "fig02b_aliasing.png")

    # 3) Laplacian pyramid + reconstruction
    gpf = [gray.astype(np.float32)]
    for _ in range(LEVELS - 1):
        gpf.append(cv2.pyrDown(gpf[-1]))
    lp = []
    for i in range(LEVELS - 1):
        up = cv2.pyrUp(gpf[i + 1], dstsize=(gpf[i].shape[1], gpf[i].shape[0]))
        lp.append(gpf[i] - up)
    lp.append(gpf[-1])
    rec = lp[-1]
    for i in range(LEVELS - 2, -1, -1):
        rec = cv2.pyrUp(rec, dstsize=(lp[i].shape[1], lp[i].shape[0])) + lp[i]
    rec_err = float(np.abs(rec - gpf[0]).max())
    print(f"Laplacian pyramid reconstruction: max abs error {rec_err:.2e} (gray levels)")

    fig, axs = plt.subplots(1, LEVELS, figsize=(13, 3))
    for i, ax in enumerate(axs):
        im = lp[i]
        ax.imshow(im if i == LEVELS - 1 else np.clip(im * 3 + 128, 0, 255), cmap="gray", vmin=0, vmax=255)
        image_axes(ax, f"L{i}" + (" (low-pass residual)" if i == LEVELS - 1 else " (detail band, x3)"))
    fig.suptitle(f"Laplacian pyramid - reconstructs the original with max error {rec_err:.1e}", x=0.01, ha="left")
    save(fig, "fig02c_laplacian_pyramid.png")

    # 4) speed: template matching at full resolution vs. pyramid level 2
    seg = cv2.imread("data/ego/seg_015.png", 0)
    ys, xs = np.where(seg == 4)                       # 4 = yellow_box
    y0, y1, x0b, x1b = ys.min(), ys.max(), xs.min(), xs.max()
    tmpl_src = cv2.imread("data/ego/rgb_012.png", 0)  # template cut from an EARLIER frame
    seg12 = cv2.imread("data/ego/seg_012.png", 0)
    ty, tx = np.where(seg12 == 4)
    tmpl = tmpl_src[ty.min():ty.max(), tx.min():tx.max()]
    timings = {}
    for lvl in [0, 2]:
        im_l, t_l = gray.copy(), tmpl.copy()
        for _ in range(lvl):
            im_l, t_l = cv2.pyrDown(im_l), cv2.pyrDown(t_l)
        t0 = time.perf_counter()
        for _ in range(20):
            res = cv2.matchTemplate(im_l, t_l, cv2.TM_CCOEFF_NORMED)
        dt = (time.perf_counter() - t0) / 20 * 1000
        _, score, _, loc = cv2.minMaxLoc(res)
        timings[lvl] = {"ms": dt, "found_xy_fullres": [loc[0] * 2 ** lvl, loc[1] * 2 ** lvl], "score": score}
    truth = [int(x0b), int(y0)]
    print(f"Template search: level 0 {timings[0]['ms']:.2f} ms, level 2 {timings[2]['ms']:.2f} ms "
          f"({timings[0]['ms'] / timings[2]['ms']:.0f}x faster); true top-left {truth}, "
          f"found L0 {timings[0]['found_xy_fullres']}, L2 {timings[2]['found_xy_fullres']}")

    log_metrics("gaussian_pyramid", {"psnr_pyramid_db": p_pyr, "psnr_naive_db": p_naive,
                                     "laplacian_reconstruction_max_abs_error": rec_err,
                                     "template_search": timings, "template_true_xy": truth})


if __name__ == "__main__":
    main()
