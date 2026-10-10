"""
Method 4 - GrabCut segmentation.

GrabCut (Rother et al., 2004) separates foreground from background inside a user-given rectangle,
by alternating between (a) fitting Gaussian mixture color models for foreground and background and
(b) solving a graph cut that balances color fit against smooth boundaries.

Experiments (ground-truth masks come straight from the simulator):
  A. Segment the Go2 robot itself from outside "observer" cameras, 3 poses x 2 views.
     The box given to GrabCut = the robot's true bounding box + 10% margin
     (simulating the output of an object detector).
  B. Segment the colored objects seen by the Go2's own front camera.
  C. Add a few "hint" strokes (sure-foreground pixels) on the hardest robot image and re-run -
     this is GrabCut's interactive refinement step.
Metric: IoU against the true mask. Baseline: calling the whole rectangle "foreground".
"""
import glob, json, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from viz_style import save, log_metrics, image_axes, SERIES, TEXT2

rng = np.random.default_rng(0)


def iou(a, b):
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def box_from_mask(mask, margin=0.10):
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    H, W = mask.shape
    x0, y0 = max(x0 - mx, 0), max(y0 - my, 0)
    x1, y1 = min(x1 + mx, W - 1), min(y1 + my, H - 1)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def grabcut_rect(img, rect, iters=5):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]), mask


def rect_mask(shape, rect):
    m = np.zeros(shape, bool); x, y, w, h = rect; m[y:y + h, x:x + w] = True
    return m


def main():
    # ---------- A: robot from observer views
    rows = []
    for f in sorted(glob.glob("data/observer/rgb_*.png")):
        tag = os.path.basename(f)[4:-4]
        img = cv2.imread(f)
        gt = cv2.imread(f"data/observer/mask_{tag}.png", 0) > 0
        rect = box_from_mask(gt)
        pred, _ = grabcut_rect(img, rect)
        rows.append({"image": tag, "iou_grabcut": float(iou(pred, gt)),
                     "iou_rectangle": float(iou(rect_mask(gt.shape, rect), gt)),
                     "robot_pixels": int(gt.sum()), "img": img, "gt": gt, "pred": pred, "rect": rect})
    print("A) Go2 robot from observer cameras (IoU):")
    for r in rows:
        print(f"   {r['image']:18s} rectangle-only {r['iou_rectangle']:.2f}  ->  GrabCut {r['iou_grabcut']:.2f}")
    mA = np.mean([r["iou_grabcut"] for r in rows]); mR = np.mean([r["iou_rectangle"] for r in rows])
    print(f"   mean: rectangle-only {mR:.2f} -> GrabCut {mA:.2f}")

    # ---------- B: colored objects from the robot's front camera
    labels = json.load(open("data/ego/poses.json"))["labels"]
    img = cv2.imread("data/ego/rgb_015.png"); seg = cv2.imread("data/ego/seg_015.png", 0)
    objrows = []
    for name, l in labels.items():
        gt = seg == l
        if gt.sum() < 400:
            continue
        rect = box_from_mask(gt)
        pred, _ = grabcut_rect(img, rect)
        objrows.append({"object": name, "iou_grabcut": float(iou(pred, gt)),
                        "iou_rectangle": float(iou(rect_mask(gt.shape, rect), gt)), "pred": pred, "rect": rect})
        print(f"B) {name:12s} rectangle-only {objrows[-1]['iou_rectangle']:.2f}  ->  GrabCut {objrows[-1]['iou_grabcut']:.2f}")

    # ---------- C: interactive refinement on the hardest robot image
    worst = min(rows, key=lambda r: r["iou_grabcut"])
    img, gt, rect = worst["img"], worst["gt"], worst["rect"]
    mask = np.zeros(gt.shape, np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    # simulate a user drawing 3 short strokes on parts of the robot GrabCut missed
    missed = gt & ~np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    ys, xs = np.where(missed if missed.sum() > 50 else gt)
    picks = rng.choice(len(ys), size=min(3, len(ys)), replace=False)
    strokes = np.zeros(gt.shape, np.uint8)
    for p in picks:
        cv2.circle(strokes, (int(xs[p]), int(ys[p])), 4, 1, -1)
    strokes &= gt.astype(np.uint8)                     # a careful user only marks robot pixels
    mask[strokes > 0] = cv2.GC_FGD
    cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    refined = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    iou_ref = float(iou(refined, gt))
    print(f"C) {worst['image']}: GrabCut {worst['iou_grabcut']:.2f} -> with 3 hint strokes {iou_ref:.2f}")

    log_metrics("grabcut", {
        "robot": [{k: r[k] for k in ("image", "iou_rectangle", "iou_grabcut", "robot_pixels")} for r in rows],
        "robot_mean": {"rectangle": float(mR), "grabcut": float(mA)},
        "objects": [{k: r[k] for k in ("object", "iou_rectangle", "iou_grabcut")} for r in objrows],
        "refinement": {"image": worst["image"], "before": worst["iou_grabcut"], "after_3_strokes": iou_ref}})

    # ---------- figures
    def overlay(img, m, color=(42, 120, 214)):
        out = img[..., ::-1].copy().astype(np.float32)
        out[m] = 0.45 * out[m] + 0.55 * np.array(color)
        return out.astype(np.uint8)

    fig, axs = plt.subplots(2, 3, figsize=(14, 6.4))
    for ax, r in zip(axs.flat, rows):
        ax.imshow(overlay(r["img"], r["pred"]))
        x, y, w, h = r["rect"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=SERIES[1], lw=1.5))
        image_axes(ax, f"{r['image']}: IoU {r['iou_grabcut']:.2f}")
    fig.suptitle("GrabCut on the Go2 (orange = input box, blue = GrabCut foreground)", x=0.01, ha="left")
    save(fig, "fig04a_grabcut_robot.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    allm = np.zeros(seg.shape, bool)
    for r in objrows:
        allm |= r["pred"]
    ego = cv2.imread("data/ego/rgb_015.png")
    axs[0].imshow(overlay(ego, allm))
    for r in objrows:
        x, y, w, h = r["rect"]; axs[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=SERIES[1], lw=1.2))
    image_axes(axs[0], "front camera: GrabCut on each object box")
    names = [r["object"].replace("_", " ") for r in objrows]
    xx = np.arange(len(objrows)); w = 0.38
    for i, (key, lab) in enumerate([("iou_rectangle", "box only"), ("iou_grabcut", "GrabCut")]):
        vals = [r[key] for r in objrows]
        axs[1].bar(xx + (i - 0.5) * w, vals, w - 0.03, color=SERIES[i], label=lab)
    axs[1].set_xticks(xx, names, rotation=20); axs[1].set_ylim(0, 1.05); axs[1].set_ylabel("IoU vs. ground truth")
    axs[1].legend(loc="lower right"); axs[1].grid(axis="x", visible=False)
    axs[1].set_title("Colored objects: box vs. GrabCut", loc="left")
    save(fig, "fig04b_grabcut_objects.png")

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    axs[0].imshow(overlay(img, worst["pred"])); image_axes(axs[0], f"GrabCut from box: IoU {worst['iou_grabcut']:.2f}")
    vis = overlay(img, refined)
    axs[1].imshow(vis)
    sy, sx = np.where(strokes > 0); axs[1].scatter(sx, sy, s=2, c=SERIES[1])
    image_axes(axs[1], f"+ 3 hint strokes (orange): IoU {iou_ref:.2f}")
    axs[2].imshow(gt, cmap="gray"); image_axes(axs[2], "ground-truth robot mask")
    fig.suptitle("Interactive refinement - the step that makes GrabCut practical", x=0.01, ha="left")
    save(fig, "fig04c_grabcut_refine.png")


if __name__ == "__main__":
    main()
