"""
Method 5 - Supervised learning: recognize what the Go2 is doing from its joint/IMU data.

Data: data/joint_states.csv.gz - synthetic Go2 "LowState" data from a MuJoCo physics simulation
      (same fields the unitree_go2_create_dataset tool records: foot_force, motor_q, motor_dq,
      motor_ddq, motor_tau_est - plus the IMU). 5 labeled behaviors x 20 episodes x 5 s @ 50 Hz:
      standing, trot_slow, trot_fast, sway, lying.

Pipeline:
  1. Cut each episode into 1-second windows (50 samples, 50% overlap).
  2. Features per window: mean, std, min, max of every channel + dominant frequency of the
     thigh joints and foot forces (trotting is periodic, so frequency is informative).
  3. Train/test split BY EPISODE (GroupKFold, 5 folds) - windows from the same episode never
     appear in both train and test. I also show what happens with a careless random split.
  4. Models: logistic regression and random forest. Feature-set comparison: which sensors matter?
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from viz_style import save, log_metrics, SERIES, TEXT2, TEXT

WIN, HOP, FS = 50, 25, 50.0
CLASSES = ["standing", "trot_slow", "trot_fast", "sway", "lying"]


def dominant_freq(x):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return freqs[1:][np.argmax(spec[1:])] if spec[1:].max() > 1e-9 else 0.0


def make_windows(df):
    chans = [c for c in df.columns if c not in ("episode", "t", "label")]
    periodic = [c for c in chans if c.startswith("q_") and "thigh" in c] + [c for c in chans if c.startswith("foot_force")]
    X, y, groups = [], [], []
    for ep, g in df.groupby("episode"):
        a = g[chans].to_numpy()
        for s in range(0, len(g) - WIN + 1, HOP):
            w = a[s:s + WIN]
            feats = np.concatenate([w.mean(0), w.std(0), w.min(0), w.max(0),
                                    [dominant_freq(g[c].to_numpy()[s:s + WIN]) for c in periodic]])
            X.append(feats); y.append(g["label"].iloc[0]); groups.append(ep)
    names = ([f"mean_{c}" for c in chans] + [f"std_{c}" for c in chans] + [f"min_{c}" for c in chans] +
             [f"max_{c}" for c in chans] + [f"freq_{c}" for c in periodic])
    return np.array(X), np.array(y), np.array(groups), names


def feature_subset(names, prefixes):
    return [i for i, n in enumerate(names) if n.split("_", 1)[1].startswith(prefixes)]


def main():
    df = pd.read_csv("data/joint_states.csv.gz")
    X, y, groups, names = make_windows(df)
    print(f"{len(df)} samples -> {len(X)} windows of 1 s, {X.shape[1]} features, {len(set(groups))} episodes")

    gkf = GroupKFold(n_splits=5)
    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "random forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    }
    results = {}
    preds = {}
    for name, mdl in models.items():
        p = cross_val_predict(mdl, X, y, groups=groups, cv=gkf)
        preds[name] = p
        results[name] = {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}
        print(f"  {name:20s} accuracy {results[name]['accuracy']:.3f}  macro-F1 {results[name]['macro_f1']:.3f}  (split by episode)")

    # which sensors carry the information? (random forest, split by episode)
    subsets = {
        "joint angles only (q)": ("q_",),
        "foot forces only": ("foot_force",),
        "IMU only": ("imu_",),
        "joint torques only": ("tau_est",),
        "all sensors": ("",),
    }
    sub_res = {}
    for sname, pref in subsets.items():
        idx = feature_subset(names, pref) if pref != ("",) else list(range(X.shape[1]))
        p = cross_val_predict(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                              X[:, idx], y, groups=groups, cv=gkf)
        sub_res[sname] = float(accuracy_score(y, p))
        print(f"  RF with {sname:24s}: accuracy {sub_res[sname]:.3f}")

    # careless random split (windows from the same episode leak into train and test)
    p_leak = cross_val_predict(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                               X[:, feature_subset(names, ("imu_",))], y,
                               cv=KFold(5, shuffle=True, random_state=0))
    leak_acc = float(accuracy_score(y, p_leak))
    print(f"  RF IMU-only with a RANDOM window split: {leak_acc:.3f} (vs {sub_res['IMU only']:.3f} split by episode)")

    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X, y)
    top = np.argsort(rf.feature_importances_)[::-1][:12]
    cm = confusion_matrix(y, preds["random forest"], labels=CLASSES)

    log_metrics("supervised_learning", {
        "n_windows": int(len(X)), "n_features": int(X.shape[1]), "models_split_by_episode": results,
        "rf_feature_subsets": sub_res, "rf_imu_random_split_accuracy": leak_acc,
        "confusion_matrix_rf": {"labels": CLASSES, "matrix": cm.tolist()},
        "top_features": [names[i] for i in top]})

    # ---- figure A: confusion matrix (single-hue sequential)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.6 else TEXT)
    ax.set_xticks(range(5), CLASSES, rotation=30); ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    ax.set_title(f"Random forest, split by episode (accuracy {results['random forest']['accuracy']:.1%})", loc="left")
    save(fig, "fig05a_confusion_matrix.png")

    # ---- figure B: accuracy by sensor subset (single series -> one color)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ks = list(sub_res)
    vals = [sub_res[k] for k in ks]
    ax.barh(range(len(ks)), vals, color=SERIES[0], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9, color=TEXT2)
    ax.set_yticks(range(len(ks)), ks); ax.invert_yaxis(); ax.set_xlim(0, 1.12)
    ax.set_xlabel("accuracy (5-fold, split by episode)"); ax.grid(axis="y", visible=False)
    ax.set_title("Which Go2 sensors tell you what the robot is doing?", loc="left")
    save(fig, "fig05b_sensor_subsets.png")

    # ---- figure C: example signals of each behavior (one thigh joint + one foot force), small multiples
    fig, axs = plt.subplots(2, 5, figsize=(15, 4.4), sharey="row", sharex=True)
    for j, c in enumerate(CLASSES):
        ep = df[df.label == c].episode.iloc[0]
        g = df[df.episode == ep]
        g = g[g.t < 2.0]
        axs[0, j].plot(g.t, g.q_FR_thigh, color=SERIES[0], lw=1.5)
        axs[1, j].plot(g.t, g.foot_force_FR, color=SERIES[0], lw=1.5)
        axs[0, j].set_title(c, loc="left")
        axs[1, j].set_xlabel("time (s)")
    axs[0, 0].set_ylabel("FR thigh angle (rad)"); axs[1, 0].set_ylabel("FR foot force (N)")
    fig.suptitle("Synthetic Go2 joint data: first 2 s of one episode per behavior", x=0.01, ha="left")
    save(fig, "fig05c_example_signals.png")


if __name__ == "__main__":
    main()
