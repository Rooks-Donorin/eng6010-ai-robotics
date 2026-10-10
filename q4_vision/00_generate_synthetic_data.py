"""
Step 0 - Generate synthetic Go2 data with MuJoCo.

Outputs (in data/):
  ego/        front-camera image sequence of the Go2 walking down a corridor
              rgb_XXX.png, depth_XXX.png (uint16, millimeters), seg_XXX.png (object labels),
              poses.json (camera pose + intrinsics for every frame -> exact ground-truth optical flow)
  lighting/   the same front-camera view under three lighting conditions (for the color-space test)
  observer/   outside views of the robot + ground-truth robot masks (for GrabCut)
  joint_states.csv.gz
              Go2 "LowState"-style data (foot_force, motor_q/dq/ddq/tau_est, IMU) from a
              physics simulation of 5 behaviors, labeled for supervised learning.

Usage:
  python 00_generate_synthetic_data.py --go2-xml ~/eng6010/unitree_mujoco/unitree_robots/go2/go2.xml
If rendering fails on a machine without a display, run with:  MUJOCO_GL=egl  (or osmesa)
"""
import argparse, json, os
import numpy as np
import cv2
import mujoco
import pandas as pd

from go2_scene import (build_model, joint_indices, robot_geom_ids, set_pose, quat_to_rpy,
                       OBJECTS, SDK_JOINTS, FEET, STAND_UP, STAND_DOWN, DEFAULT_GO2_XML)

W, H = 640, 480
LABELS = {name: i + 1 for i, name in enumerate(OBJECTS)}   # 1..6 = colored objects
ROBOT_LABEL = 7


# --------------------------------------------------------------------------- rendering helpers
def label_map_from_seg(model, seg, robot_ids):
    """Renderer segmentation output (geom id per pixel) -> small integer label image."""
    gid, gtype = seg[..., 0], seg[..., 1]
    lab = np.zeros(gid.shape, np.uint8)
    is_geom = gtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    for name, l in LABELS.items():
        lab[is_geom & (gid == model.geom(name).id)] = l
    lab[is_geom & np.isin(gid, robot_ids)] = ROBOT_LABEL
    return lab


def render_all(model, data, renderer, cam, robot_ids):
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return rgb, depth, label_map_from_seg(model, seg, robot_ids)


def cam_info(model, data, cam):
    cid = model.camera(cam).id
    fovy = float(model.cam_fovy[cid])
    f = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
    return {"pos": data.cam_xpos[cid].tolist(), "R": data.cam_xmat[cid].reshape(3, 3).tolist(),
            "f": f, "cx": (W - 1) / 2, "cy": (H - 1) / 2, "fovy": fovy}


def save_depth_mm(path, depth):
    d = np.where(depth > 40, 0, depth)          # far plane / sky -> 0 = "no depth"
    cv2.imwrite(path, np.clip(d * 1000, 0, 65535).astype(np.uint16))


# --------------------------------------------------------------------------- 1) ego-motion sequence
def make_ego_sequence(model, out, n_frames=30, fps=10.0):
    os.makedirs(out, exist_ok=True)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, H, W)
    robot_ids = robot_geom_ids(model)
    poses = []
    speed = 0.4                                   # m/s, a slow Go2 walk
    for k in range(n_frames):
        t = k / fps
        x = speed * t
        yaw = np.deg2rad(3.0) * np.sin(2 * np.pi * 0.5 * t)          # gentle heading sway
        z = 0.32 + 0.008 * np.sin(2 * np.pi * 2.0 * t)               # trot "bob"
        pitch = np.deg2rad(1.0) * np.sin(2 * np.pi * 2.0 * t + 1.0)
        set_pose(model, data, x=x, y=0.0, z=z, yaw=yaw, pitch=pitch, joints=STAND_UP)
        rgb, depth, lab = render_all(model, data, r, "front_cam", robot_ids)
        cv2.imwrite(f"{out}/rgb_{k:03d}.png", rgb[:, :, ::-1])
        save_depth_mm(f"{out}/depth_{k:03d}.png", depth)
        cv2.imwrite(f"{out}/seg_{k:03d}.png", lab)
        poses.append({"frame": k, "t": t, "base_xyz_yaw_pitch": [x, 0.0, z, yaw, pitch],
                      **cam_info(model, data, "front_cam")})
    with open(f"{out}/poses.json", "w") as fh:
        json.dump({"labels": LABELS, "robot_label": ROBOT_LABEL, "frames": poses}, fh, indent=1)
    r.close()
    print(f"  ego sequence: {n_frames} frames -> {out}")


# --------------------------------------------------------------------------- 2) lighting variants
def make_lighting_set(model, out):
    os.makedirs(out, exist_ok=True)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, H, W)
    robot_ids = robot_geom_ids(model)
    set_pose(model, data, x=0.4, joints=STAND_UP)
    base_light = model.light_diffuse.copy()
    base_hd, base_ha = model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy()
    conditions = {"normal": (1.0, [1, 1, 1]), "dim": (0.35, [1, 1, 1]), "warm": (1.25, [1.0, 0.85, 0.6])}
    for name, (gain, tint) in conditions.items():
        tint = np.array(tint)
        model.light_diffuse[:] = base_light * gain * tint
        model.vis.headlight.diffuse[:] = base_hd * gain * tint
        model.vis.headlight.ambient[:] = base_ha * gain * tint
        rgb, depth, lab = render_all(model, data, r, "front_cam", robot_ids)
        cv2.imwrite(f"{out}/rgb_{name}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"{out}/seg_{name}.png", lab)
    model.light_diffuse[:] = base_light
    model.vis.headlight.diffuse[:] = base_hd
    model.vis.headlight.ambient[:] = base_ha
    r.close()
    print(f"  lighting set: {list(conditions)} -> {out}")


# --------------------------------------------------------------------------- 3) observer views
def make_observer_set(model, out):
    os.makedirs(out, exist_ok=True)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, H, W)
    robot_ids = robot_geom_ids(model)
    poses = {"stand": (STAND_UP, 0.32, 0.0), "half": (0.5 * (STAND_UP + STAND_DOWN), 0.24, 0.4),
             "lying": (STAND_DOWN, 0.13, -0.5)}
    for pname, (joints, z, yaw) in poses.items():
        set_pose(model, data, x=0.0, z=z, yaw=yaw, joints=joints)
        for cam in ["observer0", "observer1"]:
            rgb, depth, lab = render_all(model, data, r, cam, robot_ids)
            cv2.imwrite(f"{out}/rgb_{pname}_{cam}.png", rgb[:, :, ::-1])
            cv2.imwrite(f"{out}/mask_{pname}_{cam}.png", (lab == ROBOT_LABEL).astype(np.uint8) * 255)
    r.close()
    print(f"  observer set: {len(poses) * 2} images -> {out}")


# --------------------------------------------------------------------------- 4) joint-state dataset
BEHAVIORS = ["standing", "trot_slow", "trot_fast", "sway", "lying"]


def foot_forces(model, data, foot_ids):
    """Normal contact force (N) on each foot, SDK order FR, FL, RR, RL."""
    f = np.zeros(4)
    c6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        for j, gid in enumerate(foot_ids):
            if con.geom1 == gid or con.geom2 == gid:
                mujoco.mj_contactForce(model, data, i, c6)
                f[j] += abs(c6[0])
    return f


def simulate_episode(model, data, behavior, rng, seconds=5.0, log_hz=50):
    qadr, vadr, act = joint_indices(model)
    foot_ids = [model.geom(n).id for n in FEET]
    lo, hi = model.actuator_ctrlrange[act, 0], model.actuator_ctrlrange[act, 1]
    mujoco.mj_resetData(model, data)
    nominal = STAND_DOWN if behavior == "lying" else STAND_UP
    set_pose(model, data, z=0.15 if behavior == "lying" else 0.33, joints=nominal)

    # randomize every episode so the classifier can't just memorize one trajectory
    kp, kd = rng.uniform(45, 75), rng.uniform(2.0, 4.0)
    pose_offset = rng.normal(0, 0.04, 12)
    freq = {"trot_slow": rng.uniform(1.2, 2.0), "trot_fast": rng.uniform(2.4, 3.4)}.get(behavior, 0)
    lift_amp = rng.uniform(0.15, 0.30)
    sway_amp, sway_f = rng.uniform(0.08, 0.18), rng.uniform(0.5, 1.0)
    push_times = rng.uniform(0.5, seconds - 0.5, size=2)
    base_body = model.body("base_link").id

    dt = model.opt.timestep
    n_steps = int(seconds / dt)
    every = int(round(1 / (log_hz * dt)))
    rows = []
    for k in range(n_steps):
        t = k * dt
        tgt = nominal + pose_offset
        if behavior.startswith("trot"):
            ph = 2 * np.pi * freq * t
            for leg, off in enumerate([0, np.pi, np.pi, 0]):        # diagonal pairs: FR+RL, FL+RR
                lift = lift_amp * max(0.0, np.sin(ph + off))
                tgt[3 * leg + 1] += lift
                tgt[3 * leg + 2] -= 2 * lift
        elif behavior == "sway":
            s = sway_amp * np.sin(2 * np.pi * sway_f * t)
            tgt[[0, 6]] += s                                         # right legs' hip abduction
            tgt[[3, 9]] += s                                         # left legs' hip abduction
        # random sideways/forward shoves on the body (like someone bumping the robot)
        data.xfrc_applied[base_body, :3] = 0
        for pt in push_times:
            if pt <= t < pt + 0.1:
                data.xfrc_applied[base_body, :2] = rng.normal(0, 25, 2)
        q, dq = data.qpos[qadr], data.qvel[vadr]
        data.ctrl[act] = np.clip(kp * (tgt - q) - kd * dq, lo, hi)
        mujoco.mj_step(model, data)

        if k % every == 0:
            R = data.xmat[base_body].reshape(3, 3)
            acc_world = data.qacc[0:3] - model.opt.gravity               # specific force
            row = {"t": t, "label": behavior}
            ff = foot_forces(model, data, foot_ids)
            for j, n in enumerate(FEET):
                row[f"foot_force_{n}"] = ff[j]
            for j, n in enumerate(SDK_JOINTS):
                jn = n.replace("_joint", "")
                row[f"q_{jn}"] = data.qpos[qadr[j]]
                row[f"dq_{jn}"] = data.qvel[vadr[j]]
                row[f"ddq_{jn}"] = data.qacc[vadr[j]]
                row[f"tau_est_{jn}"] = data.actuator_force[act[j]]
            rpy = quat_to_rpy(data.qpos[3:7])
            gyro = data.qvel[3:6]                     # free-joint angular velocity is in the body frame
            acc = R.T @ acc_world
            for a, v in zip("xyz", gyro):
                row[f"imu_gyro_{a}"] = v
            for a, v in zip("xyz", acc):
                row[f"imu_acc_{a}"] = v
            for a, v in zip(["roll", "pitch", "yaw"], rpy):
                row[f"imu_{a}"] = v
            rows.append(row)
    return rows


def add_sensor_noise(df, rng):
    """Add realistic-ish sensor noise to the clean simulator values."""
    noise = {"foot_force_": 3.0, "q_": 0.002, "dq_": 0.05, "ddq_": 2.0, "tau_est_": 0.2,
             "imu_gyro_": 0.01, "imu_acc_": 0.05, "imu_roll": 0.002, "imu_pitch": 0.002, "imu_yaw": 0.002}
    for prefix, s in noise.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        df[cols] = df[cols] + rng.normal(0, s, (len(df), len(cols)))
    return df


def make_joint_dataset(go2_xml, out_csv, episodes_per_class=20, seconds=5.0, seed=0):
    model = build_model(go2_xml, add_objects=False)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    all_rows = []
    ep = 0
    for behavior in BEHAVIORS:
        for _ in range(episodes_per_class):
            rows = simulate_episode(model, data, behavior, rng, seconds=seconds)
            for r_ in rows:
                r_["episode"] = ep
            all_rows += rows
            ep += 1
    df = pd.DataFrame(all_rows)
    df = add_sensor_noise(df, rng)
    cols = ["episode", "t", "label"] + [c for c in df.columns if c not in ("episode", "t", "label")]
    df[cols].to_csv(out_csv, index=False, float_format="%.5g")
    print(f"  joint dataset: {len(df)} rows, {ep} episodes, {len(BEHAVIORS)} classes -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go2-xml", default=DEFAULT_GO2_XML)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    model = build_model(args.go2_xml)
    print("Generating synthetic data...")
    make_ego_sequence(model, f"{args.out}/ego")
    make_lighting_set(model, f"{args.out}/lighting")
    make_observer_set(model, f"{args.out}/observer")
    make_joint_dataset(args.go2_xml, f"{args.out}/joint_states.csv.gz")
    print("Done.")
