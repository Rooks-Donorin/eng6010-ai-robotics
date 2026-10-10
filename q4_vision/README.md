# Question 4 – Vision Methods on Synthetic Go2 Data

This folder generates **synthetic Unitree Go2 data** with the MuJoCo physics simulator, using the real Go2 model from Unitree's [`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco) repo. It then runs the five course topics on that data:

| # | Script | Topic | What it tests |
|---|---|---|---|
| 0 | `00_generate_synthetic_data.py` | Synthetic data | Front-camera video of the Go2 walking down a corridor (with exact depth, masks and camera poses), the same view under 3 lighting conditions, outside views of the robot, and 25,000 rows of Go2 `LowState`-style joint/IMU data for 5 behaviors |
| 1 | `01_color_space.py` | Color space | Which color space keeps a color threshold working when the lighting changes |
| 2 | `02_gaussian_pyramid.py` | Gaussian pyramid | Pyramid levels, aliasing, the Laplacian pyramid, faster search |
| 3 | `03_optical_flow.py` | Optical flow | Farneback, DIS and Lucas–Kanade compared against exact ground-truth flow |
| 4 | `04_grabcut.py` | GrabCut | Segmenting the Go2 and the colored objects from a bounding box, plus interactive refinement |
| 5 | `05_supervised_learning.py` | Supervised learning | Classifying what the robot is doing (standing / slow trot / fast trot / sway / lying) from joint and IMU data |

All figures go to `results/` and all numbers go to `results/metrics.json`.

## How to run (WSL Ubuntu)

You need the `unitree_mujoco` repo for the Go2 model:

```bash
cd ~/eng6010
git clone https://github.com/unitreerobotics/unitree_mujoco.git   # skip if you already have it

cd ~/eng6010/q4_vision
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
sudo apt install -y libosmesa6          # software OpenGL for off-screen rendering

./run_all.sh ~/eng6010/unitree_mujoco/unitree_robots/go2/go2.xml
```

The whole pipeline takes about 1–2 minutes. `run_all.sh` sets `MUJOCO_GL=osmesa`, so it renders without a display or GPU.

## How the synthetic data is made

- **Scene:** `go2_scene.py` loads the Go2 MJCF (`go2.xml` and its meshes) and adds a tiled concrete floor, plaster walls, lights, six colored objects, a forward-facing camera on the robot body at about the Go2's front-camera position, and two outside "observer" cameras. The floor and wall textures are generated procedurally in `textures/`.
- **Camera data:** the robot is moved kinematically down the corridor at 0.4 m/s, with a small trot bob, pitch and heading sway, at 10 fps. For every frame the simulator also gives the **exact depth**, an **object-label mask**, and the **camera pose**. That makes it possible to grade each method against ground truth, which isn't possible with real camera images.
- **Joint data:** a physics simulation at 500 Hz, logged at 50 Hz. A PD controller tracks Unitree's stand and lie poses (from `unitree_mujoco/example/python/stand_go2.py`) plus open-loop gait patterns: diagonal-pair trot at two speeds, and a side-to-side sway. Every episode randomizes gains, pose offsets, gait frequency and amplitude, and adds random shoves and sensor noise. The columns follow the Go2 `LowState` message and the course's `unitree_go2_create_dataset` format: `foot_force`, `q`, `dq`, `ddq` and `tau_est` per motor (SDK motor order), plus the IMU (`rpy`, gyroscope, accelerometer). Foot force is in newtons here; the real robot reports raw sensor counts.

## Results summary

| Topic | Key result |
|---|---|
| Color space | Thresholds tuned under normal light: **HSV** kept IoU 0.99 under **dim** light but dropped to 0.26 under **warm** light. **Lab a\*b\*** was the opposite (0.95 warm, 0.18 dim). **RGB** collapsed to 0.00 in dim light. |
| Gaussian pyramid | At 1/8 size, the pyramid is 3 dB closer to the area-averaged reference than naive subsampling (31.5 vs 28.4 dB PSNR). The Laplacian pyramid reconstructs the image exactly. Template search on level 2 was ~18× faster and found the same location. |
| Optical flow | Against exact ground truth (~12 px motion per frame): DIS 1.2 px error, LK 0.3 px median, Farneback 6.0 px (only after contrast equalization; on raw frames it barely detected motion). For larger motion (~26 px), the pyramid cut LK's median error from 2.0 to 0.6 px. |
| GrabCut | Go2 robot: IoU 0.31 for the box alone → **0.94** with GrabCut; colored objects → 0.98–0.99. Three hint strokes raised the hardest robot image from 0.90 to 0.96. |
| Supervised learning | Random forest **97.2%**, logistic regression 96.7% (5-fold, split by episode). The IMU alone gets 83.1% when split by episode, but a careless random split inflates it to 98.2% (data leakage). |

These are synthetic results. The next step is to repeat the same experiments on real Go2 camera frames and `rt/lowstate` logs.
