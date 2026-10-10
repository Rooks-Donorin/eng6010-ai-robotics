"""
Builds a synthetic "lab" scene around the real Unitree Go2 MuJoCo model.

The Go2 model (go2.xml + meshes) comes from Unitree's unitree_mujoco repo:
    https://github.com/unitreerobotics/unitree_mujoco  (unitree_robots/go2/go2.xml)

We read go2.xml and add (as plain MJCF):
  * a tiled floor, walls, lights
  * colored objects (boxes, a cylinder, a sphere) that the vision methods can find
  * a forward-facing "front camera" on the robot body (approximate Go2 camera location)
  * two observer cameras that look at the robot from outside (used for GrabCut)
"""
import os
import re
import numpy as np
import cv2
import mujoco

# Default location if you cloned unitree_mujoco next to the SDK, as in my setup.
DEFAULT_GO2_XML = os.path.expanduser("~/eng6010/unitree_mujoco/unitree_robots/go2/go2.xml")

# Joint order used by the Unitree SDK (LowCmd / LowState motor index 0..11)
SDK_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
FEET = ["FR", "FL", "RR", "RL"]  # foot geom names, same order as SDK foot_force[4]

# Poses from unitree_mujoco/example/python/stand_go2.py (SDK joint order)
STAND_UP = np.array([0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
                     0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763])
STAND_DOWN = np.array([0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
                       0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375])

# Colored objects in the scene: name -> (geom type, size, position, RGBA)
OBJECTS = {
    "red_box":      ("box",      [0.20, 0.20, 0.20], [2.2,  0.55, 0.20], [0.85, 0.10, 0.10, 1]),
    "green_box":    ("box",      [0.15, 0.25, 0.30], [3.0, -0.70, 0.30], [0.10, 0.65, 0.20, 1]),
    "blue_box":     ("box",      [0.25, 0.15, 0.15], [3.6,  0.90, 0.15], [0.10, 0.25, 0.85, 1]),
    "yellow_box":   ("box",      [0.12, 0.12, 0.12], [1.6, -0.45, 0.12], [0.95, 0.85, 0.10, 1]),
    "orange_cyl":   ("cylinder", [0.10, 0.25, 0],    [2.7,  0.05, 0.25], [1.00, 0.50, 0.05, 1]),
    "purple_ball":  ("sphere",   [0.15, 0, 0],       [4.2, -0.15, 0.15], [0.55, 0.20, 0.70, 1]),
}


TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def _noise(size, rng, octaves=(4, 16, 64, 256), weights=(0.45, 0.3, 0.15, 0.1)):
    """Multi-scale smooth noise in [0, 1] (sum of upsampled random grids)."""
    out = np.zeros((size, size), np.float32)
    for o, w in zip(octaves, weights):
        g = rng.random((o, o)).astype(np.float32)
        out += w * cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
    out -= out.min(); out /= out.max()
    return out


def make_textures(size=1024, seed=7):
    """Procedural concrete-tile floor and plaster wall textures (so images have realistic texture)."""
    os.makedirs(TEX_DIR, exist_ok=True)
    floor_png, wall_png = os.path.join(TEX_DIR, "floor.png"), os.path.join(TEX_DIR, "wall.png")
    if os.path.exists(floor_png) and os.path.exists(wall_png):
        return floor_png, wall_png
    rng = np.random.default_rng(seed)
    # floor: 1 m texture with 2x2 polished-concrete tiles, blotchy noise + fine speckle + grout lines
    n = _noise(size, rng)
    speck = rng.random((size, size)).astype(np.float32)
    v = 0.50 + 0.18 * (n - 0.5) + 0.06 * (speck - 0.5)
    tile = (np.arange(size) // (size // 2))
    v += 0.03 * ((tile[:, None] + tile[None, :]) % 2)              # slight tile-to-tile shade change
    grout = (np.arange(size) % (size // 2) < 4)
    v[grout, :] = 0.33; v[:, grout] = 0.33
    floor = np.stack([v * 0.98, v * 0.98, v * 0.95], -1)
    cv2.imwrite(floor_png, np.clip(floor * 255, 0, 255).astype(np.uint8)[..., ::-1])
    # wall: 2 m texture, light plaster with soft stains and fine grain, plus a dark baseboard band
    n = _noise(size, rng, octaves=(3, 12, 48, 192))
    grain = rng.random((size, size)).astype(np.float32)
    w = 0.80 + 0.12 * (n - 0.5) + 0.05 * (grain - 0.5)
    wall = np.stack([w, w * 0.98, w * 0.93], -1)
    cv2.imwrite(wall_png, np.clip(wall * 255, 0, 255).astype(np.uint8)[..., ::-1])
    return floor_png, wall_png


def _cam_quat_looking(forward, up=(0, 0, 1)):
    """Quaternion (w x y z) for a MuJoCo camera (it looks along its -z axis, +y is up)."""
    f = np.asarray(forward, float); f /= np.linalg.norm(f)
    x = np.cross(f, up); x /= np.linalg.norm(x)        # camera right
    y = np.cross(x, f)                                   # camera up
    R = np.column_stack([x, y, -f])                      # columns = camera axes in parent frame
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _fmt(v):
    return " ".join(f"{float(a):.6g}" for a in v)


def build_model(go2_xml=DEFAULT_GO2_XML, add_objects=True):
    """Returns a compiled MjModel: the Go2 + floor, walls, lights, objects and cameras.

    Built with plain MJCF text (not the MjSpec API) so it works across MuJoCo versions.
    """
    if not os.path.exists(go2_xml):
        raise FileNotFoundError(
            f"Can't find {go2_xml}. Clone https://github.com/unitreerobotics/unitree_mujoco "
            "and pass --go2-xml /path/to/unitree_mujoco/unitree_robots/go2/go2.xml")
    xml = open(go2_xml).read()
    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(go2_xml)), "assets")
    xml = xml.replace('meshdir="assets"', f'meshdir="{mesh_dir}"')

    floor_png, wall_png = make_textures()
    extra_head = f"""
  <visual>
    <global offwidth="1280" offheight="960"/>
    <map zfar="50"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="tiles" type="2d" file="{floor_png}"/>
    <material name="floor_mat" texture="tiles" texrepeat="1 1" texuniform="true" reflectance="0.02"/>
    <texture name="wall_tex" type="2d" file="{wall_png}"/>
    <material name="wall_mat" texture="wall_tex" texrepeat="0.5 0.5" texuniform="true"/>
    <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.92" rgb2="0.75 0.75 0.8" width="256" height="1536"/>
  </asset>
"""
    world = """
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <light pos="3 0 3" dir="0 0 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"/>
    <geom name="wall_back"  type="box" size="0.05 2.5 1.2" pos="5.5 0 1.2"  material="wall_mat"/>
    <geom name="wall_left"  type="box" size="3.5 0.05 1.2" pos="2.0 1.6 1.2"  material="wall_mat"/>
    <geom name="wall_right" type="box" size="3.5 0.05 1.2" pos="2.0 -1.6 1.2" material="wall_mat"/>
"""
    if add_objects:
        for name, (gtype, size, pos, rgba) in OBJECTS.items():
            sz = [s for s in size if s > 0]
            world += f'    <geom name="{name}" type="{gtype}" size="{_fmt(sz)}" pos="{_fmt(pos)}" rgba="{_fmt(rgba)}"/>\n'
    for i, (p, t) in enumerate([([1.3, 0.9, 0.7], [0, 0, 0.25]), ([-1.1, -1.0, 0.8], [0, 0, 0.25])]):
        q = _cam_quat_looking(np.array(t) - np.array(p))
        world += f'    <camera name="observer{i}" pos="{_fmt(p)}" quat="{_fmt(q)}" fovy="50"/>\n'

    # Front camera on the robot body (approximate Go2 front camera position), looking forward
    q = _cam_quat_looking([1, 0, -0.08])
    front_cam = f'\n      <camera name="front_cam" pos="0.30 0 0.05" quat="{_fmt(q)}" fovy="90"/>'

    xml = re.sub(r"(<body name=\"base_link\"[^>]*>)", lambda mm: mm.group(1) + front_cam, xml, count=1)
    xml = xml.replace("</worldbody>", world + "  </worldbody>", 1)
    xml = re.sub(r"(<mujoco[^>]*>)", lambda mm: mm.group(1) + extra_head, xml, count=1)
    return mujoco.MjModel.from_xml_string(xml)


def joint_indices(model):
    """qpos / qvel addresses for the 12 joints in SDK order, and actuator ids in SDK order."""
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in SDK_JOINTS])
    vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in SDK_JOINTS])
    act = np.array([model.actuator(n.replace("_joint", "")).id for n in SDK_JOINTS])
    return qadr, vadr, act


def robot_geom_ids(model):
    """All geoms that belong to the robot (bodies under base_link)."""
    root = model.body("base_link").id
    ids = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b != 0 and b != root:
            b = model.body_parentid[b]
        if b == root:
            ids.append(g)
    return np.array(ids)


def rpy_to_quat(roll, pitch, yaw):
    """Roll/pitch/yaw (Z-Y-X convention) -> quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_to_rpy(q):
    """Quaternion (w, x, y, z) -> roll, pitch, yaw (same convention as the Go2 IMU rpy)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def set_pose(model, data, x=0.0, y=0.0, z=0.32, yaw=0.0, pitch=0.0, roll=0.0, joints=STAND_UP):
    """Kinematically place the robot (free joint) and its joints, then run forward kinematics."""
    qadr, _, _ = joint_indices(model)
    data.qpos[0:3] = [x, y, z]
    data.qpos[3:7] = rpy_to_quat(roll, pitch, yaw)
    data.qpos[qadr] = joints
    mujoco.mj_forward(model, data)
