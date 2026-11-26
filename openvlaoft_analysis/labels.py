"""Label definitions for the probing study. Everything here reads the simulator or the model's output; nothing trains.

Image coordinates: the agentview frame exactly as the policy receives it before resizing (256x256, rotated 180 degrees
by openvla-oft's get_libero_image), x to the right, y down, in pixels. `to_policy_frame` maps them into the 224x224
center-cropped policy input.
"""
import os
import re

import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z", "stay"]

# instruction phrase -> scene objects it refers to (used for the swap wording-mismatch flag)
LANDMARKS = {
    "between the plate and the ramekin": ["plate_1", "glazed_rim_porcelain_ramekin_1"],
    "next to the ramekin": ["glazed_rim_porcelain_ramekin_1"],
    "from table center": [],
    "on the cookie box": ["cookies_1"],
    "in the top drawer of the wooden cabinet": ["wooden_cabinet_1"],
    "on the ramekin": ["glazed_rim_porcelain_ramekin_1"],
    "next to the cookie box": ["cookies_1"],
    "on the stove": ["flat_stove_1"],
    "next to the plate": ["plate_1"],
    "on the wooden cabinet": ["wooden_cabinet_1"],
}


# ----------------------------------------------------------------------------------------------- image projection
def project(sim, points, H=256, W=256):
    """World points (n, 3) -> (n, 2) pixel (x, y) in the rotated agentview image the policy sees.
    robosuite projects into the raw MuJoCo frame (row, col), which is upside down; openvla-oft then rotates it
    180 degrees. Net effect on the raw (row, col): x = W - 1 - col, y = row (checked against segmentation masks)."""
    M = get_camera_transform_matrix(sim, "agentview", H, W)
    rc = project_points_from_world_to_camera(np.atleast_2d(points), M, H, W)
    return np.stack([W - 1 - rc[:, 1], rc[:, 0]], axis=1).astype(np.float32)


def to_policy_frame(xy, env_res=256, policy_res=224, crop_area=0.9):
    """256 env frame -> 224 policy input after resize + center crop (crop side = sqrt(0.9) of the image)."""
    s = np.sqrt(crop_area)
    xy224 = np.asarray(xy, np.float32) * policy_res / env_res
    return ((xy224 - policy_res * (1 - s) / 2) / s).astype(np.float32)


def patch_token(xy, grid=16, patch_px=14, first=1):
    """Sequence index of the agentview patch token covering image point xy (256 env frame). Agentview patches are
    tokens 1..256 (after <BOS>, before the wrist patches), row-major over the 16x16 grid of the 224 policy image."""
    x, y = to_policy_frame(xy)
    r, c = (int(np.clip(v // patch_px, 0, grid - 1)) for v in (y, x))
    return first + r * grid + c


def patch_block(token, radius=1, grid=16, first=1):
    """Sequence indices of the (2 radius + 1)^2 agentview patches around a patch token (clipped at the image edge)."""
    r, c = divmod(token - first, grid)
    return [first + rr * grid + cc for rr in range(max(r - radius, 0), min(r + radius, grid - 1) + 1)
            for cc in range(max(c - radius, 0), min(c + radius, grid - 1) + 1)]


# ----------------------------------------------------------------------------------------------- grasp / phase
class GraspTracker:
    """grasped(t) = contact(gripper, bowl) AND gripper command == close AND bowl z >= rest z + lift_m.
    Phase is 'before' until the first grasped step and 'after' from then on (latched)."""

    def __init__(self, rest_z, lift_m=0.02):
        self.rest_z, self.lift_m, self.phase_after = float(rest_z), lift_m, False

    def is_grasped(self, contact, gripper_close_cmd, bowl_z):
        return bool(contact and gripper_close_cmd and bowl_z >= self.rest_z + self.lift_m)

    def update(self, contact, gripper_close_cmd, bowl_z):
        grasped = self.is_grasped(contact, gripper_close_cmd, bowl_z)
        self.phase_after |= grasped
        return grasped


def gripper_contact(env, obj_name):
    inner = env.env
    return bool(inner.check_contact(inner.robots[0].gripper, inner.objects_dict[obj_name]))


# ----------------------------------------------------------------------------------------------- action labels
def direction_label(xyz, stay_threshold):
    """Axis with the largest |movement|, signed; 'stay' if every |component| < stay_threshold."""
    xyz = np.asarray(xyz, np.float32)
    if np.all(np.abs(xyz) < stay_threshold):
        return DIRECTIONS.index("stay")
    ax = int(np.argmax(np.abs(xyz)))
    return 2 * ax + (0 if xyz[ax] > 0 else 1)


def distance_bin(xyz, edges):
    """Index of ||xyz|| in quantile bins: edges are the 7 inner edges of 8 bins."""
    return int(np.searchsorted(edges, np.linalg.norm(xyz), side="right"))


# ----------------------------------------------------------------------------------------------- BDDL helpers
def bddl_init_regions(path):
    s = open(path).read()
    init = re.search(r"\(:init(.*?)\(:goal", s, re.S).group(1)
    return dict(re.findall(r"\((?:On|In)\s+(\S+)\s+([^\s)]+)\)", init))


def wording_mismatch(task_language, orig_bddl, swap_bddl):
    """True if an object the instruction uses to describe the bowl was moved by the position swap."""
    phrase = next(k for k in LANDMARKS if k in task_language)
    a, b = bddl_init_regions(orig_bddl), bddl_init_regions(swap_bddl)
    moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return any(o in moved for o in LANDMARKS[phrase]), phrase, moved


def load_init_states_safe(path):
    """Read a LIBERO .pruned_init (torch zip of one numpy array) WITHOUT unpickling: the array bytes are pulled out
    of the pickle stream with pickletools, so no code from the downloaded file can run."""
    import pickletools
    import zipfile
    z = zipfile.ZipFile(path)
    raw = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    ops = list(pickletools.genops(raw))
    ints = [a for op, a, _ in ops if op.name in ("BININT1", "BININT2", "BININT")]
    blob = next(a for op, a, _ in ops if op.name in ("BINBYTES", "SHORT_BINBYTES", "BINBYTES8") and len(a) > 1000)
    n = len(blob) // 8
    shape = next((r, c) for r, c in zip(ints, ints[1:]) if r * c == n and r > 1 and c > 1)
    assert any(op.name == "SHORT_BINUNICODE" and a == "f8" for op, a, _ in ops), "expected float64 states"
    return np.frombuffer(blob, dtype="<f8").reshape(shape).copy()


if __name__ == "__main__":                                  # print the swap wording-mismatch flags
    from libero.libero import benchmark, get_libero_path
    here = os.path.dirname(os.path.abspath(__file__))
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for t in range(suite.n_tasks):
        task = suite.get_task(t)
        orig = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        swap = os.path.join(here, "data/libero_pro/bddl_files/libero_spatial_swap", task.bddl_file)
        flag, phrase, moved = wording_mismatch(task.language, orig, swap)
        print(f"task {t} | '{phrase}' | moved by swap: {moved} | wording no longer matches: {flag}")
