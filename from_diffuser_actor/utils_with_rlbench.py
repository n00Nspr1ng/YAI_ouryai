"""
This code is originally from 3d_diffuser_actor/utils/utils_with_rlbench.py
The script is modified to launch a custom rlbench task.
Search for changed codes with "#CHANGED".
"""

import random
from typing import List, Dict, Any
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from rlbench.demo import Demo


ALL_RLBENCH_TASKS = [
    'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
    'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
    'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
    'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
    'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
    'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
    'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
    'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
    'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
    'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
    'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
    'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
    'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
    'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
    'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
    'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
    'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
    'water_plants', 'wipe_desk',
    'pick_and_lift', 'yai_pick_and_place', 'yai_pick_and_place_v2' #CHANGED
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)


class Mover:

    def __init__(self, task, disabled=False, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action, collision_checking=False):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None

        self._policy.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, rgbs, pcds, gripper,
                interpolation_length=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None, "trajectory": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(rgbs.device)
        self._task_id = self._task_id.to(rgbs.device)

        # Predict trajectory
        if self._predict_trajectory:
            print('Predict Trajectory')
            fake_traj = torch.full(
                [1, interpolation_length - 1, gripper.shape[-1]], 0
            ).to(rgbs.device)
            traj_mask = torch.full(
                [1, interpolation_length - 1], False
            ).to(rgbs.device)
            output["trajectory"] = self._policy(
                fake_traj,
                traj_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[..., :7],
                run_inference=True
            )
        else:
            print('Predict Keypose')
            pred = self._policy(
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
            )
            # Hackish, assume self._policy is an instance of Act3D
            output["action"] = self._policy.prepare_action(pred)

        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device


def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)
