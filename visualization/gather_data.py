import random
import os

import torch
import torch.nn.functional as F
import numpy as np
import einops
from tqdm import tqdm
import pickle

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import YaiPickAndPlaceCylinder
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from utils.common_utils import load_instructions
from utils.utils_with_rlbench import Actioner, Mover, transform, obs_to_attn
# from online_evaluation_rlbench.evaluate_policy import Arguments, load_models

from YAI_ouryai.visualization.visualization_utils import Arguments, load_models
from YAI_ouryai.from_diffuser_actor.utils_with_rlbench import Actioner, Mover, transform, obs_to_attn


def create_obs_config(args, apply_rgb, apply_depth, apply_pcd):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=apply_rgb,
        depth=apply_depth,
        point_cloud=apply_pcd,
        image_size=[int(x) for x in args.image_size.split(",")],
        render_mode=RenderMode.OPENGL
    )
    camera_names = args.cameras
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams

    obs_config = ObservationConfig(
        front_camera=kwargs.get("front", unused_cams),
        left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
        right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
        wrist_camera=kwargs.get("wrist", unused_cams),
        overhead_camera=kwargs.get("overhead", unused_cams),
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    
    return obs_config

# This code is integrated from utils.utils_with_rlbench.py from 3d_diffuser_actor.
def get_rgb_pcd_gripper_from_obs(apply_cameras, apply_rgb, apply_depth, apply_pcd, obs, image_size):

    # fetch state
    state_dict = {"rgb": [], "depth": [], "pc": []}
    for cam in apply_cameras:
        if apply_rgb:
            rgb = getattr(obs, "{}_rgb".format(cam))
            state_dict["rgb"] += [rgb]

        if apply_depth:
            depth = getattr(obs, "{}_depth".format(cam))
            state_dict["depth"] += [depth]

        if apply_pcd:
            pc = getattr(obs, "{}_point_cloud".format(cam))
            state_dict["pc"] += [pc]

    # fetch action
    action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
    gripper = torch.from_numpy(action).float()

    state = transform(state_dict, augmentation=False)
    state = einops.rearrange(
        state,
        "(m n ch) h w -> n m ch h w",
        ch=3,
        n=len(apply_cameras),
        m=2
    )
    rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
    pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
    gripper = gripper.unsqueeze(0)  # 1, D

    attns = torch.Tensor([])
    for cam in apply_cameras:
        u, v = obs_to_attn(obs, cam)
        attn = torch.zeros(1, 1, 1, image_size[0], image_size[1])
        if not (u < 0 or u > image_size[1] - 1 or v < 0 or v > image_size[0] - 1):
            attn[0, 0, 0, v, u] = 1
        attns = torch.cat([attns, attn], 1)
    rgb = torch.cat([rgb, attns], 2)

    return rgb, pcd, gripper


def save_pcd_color(obs):
    pcd = obs.front_point_cloud.reshape(-1, 3)
    pcd = np.concatenate((pcd, obs.wrist_point_cloud.reshape(-1, 3)), axis=0)
    pcd = np.concatenate((pcd, obs.left_shoulder_point_cloud.reshape(-1, 3)), axis=0)
    pcd = np.concatenate((pcd, obs.right_shoulder_point_cloud.reshape(-1, 3)), axis=0)

    colors = obs.front_rgb.reshape(-1, 3)
    colors = np.concatenate((colors, obs.wrist_rgb.reshape(-1, 3)), axis=0)
    colors = np.concatenate((colors, obs.left_shoulder_rgb.reshape(-1, 3)), axis=0)
    colors = np.concatenate((colors, obs.right_shoulder_rgb.reshape(-1, 3)), axis=0)

    filtered_pcd = pcd[(pcd[:, 0] > -1.0) & (pcd[:, 2] > 0.5)]
    filtered_colors = colors[(pcd[:, 0] > -1.0) & (pcd[:, 2] > 0.5)]

    return filtered_pcd, filtered_colors


if __name__ == "__main__":

    # Arguments
    args = Arguments().parse_args()
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    if args.tasks[0] != 'yai_pick_and_place':
        raise NotImplementedError
    print("Arguments:")
    print(args)
    print("-" * 100)

    # parameters
    apply_rgb = True
    apply_depth = False
    apply_pcd = True
    save_results = True

    # Save results here
    if save_results:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        outputs = {
            'pointcloud': [],
            'colors': [],
            'gripper_pose': [],
            'gripper_open': []
        }
    diffused_trajectories = []

    # Seeds - demo seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)
    model.eval()

    # Load instructions
    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    # Load actioner
    actioner = Actioner(
        policy=model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory)
    )

    # Make seperate RLBench environment    
    obs_config = create_obs_config(args, apply_rgb, apply_depth, apply_pcd)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=args.collision_checking), 
            gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=False
    )

    # Load Environment
    env.launch()
    task = env.get_task(YaiPickAndPlaceCylinder)

    # Evaluate Task 
    device = actioner.device

    rgbs = torch.Tensor([]).to(device)
    pcds = torch.Tensor([]).to(device)
    grippers = torch.Tensor([]).to(device)

    descriptions, obs = task.reset()
    if save_results:
        outputs['gripper_open'].append(obs.gripper_open)
        outputs['gripper_pose'].append(obs.gripper_pose)
        pcd, colors = save_pcd_color(obs)
        outputs['pointcloud'].append(pcd)
        outputs['colors'].append(colors)

    # loads instructions, task_id
    actioner.load_episode("yai_pick_and_place", variation=0)

    # Set Mover
    move = Mover(task, max_tries=args.max_tries)
    reward = 0.0
    max_reward = 0.0

    for step_id in range(args.max_steps):
        # Fetch the current observation, and predict one action
        rgb, pcd, gripper = get_rgb_pcd_gripper_from_obs(
            apply_cameras=args.cameras,
            apply_rgb=apply_rgb,
            apply_depth=apply_depth,
            apply_pcd=apply_pcd,
            obs=obs,
            image_size=[int(x) for x in args.image_size.split(",")]
        )
        rgb = rgb.to(device)
        pcd = pcd.to(device)
        gripper = gripper.to(device)

        rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
        pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
        grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

        # Prepare proprioception history
        rgbs_input = rgbs[:, -1:][:, :, :, :3]
        pcds_input = pcds[:, -1:]
        if args.num_history < 1:
            gripper_input = grippers[:, -1]
        else:
            gripper_input = grippers[:, -args.num_history:]
            npad = args.num_history - gripper_input.shape[1]
            gripper_input = F.pad(
                gripper_input, (0, 0, npad, 0), mode='replicate'
            )

        # Predict
        first_output = None # Run inference with first output
        keypose_diff_trajectory = np.ndarray((args.save_diff_num, args.diffusion_timesteps, 3))
        for i in range(args.save_diff_num):
            with torch.no_grad():
                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    interpolation_length=args.interpolation_length
                )
            if first_output is None:
                first_output = output
            keypose_diff_trajectory[i] = np.squeeze(np.array(actioner._policy.diffusion_trajectory)) # (100, 3)

        diffused_trajectories.append(keypose_diff_trajectory)
        output = first_output

        # Update the observation based on the predicted action
        # Erased collision_checking due to corrupted code in the original version
        try:
            # Execute entire predicted trajectory step by step
            if output.get("trajectory", None) is not None:
                trajectory = output["trajectory"][-1].cpu().numpy()
                trajectory[:, -1] = trajectory[:, -1].round()

                # execute
                for action in tqdm(trajectory):
                    obs, reward, terminate, _ = move(action, collision_checking=args.collision_checking)
                    if save_results:
                        outputs['gripper_open'].append(obs.gripper_open)
                        outputs['gripper_pose'].append(obs.gripper_pose)
                        pcd, colors = save_pcd_color(obs)
                        outputs['pointcloud'].append(pcd)
                        outputs['colors'].append(colors)
                        
            # Or plan to reach next predicted keypoint
            else:
                print("Plan with RRT")
                action = output["action"]
                action[..., -1] = torch.round(action[..., -1])
                action = action[-1].detach().cpu().numpy()

                obs, reward, terminate, _ = move(action, collision_checking=args.collision_checking)
                if save_results:
                    outputs['gripper_open'].append(obs.gripper_open)
                    outputs['gripper_pose'].append(obs.gripper_pose)
                    pcd, colors = save_pcd_color(obs)
                    outputs['pointcloud'].append(pcd)
                    outputs['colors'].append(colors)
                    
            max_reward = max(max_reward, reward)

            if reward == 1:
                print("Reward 1!")
                break

            if terminate:
                print("The episode has terminated!")

        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            print(args.seed, step_id, e)
            reward = 0
            #break
        
    print(f"Finished demo {args.seed} as max reward {max_reward}")
    if save_results:
        outputs['reward'] = max_reward

    env.shutdown()

    if save_results:
        with open(os.path.join(os.path.dirname(args.output_file), 'outputs.pkl') , "wb") as f:
            pickle.dump(outputs, f)

        with open(os.path.join(os.path.dirname(args.output_file), 'diff_traj.pkl') , "wb") as f:
            pickle.dump(diffused_trajectories, f)

    
