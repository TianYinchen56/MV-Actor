import os
import glob
from pathlib import Path
import random
import warnings

import open3d  # DON'T DELETE THIS!
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops
from matplotlib import pyplot as plt

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos


warnings.filterwarnings(
    "ignore",
    message="Object .* has ObjectType.LIGHT, which is not supported",
    category=UserWarning,
    module="pyrep.objects.object"
)


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class CameraMotion:
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class StaticCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor):
        super().__init__(cam)

    def step(self):
        pass


class TaskRecorder(object):

    def __init__(self, cams_motion, fps=30):
        self._cams_motion = cams_motion
        self._fps = fps
        self._snaps = {cam_name: [] for cam_name in self._cams_motion.keys()}

    def take_snap(self):
        for cam_name, cam_motion in self._cams_motion.items():
            cam_motion.step()
            self._snaps[cam_name].append(
                (cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))

    def save(self, path):
        print('Converting to video ...')
        path = Path(path)
        path.mkdir(exist_ok=True)
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        # Assume 'frames' is a list of RGB NumPy arrays with shape (H, W, 3)
        for cam_name, cam_motion in self._cams_motion.items():
            clip = ImageSequenceClip([image for image in self._snaps[cam_name]], fps=30)  # 30 FPS
            clip.write_videofile("output_video.mp4", codec="libx264")
            break


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (8,)
        obs = None
        terminate = None
        reward = 0

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()
        for _ in range(self._max_tries):
            obs, reward, terminate = self._task.step(action)

            # Check if we reached the desired pose (planner may be inaccurate)
            pos = obs.gripper_pose[:3]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            obs, reward, terminate = self._task.step(action)

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip', **_kwargs):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)

    @staticmethod
    def _to_cuda_tokenized(tokenized):
        if hasattr(tokenized, "items"):
            return {
                key: value.cuda(non_blocking=True)
                for key, value in tokenized.items()
            }
        if torch.is_tensor(tokenized):
            return tokenized.cuda(non_blocking=True)
        return {
            key: value.cuda(non_blocking=True)
            for key, value in tokenized.items()
        }

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self._to_cuda_tokenized(self.tokenizer(instr))

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs: (1, ncam, 3, H, W)
            pcds: (1, ncam, 3, H, W)
            gripper: (1, nhist, 8)
            prediction_len: int

        Returns:
            (1, prediction_len, 8)
        """
        return self._policy(
            None,
            torch.full([1, prediction_len, 1], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper[:, :, None, :7],
            run_inference=True
        ).view(1, prediction_len, 8)


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size

    def get_obs_action(self, obs):
        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        obs_rgb = [
            torch.tensor(state_dict["rgb"][i]).float().permute(2, 0, 1) / 255.0
            for i in range(len(state_dict["rgb"]))
        ]
        obs_pc = [
            torch.tensor(state_dict["pc"][i]).float().permute(2, 0, 1)
            if len(state_dict["pc"]) > 0 else None
            for i in range(len(state_dict["rgb"]))
        ]
        state = torch.cat(obs_rgb + obs_pc, dim=0)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    num_demos=100 // len(task_variations),
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        num_demos,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1
    ):
        success_rate = 0
        total_reward = 0

        for demo_id in range(num_demos):

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            # no need to generate demos here, just get random states
            np.random.set_state(np.random.get_state())
            descriptions, obs = task.reset()
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0

            for _ in range(max_steps):

                # Fetch the current observation, and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                # Prepare proprioception history
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                # Update the observation based on the predicted action
                # Execute entire predicted trajectory step by step
                actions = output[-1].cpu().numpy()
                actions[:, -1] = actions[:, -1].round()

                # execute
                for action in actions:
                    try:
                        obs, reward, _ = move(action, collision_checking=False)
                    except:
                        pass

                max_reward = max(max_reward, reward)

                if reward == 1:
                    success_rate += 1
                    break

            total_reward += max_reward

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id + 1}",
                f"SR: {total_reward:.2f}/{demo_id + 1}",
                "# valid demos", demo_id + 1,
            )

        return success_rate, True, num_demos

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras,
        **kwargs
    ):
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
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
            gripper_joint_positions=True
        )

        return obs_config
