import os
import glob
import random
import json
import gc
from pathlib import Path
import sys
import psutil

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, assert_action_shape
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online_evaluation_rlbench.online_extra_inputs import OnlineExtraInputBuilder


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (2, 8)
        obs = None
        terminate = None
        reward = 0
        debug_rollout = os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1"

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy() # copy gripper state
        for _ in range(self._max_tries):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # Peract2 takes (right, left) action, but we predict (left, right)
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            if debug_rollout:
                print("[move] pose_only_step begin", flush=True)
            obs, reward, terminate = self._task.step(action_collision)
            if debug_rollout:
                print(
                    f"[move] pose_only_step done reward={float(reward):.2f} terminated={bool(terminate)}",
                    flush=True,
                )

            # Check if we reached the desired pose (planner may be inaccurate)
            l_pos = obs.left.gripper_pose[:3]
            r_pos = obs.right.gripper_pose[:3]
            l_dist_pos = np.sqrt(np.square(target[0, :3] - l_pos).sum())
            r_dist_pos = np.sqrt(np.square(target[1, :3] - r_pos).sum())
            criteria = (l_dist_pos < 5e-3, r_dist_pos < 5e-3)
            if debug_rollout:
                print(
                    "[move] pose_only_check "
                    f"l_dist_m={float(l_dist_pos):.6f} "
                    f"r_dist_m={float(r_dist_pos):.6f} "
                    f"target_left={np.array2string(target[0], precision=5, suppress_small=False)} "
                    f"target_right={np.array2string(target[1], precision=5, suppress_small=False)} "
                    f"obs_left={np.array2string(obs.left.gripper_pose, precision=5, suppress_small=False)} "
                    f"obs_right={np.array2string(obs.right.gripper_pose, precision=5, suppress_small=False)}",
                    flush=True,
                )

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        gripper_changed = (
            self._last_action is not None
            and (
                action[0, 7] != self._last_action[0, 7]
                or action[1, 7] != self._last_action[1, 7]
            )
        )
        if debug_rollout and self._last_action is not None:
            print(
                "[move] gripper_decision "
                f"reward={float(reward):.2f} "
                f"changed={bool(gripper_changed)} "
                f"last_left={float(self._last_action[0, 7]):.3f} "
                f"last_right={float(self._last_action[1, 7]):.3f} "
                f"target_left={float(action[0, 7]):.3f} "
                f"target_right={float(action[1, 7]):.3f}",
                flush=True,
            )
        if (
            not reward == 1.0
            and self._last_action is not None
            and (  # if any gripper state has changed, re-execute
                action[0, 7] != self._last_action[0, 7]
                or action[1, 7] != self._last_action[1, 7]
            )
        ):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            if debug_rollout:
                print(
                    "[move] gripper_step begin "
                    f"action_left={np.array2string(action[0], precision=5, suppress_small=False)} "
                    f"action_right={np.array2string(action[1], precision=5, suppress_small=False)}",
                    flush=True,
                )
            obs, reward, terminate = self._task.step(action_collision)
            if debug_rollout:
                print(
                    f"[move] gripper_step done reward={float(reward):.2f} terminated={bool(terminate)}",
                    flush=True,
                )

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(
        self,
        policy=None,
        backbone='clip',
        pi3x_client=None,
        pi3_root=None,
        pi3_ckpt=None,
        scene_geometry_source=None,
        online_semantic_corr_source=None,
        vggt_root=None,
        vggt_ckpt=None,
        vggt_preprocess_mode=None,
        vggt_image_size=None,
        vggt_align_mode=None,
        use_pi3_pose_prior=True,
        pi3_rlbench_to_opencv=True,
        apply_cameras=None,
        actor_cameras=None,
    ):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self._instr_text = None
        self.tokenizer = fetch_tokenizers(backbone)
        self.apply_cameras = tuple(apply_cameras or ())
        if actor_cameras is None:
            actor_cameras_env = os.environ.get("ONLINE_ACTOR_CAMERAS", "").strip()
            actor_cameras = tuple(
                item.strip() for item in actor_cameras_env.split(",") if item.strip()
            )
            if len(actor_cameras) == 0:
                actor_cameras = None
        self.actor_cameras = tuple(actor_cameras or ())
        if len(self.actor_cameras) > 0 and len(self.apply_cameras) == 0:
            raise ValueError("actor_cameras requires apply_cameras")
        self.actor_camera_indices = None
        if len(self.actor_cameras) > 0:
            missing = [cam for cam in self.actor_cameras if cam not in self.apply_cameras]
            if missing:
                raise ValueError(
                    f"actor_cameras must be subset of apply_cameras, missing={missing}, "
                    f"apply_cameras={self.apply_cameras}"
                )
            self.actor_camera_indices = tuple(
                self.apply_cameras.index(cam) for cam in self.actor_cameras
            )
        self.extra_input_builder = OnlineExtraInputBuilder(
            self._policy,
            pi3x_client=pi3x_client,
            pi3_root=pi3_root,
            pi3_ckpt=pi3_ckpt,
            scene_geometry_source=scene_geometry_source,
            online_semantic_corr_source=online_semantic_corr_source,
            vggt_root=vggt_root,
            vggt_ckpt=vggt_ckpt,
            vggt_preprocess_mode=vggt_preprocess_mode,
            vggt_image_size=vggt_image_size,
            vggt_align_mode=vggt_align_mode,
            use_pi3_pose_prior=use_pi3_pose_prior,
            pi3_rlbench_to_opencv=pi3_rlbench_to_opencv,
        )
        self._last_router_trace = []

    def shutdown(self) -> None:
        self.extra_input_builder.shutdown()

    def _collect_router_trace(self):
        policy = self._policy.module if hasattr(self._policy, 'module') else self._policy
        traces = []
        for module_name, module in policy.named_modules():
            gates = getattr(module, 'last_gates', None)
            if gates is None or not torch.is_tensor(gates):
                continue
            gates_cpu = gates.detach().float().cpu()
            entry = {
                'module': module_name,
                'shape': list(gates_cpu.shape),
                'values': gates_cpu.tolist(),
            }
            if gates_cpu.ndim == 2 and gates_cpu.shape[-1] == 4:
                labels = ['sem_pos', 'geo_pos', 'sem_rot', 'geo_rot']
                entry['labels'] = labels
                entry['sample0'] = {label: float(gates_cpu[0, idx].item()) for idx, label in enumerate(labels)}
            traces.append(entry)
        return traces

    def get_last_router_trace(self):
        return list(self._last_router_trace)

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
        self._instr_text = instr[0]
        self._instr = self._to_cuda_tokenized(self.tokenizer(instr))

    def _subset_views_tensor(self, value):
        if self.actor_camera_indices is None or not torch.is_tensor(value):
            return value
        if value.ndim < 2 or value.shape[1] != len(self.apply_cameras):
            return value
        select_idx = torch.as_tensor(
            self.actor_camera_indices,
            device=value.device,
            dtype=torch.long,
        )
        return value.index_select(1, select_idx)

    def _subset_extra_inputs(self, extra_inputs):
        if extra_inputs is None or self.actor_camera_indices is None:
            return extra_inputs
        subset = {}
        for key, value in extra_inputs.items():
            subset[key] = self._subset_views_tensor(value)
        return subset

    def predict(
        self,
        rgbs,
        pcds,
        gripper,
        prediction_len=1,
        camera_intrinsics=None,
        camera_extrinsics=None,
        precomputed_extra_inputs=None,
    ):
        """
        Args:
            rgbs: (1, ncam, 3, H, W)
            pcds: (1, ncam, 3, H, W)
            gripper: (1, nhist, 2*8)
            prediction_len: int

        Returns:
            trajectory: (1, nhist, nhand=2, 8)
        """
        with torch.inference_mode():
            debug_rollout = os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1"
            if debug_rollout:
                print("[predict] extra_inputs_begin", flush=True)
            if precomputed_extra_inputs is None:
                extra_inputs = self.extra_input_builder.build(
                    rgbs,
                    pcds,
                    camera_intrinsics,
                    camera_extrinsics,
                )
            else:
                extra_inputs = dict(precomputed_extra_inputs)
            if self.actor_camera_indices is not None:
                rgbs = self._subset_views_tensor(rgbs)
                pcds = self._subset_views_tensor(pcds)
                camera_intrinsics = self._subset_views_tensor(camera_intrinsics)
                camera_extrinsics = self._subset_views_tensor(camera_extrinsics)
                extra_inputs = self._subset_extra_inputs(extra_inputs)
            if debug_rollout:
                print("[predict] extra_inputs_done", flush=True)
            model_rgbs = rgbs
            model_pcds = pcds
            if extra_inputs is not None:
                online_model_pcds = extra_inputs.pop("online_model_pcds", None)
                extra_inputs.pop("online_vggt_align_diag", None)
                online_siglip2_tokens = extra_inputs.pop("online_siglip2_tokens", None)
                semantic_xyz_tokens = extra_inputs.get("semantic_correspondence_xyz")
                if online_model_pcds is not None:
                    model_pcds = online_model_pcds
                if online_siglip2_tokens is not None and semantic_xyz_tokens is not None:
                    model_rgbs = online_siglip2_tokens
                    model_pcds = semantic_xyz_tokens
            if debug_rollout:
                print("[predict] policy_forward_begin", flush=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                out = self._policy(
                    None,
                    torch.full([1, prediction_len, 2], False).cuda(non_blocking=True),
                    model_rgbs,
                    None,
                    model_pcds,
                    self._instr,
                    gripper.unflatten(-1, (2, -1)),
                    run_inference=True,
                    extra_inputs=extra_inputs,
                )
            self._last_router_trace = self._collect_router_trace()
            if debug_rollout:
                print("[predict] policy_forward_done", flush=True)
            return out


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
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=HandoverDiscrete() if 'handover' in task_str else BimanualDiscrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless, robot_setup="dual_panda"
        )

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        rgb = torch.stack([
            torch.tensor(obs.perception_data["{}_rgb".format(cam)]).float().permute(2, 0, 1) / 255.0
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        pcds = []
        for cam in self.apply_cameras:
            point_cloud = obs.perception_data.get(f"{cam}_point_cloud", None)
            if point_cloud is None:
                h, w = obs.perception_data[f"{cam}_rgb"].shape[:2]
                point_cloud = np.zeros((h, w, 3), dtype=np.float32)
            pcds.append(torch.tensor(point_cloud).float().permute(2, 0, 1))
        pcd = torch.stack(pcds).unsqueeze(0)  # 1, N, C, H, W
        # action is an array of length 16 = (7+1)*2
        gripper = torch.from_numpy(np.concatenate([
            obs.left.gripper_pose, [obs.left.gripper_open],
            obs.right.gripper_pose, [obs.right.gripper_open]
        ])).float().unsqueeze(0)  # 1, D

        misc = getattr(obs, "misc", None)
        if misc is None:
            raise ValueError("RLBench observation is missing misc camera parameters")

        intrinsics = torch.stack([
            torch.tensor(misc[f"{cam}_camera_intrinsics"]).float()
            for cam in self.apply_cameras
        ]).unsqueeze(0)
        extrinsics = torch.stack([
            torch.tensor(misc[f"{cam}_camera_extrinsics"]).float()
            for cam in self.apply_cameras
        ]).unsqueeze(0)

        return rgb, pcd, gripper, intrinsics, extrinsics

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1,
        variation=-1,
        demo_amount=-1,
        from_episode_number=0,
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        if variation >= 0:
            task_variations = [variation]
        else:
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
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history,
                    demo_amount=demo_amount,
                    from_episode_number=from_episode_number,
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
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1,
        demo_amount=-1,
        from_episode_number=0,
    ):
        process = psutil.Process()

        def _log_memory(tag: str) -> None:
            if os.environ.get("ONLINE_DEBUG_MEMORY", "0") != "1":
                return
            mem = process.memory_info()
            gpu_alloc_gb = 0.0
            gpu_reserved_gb = 0.0
            if torch.cuda.is_available():
                gpu_alloc_gb = float(torch.cuda.memory_allocated()) / (1024 ** 3)
                gpu_reserved_gb = float(torch.cuda.memory_reserved()) / (1024 ** 3)
            print(
                f"[mem] {tag} rss_gb={mem.rss / (1024 ** 3):.3f} "
                f"vms_gb={mem.vms / (1024 ** 3):.3f} "
                f"gpu_alloc_gb={gpu_alloc_gb:.3f} "
                f"gpu_reserved_gb={gpu_reserved_gb:.3f}",
                flush=True,
            )

        success_rate = 0
        total_reward = 0
        print(
            f"[phase] get_stored_demos_begin task={task_str} variation={variation}",
            flush=True,
        )
        var_demos = get_stored_demos(
            amount=demo_amount,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=from_episode_number
        )
        print(
            f"[phase] get_stored_demos_done count={len(var_demos)}",
            flush=True,
        )

        for demo_id, demo in enumerate(var_demos):

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            print(
                f"[phase] reset_to_demo_begin demo={demo_id} episode={from_episode_number + demo_id}",
                flush=True,
            )
            descriptions, obs = task.reset_to_demo(demo)
            print(
                f"[phase] reset_to_demo_done demo={demo_id}",
                flush=True,
            )
            actioner.load_episode(descriptions)
            print(
                f"[demo] id={demo_id} task={task_str} variation={variation} "
                f"episode={from_episode_number + demo_id} instruction={actioner._instr_text}",
                flush=True,
            )

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0
            trace_steps = []
            trace_enabled = os.environ.get("TRACE_ONLINE_ROLLOUT", "0") == "1"

            for step_id in range(max_steps):
                print(f"[step] demo={demo_id} step={step_id} begin", flush=True)
                _log_memory(f"demo={demo_id} step={step_id} begin")

                # Fetch the current observation and predict one action
                rgb, pcd, gripper, intrinsics, extrinsics = self.get_rgb_pcd_gripper_from_obs(obs)
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
                current_gripper = gripper_input[0, -1].view(2, -1)

                if os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1":
                    print(f"[step] demo={demo_id} step={step_id} predict begin", flush=True)
                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len,
                    camera_intrinsics=intrinsics,
                    camera_extrinsics=extrinsics,
                )
                _log_memory(f"demo={demo_id} step={step_id} after_predict")
                if os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1":
                    print(f"[step] demo={demo_id} step={step_id} predict done", flush=True)

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu().numpy()
                    actions_before_round = actions.copy()
                    actions[..., -1] = actions[..., -1].round()
                    if os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1":
                        print(
                            f"[step] demo={demo_id} step={step_id} actions_before_round={np.array2string(actions_before_round, precision=5, suppress_small=False)}",
                            flush=True,
                        )
                        print(
                            f"[step] demo={demo_id} step={step_id} actions_after_round={np.array2string(actions, precision=5, suppress_small=False)}",
                            flush=True,
                        )

                    # execute
                    terminated = False
                    step_actions = []
                    if os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1":
                        print(f"[step] demo={demo_id} step={step_id} move begin", flush=True)
                    for action in actions:
                        obs, reward, terminated = move(action, collision_checking=False)
                        step_actions.append({
                            "action": np.asarray(action, dtype=np.float32).tolist(),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "left_obs_pose_after": np.asarray(obs.left.gripper_pose, dtype=np.float32).tolist(),
                            "right_obs_pose_after": np.asarray(obs.right.gripper_pose, dtype=np.float32).tolist(),
                            "left_gripper_open_after": float(obs.left.gripper_open),
                            "right_gripper_open_after": float(obs.right.gripper_open),
                        })
                        if terminated:
                            break
                    if os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1":
                        print(
                            f"[step] demo={demo_id} step={step_id} move done reward={float(reward):.2f} terminated={bool(terminated)}",
                            flush=True,
                        )

                    max_reward = max(max_reward, reward)

                    if trace_enabled:
                        trace_steps.append({
                            "step_id": int(step_id),
                            "instruction": actioner._instr_text,
                            "left_obs_pose_before": np.asarray(current_gripper[0, :7].cpu(), dtype=np.float32).tolist(),
                            "right_obs_pose_before": np.asarray(current_gripper[1, :7].cpu(), dtype=np.float32).tolist(),
                            "left_gripper_open_before": float(current_gripper[0, 7].item()),
                            "right_gripper_open_before": float(current_gripper[1, 7].item()),
                            "predicted_actions_before_round": np.asarray(actions_before_round, dtype=np.float32).tolist(),
                            "predicted_actions_after_round": np.asarray(actions, dtype=np.float32).tolist(),
                            "executed_actions": step_actions,
                            "reward_after_step": float(reward),
                            "max_reward_so_far": float(max_reward),
                            "router_trace": actioner.get_last_router_trace(),
                        })

                    if reward == 1:
                        success_rate += 1
                        print(
                            f"[step] demo={demo_id} step={step_id} reward={float(reward):.2f} "
                            f"max_reward={float(max_reward):.2f} success=1",
                            flush=True,
                        )
                        break

                    if terminated:
                        print(
                            f"[step] demo={demo_id} step={step_id} reward={float(reward):.2f} "
                            f"max_reward={float(max_reward):.2f} terminated=1",
                            flush=True,
                        )
                        break

                    print(
                        f"[step] demo={demo_id} step={step_id} reward={float(reward):.2f} "
                        f"max_reward={float(max_reward):.2f}",
                        flush=True,
                    )

                except (IKError, ConfigurationPathError, InvalidActionError, RuntimeError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0
                    break

                del rgb
                del pcd
                del gripper
                del intrinsics
                del extrinsics
                del rgbs_input
                del pcds_input
                del gripper_input
                del current_gripper
                del output
                torch.cuda.empty_cache()
                gc.collect()
                _log_memory(f"demo={demo_id} step={step_id} after_cleanup")

            total_reward += max_reward

            if trace_enabled:
                trace_output_file = os.environ.get("TRACE_OUTPUT_FILE")
                if trace_output_file:
                    payload = {
                        "task": task_str,
                        "variation": int(variation),
                        "demo_id": int(demo_id),
                        "instruction": actioner._instr_text,
                        "max_reward": float(max_reward),
                        "success": float(max_reward >= 1.0),
                        "steps": trace_steps,
                    }
                    with open(trace_output_file, "w") as f:
                        json.dump(payload, f, indent=2)

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
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", demo_id + 1
            )

        # Compensate for failed demos
        valid = len(var_demos) > 0

        return success_rate, valid, len(var_demos)

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
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
            render_mode=RenderMode.OPENGL3,  # note OPENGL3 for Peract2!
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams

        obs_config = ObservationConfig(
            camera_configs=cameras,
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


class HandoverDiscrete(BimanualDiscrete):
    """
    A custom gripper action mode for the handover task.
    It forces one gripper to release so that the other grasps.
    """

    def action(self, scene, action):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        if 0.0 > action[1] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        right_open_condition = all(
            x > 0.9 for x in scene.robot.right_gripper.get_open_amount())

        left_open_condition = all(
            x > 0.9 for x in scene.robot.left_gripper.get_open_amount())

        right_current_ee = 1.0 if right_open_condition else 0.0
        left_current_ee = 1.0 if left_open_condition else 0.0

        right_action = float(action[0] > 0.5)
        left_action = float(action[1] > 0.5)

        if right_current_ee != right_action or left_current_ee != left_action:
            if not self._detach_before_open:
                self._actuate(scene, action)

        # Move objects between grippers
        if right_current_ee != right_action:
            if right_action == 0.0 and self._attach_grasped_objects:
                left_grasped_objects = scene.robot.left_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in left_grasped_objects:
                        scene.robot.left_gripper.release()
                        scene.robot.right_gripper.grasp(g_obj)
                    else:
                        scene.robot.right_gripper.grasp(g_obj)
            else:
                scene.robot.right_gripper.release()
        if left_current_ee != left_action:
            if left_action == 0.0 and self._attach_grasped_objects:
                right_grasped_objects = scene.robot.right_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in right_grasped_objects:
                        scene.robot.right_gripper.release()
                        scene.robot.left_gripper.grasp(g_obj)
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                scene.robot.left_gripper.release()

        if right_current_ee != right_action or left_current_ee != left_action:
            if self._detach_before_open:
                self._actuate(scene, action)
            if right_action == 1.0 or left_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()
