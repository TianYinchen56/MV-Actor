"""Online RLBench evaluation with an in-process Pi3X geometry worker."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import fetch_dataset_class
from online_evaluation_rlbench.evaluate_policy import configure_torch_runtime, load_models, parse_arguments
from online_evaluation_rlbench.pi3x_worker import Pi3XWorkerSession
from utils.common_utils import round_floats


class InProcessPi3XClient:
    def __init__(
        self,
        *,
        pi3_root: str,
        pi3_ckpt: str,
        device: str,
        use_pi3_pose_prior: bool,
        pi3_rlbench_to_opencv: bool,
    ):
        torch_num_threads = os.environ.pop("TORCH_NUM_THREADS", None)
        torch_num_interop_threads = os.environ.pop("TORCH_NUM_INTEROP_THREADS", None)
        self.session = Pi3XWorkerSession(
            pi3_root=pi3_root,
            pi3_ckpt=pi3_ckpt,
            device=device,
            use_pi3_pose_prior=use_pi3_pose_prior,
            pi3_rlbench_to_opencv=pi3_rlbench_to_opencv,
            configure_torch_threads=False,
        )
        if torch_num_threads is not None:
            os.environ["TORCH_NUM_THREADS"] = torch_num_threads
        if torch_num_interop_threads is not None:
            os.environ["TORCH_NUM_INTEROP_THREADS"] = torch_num_interop_threads
        self.runtime_device = str(self.session.device)
        self.worker_pid = int(os.getpid())

    def request_online_inputs(self, **kwargs):
        return self.session.build_online_inputs(**kwargs)

    def close(self) -> None:
        return None


def _resolve_required_path(cli_value, env_name: str, description: str) -> str:
    value = cli_value or os.environ.get(env_name)
    if not value:
        raise ValueError(f"Set --{env_name.lower()} or ${env_name} to a local {description} path.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local {description} path does not exist: {path}")
    return str(path)


if __name__ == "__main__":
    configure_torch_runtime()
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("TRACE_ONLINE_ROLLOUT", "0") == "1":
        trace_path = str(Path(args.output_file).with_suffix(".trace.json"))
        os.environ["TRACE_OUTPUT_FILE"] = trace_path

    if args.bimanual:
        from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner
    elif "peract" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_rlbench import RLBenchEnv, Actioner
    else:
        from online_evaluation_rlbench.utils_with_hiveformer_rlbench import RLBenchEnv, Actioner

    dataset_class = fetch_dataset_class(args.dataset)
    if args.apply_cameras is None:
        apply_cameras = tuple(dataset_class.cameras)
    else:
        apply_cameras = tuple(item.strip() for item in str(args.apply_cameras).split(",") if item.strip())
    if args.actor_cameras is None:
        actor_cameras = None
    else:
        actor_cameras = tuple(item.strip() for item in str(args.actor_cameras).split(",") if item.strip())

    need_pi3x_worker = bool(args.enable_interleaved_point_fusion) or str(args.scene_geometry_source).strip().lower() == "pi3x"
    pi3x_client = None
    if need_pi3x_worker:
        pi3_root = _resolve_required_path(args.pi3_root, "PI3_ROOT", "Pi3 code directory")
        pi3_ckpt = _resolve_required_path(args.pi3_ckpt, "PI3X_CKPT", "Pi3X checkpoint")
        runtime_device = os.environ.get("ONLINE_PI3X_WORKER_DEVICE", "cuda:0")
        pi3x_client = InProcessPi3XClient(
            pi3_root=pi3_root,
            pi3_ckpt=pi3_ckpt,
            device=runtime_device,
            use_pi3_pose_prior=args.pi3_use_pose_prior,
            pi3_rlbench_to_opencv=args.pi3_rlbench_to_opencv,
        )
        print(f"[pi3x_inprocess] pid={pi3x_client.worker_pid} device={pi3x_client.runtime_device}", flush=True)

    model = load_models(args)
    print("[phase] actioner_init_begin", flush=True)
    actioner = Actioner(
        model,
        backbone=args.backbone,
        pi3x_client=pi3x_client,
        pi3_root=args.pi3_root,
        pi3_ckpt=args.pi3_ckpt,
        scene_geometry_source=args.scene_geometry_source,
        online_semantic_corr_source=args.online_semantic_corr_source,
        vggt_root=args.vggt_root,
        vggt_ckpt=args.vggt_ckpt,
        vggt_preprocess_mode=args.vggt_preprocess_mode,
        vggt_image_size=args.vggt_image_size,
        vggt_align_mode=args.vggt_align_mode,
        use_pi3_pose_prior=args.pi3_use_pose_prior,
        pi3_rlbench_to_opencv=args.pi3_rlbench_to_opencv,
        apply_cameras=apply_cameras,
        actor_cameras=actor_cameras,
    )
    print("[phase] actioner_init_done", flush=True)

    try:
        task_success_rates = {}
        for task_str in [args.task]:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

            scene_geometry_source = str(args.scene_geometry_source).strip().lower()
            need_raw_point_cloud = scene_geometry_source not in {"pi3x", "vggt"}
            print("[phase] env_init_begin", flush=True)
            env = RLBenchEnv(
                data_path=args.data_dir,
                task_str=task_str,
                image_size=[int(x) for x in args.image_size.split(",")],
                apply_rgb=True,
                apply_pc=need_raw_point_cloud,
                headless=bool(args.headless),
                apply_cameras=apply_cameras,
                collision_checking=bool(args.collision_checking),
            )
            print("[phase] env_init_done", flush=True)

            eval_kwargs = dict(
                task_str=task_str,
                max_steps=args.max_steps,
                actioner=actioner,
                max_tries=args.max_tries,
                prediction_len=args.prediction_len,
                num_history=args.num_history,
            )
            if args.bimanual:
                eval_kwargs.update(
                    variation=args.variation,
                    demo_amount=args.demo_amount,
                    from_episode_number=args.from_episode_number,
                )
                var_success_rates = env.evaluate_task_on_multiple_variations(**eval_kwargs)
            else:
                var_success_rates = []
                for variation in env.variation_numbers:
                    success_rate = env.evaluate_task_on_one_variation(variation=variation, **eval_kwargs)
                    var_success_rates.append(success_rate)

            env.env.shutdown()
            torch.cuda.empty_cache()
            task_success_rates[task_str] = var_success_rates

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
        print(f"Results saved to {args.output_file}")
    finally:
        if pi3x_client is not None:
            pi3x_client.close()
