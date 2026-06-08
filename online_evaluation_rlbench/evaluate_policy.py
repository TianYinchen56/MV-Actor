"""Online evaluation script on RLBench."""

import argparse
import random
from pathlib import Path
import json
import os
import sys

import torch
import numpy as np
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none, round_floats


def configure_torch_runtime() -> None:
    default_threads = int(os.environ.get("ONLINE_EVAL_NUM_THREADS", "1"))
    num_threads = int(os.environ.get("TORCH_NUM_THREADS", str(default_threads)))
    num_interop_threads = int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, str(num_threads))
    os.environ.setdefault("TORCH_NUM_THREADS", str(num_threads))
    os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", str(num_interop_threads))
    threadpool_limits(limits=num_threads)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    print(
        f"[torch_runtime] num_threads={torch.get_num_threads()} "
        f"num_interop_threads={torch.get_num_interop_threads()} "
        f"omp={os.environ.get('OMP_NUM_THREADS')} "
        f"mkl={os.environ.get('MKL_NUM_THREADS')} "
        f"openblas={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"numexpr={os.environ.get('NUMEXPR_NUM_THREADS')}",
        flush=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Testing arguments
        ('checkpoint', str_none, None),
        ('task', str, "close_jar"),
        ('variation', int, -1),
        ('demo_amount', int, -1),
        ('from_episode_number', int, 0),
        ('max_tries', int, 10),
        ('max_steps', int, 25),
        ('headless', str2bool, False),
        ('collision_checking', str2bool, False),
        ('seed', int, 0),
        # Dataset arguments
        ('data_dir', Path, Path(__file__).parent / "demos"),
        ('dataset', str, "Peract"),
        ('image_size', str, "256,256"),
        ('apply_cameras', str_none, None),
        ('actor_cameras', str_none, None),
        # Logging arguments
        ('output_file', Path, Path(__file__).parent / "eval.json"),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('prediction_len', int, 1),
        # Model arguments: encoder
        ('backbone', str, "siglip2"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        ('enable_interleaved_point_fusion', str2bool, False),
        ('interleaved_before_language', str2bool, False),
        ('interleaved_text_between_self_and_geo', str2bool, False),
        ('interleaved_cross_overlap_only', str2bool, False),
        ('interleaved_point_fusion_layers', int, 2),
        ('interleaved_semantic_mode', str, "cross_view_reproject"),
        ('interleaved_cross_knn', int, 16),
        ('interleaved_cross_residual_scale', float, 0.1),
        ('interleaved_knn_chunk_size', int, 256),
        ('interleaved_cross_view_radius', int, 1),
        ('interleaved_cross_view_max_reproj_error', float, 1.5),
        ('interleaved_cross_view_max_depth_error', float, 0.05),
        ('pi3x_geometry_mode', str, "interleaved"),
        ('geo_upsample_mode', str, "nearest_copy"),
        ('pointwise_parallel_branch_fusion', str2bool, False),
        ('pointwise_parallel_branch_text_last', str2bool, False),
        ('interleaved_semantic_use_pos', str2bool, True),
        ('disable_semantic_fusion', str2bool, False),
        ('disable_geometry_fusion', str2bool, False),
        ('lv2_batch_size', int, 1),
        ('pi3_root', str_none, None),
        ('pi3_ckpt', str_none, None),
        ('scene_geometry_source', str, "raw_pcd"),
        ('online_semantic_corr_source', str_none, None),
        ('vggt_root', str_none, None),
        ('vggt_ckpt', str_none, None),
        ('vggt_preprocess_mode', str, "crop"),
        ('vggt_image_size', int, 518),
        ('vggt_align_mode', str, "pose_ransac"),
        ('pi3_use_pose_prior', str2bool, True),
        ('pi3_rlbench_to_opencv', str2bool, True),
        # Model arguments: encoder and head
        ('embedding_dim', int, 144),
        ('num_attn_heads', int, 9),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 0),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow")
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])
    return parser


def parse_arguments():
    return build_argument_parser().parse_args()


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    model_class = fetch_model_class(args.model_type)
    model = model_class(
        backbone=args.backbone,
        finetune_backbone=args.finetune_backbone,
        finetune_text_encoder=args.finetune_text_encoder,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        enable_interleaved_point_fusion=args.enable_interleaved_point_fusion,
        interleaved_before_language=args.interleaved_before_language,
        interleaved_text_between_self_and_geo=args.interleaved_text_between_self_and_geo,
        interleaved_cross_overlap_only=args.interleaved_cross_overlap_only,
        interleaved_point_fusion_layers=args.interleaved_point_fusion_layers,
        interleaved_semantic_mode=args.interleaved_semantic_mode,
        interleaved_cross_knn=args.interleaved_cross_knn,
        interleaved_cross_residual_scale=args.interleaved_cross_residual_scale,
        interleaved_knn_chunk_size=args.interleaved_knn_chunk_size,
        interleaved_cross_view_radius=args.interleaved_cross_view_radius,
        interleaved_cross_view_max_reproj_error=args.interleaved_cross_view_max_reproj_error,
        interleaved_cross_view_max_depth_error=args.interleaved_cross_view_max_depth_error,
        pi3x_geometry_mode=args.pi3x_geometry_mode,
        geo_upsample_mode=args.geo_upsample_mode,
        pointwise_parallel_branch_fusion=args.pointwise_parallel_branch_fusion,
        pointwise_parallel_branch_text_last=args.pointwise_parallel_branch_text_last,
        interleaved_semantic_use_pos=args.interleaved_semantic_use_pos,
        disable_semantic_fusion=args.disable_semantic_fusion,
        disable_geometry_fusion=args.disable_geometry_fusion,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        lv2_batch_size=args.lv2_batch_size,
    )

    # Load model weights
    model_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    missing, unexpected = model.load_state_dict(model_dict_weight, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    model.eval()

    return model.cuda()


if __name__ == "__main__":
    configure_torch_runtime()
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.environ.get("TRACE_ONLINE_ROLLOUT", "0") == "1":
        trace_path = str(Path(args.output_file).with_suffix(".trace.json"))
        os.environ["TRACE_OUTPUT_FILE"] = trace_path

    # Bimanual vs single-arm utils
    if args.bimanual:
        from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner
    elif "peract" in args.dataset.lower():
        from online_evaluation_rlbench.utils_with_rlbench import RLBenchEnv, Actioner
    else:
        from online_evaluation_rlbench.utils_with_hiveformer_rlbench import RLBenchEnv, Actioner

    # Dataset class (for getting cameras and tasks/variations)
    dataset_class = fetch_dataset_class(args.dataset)
    if args.apply_cameras is None:
        apply_cameras = tuple(dataset_class.cameras)
    else:
        apply_cameras = tuple(
            item.strip() for item in str(args.apply_cameras).split(",") if item.strip()
        )
    if args.actor_cameras is None:
        actor_cameras = None
    else:
        actor_cameras = tuple(
            item.strip() for item in str(args.actor_cameras).split(",") if item.strip()
        )

    if str(args.scene_geometry_source).strip().lower() in {"pi3x", "vggt"}:
        raise NotImplementedError(
            "This evaluation-only release supports raw_pcd online evaluation only. "
            "Set --scene_geometry_source raw_pcd."
        )
    pi3x_client = None

    # Load models
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
        # Evaluate - reload environment for each task (crashes otherwise)
        task_success_rates = {}
        for task_str in [args.task]:

            # Seeds - re-seed for each task
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

            scene_geometry_source = str(args.scene_geometry_source).strip().lower()
            need_raw_point_cloud = scene_geometry_source not in {"pi3x", "vggt"}
            # Load RLBench environment
            print("[phase] env_init_begin", flush=True)
            env = RLBenchEnv(
                data_path=args.data_dir,
                task_str=task_str,
                image_size=[int(x) for x in args.image_size.split(",")],
                apply_rgb=True,
                apply_pc=need_raw_point_cloud,
                headless=bool(args.headless),
                apply_cameras=apply_cameras,
                collision_checking=bool(args.collision_checking)
            )
            print("[phase] env_init_done", flush=True)

            # Evaluate
            eval_kwargs = dict(
                task_str=task_str,
                max_steps=args.max_steps,
                actioner=actioner,
                max_tries=args.max_tries,
                prediction_len=args.prediction_len,
                num_history=args.num_history,
            )
            if args.bimanual:
                eval_kwargs["variation"] = args.variation
                eval_kwargs["demo_amount"] = args.demo_amount
                eval_kwargs["from_episode_number"] = args.from_episode_number
            var_success_rates = env.evaluate_task_on_multiple_variations(**eval_kwargs)
            print()
            print(
                f"{task_str} variation success rates:",
                round_floats(var_success_rates)
            )
            print(
                f"{task_str} mean success rate:",
                round_floats(var_success_rates["mean"])
            )

            task_success_rates[task_str] = var_success_rates
            with open(args.output_file, "w") as f:
                json.dump(round_floats(task_success_rates), f, indent=4)
    finally:
        actioner.shutdown()
