from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim3_ransac import apply_sim3_to_points, estimate_pose_aware_sim3_ransac, transform_c2w_with_sim3


def _rss_gb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 3)


def _cuda_mem_gb(device: torch.device) -> tuple[float, float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    alloc = float(torch.cuda.memory_allocated(dev_idx)) / (1024 ** 3)
    reserved = float(torch.cuda.memory_reserved(dev_idx)) / (1024 ** 3)
    peak = float(torch.cuda.max_memory_allocated(dev_idx)) / (1024 ** 3)
    return alloc, reserved, peak


def _log_mem(prefix: str, device: torch.device) -> None:
    alloc, reserved, peak = _cuda_mem_gb(device)
    print(
        f"[pi3x_worker][mem] {prefix} rss_gb={_rss_gb():.3f} "
        f"cuda_alloc_gb={alloc:.3f} cuda_reserved_gb={reserved:.3f} cuda_peak_gb={peak:.3f}",
        flush=True,
    )


def _arrays_nbytes(payload: dict[str, np.ndarray]) -> int:
    return sum(int(value.nbytes) for value in payload.values() if isinstance(value, np.ndarray))


def _otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    vmin = float(values.min())
    vmax = float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
        return vmax
    hist, edges = np.histogram(values, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_total = hist.sum()
    sum_total = float((hist * centers).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_score = -1.0
    best_thr = centers[0]
    for idx in range(hist.shape[0]):
        weight_bg += hist[idx]
        if weight_bg <= 0:
            continue
        weight_fg = weight_total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += hist[idx] * centers[idx]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        score = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if score > best_score:
            best_score = score
            best_thr = centers[idx]
    return float(best_thr)


def _resize_depths_np(depths_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    depths_np = np.asarray(depths_np, dtype=np.float32)
    batch, views, height, width = depths_np.shape
    if height == target_h and width == target_w:
        return depths_np.astype(np.float32, copy=False)
    depths_t = torch.from_numpy(depths_np.reshape(batch * views, 1, height, width))
    resized = F.interpolate(depths_t, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=False)
    return resized.reshape(batch, views, target_h, target_w).numpy().astype(np.float32)


def _world_to_camera_points(world_points: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    world_points = np.asarray(world_points, dtype=np.float32)
    height, width, _ = world_points.shape
    world_h = np.concatenate([world_points, np.ones((height, width, 1), dtype=np.float32)], axis=-1)
    w2c = np.linalg.inv(np.asarray(pose_c2w, dtype=np.float32))
    cam = world_h @ w2c.T
    return cam[..., :3].astype(np.float32)


def _camera_to_world_points(camera_points: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    camera_points = np.asarray(camera_points, dtype=np.float32)
    pose_c2w = np.asarray(pose_c2w, dtype=np.float32)
    world = (pose_c2w[:3, :3] @ camera_points.reshape(-1, 3).T).T.reshape(camera_points.shape)
    world = world + pose_c2w[:3, 3].reshape(1, 1, 3)
    return world.astype(np.float32)


def _safe_estimate_alignment(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    pred_c2w = np.asarray(pred_c2w, dtype=np.float32)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float32)
    valid = np.isfinite(pred_c2w).all(axis=(1, 2)) & np.isfinite(gt_c2w).all(axis=(1, 2))
    pred_valid = pred_c2w[valid]
    gt_valid = gt_c2w[valid]
    if pred_valid.shape[0] < 3:
        raise ValueError(f"Need at least 3 valid pose pairs for pose-aware Sim3 RANSAC, got {pred_valid.shape[0]}")
    scale, rot, trans, _diag = estimate_pose_aware_sim3_ransac(pred_valid, gt_valid)
    return float(scale), rot.astype(np.float32), trans.astype(np.float32)


def _apply_pose_aware_sim3(
    pred_pointmaps_world: np.ndarray,
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    pred_pointmaps_world = np.asarray(pred_pointmaps_world, dtype=np.float32)
    pred_c2w = np.asarray(pred_c2w, dtype=np.float32)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float32)
    scale, rot, trans = _safe_estimate_alignment(pred_c2w, gt_c2w)
    pointmaps_metric = apply_sim3_to_points(pred_pointmaps_world, scale, rot, trans).astype(np.float32)
    pred_metric_c2w = transform_c2w_with_sim3(pred_c2w, scale, rot, trans).astype(np.float32)
    diag = {"scale": float(scale), "rotation": rot.astype(np.float32).tolist(), "translation": trans.astype(np.float32).tolist()}
    return pointmaps_metric, pred_metric_c2w, diag


def _align_with_gt_fg_scale(
    pred_pointmaps_world: np.ndarray,
    pred_c2w: np.ndarray,
    gt_depths: np.ndarray,
    gt_c2w: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    pred_pointmaps_world = np.asarray(pred_pointmaps_world, dtype=np.float32)
    pred_c2w = np.asarray(pred_c2w, dtype=np.float32)
    gt_depths = np.asarray(gt_depths, dtype=np.float32)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float32)
    if pred_pointmaps_world.shape[0] != gt_depths.shape[0]:
        raise ValueError(f"GT-depth/view mismatch: pred_views={pred_pointmaps_world.shape[0]} gt_depth_views={gt_depths.shape[0]}")
    aligned = []
    diag = []
    for view_idx in range(pred_pointmaps_world.shape[0]):
        pred_pose = pred_c2w[view_idx]
        gt_pose = gt_c2w[view_idx]
        if not np.isfinite(pred_pose).all() or not np.isfinite(gt_pose).all():
            raise RuntimeError(f"Non-finite pose at view_idx={view_idx}")
        pred_cam = _world_to_camera_points(pred_pointmaps_world[view_idx], pred_pose)
        pred_depth = pred_cam[..., 2]
        gt_depth = np.asarray(gt_depths[view_idx], dtype=np.float32)
        valid = np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 1e-6) & (pred_depth > 1e-6)
        if not np.any(valid):
            raise RuntimeError(f"No valid GT/pred depth overlap at view_idx={view_idx}")
        threshold = _otsu_threshold(gt_depth[valid])
        foreground = valid & (gt_depth <= threshold)
        if not np.any(foreground):
            raise RuntimeError(f"No GT foreground after Otsu at view_idx={view_idx}")
        gt_median = float(np.median(gt_depth[foreground]))
        pred_median = float(np.median(pred_depth[foreground]))
        if not np.isfinite(pred_median) or pred_median <= 1e-8:
            raise RuntimeError(f"Invalid predicted FG median at view_idx={view_idx}: {pred_median}")
        scale = float(gt_median / pred_median)
        scaled_cam = pred_cam * scale
        aligned_world = _camera_to_world_points(scaled_cam, gt_pose)
        invalid = ~np.isfinite(pred_cam).all(axis=-1) | ~np.isfinite(gt_depth)
        aligned_world[invalid] = np.nan
        aligned.append(aligned_world.astype(np.float32))
        diag.append(
            {
                "view_idx": int(view_idx),
                "depth_threshold": float(threshold),
                "fg_count": int(foreground.sum()),
                "gt_fg_median_depth": gt_median,
                "pred_fg_median_depth": pred_median,
                "scale": scale,
            }
        )
    return np.stack(aligned, axis=0).astype(np.float32), diag


def _add_pi3_to_path(pi3_root: str) -> None:
    candidates = [pi3_root, os.path.join(pi3_root, "pi3")]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def load_pi3x_model(pi3_root: str, ckpt: str, device: torch.device):
    pi3_root_path = Path(pi3_root)
    ckpt_path = Path(ckpt)
    if not pi3_root_path.exists():
        raise FileNotFoundError(f"Pi3 code directory does not exist: {pi3_root_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pi3X checkpoint does not exist: {ckpt_path}")
    _add_pi3_to_path(str(pi3_root_path))
    from pi3.models.pi3x import Pi3X

    model = Pi3X().eval()
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_model

        model = model.to(device)
        missing_keys, unexpected_keys = load_model(model, str(ckpt_path), strict=False, device=str(device))
    else:
        weights = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    if missing_keys:
        print(f"[pi3x] missing keys: {len(missing_keys)}", flush=True)
    if unexpected_keys:
        print(f"[pi3x] unexpected keys: {len(unexpected_keys)}", flush=True)
    return model


def _compute_target_hw(height: int, width: int, patch_size: int = 14) -> tuple[int, int]:
    target_width = max(patch_size, round(width / patch_size) * patch_size)
    target_height = max(patch_size, round(height / patch_size) * patch_size)
    return int(target_height), int(target_width)


def _rlbench_to_opencv_np(intrinsics: np.ndarray, poses_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = intrinsics.copy()
    poses_c2w = poses_c2w.copy()
    intrinsics[..., 0, 0] = np.abs(intrinsics[..., 0, 0])
    intrinsics[..., 1, 1] = np.abs(intrinsics[..., 1, 1])
    flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    poses_c2w[..., :3, :3] = poses_c2w[..., :3, :3] @ flip
    return intrinsics, poses_c2w


def build_rays_from_intrinsics(
    intrinsics_bv33: torch.Tensor,
    height: int,
    width: int,
    *,
    abs_focal: bool = True,
) -> torch.Tensor:
    fx = intrinsics_bv33[..., 0, 0]
    fy = intrinsics_bv33[..., 1, 1]
    if abs_focal:
        fx = fx.abs()
        fy = fy.abs()
    cx = intrinsics_bv33[..., 0, 2]
    cy = intrinsics_bv33[..., 1, 2]
    batch, views = fx.shape
    u = torch.arange(width, device=intrinsics_bv33.device, dtype=intrinsics_bv33.dtype)[None, None, None, :].expand(batch, views, height, width)
    v = torch.arange(height, device=intrinsics_bv33.device, dtype=intrinsics_bv33.dtype)[None, None, :, None].expand(batch, views, height, width)
    x_over_z = (u - cx[..., None, None]) / fx.clamp_min(1e-6)[..., None, None]
    y_over_z = (v - cy[..., None, None]) / fy.clamp_min(1e-6)[..., None, None]
    z = torch.ones_like(x_over_z)
    return torch.stack([x_over_z, y_over_z, z], dim=-1)


def resize_xyz_map(xyz_bv3hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    batch, views, channels, _, _ = xyz_bv3hw.shape
    xyz = F.interpolate(xyz_bv3hw.flatten(0, 1), size=(out_h, out_w), mode="bilinear", align_corners=False)
    return xyz.reshape(batch, views, channels, out_h, out_w).contiguous()


def patch_pool_xyz_tokens(xyz_bv3hw: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    batch, views, channels, height, width = xyz_bv3hw.shape
    if height % grid_h != 0 or width % grid_w != 0:
        raise ValueError(f"xyz map {(height, width)} cannot be patch-pooled to grid {(grid_h, grid_w)}")
    patch_h = height // grid_h
    patch_w = width // grid_w
    xyz = xyz_bv3hw.reshape(batch, views, channels, grid_h, patch_h, grid_w, patch_w)
    xyz = xyz.mean(dim=(4, 6))
    xyz = xyz.permute(0, 1, 3, 4, 2).reshape(batch, views, grid_h * grid_w, channels)
    return xyz.contiguous()


@torch.no_grad()
def extract_pi3x_tokens(
    model: torch.nn.Module,
    rgb_bvchw: torch.Tensor,
    *,
    intrinsics: torch.Tensor | None = None,
    rays: torch.Tensor | None = None,
    poses_c2w: torch.Tensor | None = None,
    use_point_decoder: bool = True,
    with_prior: bool = True,
) -> tuple[torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
    batch, views, _, height, width = rgb_bvchw.shape
    patch_size = int(getattr(model, "patch_size", 14))
    target_h, target_w = _compute_target_hw(height, width, patch_size=patch_size)
    scale_y = float(target_h) / float(height)
    scale_x = float(target_w) / float(width)

    x = rgb_bvchw.float()
    if (target_h, target_w) != (height, width):
        x = F.interpolate(x.flatten(0, 1), size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True)
        x = x.reshape(batch, views, 3, target_h, target_w)

    mean = model.image_mean.to(x.device)
    std = model.image_std.to(x.device)
    x = (x - mean) / std

    scaled_intrinsics = None
    if intrinsics is not None:
        scaled_intrinsics = intrinsics.clone().to(x.device)
        if (target_h, target_w) != (height, width):
            scaled_intrinsics[..., 0, 0] *= scale_x
            scaled_intrinsics[..., 0, 2] *= scale_x
            scaled_intrinsics[..., 1, 1] *= scale_y
            scaled_intrinsics[..., 1, 2] *= scale_y

    poses = poses_c2w.to(x.device) if poses_c2w is not None else None
    rays = rays.to(x.device) if rays is not None else None

    hidden, poses_rel, _use_depth_mask, use_pose_mask, _norm = model.encode(
        x,
        with_prior=with_prior,
        depths=None,
        intrinsics=scaled_intrinsics,
        poses=poses,
        rays=rays,
    )
    hidden_4d = hidden.reshape(batch, views, -1, hidden.shape[-1])
    tokens_dec_full, pos_full = model.decode(hidden_4d, views, target_h, target_w, poses_rel, use_pose_mask)
    dec_embed_dim = int(getattr(model, "dec_embed_dim", tokens_dec_full.shape[-1] // 2))
    tokens_dec_last = tokens_dec_full[..., -dec_embed_dim:]
    patch_start = int(getattr(model, "patch_start_idx", 0))
    patch_h = target_h // patch_size
    patch_w = target_w // patch_size
    tokens_per_view = patch_h * patch_w + patch_start

    ret_point = model.point_decoder(tokens_dec_full, xpos=pos_full)
    if use_point_decoder:
        tokens_patch = ret_point[:, patch_start:]
    else:
        tokens_patch = tokens_dec_last[:, patch_start:]
    tokens_patch = tokens_patch.reshape(batch, views, patch_h * patch_w, tokens_patch.shape[-1]).contiguous()

    ret_camera = model.camera_decoder(tokens_dec_full, xpos=pos_full)
    pos_hw = pos_full.reshape(batch, views * tokens_per_view, -1)
    ret_metric = model.metric_decoder(
        model.metric_token.repeat(batch, 1, 1),
        tokens_dec_full.reshape(batch, views * tokens_per_view, -1),
        xpos=pos_hw[:, 0:1],
        ypos=pos_hw,
    )

    point_head_dtype = next(model.point_head.parameters()).dtype
    camera_head_dtype = next(model.camera_head.parameters()).dtype
    metric_head_dtype = next(model.metric_head.parameters()).dtype
    with torch.amp.autocast(device_type="cuda", enabled=False):
        xy, z = model.point_head(ret_point[:, patch_start:].to(dtype=point_head_dtype), patch_h=patch_h, patch_w=patch_w)
        xy = xy.float().permute(0, 2, 3, 1).reshape(batch, views, target_h, target_w, -1)
        z = z.float().permute(0, 2, 3, 1).reshape(batch, views, target_h, target_w, -1)
        z = torch.exp(z.clamp(max=15.0))
        local_points = torch.cat([xy * z, z], dim=-1)
        camera_poses = model.camera_head(ret_camera[:, patch_start:].to(dtype=camera_head_dtype), patch_h, patch_w)
        camera_poses = camera_poses.reshape(batch, views, 4, 4).float()
        metric = model.metric_head(ret_metric.to(dtype=metric_head_dtype)).reshape(batch).float().exp()
        ones = torch.ones((*local_points.shape[:-1], 1), device=local_points.device, dtype=local_points.dtype)
        points = torch.einsum("bnij,bnhwj->bnhwi", camera_poses, torch.cat([local_points, ones], dim=-1))[..., :3]
        points = points * metric.view(batch, 1, 1, 1, 1)
        camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(batch, 1, 1)
    return tokens_patch, patch_h, patch_w, points.contiguous(), camera_poses.contiguous()


class Pi3XWorkerSession:
    def __init__(
        self,
        *,
        pi3_root: str,
        pi3_ckpt: str,
        device: str,
        use_pi3_pose_prior: bool,
        pi3_rlbench_to_opencv: bool,
        configure_torch_threads: bool = True,
    ):
        if configure_torch_threads:
            torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
            if torch_num_threads is not None:
                torch.set_num_threads(max(1, int(torch_num_threads)))
            torch_num_interop_threads = os.environ.get("TORCH_NUM_INTEROP_THREADS")
            if torch_num_interop_threads is not None:
                torch.set_num_interop_threads(max(1, int(torch_num_interop_threads)))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_pi3_pose_prior = bool(use_pi3_pose_prior)
        self.pi3_rlbench_to_opencv = bool(pi3_rlbench_to_opencv)
        self.request_count = 0
        print(
            json.dumps(
                {
                    "event": "worker_init_begin",
                    "pid": int(os.getpid()),
                    "device_arg": str(device),
                    "runtime_device": str(self.device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
                    "torch_num_threads": int(torch.get_num_threads()),
                    "torch_num_interop_threads": int(torch.get_num_interop_threads()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _log_mem("before_model_load", self.device)
        self.pi3_model = load_pi3x_model(pi3_root, pi3_ckpt, self.device)
        _log_mem("after_model_load", self.device)

    def _build_semantic_xyz_tokens_from_pi3x_dense(
        self,
        xyz_dense: torch.Tensor,
        *,
        sem_h: int,
        sem_w: int,
        backbone_name: str,
        clip_semantic_xyz_mode: str,
        siglip_image_size: int,
    ) -> torch.Tensor:
        if backbone_name == "clip" and clip_semantic_xyz_mode == "match_train_cache":
            semantic_xyz_dense = resize_xyz_map(xyz_dense, sem_h, sem_w)
            return semantic_xyz_dense.permute(0, 1, 3, 4, 2).reshape(
                xyz_dense.shape[0],
                xyz_dense.shape[1],
                sem_h * sem_w,
                3,
            ).contiguous()
        semantic_xyz_dense = resize_xyz_map(xyz_dense, siglip_image_size, siglip_image_size)
        return patch_pool_xyz_tokens(semantic_xyz_dense, sem_h, sem_w)

    @torch.no_grad()
    def build_online_inputs(
        self,
        *,
        rgbs_np: np.ndarray,
        camera_intrinsics_np: np.ndarray,
        camera_extrinsics_np: np.ndarray,
        camera_gt_depths_np: np.ndarray,
        out_h: int,
        out_w: int,
        sem_h: int,
        sem_w: int,
        backbone_name: str,
        clip_semantic_xyz_mode: str,
        siglip_image_size: int,
        disable_geometry_tokens: bool,
    ) -> dict[str, np.ndarray]:
        self.request_count += 1
        req_id = self.request_count
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        print(
            f"[pi3x_worker] request={req_id} kind=build_online_inputs "
            f"rgbs_shape={tuple(rgbs_np.shape)} intr_shape={tuple(camera_intrinsics_np.shape)} "
            f"extr_shape={tuple(camera_extrinsics_np.shape)} gt_depth_shape={tuple(camera_gt_depths_np.shape)} in_bytes_mb="
            f"{(rgbs_np.nbytes + camera_intrinsics_np.nbytes + camera_extrinsics_np.nbytes + camera_gt_depths_np.nbytes) / (1024 ** 2):.2f}",
            flush=True,
        )
        _log_mem(f"request={req_id} build_begin", self.device)
        rgbs = torch.from_numpy(rgbs_np).to(self.device).float()
        intrinsics_np = np.asarray(camera_intrinsics_np, dtype=np.float32)
        extrinsics_np = np.asarray(camera_extrinsics_np, dtype=np.float32)
        if self.pi3_rlbench_to_opencv:
            intrinsics_np, extrinsics_np = _rlbench_to_opencv_np(intrinsics_np, extrinsics_np)

        intrinsics_t = torch.from_numpy(intrinsics_np).to(self.device).float()
        extrinsics_t = torch.from_numpy(extrinsics_np).to(self.device).float()
        _, _, _, in_h, in_w = rgbs.shape
        patch_size = int(getattr(self.pi3_model, "patch_size", 14))
        target_h, target_w = _compute_target_hw(in_h, in_w, patch_size=patch_size)
        gt_depths_np = _resize_depths_np(camera_gt_depths_np, target_h=target_h, target_w=target_w)
        intrinsics_scaled = intrinsics_t.clone()
        intrinsics_scaled[..., 0, 0] *= float(target_w) / float(in_w)
        intrinsics_scaled[..., 0, 2] *= float(target_w) / float(in_w)
        intrinsics_scaled[..., 1, 1] *= float(target_h) / float(in_h)
        intrinsics_scaled[..., 1, 2] *= float(target_h) / float(in_h)
        rays = build_rays_from_intrinsics(intrinsics_scaled, height=target_h, width=target_w, abs_focal=True)

        pi3_tokens, patch_h, patch_w, pi3_points, pi3_cameras = extract_pi3x_tokens(
            self.pi3_model,
            rgbs,
            intrinsics=None,
            rays=rays,
            poses_c2w=extrinsics_t if self.use_pi3_pose_prior else None,
            use_point_decoder=False,
            with_prior=True,
        )
        _log_mem(f"request={req_id} after_extract_tokens", self.device)

        xyz_aligned = []
        align_diag = []
        for batch_idx in range(rgbs.shape[0]):
            pred_c2w = pi3_cameras[batch_idx].detach().cpu().numpy().astype(np.float32)
            gt_c2w = extrinsics_np[batch_idx].astype(np.float32)
            metric_np, metric_c2w, sim3_diag = _apply_pose_aware_sim3(
                pi3_points[batch_idx].detach().cpu().numpy().astype(np.float32),
                pred_c2w,
                gt_c2w,
            )
            aligned_np, fg_diag = _align_with_gt_fg_scale(metric_np, metric_c2w, gt_depths_np[batch_idx], gt_c2w)
            xyz_aligned.append(torch.from_numpy(aligned_np).to(pi3_points.device, dtype=pi3_points.dtype))
            align_diag.append({"sim3": sim3_diag, "fg_scale": fg_diag})

        xyz_dense = torch.stack(xyz_aligned, dim=0).permute(0, 1, 4, 2, 3).contiguous()
        xyz_tokens = patch_pool_xyz_tokens(xyz_dense, patch_h, patch_w)
        semantic_xyz_tokens = self._build_semantic_xyz_tokens_from_pi3x_dense(
            xyz_dense,
            sem_h=sem_h,
            sem_w=sem_w,
            backbone_name=str(backbone_name),
            clip_semantic_xyz_mode=str(clip_semantic_xyz_mode),
            siglip_image_size=int(siglip_image_size),
        )
        scene_pcds = resize_xyz_map(xyz_dense, int(out_h), int(out_w))
        outputs = {
            "online_model_pcds": scene_pcds.detach().cpu().numpy().astype(np.float32),
            "token_xyz_pi3_sim3": xyz_tokens.detach().cpu().numpy().astype(np.float32),
            "token_xyz_siglip2_sim3": semantic_xyz_tokens.detach().cpu().numpy().astype(np.float32),
            "pi3x_target_hw": np.asarray([target_h, target_w], dtype=np.int32),
            "pi3x_token_grid_hw": np.asarray([patch_h, patch_w], dtype=np.int32),
            "gt_fg_scale_diag": np.asarray(json.dumps(align_diag), dtype=np.str_),
        }
        if not disable_geometry_tokens:
            outputs["pi3x_tokens"] = pi3_tokens.detach().cpu().numpy().astype(np.float32)
        print(
            f"[pi3x_worker] request={req_id} kind=build_online_inputs out_bytes_mb={_arrays_nbytes(outputs) / (1024 ** 2):.2f}",
            flush=True,
        )
        _log_mem(f"request={req_id} before_cleanup", self.device)
        del rgbs, intrinsics_t, extrinsics_t, intrinsics_scaled, rays, pi3_tokens, pi3_points, pi3_cameras
        del xyz_dense, xyz_tokens, semantic_xyz_tokens, scene_pcds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_mem(f"request={req_id} after_cleanup", self.device)
        return outputs
