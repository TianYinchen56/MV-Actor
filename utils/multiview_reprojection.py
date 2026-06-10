from __future__ import annotations

import math

import torch


def rlbench_to_opencv_torch(
    intrinsics: torch.Tensor,
    extrinsics_c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = intrinsics.clone()
    extrinsics_c2w = extrinsics_c2w.clone()
    intrinsics[..., 0, 0] = intrinsics[..., 0, 0].abs()
    intrinsics[..., 1, 1] = intrinsics[..., 1, 1].abs()
    flip = torch.eye(3, device=extrinsics_c2w.device, dtype=extrinsics_c2w.dtype)
    flip[0, 0] = -1.0
    flip[1, 1] = -1.0
    extrinsics_c2w[..., :3, :3] = extrinsics_c2w[..., :3, :3] @ flip
    return intrinsics, extrinsics_c2w


def scale_intrinsics_to_grid(
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    in_h, in_w = int(image_hw[0]), int(image_hw[1])
    out_h, out_w = int(grid_hw[0]), int(grid_hw[1])
    sx = float(out_w) / float(in_w)
    sy = float(out_h) / float(in_h)
    scaled = intrinsics.clone()
    scaled[..., 0, 0] *= sx
    scaled[..., 0, 2] *= sx
    scaled[..., 1, 1] *= sy
    scaled[..., 1, 2] *= sy
    return scaled


def world_to_camera(
    points_world: torch.Tensor,
    extrinsics_c2w: torch.Tensor,
) -> torch.Tensor:
    rotation = extrinsics_c2w[..., :3, :3]
    translation = extrinsics_c2w[..., :3, 3]
    return torch.einsum("...ij,...j->...i", rotation.transpose(-1, -2), points_world - translation)


def project_world_to_grid(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_c2w: torch.Tensor,
    image_hw: tuple[int, int],
    grid_hw: tuple[int, int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intrinsics_grid = scale_intrinsics_to_grid(intrinsics, image_hw=image_hw, grid_hw=grid_hw)
    points_cam = world_to_camera(points_world, extrinsics_c2w)
    z = points_cam[..., 2].clamp_min(eps)
    u = intrinsics_grid[..., 0, 0] * (points_cam[..., 0] / z) + intrinsics_grid[..., 0, 2]
    v = intrinsics_grid[..., 1, 1] * (points_cam[..., 1] / z) + intrinsics_grid[..., 1, 2]
    return u, v, points_cam[..., 2]


def _batched_gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = points.shape[0]
    batch_idx = torch.arange(batch, device=points.device).view(batch, 1, 1).expand_as(indices)
    return points[batch_idx, indices]


def build_cross_view_token_correspondence(
    query_xyz: torch.Tensor,
    memory_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_c2w: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    grid_hw: tuple[int, int],
    search_radius: int = 1,
    max_reproj_error: float = 1.5,
    max_depth_error: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_xyz.dim() != 4 or memory_xyz.dim() != 4:
        raise ValueError(
            f"Expected query_xyz/memory_xyz as (B, V, N, 3), got {tuple(query_xyz.shape)} and {tuple(memory_xyz.shape)}"
        )
    if query_xyz.shape != memory_xyz.shape:
        raise ValueError(f"query_xyz and memory_xyz must share shape, got {tuple(query_xyz.shape)} vs {tuple(memory_xyz.shape)}")

    batch, num_views, tokens_per_view, _ = query_xyz.shape
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if grid_h * grid_w != tokens_per_view:
        raise ValueError(f"grid {grid_hw} incompatible with tokens_per_view={tokens_per_view}")
    if num_views < 2:
        raise ValueError("Cross-view correspondence requires at least 2 views")

    device = query_xyz.device
    dtype = query_xyz.dtype

    offsets = torch.arange(-search_radius, search_radius + 1, device=device, dtype=torch.long)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_y = offset_y.reshape(1, 1, -1)
    offset_x = offset_x.reshape(1, 1, -1)
    num_candidates = int(offset_x.shape[-1])
    max_neighbors = num_views - 1

    neighbor_idx = torch.zeros((batch, num_views * tokens_per_view, max_neighbors), device=device, dtype=torch.long)
    neighbor_mask = torch.zeros((batch, num_views * tokens_per_view, max_neighbors), device=device, dtype=torch.bool)

    memory_cam = []
    for target_view in range(num_views):
        target_xyz = memory_xyz[:, target_view]
        target_cam = world_to_camera(target_xyz, extrinsics_c2w[:, target_view][:, None])
        memory_cam.append(target_cam)
    memory_cam = torch.stack(memory_cam, dim=1)

    for source_view in range(num_views):
        query_world = query_xyz[:, source_view]
        global_query_start = source_view * tokens_per_view
        slot = 0
        for target_view in range(num_views):
            if target_view == source_view:
                continue

            u, v, z = project_world_to_grid(
                query_world,
                intrinsics[:, target_view][:, None],
                extrinsics_c2w[:, target_view][:, None],
                image_hw=image_hw,
                grid_hw=grid_hw,
            )
            center_x = torch.round(u).to(torch.long)
            center_y = torch.round(v).to(torch.long)

            cand_x = center_x.unsqueeze(-1) + offset_x
            cand_y = center_y.unsqueeze(-1) + offset_y
            cand_x_clamped = cand_x.clamp(0, grid_w - 1)
            cand_y_clamped = cand_y.clamp(0, grid_h - 1)
            candidate_local_idx = cand_y_clamped * grid_w + cand_x_clamped

            target_cam = memory_cam[:, target_view]
            candidate_cam = _batched_gather_points(target_cam, candidate_local_idx)
            candidate_depth = candidate_cam[..., 2]

            reproj_err = torch.sqrt(
                (cand_x_clamped.to(dtype) - u.unsqueeze(-1)) ** 2 +
                (cand_y_clamped.to(dtype) - v.unsqueeze(-1)) ** 2
            )
            depth_err = (candidate_depth - z.unsqueeze(-1)).abs()
            valid = (
                (z.unsqueeze(-1) > 0.0)
                & (candidate_depth > 0.0)
                & (reproj_err <= max_reproj_error)
                & (depth_err <= max_depth_error)
            )

            score = depth_err + 0.01 * reproj_err
            score = torch.where(valid, score, torch.full_like(score, float("inf")))
            best_local = score.argmin(dim=-1, keepdim=True)
            best_score = torch.gather(score, dim=-1, index=best_local).squeeze(-1)
            best_valid = torch.isfinite(best_score)
            best_idx_local = torch.gather(candidate_local_idx, dim=-1, index=best_local).squeeze(-1)
            best_idx_global = target_view * tokens_per_view + best_idx_local

            neighbor_idx[:, global_query_start:global_query_start + tokens_per_view, slot] = best_idx_global
            neighbor_mask[:, global_query_start:global_query_start + tokens_per_view, slot] = best_valid
            slot += 1

    return neighbor_idx, neighbor_mask


def select_balanced_anchor_indices(valid_mask: torch.Tensor, grid_hw: tuple[int, int], max_points: int) -> torch.Tensor:
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    flat = valid_mask.flatten()
    if flat.sum().item() == 0:
        return torch.empty((0,), device=valid_mask.device, dtype=torch.long)

    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=valid_mask.device),
        torch.arange(grid_w, device=valid_mask.device),
        indexing="ij",
    )
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)
    valid_coords = coords[flat]
    valid_indices = flat.nonzero(as_tuple=False).squeeze(-1)

    side = max(1, int(math.ceil(math.sqrt(max_points))))
    bins_y = torch.clamp((valid_coords[:, 0].float() / max(grid_h, 1) * side).floor().to(torch.long), 0, side - 1)
    bins_x = torch.clamp((valid_coords[:, 1].float() / max(grid_w, 1) * side).floor().to(torch.long), 0, side - 1)
    bins = bins_y * side + bins_x

    selected = []
    for bin_idx in range(side * side):
        members = valid_indices[bins == bin_idx]
        if members.numel() > 0:
            selected.append(members[members.numel() // 2])
    if not selected:
        return valid_indices[:max_points]

    selected = torch.stack(selected)
    if selected.numel() > max_points:
        step = max(1, selected.numel() // max_points)
        selected = selected[::step][:max_points]
    return selected
