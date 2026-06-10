from __future__ import annotations

from itertools import combinations

import numpy as np


RANSAC_SUBSET_SIZE = 3
RANSAC_TRANS_THRESH_M = 0.20
RANSAC_ROT_THRESH_DEG = 15.0


def vggt_w2c34_to_c2w44(extrinsics_v34: np.ndarray) -> np.ndarray:
    extrinsics_v34 = np.asarray(extrinsics_v34, dtype=np.float32)
    if extrinsics_v34.ndim != 3 or extrinsics_v34.shape[1:] != (3, 4):
        raise ValueError(f"Expected VGGT extrinsics with shape (V,3,4), got {extrinsics_v34.shape}")
    rot = extrinsics_v34[:, :3, :3]
    trans = extrinsics_v34[:, :3, 3]
    rot_t = np.transpose(rot, (0, 2, 1))
    cam_center = -np.einsum("vij,vj->vi", rot_t, trans)
    c2w = np.tile(np.eye(4, dtype=np.float32)[None], (extrinsics_v34.shape[0], 1, 1))
    c2w[:, :3, :3] = rot_t.astype(np.float32)
    c2w[:, :3, 3] = cam_center.astype(np.float32)
    return c2w


def apply_sim3_to_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    rot = np.asarray(rotation, dtype=np.float32)
    trans = np.asarray(translation, dtype=np.float32)
    return float(scale) * np.einsum("ij,...j->...i", rot, pts) + trans


def transform_c2w_with_sim3(c2w: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected c2w pose(s) with shape (...,4,4), got {c2w.shape}")
    rot_old = c2w[..., :3, :3]
    trans_old = c2w[..., :3, 3]
    rot = np.asarray(rotation, dtype=np.float32)
    trans = np.asarray(translation, dtype=np.float32)
    rot_new = np.einsum("ij,...jk->...ik", rot, rot_old).astype(np.float32)
    trans_new = apply_sim3_to_points(trans_old, scale, rot, trans).astype(np.float32)
    out = np.broadcast_to(np.eye(4, dtype=np.float32), c2w.shape).copy()
    out[..., :3, :3] = rot_new
    out[..., :3, 3] = trans_new
    return out


def invert_sim3(scale: float, rotation: np.ndarray, translation: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Cannot invert Sim3 with invalid scale: {scale}")
    rot = np.asarray(rotation, dtype=np.float32)
    trans = np.asarray(translation, dtype=np.float32)
    inv_scale = 1.0 / scale
    inv_rot = rot.T.astype(np.float32)
    inv_trans = (-inv_scale * (inv_rot @ trans)).astype(np.float32)
    return float(inv_scale), inv_rot, inv_trans


def geodesic_rotation_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", rot_b, rot_a)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def estimate_global_rotation_from_pose_axes(src_pose: np.ndarray, dst_pose: np.ndarray) -> np.ndarray:
    src_rot = np.asarray(src_pose, dtype=np.float64)[..., :3, :3]
    dst_rot = np.asarray(dst_pose, dtype=np.float64)[..., :3, :3]
    src_axes = np.transpose(src_rot, (0, 2, 1)).reshape(-1, 3)
    dst_axes = np.transpose(dst_rot, (0, 2, 1)).reshape(-1, 3)
    covariance = dst_axes.T @ src_axes
    u, _, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    return rotation.astype(np.float32)


def estimate_pose_aware_sim3(src_pose: np.ndarray, dst_pose: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_pose = np.asarray(src_pose, dtype=np.float32)
    dst_pose = np.asarray(dst_pose, dtype=np.float32)
    if src_pose.shape != dst_pose.shape or src_pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected matching pose arrays with shape (N,4,4), got {src_pose.shape} and {dst_pose.shape}")
    if src_pose.shape[0] < 2:
        raise ValueError(f"Need at least 2 poses for Sim3, got {src_pose.shape[0]}")
    rotation = estimate_global_rotation_from_pose_axes(src_pose, dst_pose).astype(np.float64)
    src_centers = src_pose[:, :3, 3].astype(np.float64)
    dst_centers = dst_pose[:, :3, 3].astype(np.float64)
    mu_src = src_centers.mean(axis=0)
    mu_dst = dst_centers.mean(axis=0)
    src_centered = src_centers - mu_src
    dst_centered = dst_centers - mu_dst
    rotated_src = (rotation @ src_centered.T).T
    numerator = float(np.sum(rotated_src * dst_centered))
    denominator = float(np.sum(src_centered * src_centered))
    if denominator <= 1e-12:
        raise ValueError(f"Degenerate source camera centers for Sim3, denominator={denominator}")
    scale = numerator / denominator
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Invalid pose-aware Sim3 scale: {scale}")
    translation = mu_dst - scale * (rotation.astype(np.float64) @ mu_src)
    return float(scale), rotation.astype(np.float32), translation.astype(np.float32)


def evaluate_pose_aware_sim3(
    src_pose: np.ndarray,
    dst_pose: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_centers = apply_sim3_to_points(src_pose[:, :3, 3], scale, rotation, translation).astype(np.float32)
    gt_centers = dst_pose[:, :3, 3].astype(np.float32)
    center_err_m = np.linalg.norm(pred_centers - gt_centers, axis=1).astype(np.float32)
    pred_rot = np.einsum("ij,njk->nik", rotation.astype(np.float32), src_pose[:, :3, :3].astype(np.float32)).astype(np.float32)
    rot_err_deg = geodesic_rotation_deg(pred_rot, dst_pose[:, :3, :3].astype(np.float32))
    return center_err_m, rot_err_deg


def estimate_pose_aware_sim3_ransac(src_pose: np.ndarray, dst_pose: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    src_pose = np.asarray(src_pose, dtype=np.float32)
    dst_pose = np.asarray(dst_pose, dtype=np.float32)
    fit_indices = np.arange(src_pose.shape[0], dtype=np.int64)
    if src_pose.shape[0] < RANSAC_SUBSET_SIZE:
        raise ValueError(f"Need at least {RANSAC_SUBSET_SIZE} poses for RANSAC, got {src_pose.shape[0]}")
    if src_pose.shape[0] == RANSAC_SUBSET_SIZE:
        scale, rotation, translation = estimate_pose_aware_sim3(src_pose, dst_pose)
        center_err_m, rot_err_deg = evaluate_pose_aware_sim3(src_pose, dst_pose, scale, rotation, translation)
        inliers = (center_err_m <= RANSAC_TRANS_THRESH_M) & (rot_err_deg <= RANSAC_ROT_THRESH_DEG)
        diagnostics = {
            "fit_indices": fit_indices.astype(np.int32).tolist(),
            "ransac_subset_size": int(RANSAC_SUBSET_SIZE),
            "ransac_trans_thresh_m": float(RANSAC_TRANS_THRESH_M),
            "ransac_rot_thresh_deg": float(RANSAC_ROT_THRESH_DEG),
            "best_subset_indices": fit_indices.astype(np.int32).tolist(),
            "best_subset_inlier_count": int(np.sum(inliers)),
            "refine_inlier_indices": np.flatnonzero(inliers).astype(np.int32).tolist(),
            "center_err_m": center_err_m.astype(np.float32).tolist(),
            "rot_err_deg": rot_err_deg.astype(np.float32).tolist(),
        }
        return float(scale), rotation.astype(np.float32), translation.astype(np.float32), diagnostics

    best = None
    for subset in combinations(range(src_pose.shape[0]), RANSAC_SUBSET_SIZE):
        subset_idx = np.asarray(subset, dtype=np.int64)
        try:
            cand_scale, cand_rot, cand_trans = estimate_pose_aware_sim3(src_pose[subset_idx], dst_pose[subset_idx])
        except ValueError:
            continue
        center_err_m, rot_err_deg = evaluate_pose_aware_sim3(src_pose, dst_pose, cand_scale, cand_rot, cand_trans)
        inliers = (center_err_m <= RANSAC_TRANS_THRESH_M) & (rot_err_deg <= RANSAC_ROT_THRESH_DEG)
        inlier_count = int(np.sum(inliers))
        robust_score = float(
            np.mean(
                np.minimum(center_err_m / RANSAC_TRANS_THRESH_M, 5.0)
                + np.minimum(rot_err_deg / RANSAC_ROT_THRESH_DEG, 5.0)
            )
        )
        tie_break = float(np.mean(center_err_m) + 0.01 * np.mean(rot_err_deg))
        candidate = {
            "subset_idx": subset_idx,
            "scale": float(cand_scale),
            "rotation": cand_rot.astype(np.float32),
            "translation": cand_trans.astype(np.float32),
            "inliers": inliers,
            "inlier_count": inlier_count,
            "robust_score": robust_score,
            "tie_break": tie_break,
        }
        if best is None:
            best = candidate
            continue
        if candidate["inlier_count"] > best["inlier_count"]:
            best = candidate
            continue
        if candidate["inlier_count"] == best["inlier_count"] and candidate["robust_score"] < best["robust_score"] - 1e-8:
            best = candidate
            continue
        if (
            candidate["inlier_count"] == best["inlier_count"]
            and abs(candidate["robust_score"] - best["robust_score"]) <= 1e-8
            and candidate["tie_break"] < best["tie_break"]
        ):
            best = candidate

    if best is None:
        raise RuntimeError("Rotation-aware RANSAC failed to produce any valid Sim3 candidate.")

    refine_indices = np.flatnonzero(best["inliers"]).astype(np.int64)
    if refine_indices.shape[0] < 2:
        raise RuntimeError(
            f"Rotation-aware RANSAC produced fewer than 2 inliers: inlier_count={best['inlier_count']} "
            f"subset={best['subset_idx'].tolist()}"
        )
    ref_scale, ref_rot, ref_trans = estimate_pose_aware_sim3(src_pose[refine_indices], dst_pose[refine_indices])
    final_center_err_m, final_rot_err_deg = evaluate_pose_aware_sim3(src_pose, dst_pose, ref_scale, ref_rot, ref_trans)
    final_inliers = (final_center_err_m <= RANSAC_TRANS_THRESH_M) & (final_rot_err_deg <= RANSAC_ROT_THRESH_DEG)
    diagnostics = {
        "fit_indices": fit_indices.astype(np.int32).tolist(),
        "ransac_subset_size": int(RANSAC_SUBSET_SIZE),
        "ransac_trans_thresh_m": float(RANSAC_TRANS_THRESH_M),
        "ransac_rot_thresh_deg": float(RANSAC_ROT_THRESH_DEG),
        "best_subset_indices": best["subset_idx"].astype(np.int32).tolist(),
        "best_subset_inlier_count": int(best["inlier_count"]),
        "refine_inlier_indices": np.flatnonzero(final_inliers).astype(np.int32).tolist(),
        "center_err_m": final_center_err_m.astype(np.float32).tolist(),
        "rot_err_deg": final_rot_err_deg.astype(np.float32).tolist(),
    }
    return float(ref_scale), ref_rot.astype(np.float32), ref_trans.astype(np.float32), diagnostics
