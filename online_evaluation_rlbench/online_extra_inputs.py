from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online_evaluation_rlbench.pi3x_worker import patch_pool_xyz_tokens, resize_xyz_map
from utils.multiview_reprojection import build_cross_view_token_correspondence
DEFAULT_VGGT_IMAGE_SIZE = 518
DEFAULT_VGGT_PREPROCESS_MODE = "crop"


def _camera_space_depth_from_world_pcds(
    world_pcds: torch.Tensor,
    camera_extrinsics: torch.Tensor,
) -> torch.Tensor:
    if world_pcds.ndim != 5 or world_pcds.shape[2] != 3:
        raise ValueError(
            f"Expected world_pcds shape=(B,V,3,H,W), got {tuple(world_pcds.shape)}"
        )
    if camera_extrinsics.ndim != 4 or tuple(camera_extrinsics.shape[-2:]) != (4, 4):
        raise ValueError(
            f"Expected camera_extrinsics shape=(B,V,4,4), got {tuple(camera_extrinsics.shape)}"
        )
    world_hwc = world_pcds.permute(0, 1, 3, 4, 2).float()
    ones = torch.ones((*world_hwc.shape[:-1], 1), device=world_hwc.device, dtype=world_hwc.dtype)
    world_h = torch.cat([world_hwc, ones], dim=-1)
    w2c = torch.linalg.inv(camera_extrinsics.float())
    cam_h = torch.einsum('bvhwc,bvdc->bvhwd', world_h, w2c)
    return cam_h[..., 2].contiguous()


def _resolve_online_semantic_corr_source(
    scene_geometry_source: str,
    online_semantic_corr_source: str | None,
) -> str:
    if online_semantic_corr_source is not None:
        resolved = str(online_semantic_corr_source).strip().lower()
        if resolved:
            return resolved
    env_value = os.environ.get("ONLINE_SEMANTIC_CORR_SOURCE")
    if env_value is not None:
        resolved = str(env_value).strip().lower()
        if resolved:
            return resolved
    if str(scene_geometry_source).strip().lower() == "raw_pcd":
        return "raw_pcd"
    return "pi3x"


class OnlineExtraInputBuilder:
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        pi3x_client=None,
        pi3_root: str | None = None,
        pi3_ckpt: str | None = None,
        scene_geometry_source: str | None = None,
        online_semantic_corr_source: str | None = None,
        vggt_root: str | None = None,
        vggt_ckpt: str | None = None,
        vggt_preprocess_mode: str | None = None,
        vggt_image_size: int | None = None,
        vggt_align_mode: str | None = None,
        use_pi3_pose_prior: bool = True,
        pi3_rlbench_to_opencv: bool = True,
    ):
        self.policy = policy
        self.encoder = getattr(policy, "encoder", None)
        self.device = next(policy.parameters()).device
        self.interleaved_point_fusion = getattr(self.encoder, "interleaved_point_fusion", None)
        self.disable_semantic_fusion = bool(
            getattr(self.encoder, "disable_semantic_fusion", False)
        )
        self.disable_geometry_fusion = bool(
            getattr(self.encoder, "disable_geometry_fusion", False)
        )
        self.pointwise_pi3x_fusion = getattr(self.encoder, "pointwise_pi3x_fusion", None)
        self.semantic_grid_hw = tuple(
            getattr(self.interleaved_point_fusion, "semantic_grid_hw", (32, 32))
        )
        self.geometry_grid_hw = tuple(
            getattr(self.interleaved_point_fusion, "geometry_grid_hw", (18, 18))
        )
        self.cross_view_radius = int(
            getattr(self.interleaved_point_fusion, "cross_view_radius", 1)
        )
        self.cross_view_max_reproj_error = float(
            getattr(self.interleaved_point_fusion, "cross_view_max_reproj_error", 1.5)
        )
        self.cross_view_max_depth_error = float(
            getattr(self.interleaved_point_fusion, "cross_view_max_depth_error", 0.05)
        )
        self.pi3x_geometry_mode = str(
            getattr(self.encoder, "pi3x_geometry_mode", "interleaved")
        )
        self.use_pi3_pose_prior = bool(use_pi3_pose_prior)
        self.pi3_rlbench_to_opencv = bool(pi3_rlbench_to_opencv)
        self.backbone_name = str(getattr(self.encoder, "_backbone_name", ""))
        self.siglip_image_size = int(
            getattr(getattr(self.encoder, "backbone", None), "image_size", 512)
        )
        self.scene_geometry_source = str(
            scene_geometry_source or os.environ.get("ONLINE_SCENE_GEOMETRY_SOURCE", "raw_pcd")
        ).strip().lower()
        self.semantic_corr_source = _resolve_online_semantic_corr_source(
            self.scene_geometry_source,
            online_semantic_corr_source,
        )
        self.need_semantic_corr = (
            self.interleaved_point_fusion is not None and not self.disable_semantic_fusion
        )
        self.need_scene_geometry = self.scene_geometry_source in {"pi3x", "vggt"}
        self.need_geometry_tokens = (
            not self.disable_geometry_fusion
            and (
                self.interleaved_point_fusion is not None
                or self.pointwise_pi3x_fusion is not None
            )
        )
        self.need_pi3x = bool(
            self.need_geometry_tokens
            or self.semantic_corr_source == "pi3x"
            or self.scene_geometry_source == "pi3x"
        )
        self.clip_semantic_xyz_mode = str(
            os.environ.get("ONLINE_CLIP_SEMANTIC_XYZ_MODE", "match_train_cache")
        ).strip().lower()
        self.vggt_preprocess_mode = str(
            vggt_preprocess_mode or os.environ.get("ONLINE_VGGT_PREPROCESS_MODE", DEFAULT_VGGT_PREPROCESS_MODE)
        ).strip().lower()
        self.vggt_image_size = int(
            vggt_image_size or int(os.environ.get("ONLINE_VGGT_IMAGE_SIZE", str(DEFAULT_VGGT_IMAGE_SIZE)))
        )
        self.vggt_align_mode = str(
            vggt_align_mode or os.environ.get("ONLINE_VGGT_ALIGN_MODE", "pose_ransac")
        ).strip().lower()
        self.disable_geometry_tokens = os.environ.get("DISABLE_ONLINE_PI3X_TOKENS", "0") == "1"
        self.force_tokenized_scene_inputs = (
            os.environ.get("FORCE_ONLINE_TOKENIZED_INPUTS", "0") == "1"
        )
        self._mode_announced = False
        self.pi3x = None
        self.pi3x_client = pi3x_client
        self.vggt = None
        if self.need_pi3x and self.pi3x_client is None:
            raise NotImplementedError(
                "Pi3X online geometry is required for this checkpoint. "
                "Use online_evaluation_rlbench/evaluate_policy_inprocess_pi3x.py with --pi3_root and --pi3_ckpt."
            )
        if self.scene_geometry_source == "vggt":
            raise NotImplementedError("VGGT online geometry is not used by the released MV-Actor checkpoint.")

    def shutdown(self) -> None:
        if self.pi3x_client is not None:
            self.pi3x_client.close()

    def _build_online_siglip2_tokens(
        self,
        rgbs: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.encoder is None or self.backbone_name != "siglip2":
            return None
        normalize = getattr(self.encoder, "normalize", None)
        backbone = getattr(self.encoder, "backbone", None)
        if normalize is None or backbone is None:
            return None

        batch, num_views, _, _, _ = rgbs.shape
        rgb_flat = rgbs.flatten(0, 1)
        rgb_norm = normalize(rgb_flat)
        feat = backbone(rgb_norm)["patch"]
        feat = feat.reshape(batch, num_views, feat.shape[1], feat.shape[2], feat.shape[3])
        feat = feat.permute(0, 1, 3, 4, 2).reshape(
            batch, num_views, feat.shape[3] * feat.shape[4], feat.shape[2]
        ).contiguous()
        return feat.float()

    def _build_semantic_corr_from_tokens(
        self,
        semantic_xyz: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_views, num_tokens, _ = semantic_xyz.shape
        sem_h, sem_w = int(self.semantic_grid_hw[0]), int(self.semantic_grid_hw[1])
        if num_tokens != sem_h * sem_w:
            raise ValueError(
                f"Expected semantic xyz on grid {(sem_h, sem_w)}, got {num_tokens} tokens"
            )
        corr_idx, corr_mask = build_cross_view_token_correspondence(
            semantic_xyz.float(),
            semantic_xyz.float(),
            camera_intrinsics.float(),
            camera_extrinsics.float(),
            image_hw=image_hw,
            grid_hw=(sem_h, sem_w),
            search_radius=self.cross_view_radius,
            max_reproj_error=self.cross_view_max_reproj_error,
            max_depth_error=self.cross_view_max_depth_error,
        )
        return corr_idx, corr_mask

    def _build_semantic_corr_from_raw_pcd(
        self,
        pcds: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_views, _, _, _ = pcds.shape
        sem_h, sem_w = int(self.semantic_grid_hw[0]), int(self.semantic_grid_hw[1])
        semantic_xyz = resize_xyz_map(
            pcds.float(),
            sem_h,
            sem_w,
        )
        semantic_xyz = semantic_xyz.permute(0, 1, 3, 4, 2).reshape(
            batch, num_views, sem_h * sem_w, 3
        ).contiguous()
        corr_idx, corr_mask = self._build_semantic_corr_from_tokens(
            semantic_xyz,
            camera_intrinsics,
            camera_extrinsics,
            image_hw=image_hw,
        )
        return semantic_xyz, corr_idx, corr_mask

    def _build_semantic_xyz_tokens_from_pi3x_dense(
        self,
        xyz_dense: torch.Tensor,
    ) -> torch.Tensor:
        sem_h, sem_w = int(self.semantic_grid_hw[0]), int(self.semantic_grid_hw[1])
        if self.backbone_name == "clip" and self.clip_semantic_xyz_mode == "match_train_cache":
            semantic_xyz_dense = resize_xyz_map(
                xyz_dense,
                sem_h,
                sem_w,
            )
            return semantic_xyz_dense.permute(0, 1, 3, 4, 2).reshape(
                xyz_dense.shape[0],
                xyz_dense.shape[1],
                sem_h * sem_w,
                3,
            ).contiguous()

        semantic_xyz_dense = resize_xyz_map(
            xyz_dense,
            self.siglip_image_size,
            self.siglip_image_size,
        )
        return patch_pool_xyz_tokens(
            semantic_xyz_dense,
            sem_h,
            sem_w,
        )

    def _build_semantic_xyz_tokens_from_raw_pcd(
        self,
        pcds: torch.Tensor,
    ) -> torch.Tensor:
        sem_h, sem_w = int(self.semantic_grid_hw[0]), int(self.semantic_grid_hw[1])
        semantic_xyz_dense = resize_xyz_map(
            pcds.float(),
            sem_h,
            sem_w,
        )
        return semantic_xyz_dense.permute(0, 1, 3, 4, 2).reshape(
            pcds.shape[0],
            pcds.shape[1],
            sem_h * sem_w,
            3,
        ).contiguous()

    def _build_geometry_xyz_tokens_from_raw_pcd(
        self,
        pcds: torch.Tensor,
        *,
        target_hw: tuple[int, int],
        token_grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        grid_h, grid_w = int(token_grid_hw[0]), int(token_grid_hw[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid Pi3X target_hw={target_hw}")
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"Invalid Pi3X token_grid_hw={token_grid_hw}")
        if self.interleaved_point_fusion is not None:
            expected_grid_hw = tuple(int(v) for v in self.geometry_grid_hw)
            if expected_grid_hw != (grid_h, grid_w):
                raise ValueError(
                    f"Pi3X token_grid_hw {(grid_h, grid_w)} does not match encoder geometry_grid_hw {expected_grid_hw}"
                )
        geometry_xyz_dense = resize_xyz_map(
            pcds.float(),
            target_h,
            target_w,
        )
        return patch_pool_xyz_tokens(
            geometry_xyz_dense,
            grid_h,
            grid_w,
        )

    def _align_dense_points_with_sim3(
        self,
        world_points: torch.Tensor,
        pred_c2w: np.ndarray,
        gt_c2w: np.ndarray,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        pred_c2w = np.asarray(pred_c2w, dtype=np.float32)
        gt_c2w = np.asarray(gt_c2w, dtype=np.float32)
        valid = np.isfinite(pred_c2w).all(axis=(1, 2)) & np.isfinite(gt_c2w).all(axis=(1, 2))
        pred_valid = pred_c2w[valid]
        gt_valid = gt_c2w[valid]

        if self.vggt_align_mode != "pose_ransac":
            raise ValueError(
                f"Unsupported VGGT align mode: {self.vggt_align_mode}. Only pose_ransac is allowed."
            )
        if pred_valid.shape[0] < 3:
            raise ValueError(
                f"Need at least 3 valid pose pairs for pose-aware Sim3 RANSAC, got {pred_valid.shape[0]}"
            )
        scale, rotation, translation, ransac_diag = estimate_pose_aware_sim3_ransac(
            pred_valid,
            gt_valid,
        )
        diag = {
            "align_mode": "pose_ransac",
            "valid_pose_pairs": int(pred_valid.shape[0]),
            **ransac_diag,
        }

        rotation_t = torch.from_numpy(rotation).to(world_points.device, dtype=world_points.dtype)
        translation_t = torch.from_numpy(translation).to(world_points.device, dtype=world_points.dtype)
        aligned = scale * torch.einsum("ij,vhwj->vhwi", rotation_t, world_points)
        aligned = aligned + translation_t.view(1, 1, 1, 3)
        return aligned, diag

    def _build_vggt_scene_inputs(
        self,
        rgbs: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.vggt is None:
            return {}

        outputs = extract_vggt_tokens_and_geometry(
            self.vggt,
            rgbs.float(),
            preprocess_mode=self.vggt_preprocess_mode,
            image_size=self.vggt_image_size,
        )
        world_points = outputs["world_points"].float()
        pred_extrinsics = outputs["pred_extrinsics"].detach().cpu().numpy().astype(np.float32)
        gt_extrinsics = camera_extrinsics.detach().cpu().numpy().astype(np.float32)
        if self.pi3_rlbench_to_opencv:
            intrinsics_np = camera_intrinsics.detach().cpu().numpy().astype(np.float32)
            _, gt_extrinsics = _rlbench_to_opencv_np(
                intrinsics_np,
                gt_extrinsics,
            )

        dense_world = []
        align_diag = []
        for batch_idx in range(world_points.shape[0]):
            pred_c2w = vggt_w2c34_to_c2w44(pred_extrinsics[batch_idx])
            gt_c2w = gt_extrinsics[batch_idx]
            aligned_points, diag = self._align_dense_points_with_sim3(
                world_points[batch_idx],
                pred_c2w,
                gt_c2w,
            )
            dense_world.append(aligned_points)
            align_diag.append(diag)

        xyz_dense = torch.stack(dense_world, dim=0).permute(0, 1, 4, 2, 3).contiguous()
        xyz_dense = torch.nan_to_num(xyz_dense, nan=0.0, posinf=0.0, neginf=0.0)
        out_h, out_w = tuple(int(v) for v in rgbs.shape[-2:])
        scene_pcds = resize_xyz_map(
            xyz_dense,
            out_h,
            out_w,
        )
        outputs = {
            "online_model_pcds": scene_pcds.float(),
            "online_vggt_align_diag": align_diag,
        }
        return outputs

    def _build_pi3x_inputs(
        self,
        rgbs: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        camera_gt_depths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.pi3x_client is None:
            return {}
        debug_rollout = os.environ.get("ONLINE_DEBUG_ROLLOUT", "0") == "1"
        if debug_rollout:
            print("[pi3x] build_inputs_begin", flush=True)
        out_h, out_w = tuple(int(v) for v in rgbs.shape[-2:])
        sem_h, sem_w = int(self.semantic_grid_hw[0]), int(self.semantic_grid_hw[1])
        rgbs_np = rgbs.detach().cpu().numpy().astype(np.float32, copy=False)
        camera_intrinsics_np = camera_intrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        camera_extrinsics_np = camera_extrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        camera_gt_depths_np = camera_gt_depths.detach().cpu().numpy().astype(np.float32, copy=False)
        dump_request_path = os.environ.get("ONLINE_PI3X_DUMP_REQUEST", "").strip()
        if dump_request_path:
            dump_path = Path(dump_request_path)
            if not dump_path.exists():
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dump_path,
                    rgbs_np=rgbs_np,
                    camera_intrinsics_np=camera_intrinsics_np,
                    camera_extrinsics_np=camera_extrinsics_np,
                    camera_gt_depths_np=camera_gt_depths_np,
                    out_h=np.asarray([out_h], dtype=np.int32),
                    out_w=np.asarray([out_w], dtype=np.int32),
                    sem_h=np.asarray([sem_h], dtype=np.int32),
                    sem_w=np.asarray([sem_w], dtype=np.int32),
                )
                print(f"[pi3x] dumped_request_npz={dump_path}", flush=True)
        worker_outputs = self.pi3x_client.request_online_inputs(
            rgbs_np=rgbs_np,
            camera_intrinsics_np=camera_intrinsics_np,
            camera_extrinsics_np=camera_extrinsics_np,
            camera_gt_depths_np=camera_gt_depths_np,
            out_h=out_h,
            out_w=out_w,
            sem_h=sem_h,
            sem_w=sem_w,
            backbone_name=self.backbone_name,
            clip_semantic_xyz_mode=self.clip_semantic_xyz_mode,
            siglip_image_size=self.siglip_image_size,
            disable_geometry_tokens=self.disable_geometry_tokens,
        )
        outputs = {
            "online_model_pcds": torch.from_numpy(worker_outputs["online_model_pcds"]).to(self.device).float(),
            "token_xyz_pi3_sim3": torch.from_numpy(worker_outputs["token_xyz_pi3_sim3"]).to(self.device).float(),
            "token_xyz_siglip2_sim3": torch.from_numpy(worker_outputs["token_xyz_siglip2_sim3"]).to(self.device).float(),
            "pi3x_target_hw": tuple(int(v) for v in worker_outputs["pi3x_target_hw"].tolist()),
            "pi3x_token_grid_hw": tuple(int(v) for v in worker_outputs["pi3x_token_grid_hw"].tolist()),
        }
        if not self.disable_geometry_tokens and "pi3x_tokens" in worker_outputs:
            outputs["pi3x_tokens"] = torch.from_numpy(worker_outputs["pi3x_tokens"]).to(self.device).float()
        if debug_rollout:
            print("[pi3x] build_inputs_done", flush=True)
        return outputs

    @torch.no_grad()
    def build(
        self,
        rgbs: torch.Tensor,
        pcds: torch.Tensor,
        camera_intrinsics: torch.Tensor | None,
        camera_extrinsics: torch.Tensor | None,
    ) -> dict | None:
        need_tokenized_scene_inputs = (
            self.force_tokenized_scene_inputs and self.backbone_name == "siglip2"
        )
        if (
            not self.need_semantic_corr
            and not self.need_pi3x
            and not self.need_scene_geometry
            and not need_tokenized_scene_inputs
        ):
            return None
        if camera_intrinsics is None or camera_extrinsics is None:
            raise ValueError("Online extra_inputs require camera intrinsics and extrinsics")
        if not self._mode_announced:
            print(
                "[online_extra_inputs] "
                f"backbone={self.backbone_name} "
                f"scene_geometry_source={self.scene_geometry_source} "
                f"semantic_corr_source={self.semantic_corr_source} "
                f"clip_semantic_xyz_mode={self.clip_semantic_xyz_mode} "
                f"vggt_align_mode={self.vggt_align_mode}",
                flush=True,
            )
            self._mode_announced = True

        camera_intrinsics = camera_intrinsics.to(self.device).float()
        camera_extrinsics = camera_extrinsics.to(self.device).float()
        camera_gt_depths = _camera_space_depth_from_world_pcds(
            pcds.to(self.device).float(),
            camera_extrinsics,
        )
        image_hw = tuple(int(v) for v in rgbs.shape[-2:])

        extra_inputs = {
            "camera_intrinsics": camera_intrinsics,
            "camera_extrinsics": camera_extrinsics,
            "camera_image_hw": image_hw,
        }

        scene_pcds = pcds.to(self.device).float()
        if self.scene_geometry_source == "vggt":
            vggt_outputs = self._build_vggt_scene_inputs(
                rgbs,
                camera_intrinsics,
                camera_extrinsics,
            )
            scene_pcds = vggt_outputs.get("online_model_pcds", scene_pcds)
            extra_inputs.update(vggt_outputs)

        pi3_outputs = {}
        if self.need_pi3x:
            pi3_outputs = self._build_pi3x_inputs(
                rgbs,
                camera_intrinsics,
                camera_extrinsics,
                camera_gt_depths,
            )
            if self.scene_geometry_source == "pi3x":
                scene_pcds = pi3_outputs.get("online_model_pcds", scene_pcds)
                extra_inputs["online_model_pcds"] = scene_pcds
            if "token_xyz_pi3_sim3" in pi3_outputs:
                geometry_xyz = pi3_outputs["token_xyz_pi3_sim3"]
                if self.scene_geometry_source == "raw_pcd":
                    target_hw = pi3_outputs.get("pi3x_target_hw")
                    token_grid_hw = pi3_outputs.get("pi3x_token_grid_hw")
                    if target_hw is None or token_grid_hw is None:
                        raise ValueError(
                            "Pi3X worker outputs must include pi3x_target_hw and pi3x_token_grid_hw for raw_pcd geometry tokens"
                        )
                    geometry_xyz = self._build_geometry_xyz_tokens_from_raw_pcd(
                        scene_pcds,
                        target_hw=target_hw,
                        token_grid_hw=token_grid_hw,
                    )
                extra_inputs["token_xyz_pi3_sim3"] = geometry_xyz
            if "token_xyz_siglip2_sim3" in pi3_outputs:
                semantic_xyz = pi3_outputs["token_xyz_siglip2_sim3"]
                if self.scene_geometry_source == "raw_pcd":
                    semantic_xyz = self._build_semantic_xyz_tokens_from_raw_pcd(scene_pcds)
                extra_inputs["token_xyz_siglip2_sim3"] = semantic_xyz
            if "pi3x_target_hw" in pi3_outputs:
                extra_inputs["pi3x_target_hw"] = tuple(int(v) for v in pi3_outputs["pi3x_target_hw"])
            if "pi3x_token_grid_hw" in pi3_outputs:
                extra_inputs["pi3x_token_grid_hw"] = tuple(int(v) for v in pi3_outputs["pi3x_token_grid_hw"])
            if "pi3x_tokens" in pi3_outputs:
                extra_inputs["pi3x_tokens"] = pi3_outputs["pi3x_tokens"]

        if self.need_semantic_corr:
            semantic_xyz = None
            corr_idx = None
            corr_mask = None
            if self.semantic_corr_source == "raw_pcd":
                semantic_xyz, corr_idx, corr_mask = self._build_semantic_corr_from_raw_pcd(
                    scene_pcds,
                    camera_intrinsics,
                    camera_extrinsics,
                    image_hw=image_hw,
                )
            else:
                semantic_xyz = extra_inputs.get("token_xyz_siglip2_sim3")
                if semantic_xyz is not None:
                    corr_idx, corr_mask = self._build_semantic_corr_from_tokens(
                        semantic_xyz.float(),
                        camera_intrinsics,
                        camera_extrinsics,
                        image_hw=image_hw,
                    )
                else:
                    semantic_xyz, corr_idx, corr_mask = self._build_semantic_corr_from_raw_pcd(
                        scene_pcds,
                        camera_intrinsics,
                        camera_extrinsics,
                        image_hw=image_hw,
                    )
            extra_inputs["semantic_correspondence_xyz"] = semantic_xyz.float()
            extra_inputs["interleaved_semantic_corr_idx"] = corr_idx.to(torch.long)
            extra_inputs["interleaved_semantic_corr_mask"] = corr_mask.to(torch.bool)
        elif need_tokenized_scene_inputs:
            semantic_xyz = extra_inputs.get("token_xyz_siglip2_sim3")
            if semantic_xyz is None:
                semantic_xyz = self._build_semantic_xyz_tokens_from_raw_pcd(scene_pcds)
            extra_inputs["semantic_correspondence_xyz"] = semantic_xyz.float()

        siglip2_tokens = self._build_online_siglip2_tokens(rgbs)
        if siglip2_tokens is not None:
            extra_inputs["online_siglip2_tokens"] = siglip2_tokens

        return extra_inputs
