import math
import os

import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.layers import AttentionModule
from ...utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from ..fusion import InterleavedPointFusion, PointwiseCrossAttention
from ..vision.fpn import EfficientFeaturePyramidNetwork
from .base_encoder import Encoder as BaseEncoder


class SceneBranchRouter(nn.Module):

    @staticmethod
    def _normalize_init_bias(init_bias):
        if isinstance(init_bias, str):
            values = [float(part.strip()) for part in init_bias.split(',') if part.strip()]
        elif isinstance(init_bias, (int, float)):
            values = [float(init_bias)]
        else:
            values = [float(value) for value in init_bias]
        if len(values) == 1:
            values = values * 4
        if len(values) != 4:
            raise ValueError(f"router init bias must have 1 or 4 values, got {values}")
        return tuple(values)

    def __init__(self, embedding_dim, proprio_input_dim, init_bias=(0.0, 0.0, 0.0, 0.0)):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.proprio_input_dim = int(proprio_input_dim)
        self.proprio_proj = nn.Linear(self.proprio_input_dim, self.embedding_dim)
        self.router = nn.Sequential(
            nn.Linear(5 * self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, 4),
        )
        self.last_gates = None
        init_bias = self._normalize_init_bias(init_bias)
        with torch.no_grad():
            nn.init.normal_(self.router[-1].weight, mean=0.0, std=1e-3)
            self.router[-1].bias.copy_(torch.tensor(init_bias, dtype=torch.float32))

    @staticmethod
    def _pool_tokens(feats):
        return feats.mean(dim=1)

    def forward(self, semantic_base, instr_feats, proprio, semantic_delta, geometry_delta):
        z_scene = self._pool_tokens(semantic_base)
        z_text = self._pool_tokens(instr_feats)
        z_prop = self.proprio_proj(proprio.flatten(1))
        z_sem = self._pool_tokens(semantic_delta)
        z_geo = self._pool_tokens(geometry_delta)
        route_in = torch.cat((z_scene, z_text, z_prop, z_sem, z_geo), dim=-1)
        gates = torch.sigmoid(self.router(route_in))
        self.last_gates = gates.detach()
        return {
            'sem_pos': gates[:, 0:1].unsqueeze(1),
            'geo_pos': gates[:, 1:2].unsqueeze(1),
            'sem_rot': gates[:, 2:3].unsqueeze(1),
            'geo_rot': gates[:, 3:4].unsqueeze(1),
        }


class Encoder(BaseEncoder):

    def __init__(
        self,
        backbone="clip",
        embedding_dim=60,
        nhist=1,
        num_attn_heads=9,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        finetune_backbone=False,
        finetune_text_encoder=False,
        enable_interleaved_point_fusion=False,
        interleaved_before_language=False,
        interleaved_text_between_self_and_geo=False,
        interleaved_cross_overlap_only=False,
        interleaved_point_fusion_layers=2,
        interleaved_semantic_mode="cross_view_reproject",
        interleaved_cross_knn=16,
        interleaved_cross_residual_scale=0.1,
        interleaved_knn_chunk_size=256,
        interleaved_cross_view_radius=1,
        interleaved_cross_view_max_reproj_error=1.5,
        interleaved_cross_view_max_depth_error=0.05,
        pi3x_geometry_mode="interleaved",
        geo_upsample_mode="nearest_copy",
        pointwise_parallel_branch_fusion=False,
        pointwise_parallel_branch_text_last=False,
        interleaved_semantic_use_pos=True,
        disable_semantic_fusion=False,
        disable_geometry_fusion=False,
        router_init_bias="0,0,0,0",
    ):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder,
        )
        self.embedding_dim = int(embedding_dim)

        if self._backbone_name == "clip":
            self.output_level = "res3"
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim,
                output_level="res3",
            )
            self.rgb2d_proj = nn.Linear(1024, embedding_dim)
        if self._backbone_name == "siglip2":
            self.siglip2_patch_proj = nn.Conv2d(
                self.backbone.output_dim,
                embedding_dim,
                1,
            )

        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            n_heads=num_attn_heads,
            rotary_pe=True,
            use_adaln=False,
            pre_norm=False,
        )

        self.camera_ids = nn.Embedding(2, embedding_dim)
        self.pos_embed_2d = SinusoidalPosEmb(embedding_dim)

        self.enable_interleaved_point_fusion = enable_interleaved_point_fusion
        self.interleaved_before_language = interleaved_before_language
        self.interleaved_text_between_self_and_geo = interleaved_text_between_self_and_geo
        self.pi3x_geometry_mode = pi3x_geometry_mode
        self.geo_upsample_mode = geo_upsample_mode
        self.pointwise_parallel_branch_fusion = bool(pointwise_parallel_branch_fusion)
        self.pointwise_parallel_branch_text_last = bool(pointwise_parallel_branch_text_last)
        self.disable_semantic_fusion = disable_semantic_fusion
        self.disable_geometry_fusion = disable_geometry_fusion
        self.interleaved_point_fusion = None
        self.pointwise_pi3x_fusion = None
        if enable_interleaved_point_fusion:
            self.interleaved_point_fusion = InterleavedPointFusion(
                embedding_dim=embedding_dim,
                num_heads=num_attn_heads,
                num_layers=interleaved_point_fusion_layers,
                cross_overlap_only=interleaved_cross_overlap_only,
                semantic_mode=interleaved_semantic_mode,
                cross_knn=interleaved_cross_knn,
                cross_residual_scale=interleaved_cross_residual_scale,
                knn_chunk_size=interleaved_knn_chunk_size,
                cross_view_radius=interleaved_cross_view_radius,
                cross_view_max_reproj_error=interleaved_cross_view_max_reproj_error,
                cross_view_max_depth_error=interleaved_cross_view_max_depth_error,
                semantic_use_pos=interleaved_semantic_use_pos,
            )
            if pi3x_geometry_mode == "pointwise_sig32":
                self.pointwise_pi3x_fusion = PointwiseCrossAttention(
                    embedding_dim,
                    num_attn_heads,
                    residual_scale=interleaved_cross_residual_scale,
                    use_gate=False,
                )
                self.scene_branch_router = SceneBranchRouter(
                    embedding_dim=embedding_dim,
                    proprio_input_dim=nhist * 8,
                    init_bias=router_init_bias,
                )
            else:
                self.scene_branch_router = None
        else:
            self.scene_branch_router = None

    def encode_proprio(self, proprio, context_feats, context_pos):
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )
        proprio_pos = self.relative_pe_layer(proprio[..., :3])
        context_pos = self.relative_pe_layer(context_pos)
        proprio_feats = self.gripper_context_head(
            proprio_feats,
            context_feats,
            seq1_pos=proprio_pos,
            seq2_pos=context_pos,
        )[-1]
        return proprio_feats

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)

        num_cameras = rgb3d.shape[1]
        if rgb3d.dim() == 4:
            batch_size, ncam, num_tokens, channels = rgb3d.shape
            if channels != self.embedding_dim:
                raise ValueError(
                    f"Cached clip tokens must have dim={self.embedding_dim}, got {channels}"
                )
            feat_h, feat_w = self._infer_square_hw(num_tokens)
            rgb3d_feats = rgb3d.reshape(batch_size, ncam * num_tokens, channels)
            if not self._should_delay_language_conditioning():
                rgb3d_feats = self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)
        else:
            rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
            rgb3d = self.normalize(rgb3d)
            rgb3d_feats = self.backbone(rgb3d)
            rgb3d_feats = self.feature_pyramid(rgb3d_feats)[self.output_level]
            feat_h, feat_w = rgb3d_feats.shape[-2:]
            rgb3d_feats = einops.rearrange(
                rgb3d_feats,
                "(bt ncam) c h w -> bt (ncam h w) c",
                ncam=num_cameras,
            )
            if not self._should_delay_language_conditioning():
                rgb3d_feats = self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)

        if pcd is None:
            raise ValueError("clip encoder requires pcd input")
        num_cameras = pcd.shape[1]
        if pcd.dim() == 5:
            pcd = F.interpolate(
                einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
                (feat_h, feat_w),
                mode="bilinear",
            )
            pcd = einops.rearrange(
                pcd,
                "(bt ncam) c h w -> bt (ncam h w) c",
                ncam=num_cameras,
            )
        elif pcd.dim() == 4:
            if pcd.shape[2] != feat_h * feat_w:
                raise ValueError(
                    f"Cached clip xyz token count {pcd.shape[2]} does not match feature grid {feat_h}x{feat_w}"
                )
            pcd = pcd.flatten(1, 2)
        else:
            raise ValueError(f"Unsupported clip pcd shape={tuple(pcd.shape)}")

        rgb2d_feats = None
        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def _use_custom_pi3x_geometry_mode(self):
        return self.interleaved_point_fusion is not None and self.pi3x_geometry_mode == "pointwise_sig32"

    def _should_delay_language_conditioning(self):
        return self.interleaved_point_fusion is not None and (
            self.interleaved_before_language
            or self.interleaved_text_between_self_and_geo
            or self._use_custom_pi3x_geometry_mode()
        )

    def _apply_scene_instruction_attention(self, rgb3d_feats, instr_feats):
        return self.vl_attention(seq1=rgb3d_feats, seq2=instr_feats)[-1]

    def encode_siglip2(self, rgb3d, rgb2d, pcd, text):
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)

        num_cameras = rgb3d.shape[1]
        if rgb3d.dim() == 4:
            batch_size, ncam, num_tokens, in_channels = rgb3d.shape
            side = int(math.isqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(f"Expected square number of tokens, got N={num_tokens}.")
            rgb3d_feats = rgb3d.reshape(batch_size * ncam, side, side, in_channels).permute(0, 3, 1, 2)
            rgb3d_feats = self.siglip2_patch_proj(rgb3d_feats)
            rgb3d_feats = einops.rearrange(
                rgb3d_feats,
                "(bt ncam) c h w -> bt (ncam h w) c",
                ncam=num_cameras,
            )
            if not self._should_delay_language_conditioning():
                rgb3d_feats = self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)

            if pcd is None or pcd.dim() != 4:
                raise ValueError("Cached tokens require pcd tokens of shape (B, ncam, N, 3).")
            pcd = pcd.flatten(1, 2)
        else:
            rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
            rgb3d = self.normalize(rgb3d)
            rgb3d_feats = self.backbone(rgb3d)["patch"]
            rgb3d_feats = self.siglip2_patch_proj(rgb3d_feats)
            feat_h, feat_w = rgb3d_feats.shape[-2:]
            rgb3d_feats = einops.rearrange(
                rgb3d_feats,
                "(bt ncam) c h w -> bt (ncam h w) c",
                ncam=num_cameras,
            )
            if not self._should_delay_language_conditioning():
                rgb3d_feats = self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)

            num_cameras = pcd.shape[1]
            pcd = F.interpolate(
                einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
                (feat_h, feat_w),
                mode="bilinear",
            )
            pcd = einops.rearrange(
                pcd,
                "(bt ncam) c h w -> bt (ncam h w) c",
                ncam=num_cameras,
            )

        rgb2d_feats = None
        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def _infer_square_hw(self, num_tokens):
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Expected square token grid, got N={num_tokens}")
        return side, side

    def _resize_flat_tokens(self, features, num_cameras, out_h, out_w, mode="bilinear"):
        batch, total_tokens, channels = features.shape
        if total_tokens % num_cameras != 0:
            raise ValueError(f"Cannot split {total_tokens} tokens across {num_cameras} cameras")
        per_cam = total_tokens // num_cameras
        in_h, in_w = self._infer_square_hw(per_cam)
        if (in_h, in_w) == (out_h, out_w):
            return features
        fmap = features.reshape(batch, num_cameras, in_h, in_w, channels)
        fmap = fmap.permute(0, 1, 4, 2, 3).reshape(batch * num_cameras, channels, in_h, in_w)
        if mode == "nearest_copy":
            fmap = F.interpolate(fmap, size=(out_h, out_w), mode="nearest")
        elif mode == "bilinear":
            fmap = F.interpolate(fmap, size=(out_h, out_w), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unsupported token resize mode: {mode}")
        fmap = fmap.reshape(batch, num_cameras, channels, out_h, out_w)
        fmap = fmap.permute(0, 1, 3, 4, 2).reshape(batch, num_cameras * out_h * out_w, channels)
        return fmap.contiguous()

    def _prepare_geometry_tokens(self, geometry_feats, ref_feats):
        if geometry_feats is None:
            return None, None
        if geometry_feats.dim() != 4:
            raise ValueError(
                f"Custom Pi3X geometry modes expect geometry feats with shape (B, ncam, N, C), got {tuple(geometry_feats.shape)}"
            )
        num_cameras = int(geometry_feats.shape[1])
        geometry_flat = geometry_feats.flatten(1, 2).to(device=ref_feats.device, dtype=ref_feats.dtype)
        geometry_flat = self.interleaved_point_fusion.prepare_geometry(geometry_flat)
        return geometry_flat, num_cameras

    def _build_semantic_view_meta(self, extra_inputs, semantic_xyz):
        if extra_inputs is None:
            return None
        camera_intrinsics = extra_inputs.get("camera_intrinsics")
        camera_extrinsics = extra_inputs.get("camera_extrinsics")
        camera_image_hw = extra_inputs.get("camera_image_hw")
        semantic_correspondence_xyz = extra_inputs.get("semantic_correspondence_xyz")
        semantic_corr_idx = extra_inputs.get("interleaved_semantic_corr_idx")
        semantic_corr_mask = extra_inputs.get("interleaved_semantic_corr_mask")
        if (
            camera_intrinsics is None
            and camera_extrinsics is None
            and camera_image_hw is None
            and semantic_correspondence_xyz is None
            and semantic_corr_idx is None
            and semantic_corr_mask is None
        ):
            return None
        return {
            "camera_intrinsics": camera_intrinsics,
            "camera_extrinsics": camera_extrinsics,
            "camera_image_hw": camera_image_hw,
            "semantic_correspondence_xyz": semantic_correspondence_xyz if semantic_correspondence_xyz is not None else semantic_xyz,
            "semantic_corr_idx": semantic_corr_idx,
            "semantic_corr_mask": semantic_corr_mask,
        }

    def _run_semantic_then_language(self, rgb3d_feats, pcd, instr_feats, extra_inputs=None):
        semantic_view_meta = self._build_semantic_view_meta(extra_inputs, pcd)
        if self.interleaved_point_fusion is not None and not self.disable_semantic_fusion:
            rgb3d_feats = self.interleaved_point_fusion.forward_semantic(
                rgb3d_feats,
                pcd,
                semantic_view_meta=semantic_view_meta,
            )
        return self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)

    def _forward_pointwise_sig32(self, rgb3d_feats, pcd, instr_feats, extra_inputs=None):
        rgb3d_feats = self._run_semantic_then_language(rgb3d_feats, pcd, instr_feats, extra_inputs=extra_inputs)
        if self.disable_geometry_fusion:
            return rgb3d_feats
        if extra_inputs is None:
            return rgb3d_feats
        geometry_feats = extra_inputs.get("pi3x_tokens")
        geometry_flat, num_cameras = self._prepare_geometry_tokens(geometry_feats, rgb3d_feats)
        if geometry_flat is None:
            return rgb3d_feats
        semantic_per_cam = rgb3d_feats.shape[1] // num_cameras
        target_h, target_w = self._infer_square_hw(semantic_per_cam)
        geometry_up = self._resize_flat_tokens(geometry_flat, num_cameras, target_h, target_w, mode=self.geo_upsample_mode)
        rgb3d_feats = self.pointwise_pi3x_fusion(rgb3d_feats, geometry_up)
        return rgb3d_feats

    def _forward_pointwise_sig32_parallel_branches(self, rgb3d_feats, pcd, instr_feats, proprio, extra_inputs=None, return_head_split=False):
        def _maybe_apply_text_last(feats):
            if self.pointwise_parallel_branch_text_last:
                return self._apply_scene_instruction_attention(feats, instr_feats)
            return feats

        if self.pointwise_parallel_branch_text_last:
            semantic_base = rgb3d_feats
        else:
            semantic_base = self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)

        semantic_delta = torch.zeros_like(semantic_base)
        if self.interleaved_point_fusion is not None and not self.disable_semantic_fusion:
            semantic_view_meta = self._build_semantic_view_meta(extra_inputs, pcd)
            semantic_branch = self.interleaved_point_fusion.forward_semantic(
                semantic_base,
                pcd,
                semantic_view_meta=semantic_view_meta,
            )
            semantic_delta = semantic_branch - semantic_base

        if self.disable_geometry_fusion or self.pointwise_pi3x_fusion is None or extra_inputs is None:
            rot_out = semantic_base + semantic_delta
            rot_out = _maybe_apply_text_last(rot_out)
            if return_head_split:
                return rot_out, rot_out, rot_out
            return rot_out

        geometry_feats = extra_inputs.get("pi3x_tokens")
        geometry_flat, num_cameras = self._prepare_geometry_tokens(geometry_feats, semantic_base)
        if geometry_flat is None:
            rot_out = semantic_base + semantic_delta
            rot_out = _maybe_apply_text_last(rot_out)
            if return_head_split:
                return rot_out, rot_out, rot_out
            return rot_out

        semantic_per_cam = semantic_base.shape[1] // num_cameras
        target_h, target_w = self._infer_square_hw(semantic_per_cam)
        geometry_up = self._resize_flat_tokens(
            geometry_flat,
            num_cameras,
            target_h,
            target_w,
            mode=self.geo_upsample_mode,
        )
        geometry_branch = self.pointwise_pi3x_fusion(semantic_base, geometry_up)
        geometry_delta = geometry_branch - semantic_base

        router_gates = self.scene_branch_router(
            semantic_base=semantic_base,
            instr_feats=instr_feats,
            proprio=proprio,
            semantic_delta=semantic_delta,
            geometry_delta=geometry_delta,
        )
        pos_out = semantic_base + router_gates['sem_pos'] * semantic_delta + router_gates['geo_pos'] * geometry_delta
        rot_out = semantic_base + router_gates['sem_rot'] * semantic_delta + router_gates['geo_rot'] * geometry_delta

        pos_out = _maybe_apply_text_last(pos_out)
        rot_out = _maybe_apply_text_last(rot_out)
        if return_head_split:
            shared_out = 0.5 * (pos_out + rot_out)
            return shared_out, pos_out, rot_out
        return rot_out

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio, extra_inputs=None):
        vl_enc_fn = {
            "clip": self.encode_clip,
            "siglip2": self.encode_siglip2,
        }[self._backbone_name]
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )

        rgb3d_feats_pos = None
        rgb3d_feats_rot = None
        if self._use_custom_pi3x_geometry_mode():
            if self.pointwise_parallel_branch_fusion:
                rgb3d_feats, rgb3d_feats_pos, rgb3d_feats_rot = self._forward_pointwise_sig32_parallel_branches(
                    rgb3d_feats,
                    pcd,
                    instr_feats,
                    proprio,
                    extra_inputs=extra_inputs,
                    return_head_split=True,
                )
            else:
                rgb3d_feats = self._forward_pointwise_sig32(
                    rgb3d_feats,
                    pcd,
                    instr_feats,
                    extra_inputs=extra_inputs,
                )
                rgb3d_feats_pos = rgb3d_feats
                rgb3d_feats_rot = rgb3d_feats
        elif self.should_insert_instruction_between_fusion_stages():
            rgb3d_feats, pcd = self.fuse_scene_tokens(
                rgb3d_feats, pcd, extra_inputs=extra_inputs, stage="semantic"
            )
            rgb3d_feats = self.condition_scene_with_instruction(
                rgb3d_feats, instr_feats
            )
            rgb3d_feats, pcd = self.fuse_scene_tokens(
                rgb3d_feats, pcd, extra_inputs=extra_inputs, stage="cross"
            )
            rgb3d_feats_pos = rgb3d_feats
            rgb3d_feats_rot = rgb3d_feats
        else:
            rgb3d_feats, pcd = self.fuse_scene_tokens(
                rgb3d_feats, pcd, extra_inputs=extra_inputs
            )
            rgb3d_feats = self.condition_scene_with_instruction(
                rgb3d_feats, instr_feats
            )
            rgb3d_feats_pos = rgb3d_feats
            rgb3d_feats_rot = rgb3d_feats

        rgb2d_pos = None
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd)

        enable_subsampling = os.environ.get("ENABLE_SCENE_SUBSAMPLING", "0") == "1"
        if enable_subsampling:
            fps_scene_feats, fps_scene_pos = self.run_dps(rgb3d_feats, pcd)
            fps_scene_feats_pos, fps_scene_pos_pos = self.run_dps(rgb3d_feats_pos, pcd)
            fps_scene_feats_rot, fps_scene_pos_rot = self.run_dps(rgb3d_feats_rot, pcd)
        else:
            fps_scene_feats, fps_scene_pos = rgb3d_feats, pcd
            fps_scene_feats_pos, fps_scene_pos_pos = rgb3d_feats_pos, pcd
            fps_scene_feats_rot, fps_scene_pos_rot = rgb3d_feats_rot, pcd

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            rgb3d_feats_pos, fps_scene_feats_pos, fps_scene_pos_pos,
            rgb3d_feats_rot, fps_scene_feats_rot, fps_scene_pos_rot,
        )

    def condition_scene_with_instruction(self, rgb3d_feats, instr_feats):
        if not self._should_delay_language_conditioning():
            return rgb3d_feats
        return self._apply_scene_instruction_attention(rgb3d_feats, instr_feats)

    def should_insert_instruction_between_fusion_stages(self):
        return self.interleaved_point_fusion is not None and self.interleaved_text_between_self_and_geo

    def fuse_scene_tokens(self, rgb3d_feats, pcd, extra_inputs=None, stage="all"):
        if self.interleaved_point_fusion is None or extra_inputs is None:
            return rgb3d_feats, pcd

        geometry_feats = extra_inputs.get("pi3x_tokens")
        geometry_xyz = extra_inputs.get("token_xyz_pi3_sim3")
        cross_knn_idx = extra_inputs.get("interleaved_cross_knn_idx")
        semantic_view_meta = self._build_semantic_view_meta(extra_inputs, pcd)
        if geometry_feats is None or geometry_xyz is None or self.disable_geometry_fusion:
            if stage in {"semantic", "all"} and not self.disable_semantic_fusion:
                rgb3d_feats = self.interleaved_point_fusion.forward_semantic(
                    rgb3d_feats,
                    pcd,
                    semantic_view_meta=semantic_view_meta,
                )
            return rgb3d_feats, pcd

        if geometry_feats.dim() == 4:
            geometry_feats = geometry_feats.flatten(1, 2)
        if geometry_xyz.dim() == 4:
            geometry_xyz = geometry_xyz.flatten(1, 2)
        if geometry_feats.dim() != 3 or geometry_xyz.dim() != 3:
            return rgb3d_feats, pcd

        geometry_feats = geometry_feats.to(device=rgb3d_feats.device, dtype=rgb3d_feats.dtype)
        geometry_xyz = geometry_xyz.to(device=pcd.device, dtype=pcd.dtype)

        if stage == "semantic":
            if self.disable_semantic_fusion:
                return rgb3d_feats, pcd
            rgb3d_feats = self.interleaved_point_fusion.forward_semantic(
                rgb3d_feats,
                pcd,
                semantic_view_meta=semantic_view_meta,
            )
        elif stage == "cross":
            if self.disable_geometry_fusion:
                return rgb3d_feats, pcd
            rgb3d_feats = self.interleaved_point_fusion.forward_cross(
                rgb3d_feats,
                pcd,
                geometry_feats,
                geometry_xyz,
                cross_knn_idx=cross_knn_idx,
            )
        else:
            if self.disable_semantic_fusion and self.disable_geometry_fusion:
                return rgb3d_feats, pcd
            if self.disable_semantic_fusion:
                rgb3d_feats = self.interleaved_point_fusion.forward_cross(
                    rgb3d_feats,
                    pcd,
                    geometry_feats,
                    geometry_xyz,
                    cross_knn_idx=cross_knn_idx,
                )
            elif self.disable_geometry_fusion:
                rgb3d_feats = self.interleaved_point_fusion.forward_semantic(
                    rgb3d_feats,
                    pcd,
                    semantic_view_meta=semantic_view_meta,
                )
            else:
                rgb3d_feats = self.interleaved_point_fusion(
                    rgb3d_feats,
                    pcd,
                    geometry_feats,
                    geometry_xyz,
                    cross_knn_idx=cross_knn_idx,
                    semantic_view_meta=semantic_view_meta,
                )
        return rgb3d_feats, pcd
