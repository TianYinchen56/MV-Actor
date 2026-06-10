import torch

from ..encoder.multimodal.encoder2d import Encoder
from ..utils.position_encodings import SinusoidalPosEmb

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor
from .base_denoise_actor import TransformerHead as BaseTransformerHead


class DenoiseActor(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
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
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size
        )

        # Vision-language encoder, runs only once
        self.encoder = Encoder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers
        )

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False,
        extra_inputs=None
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, nhand, 3+4+X)
            trajectory_mask: (B, trajectory_length, nhand)
            rgb3d: (B, num_3d_cameras, 3, H, W) in [0, 1]
            rgb2d: (B, num_2d_cameras, 3, H, W) in [0, 1]
            pcd: (B, num_3d_cameras, 3, H, W) in world coordinates
            instruction: tokenized text instruction
            proprio: (B, nhist, nhand, 3+4+X)

        Note:
            The input rotation is expressed either as:
                a) quaternion (4D), then the model converts it to 6D internally.
                b) Euler angles (3D).

        Returns:
            - loss: scalar, if run_inference is False
            - trajectory: (B, trajectory_length, nhand, 3+rot+1), at inference
        """
        nhist, nhand = proprio.shape[1], proprio.shape[2]
        proprio = proprio.flatten(1, 2)  # (B, nhist*nhand, 3+4+X)
        proprio = self.convert_rot(proprio)
        proprio = proprio.reshape(len(proprio), nhist, nhand, -1)
        # Inference, don't use gt_trajectory
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb3d, rgb2d, pcd, instruction, proprio, extra_inputs=extra_inputs
            )

        # Training, use gt_trajectory to compute loss
        return self.compute_loss(
            gt_trajectory,
            rgb3d, rgb2d, pcd, instruction, proprio, extra_inputs=extra_inputs
        )


class TransformerHead(BaseTransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 num_shared_attn_layers=4,
                 rotary_pe=False):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            nhist=nhist,
            rotary_pe=False
        )
        # Positional embeddings
        self.pos_embed_2d = SinusoidalPosEmb(embedding_dim)

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        _traj_pos = torch.zeros_like(traj_feats)
        full_scene_pos = self.pos_embed_2d(
            torch.arange(0, rgb3d_feats.size(1), device=traj_feats.device)
        )[None].repeat(traj_feats.size(0), 1, 1)
        _scene_pos = self.pos_embed_2d(
            torch.arange(0, fps_scene_feats.size(1), device=traj_feats.device)
        )[None].repeat(traj_feats.size(0), 1, 1)
        _pos = torch.cat([_traj_pos, _scene_pos], 1)
        return _traj_pos, full_scene_pos, _pos

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        features = torch.cat([traj_feats, fps_scene_feats], 1)
        return features
