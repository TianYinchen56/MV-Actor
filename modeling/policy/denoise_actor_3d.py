import torch

from ..encoder.multimodal.encoder3d import Encoder
from ..utils.position_encodings import RotaryPositionEncoding3D

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
            finetune_text_encoder=finetune_text_encoder,
            enable_interleaved_point_fusion=enable_interleaved_point_fusion,
            interleaved_before_language=interleaved_before_language,
            interleaved_text_between_self_and_geo=interleaved_text_between_self_and_geo,
            interleaved_cross_overlap_only=interleaved_cross_overlap_only,
            interleaved_point_fusion_layers=interleaved_point_fusion_layers,
            interleaved_semantic_mode=interleaved_semantic_mode,
            interleaved_cross_knn=interleaved_cross_knn,
            interleaved_cross_residual_scale=interleaved_cross_residual_scale,
            interleaved_knn_chunk_size=interleaved_knn_chunk_size,
            interleaved_cross_view_radius=interleaved_cross_view_radius,
            interleaved_cross_view_max_reproj_error=interleaved_cross_view_max_reproj_error,
            interleaved_cross_view_max_depth_error=interleaved_cross_view_max_depth_error,
            pi3x_geometry_mode=pi3x_geometry_mode,
            geo_upsample_mode=geo_upsample_mode,
            pointwise_parallel_branch_fusion=pointwise_parallel_branch_fusion,
            pointwise_parallel_branch_text_last=pointwise_parallel_branch_text_last,
            interleaved_semantic_use_pos=interleaved_semantic_use_pos,
            disable_semantic_fusion=disable_semantic_fusion,
            disable_geometry_fusion=disable_geometry_fusion,
            router_init_bias=router_init_bias,
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers
        )


class TransformerHead(BaseTransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 num_shared_attn_layers=4,
                 rotary_pe=True):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            num_shared_attn_layers=num_shared_attn_layers,
            rotary_pe=rotary_pe
        )

        # Relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos)
        rel_fps_pos = self.relative_pe_layer(fps_scene_pos)
        rel_pos = torch.cat([rel_traj_pos, rel_fps_pos], 1)
        return rel_traj_pos, rel_scene_pos, rel_pos

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        features = torch.cat([traj_feats, fps_scene_feats], 1)
        return features
