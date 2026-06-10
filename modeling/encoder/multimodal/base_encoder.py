import os

import torch
from torch import nn

from ...utils.layers import AttentionModule
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__()
        self.subsampling_factor = fps_subsampling_factor
        self._backbone_name = backbone

        # Instruction encoder
        self.text_encoder, _dim = fetch_text_encoders(backbone)
        if self.text_encoder is not None:  # is None when using a VLM
            for p in self.text_encoder.parameters():
                p.requires_grad = finetune_text_encoder
            self.instruction_encoder = nn.Linear(_dim, embedding_dim)

        # Scene encoder
        self.backbone, self.normalize = fetch_visual_encoders(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Attention from vision to language
        if backbone in {'clip', 'siglip2'}:
            self.vl_attention = AttentionModule(
                num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
                dim_fw=4 * embedding_dim, n_heads=num_attn_heads, pre_norm=False
            )

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio, extra_inputs=None):
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W)
            - rgb2d: (B, ncam2d, 3, H, W)
            - pcd: (B, ncam3d, 3, H, W)
            - instruction: (B, nt), tokens
            - proprio: (B, nhist, 3+6+X)

        Returns:
            - rgb3d_feats: (B, N, F)
            - pcd: (B, N, 3)
            - rgb2d_feats: (B, N2d, F)
            - rgb2d_pos: (B, N2d, 3)
            - instr_feats: (B, L, F)
            - instr_pos: (B, L, 3)
            - proprio_feats: (B, nhist, F)
            - fps_scene_feats: (B, N, F)
            - fps_scene_pos: (B, N, 3)
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
            'siglip2': self.encode_siglip2,
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )
        if self.should_insert_instruction_between_fusion_stages():
            rgb3d_feats, pcd = self.fuse_scene_tokens(
                rgb3d_feats, pcd, extra_inputs=extra_inputs, stage="semantic"
            )
            rgb3d_feats = self.condition_scene_with_instruction(
                rgb3d_feats, instr_feats
            )
            rgb3d_feats, pcd = self.fuse_scene_tokens(
                rgb3d_feats, pcd, extra_inputs=extra_inputs, stage="cross"
            )
        else:
            rgb3d_feats, pcd = self.fuse_scene_tokens(
                rgb3d_feats, pcd, extra_inputs=extra_inputs
            )
            rgb3d_feats = self.condition_scene_with_instruction(
                rgb3d_feats, instr_feats
            )
        rgb2d_pos = None

        # Use the current end-effector position as language 'position'
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # Encode proprioception
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd)

        enable_subsampling = os.environ.get("ENABLE_SCENE_SUBSAMPLING", "0") == "1"
        if enable_subsampling:
            fps_scene_feats, fps_scene_pos = self.run_dps(rgb3d_feats, pcd)
        else:
            fps_scene_feats, fps_scene_pos = rgb3d_feats, pcd

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            rgb3d_feats, fps_scene_feats, fps_scene_pos,
            rgb3d_feats, fps_scene_feats, fps_scene_pos,
        )

    def fuse_scene_tokens(self, rgb3d_feats, pcd, extra_inputs=None, stage="all"):
        return rgb3d_feats, pcd

    def should_insert_instruction_between_fusion_stages(self):
        return False

    def condition_scene_with_instruction(self, rgb3d_feats, instr_feats):
        return rgb3d_feats

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        return None

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        return None, None, None, None

    def encode_siglip2(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings for SigLIP2.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        return None, None, None, None

    def run_dps(self, features, pos):
        if self.subsampling_factor == 1:
            return features, pos

        _, _, ch = features.shape
        sampled_inds = density_based_sampler(features, self.subsampling_factor)
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_features = torch.gather(features, 1, expanded_inds)
        if pos is None:
            return sampled_features, None

        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    _, npts, _ = features.shape
    dists = torch.cdist(features, features, p=2)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)
    num_keep = int(npts // subsample_factor)
    sampled_inds = density.topk(num_keep, dim=-1, largest=True).indices
    return sampled_inds
