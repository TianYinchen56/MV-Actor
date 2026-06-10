import os
import time

import torch
from torch import nn

from utils.multiview_reprojection import build_cross_view_token_correspondence


def _sync_time(enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def chunked_knn_indices(query_xyz, key_xyz, k, *, exclude_self=False, chunk_size=256):
    batch, num_query, _ = query_xyz.shape
    num_key = key_xyz.shape[1]
    if exclude_self:
        k = min(k, max(num_key - 1, 1))
    else:
        k = min(k, max(num_key, 1))
    outputs = []
    for start in range(0, num_query, chunk_size):
        end = min(start + chunk_size, num_query)
        dist = torch.cdist(query_xyz[:, start:end], key_xyz)
        if exclude_self:
            local = torch.arange(start, end, device=query_xyz.device)
            chunk_index = torch.arange(end - start, device=query_xyz.device)
            dist[:, chunk_index, local] = float("inf")
        outputs.append(dist.topk(k=k, dim=-1, largest=False).indices)
    return torch.cat(outputs, dim=1)


def gather_neighbors(values, neighbor_idx):
    batch, _, channels = values.shape
    _, num_query, k = neighbor_idx.shape
    batch_idx = torch.arange(batch, device=values.device).view(batch, 1, 1).expand(-1, num_query, k)
    return values[batch_idx, neighbor_idx]


def pad_neighbor_idx(idx, target_k):
    if idx.size(-1) >= target_k:
        return idx[..., :target_k]
    pad_count = target_k - idx.size(-1)
    pad = idx[..., -1:].expand(*idx.shape[:-1], pad_count)
    return torch.cat([idx, pad], dim=-1)


def chunked_knn_indices_from_candidates(query_xyz, key_xyz, candidate_idx, k, *, chunk_size=256):
    batch, num_query, _ = query_xyz.shape
    if candidate_idx.dim() == 2:
        candidate_idx = candidate_idx.unsqueeze(0).expand(batch, -1, -1)
    outputs = []
    for start in range(0, num_query, chunk_size):
        end = min(start + chunk_size, num_query)
        query_xyz_chunk = query_xyz[:, start:end]
        cand_idx_chunk = candidate_idx[:, start:end]
        cand_xyz = gather_neighbors(key_xyz, cand_idx_chunk)
        dist = ((cand_xyz - query_xyz_chunk.unsqueeze(2)) ** 2).sum(dim=-1)
        topk = min(k, cand_idx_chunk.size(-1))
        top_local = dist.topk(k=topk, dim=-1, largest=False).indices
        chosen = torch.gather(cand_idx_chunk, 2, top_local)
        outputs.append(pad_neighbor_idx(chosen, k))
    return torch.cat(outputs, dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SigmoidTokenGate(nn.Module):
    def __init__(self, dim, gate_init=-6.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, float(gate_init))

    def forward(self, x):
        return torch.sigmoid(self.proj(self.norm(x)))


class LocalPointAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        residual_scale=1.0,
        query_chunk_size=256,
        use_token_gate=False,
        token_gate_init=-6.0,
        use_feedforward=True,
        use_pos=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.residual_scale = residual_scale
        self.query_chunk_size = query_chunk_size
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.use_feedforward = bool(use_feedforward)
        self.use_pos = bool(use_pos)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        ) if self.use_pos else None
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = FeedForward(dim) if self.use_feedforward else None
        self.use_token_gate = bool(use_token_gate)
        if self.use_token_gate:
            self.token_gate = SigmoidTokenGate(dim, gate_init=token_gate_init)
        else:
            self.token_gate = None

    def forward(self, query_feats, query_xyz, memory_feats, memory_xyz, neighbor_idx, neighbor_mask=None):
        residual = query_feats
        query_feats = self.query_norm(query_feats)
        memory_feats = self.memory_norm(memory_feats)

        outputs = []
        valid_chunks = []
        num_query = query_feats.shape[1]
        for start in range(0, num_query, self.query_chunk_size):
            end = min(start + self.query_chunk_size, num_query)
            query_feats_chunk = query_feats[:, start:end]
            query_xyz_chunk = query_xyz[:, start:end]
            neighbor_idx_chunk = neighbor_idx[:, start:end]
            neighbor_mask_chunk = None if neighbor_mask is None else neighbor_mask[:, start:end]

            neighbor_feats = gather_neighbors(memory_feats, neighbor_idx_chunk)
            neighbor_xyz = gather_neighbors(memory_xyz, neighbor_idx_chunk)
            delta_xyz = neighbor_xyz - query_xyz_chunk.unsqueeze(2)

            batch, chunk_queries, num_neighbors, _ = neighbor_feats.shape
            q = self.q_proj(query_feats_chunk).reshape(batch, chunk_queries, self.num_heads, self.head_dim)
            k = self.k_proj(neighbor_feats).reshape(batch, chunk_queries, num_neighbors, self.num_heads, self.head_dim)
            v = self.v_proj(neighbor_feats).reshape(batch, chunk_queries, num_neighbors, self.num_heads, self.head_dim)
            if self.pos_proj is None:
                pos = torch.zeros_like(k)
            else:
                pos = self.pos_proj(delta_xyz).reshape(batch, chunk_queries, num_neighbors, self.num_heads, self.head_dim)

            attn_logits = ((q.unsqueeze(2) * (k + pos)).sum(dim=-1)) * self.scale
            if neighbor_mask_chunk is not None:
                mask = neighbor_mask_chunk.unsqueeze(-1)
                attn_logits = attn_logits.masked_fill(~mask, -1e9)
                attn = torch.softmax(attn_logits, dim=2)
                attn = attn * mask.to(attn.dtype)
                attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-6)
                valid_chunks.append(neighbor_mask_chunk.any(dim=-1, keepdim=True))
            else:
                attn = torch.softmax(attn_logits, dim=2)
            updated = (attn.unsqueeze(-1) * (v + pos)).sum(dim=2).reshape(batch, chunk_queries, self.dim)
            outputs.append(updated)

        updated = torch.cat(outputs, dim=1)
        delta = self.out_proj(updated)
        if valid_chunks:
            valid_queries = torch.cat(valid_chunks, dim=1).to(delta.dtype)
            delta = delta * valid_queries
        delta = self.residual_scale * delta
        if self.use_token_gate:
            token_gate = self.token_gate(residual)
            return residual + token_gate * delta
        updated = residual + delta
        if self.ffn is not None:
            return self.ffn(updated)
        return updated



class PointwiseCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, residual_scale=0.1, use_gate=False, gate_init=-6.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.residual_scale = residual_scale
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.use_gate = use_gate
        self.token_gate = SigmoidTokenGate(dim, gate_init=gate_init) if use_gate else None

    def forward(self, query_feats, memory_feats):
        if query_feats.shape != memory_feats.shape:
            raise ValueError(
                f"PointwiseCrossAttention expects matching shapes, got {tuple(query_feats.shape)} and {tuple(memory_feats.shape)}"
            )
        residual = query_feats
        query_feats = self.query_norm(query_feats)
        memory_feats = self.memory_norm(memory_feats)

        batch, num_tokens, _ = query_feats.shape
        q = self.q_proj(query_feats).reshape(batch, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(memory_feats).reshape(batch, num_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(memory_feats).reshape(batch, num_tokens, self.num_heads, self.head_dim)

        attn_logits = (q * k).sum(dim=-1) * self.scale
        attn = torch.sigmoid(attn_logits).unsqueeze(-1)
        updated = (attn * v).reshape(batch, num_tokens, self.dim)
        delta = self.residual_scale * self.out_proj(updated)
        if self.token_gate is not None:
            token_gate = self.token_gate(residual)
            return residual + token_gate * delta
        return residual + delta


class InterleavedPointFusion(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        num_layers,
        cross_overlap_only,
        semantic_mode,
        cross_knn,
        cross_residual_scale,
        knn_chunk_size,
        cross_view_radius,
        cross_view_max_reproj_error,
        cross_view_max_depth_error,
        semantic_token_gate=False,
        semantic_token_gate_init=-6.0,
        semantic_use_pos=True,
        geometry_dim=1024,
    ):
        super().__init__()
        self.cross_overlap_only = cross_overlap_only
        self.semantic_mode = semantic_mode
        self.cross_knn = cross_knn
        self.cross_residual_scale = float(cross_residual_scale)
        self.knn_chunk_size = knn_chunk_size
        self.cross_view_radius = int(cross_view_radius)
        self.cross_view_max_reproj_error = float(cross_view_max_reproj_error)
        self.cross_view_max_depth_error = float(cross_view_max_depth_error)
        self.semantic_token_gate = bool(semantic_token_gate)
        self.semantic_token_gate_init = float(semantic_token_gate_init)
        self.semantic_use_pos = bool(semantic_use_pos)
        self.geometry_dim = geometry_dim
        self.semantic_grid_hw = (32, 32)
        self.geometry_grid_hw = (18, 18)
        self._cross_overlap_cache = {}
        self.semantic_type = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.geometry_type = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.geometry_proj = nn.Linear(geometry_dim, embedding_dim)
        self.geometry_norm = nn.LayerNorm(embedding_dim)
        self.last_profile = None
        self.semantic_blocks = nn.ModuleList([
            LocalPointAttention(
                embedding_dim,
                num_heads,
                residual_scale=1.0,
                query_chunk_size=knn_chunk_size,
                use_token_gate=self.semantic_token_gate,
                token_gate_init=self.semantic_token_gate_init,
                use_feedforward=False,
                use_pos=self.semantic_use_pos,
            )
            for _ in range(num_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            LocalPointAttention(
                embedding_dim,
                num_heads,
                residual_scale=cross_residual_scale,
                query_chunk_size=knn_chunk_size,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(embedding_dim)

    def _validate_inputs(self, semantic_feats, semantic_xyz, geometry_feats=None, geometry_xyz=None):
        if semantic_feats.dim() != 3 or semantic_xyz.dim() != 3:
            raise ValueError("semantic feats/xyz must be flattened to (B, N, C/3)")
        if geometry_feats is not None:
            if geometry_feats.dim() != 3 or geometry_xyz.dim() != 3:
                raise ValueError("geometry feats/xyz must be flattened to (B, N, C/3)")

    def _prepare_semantic(self, semantic_feats, add_type=True):
        if add_type:
            return semantic_feats + self.semantic_type.to(semantic_feats.dtype)
        return semantic_feats

    def _prepare_geometry(self, geometry_feats):
        if geometry_feats.size(-1) == 2 * self.geometry_dim:
            geometry_feats = geometry_feats[..., -self.geometry_dim:]
        elif geometry_feats.size(-1) != self.geometry_dim:
            raise ValueError(
                f"Expected geometry dim {self.geometry_dim} or {2 * self.geometry_dim}, got {geometry_feats.size(-1)}"
            )
        geometry = self.geometry_proj(geometry_feats)
        geometry = self.geometry_norm(geometry) + self.geometry_type.to(geometry.dtype)
        return geometry

    def prepare_geometry(self, geometry_feats):
        if geometry_feats.dim() != 3:
            raise ValueError(f"geometry feats must be flattened to (B, N, C), got shape={tuple(geometry_feats.shape)}")
        return self._prepare_geometry(geometry_feats)

    def _prepare_cross_view_semantic(self, semantic_xyz, semantic_view_meta):
        if semantic_view_meta is None:
            raise ValueError("cross_view_reproject semantic mode requires semantic_view_meta")
        cached_idx = semantic_view_meta.get("semantic_corr_idx")
        cached_mask = semantic_view_meta.get("semantic_corr_mask")
        if cached_idx is not None and cached_mask is not None:
            return (
                cached_idx.to(device=semantic_xyz.device, dtype=torch.long),
                cached_mask.to(device=semantic_xyz.device, dtype=torch.bool),
            )
        intrinsics = semantic_view_meta.get("camera_intrinsics")
        extrinsics = semantic_view_meta.get("camera_extrinsics")
        image_hw = semantic_view_meta.get("camera_image_hw")
        source_xyz = semantic_view_meta.get("semantic_correspondence_xyz")
        if source_xyz is None:
            source_xyz = semantic_xyz
        if intrinsics is None or extrinsics is None or image_hw is None:
            raise ValueError("cross_view_reproject semantic mode requires camera_intrinsics, camera_extrinsics, and camera_image_hw")

        batch, total_tokens, _ = semantic_xyz.shape
        sem_h, sem_w = self.semantic_grid_hw
        tokens_per_view = sem_h * sem_w
        if total_tokens % tokens_per_view != 0:
            raise ValueError(f"semantic token count {total_tokens} not divisible by per-view tokens {tokens_per_view}")
        num_views = total_tokens // tokens_per_view
        query_xyz = source_xyz.reshape(batch, num_views, tokens_per_view, 3).float()
        memory_xyz = semantic_xyz.reshape(batch, num_views, tokens_per_view, 3).float()
        intrinsics = intrinsics[:, :num_views].float()
        extrinsics = extrinsics[:, :num_views].float()
        neighbor_idx, neighbor_mask = build_cross_view_token_correspondence(
            query_xyz,
            memory_xyz,
            intrinsics,
            extrinsics,
            image_hw=image_hw,
            grid_hw=self.semantic_grid_hw,
            search_radius=self.cross_view_radius,
            max_reproj_error=self.cross_view_max_reproj_error,
            max_depth_error=self.cross_view_max_depth_error,
        )
        return neighbor_idx.to(device=semantic_xyz.device, dtype=torch.long), neighbor_mask.to(device=semantic_xyz.device)

    def _prepare_cross_knn(self, semantic_xyz, geometry_xyz, cross_knn_idx):
        semantic_xyz_f = semantic_xyz.float()
        geometry_xyz_f = geometry_xyz.float()
        if self.cross_overlap_only:
            candidate_idx = self._build_overlap_candidate_idx(
                semantic_xyz.device,
                semantic_xyz.shape[1],
                geometry_xyz.shape[1],
            )
            return chunked_knn_indices_from_candidates(
                semantic_xyz_f,
                geometry_xyz_f,
                candidate_idx,
                self.cross_knn,
                chunk_size=self.knn_chunk_size,
            )
        if cross_knn_idx is None:
            return chunked_knn_indices(
                semantic_xyz_f,
                geometry_xyz_f,
                self.cross_knn,
                exclude_self=False,
                chunk_size=self.knn_chunk_size,
            )
        if cross_knn_idx.size(-1) < self.cross_knn:
            raise ValueError(
                f"Cached cross_knn_idx has K={cross_knn_idx.size(-1)} < required {self.cross_knn}"
            )
        return cross_knn_idx[..., : self.cross_knn].to(device=semantic_xyz.device, dtype=torch.long)

    def _build_overlap_candidate_idx(self, device, num_semantic, num_geometry):
        key = (str(device), num_semantic, num_geometry)
        if key in self._cross_overlap_cache:
            return self._cross_overlap_cache[key]

        sem_h, sem_w = self.semantic_grid_hw
        geo_h, geo_w = self.geometry_grid_hw
        sem_per_cam = sem_h * sem_w
        geo_per_cam = geo_h * geo_w
        if num_semantic % sem_per_cam != 0 or num_geometry % geo_per_cam != 0:
            raise ValueError(
                f"Token counts not compatible with semantic grid {self.semantic_grid_hw} and geometry grid {self.geometry_grid_hw}: {num_semantic}, {num_geometry}"
            )
        ncam_sem = num_semantic // sem_per_cam
        ncam_geo = num_geometry // geo_per_cam
        if ncam_sem != ncam_geo:
            raise ValueError(f"Mismatched camera counts: semantic={ncam_sem}, geometry={ncam_geo}")

        all_candidates = []
        eps = 1e-6
        for cam in range(ncam_sem):
            geo_base = cam * geo_per_cam
            for sy in range(sem_h):
                y0 = sy / sem_h
                y1 = (sy + 1) / sem_h
                gy0 = min(geo_h - 1, max(0, int((y0 * geo_h))))
                gy1 = min(geo_h - 1, max(gy0, int(((y1 - eps) * geo_h))))
                for sx in range(sem_w):
                    x0 = sx / sem_w
                    x1 = (sx + 1) / sem_w
                    gx0 = min(geo_w - 1, max(0, int((x0 * geo_w))))
                    gx1 = min(geo_w - 1, max(gx0, int(((x1 - eps) * geo_w))))
                    cand = []
                    for gy in range(gy0, gy1 + 1):
                        for gx in range(gx0, gx1 + 1):
                            cand.append(geo_base + gy * geo_w + gx)
                    all_candidates.append(cand)

        max_cands = max(len(c) for c in all_candidates)
        candidate_idx = torch.empty((num_semantic, max_cands), dtype=torch.long, device=device)
        for i, cand in enumerate(all_candidates):
            if len(cand) < max_cands:
                cand = cand + [cand[-1]] * (max_cands - len(cand))
            candidate_idx[i] = torch.tensor(cand, dtype=torch.long, device=device)
        self._cross_overlap_cache[key] = candidate_idx
        return candidate_idx

    def forward_semantic(self, semantic_feats, semantic_xyz, semantic_view_meta=None):
        self._validate_inputs(semantic_feats, semantic_xyz)
        semantic = self._prepare_semantic(semantic_feats, add_type=True)
        if self.semantic_mode != "cross_view_reproject":
            raise ValueError(f"Unsupported semantic_mode={self.semantic_mode}; only cross_view_reproject is kept")
        semantic_neighbor_idx, semantic_neighbor_mask = self._prepare_cross_view_semantic(semantic_xyz, semantic_view_meta)
        for semantic_block in self.semantic_blocks:
            semantic = semantic_block(
                semantic,
                semantic_xyz,
                semantic,
                semantic_xyz,
                semantic_neighbor_idx,
                neighbor_mask=semantic_neighbor_mask,
            )
        return semantic

    def forward_cross(self, semantic_feats, semantic_xyz, geometry_feats, geometry_xyz, cross_knn_idx=None):
        if abs(self.cross_residual_scale) < 1e-12:
            return semantic_feats
        self._validate_inputs(semantic_feats, semantic_xyz, geometry_feats, geometry_xyz)
        geometry = self._prepare_geometry(geometry_feats)
        cross_knn_idx = self._prepare_cross_knn(semantic_xyz, geometry_xyz, cross_knn_idx)
        semantic = semantic_feats
        for cross_block in self.cross_blocks:
            semantic = cross_block(semantic, semantic_xyz, geometry, geometry_xyz, cross_knn_idx)
        return self.output_norm(semantic)

    def forward(
        self,
        semantic_feats,
        semantic_xyz,
        geometry_feats,
        geometry_xyz,
        cross_knn_idx=None,
        semantic_view_meta=None,
    ):
        profile_enabled = os.environ.get("PROFILE_INTERLEAVED_FUSION", "0") == "1"
        total_t0 = _sync_time(profile_enabled)
        prof = {
            "fusion_total_s": 0.0,
            "fusion_knn_self_s": 0.0,
            "fusion_knn_cross_s": 0.0,
            "fusion_semantic_blocks_s": 0.0,
            "fusion_cross_blocks_s": 0.0,
            "fusion_cached_idx": float(cross_knn_idx is not None),
        } if profile_enabled else None

        cross_disabled = abs(self.cross_residual_scale) < 1e-12
        if cross_disabled:
            self._validate_inputs(semantic_feats, semantic_xyz)
        else:
            self._validate_inputs(semantic_feats, semantic_xyz, geometry_feats, geometry_xyz)
        semantic = self._prepare_semantic(semantic_feats, add_type=True)
        geometry = None if cross_disabled else self._prepare_geometry(geometry_feats)

        t0 = _sync_time(profile_enabled)
        if self.semantic_mode != "cross_view_reproject":
            raise ValueError(f"Unsupported semantic_mode={self.semantic_mode}; only cross_view_reproject is kept")
        semantic_neighbor_idx, semantic_neighbor_mask = self._prepare_cross_view_semantic(semantic_xyz, semantic_view_meta)
        if profile_enabled:
            prof["fusion_knn_self_s"] += _sync_time(True) - t0

        if not cross_disabled:
            t0 = _sync_time(profile_enabled)
            cross_knn_idx = self._prepare_cross_knn(semantic_xyz, geometry_xyz, cross_knn_idx)
            if profile_enabled:
                prof["fusion_knn_cross_s"] += _sync_time(True) - t0

        for semantic_block, cross_block in zip(self.semantic_blocks, self.cross_blocks):
            t0 = _sync_time(profile_enabled)
            semantic = semantic_block(
                semantic,
                semantic_xyz,
                semantic,
                semantic_xyz,
                semantic_neighbor_idx,
                neighbor_mask=semantic_neighbor_mask,
            )
            if profile_enabled:
                prof["fusion_semantic_blocks_s"] += _sync_time(True) - t0
            if not cross_disabled:
                t0 = _sync_time(profile_enabled)
                semantic = cross_block(semantic, semantic_xyz, geometry, geometry_xyz, cross_knn_idx)
                if profile_enabled:
                    prof["fusion_cross_blocks_s"] += _sync_time(True) - t0

        if not cross_disabled:
            semantic = self.output_norm(semantic)
        if profile_enabled:
            prof["fusion_total_s"] = _sync_time(True) - total_t0
            self.last_profile = prof
        return semantic
