# Interleaved Point Fusion

This module implements an interleaved fusion path with the CLIP semantic point cloud as the main support:

1. `Semantic Self-Attention`: aggregates features within the semantic point cloud.
2. `Semantic -> Geometry Cross-Attention`: queries the Pi3X geometry point cloud from semantic tokens.

Expected inputs:

- Semantic features and coordinates: `clip_tokens`, `token_xyz_clip_sim3`.
- Geometry features and coordinates: `pi3x_tokens`, `token_xyz_pi3_sim3`.
- Optional neighborhood caches: `interleaved_semantic_corr_idx`, `interleaved_semantic_corr_mask`, `interleaved_cross_knn_idx`.

The output stays on the CLIP semantic point support, so it can be passed back to the original 3DFA policy path without changing the action head.

If the zarr data provides semantic correspondence caches and geometry KNN caches, training and evaluation reuse them directly. The semantic branch does not use KNN in that case.
