# Interleaved Point Fusion

这个模块实现一条以 SigLIP2 语义点云为主轴的交替式融合路径：

1. `Semantic Self-Attention (语义点云内部聚合)`
2. `Semantic -> Geometry Cross-Attention (语义点云查询 Pi3X 几何点云)`

输入假定为：
- 语义特征与坐标：`siglip2_tokens`, `token_xyz_siglip2_sim3`
- 几何特征与坐标：`pi3x_tokens`, `token_xyz_pi3_sim3`
- 可选邻域缓存：`interleaved_semantic_corr_idx`, `interleaved_semantic_corr_mask`, `interleaved_cross_knn_idx`

输出仍然落在语义点云 support 上，便于无缝接回原版 3DFA policy。

如果 zarr 里提供了语义对应缓存和几何 KNN 缓存，训练/评估时会直接复用；语义侧不再使用 KNN。
