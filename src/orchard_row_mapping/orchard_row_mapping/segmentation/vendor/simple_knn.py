"""简单的 KNN 实现,替代 torch_points_kernels.knn

注意: torch_points_kernels.knn 的参数顺序是 (pos_support, pos, k)
即先支持点,后查询点
"""
import torch


def knn(pos_support, pos, k):
    """
    简单的 KNN 实现 (纯 PyTorch,无需编译)

    匹配 torch_points_kernels.knn 的接口:
    - 参数顺序: (pos_support, pos, k)
    - 返回: (idx, dist2) 其中 dist2 是平方距离

    Args:
        pos_support: (B, N, 3) 支持点 (被搜索的点集)
        pos: (B, M, 3) 查询点 (查询中心)
        k: int, 最近邻数量

    Returns:
        idx: (B, M, k) 最近邻索引
        dist2: (B, M, k) 平方距离
    """
    # 确保输入是 3D 张量
    assert pos_support.dim() == 3 and pos.dim() == 3, \
        f"Expected 3D tensors, got pos_support: {pos_support.dim()}D, pos: {pos.dim()}D"

    B, M, _ = pos.shape
    N = pos_support.shape[1]

    # 计算距离矩阵 (B, M, N)
    # pos: (B, M, 3) -> (B, M, 1, 3)
    # pos_support: (B, N, 3) -> (B, 1, N, 3)
    diff = pos.unsqueeze(2) - pos_support.unsqueeze(1)  # (B, M, N, 3)
    dist2_matrix = torch.sum(diff ** 2, dim=-1)  # (B, M, N)

    # 找到 k 个最近邻
    # 注意: k 不能大于 N
    k = min(k, N)
    dist2, idx = torch.topk(dist2_matrix, k, dim=-1, largest=False, sorted=True)

    return idx, dist2
