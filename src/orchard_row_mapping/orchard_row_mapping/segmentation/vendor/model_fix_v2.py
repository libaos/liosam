import torch
import torch.nn as nn
import logging

# 配置日志
logger = logging.getLogger(__name__)

# Try torch_points_kernels for fast KNN; fall back to other GPU paths and pure PyTorch.
try:
    from torch_points_kernels import knn as tpk_knn
    _TPK_AVAILABLE = True
except Exception:
    tpk_knn = None
    _TPK_AVAILABLE = False
    logger.warning(
        "torch_points_kernels not available; will try torch_cluster for GPU KNN and fall back to simple KNN."
    )

try:
    from torch_cluster import knn as tc_knn
    _TC_AVAILABLE = True
except Exception:
    tc_knn = None
    _TC_AVAILABLE = False

from .simple_knn import knn as simple_knn


def _torch_cluster_knn(pos_support, pos, k):
    """KNN via torch_cluster; returns (idx, dist2) like torch_points_kernels."""
    if tc_knn is None:
        raise RuntimeError("torch_cluster is unavailable")
    if pos_support.dim() != 3 or pos.dim() != 3:
        raise ValueError("pos_support/pos must be (B, N, 3) and (B, M, 3)")

    batch_size, n_support, _ = pos_support.shape
    n_query = pos.shape[1]
    device = pos_support.device

    x = pos_support.reshape(batch_size * n_support, 3)
    y = pos.reshape(batch_size * n_query, 3)
    batch_x = torch.arange(batch_size, device=device).repeat_interleave(n_support)
    batch_y = torch.arange(batch_size, device=device).repeat_interleave(n_query)

    row, col = tc_knn(x, y, k, batch_x=batch_x, batch_y=batch_y)
    if row.numel() != batch_size * n_query * k:
        raise RuntimeError("torch_cluster knn returned unexpected edge count")

    order = row.argsort()
    row = row[order]
    col = col[order]

    idx = (col % n_support).view(batch_size, n_query, k)
    neigh = x[col].view(batch_size * n_query, k, 3)
    centers = y[row].view(batch_size * n_query, k, 3)
    dist2 = (neigh - centers).pow(2).sum(dim=-1).view(batch_size, n_query, k)
    return idx, dist2


def _knn_with_fallback(pos_support, pos, k):
    """Run KNN on the current device; fall back if CUDA kernels are unavailable."""
    global _TPK_AVAILABLE, _TC_AVAILABLE

    if _TPK_AVAILABLE and pos_support.is_cuda:
        try:
            return tpk_knn(pos_support, pos, k)
        except Exception as exc:
            # Disable torch_points_kernels after the first runtime failure to avoid
            # exception overhead + log spam inside the forward pass.
            _TPK_AVAILABLE = False
            logger.warning("torch_points_kernels knn failed on %s: %s", pos_support.device, exc)

    if _TC_AVAILABLE and pos_support.is_cuda:
        try:
            return _torch_cluster_knn(pos_support, pos, k)
        except Exception as exc:
            # Same idea: disable after first failure to keep logs clean.
            _TC_AVAILABLE = False
            logger.warning("torch_cluster knn failed on %s: %s", pos_support.device, exc)
    try:
        return simple_knn(pos_support, pos, k)
    except RuntimeError as exc:
        if pos_support.is_cuda:
            logger.warning("simple_knn failed on CUDA (%s); retrying on CPU", exc)
            pos_support_cpu = pos_support.detach().cpu()
            pos_cpu = pos.detach().cpu()
            idx, dist = simple_knn(pos_support_cpu, pos_cpu, k)
            return idx.to(pos_support.device), dist.to(pos_support.device)
        raise

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            knn_output: tuple or other format
                可能是(idx, dist)元组或其他格式的KNN输出

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # 处理knn_output，确保我们有idx和dist
        if isinstance(knn_output, tuple) and len(knn_output) >= 2:
            # 如果是元组格式，直接解包
            idx, dist = knn_output[0], knn_output[1]
        else:
            # 如果不是元组或元组长度不对，假设knn_output就是idx
            idx = knn_output
            # 计算距离（简单的占位值）
            B, N, K = idx.size()
            dist = torch.ones((B, N, K), device=idx.device)
            
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors
        self.device = device

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        # Keep KNN on the same device to enable CUDA acceleration when available.
        coords_knn = coords.contiguous()
        
        try:
            # 尝试使用标准KNN函数
            # 使用安全的KNN调用方式
            knn_output = _knn_with_fallback(coords_knn, coords_knn, self.num_neighbors)
            # 将结果移回到与特征相同的设备上
            if isinstance(knn_output, tuple):
                idx = knn_output[0].to(features.device)
                dist = knn_output[1].to(features.device) if len(knn_output) > 1 else None
            else:
                idx = knn_output.to(features.device)
                dist = None
                
            # 如果没有距离信息，创建一个占位值
            if dist is None:
                B, N, K = idx.size()
                dist = torch.ones((B, N, K), device=features.device)
        except Exception as e:
            logger.warning("KNN calculation failed: %s; using fallback indices", e)
            # 备用方法：简单地使用最近的点
            B, N, _ = coords.shape
            idx = torch.arange(N, device=features.device).view(1, N, 1).repeat(B, 1, self.num_neighbors)
            dist = torch.ones((B, N, self.num_neighbors), device=features.device)

        x = self.mlp1(features)

        x = self.lse1(coords, x, idx)
        x = self.pool1(x)

        x = self.lse2(coords, x, idx)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.device = device

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

        self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N).to(input.device)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            # Keep KNN on the same device as coords when possible.
            coords_current = coords[:,:N//decimation_ratio].contiguous()
            coords_upsampled = coords[:,:d*N//decimation_ratio].contiguous()
            
            try:
                # 使用安全的KNN调用方式
                knn_output = _knn_with_fallback(coords_current, coords_upsampled, 1)
                if isinstance(knn_output, tuple):
                    neighbors = knn_output[0].to(input.device)
                else:
                    neighbors = knn_output.to(input.device)
            except Exception as e:
                logger.warning("Decoder KNN failed: %s; using fallback indices", e)
                B = coords.shape[0]
                neighbors = torch.zeros((B, N//decimation_ratio, 1), dtype=torch.long, device=input.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)


# 4D输入修复的RandLANet
class Fixed4DRandLANet(RandLANet):
    def forward(self, input):
        """
        前向传播函数 - 自动处理输入维度问题
        
        参数:
        - input: 形状为 (B, N, d_in) 或 (N, d_in) 的点云数据
        
        返回:
        - 形状为 (B, num_classes, N) 的分割得分
        """
        # 确保输入是批次格式 (B, N, d_in)
        if input.dim() == 2:
            input = input.unsqueeze(0)  # 添加批次维度
            
        return super().forward(input)


if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 7
    cloud = 1000*torch.randn(1, 2**16, d_in).to(device)
    model = Fixed4DRandLANet(d_in, 6, 16, 4, device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    model.eval()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    # print(pred)
    print(t1-t0)
