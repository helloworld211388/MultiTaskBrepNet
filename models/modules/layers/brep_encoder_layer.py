from typing import Callable, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from .multihead_attention import MultiheadAttention
from .feature_encoders import SurfaceEncoder, CurveEncoder

class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        if activation_fn == "mish":
            self.activation_fn = F.mish
        else:
            self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NonLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = F.relu(self.bn2(self.linear2(x)))
        return x


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """
#对应论文的Feature encoding过程
    def __init__(
            self, num_heads, num_degree, hidden_dim, n_layers, lpe_dim, lpe_n_heads, lpe_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.lpe_dim = lpe_dim

        # --- 修改开始: 为LPE编码调整特征维度 ---
        # 目标总维度: hidden_dim (e.g., 256)
        # LPE维度: lpe_dim (e.g., 32)
        # 剩余维度: remaining_dim = hidden_dim - lpe_dim
        remaining_dim = hidden_dim - lpe_dim

        # 重新分配剩余维度给其他特征
        surf_dim = int(remaining_dim * 0.375)
        centroid_dim = int(remaining_dim * 0.125)
        adj_stats_dim = int(remaining_dim * 0.125)


        other_dim_count = 7  # 新值: 7 (face_area, face_type, face_loop, degree, curvature, inner_loops, rational)
        #计算每个其它特征的维度
        other_dim = (remaining_dim - surf_dim - centroid_dim - adj_stats_dim) // other_dim_count

        # 原有的编码器，使用新的输出维度
        self.surf_encoder = SurfaceEncoder(
            in_channels=7, output_dims=surf_dim
        )
        self.face_area_encoder = NonLinear(1, other_dim)
        self.face_type_encoder = nn.Embedding(9, other_dim, padding_idx=0)#Embedding类似于词袋模型的权重矩阵，可以输入一个词的索引，输出该词的向量表示
        self.face_loop_encoder = nn.Embedding(256, other_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(num_degree, other_dim, padding_idx=0)
        self.centroid_encoder = NonLinear(3, centroid_dim)
        self.curvature_encoder = nn.Embedding(3, other_dim)
        self.inner_loops_encoder = NonLinear(2, other_dim)#只有2个吗？
        self.adj_stats_encoder = NonLinear(18, adj_stats_dim)
        self.rational_encoder = nn.Embedding(2, other_dim)

        # 新增: LPE处理模块
        self.linear_A = nn.Linear(2, lpe_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=lpe_dim, nhead=lpe_n_heads,
                                                   batch_first=False)  # SAN使用的是(seq, batch, feature)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=lpe_layers)
        # --- 修改结束 ---

        # 全局图标志
        self.graph_token = nn.Embedding(1, hidden_dim)#这个是一个全局的图标志
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, face_area, face_type, face_loop, face_degree,
                centroid, curvature, inner_loops, adj_stats, rational,
                EigVecs, EigVals,
                padding_mask):

        # ======================= 调试代码块 (开始) =======================
        # 目标: 逐一检查输入和中间变量，找到第一个出现 NaN/inf 的地方

        print("\n--- [调试] 进入 GraphNodeFeature.forward ---")

        def check_tensor(name, tensor):
            if not isinstance(tensor, torch.Tensor):
                print(f"  - {name}: 不是一个张量 (类型: {type(tensor)})")
                return

            # is_initialized = tensor.is_cuda and tensor.storage().size() > 0 # 检查是否已初始化
            # if not is_initialized:
            #      print(f"  - {name}: 张量未初始化!")
            #      return

            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()

            if has_nan or has_inf:
                print(f"  - [!!! 问题定位 !!!] {name}: 包含无效值! NaN: {has_nan}, Inf: {has_inf}")
            else:
                # 只对非空张量计算统计值
                if tensor.numel() > 0:
                    print(
                        f"  - {name}: 状态正常. Shape: {tensor.shape}, Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}")
                else:
                    print(f"  - {name}: 状态正常. Shape: {tensor.shape} (空张量)")

        # 1. 检查原始输入
        print("\n--- 1. 检查从数据加载器传入的原始张量 ---")
        check_tensor("输入: x (UV grid)", x)
        check_tensor("输入: face_area", face_area)
        check_tensor("输入: centroid", centroid)
        check_tensor("输入: inner_loops", inner_loops)
        check_tensor("输入: adj_stats", adj_stats)
        check_tensor("输入: EigVecs", EigVecs)
        check_tensor("输入: EigVals", EigVals)

        n_graph, n_node = padding_mask.size()[:2]
        node_pos = torch.where(padding_mask == False)

        # 2. 检查LPE计算过程
        print("\n--- 2. 检查LPE (拉普拉斯位置编码) 计算过程 ---")
        EigVecs_safe = torch.nan_to_num(EigVecs, nan=0.0, posinf=0.0, neginf=0.0)
        EigVals_safe = torch.nan_to_num(EigVals, nan=0.0, posinf=0.0, neginf=0.0)

        eig_vecs_norm = F.normalize(EigVecs_safe, p=2, dim=1, eps=1e-12)
        check_tensor("LPE: eig_vecs_norm (归一化后)", eig_vecs_norm)

        eig_vals_clamped = torch.clamp(EigVals_safe, min=-100.0, max=100.0)
        num_nodes_per_graph = (~padding_mask).sum(dim=1)
        EigVals_repeated = torch.repeat_interleave(eig_vals_clamped, num_nodes_per_graph, dim=0)

        PosEnc = torch.cat(
            (eig_vecs_norm.unsqueeze(2), EigVals_repeated.unsqueeze(2)), dim=2
        ).float()
        check_tensor("LPE: PosEnc (拼接后)", PosEnc)

        empty_mask = torch.isnan(PosEnc)
        PosEnc[empty_mask] = 0
        PosEnc = torch.transpose(PosEnc, 0, 1).float()
        PosEnc = self.linear_A(PosEnc)
        check_tensor("LPE: PosEnc (linear_A后)", PosEnc)

        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:, :, 0])
        check_tensor("LPE: PosEnc (Transformer后)", PosEnc)

        PosEnc[torch.transpose(empty_mask, 0, 1)[:, :, 0]] = float('nan')
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        PosEnc = torch.nan_to_num(PosEnc, nan=0.0, posinf=0.0, neginf=0.0)
        check_tensor("LPE: PosEnc (最终)", PosEnc)

        # 3. 检查其他特征编码器
        print("\n--- 3. 检查其他特征编码器的输出 ---")
        x = x.permute(0, 3, 1, 2)
        x_ = self.surf_encoder(x)
        check_tensor("特征: x_ (surf_encoder)", x_)

        face_area_ = self.face_area_encoder(face_area.unsqueeze(dim=1))
        check_tensor("特征: face_area_", face_area_)

        face_type_ = self.face_type_encoder(face_type)
        face_loop_ = self.face_loop_encoder(face_loop)
        face_degree_ = self.degree_encoder(face_degree)

        centroid_ = self.centroid_encoder(centroid)
        check_tensor("特征: centroid_", centroid_)

        curvature_ = self.curvature_encoder(curvature)

        inner_loops_ = self.inner_loops_encoder(inner_loops)
        check_tensor("特征: inner_loops_", inner_loops_)

        adj_stats_ = self.adj_stats_encoder(adj_stats)
        check_tensor("特征: adj_stats_", adj_stats_)

        rational_ = self.rational_encoder(rational)

        # 4. 检查最终拼接的特征
        print("\n--- 4. 检查最终拼接的节点特征 ---")
        node_feature = torch.cat((
            x_, face_area_, face_type_, face_loop_, face_degree_,
            centroid_, curvature_, inner_loops_, rational_, adj_stats_,
            PosEnc
        ), dim=-1)
        check_tensor("最终: node_feature", node_feature)

        print("--- [调试] GraphNodeFeature.forward 结束 ---\n")
        # ======================= 调试代码块 (结束) =======================

        # --- 原有的 forward 逻辑 ---
        face_feature = torch.zeros([n_graph, n_node, self.hidden_dim], device=x.device, dtype=x.dtype)
        face_feature[node_pos] = node_feature[:]

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, face_feature], dim=1)
        return graph_node_feature, node_feature


class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h), inplace= True)
        return h


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(
            self,
            dim_node,
            gamma,
            num_heads,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.multi_hop_max_dist = multi_hop_max_dist

        # spatial_feature encode
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # D2 A3 encoder
        # self.d2_pos_encoder = nn.Linear(64, num_heads, bias=False)
        # self.ang_pos_encoder = nn.Linear(64, num_heads, bias=False)
        self.d2_pos_encoder = NonLinear(64, num_heads)
        self.ang_pos_encoder = NonLinear(64, num_heads)
        # 【新增】为全局质心距离矩阵创建编码器
        self.centroid_dist_encoder = NonLinear(1, num_heads)
        # edge_feature encode
        self.curv_encoder = CurveEncoder(in_channels=12, output_dims=num_heads)
        self.edge_type_encoder = nn.Embedding(8, num_heads, padding_idx=0)
        self.edge_len_encoder = NonLinear(1, num_heads)
        self.edge_ang_encoder = NonLinear(1, num_heads)
        self.edge_conv_encoder = nn.Embedding(4, num_heads, padding_idx=0)

        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Parameter(torch.randn(num_edge_dis, num_heads, num_heads))
            self.node_cat = _EdgeConv(
                edge_feats=num_heads,
                out_feats=num_heads,
                node_feats=dim_node,
            )

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, attn_bias, spatial_pos, d2_distance, ang_distance,
                centroid_distance, edge_data, edge_type, edge_len, edge_ang, edge_conv, edge_path, edge_padding_mask,
                graph, node_feat):

        n_graph, n_node = edge_path.size()[:2]

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # [n_graph, n_head, n_node+1, n_node+1]

        # 1. 空间位置编码 (spatial_pos)
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # 2. D2 距离编码
        d2_distance = d2_distance.reshape(-1, 64)
        d2_pos_bias = self.d2_pos_encoder(d2_distance).reshape(n_graph, n_node, n_node, self.num_heads).permute(0, 3, 1,
                                                                                                                2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + d2_pos_bias

        # 3. A3 角度编码
        ang_distance = ang_distance.reshape(-1, 64)
        ang_pos_bias = self.ang_pos_encoder(ang_distance).reshape(n_graph, n_node, n_node, self.num_heads).permute(0, 3,
                                                                                                                   1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + ang_pos_bias

        # 4. 质心距离编码
        centroid_dist_with_channel = centroid_distance.unsqueeze(-1)
        reshaped_input = centroid_dist_with_channel.reshape(-1, 1)
        encoded_bias = self.centroid_dist_encoder(reshaped_input)
        centroid_dist_bias = encoded_bias.reshape(n_graph, n_node, n_node, self.num_heads).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + centroid_dist_bias

        # 5. 边特征编码 (multi_hop)
        # if self.edge_type == "multi_hop":
        #     spatial_pos_ = spatial_pos.clone()
        #     spatial_pos_[spatial_pos_ == 0] = 1
        #     spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
        #     spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
        #
        #     max_dist = self.multi_hop_max_dist
        #     edge_data = edge_data.permute(0, 2, 1)
        #     edge_data_ = self.curv_encoder(edge_data)
        #     edge_type_ = self.edge_type_encoder(edge_type)
        #     normalized_edge_len = torch.log1p(edge_len)
        #     edge_len_ = self.edge_len_encoder(normalized_edge_len.unsqueeze(dim=1))
        #     edge_ang_ = self.edge_ang_encoder(edge_ang.unsqueeze(dim=1))
        #     edge_conv_ = self.edge_conv_encoder(edge_conv)
        #     edge_feat = edge_data_ + edge_type_ + edge_len_ + edge_ang_ + edge_conv_
        #
        #     edge_feat_ = self.node_cat(graph, node_feat, edge_feat)
        #
        #     zero_feature = torch.zeros(1, edge_feat_.size(-1), device=edge_feat_.device, dtype=edge_feat_.dtype)
        #     edge_feature_global = torch.cat([zero_feature, edge_feat_], dim=0)
        #
        #     edge_path_reshaped = edge_path.reshape(n_graph, n_node * n_node * max_dist)
        #     edge_bias = edge_feature_global[edge_path_reshaped]
        #     edge_bias = edge_bias.reshape(n_graph, n_node, n_node, max_dist, self.num_heads)
        #
        #     edge_bias = edge_bias.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        #     edge_bias = torch.bmm(edge_bias, self.edge_dis_encoder[:max_dist, :, :])
        #     edge_bias = edge_bias.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
        #     edge_bias = (edge_bias.sum(-2) / (spatial_pos_.float().unsqueeze(-1)))
        #     edge_bias = edge_bias.permute(0, 3, 1, 2)
        #     graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_bias

        # --- 修改的核心逻辑: 在所有边相关的偏置计算完后，应用 gamma 缩放 ---
        # 使用 spatial_pos (最短路径距离) 来识别虚拟边
        fake_edge_mask = (spatial_pos > 1).unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 对虚拟边对应的注意力偏置乘以 gamma
        graph_attn_bias[:, :, 1:, 1:][fake_edge_mask] *= self.gamma
        # --- 修改结束 ---

        # 6. 添加全局 Token 的偏置
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # 7. 添加原始偏置并返回
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias