from typing import Optional, Tuple
import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .layers.multihead_attention import MultiheadAttention
from .layers.brep_encoder_layer import GraphEncoderLayer, GraphNodeFeature, GraphAttnBias


def init_params(module):
    def normal_(data):

        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)

        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class BrepEncoder(nn.Module):
    def __init__(
            self,
            num_degree: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            # 原参数 num_encoder_layers 被替换为更具体的层数定义
            num_shared_layers: int = 4,
            num_semantic_layers: int = 3,
            num_instance_layers: int = 4,
            gamma: float = 0.1,
            # 新增LPE相关参数
            lpe_dim: int = 32,
            lpe_n_heads: int = 4,
            lpe_layers: int = 2,
            embedding_dim: int = 128,
            ffn_embedding_dim: int = 128,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            pre_layernorm: bool = False,
            apply_params_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
    ) -> None:
        super().__init__()

        # --- 代码来源，未做修改 ---
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_params_init = apply_params_init
        self.traceable = traceable


        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_degree=num_degree,
            hidden_dim=embedding_dim,
            n_layers=num_shared_layers + num_semantic_layers + num_instance_layers,
            lpe_dim=lpe_dim,
            lpe_n_heads=lpe_n_heads,
            lpe_layers=lpe_layers,
        )


        self.graph_attn_bias = GraphAttnBias(
            dim_node=embedding_dim,
            gamma=gamma,
            num_heads=num_attention_heads,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_shared_layers + num_semantic_layers + num_semantic_layers,
        )


        self.embed_scale = embed_scale

        self.tanh = nn.Tanh()

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),  # --- 代码来源，未做修改 ---
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:  # --- 代码来源，未做修改 ---
            self.emb_layer_norm = None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            # --- 修改开始 ---
            # 将原有的单一 self.layers 拆分为共享层和任务特定层
            self.shared_layers = LayerDropModuleList(p=self.layerdrop)
            self.semantic_layers = LayerDropModuleList(p=self.layerdrop)
            self.instance_layers = LayerDropModuleList(p=self.layerdrop)
            # --- 修改结束 ---
        else:
            # --- 修改开始 ---
            self.shared_layers = nn.ModuleList([])
            self.semantic_layers = nn.ModuleList([])
            self.instance_layers = nn.ModuleList([])
            # --- 修改结束 ---

        # --- 修改开始 ---
        # 填充共享编码模块
        self.shared_layers.extend(
            [
                self.build_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,  # --- 代码来源，未做修改 ---
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,  # --- 代码来源，未做修改 ---
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_shared_layers)
            ]
        )

        # 填充语义分割任务特定编码模块
        self.semantic_layers.extend(
            [
                self.build_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_semantic_layers)
            ]
        )

        # 填充实例分割任务特定编码模块
        self.instance_layers.extend(
            [
                self.build_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_instance_layers)
            ]
        )
        # --- 修改结束 ---

        # Apply initialization of model params after building the model
        if self.apply_params_init:  # --- 代码来源，未做修改 ---
            self.apply(init_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False  # --- 代码来源，未做修改 ---

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        # --- 修改开始 ---
        # 冻结层数时，仅冻结共享层
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.shared_layers[layer])
        # --- 修改结束 ---

    def build_graph_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,  # --- 代码来源，未做修改 ---
            activation_dropout,
            activation_fn,
            export,
            q_noise,
            qn_block_size,
            pre_layernorm,
    ):
        return GraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,  # --- 代码来源，未做修改 ---
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,  # --- 代码来源，未做修改 ---
        )

    def forward(
            self,
            batch_data,
            perturb=None,
            last_state_only: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        padding_mask = batch_data["padding_mask"]
        n_graph, n_node = padding_mask.size()[:2]

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x, x_0 = self.graph_node_feature(batch_data["node_data"],
                                             batch_data["face_area"],
                                             batch_data["face_type"],
                                             batch_data["face_loop"],
                                             batch_data["in_degree"],
                                             batch_data["centroid"],
                                             batch_data["curvature"],
                                             batch_data["inner_loops"],
                                             batch_data["adj_stats"],
                                             batch_data["rational"],
                                             batch_data["EigVecs"],
                                             batch_data["EigVals"],
                                             batch_data["padding_mask"])

        # ======================= BrepEncoder 调试点 1: 检查初始节点特征 =======================
        if not torch.all(torch.isfinite(x)):
            print("\n--- BrepEncoder 调试 (1): 初始节点特征 'x' 包含无效值 ---")
        # =================================================================================

        if perturb is not None:
            x[:, 1:, :] += perturb

        attn_bias = self.graph_attn_bias(batch_data["attn_bias"],
                                         batch_data["spatial_pos"],
                                         batch_data["d2_distance"],
                                         batch_data["angle_distance"],
                                         batch_data["centroid_distance"],
                                         batch_data["edge_data"],
                                         batch_data["edge_type"],
                                         batch_data["edge_len"],
                                         batch_data["edge_ang"],
                                         batch_data["edge_conv"],
                                         batch_data["edge_path"],
                                         batch_data["edge_padding_mask"],
                                         batch_data["graph"],
                                         x_0
                                         )

        # ======================= BrepEncoder 调试点 2: 检查注意力偏置 =======================
        if not torch.all(torch.isfinite(attn_bias)):
            print("\n--- BrepEncoder 调试 (2): 'attn_bias' 包含无效值 ---")
        # =================================================================================

        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)
        x = x.transpose(0, 1)

        # 1. 通过共享编码模块
        for i, layer in enumerate(self.shared_layers):
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            # ======================= BrepEncoder 调试点 3: 检查共享层输出 =======================
            if not torch.all(torch.isfinite(x)):
                print(f"\n--- BrepEncoder 调试 (3): 共享层 {i + 1}/{len(self.shared_layers)} 的输出 'x' 包含无效值 ---")
                # 如果发现无效值，立即返回，防止后续代码崩溃
                nan_tensor = torch.full_like(x, float('nan'))
                return nan_tensor, nan_tensor, nan_tensor, torch.full_like(x[0, :, :], float('nan'))
            # =====================================================================================

        shared_features = x
        graph_rep = shared_features[0, :, :]

        # 3. 通过任务特定分支
        # 语义分割分支
        semantic_x = shared_features
        for i, layer in enumerate(self.semantic_layers):
            semantic_x, _ = layer(
                semantic_x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            # ======================= BrepEncoder 调试点 4: 检查语义层输出 =======================
            if not torch.all(torch.isfinite(semantic_x)):
                print(
                    f"\n--- BrepEncoder 调试 (4): 语义层 {i + 1}/{len(self.semantic_layers)} 的输出 'semantic_x' 包含无效值 ---")
                nan_tensor = torch.full_like(x, float('nan'))
                return nan_tensor, nan_tensor, nan_tensor, torch.full_like(x[0, :, :], float('nan'))
            # =====================================================================================

        semantic_features = self.tanh(semantic_x)

        # 实例分割分支
        instance_x = shared_features
        for i, layer in enumerate(self.instance_layers):
            instance_x, _ = layer(
                instance_x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            # ======================= BrepEncoder 调试点 5: 检查实例层输出 =======================
            if not torch.all(torch.isfinite(instance_x)):
                print(
                    f"\n--- BrepEncoder 调试 (5): 实例层 {i + 1}/{len(self.instance_layers)} 的输出 'instance_x' 包含无效值 ---")
                nan_tensor = torch.full_like(x, float('nan'))
                return nan_tensor, nan_tensor, nan_tensor, torch.full_like(x[0, :, :], float('nan'))
            # =====================================================================================

        instance_features = self.tanh(instance_x)

        shared_features = shared_features.transpose(0, 1)
        semantic_features = semantic_features.transpose(0, 1)
        instance_features = instance_features.transpose(0, 1)

        return shared_features, semantic_features, instance_features, graph_rep