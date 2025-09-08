# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    注意力融合模块，用于动态地、自适应地融合共享特征和任务特定特征。

    该网络会为每个任务（例如语义、实例）分别学习一组注意力权重，
    然后使用这些权重对共享特征和任务特定特征进行加权求和，从而得到最终的融合特征。

    融合公式:
    fused_feature = weight_shared * shared_feature + weight_task * task_specific_feature
    其中，权重是通过一个小型网络从拼接的特征中预测得来。
    """

    def __init__(self, embedding_dim: int):
        """
        初始化注意力融合模块。

        Args:
            embedding_dim (int): 输入特征的维度。
                               共享特征和所有任务特定特征都应具有此维度。
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # 用于预测权重的网络的输入是拼接后的共享特征和任务特定特征
        fusion_input_dim = embedding_dim * 2

        # 为“语义分割”任务定义的注意力权重预测网络
        # 线性层输出2个值，分别对应共享特征和语义特征的重要性得分
        self.semantic_attention_net = nn.Linear(fusion_input_dim, 2)

        # 为“实例分割”任务定义的注意力权重预测网络
        self.instance_attention_net = nn.Linear(fusion_input_dim, 2)

    def forward(self, shared_features: torch.Tensor,
                semantic_features: torch.Tensor,
                instance_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行前向传播，通过注意力机制融合特征。

        Args:
            shared_features (torch.Tensor): 来自共享编码器的特征张量。
                                            Shape: (..., embedding_dim)
            semantic_features (torch.Tensor): 来自语义分割特定编码器的特征张量。
                                              Shape: (..., embedding_dim)
            instance_features (torch.Tensor): 来自实例分割特定编码器的特征张量。
                                              Shape: (..., embedding_dim)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - fused_semantic_features (torch.Tensor): 融合后的语义特征。
                - fused_instance_features (torch.Tensor): 融合后的实例特征。
        """
        # --- 语义任务特征融合 ---
        # 1. 拼接共享特征和语义特定特征
        semantic_concat = torch.cat((shared_features, semantic_features), dim=-1)
        # 2. 预测注意力得分
        semantic_scores = self.semantic_attention_net(semantic_concat)
        # 3. 通过Softmax将得分转换为权重
        semantic_weights = F.softmax(semantic_scores, dim=-1)
        # 4. 进行加权求和
        #    为了高效计算，我们将权重变形以利用广播机制
        fused_semantic_features = (semantic_weights[..., 0].unsqueeze(-1) * shared_features +
                                   semantic_weights[..., 1].unsqueeze(-1) * semantic_features)

        # --- 实例任务特征融合 ---
        # 1. 拼接共享特征和实例特定特征
        instance_concat = torch.cat((shared_features, instance_features), dim=-1)
        # 2. 预测注意力得分
        instance_scores = self.instance_attention_net(instance_concat)
        # 3. 通过Softmax将得分转换为权重
        instance_weights = F.softmax(instance_scores, dim=-1)
        # 4. 进行加权求和
        fused_instance_features = (instance_weights[..., 0].unsqueeze(-1) * shared_features +
                                   instance_weights[..., 1].unsqueeze(-1) * instance_features)

        return fused_semantic_features, fused_instance_features