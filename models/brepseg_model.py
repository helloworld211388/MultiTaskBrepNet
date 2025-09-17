# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import networkx as nx
from .modules.brep_encoder import BrepEncoder
from .modules.attention_fusion import AttentionFusion
from torch_geometric.utils import negative_sampling
# 确保导入了评估器模块
from .modules.utils import instance_evaluator as evaluator
from .modules.utils.macro import *
from torch.cuda.amp import autocast
# 导入 torchmetrics 的评估函数
from torchmetrics.functional.classification import binary_f1_score, binary_average_precision


class MultiFocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, preds, labels):
        pt = F.softmax(preds, dim=1)
        class_mask = F.one_hot(labels, self.class_num)
        ids = labels.view(-1, 1)
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = torch.log(probs + 1e-7)
        loss = -torch.pow((1 - probs), self.gamma) * log_p
        return loss.mean()


class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=dropout)
        self.linear4 = nn.Linear(256, num_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = self.dense_weight(stacked)
        weights = F.softmax(weights, dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class MultiTaskBrepNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.num_semantic_classes = args.num_classes
        self.num_instance_classes = getattr(args, 'num_instance_classes', args.num_classes)
        self.ortho_loss_weight = 0.1
        self.semantic_loss_weight = getattr(args, 'semantic_loss_weight', 0.5)

        self.lr = getattr(args, 'lr', 5e-5)
        self.warmup_steps = getattr(args, 'warmup_steps', 5000)
        self.total_steps = getattr(args, 'total_steps', 100000)
        self.temperature = getattr(args, 'temperature', 0.1)

        self.brep_encoder = BrepEncoder(
            num_degree=512, num_spatial=64, num_edge_dis=64,
            edge_type="multi_hop", multi_hop_max_dist=16,#在这里启用了 multi-hop 边，如果是这个值，那么就不会有边偏置
            num_shared_layers=getattr(args, 'num_shared_layers', 4),
            num_semantic_layers=getattr(args, 'num_semantic_layers', 3),
            num_instance_layers=getattr(args, 'num_instance_layers', 4),
            embedding_dim=args.dim_node, ffn_embedding_dim=args.d_model,
            num_attention_heads=args.n_heads, dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout, layerdrop=0.1,
            encoder_normalize_before=True, pre_layernorm=True,
            apply_params_init=True, activation_fn="mish",
        )

        self.attention_fusion = AttentionFusion(embedding_dim=args.dim_node)
        self.semantic_attention = Attention(args.dim_node)
        self.semantic_classifier = NonLinearClassifier(args.dim_node, self.num_semantic_classes, args.dropout)
        self.instance_attention = Attention(args.dim_node)
        combined_input_dim = args.dim_node * 2
        self.instance_embedding_dim = args.dim_node
        self.instance_embedding_projector = nn.Sequential(
            nn.Linear(combined_input_dim, self.instance_embedding_dim),
            nn.Mish(),
            LayerNorm(self.instance_embedding_dim)
        )

        self.semantic_criterion = MultiFocalLoss(class_num=self.num_semantic_classes)

        self.semantic_preds, self.semantic_labels = [], []
        # --- 恢复：为两种实例评估方法创建存储列表 ---
        self.instance_eval_data = [] # 用于链接预测评估
        self.instance_post_process_data = [] # 用于 rec/loc F1 评估


    def _compute_multi_positive_loss(self, embeddings, pos_edge_index):

        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        unique_anchors_indices, inverse_indices = torch.unique(pos_edge_index[0], return_inverse=True)
        anchor_embeddings = normalized_embeddings[unique_anchors_indices]

        all_embeddings = normalized_embeddings
        sim_matrix = torch.matmul(anchor_embeddings, all_embeddings.t()) / self.temperature

        # 2. 检查 exp 指数操作后的结果
        # 这是最容易发生数值溢出的地方
        exp_sim_matrix = torch.exp(sim_matrix)

        # 3. 检查分子和分母
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask[inverse_indices, pos_edge_index[1]] = True

        denominator = exp_sim_matrix.sum(dim=1)
        numerator = (exp_sim_matrix * pos_mask).sum(dim=1)


        # 4. 检查最终的损失值
        # 在log前检查 ratio 是否接近0
        ratio = numerator / (denominator + 1e-9)  # 增加一个小的epsilon防止除以0

        loss_per_anchor = -torch.log(ratio)

        return loss_per_anchor.mean()

    def forward(self, batch):
        shared_features, semantic_features, instance_features, graph_rep = self.brep_encoder(batch)
        shared_nodes = shared_features[:, 1:, :]
        semantic_nodes = semantic_features[:, 1:, :]
        instance_nodes = instance_features[:, 1:, :]
        fused_semantic_nodes, fused_instance_nodes = self.attention_fusion(shared_nodes, semantic_nodes, instance_nodes)
        padding_mask = batch["padding_mask"]
        node_pos = torch.where(padding_mask == False)
        if node_pos[0].numel() == 0:
            return torch.empty(0, self.num_semantic_classes, device=batch["node_data"].device), \
                torch.empty(0, self.instance_embedding_dim, device=batch["node_data"].device), \
                None, None
        num_nodes_per_graph = torch.sum(~padding_mask, dim=-1)
        semantic_z_nodes = fused_semantic_nodes[node_pos]
        semantic_graph_rep = graph_rep.repeat_interleave(num_nodes_per_graph, dim=0)
        semantic_z = self.semantic_attention([semantic_z_nodes, semantic_graph_rep])
        semantic_logits = self.semantic_classifier(semantic_z)
        instance_z_nodes = fused_instance_nodes[node_pos]
        instance_graph_rep = graph_rep.repeat_interleave(num_nodes_per_graph, dim=0)
        instance_z = self.instance_attention([instance_z_nodes, instance_graph_rep])
        combined_z_for_instance = torch.cat([instance_z, semantic_z.detach()], dim=-1)
        instance_embeddings = self.instance_embedding_projector(combined_z_for_instance)
        return semantic_logits, instance_embeddings, semantic_nodes, instance_nodes

    def training_step(self, batch, batch_idx):
        self.train()
        semantic_logits, instance_embeddings, semantic_nodes, instance_nodes = self.forward(batch)

        if semantic_logits.numel() == 0: return None

        semantic_labels = batch["label_feature"].long()
        mask = (semantic_labels != 24) & (semantic_labels != 25) & (semantic_labels != 26)
        if not mask.any(): return None

        with autocast(enabled=False):
            loss_semantic = self.semantic_criterion(semantic_logits[mask], semantic_labels[mask])
            loss_instance = self._compute_multi_positive_loss(
                instance_embeddings, batch["instance_pos_edge_index"]
            )
            loss_ortho = 0.0
            padding_mask = batch["padding_mask"]
            valid_nodes_mask = ~padding_mask
            if valid_nodes_mask.any():
                semantic_features_valid = semantic_nodes[valid_nodes_mask]
                instance_features_valid = instance_nodes[valid_nodes_mask]
                semantic_features_norm = F.normalize(semantic_features_valid, p=2, dim=1)
                instance_features_norm = F.normalize(instance_features_valid, p=2, dim=1)
                cos_sim = (semantic_features_norm * instance_features_norm).sum(dim=1)
                loss_ortho = torch.mean(cos_sim ** 2)

            loss_semantic_fp32 = loss_semantic.float()
            loss_instance_fp32 = loss_instance.float()

            main_loss = self.semantic_loss_weight * loss_semantic_fp32 + (
                        1 - self.semantic_loss_weight) * loss_instance_fp32
            total_loss = main_loss + self.ortho_loss_weight * loss_ortho



        batch_size = batch["padding_mask"].shape[0]
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_loss_semantic", loss_semantic, on_step=False, on_epoch=True)
        self.log("train_loss_instance", loss_instance, on_step=False, on_epoch=True)
        self.log("train_loss_ortho", loss_ortho, on_step=False, on_epoch=True)

        # 只有当损失值是有效的时候才返回，防止 NaN 损失污染 Lightning 的内部状态
        if torch.isfinite(total_loss):
            return total_loss
        else:
            print(f"\n[!!!] 在批次 {batch_idx} 检测到无效的总损失值 (NaN/Inf)，跳过此批次的更新。")
            return None

    def validation_step(self, batch, batch_idx):
        self.eval()
        semantic_logits, instance_embeddings, semantic_nodes, instance_nodes = self.forward(batch)
        if semantic_logits.numel() == 0:
            self.log("eval_loss", 0.0)
            return None

        semantic_labels = batch["label_feature"].long()
        mask = (semantic_labels != 24) & (semantic_labels != 25) & (semantic_labels != 26)
        if not mask.any(): return None

        valid_logits = semantic_logits[mask]
        valid_labels = semantic_labels[mask]

        self.semantic_preds.append(torch.argmax(valid_logits, dim=-1).cpu())
        self.semantic_labels.append(valid_labels.cpu())

        # --- 数据收集 Section 1: 用于链接预测评估 ---
        num_nodes_total = instance_embeddings.size(0)
        pos_edge_index = batch["instance_pos_edge_index"]
        self.instance_eval_data.append({
            'embeddings': instance_embeddings.cpu(),
            'pos_edge_index': pos_edge_index.cpu(),
            'num_nodes': num_nodes_total
        })

        # --- 恢复：数据收集 Section 2: 用于 rec/loc F1 评估 ---
        num_nodes_per_graph = batch['graph'].batch_num_nodes().cpu().tolist()
        split_instance_embeddings = torch.split(instance_embeddings, num_nodes_per_graph, dim=0)
        split_semantic_logits = torch.split(semantic_logits, num_nodes_per_graph, dim=0)
        split_instance_labels_gt = torch.split(batch["instance_label"], num_nodes_per_graph, dim=0)
        split_semantic_labels_gt = torch.split(batch["label_feature"], num_nodes_per_graph, dim=0)
        split_ids = torch.split(batch["id"], 1)
        for i in range(len(num_nodes_per_graph)):
            self.instance_post_process_data.append({
                'embeddings': split_instance_embeddings[i].cpu(),
                'logits': split_semantic_logits[i].cpu(),
                'instance_gt': split_instance_labels_gt[i].cpu(),
                'semantic_gt': split_semantic_labels_gt[i].cpu(),
                'id': split_ids[i].cpu().item(),
            })

        # --- 损失计算 ---
        with autocast(enabled=False):
            loss_semantic = self.semantic_criterion(valid_logits, valid_labels)
            loss_instance = self._compute_multi_positive_loss(
                instance_embeddings, batch["instance_pos_edge_index"]
            )
            loss_ortho = 0.0
            padding_mask = batch["padding_mask"]
            valid_nodes_mask = ~padding_mask
            if valid_nodes_mask.any():
                semantic_features_valid = semantic_nodes[valid_nodes_mask]
                instance_features_valid = instance_nodes[valid_nodes_mask]
                semantic_features_norm = F.normalize(semantic_features_valid, p=2, dim=1)
                instance_features_norm = F.normalize(instance_features_valid, p=2, dim=1)
                cos_sim = (semantic_features_norm * instance_features_norm).sum(dim=1)
                loss_ortho = torch.mean(cos_sim ** 2)

            loss_semantic_fp32 = loss_semantic.float()
            loss_instance_fp32 = loss_instance.float()
            main_loss = self.semantic_loss_weight * loss_semantic_fp32 + (1 - self.semantic_loss_weight) * loss_instance_fp32
            total_loss = main_loss + self.ortho_loss_weight * loss_ortho

        self.log("eval_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_epoch_end(self, val_step_outputs):
        if not val_step_outputs:
            return

        # --- 语义分割指标 ---
        if self.semantic_preds:
            preds_np = torch.cat(self.semantic_preds).numpy()
            labels_np = torch.cat(self.semantic_labels).numpy()
            self.semantic_preds.clear()
            self.semantic_labels.clear()
            semantic_acc = np.mean((preds_np == labels_np).astype(int))
            self.log("semantic_accuracy", semantic_acc, prog_bar=True)

        # --- 实例评估 Section 1: 链接预测 F1/AP ---
        if self.instance_eval_data:
            all_f1_scores, all_ap_scores = [], []
            for data in self.instance_eval_data:
                embeddings, pos_edge_index, num_nodes = data['embeddings'], data['pos_edge_index'], data['num_nodes']
                if pos_edge_index.numel() == 0: continue

                num_neg_samples = pos_edge_index.size(1)
                neg_edge_index = negative_sampling(pos_edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
                pos_logits = (embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]]).sum(dim=-1)
                neg_logits = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=-1)
                logits = torch.cat([pos_logits, neg_logits], dim=0)
                preds_prob = torch.sigmoid(logits)
                labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0).long()
                all_f1_scores.append(binary_f1_score(preds_prob, labels).item())
                all_ap_scores.append(binary_average_precision(preds_prob, labels).item())
            
            avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
            avg_ap = np.mean(all_ap_scores) if all_ap_scores else 0.0
            self.log("instance_f1_score", avg_f1, prog_bar=True)
            self.log("instance_ap_score", avg_ap, prog_bar=True)
            self.instance_eval_data.clear()

        # --- 恢复：实例评估 Section 2: rec/loc F1 ---
        if not self.instance_post_process_data:
            self.log_dict({"rec_f1": 0.0, "loc_f1": 0.0})
        else:
            rec_predictions, rec_truelabels, rec_truepositives = np.zeros(24, dtype=int), np.zeros(24, dtype=int), np.zeros(24, dtype=int)
            loc_predictions, loc_truelabels, loc_truepositives = np.zeros(24, dtype=int), np.zeros(24, dtype=int), np.zeros(24, dtype=int)
            for data in self.instance_post_process_data:
                predicted_features = evaluator.post_process_instances(instance_embeddings=data['embeddings'], semantic_logits=data['logits'], data_id=data.get('id', -1),)
                gt_features = evaluator.parse_ground_truth(instance_labels=data['instance_gt'], semantic_labels=data['semantic_gt'])
                pred, gt, tp = evaluator.cal_recognition_performance(predicted_features, gt_features)
                rec_predictions += pred
                rec_truelabels += gt
                rec_truepositives += tp
                pred, gt, tp = evaluator.cal_localization_performance(predicted_features, gt_features)
                loc_predictions += pred
                loc_truelabels += gt
                loc_truepositives += tp
            rec_precision, rec_recall = evaluator.eval_metric(rec_predictions, rec_truelabels, rec_truepositives)
            rec_f1 = (2 * rec_precision.mean() * rec_recall.mean()) / (rec_precision.mean() + rec_recall.mean() + evaluator.EPS)
            loc_precision, loc_recall = evaluator.eval_metric(loc_predictions, loc_truelabels, loc_truepositives)
            loc_f1 = (2 * loc_precision.mean() * loc_recall.mean()) / (loc_precision.mean() + loc_recall.mean() + evaluator.EPS)
            self.log_dict({"rec_f1": rec_f1, "loc_f1": loc_f1}, prog_bar=True)
            self.instance_post_process_data.clear()

    def test_step(self, batch, batch_idx):
        self.eval()
        semantic_logits, instance_embeddings, _, _ = self.forward(batch)
        if semantic_logits.numel() == 0: return

        semantic_labels = batch["label_feature"].long()
        mask = (semantic_labels != 24) & (semantic_labels != 25) & (semantic_labels != 26)
        if not mask.any(): return

        valid_logits = semantic_logits[mask]
        valid_labels = semantic_labels[mask]
        self.semantic_preds.append(torch.argmax(valid_logits, dim=-1).cpu())
        self.semantic_labels.append(valid_labels.cpu())

        # --- 数据收集 Section 1: 用于链接预测评估 ---
        num_nodes_total = instance_embeddings.size(0)
        pos_edge_index = batch["instance_pos_edge_index"]
        self.instance_eval_data.append({
            'embeddings': instance_embeddings.cpu(),
            'pos_edge_index': pos_edge_index.cpu(),
            'num_nodes': num_nodes_total
        })

        # # --- 恢复：数据收集 Section 2: 用于 rec/loc F1 评估 ---
        # num_nodes_per_graph = batch['graph'].batch_num_nodes().cpu().tolist()
        # split_instance_embeddings = torch.split(instance_embeddings, num_nodes_per_graph, dim=0)
        # split_semantic_logits = torch.split(semantic_logits, num_nodes_per_graph, dim=0)
        # split_instance_labels_gt = torch.split(batch["instance_label"], num_nodes_per_graph, dim=0)
        # split_semantic_labels_gt = torch.split(batch["label_feature"], num_nodes_per_graph, dim=0)
        # split_ids = torch.split(batch["id"], 1)
        # for i in range(len(num_nodes_per_graph)):
        #     self.instance_post_process_data.append({
        #         'embeddings': split_instance_embeddings[i].cpu(),
        #         'logits': split_semantic_logits[i].cpu(),
        #         'instance_gt': split_instance_labels_gt[i].cpu(),
        #         'semantic_gt': split_semanti王者c_labels_gt[i].cpu(),
        #         'id': split_ids[i].cpu().item(),
        #     })

    def test_epoch_end(self, outputs):
        if not self.semantic_preds: return

        # --- 语义分割指标 ---
        preds_np = torch.cat(self.semantic_preds).numpy()
        labels_np = torch.cat(self.semantic_labels).numpy()
        semantic_acc = np.mean((preds_np == labels_np).astype(int))
        self.log("test_semantic_accuracy", semantic_acc)
        print("Test Semantic Accuracy:", semantic_acc)
        self.semantic_preds.clear()
        self.semantic_labels.clear()

        # --- 实例评估 Section 1: 链接预测 F1/AP ---
        if self.instance_eval_data:
            all_f1_scores, all_ap_scores = [], []
            for data in self.instance_eval_data:
                embeddings, pos_edge_index, num_nodes = data['embeddings'], data['pos_edge_index'], data['num_nodes']
                if pos_edge_index.numel() == 0: continue

                num_neg_samples = pos_edge_index.size(1)
                neg_edge_index = negative_sampling(pos_edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
                pos_logits = (embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]]).sum(dim=-1)
                neg_logits = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=-1)
                logits = torch.cat([pos_logits, neg_logits], dim=0)
                preds_prob = torch.sigmoid(logits)
                labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0).long()
                all_f1_scores.append(binary_f1_score(preds_prob, labels).item())
                all_ap_scores.append(binary_average_precision(preds_prob, labels).item())
            
            avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
            avg_ap = np.mean(all_ap_scores) if all_ap_scores else 0.0
            self.log("test_instance_f1_score", avg_f1)
            print("Test Instance F1 Score:", avg_f1)
            self.log("test_instance_ap_score", avg_ap)
            print("Test Instance AP Score:", avg_ap)
            self.instance_eval_data.clear()

        # # --- 恢复：实例评估 Section 2: rec/loc F1 ---
        # if not self.instance_post_process_data:
        #     self.log_dict({"test_rec_f1": 0.0, "test_loc_f1": 0.0})
        # else:
        #     rec_predictions, rec_truelabels, rec_truepositives = np.zeros(24, dtype=int), np.zeros(24, dtype=int), np.zeros(24, dtype=int)
        #     loc_predictions, loc_truelabels, loc_truepositives = np.zeros(24, dtype=int), np.zeros(24, dtype=int), np.zeros(24, dtype=int)
        #     for data in self.instance_post_process_data:
        #         predicted_features = evaluator.post_process_instances(instance_embeddings=data['embeddings'], semantic_logits=data['logits'],data_id=data.get('id', -1),)
        #         gt_features = evaluator.parse_ground_truth(instance_labels=data['instance_gt'], semantic_labels=data['semantic_gt'])
        #         pred, gt, tp = evaluator.cal_recognition_performance(predicted_features, gt_features)
        #         rec_predictions += pred; rec_truelabels += gt; rec_truepositives += tp
        #         pred, gt, tp = evaluator.cal_localization_performance(predicted_features, gt_features)
        #         loc_predictions += pred; loc_truelabels += gt; loc_truepositives += tp
        #     rec_precision, rec_recall = evaluator.eval_metric(rec_predictions, rec_truelabels, rec_truepositives)
        #     rec_f1 = (2 * rec_precision.mean() * rec_recall.mean()) / (rec_precision.mean() + rec_recall.mean() + evaluator.EPS)
        #     loc_precision, loc_recall = evaluator.eval_metric(loc_predictions, loc_truelabels, loc_truepositives)
        #     loc_f1 = (2 * loc_precision.mean() * loc_recall.mean()) / (loc_precision.mean() + loc_recall.mean() + evaluator.EPS)
        #     self.log_dict({"test_rec_f1": rec_f1, "test_loc_f1": loc_f1})
            self.instance_post_process_data.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_steps)
        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                    milestones=[self.warmup_steps])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        }
