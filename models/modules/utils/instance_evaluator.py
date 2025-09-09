import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple

# AAGNet中的常量，用于评估
EPS = 1e-6


class FeatureInstance:
    """
    一个简单的数据结构，用于存储一个特征实例的信息。
    """

    def __init__(self, name: int = None, faces: np.ndarray = None):
        self.name = name
        self.faces = faces if faces is not None else np.array([])


def post_process_instances(
        instance_embeddings: torch.Tensor,
        semantic_logits: torch.Tensor,
        similarity_threshold: float = 0.5,
        data_id: int = -1
) -> List[FeatureInstance]:
    """
    使用基于相似度阈值的方法对实例进行后处理。
    """
    if instance_embeddings.numel() == 0:
        return []

    # 归一化嵌入向量
    instance_embeddings = torch.nn.functional.normalize(instance_embeddings, p=2, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(instance_embeddings, instance_embeddings.T)

    # 基于相似度阈值构建图
    num_faces = instance_embeddings.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_faces))

    adj_matrix = (similarity_matrix > similarity_threshold).cpu().numpy()
    rows, cols = np.where(adj_matrix)
    edges = zip(rows.tolist(), cols.tolist())
    graph.add_edges_from(edges)

    # 寻找连通分量作为实例分组
    clustered_instances = list(nx.connected_components(graph))

    semantic_preds_np = torch.argmax(semantic_logits, dim=-1).cpu().numpy()

    predicted_features = []
    for instance_faces_set in clustered_instances:
        instance_faces = np.array(list(instance_faces_set))
        valid_semantic_preds = semantic_preds_np[instance_faces]

        # 过滤掉基面等非特征类别
        valid_semantic_preds = valid_semantic_preds[valid_semantic_preds < 24]

        if len(valid_semantic_preds) == 0:
            continue

        # 通过多数投票决定实例的类别
        instance_class = np.bincount(valid_semantic_preds).argmax()

        predicted_features.append(FeatureInstance(name=instance_class, faces=instance_faces))

    return predicted_features


def parse_ground_truth(
        instance_labels: torch.Tensor,
        semantic_labels: torch.Tensor
) -> List[FeatureInstance]:
    """
    从真实标签中解析出基准特征实例。
    """
    gt_instances = []
    instance_labels_np = instance_labels.cpu().numpy()
    semantic_labels_np = semantic_labels.cpu().numpy()

    unique_instance_ids = np.unique(instance_labels_np[instance_labels_np > 0])

    for inst_id in unique_instance_ids:
        face_indices = np.where(instance_labels_np == inst_id)[0]

        if len(face_indices) > 0:
            instance_class = semantic_labels_np[face_indices[0]]
            if instance_class < 24:
                gt_instances.append(FeatureInstance(name=instance_class, faces=face_indices))

    return gt_instances


def cal_recognition_performance(
        feature_list: List[FeatureInstance],
        label_list: List[FeatureInstance]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算识别性能。"""
    pred = np.zeros(24, dtype=int)
    gt = np.zeros(24, dtype=int)
    for feature in feature_list:
        if feature.name < 24:
            pred[feature.name] += 1
    for label in label_list:
        if label.name < 24:
            gt[label.name] += 1
    tp = np.minimum(gt, pred)
    return pred, gt, tp


def cal_localization_performance(
        feature_list: List[FeatureInstance],
        label_list: List[FeatureInstance]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算定位性能。"""
    pred = np.zeros(24, dtype=int)
    gt = np.zeros(24, dtype=int)
    for feature in feature_list:
        if feature.name < 24:
            pred[feature.name] += 1
    for label in label_list:
        if label.name < 24:
            gt[label.name] += 1

    tp = np.zeros(24, dtype=int)
    found_lbl = np.zeros(len(label_list), dtype=bool)

    for pred_instance in feature_list:
        if pred_instance.name >= 24:
            continue

        for lbl_i, lbl_instance in enumerate(label_list):
            if lbl_instance.name != pred_instance.name or found_lbl[lbl_i]:
                continue

            pred_faces = pred_instance.faces
            lbl_faces = lbl_instance.faces
            intersection = np.intersect1d(pred_faces, lbl_faces, assume_unique=True)
            union = np.union1d(pred_faces, lbl_faces)

            iou = len(intersection) / len(union) if len(union) > 0 else 0.0

            if iou >= 1.0 - EPS:
                found_lbl[lbl_i] = True
                tp[pred_instance.name] += 1
                break

    return pred, gt, tp


def eval_metric(
        pre: np.ndarray,
        trul: np.ndarray,
        tp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """计算精确率和召回率。"""
    precision = tp / (pre + EPS)
    recall = tp / (trul + EPS)
    precision[trul == 0] = 1.0
    recall[trul == 0] = 1.0
    return precision, recall