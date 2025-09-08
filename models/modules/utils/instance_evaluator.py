import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from sklearn.cluster import MeanShift, estimate_bandwidth
# import umap
# AAGNet中的常量，用于评估
EPS = 1e-6


class FeatureInstance:
    """
    一个简单的数据结构，用于存储一个特征实例的信息。
    直接从 AAGNet 的代码中借鉴而来。
    """

    def __init__(self, name: int = None, faces: np.ndarray = None):
        self.name = name
        self.faces = faces if faces is not None else np.array([])


def post_process_instances(
        instance_embeddings: torch.Tensor,
        semantic_logits: torch.Tensor,
        similarity_threshold: float = 0.5,
        data_id: int = -1  # 接收ID参数
) -> List[FeatureInstance]:
    # --- 调试代码：在所有操作前打印诊断信息 ---
    num_faces = instance_embeddings.shape[0]
    print(f"DEBUG: Attempting to process sample ID [{data_id}] which has [{num_faces}] faces.")
    # --- 调试代码结束 ---

    if instance_embeddings.numel() == 0:
        return []

    instance_embeddings = torch.nn.functional.normalize(instance_embeddings, p=2, dim=1)

    embeddings_np = instance_embeddings.cpu().numpy()
    semantic_preds_np = torch.argmax(semantic_logits, dim=-1).cpu().numpy()

    try:
        # --- 调试代码：在UMAP前也检查一下 ---
        if embeddings_np.shape[0] < 15:
            print(
                f"DEBUG: Low face count ({embeddings_np.shape[0]}) for sample ID [{data_id}]. This is a likely cause of failure.")

        reducer = umap.UMAP(
            n_neighbors=min(15, embeddings_np.shape[0] - 1) if embeddings_np.shape[0] > 1 else 1,
            n_components=32,
            min_dist=0.0,
            random_state=42,
        )
        embeddings_reduced_np = reducer.fit_transform(embeddings_np)

        bandwidth = estimate_bandwidth(embeddings_reduced_np, quantile=0.3)

        if bandwidth is None or bandwidth <= 1e-6:
            return []

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(embeddings_reduced_np)
        cluster_labels = ms.labels_

    except Exception as e:  # 捕获所有可能的异常
        print("\n" + "=" * 50)
        print(f"!!! DEBUG: An error occurred for sample ID [{data_id}] with [{num_faces}] faces. !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("This sample will be skipped.")
        print("=" * 50 + "\n")
        return []

    # 后续代码不变
    unique_labels = np.unique(cluster_labels)
    clustered_instances = []
    for label in unique_labels:
        if label == -1:
            continue
        face_indices = np.where(cluster_labels == label)[0]
        clustered_instances.append(face_indices)

    predicted_features = []
    for instance_faces in clustered_instances:
        valid_semantic_preds = semantic_preds_np[instance_faces]
        valid_semantic_preds = valid_semantic_preds[valid_semantic_preds < 24]

        if len(valid_semantic_preds) == 0:
            continue

        instance_class = np.bincount(valid_semantic_preds).argmax()

        predicted_features.append(FeatureInstance(name=instance_class, faces=instance_faces))

    return predicted_features


def parse_ground_truth(
        instance_labels: torch.Tensor,
        semantic_labels: torch.Tensor
) -> List[FeatureInstance]:
    """
    从真实标签中解析出基准特征实例。

    Args:
        instance_labels (torch.Tensor): 单个CAD模型的实例ID标签 (num_faces,)。
        semantic_labels (torch.Tensor): 单个CAD模型的语义类别标签 (num_faces,)。

    Returns:
        List[FeatureInstance]: 真实的特征实例列表。
    """
    gt_instances = []
    instance_labels_np = instance_labels.cpu().numpy()
    semantic_labels_np = semantic_labels.cpu().numpy()

    # 实例ID大于0的为有效特征实例
    unique_instance_ids = np.unique(instance_labels_np[instance_labels_np > 0])

    for inst_id in unique_instance_ids:
        face_indices = np.where(instance_labels_np == inst_id)[0]

        if len(face_indices) > 0:
            # 实例的类别由其第一个面的语义标签决定
            instance_class = semantic_labels_np[face_indices[0]]

            # 同样，我们只关心非基体的特征
            if instance_class < 24:
                gt_instances.append(FeatureInstance(name=instance_class, faces=face_indices))

    return gt_instances


def cal_recognition_performance(
        feature_list: List[FeatureInstance],
        label_list: List[FeatureInstance]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算识别性能 (直接从 AAGNet 移植)。"""
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
    """计算定位性能 (直接从 AAGNet 移植)。"""
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

            # 计算 IoU
            pred_faces = pred_instance.faces
            lbl_faces = lbl_instance.faces
            intersection = np.intersect1d(pred_faces, lbl_faces, assume_unique=True)
            union = np.union1d(pred_faces, lbl_faces)

            if len(union) == 0:
                iou = 0.0
            else:
                iou = len(intersection) / len(union)

            if iou >= 1.0 - EPS:
                found_lbl[lbl_i] = True
                tp[pred_instance.name] += 1
                break  # 一个预测最多只能匹配一个真实实例

    return pred, gt, tp


def eval_metric(
        pre: np.ndarray,
        trul: np.ndarray,
        tp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """计算精确率和召回率 (直接从 AAGNet 移植)。"""
    precision = tp / (pre + EPS)
    recall = tp / (trul + EPS)

    # 对于数据集中不存在的类别，其P和R没有意义，设为1
    precision[trul == 0] = 1.0
    recall[trul == 0] = 1.0

    return precision, recall
