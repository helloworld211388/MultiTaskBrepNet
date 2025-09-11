# -*- coding: utf-8 -*-
import torch
import dgl
import sys

sys.path.append('..')
from models.modules.utils.macro import *


# --- 填充函数未做修改，保持原样 ---
def pad_mask_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_float_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_face_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_d2_pos_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 64], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_ang_pos_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 64], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)



def collator(items, multi_hop_max_dist, spatial_pos_max):
    # ======================= 在此插入最终的调试代码块 (开始) =======================
    # 在函数最开始，对原始的 item 对象列表进行图结构完整性检查
    try:
        for item in items:
            graph = item.graph
            data_id = item.data_id

            num_nodes = graph.num_nodes()
            if num_nodes == 0:
                continue  # 跳过空图

            src, dst = graph.edges()

            # 确保图中存在边才进行检查
            if len(src) > 0:
                max_src_id = torch.max(src)
                assert max_src_id < num_nodes, \
                    f"文件 ID {data_id} 对应的图中发现悬空边! " \
                    f"图的总节点数为 {num_nodes} (有效ID: 0 至 {num_nodes - 1}), " \
                    f"但边的源节点中发现了最大ID: {max_src_id}."

            if len(dst) > 0:
                max_dst_id = torch.max(dst)
                assert max_dst_id < num_nodes, \
                    f"文件 ID {data_id} 对应的图中发现悬空边! " \
                    f"图的总节点数为 {num_nodes} (有效ID: 0 至 {num_nodes - 1}), " \
                    f"但边的目标节点中发现了最大ID: {max_dst_id}."

    except AssertionError as e:
        print(f"\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!   DGL 图结构校验失败 (在Collator顶层捕获)   !!!")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print(f"错误详情 (Error Detail): {e}")
        print(f"\n已定位到问题的根源。请检查数据生成脚本 (multi_StepToBin.py) 中构建邻接图的逻辑。")
        raise e
    # ======================= 在此插入最终的调试代码块 (结束) =======================
    # ======================================================================
    # 【核心修正】将每个图的局部 edge_path 索引转换为批次内的全局索引
    # ======================================================================
    edge_offset = 0
    # 遍历每个数据项，但这次我们只为了计算偏移量和准备新数据
    for item in items:
        # 关键：创建 edge_path 的一个副本进行修改，而不是直接修改原始数据
        new_edge_path = item.edge_path.clone()

        valid_paths = new_edge_path > 0
        if valid_paths.any():  # 仅在存在有效路径时才增加偏移
            new_edge_path[valid_paths] += edge_offset

        # 将修改后的副本替换掉原始的张量
        item.edge_path = new_edge_path

        # 更新偏移量
        edge_offset += item.graph.num_edges()
    # ======================================================================

    items = [
        (
            item.graph,
            item.node_data,
            item.face_area,
            item.face_type,
            item.face_loop,
            item.face_adj,
            item.centroid,
            item.curvature,
            item.inner_loops,
            item.adj_stats,
            item.edge_data,
            item.edge_type,
            item.edge_len,
            item.edge_ang,
            item.edge_conv,
            item.node_degree,
            item.attn_bias,
            item.spatial_pos,
            item.d2_distance,
            item.angle_distance,
            item.centroid_distance,
            item.edge_path[:, :, :multi_hop_max_dist],
            item.label_feature,
            item.instance_pos_edge_index,
            item.instance_label,
            item.rational,
            item.data_id,
            item.EigVecs, # 新增 EigVecs
            item.EigVals, # 新增 EigVals
        )
        for item in items
    ]

    (
        graphs,
        node_datas,
        face_areas,
        face_types,
        face_loops,
        face_adjs,
        centroids,
        curvatures,
        inner_loops_list,
        adj_stats_list,
        edge_datas,
        edge_types,
        edge_lens,
        edge_angs,
        edge_convs,
        node_degrees,
        attn_biases,
        spatial_poses,
        d2_distances,
        angle_distances,
        centroid_distances,
        edge_paths,
        label_features,
        instance_pos_edge_indices,
        instance_labels,
        rationals,
        data_ids,
        # ======================= 修改点 2 (开始) =======================
        EigVecs_list, # 新增 EigVecs 列表
        EigVals_list, # 新增 EigVals 列表
        # ======================= 修改点 2 (结束) =======================
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in node_datas)
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)
    max_dist = max(max_dist, multi_hop_max_dist)

    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])

    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])

    node_data = torch.cat([i for i in node_datas])
    face_area = torch.cat([i for i in face_areas])
    face_type = torch.cat([i for i in face_types])
    face_loop = torch.cat([i for i in face_loops])
    face_adj = torch.cat([i for i in face_adjs])
    centroid = torch.cat([i for i in centroids])
    curvature = torch.cat([i for i in curvatures])
    inner_loops = torch.cat([i for i in inner_loops_list])
    adj_stats = torch.cat([i for i in adj_stats_list])
    edge_data = torch.cat([i for i in edge_datas])
    edge_type = torch.cat([i for i in edge_types])
    edge_len = torch.cat([i for i in edge_lens])
    edge_ang = torch.cat([i for i in edge_angs])
    edge_conv = torch.cat([i for i in edge_convs])

    edge_path = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    d2_distance = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distances]
    )
    angle_distance = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distances]
    )
    centroid_distance = torch.cat(
        [pad_2d_float_unsqueeze(i, max_node_num) for i in centroid_distances]
    )

    in_degree = torch.cat([i for i in node_degrees])

    batched_instance_pos_edge_index_list = []
    node_offset = 0
    for i, pos_edge_index in enumerate(instance_pos_edge_indices):
        if pos_edge_index.numel() > 0:
            batched_instance_pos_edge_index_list.append(pos_edge_index + node_offset)
        node_offset += node_datas[i].size(0)

    if not batched_instance_pos_edge_index_list:
        batched_instance_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        batched_instance_pos_edge_index = torch.cat(batched_instance_pos_edge_index_list, dim=1)

    batched_graph = dgl.batch([i for i in graphs])
    batched_label_feature = torch.cat([i for i in label_features])
    batched_instance_label = torch.cat([i for i in instance_labels])
    rational = torch.cat([i for i in rationals])
    data_ids = torch.tensor([i for i in data_ids])

    # ======================= 修改点 3 (开始) =======================
    # 拼接 EigVecs 和 EigVals
    batched_EigVecs = torch.cat([i for i in EigVecs_list])
    # 将 EigVals 堆叠，而不是拼接，以保留批次维度
    batched_EigVals = torch.stack(EigVals_list, dim=0)  # 旧: torch.cat
    # ======================= 修改点 3 (结束) =======================


    batch_data = dict(
        padding_mask=padding_mask,
        edge_padding_mask=edge_padding_mask,
        graph=batched_graph,
        node_data=node_data,
        face_area=face_area,
        face_type=face_type,
        face_loop=face_loop,
        face_adj=face_adj,
        centroid=centroid,
        curvature=curvature,
        inner_loops=inner_loops,
        adj_stats=adj_stats,
        edge_data=edge_data,
        edge_type=edge_type,
        edge_len=edge_len,
        edge_ang=edge_ang,
        edge_conv=edge_conv,
        in_degree=in_degree,
        out_degree=in_degree,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        d2_distance=d2_distance,
        angle_distance=angle_distance,
        centroid_distance=centroid_distance,
        edge_path=edge_path,
        label_feature=batched_label_feature,
        instance_label=batched_instance_label,
        instance_pos_edge_index=batched_instance_pos_edge_index,
        rational=rational,
        id=data_ids,
        # ======================= 修改点 4 (开始) =======================
        EigVecs=batched_EigVecs, # 将拼接好的张量添加到字典中
        EigVals=batched_EigVals, # 将拼接好的张量添加到字典中
        # ======================= 修改点 4 (结束) =======================
    )
    return batch_data

def collator_st(items, multi_hop_max_dist, spatial_pos_max):
    items_source_data = [item["source_data"] for item in items]
    items_target_data = [item["target_data"] for item in items]
    all_items_data = items_source_data + items_target_data

    # ======================================================================
    # 【核心修正】对源域和目标域的数据统一进行索引转换
    # ======================================================================
    edge_offset = 0
    for item_data in all_items_data:
        # 关键：创建 edge_path 的一个副本进行修改
        new_edge_path = item_data.edge_path.clone()

        valid_paths = new_edge_path > 0
        if valid_paths.any():
            new_edge_path[valid_paths] += edge_offset

        # 将修改后的副本替换掉原始的张量
        item_data.edge_path = new_edge_path

        edge_offset += item_data.graph.num_edges()
    # ======================================================================

    # 从已修改的 item_data 对象中重新构建 items 列表以进行解包
    items_processed = []
    for item_data in all_items_data:
        items_processed.append((
            item_data.graph, item_data.node_data, item_data.face_area,
            item_data.face_type, item_data.face_loop, item_data.face_adj,
            item_data.centroid, item_data.curvature, item_data.inner_loops,
            item_data.adj_stats, item_data.edge_data, item_data.edge_type,
            item_data.edge_len, item_data.edge_ang, item_data.edge_conv,
            item_data.in_degree, item_data.attn_bias, item_data.spatial_pos,
            item_data.d2_distance, item_data.angle_distance,
            item_data.centroid_distance, item_data.edge_path[:, :, :multi_hop_max_dist],
            item_data.label_feature, item_data.instance_label, item_data.data_id
        ))

    (
        graphs, node_datas, face_areas, face_types, face_loops, face_adjs,
        centroids, curvatures, inner_loops_list, adj_stats_list, edge_datas,
        edge_types, edge_lens, edge_angs, edge_convs, in_degrees,
        attn_biases, spatial_poses, d2_distancees, angle_distancees,
        centroid_distances, edge_paths, label_features, instance_labels, data_ids
    ) = zip(*items_processed)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    batched_graph = dgl.batch([i for i in graphs])

    max_node_num = max(i.size(0) for i in node_datas)
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)
    max_dist = max(max_dist, multi_hop_max_dist)

    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])

    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])

    node_data = torch.cat([i for i in node_datas])
    face_area = torch.cat([i for i in face_areas])
    face_type = torch.cat([i for i in face_types])
    face_loop = torch.cat([i for i in face_loops])
    face_adj = torch.cat([i for i in face_adjs])
    centroid = torch.cat([i for i in centroids])
    curvature = torch.cat([i for i in curvatures])
    inner_loops = torch.cat([i for i in inner_loops_list])
    adj_stats = torch.cat([i for i in adj_stats_list])
    edge_data = torch.cat([i for i in edge_datas])
    edge_type = torch.cat([i for i in edge_types])
    edge_len = torch.cat([i for i in edge_lens])
    edge_ang = torch.cat([i for i in edge_angs])
    edge_conv = torch.cat([i for i in edge_convs])

    edge_path = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    d2_distance = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distancees]
    )
    angle_distance = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distancees]
    )

    centroid_distance = torch.cat(
        [pad_2d_float_unsqueeze(i, max_node_num) for i in centroid_distances]
    )

    in_degree = torch.cat([i for i in in_degrees])
    batched_label_feature = torch.cat([i for i in label_features])
    batched_instance_label = torch.cat([i for i in instance_labels])
    data_ids = torch.tensor([i for i in data_ids])

    batch_data = dict(
        padding_mask=padding_mask,
        edge_padding_mask=edge_padding_mask,
        graph=batched_graph,
        node_data=node_data,
        face_area=face_area,
        face_type=face_type,
        face_loop=face_loop,
        face_adj=face_adj,
        centroid=centroid,
        curvature=curvature,
        inner_loops=inner_loops,
        adj_stats=adj_stats,
        edge_data=edge_data,
        edge_type=edge_type,
        edge_len=edge_len,
        edge_ang=edge_ang,
        edge_conv=edge_conv,
        in_degree=in_degree,
        out_degree=in_degree,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        d2_distance=d2_distance,
        angle_distance=angle_distance,
        centroid_distance=centroid_distance,
        edge_path=edge_path,
        label_feature=batched_label_feature,
        instance_label=batched_instance_label,
        id=data_ids
    )
    return batch_data