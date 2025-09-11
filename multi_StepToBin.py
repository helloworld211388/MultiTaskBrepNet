# -*- coding: utf-8 -*-

# ==========================================================================================
# 脚本用途说明 - (V4 - MFTReNet 特征对齐修改版)
# ==========================================================================================
# 本脚本的核心功能是读取一个 STEP 格式的 3D CAD 模型文件（.stp 或 .step）及其对应的
# JSON 标签文件，然后遵循 BrepMFR 论文并结合 MFTReNet 的特征工程方法，提取模型的 B-rep 信息，
# 并将其转换为 DGL 图的格式，最终保存为一个二进制（.bin）文件。
#
# 【版本 V4 - MFTReNet 特征对齐修改】:
#   1. 增强边的U-Grid特征: 为每个边的采样点提取坐标(3D)、切线(3D)、左相邻面法向量(3D)
#      和右相邻面法向量(3D)，形成12D特征，替换了原有的7D特征。
#   2. 增加面的属性: 为每个面增加一个布尔特征，判断其是否为有理B样条曲面(Rational NURBS)。
#   3. 在图构建中不跳过基体面（标签24, 25, 26）。
#
# 【原有功能】:
#   - 批量并行处理与GPU加速。
#   - 提取多种面/边/图级别特征。
#   -
# ==========================================================================================


import os
import sys
import json
import argparse
import numpy as np
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from occwl.uvgrid import ugrid, uvgrid
from occwl.face import Face
from occwl.solid import Solid
from occwl.edge import Edge
from occwl.wire import Wire
import glob
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial

# 导入 OpenCASCADE (OCC) 核心模块
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_IN, TopAbs_OUT, TopAbs_ON
from OCC.Core.TopoDS import topods, TopoDS_Face, TopoDS_Edge, TopoDS_Shape
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties, brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle,
                              GeomAbs_Ellipse, GeomAbs_Parabola, GeomAbs_Hyperbola,
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OffsetSurface,
                              GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion)
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Pnt2d, gp_Trsf
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeFace
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepTools import breptools_OuterWire
from occwl.graph import face_adjacency

# 【MFTReNet方案一: 增强边的U-Grid特征】 导入 EdgeDataExtractor
from occwl.edge_data_extractor import EdgeDataExtractor
from scipy import sparse as sp
import torch.nn.functional as F

# --- 论文中定义的常量 ---
UV_GRID_SIZE = 5
POINT_SAMPLES_FOR_D2_A3 = 512
HISTOGRAM_BINS = 64
MAX_HOP_DISTANCE = 6
SPATIAL_POS_MAX = 32

# 【新增】为新特征定义的常量
NUM_ADJ_STATS = 18
LABELS_TO_SKIP = {24, 25, 26}
MAX_FREQS = 10  # Define max_freqs for Laplacian decomposition


def laplace_decomposition(g, max_freqs):
    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(np.array(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[
        :, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        EigVecs = F.pad(EigVecs, (0, max_freqs - n), value=float('nan'))

    # Save eigenvalues and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(
        EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

    if n < max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs - n), value=float('nan')).unsqueeze(0)
    else:
        EigVals = EigVals.unsqueeze(0)

    return EigVecs, EigVals


def normalize_shape(shape: TopoDS_Shape) -> TopoDS_Shape:
    """
    将输入的 TopoDS_Shape 对象归一化，使其边界盒适配于 [-1, 1] 的立方体中。
    """
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    if bbox.IsVoid():
        return shape
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center_x, center_y, center_z = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    max_dim = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if max_dim < 1e-6:
        return shape
    scale_factor = 2.0 / max_dim
    translation_transform = gp_Trsf()
    translation_vector = gp_Vec(-center_x, -center_y, -center_z)
    translation_transform.SetTranslation(translation_vector)
    scaling_transform = gp_Trsf()
    scaling_transform.SetScale(gp_Pnt(0, 0, 0), scale_factor)
    final_transform = scaling_transform * translation_transform
    transformer = BRepBuilderAPI_Transform(shape, final_transform, True)
    transformer.Build()
    if transformer.IsDone():
        return transformer.Shape()
    else:
        return shape


def get_surface_type_enum(face: TopoDS_Face) -> int:
    surf = BRepAdaptor_Surface(face, True)
    stype = surf.GetType()

    if stype == GeomAbs_Plane:
        return 0
    elif stype == GeomAbs_Cylinder:
        return 1
    elif stype == GeomAbs_Cone:
        return 2
    elif stype == GeomAbs_Sphere:
        return 3
    elif stype == GeomAbs_Torus:
        return 4
    elif stype == GeomAbs_SurfaceOfRevolution:
        return 5
    elif stype == GeomAbs_SurfaceOfExtrusion:
        return 6
    elif stype == GeomAbs_OffsetSurface:
        return 7
    else:
        return 8


def get_curve_type_enum(edge: TopoDS_Edge) -> int:
    curve_adaptor = BRepAdaptor_Curve(edge)
    ctype = curve_adaptor.GetType()
    type_map = {
        GeomAbs_Line: 0, GeomAbs_Circle: 1, GeomAbs_Ellipse: 2,
        GeomAbs_Parabola: 3, GeomAbs_Hyperbola: 4, GeomAbs_BezierCurve: 5,
        GeomAbs_BSplineCurve: 6,
    }
    return type_map.get(ctype, 7)


class BrepDataExtractor:
    def __init__(self, shape, full_feature_labels, device='cpu'):
        if shape.IsNull(): raise ValueError("输入的 shape 为空，无法进行处理。")
        self.device = device
        self.shape = shape
        self.solid_classifier = BRepClass3d_SolidClassifier(self.shape)
        # 1. 初始构建包含所有面的实体映射
        self._build_initial_entity_maps(shape)

        # 2. 【修改】将所有面都视为有效面，不再根据标签进行筛选
        self.valid_original_indices = list(range(len(full_feature_labels)))

        if not self.valid_original_indices:
            raise ValueError("模型中未找到任何有效面。")

        # 由于所有面都被保留，原始索引和新索引将是一致的
        self.original_to_new_face_map = {orig_idx: orig_idx for orig_idx in self.valid_original_indices}

        # 3. 【修改】基于所有面来构建核心属性
        self.faces = [self.all_faces[i] for i in self.valid_original_indices]
        self.face_map = {face: i for i, face in enumerate(self.faces)}
        self.original_surface_types = [get_surface_type_enum(face) for face in self.faces]
        self._build_adjacency_graph(shape)
        self._precompute_face_centroids()

    def _build_initial_entity_maps(self, shape_to_process):
        self.all_faces = []
        face_explorer = TopExp_Explorer(shape_to_process, TopAbs_FACE)
        while face_explorer.More():
            self.all_faces.append(topods.Face(face_explorer.Current()))
            face_explorer.Next()
        edge_explorer = TopExp_Explorer(shape_to_process, TopAbs_EDGE)
        unique_edges_dict = {}
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_hash = hash(edge)
            if edge_hash not in unique_edges_dict: unique_edges_dict[edge_hash] = edge
            edge_explorer.Next()
        self.edges = list(unique_edges_dict.values())
        self.edge_map = {edge_obj: i for i, edge_obj in enumerate(self.edges)}

    def _build_adjacency_graph(self, shape_to_process):
        solid = Solid(shape_to_process)
        occwl_graph = face_adjacency(solid)
        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(range(len(self.faces)))
        self.adj_map = {i: [] for i in range(len(self.faces))}
        valid_original_indices_set = set(self.valid_original_indices)
        for u_orig, v_orig, data in occwl_graph.edges(data=True):
            if u_orig in valid_original_indices_set and v_orig in valid_original_indices_set:
                u_new = self.original_to_new_face_map[u_orig]
                v_new = self.original_to_new_face_map[v_orig]
                shared_topo_edge = data['edge'].topods_shape()
                edge_idx = self.edge_map.get(shared_topo_edge)
                if edge_idx is not None and not self.nx_graph.has_edge(u_new, v_new):
                    self.nx_graph.add_edge(u_new, v_new, edge_idx=edge_idx)
                    if v_new not in self.adj_map[u_new]: self.adj_map[u_new].append(v_new)
                    if u_new not in self.adj_map[v_new]: self.adj_map[v_new].append(u_new)

    def _precompute_face_centroids(self):
        self.face_centroids = np.zeros((len(self.faces), 3), dtype=np.float32)
        props = GProp_GProps()
        for i, face in enumerate(self.faces):
            brepgprop_SurfaceProperties(face, props)
            centroid = props.CentreOfMass()
            self.face_centroids[i, :] = [centroid.X(), centroid.Y(), centroid.Z()]

    def process(self, feature_labels, instance_labels):
        num_nodes = len(self.faces)
        node_data_uv = np.zeros((num_nodes, UV_GRID_SIZE, UV_GRID_SIZE, 7), dtype=np.float32)
        face_types = np.zeros(num_nodes, dtype=np.int32)
        face_areas = np.zeros(num_nodes, dtype=np.float32)
        face_loops = np.zeros(num_nodes, dtype=np.int32)
        face_adjs = np.zeros(num_nodes, dtype=np.int32)
        face_centroids_data = np.zeros((num_nodes, 3), dtype=np.float32)
        face_curvatures = np.zeros(num_nodes, dtype=np.int32)
        inner_loop_props = np.zeros((num_nodes, 2), dtype=np.int32)
        # 【MFTReNet方案二】为'是否为有理NURBS'属性分配空间
        face_is_rational = np.zeros(num_nodes, dtype=np.int32)

        for i in range(num_nodes):
            (node_data_uv[i, ...], face_types[i], face_areas[i],
             face_loops[i], face_adjs[i], face_centroids_data[i, :],
             face_curvatures[i], inner_loop_props[i, :],
             face_is_rational[i]) = self._extract_face_attributes(i)  # 解包新返回值

        num_edges_in_graph = self.nx_graph.number_of_edges()
        src_nodes, dst_nodes = [], []
        edge_idx_map = {data['edge_idx']: new_idx for new_idx, (_, _, data) in
                        enumerate(self.nx_graph.edges(data=True))}
        # 【MFTReNet方案一】更新edge_data_uv的形状
        edge_data_uv = np.zeros((num_edges_in_graph, UV_GRID_SIZE, 12), dtype=np.float32)
        edge_types = np.zeros(num_edges_in_graph, dtype=np.int32)
        edge_lens = np.zeros(num_edges_in_graph, dtype=np.float32)
        edge_angs = np.zeros(num_edges_in_graph, dtype=np.float32)
        edge_convs = np.zeros(num_edges_in_graph, dtype=np.int32)

        edge_counter = 0
        for u, v, data in self.nx_graph.edges(data=True):
            edge_original_idx = data['edge_idx']
            src_nodes.extend([u, v])
            dst_nodes.extend([v, u])
            (edge_data_uv[edge_counter, ...], edge_types[edge_counter], edge_lens[edge_counter],
             edge_angs[edge_counter], edge_convs[edge_counter]) = self._extract_edge_attributes(edge_original_idx, u, v)
            edge_counter += 1

        edge_data_uv = np.repeat(edge_data_uv, 2, axis=0)
        edge_types = np.repeat(edge_types, 2, axis=0)
        edge_lens = np.repeat(edge_lens, 2, axis=0)
        edge_angs = np.repeat(edge_angs, 2, axis=0)
        edge_convs = np.repeat(edge_convs, 2, axis=0)

        adj_stats = self._extract_adj_face_stats(edge_convs)

        dgl_graph = dgl.graph((torch.tensor(src_nodes, dtype=torch.long), torch.tensor(dst_nodes, dtype=torch.long)),
                              num_nodes=num_nodes)
        dgl_graph.ndata['x'] = torch.from_numpy(node_data_uv)
        dgl_graph.ndata['z'] = torch.from_numpy(face_types)
        dgl_graph.ndata['y'] = torch.from_numpy(face_areas)
        dgl_graph.ndata['l'] = torch.from_numpy(face_loops)
        dgl_graph.ndata['a'] = torch.from_numpy(face_adjs)
        dgl_graph.ndata['f'] = torch.tensor(feature_labels, dtype=torch.int)
        dgl_graph.ndata['centroid'] = torch.from_numpy(face_centroids_data)
        dgl_graph.ndata['curvature'] = torch.from_numpy(face_curvatures)
        dgl_graph.ndata['inner_loops'] = torch.from_numpy(inner_loop_props)
        dgl_graph.ndata['adj_stats'] = torch.from_numpy(adj_stats)
        # 【MFTReNet方案二】将新属性添加到图中
        dgl_graph.ndata['rational'] = torch.from_numpy(face_is_rational)
        dgl_graph.ndata['i'] = torch.tensor(instance_labels, dtype=torch.int)
        dgl_graph.edata['x'] = torch.from_numpy(edge_data_uv)
        dgl_graph.edata['t'] = torch.from_numpy(edge_types)
        dgl_graph.edata['l'] = torch.from_numpy(edge_lens)
        dgl_graph.edata['a'] = torch.from_numpy(edge_angs)
        dgl_graph.edata['c'] = torch.from_numpy(edge_convs)

        spatial_pos, d2_dist, a3_dist, centroid_dist = self._extract_proximity_features()
        edge_path = self._extract_edge_paths(MAX_HOP_DISTANCE, edge_idx_map)

        # Laplacian decomposition
        EigVecs, EigVals = laplace_decomposition(dgl_graph, MAX_FREQS)

        graph_labels = {
            "spatial_pos": torch.from_numpy(spatial_pos).int(),
            "d2_distance": torch.from_numpy(d2_dist).float(),
            "angle_distance": torch.from_numpy(a3_dist).float(),
            "edges_path": torch.from_numpy(edge_path).int(),
            "centroid_distance": torch.from_numpy(centroid_dist).float(),
            "EigVecs": EigVecs,
            "EigVals": EigVals,
        }
        return dgl_graph, graph_labels

    def _extract_face_attributes(self, face_idx):
        original_face_topo = self.faces[face_idx]

        # 【MFTReNet方案二】 增加'是否为有理NURBS'属性的辅助函数
        def _is_rational_nurbs(face_topo):
            surf = BRepAdaptor_Surface(face_topo, True)
            stype = surf.GetType()
            if stype == GeomAbs_BSplineSurface:
                bspline = surf.BSpline()
                if bspline.IsURational() or bspline.IsVRational():
                    return 1
            elif stype == GeomAbs_BezierSurface:
                bezier = surf.Bezier()
                if bezier.IsURational() or bezier.IsVRational():
                    return 1
            return 0

        nurbs_converter = BRepBuilderAPI_NurbsConvert(original_face_topo, False)
        nurbs_face_topo = topods.Face(nurbs_converter.Shape())
        face = Face(nurbs_face_topo)
        points = uvgrid(face, method="point", num_u=UV_GRID_SIZE, num_v=UV_GRID_SIZE)
        normals = uvgrid(face, method="normal", num_u=UV_GRID_SIZE, num_v=UV_GRID_SIZE)
        visibility_status = uvgrid(face, method="visibility_status", num_u=UV_GRID_SIZE, num_v=UV_GRID_SIZE)
        mask = np.logical_or(visibility_status == 0, visibility_status == 2).reshape(UV_GRID_SIZE, UV_GRID_SIZE, 1)
        uv_grid_features = np.concatenate((points, normals, mask), axis=-1).astype(np.float32)
        face_type = self.original_surface_types[face_idx]
        original_face_wrapper = Face(original_face_topo)
        face_area = original_face_wrapper.area()
        face_loops = original_face_wrapper.num_wires()
        face_adj = len(self.adj_map.get(face_idx, []))
        face_centroid = self.face_centroids[face_idx]
        face_curvature = self._get_face_curvature(original_face_topo)
        inner_loop_prop = self._get_inner_loop_properties(original_face_topo)
        # 【MFTReNet方案二】 调用辅助函数获取新属性
        is_rational = _is_rational_nurbs(original_face_topo)

        return (uv_grid_features, face_type, float(face_area), int(face_loops), int(face_adj),
                face_centroid, face_curvature, inner_loop_prop, is_rational)

    def _get_face_curvature(self, face: TopoDS_Face) -> int:
        adaptor = BRepAdaptor_Surface(face, True)
        if adaptor.GetType() == GeomAbs_Plane: return 0
        u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
        props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
        if not props.IsNormalDefined(): return 0
        pnt = props.Value()
        normal = props.Normal()
        pnt_outside = gp_Pnt(pnt.X() + normal.X() * 1e-4, pnt.Y() + normal.Y() * 1e-4, pnt.Z() + normal.Z() * 1e-4)
        self.solid_classifier.Perform(pnt_outside, 1e-6)
        state = self.solid_classifier.State()
        if state == TopAbs_OUT:
            return 1
        elif state == TopAbs_IN:
            return 2
        else:
            return 0

    def _get_inner_loop_properties(self, face: TopoDS_Face) -> np.ndarray:
        num_concave_loops, num_convex_loops = 0, 0
        occwl_face = Face(face)
        outer_topo_wire = breptools_OuterWire(face)
        all_occwl_wires = occwl_face.wires()
        inner_wires = []
        if not outer_topo_wire.IsNull():
            inner_wires = [w for w in all_occwl_wires if not w.topods_shape().IsSame(outer_topo_wire)]
        if not inner_wires: return np.array([0, 0], dtype=np.int32)
        for wire in inner_wires:
            try:
                topo_wire = wire.topods_shape()
                mk_face_builder = BRepBuilderAPI_MakeFace(face, topo_wire)
                mk_face_builder.Build()
                if not mk_face_builder.IsDone(): continue
                temp_face = mk_face_builder.Face()
                props = GProp_GProps()
                brepgprop_SurfaceProperties(temp_face, props)
                pnt_on_wire_face = props.CentreOfMass()
                adaptor = BRepAdaptor_Surface(face, True)
                proj = GeomAPI_ProjectPointOnSurf(pnt_on_wire_face, adaptor.Surface().Surface())
                if proj.NbPoints() == 0: continue
                u, v = proj.LowerDistanceParameters()
                sl_props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
                if not sl_props.IsNormalDefined(): continue
                normal = sl_props.Normal()
                pnt_inside_feature = gp_Pnt(pnt_on_wire_face.X() - normal.X() * 1e-4,
                                            pnt_on_wire_face.Y() - normal.Y() * 1e-4,
                                            pnt_on_wire_face.Z() - normal.Z() * 1e-4)
                self.solid_classifier.Perform(pnt_inside_feature, 1e-6)
                state = self.solid_classifier.State()
                if state == TopAbs_IN:
                    num_convex_loops += 1
                elif state == TopAbs_OUT:
                    num_concave_loops += 1
            except Exception:
                continue
        return np.array([num_concave_loops, num_convex_loops], dtype=np.int32)

    def _extract_adj_face_stats(self, edge_convs: np.ndarray) -> np.ndarray:
        num_faces = len(self.faces)
        adj_stats = np.zeros((num_faces, NUM_ADJ_STATS), dtype=np.int32)
        edge_idx_counter = 0
        for u, v, _ in self.nx_graph.edges(data=True):
            conv_type = edge_convs[edge_idx_counter]
            if conv_type in [1, 2]:
                v_type = self.original_surface_types[v]
                index_for_u = v_type * 2 + (conv_type - 1)
                adj_stats[u, index_for_u] += 1
                u_type = self.original_surface_types[u]
                index_for_v = u_type * 2 + (conv_type - 1)
                adj_stats[v, index_for_v] += 1
            edge_idx_counter += 1
        return adj_stats

    def _extract_edge_attributes(self, edge_idx, face1_idx, face2_idx):
        topods_edge = self.edges[edge_idx]
        occwl_edge = Edge(topods_edge)

        # 【MFTReNet方案一: 增强边的U-Grid特征】
        # 使用 EdgeDataExtractor 提取更丰富的边信息
        face1_occwl = Face(self.faces[face1_idx])
        face2_occwl = Face(self.faces[face2_idx])

        # 确保传递给Extractor的面是正确的邻接面
        # occwl内部会根据边的方向确定左右面，这里我们提供所有可能的邻接面
        # 注意：occwl.solid.Solid.faces_from_edge 会返回正确的邻接面
        solid_for_edge = Solid(self.shape)
        faces_of_edge = list(solid_for_edge.faces_from_edge(occwl_edge))

        edge_data = EdgeDataExtractor(occwl_edge, faces_of_edge, num_samples=UV_GRID_SIZE, use_arclength_params=True)

        if not edge_data.good:
            # 对于无法处理的边（如退化边），返回零矩阵
            uv_grid_features = np.zeros((UV_GRID_SIZE, 12), dtype=np.float32)
        else:
            uv_grid_features = np.concatenate(
                [
                    edge_data.points,
                    edge_data.tangents,
                    edge_data.left_normals,
                    edge_data.right_normals
                ],
                axis=1
            ).astype(np.float32)

        # 提取其他边属性（与之前逻辑类似）
        edge_type = get_curve_type_enum(topods_edge)
        props = GProp_GProps()
        brepgprop.LinearProperties(topods_edge, props)
        edge_len = props.Mass()

        curve_adaptor = BRepAdaptor_Curve(topods_edge)
        u_min, u_max = curve_adaptor.FirstParameter(), curve_adaptor.LastParameter()
        if not np.isfinite(u_min) or not np.isfinite(u_max): u_min, u_max = 0., 1.
        mid_param = (u_min + u_max) / 2.0

        # 复用 EdgeDataExtractor 的结果来计算角度和凹凸性
        if edge_data.good:
            mid_idx = UV_GRID_SIZE // 2
            n1_vec_np = edge_data.left_normals[mid_idx]
            n2_vec_np = edge_data.right_normals[mid_idx]
            tangent_vec_np = edge_data.tangents[mid_idx]

            n1_vec = gp_Vec(*n1_vec_np)
            n2_vec = gp_Vec(*n2_vec_np)
            tangent_vec = gp_Vec(*tangent_vec_np)

            edge_ang = n1_vec.Angle(n2_vec)
            edge_ang = min(edge_ang, 2 * np.pi - edge_ang)

            if abs(edge_ang) < 1e-4:
                edge_conv = 3  # Smooth
            elif n1_vec.Crossed(n2_vec).Dot(tangent_vec) > 0:
                edge_conv = 2  # Convex
            else:
                edge_conv = 1  # Concave
        else:
            # 如果 EdgeDataExtractor 失败，则退回旧的计算方式或默认值
            edge_ang = 0.0
            edge_conv = 3

        return uv_grid_features, edge_type, float(edge_len), float(edge_ang), int(edge_conv)

    def _extract_proximity_features(self):
        num_faces = len(self.faces)
        path_lengths = nx.floyd_warshall_numpy(self.nx_graph, weight=None)
        path_lengths[path_lengths > SPATIAL_POS_MAX] = SPATIAL_POS_MAX
        bbox = Bnd_Box()
        brepbndlib.Add(self.shape, bbox)
        diag = bbox.CornerMin().Distance(bbox.CornerMax())
        if diag < 1e-6: diag = 1.0
        d2_distances = np.zeros((num_faces, num_faces, HISTOGRAM_BINS), dtype=np.float32)
        a3_distances = np.zeros((num_faces, num_faces, HISTOGRAM_BINS), dtype=np.float32)
        sampled_points = [self._sample_random_points(i, POINT_SAMPLES_FOR_D2_A3) for i in range(num_faces)]
        points_tensors = [torch.from_numpy(p).to(self.device) for p, n in sampled_points]
        normals_tensors = [torch.from_numpy(n).to(self.device) for p, n in sampled_points]
        for i in range(num_faces):
            for j in range(i, num_faces):
                if i == j:
                    d2_distances[i, i, 0], a3_distances[i, i, 0] = 1.0, 1.0
                    continue
                p_i, n_i, p_j, n_j = points_tensors[i], normals_tensors[i], points_tensors[j], normals_tensors[j]
                dists = torch.cdist(p_i, p_j) / diag
                d2_hist = torch.histc(dists, bins=HISTOGRAM_BINS, min=0.0, max=1.0)
                d2_hist_norm = d2_hist / (d2_hist.sum() + 1e-9)
                d2_distances[i, j, :] = d2_hist_norm.cpu().numpy()
                d2_distances[j, i, :] = d2_distances[i, j, :]
                cos_angles = torch.einsum('ik,jk->ij', n_i, n_j).flatten().clamp(-1.0, 1.0)
                angles = torch.acos(cos_angles)
                a3_hist = torch.histc(angles, bins=HISTOGRAM_BINS, min=0.0, max=np.pi)
                a3_hist_norm = a3_hist / (a3_hist.sum() + 1e-9)
                a3_distances[i, j, :] = a3_hist_norm.cpu().numpy()
                a3_distances[j, i, :] = a3_distances[i, j, :]
        centroids_tensor = torch.from_numpy(self.face_centroids).to(self.device)
        centroid_dist_matrix = (torch.cdist(centroids_tensor, centroids_tensor) / diag).cpu().numpy()
        return path_lengths, d2_distances, a3_distances, centroid_dist_matrix

    def _sample_random_points(self, face_idx, n_points):
        face = self.faces[face_idx]
        surface_adaptor = BRepAdaptor_Surface(face, True)
        try:
            u_min, u_max, v_min, v_max = surface_adaptor.FirstUParameter(), surface_adaptor.LastUParameter(), surface_adaptor.FirstVParameter(), surface_adaptor.LastVParameter()
        except:
            return np.zeros((n_points, 3), dtype=np.float32), np.array([[0, 0, 1]] * n_points, dtype=np.float32)
        if not all(np.isfinite([u_min, u_max, v_min, v_max])): u_min, u_max, v_min, v_max = -1., 1., -1., 1.
        points, normals = [], []
        attempts = 0
        classifier = BRepClass_FaceClassifier()
        while len(points) < n_points and attempts < n_points * 10:
            attempts += 1
            u, v = np.random.uniform(u_min, u_max), np.random.uniform(v_min, v_max)
            uv_pnt = gp_Pnt2d(u, v)
            classifier.Perform(face, uv_pnt, 1e-6)
            if classifier.State() == TopAbs_IN:
                pnt = surface_adaptor.Value(u, v)
                sl_props = BRepLProp_SLProps(surface_adaptor, u, v, 1, 1e-6)
                if sl_props.IsNormalDefined():
                    normal = sl_props.Normal()
                    points.append([pnt.X(), pnt.Y(), pnt.Z()])
                    normals.append([normal.X(), normal.Y(), normal.Z()])
        if not points: return np.zeros((n_points, 3), dtype=np.float32), np.array([[0, 0, 1]] * n_points,
                                                                                  dtype=np.float32)
        num_existing = len(points)
        points_arr, normals_arr = np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)
        if 0 < num_existing < n_points:
            indices_to_repeat = np.random.choice(num_existing, n_points - num_existing, replace=True)
            points_arr = np.vstack([points_arr, points_arr[indices_to_repeat]])
            normals_arr = np.vstack([normals_arr, normals_arr[indices_to_repeat]])
        return points_arr, normals_arr

    def _extract_edge_paths(self, max_len, edge_idx_map):
        num_faces = len(self.faces)
        try:
            all_paths = dict(nx.all_pairs_shortest_path(self.nx_graph))
        except nx.NetworkXNoPath:
            all_paths = {}
        edge_paths = np.zeros((num_faces, num_faces, max_len), dtype=int)
        for i in range(num_faces):
            for j in range(num_faces):
                if i == j: continue
                path_nodes = all_paths.get(i, {}).get(j, [])
                if len(path_nodes) > 1:
                    path_edge_indices = []
                    for k in range(len(path_nodes) - 1):
                        u, v = path_nodes[k], path_nodes[k + 1]
                        edge_data = self.nx_graph.get_edge_data(u, v)
                        if edge_data and 'edge_idx' in edge_data:
                            old_idx = edge_data['edge_idx']
                            new_idx = edge_idx_map.get(old_idx)
                            if new_idx is not None: path_edge_indices.append(new_idx + 1)
                    path_len = len(path_edge_indices)
                    edge_paths[i, j, :min(path_len, max_len)] = path_edge_indices[:max_len]
        return edge_paths


def process_file(file_path, label_dir, output_dir, device):
    basename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(basename)[0]
    label_file = os.path.join(label_dir, filename_no_ext + ".json")
    output_file = os.path.join(output_dir, filename_no_ext + ".bin")
    try:
        if not os.path.exists(label_file): return f"Skipped (no label): {basename}"
        reader = STEPControl_Reader()
        if reader.ReadFile(file_path) != 1: raise IOError(f"读取STEP文件失败: {file_path}")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape is None or shape.IsNull(): raise ValueError("从STEP文件中未能获取到有效的Shape。")
        shape = normalize_shape(shape)
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'cls' not in data or 'seg' not in data: raise ValueError(f"JSON文件 {label_file} 的格式不正确。")
        feature_labels_dict, instance_groups_list = data['cls'], data['seg']
        temp_face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        num_original_faces = 0
        while temp_face_explorer.More():
            num_original_faces += 1
            temp_face_explorer.Next()
        full_feature_labels = [0] * num_original_faces
        for face_idx_str, label in feature_labels_dict.items():
            face_idx = int(face_idx_str)
            if face_idx < num_original_faces: full_feature_labels[face_idx] = label
        extractor = BrepDataExtractor(shape, full_feature_labels, device=device)
        num_nodes_after_filtering = len(extractor.faces)
        if num_nodes_after_filtering <= 1: return f"Skipped (trivial graph with {num_nodes_after_filtering} nodes): {basename}"
        if extractor.nx_graph.number_of_edges() == 0: return f"Skipped (graph has no edges after filtering): {basename}"
        valid_indices = extractor.valid_original_indices
        filtered_feature_labels = [full_feature_labels[i] for i in valid_indices]
        full_instance_labels = [0] * num_original_faces
        for instance_id, face_group in enumerate(instance_groups_list, start=1):
            for face_idx in face_group:
                if face_idx < num_original_faces: full_instance_labels[face_idx] = instance_id
        filtered_instance_labels = [full_instance_labels[i] for i in valid_indices]
        remaining_instance_ids = sorted(list(set(i for i in filtered_instance_labels if i > 0)))
        id_map = {old_id: new_id for new_id, old_id in enumerate(remaining_instance_ids, start=1)}
        final_instance_labels = [id_map.get(i, 0) for i in filtered_instance_labels]
        graph, graph_labels = extractor.process(filtered_feature_labels, final_instance_labels)
        try:
            data_id = int(filename_no_ext.split('_')[-1])
        except (ValueError, IndexError):
            data_id = 0
        graph.data_id = data_id
        dgl.data.utils.save_graphs(output_file, [graph], graph_labels)
        return f"Success: {basename}"
    except Exception as e:
        return f"Failed: {basename} -> {e}"


def main():
    parser = argparse.ArgumentParser(description="从STEP文件夹批量、并行地提取B-rep数据到DGL图的.bin格式。")
    parser.add_argument("-i", "--input_dir", required=True, help="包含输入STEP (.stp, .step) 文件的文件夹路径。")
    parser.add_argument("-l", "--label_dir", required=True, help="包含面标签JSON文件的文件夹路径。")
    parser.add_argument("-o", "--output_dir", required=True, help="用于存储输出.bin文件的文件夹路径。")
    parser.add_argument("-n", "--num_workers", type=int, default=cpu_count(),
                        help=f"用于并行处理的进程数 (默认为: {cpu_count()})。")
    parser.add_argument("-t", "--gene_test", type=bool, default=False, help=f"用于判断是不是测试生成数据集。")
    args = parser.parse_args()
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"=================================================\n        B-rep to DGL Graph Conversion (Batch)      \n=================================================\n检测到可用设备: {device.upper()}\n输入 STEP 目录: {args.input_dir}\n输入 Label 目录: {args.label_dir}\n输出 Bin 目录: {args.output_dir}\n并行工作进程数: {args.num_workers}\n-------------------------------------------------")
    os.makedirs(args.output_dir, exist_ok=True)
    all_step_files = glob.glob(os.path.join(args.input_dir, "*.stp")) + glob.glob(
        os.path.join(args.input_dir, "*.step"))
    if not all_step_files:
        print("错误: 在输入目录中未找到任何 .stp 或 .step 文件。")
        return
    unprocessed_files = []
    for file_path in all_step_files:
        basename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(basename)[0]
        output_file = os.path.join(args.output_dir, filename_no_ext + ".bin")
        if not os.path.exists(output_file): unprocessed_files.append(file_path)
    print(f"在输入目录中总共发现 {len(all_step_files)} 个 STEP 文件。")
    print(f"其中 {len(unprocessed_files)} 个是尚未处理的新文件。")
    if args.gene_test:
        TARGET_FILE_COUNT = 5000
        print(
            f"测试模式已启用：目标是直到输出文件夹下有 {TARGET_FILE_COUNT} 个输出文件或输入文件夹的输入已经全部使用完。")
        results_aggregator = []
        while unprocessed_files:
            current_bin_count = len(glob.glob(os.path.join(args.output_dir, "*.bin")))
            if current_bin_count >= TARGET_FILE_COUNT:
                print(f"\n目标达成：输出文件夹中已有 {current_bin_count} 个文件。任务结束。")
                break
            batch_size = min(len(unprocessed_files), args.num_workers * 4 if args.num_workers > 0 else 8)
            batch_to_process = unprocessed_files[:batch_size]
            unprocessed_files = unprocessed_files[batch_size:]
            print(
                f"\n--- 输出: {current_bin_count}/{TARGET_FILE_COUNT} | 队列剩余: {len(unprocessed_files)} | 本轮处理: {len(batch_to_process)} ---")
            process_func = partial(process_file, label_dir=args.label_dir, output_dir=args.output_dir, device=device)
            with Pool(processes=args.num_workers) as pool:
                batch_results = list(tqdm(pool.imap(process_func, batch_to_process), total=len(batch_to_process),
                                          desc=f"进度 (目标 {TARGET_FILE_COUNT})"))
            results_aggregator.extend(batch_results)
        if not unprocessed_files and len(glob.glob(os.path.join(args.output_dir, "*.bin"))) < TARGET_FILE_COUNT:
            print(f"\n已处理完所有输入文件，但未达到 {TARGET_FILE_COUNT} 的目标。")
        results = results_aggregator
    else:
        print("普通模式：将处理所有新发现的文件。")
        if not unprocessed_files:
            print("没有待处理的新文件，任务结束。")
            return
        print(f"本次任务将处理 {len(unprocessed_files)} 个文件。")
        process_func = partial(process_file, label_dir=args.label_dir, output_dir=args.output_dir, device=device)
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(pool.imap(process_func, unprocessed_files), total=len(unprocessed_files), desc="总体进度"))
    print("-------------------------------------------------\n所有处理完成。")
    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))
    print(f"总成功: {success_count}, 总失败: {failed_count}, 总跳过: {skipped_count}")
    if failed_count > 0:
        print("\n失败文件详情:")
        for r in results:
            if r.startswith("Failed"): print(f"- {r}")


if __name__ == '__main__':
    main()