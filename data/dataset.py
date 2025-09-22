# -*- coding: utf-8 -*-
import os
import pathlib
from tqdm import tqdm
import random
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from prefetch_generator import BackgroundGenerator

from .collator import collator, collator_st
from .utils import get_random_rotation, rotate_uvgrid


class DataLoaderX(DataLoader):
    def __iter__(self):
        # --- 代码来源，未做修改 ---
        return BackgroundGenerator(super().__iter__())


class CADSynth(Dataset):
    def __init__(
            self,
            root_dir,
            split="train",
            random_rotate=False,
            num_class=25,
            num_workers=1,
    ):
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)
        self.split = split
        self.num_class = num_class
        self.random_rotate = random_rotate
        self.num_workers = num_workers
        self.file_paths = []
        self._get_filenames(path, filelist=split + ".txt")

    def _get_filenames(self, root_dir, filelist):
        # --- 代码来源，未做修改 ---
        print(f"Loading data...")
        with open(str(root_dir / f"{filelist}"), "r") as f:
            file_list = [x.strip() for x in f.readlines()]
        bin_dir = root_dir / "bin"
        for x in tqdm(bin_dir.glob('**/*.bin')):
            if x.stem in file_list:
                self.file_paths.append(x)
        print("Done loading {} files".format(len(self.file_paths)))

    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        if (self.random_rotate):
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)

        pyg_graph.face_type = graph.ndata["z"].type(torch.int)
        pyg_graph.face_area = graph.ndata["y"].type(torch.float)
        pyg_graph.face_loop = graph.ndata["l"].type(torch.int)
        pyg_graph.face_adj = graph.ndata["a"].type(torch.int)
        pyg_graph.label_feature = graph.ndata["f"].type(torch.int)

        # 假设实例标签存储在 graph.ndata 的 "i" 键中
        if "i" in graph.ndata:
            pyg_graph.instance_label = graph.ndata["i"].type(torch.int)
        else:
            # 如果找不到实例标签，则用一个占位符（例如全0）填充，并打印警告
            print(f"Warning: Instance label key 'i' not found in {file_path}. Using zeros as placeholder.")
            num_nodes = graph.num_nodes()
            pyg_graph.instance_label = torch.zeros(num_nodes, dtype=torch.int)

        instance_labels = pyg_graph.instance_label
        unique_ids = torch.unique(instance_labels[instance_labels > 0])  # 忽略背景ID 0
        positive_pairs = []
        for inst_id in unique_ids:
            # 找到属于当前实例的所有面的索引
            face_indices = torch.where(instance_labels == inst_id)[0]
            # 如果实例内有多个面，则生成所有可能的配对 (combinations)
            if len(face_indices) > 1:
                pairs = torch.combinations(face_indices, r=2)
                positive_pairs.append(pairs)

        if positive_pairs:
            # 将所有配对拼接成一个 [num_pairs, 2] 的张量，然后转置为 [2, num_pairs]
            pyg_graph.instance_pos_edge_index = torch.cat(positive_pairs, dim=0).t().contiguous()
        else:
            # 如果没有找到任何正样本对，则创建一个空的张量
            pyg_graph.instance_pos_edge_index = torch.empty((2, 0), dtype=torch.long)

        if "rational" in graph.ndata:
            pyg_graph.rational = graph.ndata["rational"].type(torch.int)
        else:
            print(f"Warning: Rational NURBS key 'rational' not found in {file_path}. Using zeros as placeholder.")
            num_nodes = graph.num_nodes()
            pyg_graph.rational = torch.zeros(num_nodes, dtype=torch.int)

        pyg_graph.edge_type = graph.edata["t"].type(torch.int)
        pyg_graph.edge_len = graph.edata["l"].type(torch.float)
        pyg_graph.edge_ang = graph.edata["a"].type(torch.float)
        pyg_graph.edge_conv = graph.edata["c"].type(torch.int)

        dense_adj = graph.adj().to_dense().type(torch.int)
        n_nodes = graph.num_nodes()
        pyg_graph.node_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.edge_path = graphfile[1]["edges_path"]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]
        try:
            pyg_graph.centroid = graph.ndata["centroid"].type(torch.float)
        except KeyError:
            print(f"\nError: 'centroid' key not found in ndata for file: {file_path}")
            num_nodes = graph.num_nodes()
            pyg_graph.centroid = torch.zeros(num_nodes, 3, dtype=torch.float)
        pyg_graph.curvature = graph.ndata["curvature"].type(torch.int)
        pyg_graph.inner_loops = graph.ndata["inner_loops"].type(torch.float)
        pyg_graph.adj_stats = graph.ndata["adj_stats"].type(torch.float)
        pyg_graph.centroid_distance = graphfile[1]["centroid_distance"]

        # Load Laplacian features
        pyg_graph.EigVecs = graphfile[1]["EigVecs"]
        pyg_graph.EigVals = graphfile[1]["EigVals"]

        _, file_extension = os.path.splitext(file_path)
        basename = os.path.basename(file_path).replace(file_extension, "")
        pyg_graph.data_id = int(basename.split("_")[-2])

        if (torch.max(pyg_graph.label_feature) > 26 or torch.max(pyg_graph.label_feature) < 0):
            print(pyg_graph.data_id)

        return pyg_graph

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # --- 代码来源，未做修改 ---
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)
        return sample

    def _collate(self, batch):
        return collator(
            batch,
            multi_hop_max_dist=6,#减少显存消耗
            spatial_pos_max=32,
        )

    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoaderX(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            # 使用 self.num_workers
            num_workers=self.num_workers,
            # 修改 drop_last 行为
            drop_last=(self.split == 'train'),  # 仅在训练时丢弃最后一个不完整的批次
            pin_memory=True,
            persistent_workers=False if self.num_workers > 0 else False,
        )


class TransferDataset(Dataset):
    def __init__(
            self,
            root_dir_source,
            root_dir_target,
            split="train",
            random_rotate=False,
            num_class=25,
            open_set=0,
            num_workers=0,
    ):
        assert split in ("train", "val", "test")
        source_path = pathlib.Path(root_dir_source)
        target_path = pathlib.Path(root_dir_target)
        self.split = split
        self.random_rotate = random_rotate
        self.num_class = num_class
        self.num_workers = num_workers
        self.open_set = bool(open_set)

        # 初始化路径列表
        self.source_file_paths = []
        self.target_file_paths = []

        # 分别为源和目标加载文件名
        print("--- Loading Source Data ---")
        self._get_filenames(source_path, split + ".txt", self.source_file_paths)
        print("--- Loading Target Data ---")
        self._get_filenames(target_path, split + ".txt", self.target_file_paths)

    def _get_filenames(self, root_dir, filelist_name, path_list):
        """
        从 root_dir 中根据 filelist_name 查找文件，并填充到 path_list 中。
        """
        print(f"--- Debugging Data Loading ---")
        print(f"Root directory: {root_dir}")
        filelist_path = root_dir / filelist_name
        print(f"File list path: {filelist_path}")

        if not filelist_path.exists():
            print(f"Warning: File list not found at {filelist_path}. Skipping.")
            return

        with open(str(filelist_path), "r") as f:
            file_list = {x.strip() for x in f.readlines()}  # 使用集合以提高查找效率
        print(f"Loaded {len(file_list)} names from {filelist_name}.")

        bin_dir = root_dir / "bin"
        print(f"Searching for .bin files in: {bin_dir}")

        if not bin_dir.exists():
            print(f"Warning: Bin directory not found at {bin_dir}. Skipping.")
            return

        for x in tqdm(bin_dir.glob('**/*.bin')):
            if x.stem in file_list:
                path_list.append(x)

        print(f"Done loading {len(path_list)} files.")
        print(f"--- End Debugging ---")

    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        if (self.random_rotate):
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)

        pyg_graph.face_type = graph.ndata["z"].type(torch.int)
        pyg_graph.face_area = graph.ndata["y"].type(torch.float)
        pyg_graph.face_loop = graph.ndata["l"].type(torch.int)
        pyg_graph.face_adj = graph.ndata["a"].type(torch.int)
        pyg_graph.label_feature = graph.ndata["f"].type(torch.int)

        # --- 修改开始 ---
        # 同样为 TransferDataset 添加实例标签的加载逻辑
        if "i" in graph.ndata:
            pyg_graph.instance_label = graph.ndata["i"].type(torch.int)
        else:
            print(f"Warning: Instance label key 'i' not found in {file_path}. Using zeros as placeholder.")
            num_nodes = graph.num_nodes()
            pyg_graph.instance_label = torch.zeros(num_nodes, dtype=torch.int)
        # --- 修改结束 ---

        pyg_graph.edge_type = graph.edata["t"].type(torch.int)
        pyg_graph.edge_len = graph.edata["l"].type(torch.float)
        pyg_graph.edge_ang = graph.edata["a"].type(torch.float)
        pyg_graph.edge_conv = graph.edata["c"].type(torch.int)

        dense_adj = graph.adj().to_dense().type(torch.int)
        n_nodes = graph.num_nodes()
        pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.edge_path = graphfile[1]["edges_path"]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]
        try:
            pyg_graph.centroid = graph.ndata["centroid"].type(torch.float)
        except KeyError:
            print(f"\nError: 'centroid' key not found in ndata for file: {file_path}")
            pyg_graph.centroid = None
        pyg_graph.curvature = graph.ndata["curvature"].type(torch.int)
        pyg_graph.inner_loops = graph.ndata["inner_loops"].type(torch.float)
        pyg_graph.adj_stats = graph.ndata["adj_stats"].type(torch.float)
        pyg_graph.centroid_distance = graphfile[1]["centroid_distance"]

        # Load Laplacian features
        pyg_graph.EigVecs = graphfile[1]["EigVecs"]
        pyg_graph.EigVals = graphfile[1]["EigVals"]

        _, file_extension = os.path.splitext(file_path)
        basename = os.path.basename(file_path).replace(file_extension, "")
        pyg_graph.data_id = int(basename.split("_")[-2])

        return pyg_graph

    def __len__(self):
        # --- 代码来源，未做修改 ---
        if self.split == "train":
            return max(len(self.source_file_paths), len(self.target_file_paths))
        else:
            return len(self.target_file_paths)

    def __getitem__(self, idx):
        # --- 代码来源，未做修改 ---
        idx_s = idx
        idx_t = idx
        if idx_s >= len(self.source_file_paths):
            idx_s = random.randint(0, len(self.source_file_paths) - 1)
        if idx_t >= len(self.target_file_paths):
            idx_t = random.randint(0, len(self.target_file_paths) - 1)

        fn_s = self.source_file_paths[idx_s]
        fn_t = self.target_file_paths[idx_t]

        sample_s = self.load_one_graph(fn_s)
        sample_t = self.load_one_graph(fn_t)
        sample = {"source_data": sample_s, "target_data": sample_t}
        return sample

    def _collate(self, batch):
        # --- 代码来源，未做修改 ---
        return collator_st(
            batch,
            multi_hop_max_dist=16,
            spatial_pos_max=32,
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        # --- 代码来源，未做修改 ---
        return DataLoaderX(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )