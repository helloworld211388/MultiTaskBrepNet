# -*- coding: utf-8 -*-
import os
import pathlib
import json  # 新增导入
import random
from tqdm import tqdm
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader, Sampler  # 新增导入 Sampler
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from prefetch_generator import BackgroundGenerator

from .collator import collator, collator_st
from .utils import get_random_rotation, rotate_uvgrid

multi_hop_max_dist=16;
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# --- 新增：BucketBatchSampler ---
class BucketBatchSampler(Sampler):
    """
    一个自定义的 Sampler，用于实现数据分桶。
    它将数据按尺寸（节点数）排序，然后创建大小相近的批次。
    为了保持一定的随机性，它会先将数据分到大的桶（bucket）中，
    打乱桶的顺序，然后在桶内进行批次划分。
    """

    def __init__(self, data_source, batch_size, drop_last):
        super().__init__(data_source)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 从 data_source 获取预先计算好的节点数
        # 我们假设 data_source 有一个名为 `get_num_nodes_list` 的方法
        indices_and_sizes = data_source.get_indices_and_num_nodes()

        # 按节点数对 (index, size) 对进行排序
        self.sorted_indices = [idx for idx, size in sorted(indices_and_sizes, key=lambda x: x[1])]

    def __iter__(self):
        # 创建一个索引副本进行操作
        indices = self.sorted_indices[:]

        # 在一个大的桶（比如100个批次大小）内进行随机化
        bucket_size = self.batch_size * 100

        # 将索引分桶
        buckets = [indices[i:i + bucket_size] for i in range(0, len(indices), bucket_size)]

        # 打乱桶的顺序，但不打乱桶内的元素顺序
        random.shuffle(buckets)

        # 展平，得到一个半随机的序列
        shuffled_indices = [idx for bucket in buckets for idx in bucket]

        # 产生批次
        for i in range(0, len(shuffled_indices), self.batch_size):
            batch_indices = shuffled_indices[i:i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


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
        self.root_path = pathlib.Path(root_dir)  # 修改为实例变量
        self.split = split
        self.num_class = num_class
        self.random_rotate = random_rotate
        self.num_workers = num_workers
        self.file_paths = []

        # --- 修改：加载文件名和预计算的节点数 ---
        self._load_metadata()

    def _load_metadata(self):
        """加载文件名和预计算的节点数缓存。"""
        print(f"Loading data for split: {self.split}...")

        # 1. 加载文件名列表 (train.txt, val.txt, test.txt)
        filelist_path = self.root_path / f"{self.split}.txt"
        try:
            with open(str(filelist_path), "r") as f:
                self.file_list_stems = {x.strip() for x in f.readlines()}
        except FileNotFoundError:
            print(f"Error: File list not found at {filelist_path}. Cannot proceed.")
            raise

        # 2. 加载预计算的节点数缓存
        cache_path = self.root_path / "node_counts.json"
        try:
            with open(cache_path, "r") as f:
                self.node_counts = json.load(f)
        except FileNotFoundError:
            print("\n" + "=" * 50)
            print("Warning: `node_counts.json` not found!")
            print("Please run `precompute_node_counts.py` first to generate the cache file.")
            print("This will significantly speed up data loading for bucketing.")
            print("=" * 50 + "\n")
            raise

        # 3. 筛选出当前split需要的文件路径，并存储它们的节点数
        bin_dir = self.root_path / "bin"
        self.file_paths = []
        self.indices_and_sizes = []

        for file_path in tqdm(bin_dir.glob('**/*.bin'), desc=f"Filtering files for '{self.split}' split"):
            stem = file_path.stem
            if stem in self.file_list_stems:
                self.file_paths.append(file_path)

                # 存储索引和对应的节点数
                if stem in self.node_counts:
                    self.indices_and_sizes.append((len(self.file_paths) - 1, self.node_counts[stem]))
                else:
                    # 如果缓存中没有，给出警告
                    print(f"Warning: Node count for '{stem}' not found in cache. Bucketing might be inaccurate.")
                    # 也可以选择在这里动态计算，作为后备方案
                    # _, count = get_node_count(file_path)
                    # self.indices_and_sizes.append((len(self.file_paths) - 1, count))

        print(f"Done loading metadata. Found {len(self.file_paths)} files for '{self.split}' split.")

    def get_indices_and_num_nodes(self):
        """为 BucketBatchSampler 提供数据。"""
        return self.indices_and_sizes

    # ... (load_one_graph, __len__, __getitem__ 方法保持不变) ...
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
        if "i" in graph.ndata:
            pyg_graph.instance_label = graph.ndata["i"].type(torch.int)
        else:
            print(f"Warning: Instance label key 'i' not found in {file_path}. Using zeros as placeholder.")
            num_nodes = graph.num_nodes()
            pyg_graph.instance_label = torch.zeros(num_nodes, dtype=torch.int)
        instance_labels = pyg_graph.instance_label
        unique_ids = torch.unique(instance_labels[instance_labels > 0])
        positive_pairs = []
        for inst_id in unique_ids:
            face_indices = torch.where(instance_labels == inst_id)[0]
            if len(face_indices) > 1:
                pairs = torch.combinations(face_indices, r=2)
                positive_pairs.append(pairs)
        if positive_pairs:
            pyg_graph.instance_pos_edge_index = torch.cat(positive_pairs, dim=0).t().contiguous()
        else:
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
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)
        return sample

    def _collate(self, batch):
        return collator(
            batch,
            multi_hop_max_dist=multi_hop_max_dist,
            spatial_pos_max=32,
        )

    def get_dataloader(self, batch_size, shuffle=True):
        # --- 修改：使用 BucketBatchSampler ---
        if shuffle:
            # 训练时使用分桶采样器
            bucket_sampler = BucketBatchSampler(
                data_source=self,
                batch_size=batch_size,
                drop_last=(self.split == 'train')
            )

            return DataLoaderX(
                dataset=self,
                batch_sampler=bucket_sampler,  # 注意：使用 batch_sampler
                collate_fn=self._collate,
                num_workers=self.num_workers,
                pin_memory=True, # 保持开启，加速CPU到GPU的传输
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
        else:
            # 验证/测试时，通常不需要分桶，按顺序即可
            return DataLoaderX(
                dataset=self,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
                persistent_workers=False,
            )


# ... (TransferDataset 类保持不变) ...
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
        self.source_file_paths = []
        self.target_file_paths = []
        print("--- Loading Source Data ---")
        self._get_filenames(source_path, split + ".txt", self.source_file_paths)
        print("--- Loading Target Data ---")
        self._get_filenames(target_path, split + ".txt", self.target_file_paths)

    def _get_filenames(self, root_dir, filelist_name, path_list):
        print(f"--- Debugging Data Loading ---")
        print(f"Root directory: {root_dir}")
        filelist_path = root_dir / filelist_name
        print(f"File list path: {filelist_path}")
        if not filelist_path.exists():
            print(f"Warning: File list not found at {filelist_path}. Skipping.")
            return
        with open(str(filelist_path), "r") as f:
            file_list = {x.strip() for x in f.readlines()}
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
        if "i" in graph.ndata:
            pyg_graph.instance_label = graph.ndata["i"].type(torch.int)
        else:
            print(f"Warning: Instance label key 'i' not found in {file_path}. Using zeros as placeholder.")
            num_nodes = graph.num_nodes()
            pyg_graph.instance_label = torch.zeros(num_nodes, dtype=torch.int)
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
        pyg_graph.EigVecs = graphfile[1]["EigVecs"]
        pyg_graph.EigVals = graphfile[1]["EigVals"]
        _, file_extension = os.path.splitext(file_path)
        basename = os.path.basename(file_path).replace(file_extension, "")
        pyg_graph.data_id = int(basename.split("_")[-2])
        return pyg_graph

    def __len__(self):
        if self.split == "train":
            return max(len(self.source_file_paths), len(self.target_file_paths))
        else:
            return len(self.target_file_paths)

    def __getitem__(self, idx):
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
        return collator_st(
            batch,
            multi_hop_max_dist=multi_hop_max_dist,
            spatial_pos_max=32,
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
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