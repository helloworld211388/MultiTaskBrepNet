import os
import pathlib
import json
import argparse
from tqdm import tqdm
from dgl.data.utils import load_graphs
from multiprocessing import Pool, cpu_count

def get_node_count(file_path):
    """加载单个.bin文件并返回图的节点数。"""
    try:
        graph_list, _ = load_graphs(str(file_path))
        if graph_list:
            # 文件名作为key，节点数作为value
            return file_path.stem, graph_list[0].num_nodes()
    except Exception as e:
        print(f"Warning: Could not process file {file_path}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Pre-computes and caches the number of nodes for each graph in the dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default='./datasets',
        help="Path to the root directory of the dataset (e.g., ./datasets).",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=cpu_count(),
        help=f"Number of worker processes to use (default: {cpu_count()}).",
    )
    args = parser.parse_args()

    root_path = pathlib.Path(args.dataset_path)
    bin_dir = root_path / "bin"
    output_cache_file = root_path / "node_counts.json"

    if not bin_dir.is_dir():
        print(f"Error: 'bin' directory not found in {root_path}")
        return

    all_bin_files = list(bin_dir.glob('**/*.bin'))
    if not all_bin_files:
        print(f"No .bin files found in {bin_dir}")
        return

    print(f"Found {len(all_bin_files)} .bin files. Starting node count pre-computation...")

    # 使用多进程并行处理
    with Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(get_node_count, all_bin_files),
                total=len(all_bin_files),
                desc="Processing files"
            )
        )

    # 过滤掉处理失败的文件
    node_counts = {filename: count for filename, count in results if filename is not None}

    # 保存到JSON文件
    with open(output_cache_file, "w") as f:
        json.dump(node_counts, f, indent=4)

    print(f"\nSuccessfully pre-computed node counts for {len(node_counts)} graphs.")
    print(f"Cache saved to: {output_cache_file}")

if __name__ == "__main__":
    main()
