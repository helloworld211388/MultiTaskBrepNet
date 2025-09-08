import os
import random


def split_dataset(folder_path, is_test_mode=False):
    """
    读取指定文件夹下所有.bin文件的名字（不含后缀），并根据模式进行划分。
    标准模式：使用所有文件。
    测试模式：最多随机抽取5000个文件。

    划分比例固定为 70% 训练集, 15% 验证集, 15% 测试集。
    将这些名字分别保存为 train.txt, val.txt, 和 test.txt。

    参数:
    folder_path (str): .bin文件所在的文件夹路径。
    is_test_mode (bool): 是否启用测试模式。
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    # 1. 读取所有.bin文件的名字（不含后缀）
    try:
        filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.bin')]
    except OSError as e:
        print(f"错误：无法读取文件夹 '{folder_path}'。请检查权限。")
        print(e)
        return

    # 如果是测试模式，则进行抽样
    if is_test_mode:
        print("--- 当前运行在测试模式 ---")
        max_samples = 5000
        if len(filenames) > max_samples:
            print(f"样本总数 ({len(filenames)}) 大于 {max_samples}。将随机抽取 {max_samples} 个样本进行划分。")
            filenames = random.sample(filenames, max_samples)
        else:
            print(f"样本总数 ({len(filenames)}) 不超过 {max_samples}。将使用所有样本进行划分。")
        print("--------------------------")

    # 2. 统计最终要处理的数量
    total_files = len(filenames)
    if total_files == 0:
        print(f"警告：在 '{folder_path}' 中没有找到任何.bin文件。")
        return

    print(f"共使用 {total_files} 个.bin文件进行划分。")

    # 3. 随机打乱文件名列表
    random.shuffle(filenames)

    # 4. 计算 70%:15%:15% 比例的分割点
    train_split_index = int(total_files * 0.70)
    val_split_index = int(total_files * 0.85)  # 70% + 15%

    # 5. 分割数据集
    train_files = filenames[:train_split_index]
    val_files = filenames[train_split_index:val_split_index]
    test_files = filenames[val_split_index:]

    # 打印各个集合的数量
    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}")
    print(f"测试集数量: {len(test_files)}")
    print("------")
    print(f"总计: {len(train_files) + len(val_files) + len(test_files)}")

    # 6. 将文件名列表写入txt文件
    try:
        with open('train.txt', 'w') as f:
            for name in train_files:
                f.write(name + '\n')

        with open('val.txt', 'w') as f:
            for name in val_files:
                f.write(name + '\n')

        with open('test.txt', 'w') as f:
            for name in test_files:
                f.write(name + '\n')

        print("\n成功创建 train.txt, val.txt, 和 test.txt 文件。")

    except IOError as e:
        print(f"错误：写入文件时发生错误。")
        print(e)


# --- 使用说明 ---
if __name__ == '__main__':
    # --- 设置 ---
    # 是否启用测试模式？
    # True: 最多随机抽取5000个样本进行划分。
    # False: 使用文件夹中的所有文件。
    TEST_MODE = False  # <--- 在这里修改 True/False

    # *** 请修改为你存放.bin文件的文件夹路径 ***
    # 例如:
    #   Windows: "C:\\Users\\YourUser\\Desktop\\my_bins"
    #   macOS/Linux: "/home/user/data/bins"
    target_folder = "./datasets/bin"  # <--- 在这里替换你的路径

    # 根据设置的模式运行脚本
    split_dataset(target_folder, is_test_mode=TEST_MODE)
