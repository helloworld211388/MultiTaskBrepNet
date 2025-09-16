# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import CADSynth
# --- 修改开始 ---
# 导入重构后的多任务模型 MultiTaskBrepNet，替换原有的 BrepSeg
from models.brepseg_model import MultiTaskBrepNet
# --- 修改结束 ---
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("BrepMFR Multi-Task Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument("--num_classes", type=int, default=27, help="Number of semantic classes")
# --- 修改开始 ---
# 1. 为实例分割任务添加类别数量参数
parser.add_argument("--num_instance_classes", type=int, default=26,
                    help="Number of instance classes. Defaults to num_classes if not specified.")
# --- 修改结束 ---
parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="cadsynth", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=6,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="MultiTaskBrepNet",  # 修改默认实验名称
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

# 设置transformer模块的默认参数
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--attention_dropout", type=float, default=0.3)
parser.add_argument("--act-dropout", type=float, default=0.3)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dim_node", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=32)
# --- 修改开始 ---
# 2. 添加用于配置“共享+任务特定”编码器结构的参数
parser.add_argument("--n_layers_encode", type=int, default=8,
                    help="Total number of encoder layers, used for compatibility. Will be overridden by shared/specific layers.")
parser.add_argument("--num_shared_layers", type=int, default=4, help="Number of shared layers in the encoder.")
parser.add_argument("--num_semantic_layers", type=int, default=4,
                    help="Number of task-specific layers in the SEMANTIC branch of the encoder.")
parser.add_argument("--num_instance_layers", type=int, default=4,
                    help="Number of task-specific layers in the INSTANCE branch of the encoder.")
parser.add_argument("--semantic_loss_weight", type=float, default=0.5,
                    help="The weight for the semantic loss component in the total loss.")
# --- 修改结束 ---

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

# 如果未指定实例类别数，则使其与语义类别数相同
if args.num_instance_classes is None:
    args.num_instance_classes = args.num_classes

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="eval_loss",  # 监控验证集总损失
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best-{epoch}-{semantic_accuracy:.4f}-{instance_accuracy:.4f}",
    save_top_k=3,
    save_last=True,
)
# 新增：实例化EarlyStopping回调
early_stop_callback = EarlyStopping(
   monitor="eval_loss",   # 监控的指标，必须是在验证集上记录的
   min_delta=0.00001,        # 认为模型有提升的最小变化量
   patience=10,           # 在停止前，允许指标不提升的epoch数
   verbose=True,         # 在终端打印早停信息
   mode="min"             # "min"表示监控的指标越小越好（如loss），"max"则相反（如F1分数）
)
trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True,
    gradient_clip_val=1.0,
    precision=32, # 启用16位混合精度
    amp_backend='native' # 指定使用PyTorch原生AMP后端
)

if args.dataset == "cadsynth":
    Dataset = CADSynth
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train":
    # Train/val
    print(
        f"""
-----------------------------------------------------------------------------------
B-rep model Multi-Task Feature Recognition (Semantic + Instance)
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best-....ckpt
-----------------------------------------------------------------------------------
    """
    )

    # 3. 实例化新的 MultiTaskBrepNet 模型
    model = MultiTaskBrepNet(args)


    train_data = Dataset(root_dir=args.dataset_path, split="train", random_rotate=True, num_class=args.num_classes, num_workers=args.num_workers)
    val_data = Dataset(root_dir=args.dataset_path, split="val", random_rotate=False, num_class=args.num_classes, num_workers=args.num_workers)
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False
    )
    trainer.fit(model, train_loader, val_loader)

else:
    # Test
    assert (
            args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=args.num_classes)
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False
    )
    # --- 修改开始 ---
    # 4. 从 checkpoint 加载时，同样使用新的模型类
    model = MultiTaskBrepNet.load_from_checkpoint(args.checkpoint)
    # --- 修改结束 ---
    trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint, verbose=False)
