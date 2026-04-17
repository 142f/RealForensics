
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torchvision.transforms.functional

# 兼容某些 torchvision / pytorchvideo 版本的导入路径差异
sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional

from pytorchvideo.transforms import RemoveKey
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    RandomApply,
    RandomErasing,
    RandomGrayscale,
    Resize,
)
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_ROOT = REPO_ROOT / "stage2"
for path in (str(REPO_ROOT), str(STAGE2_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from stage2.data.pytorchvideo_utils import labeled_video_dataset_with_fix
from stage2.metrics import VideoLevelAUROC, VideoLevelAcc, VideoLevelAUROCCDF, VideoLevelAUROCDFDC
from stage2.models.model_combined import ModelCombined
from stage2.data.transforms import LambdaModule, TimeMask, TimeMaskV2, ZeroPadTemp
from pytorchvideo.data.clip_sampling import ClipInfo, UniformClipSampler


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*Defaults list is missing `_self_`.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    # 这个告警通常来自只含单一类别时的内部指标计算，对最终自定义二分类汇总没有影响
    warnings.filterwarnings(
        "ignore",
        message=r".*No positive samples in targets.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*No negative samples in targets.*",
        category=UserWarning,
    )


def seed_everything_simple(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ApplyTransformToKeyAug:
    def __init__(self, transform_video, transform_video_aug=None, time_mask_video=None, args=None):
        self._transform_video = transform_video
        self._transform_video_aug = transform_video_aug
        self._time_mask_video = time_mask_video
        self._pad_video = ZeroPadTemp(args.num_frames)
        self._args = args

    def __call__(self, x):
        if "video_name" in x:
            del x["video_name"]
        x["video_aug"], _ = self._pad_video(self._transform_video_aug(x["video"]))
        x["video"], x["mask"] = self._pad_video(self._transform_video(x["video"]))

        if self._time_mask_video is not None and torch.rand(1) < self._args.time_mask_prob_video:
            x["video_aug"], time_mask = self._time_mask_video(x["video_aug"])
            bool_mask = torch.ones(self._args.num_frames, dtype=x["mask"].dtype, device=x["mask"].device)
            bool_mask[time_mask] = 0
            x["mask"] *= bool_mask
        return x


class UniformClipSamplerWithDuration(UniformClipSampler):
    def __init__(self, clip_duration, video_duration, stride=None, backpad_last=False, eps=1e-6):
        super().__init__(clip_duration, stride, backpad_last, eps)
        self._video_duration = video_duration

    def __call__(self, last_clip_time, video_duration, annotation):
        video_duration = self._video_duration
        clip_start, clip_end = self._clip_start_end(last_clip_time, video_duration, backpad_last=self._backpad_last)
        _, next_clip_end = self._clip_start_end(clip_end, video_duration, backpad_last=self._backpad_last)
        if self._backpad_last:
            is_last_clip = abs(next_clip_end - clip_end) < self._eps
        else:
            is_last_clip = next_clip_end > video_duration
        clip_index = self._current_clip_index
        self._current_clip_index += 1
        return ClipInfo(clip_start, clip_end, clip_index, 0, is_last_clip)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate custom AVI datasets with RealForensics stage2 official-equivalent logic."
    )
    parser.add_argument("--weights", required=True, help="stage2 权重路径，或 stage2/weights 下的文件名。")
    parser.add_argument(
        "--test-root",
        required=True,
        help="支持两种结构：1) 根目录下 0_real/1_fake/*.avi；2) 根目录下 Real/<fake_type>/c23/cropped_faces/*.avi",
    )
    parser.add_argument("--real-csv", default=None, help="真实类 CSV 清单路径；留空时自动生成。")
    parser.add_argument("--fake-csv", default=None, help="伪造类 CSV 清单路径；留空时自动生成。")
    parser.add_argument("--fake-type-name", required=True, help="伪造类别名称。")
    parser.add_argument("--output-dir", required=True, help="输出目录。")
    parser.add_argument("--devices", type=int, default=1, help="使用 GPU 数量；为 0 时走 CPU。注意：这不是 GPU 编号。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数。")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小。")
    parser.add_argument("--compression", default="c23", help="压缩分支名。")
    parser.add_argument("--crop-type", default="full_face", help="Hydra crop_type 配置名。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--cdf-dfdc-test", action="store_true", help="是否额外启用 CelebDF/DFDC 官方测试。")
    parser.add_argument("--progress-refresh", type=int, default=1, help="每处理多少个 batch 更新一次进度条后缀。")
    parser.add_argument("--quiet-warnings", action="store_true", help="屏蔽不会影响结果的 Hydra / torch.load / 单类指标提示。")
    return parser.parse_args()


def resolve_weights_path(weights_arg: str) -> Path:
    weights_path = Path(weights_arg)
    if weights_path.is_absolute() or weights_path.exists():
        return weights_path.resolve()
    return (STAGE2_ROOT / "weights" / weights_arg).resolve()


def safe_torch_load_state_dict(weights_path: Path):
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def is_flat_two_dir_root(test_root: Path) -> bool:
    return (test_root / "0_real").is_dir() and (test_root / "1_fake").is_dir()


def collect_avi_paths(video_dir: Path) -> list[Path]:
    video_paths = sorted(video_dir.rglob("*.avi"), key=lambda path: path.as_posix().lower())
    if not video_paths:
        raise RuntimeError(f"No AVI files found in {video_dir}")
    return video_paths


def write_auto_manifest(video_dir: Path, csv_path: Path, label: int) -> None:
    video_paths = collect_avi_paths(video_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{path.relative_to(video_dir).as_posix()} {label}" for path in video_paths]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_eval_inputs(args, output_dir: Path):
    test_root = Path(args.test_root).resolve()
    args.test_root = str(test_root)

    if is_flat_two_dir_root(test_root):
        auto_csv_dir = output_dir / f"auto_csv_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if args.real_csv is None:
            args.real_csv = str(auto_csv_dir / "test_real.csv")
            write_auto_manifest(test_root / "0_real", Path(args.real_csv), 0)
        if args.fake_csv is None:
            args.fake_csv = str(auto_csv_dir / "test_fake.csv")
            write_auto_manifest(test_root / "1_fake", Path(args.fake_csv), 1)
        args.flat_two_dir = True
        return args

    real_dir = test_root / "Real" / args.compression / "cropped_faces"
    fake_dir = test_root / args.fake_type_name / args.compression / "cropped_faces"
    if real_dir.is_dir() and fake_dir.is_dir():
        auto_csv_dir = output_dir / f"auto_csv_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if args.real_csv is None:
            args.real_csv = str(auto_csv_dir / "test_real.csv")
            write_auto_manifest(real_dir, Path(args.real_csv), 0)
        if args.fake_csv is None:
            args.fake_csv = str(auto_csv_dir / "test_fake.csv")
            write_auto_manifest(fake_dir, Path(args.fake_csv), 1)
        args.flat_two_dir = False
        return args

    raise RuntimeError(
        "Unsupported --test-root layout. Expected either root/0_real + root/1_fake, or root/Real/<fake>/c23/cropped_faces."
    )


def validate_and_resolve_device_request(args):
    requested_devices = int(max(0, args.devices))
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    if requested_devices > 0 and not cuda_available:
        print("CUDA 不可用，--devices 已自动回退为 0（CPU 模式）。")
        args.devices = 0
    elif requested_devices > cuda_count and cuda_count > 0:
        print(f"请求的 GPU 数量为 {requested_devices}，但当前仅检测到 {cuda_count} 张卡，已自动修正为 {cuda_count}。")
        args.devices = cuda_count
    else:
        args.devices = requested_devices

    return {
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(cuda_count),
        "requested_devices": int(requested_devices),
        "effective_devices": int(args.devices),
    }


def load_cfg(args):
    overrides = [
        f"model.weights_filename={resolve_weights_path(args.weights).as_posix()}",
        f"data.dataset_df.types_val=[Real,{args.fake_type_name}]",
        f"data.dataset_df.ds_type={args.compression}",
        f"data/crop_type={args.crop_type}",
        f"trainer.gpus={max(1, args.devices)}",
        f"num_workers={args.num_workers}",
        f"batch_size={args.batch_size}",
        f"data.dataset_df.cdf_dfdc_test={'True' if args.cdf_dfdc_test else 'False'}",
    ]
    if args.devices < 2:
        overrides.append("trainer.accelerator=null")

    with initialize_config_dir(version_base=None, config_dir=str(STAGE2_ROOT / "conf")):
        cfg = compose(config_name="config_combined", overrides=overrides)

    cfg.gpus = max(1, args.devices)
    if args.devices < 2:
        cfg.trainer.accelerator = None
    return cfg


def _div_255(x):
    return x / 255.0


def _transpose_0_1(x):
    return x.transpose(0, 1)


def build_val_transform(cfg):
    args = cfg.data
    transform = [
        LambdaModule(_div_255),
        CenterCrop(args.crop_type.random_crop_dim),
        Resize(args.crop_type.resize_dim),
    ]

    if args.channel.in_video_channels == 1:
        transform.extend([
            LambdaModule(_transpose_0_1),
            Grayscale(),
            LambdaModule(_transpose_0_1),
        ])

    if args.channel.in_video_channels != 1 and math.isclose(args.channel.grayscale_prob, 1.0):
        transform.extend([
            LambdaModule(_transpose_0_1),
            RandomGrayscale(args.channel.grayscale_prob),
            LambdaModule(_transpose_0_1),
        ])

    transform_aug_extra = []
    if args.channel.in_video_channels != 1 and not math.isclose(args.channel.grayscale_prob, 0.0):
        transform_aug_extra.extend([
            LambdaModule(_transpose_0_1),
            RandomGrayscale(args.channel.grayscale_prob),
            LambdaModule(_transpose_0_1),
        ])

    transform_aug = [
        *deepcopy(transform),
        RandomApply(nn.ModuleList(transform_aug_extra), p=args.aug_prob),
        instantiate(args.channel.obj),
    ]

    if not math.isclose(args.crop_type.random_erasing_prob, 0.0):
        transform_aug.append(
            RandomErasing(
                p=args.crop_type.random_erasing_prob,
                scale=OmegaConf.to_object(args.crop_type.random_erasing_scale),
            )
        )

    transform.append(instantiate(args.channel.obj))

    time_mask_video = None
    if (args.mask_version == "v1" and args.n_time_mask_video != 0) or (
        args.mask_version == "v2" and not math.isclose(args.time_mask_prob_video, 0.0)
    ):
        if args.mask_version == "v1":
            time_mask_video = TimeMask(T=args.time_mask_video, n_mask=args.n_time_mask_video, replace_with_zero=True)
        else:
            time_mask_video = TimeMaskV2(p=args.time_mask_prob_video, T=args.time_mask_video, replace_with_zero=True)

    return Compose((ApplyTransformToKeyAug(Compose(transform), Compose(transform_aug), time_mask_video, args), RemoveKey("audio")))


def build_runtime_model(cfg, device: torch.device):
    model = ModelCombined(cfg)
    weights_path = resolve_weights_path(str(cfg.model.weights_filename))
    state_dict = safe_torch_load_state_dict(weights_path)

    backbone_state = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
    df_head_state = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
    missing_b, unexpected_b = model.backbone.load_state_dict(backbone_state, strict=False)
    missing_h, unexpected_h = model.df_head.load_state_dict(df_head_state, strict=False)
    if missing_b or unexpected_b or missing_h or unexpected_h:
        raise RuntimeError(
            "Checkpoint and model state_dict mismatch. "
            f"backbone missing={missing_b}, backbone unexpected={unexpected_b}, "
            f"head missing={missing_h}, head unexpected={unexpected_h}"
        )
    model.to(device).eval()
    return model


def make_dataloader(ds, cfg, args):
    batch_size = max(1, int(cfg.batch_size // max(1, cfg.gpus)))
    num_workers = max(0, int(args.num_workers // max(1, cfg.gpus)))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(args.devices > 0 and torch.cuda.is_available()),
        drop_last=False,
        sampler=None,
        shuffle=False,
    )


def build_test_loaders(cfg, args):
    offset = 0.04
    transform = build_val_transform(cfg)
    loaders = []

    for ds_type in cfg.data.dataset_df.types_val:
        if cfg.data.dataset_df.only_ff_val and ds_type in ("DeeperForensics", "FaceShifter"):
            continue

        csv_path = Path(args.real_csv) if ds_type == "Real" else Path(args.fake_csv)
        if args.flat_two_dir:
            video_prefix = Path(args.test_root) / ("0_real" if ds_type == "Real" else "1_fake")
        else:
            video_prefix = Path(args.test_root) / ds_type / args.compression / cfg.data.crop_type.video_dir_df

        ds = labeled_video_dataset_with_fix(
            data_path=str(csv_path),
            clip_sampler=UniformClipSamplerWithDuration(
                cfg.data.num_frames / cfg.data.dataset_df.fps - offset,
                110 / cfg.data.dataset_df.fps,
                backpad_last=True,
            ),
            video_path_prefix=str(video_prefix),
            transform=transform,
            video_sampler=torch.utils.data.SequentialSampler,
            with_length=False,
        )
        loaders.append(make_dataloader(ds, cfg, args))

    return loaders


def export_csv(rows, output_path, fieldnames):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_to_python_metrics(metric_dict):
    result = {}
    for key, value in metric_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
        else:
            result[key] = value
    return result


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def compute_binary_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_ranks = np.empty(len(y_score), dtype=np.float64)

    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        sorted_ranks[i:j] = avg_rank
        i = j

    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = sorted_ranks
    auc = (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return safe_float(max(0.0, min(1.0, auc)))


def compute_binary_average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    positive_positions = np.where(y_true_sorted == 1)[0]
    if len(positive_positions) == 0:
        return None

    ap = 0.0
    prev_recall = 0.0
    for idx in positive_positions:
        recall_now = float(recall[idx])
        ap += float(precision[idx]) * max(0.0, recall_now - prev_recall)
        prev_recall = recall_now

    return safe_float(ap)


def compute_binary_metrics_from_rows(rows, prob_key: str, pred_key: str):
    if not rows:
        return {
            "samples_total": 0,
            "positives": 0,
            "negatives": 0,
            "threshold_prob_fake": 0.5,
            "threshold_logit": 0.0,
            "auc": None,
            "ap": None,
            "acc": None,
            "f1": None,
            "fnr": None,
            "fpr": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    y_true = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    y_score = np.asarray([float(row[prob_key]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([int(row[pred_key]) for row in rows], dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = 0.0 if (tp + fp + fn) > 0 else None
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    return {
        "samples_total": int(len(rows)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "threshold_prob_fake": 0.5,
        "threshold_logit": 0.0,
        "auc": compute_binary_auc(y_true, y_score),
        "ap": compute_binary_average_precision(y_true, y_score),
        "acc": safe_div(tp + tn, len(rows)),
        "f1": safe_float(f1),
        "fnr": safe_div(fn, tp + fn),
        "fpr": safe_div(fp, fp + tn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_pair_metrics_summary(clip_rows, video_rows):
    return {
        "clip_level": compute_binary_metrics_from_rows(clip_rows, prob_key="clip_prob_fake", pred_key="clip_pred_label"),
        "video_level": compute_binary_metrics_from_rows(video_rows, prob_key="prob_fake", pred_key="pred_label"),
    }


def export_pair_metrics_summary_csv(pair_metrics_summary, output_path):
    rows = []
    for level_name, metrics in pair_metrics_summary.items():
        rows.append({"level": level_name, **metrics})
    export_csv(
        rows,
        output_path,
        [
            "level",
            "samples_total",
            "positives",
            "negatives",
            "threshold_prob_fake",
            "threshold_logit",
            "auc",
            "ap",
            "acc",
            "f1",
            "fnr",
            "fpr",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )


def build_official_metrics(cfg):
    ff_types = list(cfg.data.dataset_df.types_val)
    if cfg.data.dataset_df.only_ff_val:
        ff_types = [x for x in ff_types if x not in ("DeeperForensics", "FaceShifter")]

    metric_auc_ff = VideoLevelAUROC(ff_types) if ff_types else None
    metric_acc_ff = VideoLevelAcc(ff_types) if ff_types else None
    metric_auc_cdf = VideoLevelAUROCCDF(("Real", "Fake"), multi_gpu=False)
    metric_auc_dfdc = VideoLevelAUROCDFDC(multi_gpu=False)
    return metric_auc_ff, metric_acc_ff, metric_auc_cdf, metric_auc_dfdc


def summarize_current_custom_pair(video_rows, cfg):
    target_groups = set(cfg.data.dataset_df.types_val)
    selected = [row for row in video_rows if row["dataset_group"] in target_groups]
    if not selected:
        return {"videos_total": 0, "videos_real": 0, "videos_fake": 0, "acc_at_logit_0": None}

    videos_total = len(selected)
    videos_real = sum(row["label"] == 0 for row in selected)
    videos_fake = sum(row["label"] == 1 for row in selected)
    correct = sum(int(row["pred_label"] == row["label"]) for row in selected)
    return {
        "videos_total": int(videos_total),
        "videos_real": int(videos_real),
        "videos_fake": int(videos_fake),
        "acc_at_logit_0": float(correct / videos_total),
    }


def make_progress_bar(loader, progress_name: str, dataset_index: int, total_datasets: int):
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    bar = tqdm(
        loader,
        total=total_batches,
        desc=f"[{dataset_index}/{total_datasets}] {progress_name}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        smoothing=0.1,
    )
    return bar, total_batches


def evaluate_with_official_logic(model, loaders, cfg, device, progress_refresh: int = 1):
    metric_auc_ff, metric_acc_ff, metric_auc_cdf, metric_auc_dfdc = build_official_metrics(cfg)

    clip_rows = []
    grouped_video_logits = defaultdict(list)
    grouped_video_meta = {}

    ff_types = list(cfg.data.dataset_df.types_val)
    if cfg.data.dataset_df.only_ff_val:
        ff_types = [x for x in ff_types if x not in ("DeeperForensics", "FaceShifter")]
    num_ff_types = len(ff_types)
    total_datasets = len(loaders)

    with torch.inference_mode():
        for dataloader_idx, loader in enumerate(loaders):
            dataset = loader.dataset
            path_label_pairs = dataset._labeled_videos._paths_and_labels
            total_videos = len(path_label_pairs)

            if dataloader_idx < num_ff_types:
                ds_type_name = ff_types[dataloader_idx]
                metric_ds_type = "Real" if (cfg.data.dataset_df.aggregate_scores and ds_type_name == "Real") else (
                    "FaceForensics" if cfg.data.dataset_df.aggregate_scores else ds_type_name
                )
                progress_name = f"FF {ds_type_name}"
            elif dataloader_idx in (num_ff_types, num_ff_types + 1):
                ds_type_name = "Real" if dataloader_idx == num_ff_types else "Fake"
                metric_ds_type = ds_type_name
                progress_name = f"CelebDF {ds_type_name}"
            else:
                ds_type_name = "DFDC"
                metric_ds_type = None
                progress_name = "DFDC"

            progress_bar, total_batches = make_progress_bar(loader, progress_name, dataloader_idx + 1, total_datasets)
            seen_video_indexes = set()
            clips_seen = 0

            for batch_idx, batch in enumerate(progress_bar, start=1):
                videos = batch["video"].to(device, non_blocking=True)
                labels = batch["label"]
                video_indexes = batch["video_index"]
                clip_indexes = batch["clip_index"]
                logits = model.df_head(model.backbone(videos))

                logits_metric = logits.detach().cpu()
                labels_metric = labels.detach().cpu()
                video_indexes_metric = video_indexes.detach().cpu()

                if dataloader_idx < num_ff_types:
                    metric_auc_ff.update(logits_metric, labels_metric, video_indexes_metric, metric_ds_type)
                    metric_acc_ff.update(logits_metric, labels_metric, video_indexes_metric, metric_ds_type)
                elif dataloader_idx in (num_ff_types, num_ff_types + 1):
                    metric_auc_cdf.update(logits_metric, labels_metric, video_indexes_metric, metric_ds_type)
                else:
                    metric_auc_dfdc.update(logits_metric, labels_metric, video_indexes_metric)

                logits_list = logits_metric.squeeze(-1).tolist()
                labels_list = labels_metric.tolist()
                video_indexes_list = video_indexes_metric.tolist()
                clip_indexes_list = clip_indexes.detach().cpu().tolist()

                clips_seen += len(logits_list)
                seen_video_indexes.update(video_indexes_list)

                for logit, label, video_index, clip_index in zip(logits_list, labels_list, video_indexes_list, clip_indexes_list):
                    prob_fake = float(torch.sigmoid(torch.tensor(logit)).item())
                    clip_pred_label = int(logit > 0.0)
                    video_name = path_label_pairs[video_index][0]
                    clip_rows.append(
                        {
                            "dataset_group": ds_type_name,
                            "metric_group": metric_ds_type if metric_ds_type is not None else "DFDC",
                            "video_index": int(video_index),
                            "video_name": video_name,
                            "label": int(label),
                            "clip_index": int(clip_index),
                            "clip_logit": float(logit),
                            "clip_prob_fake": prob_fake,
                            "clip_pred_label": clip_pred_label,
                            "clip_correct": int(clip_pred_label == int(label)),
                        }
                    )
                    key = (ds_type_name, int(video_index), video_name, int(label))
                    grouped_video_logits[key].append(float(logit))
                    grouped_video_meta[key] = {
                        "dataset_group": ds_type_name,
                        "metric_group": metric_ds_type if metric_ds_type is not None else "DFDC",
                        "video_index": int(video_index),
                        "video_name": video_name,
                        "label": int(label),
                    }

                if progress_refresh > 0 and (batch_idx % progress_refresh == 0 or total_batches is None or batch_idx == total_batches):
                    progress_bar.set_postfix(
                        videos=f"{len(seen_video_indexes)}/{total_videos}",
                        clips=clips_seen,
                        batch=batch_idx if total_batches is None else f"{batch_idx}/{total_batches}",
                    )
            progress_bar.close()

    video_rows = []
    for key in sorted(grouped_video_logits.keys(), key=lambda x: (str(x[0]), int(x[1]), str(x[2]))):
        logits = grouped_video_logits[key]
        meta = grouped_video_meta[key]
        mean_logit = float(sum(logits) / len(logits))
        prob_fake = float(torch.sigmoid(torch.tensor(mean_logit)).item())
        pred_label = int(mean_logit > 0.0)
        video_rows.append({
            **meta,
            "num_clips": int(len(logits)),
            "mean_logit": mean_logit,
            "prob_fake": prob_fake,
            "pred_label": pred_label,
            "video_correct": int(pred_label == int(meta["label"])),
        })

    official_metrics = {}
    if metric_auc_ff is not None and metric_acc_ff is not None:
        official_metrics.update(tensor_to_python_metrics(metric_auc_ff.compute()))
        official_metrics.update(tensor_to_python_metrics(metric_acc_ff.compute()))
    if cfg.data.dataset_df.cdf_dfdc_test:
        official_metrics.update(tensor_to_python_metrics(metric_auc_cdf.compute()))
        official_metrics.update(tensor_to_python_metrics(metric_auc_dfdc.compute()))

    pair_metrics_summary = build_pair_metrics_summary(clip_rows, video_rows)

    return {
        "official_metrics": official_metrics,
        "pair_metrics_summary": pair_metrics_summary,
        "supplemental_custom_pair_summary": summarize_current_custom_pair(video_rows, cfg),
        "clip_rows": clip_rows,
        "video_rows": video_rows,
    }


def main():
    args = parse_args()
    if args.quiet_warnings:
        configure_warnings()
    seed_everything_simple(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = configure_eval_inputs(args, output_dir)
    device_info = validate_and_resolve_device_request(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.devices > 0 else "cpu")
    cfg = load_cfg(args)
    model = build_runtime_model(cfg, device)
    loaders = build_test_loaders(cfg, args)

    print(f"Using device: {device}")
    print(f"Real CSV: {args.real_csv}")
    print(f"Fake CSV: {args.fake_csv}")
    print(f"Datasets to evaluate: {cfg.data.dataset_df.types_val}")

    results = evaluate_with_official_logic(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
        progress_refresh=max(1, args.progress_refresh),
    )

    result_json = {
        "weights": str(resolve_weights_path(args.weights)),
        "test_root": str(Path(args.test_root).resolve()),
        "real_csv": str(Path(args.real_csv).resolve()),
        "fake_csv": str(Path(args.fake_csv).resolve()),
        "fake_type_name": args.fake_type_name,
        "settings": {
            "devices": args.devices,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "compression": args.compression,
            "crop_type": args.crop_type,
            "seed": args.seed,
            "cdf_dfdc_test": bool(args.cdf_dfdc_test),
            "progress_refresh": int(max(1, args.progress_refresh)),
            "quiet_warnings": bool(args.quiet_warnings),
        },
        "device_info": device_info,
        "data_summary": {
            "videos_total": len(results["video_rows"]),
            "videos_real": sum(row["label"] == 0 for row in results["video_rows"]),
            "videos_fake": sum(row["label"] == 1 for row in results["video_rows"]),
            "clips_total": len(results["clip_rows"]),
        },
        "official_metrics": results["official_metrics"],
        "pair_metrics_summary": results["pair_metrics_summary"],
        "supplemental_custom_pair_summary": results["supplemental_custom_pair_summary"],
    }

    export_csv(
        results["clip_rows"],
        output_dir / "clip_scores.csv",
        [
            "dataset_group",
            "metric_group",
            "video_index",
            "video_name",
            "label",
            "clip_index",
            "clip_logit",
            "clip_prob_fake",
            "clip_pred_label",
            "clip_correct",
        ],
    )
    export_csv(
        results["video_rows"],
        output_dir / "video_scores.csv",
        [
            "dataset_group",
            "metric_group",
            "video_index",
            "video_name",
            "label",
            "num_clips",
            "mean_logit",
            "prob_fake",
            "pred_label",
            "video_correct",
        ],
    )
    export_pair_metrics_summary_csv(results["pair_metrics_summary"], output_dir / "pair_metrics_summary.csv")
    (output_dir / "metrics.json").write_text(json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result_json, indent=2, ensure_ascii=False))
    print(f"clip_scores.csv 已保存到: {output_dir / 'clip_scores.csv'}")
    print(f"video_scores.csv 已保存到: {output_dir / 'video_scores.csv'}")
    print(f"pair_metrics_summary.csv 已保存到: {output_dir / 'pair_metrics_summary.csv'}")
    print(f"metrics.json 已保存到: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
