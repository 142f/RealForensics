import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.models.backbones.csn import csn_temporal_no_head
from stage2.models.linear import MeanLinear


MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    label: int
    subtype: str
    frame_paths: tuple[Path, ...]
    num_frames_original: int
    num_frames_used: int


@dataclass(frozen=True)
class ClipRecord:
    video_id: str
    label: int
    subtype: str
    clip_index: int
    frame_paths: tuple[Path, ...]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RealForensics weights on per-video frame folders.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing 0_real and 1_fake folders.")
    parser.add_argument("--weights", required=True, help="Path to a RealForensics stage2 .pth checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to write metrics and prediction CSV files.")
    parser.add_argument("--real-split-name", default="0_real", help="Directory name for the real split.")
    parser.add_argument("--fake-split-name", default="1_fake", help="Directory name for the fake split.")
    parser.add_argument("--limit-real", type=int, default=0, help="Optional cap on the number of real videos.")
    parser.add_argument("--limit-fake", type=int, default=0, help="Optional cap on the number of fake videos.")
    parser.add_argument("--num-frames", type=int, default=25, help="Frames per clip.")
    parser.add_argument("--max-frames", type=int, default=110, help="Maximum frames to use per video.")
    parser.add_argument("--crop-size", type=int, default=140, help="Center crop size before resize.")
    parser.add_argument("--resize", type=int, default=112, help="Final spatial size.")
    parser.add_argument("--batch-size", type=int, default=4, help="Clip batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    return parser.parse_args()


def center_crop_tensor(x: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, h, w = x.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return x[:, :, top:top + crop_h, left:left + crop_w]


def load_frame(frame_path: Path) -> torch.Tensor:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).permute(2, 0, 1).float()


class FrameClipDataset(Dataset):
    def __init__(self, clips: list[ClipRecord], crop_size: int, resize: int, num_frames: int):
        self.clips = clips
        self.crop_size = crop_size
        self.resize = resize
        self.num_frames = num_frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        frames = [load_frame(path) for path in clip.frame_paths]
        valid_frames = min(len(frames), self.num_frames)
        video = torch.stack(frames[:valid_frames], dim=1) / 255.0
        video = center_crop_tensor(video, self.crop_size)
        video = F.interpolate(
            video.permute(1, 0, 2, 3),
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False
        ).permute(1, 0, 2, 3)
        video = (video - MEAN) / STD
        if video.size(1) < self.num_frames:
            padded = torch.zeros((video.size(0), self.num_frames, video.size(2), video.size(3)), dtype=video.dtype)
            padded[:, :video.size(1), ...] = video
            video = padded

        return {
            "video": video,
            "label": torch.tensor(clip.label, dtype=torch.long),
            "video_id": clip.video_id,
            "clip_index": torch.tensor(clip.clip_index, dtype=torch.long),
            "subtype": clip.subtype,
        }


def discover_videos(
    dataset_root: Path,
    max_frames: int,
    real_split_name: str,
    fake_split_name: str,
    limit_real: int,
    limit_fake: int,
) -> list[VideoRecord]:
    split_to_label = {real_split_name: 0, fake_split_name: 1}
    split_limits = {real_split_name: limit_real, fake_split_name: limit_fake}
    videos = []

    for split_name, label in split_to_label.items():
        split_dir = dataset_root / split_name
        if not split_dir.is_dir():
            raise RuntimeError(f"Missing split directory: {split_dir}")

        video_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        limit = split_limits[split_name]
        if limit > 0:
            video_dirs = video_dirs[:limit]

        for video_dir in video_dirs:
            frame_paths = tuple(sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png")))
            if not frame_paths:
                continue

            used_paths = frame_paths[:max_frames]
            subtype = video_dir.name.rsplit("_", 1)[-1] if "_" in video_dir.name else split_name
            # Use split-prefixed IDs so same folder names in real/fake do not collide.
            unique_video_id = f"{split_name}/{video_dir.name}"
            videos.append(
                VideoRecord(
                    video_id=unique_video_id,
                    label=label,
                    subtype=subtype,
                    frame_paths=used_paths,
                    num_frames_original=len(frame_paths),
                    num_frames_used=len(used_paths),
                )
            )

    return videos


def build_clips(videos: list[VideoRecord], num_frames: int) -> list[ClipRecord]:
    clips = []
    for video in videos:
        num_clips = max(1, math.ceil(video.num_frames_used / num_frames))
        for clip_index in range(num_clips):
            start = clip_index * num_frames
            end = min(start + num_frames, video.num_frames_used)
            clip_paths = video.frame_paths[start:end]
            if not clip_paths:
                clip_paths = (video.frame_paths[-1],)
            clips.append(
                ClipRecord(
                    video_id=video.video_id,
                    label=video.label,
                    subtype=video.subtype,
                    clip_index=clip_index,
                    frame_paths=clip_paths,
                )
            )
    return clips


def load_model(weights_path: Path, device: torch.device):
    backbone = csn_temporal_no_head(model_depth=101, input_channel=3)
    df_head = MeanLinear(in_dim=2048, out_dim=1, norm_linear=True, scale=64)

    state_dict = torch.load(weights_path, map_location="cpu")
    backbone_state = {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    head_state = {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith("df_head.")}
    backbone.load_state_dict(backbone_state, strict=True)
    df_head.load_state_dict(head_state, strict=True)

    backbone.to(device).eval()
    df_head.to(device).eval()
    return backbone, df_head


def evaluate(backbone, df_head, loader, device: torch.device):
    logits_by_video = defaultdict(list)
    labels_by_video = {}
    subtype_by_video = {}

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating clips"):
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["label"]
            video_ids = batch["video_id"]
            subtypes = batch["subtype"]

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = df_head(backbone(videos)).squeeze(-1)

            for logit, label, video_id, subtype in zip(logits.detach().cpu(), labels, video_ids, subtypes):
                logits_by_video[video_id].append(float(logit))
                labels_by_video[video_id] = int(label)
                subtype_by_video[video_id] = subtype

    rows = []
    for video_id, clip_logits in sorted(logits_by_video.items()):
        mean_logit = float(sum(clip_logits) / len(clip_logits))
        probability = float(torch.sigmoid(torch.tensor(mean_logit)).item())
        rows.append(
            {
                "video_id": video_id,
                "label": labels_by_video[video_id],
                "subtype": subtype_by_video[video_id],
                "num_clips": len(clip_logits),
                "mean_logit": mean_logit,
                "prob_fake": probability,
                "pred_label": int(mean_logit > 0.0),
            }
        )
    return rows


def compute_metrics(rows: list[dict]):
    y_true = [row["label"] for row in rows]
    y_score = [row["prob_fake"] for row in rows]
    y_pred = [row["pred_label"] for row in rows]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
    }


def metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (torch.tensor(y_score) >= threshold).long().numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
    }


def threshold_sensitivity_analysis(rows: list[dict]):
    y_true = [row["label"] for row in rows]
    y_score = [row["prob_fake"] for row in rows]
    thresholds = sorted(set([0.0, 1.0, *y_score]))

    best_f1 = None
    best_balanced = None
    best_youden = None

    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_score, threshold)
        youden_j = metrics["recall"] + metrics["specificity"] - 1.0

        if best_f1 is None or (
            metrics["f1"], metrics["balanced_acc"], metrics["acc"]
        ) > (
            best_f1["f1"], best_f1["balanced_acc"], best_f1["acc"]
        ):
            best_f1 = metrics

        if best_balanced is None or (
            metrics["balanced_acc"], metrics["f1"], metrics["acc"]
        ) > (
            best_balanced["balanced_acc"], best_balanced["f1"], best_balanced["acc"]
        ):
            best_balanced = metrics

        if best_youden is None or youden_j > best_youden["youden_j"]:
            best_youden = {**metrics, "youden_j": float(youden_j)}

    return {
        "warning": "Thresholds below are optimized on the evaluated set and are suitable only for sensitivity analysis, not as the main paper result.",
        "best_f1_on_test": best_f1,
        "best_balanced_acc_on_test": best_balanced,
        "best_youden_j_on_test": best_youden,
    }


def bootstrap_confidence_intervals(rows: list[dict], threshold_for_thresholded_metrics, num_bootstrap=3000, seed=42):
    y_true = torch.tensor([row["label"] for row in rows], dtype=torch.long).numpy()
    y_score = torch.tensor([row["prob_fake"] for row in rows], dtype=torch.float32).numpy()
    rng = torch.Generator().manual_seed(seed)

    auc_vals = []
    ap_vals = []
    acc_vals = []
    f1_vals = []
    bacc_vals = []

    n = len(rows)
    for _ in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), generator=rng).numpy()
        yb = y_true[idx]
        sb = y_score[idx]
        if len(set(yb.tolist())) < 2:
            continue

        auc_vals.append(float(roc_auc_score(yb, sb)))
        ap_vals.append(float(average_precision_score(yb, sb)))
        pred_b = (sb >= threshold_for_thresholded_metrics).astype(int)
        acc_vals.append(float(accuracy_score(yb, pred_b)))
        f1_vals.append(float(f1_score(yb, pred_b, zero_division=0)))
        bacc_vals.append(float(balanced_accuracy_score(yb, pred_b)))

    def summarize(values):
        values_t = torch.tensor(values, dtype=torch.float32)
        return {
            "mean": float(values_t.mean().item()),
            "ci95": [
                float(torch.quantile(values_t, 0.025).item()),
                float(torch.quantile(values_t, 0.975).item()),
            ],
        }

    return {
        "num_bootstrap": num_bootstrap,
        "threshold_for_thresholded_metrics": float(threshold_for_thresholded_metrics),
        "auc": summarize(auc_vals),
        "ap": summarize(ap_vals),
        "acc": summarize(acc_vals),
        "f1": summarize(f1_vals),
        "balanced_acc": summarize(bacc_vals),
    }


def summarize_data(videos: list[VideoRecord], rows: list[dict]):
    num_videos_real = sum(video.label == 0 for video in videos)
    num_videos_fake = sum(video.label == 1 for video in videos)
    num_videos_total = len(videos)
    num_clips_total = sum(row["num_clips"] for row in rows)
    num_clips_real = sum(row["num_clips"] for row in rows if row["label"] == 0)
    num_clips_fake = sum(row["num_clips"] for row in rows if row["label"] == 1)
    num_frames_original = sum(video.num_frames_original for video in videos)
    num_frames_used = sum(video.num_frames_used for video in videos)
    subtype_counter = Counter(video.subtype for video in videos)

    return {
        "videos": {
            "total": num_videos_total,
            "real": num_videos_real,
            "fake": num_videos_fake,
            "real_ratio": num_videos_real / num_videos_total if num_videos_total else 0.0,
            "fake_ratio": num_videos_fake / num_videos_total if num_videos_total else 0.0,
        },
        "clips": {
            "total": num_clips_total,
            "real": num_clips_real,
            "fake": num_clips_fake,
            "real_ratio": num_clips_real / num_clips_total if num_clips_total else 0.0,
            "fake_ratio": num_clips_fake / num_clips_total if num_clips_total else 0.0,
        },
        "frames": {
            "total_original": num_frames_original,
            "total_used": num_frames_used,
            "avg_original_per_video": num_frames_original / num_videos_total if num_videos_total else 0.0,
            "avg_used_per_video": num_frames_used / num_videos_total if num_videos_total else 0.0,
        },
        "subtypes": dict(sorted(subtype_counter.items())),
    }


def write_predictions(rows: list[dict], videos: list[VideoRecord], output_csv: Path):
    video_meta = {video.video_id: video for video in videos}
    fieldnames = [
        "video_id",
        "label",
        "subtype",
        "num_frames_original",
        "num_frames_used",
        "num_clips",
        "mean_logit",
        "prob_fake",
        "pred_label",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = video_meta[row["video_id"]]
            writer.writerow(
                {
                    **row,
                    "num_frames_original": meta.num_frames_original,
                    "num_frames_used": meta.num_frames_used,
                }
            )


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    videos = discover_videos(
        dataset_root,
        max_frames=args.max_frames,
        real_split_name=args.real_split_name,
        fake_split_name=args.fake_split_name,
        limit_real=args.limit_real,
        limit_fake=args.limit_fake,
    )
    if not videos:
        raise RuntimeError("No videos discovered in the dataset root.")
    clips = build_clips(videos, num_frames=args.num_frames)

    dataset = FrameClipDataset(
        clips=clips,
        crop_size=args.crop_size,
        resize=args.resize,
        num_frames=args.num_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    backbone, df_head = load_model(weights_path, device)
    rows = evaluate(backbone, df_head, loader, device)
    metrics = compute_metrics(rows)
    data_summary = summarize_data(videos, rows)
    threshold_analysis = threshold_sensitivity_analysis(rows)
    bootstrap = bootstrap_confidence_intervals(
        rows,
        threshold_for_thresholded_metrics=threshold_analysis["best_youden_j_on_test"]["threshold"]
    )

    result = {
        "dataset_root": str(dataset_root),
        "weights": str(weights_path),
        "device": str(device),
        "settings": {
            "real_split_name": args.real_split_name,
            "fake_split_name": args.fake_split_name,
            "limit_real": args.limit_real,
            "limit_fake": args.limit_fake,
            "num_frames": args.num_frames,
            "max_frames": args.max_frames,
            "crop_size": args.crop_size,
            "resize": args.resize,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "data_summary": data_summary,
        "metrics_default_threshold": metrics,
        "threshold_sensitivity_analysis": threshold_analysis,
        "bootstrap_confidence_intervals": bootstrap,
    }

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_predictions(rows, videos, predictions_path)

    print(json.dumps(result, indent=2))
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
