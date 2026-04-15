import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_ROOT = REPO_ROOT / "stage2"
for path in (str(REPO_ROOT), str(STAGE2_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from stage2.combined_learner import CombinedLearner
from stage2.data.combined_dm import DataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a custom dataset with RealForensics official logic.")
    parser.add_argument("--weights", required=True, help="Path to stage2 weights file.")
    parser.add_argument("--test-root", required=True, help="Prepared dataset root with Real/<ds>/c23/cropped_faces.")
    parser.add_argument("--real-csv", required=True, help="CSV manifest for real videos.")
    parser.add_argument("--fake-csv", required=True, help="CSV manifest for fake videos.")
    parser.add_argument("--fake-type-name", required=True, help="Dataset type name used under test root, e.g. LAVDF.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and score CSV files.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPU devices to use.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size override.")
    parser.add_argument("--compression", default="c23", help="Compression split name.")
    parser.add_argument("--crop-type", default="full_face", help="Hydra crop_type config option.")
    return parser.parse_args()


def load_cfg(args):
    overrides = [
        f"model.weights_filename={Path(args.weights).as_posix()}",
        f"data.dataset_df.types_val=[Real,{args.fake_type_name}]",
        f"data.dataset_df.test_root={Path(args.test_root).as_posix()}",
        f"data.dataset_df.test_real_csv={Path(args.real_csv).as_posix()}",
        f"data.dataset_df.test_fake_csv={Path(args.fake_csv).as_posix()}",
        f"data.dataset_df.ds_type={args.compression}",
        f"data/crop_type={args.crop_type}",
        f"trainer.gpus={args.devices}",
        "trainer.accelerator=null",
        f"num_workers={args.num_workers}",
        f"batch_size={args.batch_size}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(STAGE2_ROOT / "conf")):
        cfg = compose(config_name="config_combined", overrides=overrides)
    cfg.gpus = args.devices
    return cfg


def load_model_and_data(cfg):
    learner = CombinedLearner(cfg)
    if cfg.model.weights_filename:
        weights_path = Path(cfg.model.weights_filename)
        state_dict = torch.load(weights_path, map_location="cpu")
        weights_backbone = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
        learner.model.backbone.load_state_dict(weights_backbone)
        weights_df_head = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
        learner.model.df_head.load_state_dict(weights_df_head)

    dm = DataModule(cfg, root=str(REPO_ROOT))
    loaders = dm.test_dataloader()
    return learner, loaders


def compute_scalar_metrics(video_rows):
    y_true = [row["label"] for row in video_rows]
    y_score = [row["prob_fake"] for row in video_rows]
    y_pred = [row["pred_label"] for row in video_rows]

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


def export_csv(rows, output_path, fieldnames):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args)
    learner, loaders = load_model_and_data(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and args.devices > 0 else "cpu")
    learner.model.to(device).eval()

    clip_rows = []
    grouped = defaultdict(list)
    ds_types = cfg.data.dataset_df.types_val

    with torch.inference_mode():
        for dataloader_idx, loader in enumerate(loaders):
            ds_type = ds_types[dataloader_idx]
            dataset = loader.dataset
            path_label_pairs = dataset._labeled_videos._paths_and_labels

            for batch in tqdm(loader, desc=f"Official {ds_type}"):
                videos = batch["video"].to(device, non_blocking=True)
                logits = learner.model.df_head(learner.model.backbone(videos)).squeeze(-1).detach().cpu().tolist()
                labels = batch["label"].tolist()
                video_indexes = batch["video_index"].tolist()
                clip_indexes = batch["clip_index"].tolist()

                for logit, label, video_index, clip_index in zip(logits, labels, video_indexes, clip_indexes):
                    video_name = path_label_pairs[video_index][0]
                    clip_row = {
                        "ds_type": ds_type,
                        "video_index": int(video_index),
                        "video_name": video_name,
                        "label": int(label),
                        "clip_index": int(clip_index),
                        "clip_logit": float(logit),
                    }
                    clip_rows.append(clip_row)
                    grouped[(ds_type, int(video_index), video_name, int(label))].append(float(logit))

    video_rows = []
    for (ds_type, video_index, video_name, label), logits in grouped.items():
        mean_logit = float(sum(logits) / len(logits))
        prob_fake = float(torch.sigmoid(torch.tensor(mean_logit)).item())
        video_rows.append(
            {
                "ds_type": ds_type,
                "video_index": video_index,
                "video_name": video_name,
                "label": label,
                "num_clips": len(logits),
                "mean_logit": mean_logit,
                "prob_fake": prob_fake,
                "pred_label": int(mean_logit > 0.0),
            }
        )
    video_rows.sort(key=lambda row: (row["ds_type"], row["video_index"]))

    metrics = compute_scalar_metrics(video_rows)
    result = {
        "weights": str(Path(args.weights)),
        "test_root": str(Path(args.test_root)),
        "fake_type_name": args.fake_type_name,
        "settings": {
            "devices": args.devices,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "compression": args.compression,
            "crop_type": args.crop_type,
        },
        "data_summary": {
            "videos_total": len(video_rows),
            "videos_real": sum(row["label"] == 0 for row in video_rows),
            "videos_fake": sum(row["label"] == 1 for row in video_rows),
            "clips_total": len(clip_rows),
        },
        "metrics": metrics,
    }

    export_csv(
        clip_rows,
        output_dir / "clip_scores.csv",
        ["ds_type", "video_index", "video_name", "label", "clip_index", "clip_logit"],
    )
    export_csv(
        video_rows,
        output_dir / "video_scores.csv",
        ["ds_type", "video_index", "video_name", "label", "num_clips", "mean_logit", "prob_fake", "pred_label"],
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
