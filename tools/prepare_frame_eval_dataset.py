import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ClipTask:
    split_name: str
    clip_dir: str
    output_path: str
    label: int
    fps: float
    codec: str
    image_patterns: tuple[str, ...]


def natural_key(path_like) -> list:
    text = Path(path_like).name if not isinstance(path_like, str) else path_like
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def parse_patterns(pattern_text: str) -> tuple[str, ...]:
    parts = [item.strip() for item in pattern_text.split(",") if item.strip()]
    if not parts:
        raise ValueError("--image-patterns 不能为空。")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert per-video frame folders into RealForensics-compatible AVI videos and CSV manifests."
    )
    parser.add_argument("--input-root", required=True, help="Root folder with 0_real and 1_fake frame folders.")
    parser.add_argument("--output-root", required=True, help="Output root for the converted dataset.")
    parser.add_argument("--fake-type-name", default="FakeAVCeleb", help="Folder name to use for the fake class.")
    parser.add_argument("--compression-name", default="c23", help="Compression split name used by the eval config.")
    parser.add_argument("--video-dir-name", default="cropped_faces", help="Video directory name used by the eval config.")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS assigned to the generated AVI files.")
    parser.add_argument(
        "--image-patterns",
        default="*.jpg,*.jpeg,*.png",
        help="Comma-separated glob patterns searched inside each clip folder, for example '*.jpg,*.png'.",
    )
    parser.add_argument("--codec", default="FFV1", choices=("FFV1", "MJPG"), help="Preferred AVI codec.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit per split for quick smoke tests.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of CPU worker processes for conversion.")
    parser.add_argument(
        "--allow-mixed-extensions",
        action="store_true",
        help="Allow a clip folder to contain multiple image extensions. Disabled by default to surface hidden data issues.",
    )
    parser.add_argument(
        "--skip-broken",
        action="store_true",
        help="Skip broken clips instead of stopping the whole run. Broken clip details will be written to failures.csv.",
    )
    return parser.parse_args()


def iter_clip_dirs(split_dir: Path, limit: int) -> list[Path]:
    clip_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    clip_dirs.sort(key=natural_key)
    if limit > 0:
        clip_dirs = clip_dirs[:limit]
    return clip_dirs


def find_frame_paths(clip_dir: Path, image_patterns: tuple[str, ...]) -> list[Path]:
    seen = set()
    frame_paths = []
    for pattern in image_patterns:
        for path in clip_dir.glob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            path_str = str(path.resolve())
            if path_str in seen:
                continue
            seen.add(path_str)
            frame_paths.append(path)
    frame_paths.sort(key=natural_key)
    return frame_paths


def make_writer(output_path: Path, size: tuple[int, int], fps: float, codec: str):
    codec_candidates = [codec]
    if codec != "MJPG":
        codec_candidates.append("MJPG")

    for candidate in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*candidate),
            fps,
            size,
        )
        if writer.isOpened():
            return writer, candidate
        writer.release()

    raise RuntimeError(f"Could not create a video writer for {output_path}.")


def validate_extensions(frame_paths: list[Path], allow_mixed_extensions: bool):
    suffixes = {path.suffix.lower() for path in frame_paths}
    if len(suffixes) > 1 and not allow_mixed_extensions:
        raise RuntimeError(
            "Mixed image extensions found in one clip folder. "
            f"Please normalize extensions first or pass --allow-mixed-extensions. Extensions: {sorted(suffixes)}"
        )


def convert_clip(task: ClipTask, allow_mixed_extensions: bool):
    clip_dir = Path(task.clip_dir)
    output_path = Path(task.output_path)
    frame_paths = find_frame_paths(clip_dir, task.image_patterns)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {clip_dir} for patterns {task.image_patterns}.")

    validate_extensions(frame_paths, allow_mixed_extensions)

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    size = (first_frame.shape[1], first_frame.shape[0])
    writer, used_codec = make_writer(output_path, size, task.fps, task.codec)
    frame_count = 0
    try:
        writer.write(first_frame)
        frame_count += 1
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if (frame.shape[1], frame.shape[0]) != size:
                raise RuntimeError(
                    f"Inconsistent frame size in {clip_dir}: expected {size[::-1]}, got {(frame.shape[0], frame.shape[1])} at {frame_path.name}"
                )
            writer.write(frame)
            frame_count += 1
    finally:
        writer.release()

    return {
        "split_name": task.split_name,
        "clip_dir": clip_dir.name,
        "output_name": output_path.name,
        "label": task.label,
        "frame_count": frame_count,
        "height": size[1],
        "width": size[0],
        "fps": task.fps,
        "codec": used_codec,
    }


def submit_tasks(
    clip_dirs: Iterable[Path],
    split_name: str,
    output_split_dir: Path,
    label: int,
    fps: float,
    codec: str,
    image_patterns: tuple[str, ...],
) -> list[ClipTask]:
    tasks = []
    for clip_dir in clip_dirs:
        output_name = f"{clip_dir.name}.avi"
        output_path = output_split_dir / output_name
        tasks.append(
            ClipTask(
                split_name=split_name,
                clip_dir=str(clip_dir),
                output_path=str(output_path),
                label=label,
                fps=fps,
                codec=codec,
                image_patterns=image_patterns,
            )
        )
    return tasks


def process_split(
    input_split_dir: Path,
    output_split_dir: Path,
    label: int,
    fps: float,
    image_patterns: tuple[str, ...],
    codec: str,
    limit: int,
    num_workers: int,
    allow_mixed_extensions: bool,
    skip_broken: bool,
):
    clip_dirs = iter_clip_dirs(input_split_dir, limit)
    if not clip_dirs:
        raise RuntimeError(f"No clip directories found in {input_split_dir}.")

    output_split_dir.mkdir(parents=True, exist_ok=True)
    tasks = submit_tasks(
        clip_dirs=clip_dirs,
        split_name=input_split_dir.name,
        output_split_dir=output_split_dir,
        label=label,
        fps=fps,
        codec=codec,
        image_patterns=image_patterns,
    )

    successes = []
    failures = []

    if num_workers <= 1:
        iterator = tasks
        for task in tqdm(iterator, desc=input_split_dir.name):
            try:
                successes.append(convert_clip(task, allow_mixed_extensions))
            except Exception as exc:
                error = {"split_name": input_split_dir.name, "clip_dir": Path(task.clip_dir).name, "error": str(exc)}
                if skip_broken:
                    failures.append(error)
                    continue
                raise
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(convert_clip, task, allow_mixed_extensions): task for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc=input_split_dir.name):
                task = future_to_task[future]
                try:
                    successes.append(future.result())
                except Exception as exc:
                    error = {"split_name": input_split_dir.name, "clip_dir": Path(task.clip_dir).name, "error": str(exc)}
                    if skip_broken:
                        failures.append(error)
                        continue
                    raise

    successes.sort(key=lambda row: natural_key(row["clip_dir"]))
    return successes, failures


def write_manifest(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row['output_name']} {row['label']}" for row in rows]
    csv_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_details_csv(csv_path: Path, rows: list[dict], fieldnames: list[str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    image_patterns = parse_patterns(args.image_patterns)

    real_input = input_root / "0_real"
    fake_input = input_root / "1_fake"
    if not real_input.is_dir() or not fake_input.is_dir():
        raise RuntimeError("Expected input root to contain 0_real and 1_fake directories.")

    real_output = output_root / "Real" / args.compression_name / args.video_dir_name
    fake_output = output_root / args.fake_type_name / args.compression_name / args.video_dir_name
    csv_root = output_root / "csv_files"

    real_rows, real_failures = process_split(
        input_split_dir=real_input,
        output_split_dir=real_output,
        label=0,
        fps=args.fps,
        image_patterns=image_patterns,
        codec=args.codec,
        limit=args.limit,
        num_workers=args.num_workers,
        allow_mixed_extensions=args.allow_mixed_extensions,
        skip_broken=args.skip_broken,
    )
    fake_rows, fake_failures = process_split(
        input_split_dir=fake_input,
        output_split_dir=fake_output,
        label=1,
        fps=args.fps,
        image_patterns=image_patterns,
        codec=args.codec,
        limit=args.limit,
        num_workers=args.num_workers,
        allow_mixed_extensions=args.allow_mixed_extensions,
        skip_broken=args.skip_broken,
    )

    write_manifest(csv_root / "test_real.csv", real_rows)
    write_manifest(csv_root / "test_fake.csv", fake_rows)
    write_details_csv(
        output_root / "conversion_details.csv",
        real_rows + fake_rows,
        ["split_name", "clip_dir", "output_name", "label", "frame_count", "height", "width", "fps", "codec"],
    )

    all_failures = real_failures + fake_failures
    if all_failures:
        write_details_csv(output_root / "failures.csv", all_failures, ["split_name", "clip_dir", "error"])

    print(f"Prepared {len(real_rows)} real videos and {len(fake_rows)} fake videos.")
    print(f"Output root: {output_root}")
    print(f"Eval types_val: [Real,{args.fake_type_name}]")
    print(f"Image patterns: {image_patterns}")
    print(f"CPU workers: {args.num_workers}")
    if all_failures:
        print(f"Skipped broken clips: {len(all_failures)} (see {output_root / 'failures.csv'})")


if __name__ == "__main__":
    main()
