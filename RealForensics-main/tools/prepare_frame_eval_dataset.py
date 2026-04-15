import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert per-video frame folders into RealForensics-compatible test videos and CSVs."
    )
    parser.add_argument("--input-root", required=True, help="Root folder with 0_real and 1_fake frame folders.")
    parser.add_argument("--output-root", required=True, help="Output root for the converted dataset.")
    parser.add_argument("--fake-type-name", default="FakeAVCeleb", help="Folder name to use for the fake class.")
    parser.add_argument("--compression-name", default="c23", help="Compression split name used by the eval config.")
    parser.add_argument("--video-dir-name", default="cropped_faces", help="Video directory name used by the eval config.")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS assigned to the generated AVI files.")
    parser.add_argument("--image-pattern", default="frame_*.jpg", help="Glob pattern for frame files inside each clip.")
    parser.add_argument("--codec", default="FFV1", choices=("FFV1", "MJPG"), help="Preferred AVI codec.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit per split for quick smoke tests.")
    return parser.parse_args()


def iter_clip_dirs(split_dir, limit):
    clip_dirs = [path for path in sorted(split_dir.iterdir()) if path.is_dir()]
    if limit > 0:
        clip_dirs = clip_dirs[:limit]
    return clip_dirs


def make_writer(output_path, size, fps, codec):
    codec_candidates = [codec]
    if codec != "MJPG":
        codec_candidates.append("MJPG")

    for candidate in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*candidate),
            fps,
            size
        )
        if writer.isOpened():
            return writer, candidate
        writer.release()

    raise RuntimeError(f"Could not create a video writer for {output_path}.")


def convert_clip(clip_dir, output_path, fps, image_pattern, codec):
    frame_paths = sorted(clip_dir.glob(image_pattern))
    if not frame_paths:
        raise RuntimeError(f"No frames matching {image_pattern} found in {clip_dir}.")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read {frame_paths[0]}.")

    size = (first_frame.shape[1], first_frame.shape[0])
    writer, used_codec = make_writer(output_path, size, fps, codec)
    try:
        writer.write(first_frame)
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read {frame_path}.")
            if (frame.shape[1], frame.shape[0]) != size:
                raise RuntimeError(f"Inconsistent frame size in {clip_dir}.")
            writer.write(frame)
    finally:
        writer.release()

    return used_codec


def build_split(input_split_dir, output_split_dir, label, csv_path, fps, image_pattern, codec, limit):
    output_split_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    used_codec = None

    for clip_dir in tqdm(iter_clip_dirs(input_split_dir, limit), desc=input_split_dir.name):
        output_name = f"{clip_dir.name}.avi"
        output_path = output_split_dir / output_name
        used_codec = convert_clip(clip_dir, output_path, fps, image_pattern, codec)
        lines.append(f"{output_name} {label}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return used_codec, len(lines)


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    real_input = input_root / "0_real"
    fake_input = input_root / "1_fake"
    if not real_input.is_dir() or not fake_input.is_dir():
        raise RuntimeError("Expected input root to contain 0_real and 1_fake directories.")

    real_output = output_root / "Real" / args.compression_name / args.video_dir_name
    fake_output = output_root / args.fake_type_name / args.compression_name / args.video_dir_name
    csv_root = output_root / "csv_files"

    real_codec, real_count = build_split(
        real_input,
        real_output,
        label=0,
        csv_path=csv_root / "test_real.csv",
        fps=args.fps,
        image_pattern=args.image_pattern,
        codec=args.codec,
        limit=args.limit
    )
    fake_codec, fake_count = build_split(
        fake_input,
        fake_output,
        label=1,
        csv_path=csv_root / "test_fake.csv",
        fps=args.fps,
        image_pattern=args.image_pattern,
        codec=args.codec,
        limit=args.limit
    )

    print(f"Prepared {real_count} real videos and {fake_count} fake videos.")
    print(f"Output root: {output_root}")
    print(f"Video codec used: real={real_codec}, fake={fake_codec}")
    print(f"Eval types_val: [Real,{args.fake_type_name}]")


if __name__ == "__main__":
    main()
