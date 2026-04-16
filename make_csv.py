from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Optional

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aac"}


def parse_transcription_line(line: str) -> Optional[tuple[str, str]]:
    """
    Expected formats per line (common cases):
      1) 123456 some transcription here
      2) 123456|some transcription here
      3) 123456\t some transcription here

    Returns: (audio_id, transcription) or None if the line can't be parsed.
    """
    line = line.strip()
    if not line:
        return None

    # Try separators first
    for sep in ["|", "\t"]:
        if sep in line:
            left, right = line.split(sep, 1)
            audio_id = left.strip()
            transcription = right.strip()
            if audio_id and transcription:
                return audio_id, transcription

    # Fallback: first token = id, rest = transcription
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None
    audio_id, transcription = parts[0].strip(), parts[1].strip()
    if not audio_id or not transcription:
        return None
    return audio_id, transcription


def build_audio_index(root: Path, exts: set[str]) -> dict[str, Path]:
    """
    Index audio files by their stem (filename without extension).
    Example: /data/.../12345.wav -> key "12345"
    If duplicates exist, first one wins (you can change this).
    """
    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            stem = p.stem
            if stem not in index:
                index[stem] = p
    return index


def find_txt_files(root: Path) -> Iterable[Path]:
    """Find all .txt files under root."""
    return root.rglob("*.txt")


def main(
    root_dir: str,
    output_csv: str = "dataset.csv",
    strict: bool = False,
) -> None:
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    # 1) Index all audio files once (fast lookup)
    audio_index = build_audio_index(root, AUDIO_EXTS)

    rows = []
    missing_audio = 0
    bad_lines = 0

    # 2) Read every txt and map ids -> audio paths
    for txt_path in find_txt_files(root):
        with txt_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                parsed = parse_transcription_line(line)
                if parsed is None:
                    bad_lines += 1
                    if strict:
                        raise ValueError(f"Bad line in {txt_path} at {line_no}: {line!r}")
                    continue

                audio_id, transcription = parsed

                # Sometimes audio_id may include extension; remove it if present
                audio_id = Path(audio_id).stem

                audio_path = audio_index.get(audio_id)
                if audio_path is None:
                    missing_audio += 1
                    if strict:
                        raise FileNotFoundError(
                            f"Audio not found for id={audio_id} (from {txt_path}:{line_no})"
                        )
                    continue

                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "transcription": transcription,
                    }
                )

    # 3) Write CSV
    out_path = Path(output_csv).resolve()
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["audio_path", "transcription"])
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Done!")
    print(f"- Root: {root}")
    print(f"- CSV:  {out_path}")
    print(f"- Rows written: {len(rows)}")
    print(f"- Missing audio lines skipped: {missing_audio}")
    print(f"- Bad/unparsed lines skipped: {bad_lines}")


if __name__ == "__main__":
    # Example usage:
    # main(r"C:\data\my_dataset", "my_dataset.csv")
   main(
    root_dir=r"C:\Users\jfvdk\Desktop\Git\Deep-Learning\SignWave-Models-transcription\data",
    output_csv="dataset.csv",
    strict=False
)

