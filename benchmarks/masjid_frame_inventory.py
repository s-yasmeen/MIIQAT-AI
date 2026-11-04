#!/usr/bin/env python3
"""Inventory permitted Masjid al-Nabawi RawDataFrames without opening image pixels.

This utility records file-level metadata only. It deliberately ignores directories
whose names suggest face crops, identities, or processed-person imagery.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

EXCLUDED_DIR_TOKENS = ("face", "person", "identity", "subject", "processed")


def is_excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return True
    return any(
        token in part.lower()
        for part in parts
        for token in EXCLUDED_DIR_TOKENS
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, include_hashes: bool = False):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"RawDataFrames directory does not exist: {root}")

    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or is_excluded(path, root):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        relative = path.relative_to(root)
        record = {
            "relative_path": relative.as_posix(),
            "top_level_group": relative.parts[0] if len(relative.parts) > 1 else "",
            "extension": suffix,
            "size_bytes": path.stat().st_size,
        }
        if include_hashes:
            record["sha256"] = sha256_file(path)
        records.append(record)

    if not records:
        raise ValueError("No supported image files found under RawDataFrames")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-frames", required=True, type=Path)
    parser.add_argument("--out-dir", default="masjid_inventory")
    parser.add_argument("--hashes", action="store_true")
    args = parser.parse_args()

    records = inventory(args.raw_frames, include_hashes=args.hashes)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "image_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "source_directory": str(args.raw_frames.resolve()),
        "image_count": len(records),
        "total_size_bytes": sum(row["size_bytes"] for row in records),
        "extension_counts": dict(Counter(row["extension"] for row in records)),
        "top_level_group_counts": dict(Counter(row["top_level_group"] for row in records)),
        "pixel_dimensions": None,
        "note": (
            "Metadata-only inventory. Image pixels were not decoded; no faces, "
            "identities, or individual trajectories were analysed. Camera IDs, "
            "timestamps and dimensions require verified source metadata."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
