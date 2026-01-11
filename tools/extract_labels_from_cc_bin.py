#!/usr/bin/env python3

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class PcdBinaryXYZ:
    points: int
    payload: bytes


@dataclass(frozen=True)
class PointBlock:
    offset: int
    count: int


def _read_pcd_binary_xyz(pcd_path: Path) -> PcdBinaryXYZ:
    fields: List[str] = []
    sizes: List[int] = []
    types: List[str] = []
    counts: List[int] = []
    points: Optional[int] = None
    data_kind: Optional[str] = None

    with pcd_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"{pcd_path} has no DATA section")
            line_str = line.decode("ascii", "replace").strip()
            if not line_str or line_str.startswith("#"):
                continue
            if line_str.startswith("FIELDS "):
                fields = line_str.split()[1:]
            elif line_str.startswith("SIZE "):
                sizes = [int(x) for x in line_str.split()[1:]]
            elif line_str.startswith("TYPE "):
                types = line_str.split()[1:]
            elif line_str.startswith("COUNT "):
                counts = [int(x) for x in line_str.split()[1:]]
            elif line_str.startswith("POINTS "):
                points = int(line_str.split()[1])
            elif line_str.startswith("DATA "):
                data_kind = line_str.split()[1]
                break

        if data_kind != "binary":
            raise RuntimeError(f"{pcd_path} must be binary PCD (got DATA {data_kind})")

        if fields != ["x", "y", "z"]:
            raise RuntimeError(f"{pcd_path} must have FIELDS x y z (got {fields})")
        if sizes != [4, 4, 4] or types != ["F", "F", "F"] or counts != [1, 1, 1]:
            raise RuntimeError(
                f"{pcd_path} must be float32 x/y/z only (SIZE={sizes}, TYPE={types}, COUNT={counts})"
            )

        payload = f.read()

    if len(payload) % 12 != 0:
        raise RuntimeError(f"{pcd_path} binary payload length {len(payload)} not multiple of 12 bytes")

    if points is None:
        points = len(payload) // 12
    elif points != len(payload) // 12:
        raise RuntimeError(
            f"{pcd_path} POINTS={points} but payload has {len(payload)//12} points (size mismatch)"
        )

    return PcdBinaryXYZ(points=points, payload=payload)


def _write_pcd_xyz(pcd_path: Path, payload: bytes, points: int) -> None:
    pcd_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {points}\n"
        "DATA binary\n"
    ).encode("ascii")
    with pcd_path.open("wb") as f:
        f.write(header)
        f.write(payload)


def _sample_membership_ratio(
    bin_data: bytes,
    offset: int,
    count: int,
    reference_point_set: Set[bytes],
    max_samples: int = 48,
) -> float:
    start = offset + 4
    if count <= 0:
        return 0.0

    sample_indices = {0, count - 1, count // 2, count // 3, (2 * count) // 3}
    step = max(1, count // 25)
    for i in range(0, count, step):
        sample_indices.add(i)
        if len(sample_indices) >= max_samples:
            break

    hits = 0
    for i in sorted(sample_indices):
        if i < 0 or i >= count:
            continue
        b = bin_data[start + i * 12 : start + (i + 1) * 12]
        if b in reference_point_set:
            hits += 1
    return hits / max(1, len(sample_indices))


def _find_point_blocks(
    bin_data: bytes,
    reference_point_set: Set[bytes],
    min_count: int = 3,
    min_membership_ratio: float = 0.95,
) -> List[PointBlock]:
    blocks: List[PointBlock] = []
    file_len = len(bin_data)
    for offset in range(0, file_len - 16):
        count = struct.unpack_from("<I", bin_data, offset)[0]
        if count < min_count:
            continue
        end = offset + 4 + count * 12
        if end > file_len:
            continue
        first_pt = bin_data[offset + 4 : offset + 16]
        if first_pt not in reference_point_set:
            continue
        last_pt = bin_data[end - 12 : end]
        if last_pt not in reference_point_set:
            continue

        ratio = _sample_membership_ratio(bin_data, offset, count, reference_point_set)
        if ratio < min_membership_ratio:
            continue

        blocks.append(PointBlock(offset=offset, count=count))

    blocks = sorted({b.offset: b for b in blocks}.values(), key=lambda b: b.offset)
    return blocks


def _iter_block_points(bin_data: bytes, block: PointBlock) -> Iterable[bytes]:
    start = block.offset + 4
    for i in range(0, block.count * 12, 12):
        yield bin_data[start + i : start + i + 12]


def _payload_from_points(points_iter: Iterable[bytes]) -> bytes:
    buf = bytearray()
    for pt in points_iter:
        buf.extend(pt)
    return bytes(buf)


def _write_report(path: Path, report: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract two labeled subsets (label=0/1) from a CloudCompare .bin by"
            " finding embedded XYZ point blocks that match a reference binary PCD (x,y,z float32)."
        )
    )
    parser.add_argument("--bin", dest="bin_path", type=Path, required=True, help="CloudCompare .bin file path")
    parser.add_argument(
        "--ref-pcd",
        dest="ref_pcd",
        type=Path,
        required=True,
        help="Reference PCD used to validate point blocks (binary x/y/z float32)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=Path("maps"),
        help="Output directory (default: maps)",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        default=None,
        help="Output filename prefix (default: <bin stem>)",
    )
    parser.add_argument(
        "--subset-label",
        dest="subset_label",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which label the extracted subset corresponds to (default: 0)",
    )
    parser.add_argument(
        "--write-blocks",
        dest="write_blocks",
        action="store_true",
        help="Also write each detected block as a separate PCD (debug, can create many files)",
    )
    args = parser.parse_args()

    ref = _read_pcd_binary_xyz(args.ref_pcd)
    ref_point_set = set(ref.payload[i : i + 12] for i in range(0, len(ref.payload), 12))

    bin_data = args.bin_path.read_bytes()
    blocks = _find_point_blocks(bin_data, ref_point_set)

    subset_points: List[bytes] = []
    subset_set: Set[bytes] = set()
    dup = 0
    for blk in blocks:
        for pt in _iter_block_points(bin_data, blk):
            if pt in subset_set:
                dup += 1
                continue
            subset_set.add(pt)
            subset_points.append(pt)

    # Complement is computed against the reference PCD.
    complement_points: List[bytes] = []
    for i in range(0, len(ref.payload), 12):
        pt = ref.payload[i : i + 12]
        if pt not in subset_set:
            complement_points.append(pt)

    prefix = args.prefix or args.bin_path.stem
    label_a = args.subset_label
    label_b = 1 - label_a

    out_a = args.out_dir / f"{prefix}_label{label_a}.pcd"
    out_b = args.out_dir / f"{prefix}_label{label_b}.pcd"
    report_path = args.out_dir / f"{prefix}_extract_report.json"

    payload_a = _payload_from_points(subset_points)
    payload_b = _payload_from_points(complement_points)

    _write_pcd_xyz(out_a, payload_a, points=len(subset_points))
    _write_pcd_xyz(out_b, payload_b, points=len(complement_points))

    if args.write_blocks:
        blocks_dir = args.out_dir / f"{prefix}_blocks"
        for i, blk in enumerate(blocks):
            payload = _payload_from_points(_iter_block_points(bin_data, blk))
            _write_pcd_xyz(blocks_dir / f"block_{i:03d}_count_{blk.count}.pcd", payload, points=blk.count)

    report: Dict = {
        "bin": str(args.bin_path),
        "ref_pcd": str(args.ref_pcd),
        "detected_blocks": [{"offset": b.offset, "count": b.count} for b in blocks],
        "detected_block_count": len(blocks),
        "subset_label": label_a,
        "label_counts": {str(label_a): len(subset_points), str(label_b): len(complement_points)},
        "reference_points": ref.points,
        "subset_coverage": (len(subset_points) / ref.points) if ref.points else 0.0,
        "duplicate_points_ignored": dup,
        "outputs": {"label_subset": str(out_a), "label_complement": str(out_b)},
    }
    _write_report(report_path, report)

    print(f"[OK] Blocks: {len(blocks)}")
    print(f"[OK] label{label_a}: {len(subset_points)} points -> {out_a}")
    print(f"[OK] label{label_b}: {len(complement_points)} points -> {out_b}")
    print(f"[OK] Report: {report_path}")

    if len(subset_points) + len(complement_points) != ref.points:
        print(
            f"[WARN] label0+label1 != reference points ({len(subset_points)} + {len(complement_points)} != {ref.points})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
