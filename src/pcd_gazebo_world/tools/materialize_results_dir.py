#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except Exception:
        return False


def _materialize_symlink(link_path: Path, results_root: Path, *, dry_run: bool) -> bool:
    if not link_path.is_symlink():
        return False

    # If the link *text* stays within results_root (e.g. "latest/实验01_一"),
    # keep it as an internal navigation entry even if it currently resolves to
    # something external due to intermediate links.
    try:
        link_text = os.readlink(link_path)
        if os.path.isabs(link_text):
            lexical = Path(os.path.normpath(link_text))
        else:
            lexical = Path(os.path.normpath(str(link_path.parent / link_text)))
        # NOTE: This is a *lexical* check; do not resolve symlinks here, otherwise
        # an internal link like "latest/实验01_一" could be treated as external if
        # "latest" temporarily points to a directory containing external links.
        lexical_abs = lexical.absolute()
        root_abs = results_root.resolve()
        if os.path.commonpath([str(lexical_abs), str(root_abs)]) == str(root_abs):
            return False
    except OSError:
        pass

    try:
        target = link_path.resolve(strict=False)
    except Exception as exc:
        print(f"[warn] cannot resolve symlink: {link_path} ({exc})", file=sys.stderr)
        return False

    if _is_within(target, results_root):
        return False

    if not target.exists():
        print(f"[warn] broken symlink: {link_path} -> {target}", file=sys.stderr)
        return False

    if dry_run:
        print(f"[dry-run] materialize: {link_path} -> {target}")
        return True

    if target.is_dir():
        link_path.unlink()
        shutil.copytree(target, link_path, symlinks=False, copy_function=shutil.copy2)
        print(f"[OK] dir: {link_path} <- {target}")
        return True

    if target.is_file():
        link_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{link_path.name}.",
            suffix=".tmp",
            dir=str(link_path.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copy2(target, tmp_path)
            link_path.unlink()
            os.replace(str(tmp_path), str(link_path))
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
        print(f"[OK] file: {link_path} <- {target}")
        return True

    print(f"[warn] unsupported symlink target type: {link_path} -> {target}", file=sys.stderr)
    return False


def materialize_results(results_root: Path, *, dry_run: bool) -> int:
    results_root = results_root.expanduser().resolve()
    if not results_root.exists():
        print(f"ERROR: results dir not found: {results_root}", file=sys.stderr)
        return 2
    if not results_root.is_dir():
        print(f"ERROR: results root is not a directory: {results_root}", file=sys.stderr)
        return 2

    changed = 0
    # Iterate until stable: converting a symlink-dir into a real dir may introduce new
    # paths to scan (os.walk won't descend into symlink dirs by default).
    for _ in range(3):
        round_changed = 0
        for link_path in results_root.rglob("*"):
            if _materialize_symlink(link_path, results_root, dry_run=dry_run):
                round_changed += 1
        changed += round_changed
        if round_changed == 0:
            break

    return 0 if changed >= 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="把 结果/ 内部指向外部的符号链接实化为真实文件（复制到 结果/ 里）")
    parser.add_argument(
        "--results",
        type=str,
        default="结果",
        help="结果目录（默认：结果）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印不执行")
    args = parser.parse_args()

    return materialize_results(Path(args.results), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
