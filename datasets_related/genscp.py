# -*- encoding: utf-8 -*-
"""
Generate .scp files for training/validation based on directories defined in train.yml,
and write them to the relative paths under datasets_related as recorded in train.yml.

Usage:
  python datasets_related/genscp.py --opt train.yml \
      [--exts .wav,.flac] [--key-scheme basename|relpath] [--min-seconds 0] \
      [--relative-to /base/for/paths]

YAML requirements (train.yml):
  light_conf:
    # Output .scp paths (relative to repo root). Example values shown.
    train_mix_scp: datasets_related/tr_mix.scp
    train_ref_scp:
      - datasets_related/tr_s1.scp
      - datasets_related/tr_s2.scp
    val_mix_scp: datasets_related/cv_mix.scp
    val_ref_scp:
      - datasets_related/cv_s1.scp
      - datasets_related/cv_s2.scp

  data_dirs:
    # Input audio directories. Entries can be plain paths or dicts with "path"
    # and optional "key_replace" instructions to normalize filenames. If you keep
    # them as simple strings and the directory name contains tokens like "noisy"
    # or "clean", the script automatically strips `_noisy`/`_clean` (and `-noisy`/
    # `-clean`) from filenames to match mix/ref pairs.
    train:
      mix: ../DNS3/train_noisy
      s1:  ../DNS3/train_clean
    val:
      mix: ../DNS3/dev_noisy
      s1:  ../DNS3/dev_clean
  paths:
    datasets_related: datasets_related
    dns_root: ../DNS3

Key points:
  - References are inferred from every key in data_dirs[split] except "mix".
    Their alphabetical order must match light_conf.*_ref_scp entries.
  - `key_replace` allows aligning filenames such as "foo_noisy" vs "foo_clean" by
    stripping suffixes or performing arbitrary string replacements.
  - Only the intersection of keys across mix/ref directories is written.
  - `--min-seconds` filters out samples shorter than the threshold using
    header metadata (no audio decoding required).
  - `--relative-to` writes file paths relative to the provided base directory
    (default: absolute paths). Make sure you run training from the same base
    so the relative paths resolve correctly.
  - You can define reusable placeholders (e.g. `paths.datasets_related`) and
    refer to them via `{datasets_related}` anywhere in the YAML.

The Datasets class will then consume these scp files during training.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from option import load_config


_AUTO_SUFFIX_HINTS = {
    "noisy": ["_noisy", "-noisy"],
    "clean": ["_clean", "-clean"],
}


def _make_key(path: str, base_dir: str, scheme: str) -> str:
    if scheme == "basename":
        return os.path.splitext(os.path.basename(path))[0]
    if scheme == "relpath":
        rel = os.path.relpath(path, base_dir)
        return os.path.splitext(rel)[0].replace(os.sep, "/")
    raise ValueError(f"Unknown key scheme: {scheme}")


def _apply_replacements(text: str, replacements: Sequence[Tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _normalize_replacements(entries: Sequence) -> List[Tuple[str, str]]:
    normalized: List[Tuple[str, str]] = []
    for item in entries:
        if isinstance(item, str):
            if "->" in item:
                old, new = item.split("->", 1)
            else:
                old, new = item, ""
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            old, new = item
        elif isinstance(item, dict):
            old = item.get("from") or item.get("src") or item.get("old")
            if old is None:
                raise ValueError(f"key_replace dict requires 'from'/'src'/'old': {item}")
            new = item.get("to") or item.get("dst") or item.get("new") or ""
        else:
            raise ValueError(f"Unsupported key_replace entry: {item}")
        normalized.append((str(old), str(new)))
    return normalized


def _auto_replacements_for_dir(path: str) -> List[Tuple[str, str]]:
    base = os.path.basename(os.path.normpath(path)).lower()
    replacements: List[Tuple[str, str]] = []
    for token, patterns in _AUTO_SUFFIX_HINTS.items():
        if token in base:
            replacements.extend((pattern, "") for pattern in patterns)
    return replacements


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _parse_source_entry(entry, base_dir: str) -> Dict:
    if isinstance(entry, str):
        path = entry
        replacements: Sequence = []
    elif isinstance(entry, dict):
        if "path" not in entry:
            raise KeyError("Each data_dirs entry dict must include 'path'")
        path = entry["path"]
        replacements = entry.get("key_replace", [])
    else:
        raise TypeError("data_dirs entries must be either string paths or dicts")

    resolved_path = _resolve_path(path, base_dir)
    if replacements:
        reps = _normalize_replacements(replacements)
    else:
        reps = _auto_replacements_for_dir(resolved_path)
    return {"path": resolved_path, "replacements": reps}


def _list_audio(source_conf: Dict, exts: Tuple[str, ...], key_scheme: str) -> Dict[str, str]:
    dir_path = source_conf["path"]
    replacements = source_conf["replacements"]
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    mapping: Dict[str, str] = {}
    for ext in exts:
        pattern = os.path.join(dir_path, f"**/*{ext}")
        for p in glob.glob(pattern, recursive=True):
            key = _make_key(p, dir_path, key_scheme)
            key = _apply_replacements(key, replacements)
            if key not in mapping:
                mapping[key] = os.path.abspath(p)
    return mapping


def _write_scp(
    out_path: str,
    mapping: Dict[str, str],
    keys: Sequence[str],
    relative_to: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k in keys:
            path = mapping[k]
            if relative_to:
                path = os.path.relpath(path, relative_to)
            f.write(f"{k} {path}\n")


def _filter_by_duration(keys: List[str], mix_map: Dict[str, str], min_seconds: float) -> List[str]:
    if not min_seconds or min_seconds <= 0:
        return keys
    kept: List[str] = []
    for k in keys:
        path = mix_map[k]
        try:
            info = sf.info(path)
            duration = info.frames / float(info.samplerate or 1)
        except Exception:
            continue
        if duration >= min_seconds:
            kept.append(k)
    if not kept:
        raise RuntimeError("All samples filtered out by min-seconds criterion.")
    return kept


def _generate_split(
    split_cfg: Dict,
    out_mix_scp: str,
    out_ref_scps: Sequence[str],
    exts: Tuple[str, ...],
    key_scheme: str,
    min_seconds: float,
    base_dir: str,
    relative_to: Optional[str],
) -> None:
    if "mix" not in split_cfg:
        raise KeyError("Each split must define a 'mix' directory under data_dirs")

    mix_source = _parse_source_entry(split_cfg["mix"], base_dir)
    ref_items = sorted((k, v) for k, v in split_cfg.items() if k != "mix")
    if len(ref_items) != len(out_ref_scps):
        raise ValueError(
            f"Number of reference directories ({len(ref_items)}) does not match the number of output scp files ({len(out_ref_scps)})."
        )

    ref_sources = [_parse_source_entry(cfg, base_dir) for _, cfg in ref_items]

    mix_map = _list_audio(mix_source, exts, key_scheme)
    ref_maps = [_list_audio(src, exts, key_scheme) for src in ref_sources]

    common_keys = set(mix_map.keys())
    for ref_map in ref_maps:
        common_keys &= set(ref_map.keys())

    keys_sorted = sorted(common_keys)
    if not keys_sorted:
        raise RuntimeError("No overlapping keys found across mix/ref directories. Check naming or key_replace rules.")

    keys_filtered = _filter_by_duration(keys_sorted, mix_map, min_seconds)

    _write_scp(out_mix_scp, mix_map, keys_filtered, relative_to)
    for ref_path, ref_map in zip(out_ref_scps, ref_maps):
        _write_scp(ref_path, ref_map, keys_filtered, relative_to)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="train.yml", help="Path to option YAML file")
    parser.add_argument("--exts", type=str, default=".wav,.flac", help="Comma-separated audio extensions to scan")
    parser.add_argument(
        "--key-scheme",
        type=str,
        default="basename",
        choices=["basename", "relpath"],
        help="Key scheme for scp entries",
    )
    parser.add_argument("--min-seconds", type=float, default=0.0, help="Minimum duration (seconds) to include")
    parser.add_argument(
        "--relative-to",
        type=str,
        default=None,
        help="Base directory to which written paths should be relative (default: absolute paths)",
    )
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.opt)
    cfg_dir = os.path.dirname(cfg_path)
    cfg = load_config(cfg_path)

    light = cfg.get("light_conf", {})
    data_dirs = cfg.get("data_dirs", {})
    if not data_dirs:
        raise KeyError("'data_dirs' section not found in YAML. See genscp.py header for required structure.")

    def resolve_output(path_or_rel: str) -> str:
        return _resolve_path(path_or_rel, cfg_dir)

    train_mix_out = resolve_output(light["train_mix_scp"])
    train_refs_out = [resolve_output(p) for p in light["train_ref_scp"]]
    val_mix_out = resolve_output(light["val_mix_scp"])
    val_refs_out = [resolve_output(p) for p in light["val_ref_scp"]]

    exts = tuple(x.strip() for x in args.exts.split(",") if x.strip())
    relative_base = args.relative_to
    if relative_base:
        relative_base = _resolve_path(relative_base, cfg_dir)

    if "train" not in data_dirs:
        raise KeyError("'data_dirs.train' not found in YAML")
    _generate_split(
        data_dirs["train"],
        train_mix_out,
        train_refs_out,
        exts,
        key_scheme=args.key_scheme,
        min_seconds=args.min_seconds,
        base_dir=cfg_dir,
        relative_to=relative_base,
    )

    if "val" not in data_dirs:
        raise KeyError("'data_dirs.val' not found in YAML")
    _generate_split(
        data_dirs["val"],
        val_mix_out,
        val_refs_out,
        exts,
        key_scheme=args.key_scheme,
        min_seconds=args.min_seconds,
        base_dir=cfg_dir,
        relative_to=relative_base,
    )

    print("SCP generation completed.")
    print(f"  Train mix: {train_mix_out}")
    print(f"  Train ref: {train_refs_out}")
    print(f"  Val   mix: {val_mix_out}")
    print(f"  Val   ref: {val_refs_out}")


if __name__ == "__main__":
    main()
