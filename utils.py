# -*- encoding: utf-8 -*-
'''
@Filename    :utils.py
@Time        :2020/07/10 23:23:35
@Author      :Kai Li
@Version     :1.0
'''

import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

import soundfile as sf
import torch
import yaml

def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict


def snapshot_experiment(run_dir, config=None, code_root=None, extra_dirs: Optional[Iterable[str]] = None):
    """Copy config and source files into the current run directory.

    Args:
        run_dir: Destination directory (e.g. wandb run files folder).
        config: Dict-like configuration to dump as YAML (optional).
        code_root: Path to the project root whose *.py files should be copied.
        extra_dirs: Iterable of subdirectories under code_root to copy recursively
            (e.g. ["models"]). Defaults to ("models",) if None.
    """

    dest = Path(run_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoints").mkdir(exist_ok=True)
    (dest / "val_samples").mkdir(exist_ok=True)
    (dest / "logs").mkdir(exist_ok=True)

    if config is not None:
        cfg_path = dest / "config_snapshot.yaml"
        try:
            with open(cfg_path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
        except Exception:
            logging.exception("Failed to write config snapshot to %s", cfg_path)

    if code_root is None:
        return

    code_root = Path(code_root)
    if not code_root.exists():
        logging.warning("Code root %s does not exist; skipping code snapshot", code_root)
        return

    codes_dest = dest / "codes"
    codes_dest.mkdir(exist_ok=True)

    for item in code_root.glob("*.py"):
        try:
            shutil.copy2(item, codes_dest / item.name)
        except Exception:
            logging.exception("Failed to copy %s to %s", item, codes_dest)

    dirs_to_copy = extra_dirs if extra_dirs is not None else ("models",)
    for subdir in dirs_to_copy:
        src = code_root / subdir
        if not src.exists():
            continue
        try:
            shutil.copytree(src, codes_dest / subdir, dirs_exist_ok=True)
        except Exception:
            logging.exception("Failed to copy directory %s to %s", src, codes_dest / subdir)


def save_validation_samples(
    sample_dir,
    noisy,
    clean,
    enhanced,
    sample_rate: int,
    epoch: int,
    max_samples: int = 3,
    start_index: int = 0,
):
    """Save a few validation audio samples for inspection.

    Returns number of samples actually written.
    """

    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    def _to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if hasattr(tensor, "cpu"):
            return tensor.cpu().numpy()
        return tensor

    noisy_np = _to_numpy(noisy)
    clean_np = _to_numpy(clean)
    enhanced_np = _to_numpy(enhanced)

    batch = noisy_np.shape[0]
    to_save = min(max_samples, batch)
    saved = 0
    for idx in range(to_save):
        global_idx = start_index + saved + 1
        base_name = f"sample_{global_idx:03d}"
        noisy_path = sample_dir / f"{base_name}_noisy.wav"
        clean_path = sample_dir / f"{base_name}_clean.wav"
        enh_path = sample_dir / f"{base_name}_epoch{epoch:03d}_enh.wav"

        for path, array in (
            (noisy_path, noisy_np[idx]),
            (clean_path, clean_np[idx]),
        ):
            if not path.exists():
                sf.write(path, array.squeeze(), sample_rate)

        sf.write(enh_path, enhanced_np[idx].squeeze(), sample_rate)
        saved += 1

    return saved
