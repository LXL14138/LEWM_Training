"""Implementation hashes: SHA256 of the wm_training source files that produced a result (audit F3).

Evaluation and probe scripts record ``"implementation_sha256": {file name: sha256}`` in their
run.json so a result directory can be tied to the exact code that wrote it, the way
finetune_tilt.py records ``adapter_sha256``. Standard library only (hashlib / pathlib): the
stop-policy evaluation and its tests run under the numpy-only interpreter, so this module must
never import h5py, PIL or torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_hashes(names, package_dir=None):
    """{file name: sha256} of wm_training module files, in the given order.

    ``names`` are plain ``*.py`` file names inside the package (no directories); every file must
    exist and no name may repeat, so a typo in a script's file list fails loudly instead of
    silently dropping a hash.
    """
    package_dir = PACKAGE_DIR if package_dir is None else Path(package_dir)
    names = [str(name) for name in names]
    if not names:
        raise ValueError("implementation_hashes needs at least one file name")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate implementation file names: {names}")
    out = {}
    for name in names:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError(f"Expected a plain wm_training module file name, got {name!r}")
        path = package_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Implementation file missing: {path}")
        out[name] = file_sha256(path)
    return out
