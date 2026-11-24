"""Download the two TabPFN-2 checkpoints into /models at image build, each pinned to a Hugging Face commit SHA.

A download at boot would sit inside Railway's healthcheck window, so the weights are part of the image.
Both repositories are ungated (read 2026-11-24) and declare `license: other` = Prior Labs License 1.1
(Apache 2.0 plus an attribution requirement, text in /licenses). Only these TabPFN-2 weights are licensed for
commercial use; the later TabPFN-2.5 to 3.5 weights are non-commercial and gated, and are never fetched here.
"""
import hashlib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
ROOT.mkdir(parents=True, exist_ok=True)
FILES = [
    ("Prior-Labs/TabPFN-v2-clf", "275f19d2add9c2c0de38f9ebaded3856599fa2e7", "tabpfn-v2-classifier.ckpt"),
    ("Prior-Labs/TabPFN-v2-reg", "02a44705ff7790b50cded404bc59adb94da456a1", "tabpfn-v2-regressor.ckpt"),
]
for repo, rev, name in FILES:
    path = hf_hub_download(repo, name, revision=rev, local_dir=ROOT)
    data = Path(path).read_bytes()
    print(f"{len(data):>12} sha256={hashlib.sha256(data).hexdigest()} {repo}@{rev[:10]} {name}")
