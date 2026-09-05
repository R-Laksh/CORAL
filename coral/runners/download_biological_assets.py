"""Fetch pinned, public benchmark assets; no credential or deployment setup."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

from coral.datasets.gb1 import URL, SHA256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--esmc", action="store_true")
    parser.add_argument("--bpnet", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gb1.csv"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("GB1 source checksum mismatch")
    from huggingface_hub import snapshot_download
    if args.esmc:
        snapshot_download("biohub/ESMC-300M", revision="a59b831785f907e96e6a246b1d142bfb76df31ee",
                          local_dir=root / "esmc300m", allow_patterns=["*.json", "model.safetensors", "README.md"])
    if args.bpnet:
        snapshot_download("kundajelab/encode-bpnet-FOXA1-ChIP-seq-HepG2-ENCSR865RXA-ENCSR337KST",
                          revision="adde8fa27ceb7647e35aebe456620ea6b40d4ec3", local_dir=root / "foxa1_bpnet",
                          allow_patterns=["README.md", "config.json", "fold_0/model.h5", "fold_0/saved_model/*"])


if __name__ == "__main__":
    main()
