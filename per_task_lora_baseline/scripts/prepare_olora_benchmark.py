from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


OLORA_MAIN_ZIP = "https://github.com/cmnfriend/O-LoRA/archive/refs/heads/main.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="CL_Benchmark")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            print(f"{output_dir} already exists; use --overwrite to replace it.")
            return
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "O-LoRA-main.zip"
        print(f"Downloading O-LoRA benchmark archive from {OLORA_MAIN_ZIP}")
        urllib.request.urlretrieve(OLORA_MAIN_ZIP, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)

        source_dir = tmp_dir / "O-LoRA-main" / "CL_Benchmark"
        if not source_dir.exists():
            raise FileNotFoundError(f"CL_Benchmark not found inside archive: {source_dir}")
        shutil.copytree(source_dir, output_dir)

    print(f"Saved O-LoRA CL_Benchmark to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
