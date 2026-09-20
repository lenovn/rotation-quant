"""Reuse an audited INT8 calibration overlay as a complete postprocessing parent."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from experiments.phase3.common import SOURCE_ROOT, format_description, save_frozen, write_json
from experiments.phase3.postprocess import load_static
from experiments.phase3.run import progress, source_record
from experiments.phase3.uniform_baseline import apply_down_int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    (snapshot / "tracked.diff").write_bytes(subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "diff", "--binary", "HEAD"]))
    write_json(args.output / "settings.json", dict(arguments={key: str(value) for key, value in vars(args).items()},
        source=source_record(), pid=os.getpid(), calibration=False, training=False))
    try:
        model, records = load_static(args.parent)
        overlay = apply_down_int8(model, args.scales, args.parent)
        package = args.output / "static_w4a8.pt"
        metadata = dict(parent=str(args.parent), reused_down_int8_overlay=overlay,
                        method="Joint100 matched INT8 calibration; no discrete postprocessing or QAT yet")
        save_frozen(model, records, package, metadata)
        write_json(args.output / "result.json", dict(package=str(package), format=format_description(model),
            **metadata, calibration=False, training=False, reused_existing_calibration=True))
        progress(args.output, "completed", package=str(package))
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
