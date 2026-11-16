"""Run the presentation experiments on a fresh checkout, without overwrites."""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_OUTPUTS = [
    "data/lfw_matching",
    "artifacts/casia_matching_encoded",
    "artifacts/casia_pairwise_development",
    "artifacts/casia_dynamic_extended_development",
    "artifacts/casia_dynamic_heldout",
]
APPENDIX_OUTPUTS = [
    "artifacts/casia_search_protocol.json",
    "artifacts/casia_search_development",
    "artifacts/casia_direct_count_development",
]


def commands(stage):
    """Keep the individual experiment CLIs available for inspecting each stage."""
    if stage == "main":
        return [
            ["evaluation.pairwise", "prepare"],
            ["evaluation.pairwise", "encode"],
            ["evaluation.pairwise", "evaluate"],
            [
                "evaluation.dynamic", "--grid", "presentation", "--selection", "count",
                "--output", MAIN_OUTPUTS[3],
            ],
            ["evaluation.heldout", "freeze", "--output", MAIN_OUTPUTS[4]],
            ["evaluation.heldout", "run", "--output", MAIN_OUTPUTS[4]],
        ]
    return [
        ["evaluation.fixed_gallery", "prepare"],
        ["evaluation.fixed_gallery", "evaluate"],
        ["evaluation.direct_count", "--output", APPENDIX_OUTPUTS[2]],
    ]


def check_inputs():
    # Import here so --dry-run works before dependencies or assets are installed.
    from evaluation.pairwise import ARCHIVE_SHA256
    from face_velocity.inputs import EXPECTED_MODELS

    expected = {
        "data/downloads/lfw.tgz": ARCHIVE_SHA256,
        "models/buffalo_l/det_10g.onnx": EXPECTED_MODELS["detector"],
        "models/casia_r50/cisia_r50.onnx": EXPECTED_MODELS["recognizer"],
    }
    errors = []
    for name, wanted in expected.items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing: {name}")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != wanted:
            errors.append(f"SHA-256 mismatch: {name}")
        else:
            print(f"Verified: {name}", flush=True)
    if errors:
        raise ValueError("\n".join(errors) + "\nSee README: Required inputs.")


def check_outputs(stage, root=ROOT):
    outputs = MAIN_OUTPUTS if stage == "main" else APPENDIX_OUTPUTS
    occupied = [name for name in outputs if (root / name).exists()]
    if occupied:
        raise FileExistsError(
            "Existing outputs will not be overwritten:\n  "
            + "\n  ".join(occupied)
            + "\nUse a fresh checkout, or individual CLIs with fresh output paths."
        )
    if stage == "appendix":
        required = [
            "data/lfw_matching/development/manifest.json",
            "artifacts/casia_matching_encoded/report.json",
            "artifacts/casia_dynamic_extended_development/protocol.json",
            "artifacts/casia_dynamic_extended_development/summary.json",
        ]
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError("Run the main stage first. Missing: " + missing[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["main", "appendix"], default="main")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify input hashes only")
    mode.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()
    if args.dry_run:
        for command in commands(args.stage):
            print(shlex.join(["python", "-m", *command]))
        return
    try:
        if args.check:
            check_inputs()
            return
        check_outputs(args.stage)
        check_inputs()
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for command in commands(args.stage):
            print("\nRunning: " + shlex.join(command), flush=True)
            subprocess.run(
                [sys.executable, "-m", *command], cwd=ROOT, env=environment, check=True
            )
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")
    print("\nComplete. See README: Expected results and presentation map.")


if __name__ == "__main__":
    main()
