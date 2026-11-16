"""Freeze a development-selected policy, then evaluate reserved identities once.

The development diagnostic CLIs continue to refuse final-test manifests.
This entry point saves the policy/protocol before materializing test photos.
"""

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.dynamic import Policy, Scores, aggregate, evaluate_stream, write_trace
from evaluation.pairwise import (
    EXPECTED_MODELS,
    digest,
    encode,
    read_manifest,
    save_json,
)
from face_velocity.inputs import load_inputs, normalize_vectors

SOURCES = [
    "evaluation/heldout.py",
    "evaluation/dynamic.py",
    "evaluation/pairwise.py",
    "evaluation/fixed_gallery.py",
    "face_velocity/extraction.py",
    "face_velocity/pipeline.py",
    "face_velocity/inputs.py",
]
PACKAGES = ["insightface", "onnxruntime", "numpy", "opencv-python", "scikit-image"]


def validate_split(development, final):
    """Check identities, image IDs, selected bytes and prior illustration exposure."""
    for key in ["identity", "image_id", "sha256"]:
        left = {r[key] for r in development["images"]}
        right = {r[key] for r in final["images"]}
        if left & right:
            raise ValueError(f"Development/final overlap in {key}")
    identities = {r["identity"] for r in final["images"]}
    if identities & set(final["previously_exposed_identities"]):
        raise ValueError("Final identities were previously exposed")


def freeze(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    development = read_manifest(args.development_manifest)
    final = read_manifest(args.manifest, expected_partition="final_test")
    validate_split(development, final)
    source = json.loads(args.development_summary.read_text())
    chosen = source["selected"]
    if (chosen["acceptance"], chosen["update_delta"]) != (0.385, 0.05):
        raise ValueError("Development candidate differs from the authorized policy")
    minimum = min(
        source["configurations"], key=lambda r: (r["membership_cost_mean"], r["policy"])
    )
    if chosen != minimum:
        raise ValueError("Candidate is not the recorded count-objective minimum")
    model_paths = {"detector": args.detector, "recognizer": args.recognizer}
    if {k: digest(v) for k, v in model_paths.items()} != EXPECTED_MODELS:
        raise ValueError("Model weights differ from development")
    if digest(args.archive) != final["archive_sha256"]:
        raise ValueError("LFW archive hash differs from the prepared split")
    images = sorted(r["image_id"] for r in final["images"])
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "partition": "final_test",
        "purpose": "Freeze the development-selected policy before held-out inference",
        "paths": {
            "manifest": str(args.manifest),
            "development_manifest": str(args.development_manifest),
            "development_summary": str(args.development_summary),
            "archive": str(args.archive),
            **{k: str(v) for k, v in model_paths.items()},
        },
        "hashes": {
            "manifest": digest(args.manifest),
            "development_manifest": digest(args.development_manifest),
            "development_summary": digest(args.development_summary),
            "archive": final["archive_sha256"],
            **EXPECTED_MODELS,
        },
        "source_hashes": {p: digest(p) for p in SOURCES},
        "packages": {p: version(p) for p in PACKAGES},
        "policy": {
            k: chosen[k]
            for k in ["policy", "acceptance", "update_delta", "update_threshold"]
        },
        "orders": {
            str(seed): np.random.default_rng(seed).permutation(images).tolist()
            for seed in [42, 43, 44]
        },
        "image_count": len(images),
        "identity_count": len({r["identity"] for r in final["images"]}),
        "initial_gallery": "empty for each order; no development references",
        "interval_seconds": 30,
        "window_seconds": 86400,
        "aggregation": "Max per identity, rank 1, earliest group tie-break",
        "reference_rule": "Enroll first usable reference for a new identity. "
        "Accepted events count. Append only at score >= 0.435. "
        "No cap; references persist after count-window expiry.",
        "extraction": "Unchanged SCRFD-10GF, central target, five-point alignment, "
        "CASIA R50. Detector 640x640, confidence 0.5, NMS 0.4.",
        "objective": "Report mean extra+missed per scored query, with equal weights. "
        "No selection, tuning or feasibility filter on test results.",
        "reporting": "All three orders and their mean/range, extra and missed "
        "separately, count MAE/exact accuracy, extraction coverage, "
        "all-input exact count rate, returning-query metrics and "
        "dynamic FPIR/FNIR with raw denominators.",
        "failures": "Unresolved inputs produce no count, remain in true histories "
        "and all-input denominator, excluded from conditional rates.",
        "limits": "Identity-disjoint from development, not proof of disjointness "
        "from pretraining. Synthetic order/time, not natural chronology. "
        "At 2482 events and 30 seconds, duration <24h: no window expiry. "
        "Smaller gallery than development; absolute rates not directly "
        "comparable. Three orders are sensitivity, not independent CIs.",
    }
    args.output.mkdir(parents=True)
    save_json(args.output / "protocol.json", protocol)
    (args.output / "protocol.sha256").write_text(digest(args.output / "protocol.json"))
    print(
        json.dumps(
            {
                "frozen": str(args.output / "protocol.json"),
                "policy": protocol["policy"],
                "images": len(images),
            }
        )
    )


def verify_protocol(output):
    protocol_path = output / "protocol.json"
    if digest(protocol_path) != (output / "protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(protocol_path.read_text())
    for key, path in protocol["paths"].items():
        if digest(path) != protocol["hashes"][key]:
            raise ValueError(f"Frozen input changed: {key}")
    for path, expected in protocol["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen evaluation source changed: {path}")
    if {p: version(p) for p in PACKAGES} != protocol["packages"]:
        raise ValueError("Inference packages changed")
    return protocol


def materialize(manifest, archive_path):
    """Read exact archive members, verify bytes, and write only manifest paths."""
    pending = {row["source_member"]: row for row in manifest["images"]}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name not in pending:
                continue
            row = pending.pop(member.name)
            if not member.isfile():
                raise ValueError("Expected a regular image archive member")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Image bytes differ from reserved manifest")
            destination = Path(row["path"])
            if destination.exists():
                if digest(destination) != row["sha256"]:
                    raise ValueError("Existing test image differs")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    if pending:
        raise ValueError("Reserved images are missing from the archive")


def run(args):
    protocol = verify_protocol(args.output)
    if (args.output / "summary.json").exists() or (args.output / "traces").exists():
        raise FileExistsError("Final replay already started; preserve its results")
    paths = {k: Path(v) for k, v in protocol["paths"].items()}
    final = read_manifest(paths["manifest"], expected_partition="final_test")
    validate_split(read_manifest(paths["development_manifest"]), final)
    materialize(final, paths["archive"])
    encoded = args.output / "encoded"
    encode(
        SimpleNamespace(
            manifest=paths["manifest"],
            output=encoded,
            detector=paths["detector"],
            recognizer=paths["recognizer"],
        ),
        expected_partition="final_test",
    )
    _, images, records, report = load_inputs(
        paths["manifest"], encoded, expected_partition="final_test"
    )
    valid_ids = sorted(
        k for k, r in records.items() if r["status"] == "valid_embedding"
    )
    vectors = normalize_vectors(encoded / "embeddings.npz", valid_ids)
    matrix = np.stack([vectors[k] for k in valid_ids])
    scores = Scores(valid_ids, np.clip(matrix @ matrix.T, -1, 1))
    policy = Policy(
        protocol["policy"]["acceptance"], protocol["policy"]["update_delta"]
    )
    (args.output / "traces").mkdir()
    summaries = []
    for seed, order in protocol["orders"].items():
        summary, rows = evaluate_stream(
            images,
            scores,
            policy,
            order,
            int(seed),
            interval=protocol["interval_seconds"],
            window=protocol["window_seconds"],
        )
        write_trace(args.output / "traces" / f"{policy.name}_s{seed}.csv", rows)
        summaries.append(summary)
        print(
            f"Frozen test seed {seed}: cost={summary['membership_cost']:.6f}, "
            f"count exact={summary['count_exact_rate']:.4%}",
            flush=True,
        )
    # Aggregation of repeated orders for ONE frozen policy, never policy selection.
    combined, _ = aggregate(summaries, selection="count")
    assert len(combined) == 1
    result = combined[0]
    result.pop("feasible", None)  # Historical development constraint is irrelevant.
    save_json(
        args.output / "summary.json",
        {
            "partition": "final_test",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(args.output / "protocol.json"),
            "extraction_report_sha256": digest(encoded / "report.json"),
            "frozen_policy": protocol["policy"],
            "per_replay": summaries,
            "result": result,
            "extraction": {k: report[k] for k in ["attempted", "valid", "statuses"]},
            "selection_performed": False,
            "scope": protocol["limits"],
        },
    )
    print(json.dumps({"output": str(args.output), "result": result}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lfw_matching/final_test/manifest.json"),
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/lfw_matching/development/manifest.json"),
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=Path("artifacts/casia_dynamic_extended_development/summary.json"),
    )
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/lfw.tgz"))
    parser.add_argument(
        "--detector", type=Path, default=Path("models/buffalo_l/det_10g.onnx")
    )
    parser.add_argument(
        "--recognizer", type=Path, default=Path("models/casia_r50/cisia_r50.onnx")
    )
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
