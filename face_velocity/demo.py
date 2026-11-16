"""Read experiment evidence and supply causal replay inputs, never match labels."""

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import isclose
from pathlib import Path

from face_velocity.inputs import load_inputs, normalize_vectors
from face_velocity.pipeline import FROZEN_POLICY

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "development": (
        "data/lfw_matching/development/manifest.json",
        "artifacts/casia_matching_encoded",
        "artifacts/casia_dynamic_extended_development",
    ),
    "heldout": (
        "data/lfw_matching/final_test/manifest.json",
        "artifacts/casia_dynamic_heldout/encoded",
        "artifacts/casia_dynamic_heldout",
    ),
}


def read_json(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def source(name):
    manifest_path, encoded_path, evidence_path = [ROOT / p for p in SOURCES[name]]
    partition = "final_test" if name == "heldout" else "development"
    manifest, images, records, extraction = load_inputs(
        manifest_path, encoded_path, expected_partition=partition
    )
    valid = sorted(k for k, r in records.items() if r["status"] == "valid_embedding")
    vectors = normalize_vectors(encoded_path / "embeddings.npz", valid)
    protocol = read_json(evidence_path / "protocol.json")
    summary = read_json(evidence_path / "summary.json")
    if name == "heldout":
        expected = (evidence_path / "protocol.sha256").read_text().strip()
        if (
            hashlib.sha256((evidence_path / "protocol.json").read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("Saved held-out protocol changed")
    result = summary["result"] if name == "heldout" else summary["selected"]
    protocol_hash = hashlib.sha256(
        (evidence_path / "protocol.json").read_bytes()
    ).hexdigest()
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("Summary belongs to a different replay protocol")
    if not (
        isclose(result["acceptance"], FROZEN_POLICY.acceptance, abs_tol=1e-12)
        and isclose(
            result["update_threshold"], FROZEN_POLICY.reference_update, abs_tol=1e-12
        )
    ):
        raise ValueError("Saved evidence uses a different policy from the demo")
    return dict(
        images=images,
        records=records,
        vectors=vectors,
        protocol=protocol,
        result=result,
        extraction=extraction,
        evidence_path=evidence_path,
        identities=len({r["identity"] for r in manifest["images"]}),
    )


def evidence(name):
    data = source(name)
    result = data["result"]
    return dict(
        source=name,
        scope="Saved experiment results across all three event orders",
        images=len(data["images"]),
        identities=data["identities"],
        usable=data["extraction"]["valid"],
        count_accuracy=result["count_exact_rate_mean"],
        accuracy_min=result["count_exact_rate_min"],
        accuracy_max=result["count_exact_rate_max"],
        extra=result["extra_per_event_mean"],
        missed=result["missed_per_event_mean"],
        cost=result["membership_cost_mean"],
        seeds=[42, 43, 44],
        limitations=(
            "Synthetic times and order. Held-out identities differ from development, "
            "but pretraining overlap cannot be ruled out. The held-out stream is "
            "shorter than 24 hours; it does not exercise window expiry."
        ),
    )


def replay_input(name, seed, index):
    data = source(name)
    order = data["protocol"]["orders"][str(seed)]
    if index >= len(order):
        return None
    image_id = order[index]
    row = data["images"][image_id]
    path = (ROOT / row["path"]).resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("Replay image is outside the local dataset")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("Replay image bytes changed")
    metadata = {
        k: v
        for k, v in data["records"][image_id].items()
        if k not in {"identity", "path", "image_id", "source_member", "deck_usage"}
    }
    metadata["mode"] = "cached evaluated CASIA embedding"
    vector = data["vectors"].get(image_id)
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=index * data["protocol"]["interval_seconds"]
    )
    return dict(
        event_id=f"e{index:05d}",
        image_id=image_id,
        event_time=instant,
        image_bytes=content,
        metadata=metadata,
        vector=None if vector is None else vector.tolist(),
    )


def score_history(name, events):
    """Ground truth is consulted only AFTER assignment and never enters storage."""
    images = source(name)["images"]
    counts, extra, missed, exact = 0, 0, 0, 0
    annotations = {}
    histories = {}
    truth_groups = {}
    for row in events:
        identity = images[row["image_id"]]["identity"]
        truth_groups.setdefault(identity, len(truth_groups) + 1)
        instant = datetime.fromisoformat(row["event_time"])
        history = histories.setdefault(identity, [])
        history.append((instant, row["event_id"]))
        expected = {eid for time, eid in history if time > instant - timedelta(days=1)}
        if row["count"] is None:
            annotations[row["event_id"]] = dict(
                true_count=len(expected),
                unresolved=True,
                truth_group=truth_groups[identity],
            )
            continue
        returned = set(row["counted_events"])
        e, m = sorted(returned - expected), sorted(expected - returned)
        annotations[row["event_id"]] = dict(
            true_count=len(expected),
            extra=e,
            missed=m,
            truth_group=truth_groups[identity],
            exact=row["count"] == len(expected),
        )
        extra += len(e)
        missed += len(m)
        exact += row["count"] == len(expected)
        counts += 1
    return dict(
        scored=counts,
        attempted=len(events),
        extra=extra,
        missed=missed,
        mean_cost=(extra + missed) / counts if counts else None,
        count_accuracy=exact / counts if counts else None,
        events=annotations,
    )


def saved_trace(name, seed):
    path = source(name)["evidence_path"] / "traces" / f"a0.385000_u0.05_s{seed}.csv"
    with path.open() as stream:
        return list(csv.DictReader(stream))
