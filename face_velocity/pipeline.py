"""Frozen CASIA extraction and label-free dynamic identity decisions."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class Policy:
    acceptance: float = 0.385
    reference_update: float = 0.435
    window_seconds: int = 86400

    def __post_init__(self):
        if not -1 <= self.acceptance <= self.reference_update <= 1:
            raise ValueError("Invalid acceptance/reference thresholds")
        if self.window_seconds <= 0:
            raise ValueError("Window must be positive")

    def configuration(self):
        return asdict(self)


FROZEN_POLICY = Policy()


def select_target(image_record):
    """Select the unique center-containing face without using identity labels."""
    if "image_shape" not in image_record:
        return image_record["status"], None
    height, width = image_record["image_shape"][:2]
    center_x, center_y = width // 2, height // 2
    center_faces = [
        face
        for face in image_record["faces"]
        if face["bbox"][0] <= center_x <= face["bbox"][2]
        and face["bbox"][1] <= center_y <= face["bbox"][3]
    ]
    if len(center_faces) != 1:
        return ("ambiguous_target" if center_faces else "target_not_detected"), None
    return center_faces[0]["status"], center_faces[0]


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 0:
        raise ValueError("Embedding must be a finite, nonzero vector")
    return vector / norm


def associate(vector, references, next_identity, policy=FROZEN_POLICY):
    """Max per identity, rank 1. References precede the current query.

    Sorting by group then admission order matches the experiment's tie-break.
    No dataset labels or previously computed decisions enter this function.
    """
    result = dict(
        decision="unresolved",
        assigned_identity=None,
        top_score=None,
        winning_reference=None,
        reference_added=False,
        candidates=[],
    )
    if vector is None:
        return result
    query = normalize(vector)
    if references:
        ordered = sorted(references, key=lambda r: (r["identity"], r["sequence"]))
        matrix = np.asarray([r["embedding"] for r in ordered], dtype=np.float64)
        scores = np.clip(matrix @ query, -1, 1)
        best = int(np.argmax(scores))
        winner, score = ordered[best], float(scores[best])
        by_identity = {}
        for r, value in zip(ordered, scores):
            if (
                r["identity"] not in by_identity
                or value > by_identity[r["identity"]]["score"]
            ):
                by_identity[r["identity"]] = dict(
                    identity=r["identity"], score=float(value), event_id=r["event_id"]
                )
        result.update(
            top_score=score,
            winning_reference=winner["event_id"],
            candidates=sorted(
                by_identity.values(), key=lambda r: (-r["score"], r["identity"])
            )[:3],
        )
        if score >= policy.acceptance:
            return dict(
                result,
                decision="accept",
                assigned_identity=winner["identity"],
                reference_added=score >= policy.reference_update,
            )
    return dict(
        result,
        decision="enroll",
        assigned_identity=next_identity,
        reference_added=True,
    )


class CasiaEncoder:
    """Use the exact SCRFD/alignment/CASIA extraction used by the experiments."""

    def __init__(self, detector_path, recognizer_path):
        from face_velocity.extraction import load_models
        from face_velocity.inputs import EXPECTED_MODELS, digest

        paths = dict(detector=Path(detector_path), recognizer=Path(recognizer_path))
        hashes = {key: digest(path) for key, path in paths.items()}
        if hashes != EXPECTED_MODELS:
            raise ValueError("Model weights differ from the evaluated checkpoints")
        self.models = load_models(paths["detector"], paths["recognizer"])
        self.lock = Lock()
        self.configuration = dict(
            models=hashes,
            input_size=640,
            detector_threshold=0.5,
            nms_iou=0.4,
            target="unique center-containing face",
            aligned_size=112,
        )

    def extract(self, image_bytes):
        from face_velocity.extraction import extract

        with self.lock, TemporaryDirectory(prefix="face-velocity-") as directory:
            path = Path(directory) / "input.jpg"
            path.write_bytes(image_bytes)
            row = dict(
                image_id="upload",
                path=str(path),
                sha256=hashlib.sha256(image_bytes).hexdigest(),
            )
            metadata, vector = extract(row, *self.models, output=None)
        if metadata["status"] == "unreadable":
            raise ValueError("The upload is not a readable image")
        for key in ["path", "image_id", "sha256"]:
            metadata.pop(key, None)
        return metadata, normalize(vector).tolist() if vector is not None else None


def service_configuration(encoder):
    """Version the deployed code separately from preserved research results."""
    config = dict(
        policy=FROZEN_POLICY.configuration(), extraction=encoder.configuration
    )
    config["source_hashes"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in [
            "pipeline.py",
            "storage.py",
            "app.py",
            "demo.py",
            "extraction.py",
            "inputs.py",
        ]
    }
    config["version"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return config
