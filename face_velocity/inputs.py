"""Model fingerprints and validated inputs shared by replay and evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np

OVERLAP_SOURCE = (
    "https://github.com/happynear/FaceDatasets/blob/master/"
    "CASIA/webface_lfw_overlap_detail.txt"
)
# Exact LFW-side names reported by that source. This is conservative name-based
# exclusion, not an independently verified or exhaustive identity-overlap audit.
EXCLUDED = set(
    """William_Macy Vanessa_Williams Mary_Blige Laura_Elena_Harring
David_Kelley Norm_Macdonald Hillary_Clinton Wanda_de_la_Jesus Michael_Jordan
Nicole_Parker Zhang_Ziyi Prince_William Liu_Ye Randy_Jackson Jesse_James
John_Mabry Richard_Cohen""".split()
)
EXPECTED_MODELS = {
    "detector": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "recognizer": "6fe59fee63a4d0f9a6ce2f0cbd1ccd91eab21b1e146181610060da2c955bcc7a",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path, *, expected_partition="development"):
    manifest = json.loads(path.read_text())
    if expected_partition not in {"development", "final_test"}:
        raise ValueError("Unknown evaluation partition")
    if manifest["partition"] != expected_partition:
        if expected_partition == "development":
            raise ValueError("This diagnostic workflow refuses final-test inference.")
        raise ValueError("Manifest partition does not match the requested workflow")
    images = manifest["images"]
    if len({r["image_id"] for r in images}) != len(images):
        raise ValueError("Duplicate image IDs")
    if len({r["sha256"] for r in images}) != len(images):
        raise ValueError("Duplicate selected file bytes")
    if {r["identity"] for r in images} & EXCLUDED:
        raise ValueError("Reported excluded identity present")
    return manifest


def load_inputs(manifest_path, encoded_path, *, expected_partition="development"):
    manifest = read_manifest(manifest_path, expected_partition=expected_partition)
    report_path = encoded_path / "report.json"
    report = json.loads(report_path.read_text())
    if report["configuration"]["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Extraction belongs to a different manifest")
    if report["configuration"]["models"] != EXPECTED_MODELS:
        raise ValueError("Unexpected cached model weights")
    for filename, key in [
        ("records.jsonl", "records_sha256"),
        ("embeddings.npz", "embeddings_sha256"),
    ]:
        if digest(encoded_path / filename) != report[key]:
            raise ValueError(f"Changed cached file: {filename}")
    records_list = [
        json.loads(line)
        for line in (encoded_path / "records.jsonl").read_text().splitlines()
    ]
    records = {r["image_id"]: r for r in records_list}
    images = {r["image_id"]: r for r in manifest["images"]}
    if len(records) != len(records_list) or set(records) != set(images):
        raise ValueError("Incomplete or duplicate extraction records")
    for image_id, row in images.items():
        if records[image_id]["sha256"] != row["sha256"]:
            raise ValueError("Extraction image hash differs from input manifest")
    return manifest, images, records, report


def normalize_vectors(path, valid_ids):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(valid_ids):
            raise ValueError("Vector keys disagree with extraction statuses")
        vectors = {}
        for key in saved.files:
            vector = np.asarray(saved[key], dtype=np.float64)
            norm = np.linalg.norm(vector)
            if vector.shape != (512,) or not np.isfinite(vector).all() or norm <= 0:
                raise ValueError("Invalid cached vector")
            vectors[key] = vector / norm
    return vectors
