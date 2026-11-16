"""Audit a fixed development selection and render its extraction evidence.

Run from the repository root. This checks interfaces and visible preprocessing,
not recognition accuracy. Identity labels never enter extraction or selection.
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/face-velocity-matplotlib")

import cv2
import matplotlib
import numpy as np

from face_velocity.extraction import extract, load_models
from face_velocity.inputs import digest as file_hash

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_detection(axis, record):
    axis.axis("off")
    image = cv2.imread(record["path"])
    if image is None:
        axis.text(0.1, 0.5, "Unreadable input")
        return
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for face in record["faces"]:
        selected = face["face_index"] == record["selected_face_index"]
        color = "#00b99a" if selected else "#bbbbbb"
        left, top, right, bottom = face["bbox"]
        axis.add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                fill=False,
                edgecolor=color,
                linewidth=1.6,
            )
        )
        points = np.asarray(face["landmarks"])
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=9,
            c="#ffd84d" if selected else color,
        )
    axis.scatter(image.shape[1] // 2, image.shape[0] // 2, marker="+", c="#ff7800")


def render_rows(records, path, heading):
    figure, axes = plt.subplots(len(records), 3, figsize=(12, 2.7 * len(records)))
    axes = np.asarray(axes).reshape(len(records), 3)
    for row_axes, record in zip(axes, records):
        draw_detection(row_axes[0], record)
        row_axes[0].set_title(Path(record["source_member"]).stem, fontsize=10)
        row_axes[1].axis("off")
        if "crop_path" in record:
            crop = cv2.imread(record["crop_path"])
            row_axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        row_axes[1].set_title("112 x 112 aligned crop", fontsize=10)
        row_axes[2].axis("off")
        origin = "Already in deck" if record["deck_usage"] else "Supplement"
        lines = [
            record["image_id"],
            origin,
            record["status"],
            f"Detected faces: {record.get('face_count', 0)}",
        ]
        if record["status"] == "valid_embedding":
            values = record["first_eight_raw_features"]
            lines += [
                "Output: 512 finite values",
                f"Raw L2 norm: {record['raw_l2_norm']:.3f}",
                "First 8 raw coordinates:",
                ", ".join(f"{x:.3f}" for x in values[:4]),
                ", ".join(f"{x:.3f}" for x in values[4:]),
            ]
        row_axes[2].text(0, 0.94, "\n".join(lines), va="top", fontsize=11)
    figure.suptitle(heading, fontsize=17, color="#172d47")
    figure.text(
        0.04,
        0.006,
        "LFW development audit. Green: selected box; yellow: landmarks; "
        "gray: other detections; orange: image center.\n"
        "Overlay marks are not encoder input. This is not recognition accuracy.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_summary(records, output, counts):
    figure, axes = plt.subplots(5, 6, figsize=(15, 13))
    for axis, record in zip(axes.flat, records):
        axis.axis("off")
        if "crop_path" in record:
            crop = cv2.imread(record["crop_path"])
            axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        else:
            axis.text(0, 0.5, record["status"])
        origin = "DECK" if record["deck_usage"] else "SUPPLEMENT"
        axis.set_title(
            Path(record["source_member"]).stem.replace("_", " ") + "\n" + origin,
            fontsize=9,
        )
    figure.suptitle("Fixed development audit: 30 images, 10 identities", fontsize=18)
    figure.text(
        0.03,
        0.01,
        "LFW labels. Three photos per identity. "
        "Crops shown for review; no recognition accuracy claim.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(output / "overview.png", dpi=150)
    plt.close(figure)

    labels = [
        "Attempted images",
        "Unique target selected",
        "Aligned crop",
        "Finite, nonzero 512-D vector",
    ]
    values = [counts[k] for k in ["attempted", "selected", "aligned", "valid"]]
    figure, axis = plt.subplots(figsize=(12, 5))
    bars = axis.barh(labels, values, color="#007f82")
    axis.bar_label(
        bars,
        labels=[f"{v} / {counts['attempted']}" for v in values],
        padding=8,
        fontsize=14,
    )
    axis.invert_yaxis()
    axis.set_xlim(0, counts["attempted"] * 1.2)
    axis.set_xlabel("Images in the fixed audit selection")
    axis.set_title("Extraction checks on 30 development images", fontsize=19)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.04,
        0.02,
        "Engineering check; no threshold tuning. "
        "These counts do not measure recognition accuracy.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "extraction_summary.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--detector", type=Path, default=Path("models/buffalo_l/det_10g.onnx")
    )
    parser.add_argument(
        "--recognizer", type=Path, default=Path("models/casia_r50/cisia_r50.onnx")
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if len(selection["images"]) != 30:
        raise ValueError("This fixed audit expects exactly 30 selected images.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "crops").mkdir()
    detector, recognizer = load_models(args.detector, args.recognizer)
    records, embeddings = [], {}
    for row in selection["images"]:
        record, features = extract(row, detector, recognizer, args.output)
        records.append(record)
        if features is not None:
            embeddings[row["image_id"]] = features
        print(row["image_id"], record["status"], flush=True)
    counts = dict(
        attempted=len(records),
        selected=sum(r["selected_face_index"] is not None for r in records),
        aligned=sum("crop_path" in r for r in records),
        valid=len(embeddings),
        multiple_detections=sum(r.get("face_count", 0) > 1 for r in records),
        statuses=dict(Counter(r["status"] for r in records)),
    )
    report = dict(
        completed_at=datetime.now(timezone.utc).isoformat(),
        scope="Engineering extraction audit; no matching, tuning or final test",
        selection_path=str(args.selection),
        selection_sha256=file_hash(args.selection),
        script_sha256=file_hash(__file__),
        models={str(p): file_hash(p) for p in [args.detector, args.recognizer]},
        settings=dict(
            detector_size=[640, 640],
            detector_threshold=0.5,
            nms_iou=0.4,
            target_rule="unique center-containing box",
            crop_size=[112, 112],
            recognition_mean=recognizer.input_mean,
            recognition_std=recognizer.input_std,
            provider="CPU",
        ),
        environment={
            n: version(n)
            for n in ["insightface", "onnxruntime", "numpy", "opencv-python"]
        },
        counts=counts,
        records=records,
        visual_review="Pending review of all contact sheets; numerical validity "
        "does not establish correct face selection or alignment.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "embeddings.npz", **embeddings)
    for offset in range(0, len(records), 6):
        render_rows(
            records[offset : offset + 6],
            args.output / f"inspection_{offset // 6 + 1:02}.png",
            f"Extraction audit: images {offset + 1}-{offset + 6}",
        )
    examples = [
        r for r in records if "slides12/15 alignment example" in r["deck_usage"]
    ]
    render_rows(
        examples,
        args.output / "deck_examples.png",
        "Existing deck examples with measured CASIA outputs",
    )
    render_summary(records, args.output, counts)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
