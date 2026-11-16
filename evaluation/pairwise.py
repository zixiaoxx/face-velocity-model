"""Reproducible CASIA development pairs: prepare, encode, then evaluate.

This custom protocol is not the official LFW 6,000-pair benchmark. Final-test
identities are reserved at preparation and explicitly refused by later stages.
"""

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from face_velocity.inputs import (
    EXCLUDED,
    EXPECTED_MODELS,
    OVERLAP_SOURCE,
    digest,
    read_manifest,
)

ARCHIVE_SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_bin(count):
    return next(
        i for i, upper in enumerate([2, 4, 9, 19, float("inf")]) if count <= upper
    )


def choose_final_identities(groups, exposed, count, random_generator):
    """Allocate held-out identities proportionally across photo-count strata."""
    bins = defaultdict(list)
    for identity, members in sorted(groups.items()):
        if len(members) >= 2:
            bins[count_bin(len(members))].append(identity)
    total = sum(map(len, bins.values()))
    ideal = {k: count * len(v) / total for k, v in bins.items()}
    quotas = {k: int(v) for k, v in ideal.items()}
    for k in sorted(ideal, key=lambda k: (-(ideal[k] - quotas[k]), k))[
        : count - sum(quotas.values())
    ]:
        quotas[k] += 1
    final = set()
    for k, names in sorted(bins.items()):
        candidates = sorted(set(names) - exposed)
        if len(candidates) < quotas[k]:
            raise ValueError("Exposed identities prevent requested stratification")
        final.update(random_generator.choice(candidates, quotas[k], replace=False))
    assert len(final) == count and not final & exposed
    return final, quotas


def make_pairs(images, seed, impostor_count=50000, genuine_cap=20):
    random_generator = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in images:
        groups[row["identity"]].append(row["image_id"])
    pairs = []
    for identity, ids in sorted(groups.items()):
        available = list(combinations(sorted(ids), 2))
        selected = random_generator.choice(
            len(available), min(genuine_cap, len(available)), replace=False
        )
        for index in sorted(selected):
            left, right = available[index]
            pairs.append(dict(left=left, right=right, kind="genuine"))
    identities = sorted(groups)
    total_negative_pairs = (
        len(images) ** 2 - sum(len(v) ** 2 for v in groups.values())
    ) // 2
    if impostor_count > total_negative_pairs:
        raise ValueError("Requested more distinct impostor pairs than available")
    negatives = set()
    while len(negatives) < impostor_count:
        indices = random_generator.choice(len(identities), 2, replace=False)
        left, right = [
            str(random_generator.choice(groups[identities[i]])) for i in indices
        ]
        negatives.add(tuple(sorted((left, right))))
    pairs.extend(dict(left=a, right=b, kind="impostor") for a, b in sorted(negatives))
    return pairs


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.archive) != ARCHIVE_SHA256:
        raise ValueError("Original LFW archive hash mismatch")
    groups, exposed_members = defaultdict(list), defaultdict(set)
    exposure_sources = {}
    for path in args.exposure_manifest:
        data = json.loads(path.read_text())
        exposure_sources[str(path)] = digest(path)
        for row in data["images"]:
            member = row["source_member"]
            exposed_members[PurePosixPath(member).parts[1]].add(member)
    random_generator = np.random.default_rng(args.seed)
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        for name in sorted(members):
            parts = PurePosixPath(name).parts
            if len(parts) == 3 and parts[0] == "lfw" and name.endswith(".jpg"):
                groups[parts[1]].append(name)
        excluded_counts = {n: len(groups[n]) for n in sorted(EXCLUDED)}
        eligible = {n: v for n, v in groups.items() if n not in EXCLUDED}
        final_multi, quotas = choose_final_identities(
            eligible, set(exposed_members), 502, random_generator
        )
        singles = sorted(n for n, v in eligible.items() if len(v) == 1)
        selected_singles = list(random_generator.choice(singles, 1000, replace=False))
        # Existing manifests currently contain multi-photo identities only.
        if set(selected_singles) & set(exposed_members):
            raise ValueError(
                "Exposed singleton requires explicit development placement"
            )
        final_single = set(selected_singles[:300])
        names = sorted(n for n, v in eligible.items() if len(v) >= 2)
        names = sorted(names + selected_singles)
        splits = {"development": [], "final_test": []}
        for identity in names:
            available = eligible[identity]
            preserved = sorted(exposed_members.get(identity, set()))
            if not set(preserved) <= set(available) or len(preserved) > 20:
                raise ValueError("Prior exposure cannot be preserved under cap20")
            remaining = sorted(set(available) - set(preserved))
            chosen = preserved + sorted(
                random_generator.choice(
                    remaining, min(20, len(available)) - len(preserved), replace=False
                ).tolist()
            )
            partition = (
                "final_test"
                if identity in final_multi | final_single
                else "development"
            )
            for member in chosen:
                splits[partition].append(
                    dict(
                        image_id=PurePosixPath(member).stem,
                        identity=identity,
                        source_member=member,
                        path=str(
                            args.output
                            / partition
                            / "images"
                            / PurePosixPath(member).name
                        ),
                        deck_usage=[],
                    )
                )
        assert len(splits["development"]) + len(splits["final_test"]) == 8347
        dev_ids = {r["identity"] for r in splits["development"]}
        final_ids = {r["identity"] for r in splits["final_test"]}
        assert len(dev_ids) == 1871 and len(final_ids) == 802
        assert not dev_ids & final_ids and not final_ids & set(exposed_members)
        args.output.mkdir(parents=True)
        (args.output / "development/images").mkdir(parents=True)
        (args.output / "final_test").mkdir()
        # Hash all selected bytes, but only materialize development images.
        # No final-test decoding, face inference, pairs or scores in this run.
        row_by_member = {
            r["source_member"]: r for rows in splits.values() for r in rows
        }
        hashes = {}
        for entry in archive.getmembers():
            if entry.name not in row_by_member:
                continue
            data = archive.extractfile(entry).read()
            row = row_by_member[entry.name]
            row["sha256"] = hashlib.sha256(data).hexdigest()
            if row["sha256"] in hashes:
                raise ValueError("Duplicate image bytes: inspect before evaluation")
            hashes[row["sha256"]] = entry.name
            if row["identity"] in dev_ids:
                Path(row["path"]).write_bytes(data)
    protocol = dict(
        seed=args.seed,
        archive_sha256=ARCHIVE_SHA256,
        overlap_source=OVERLAP_SOURCE,
        excluded_images_by_identity=excluded_counts,
        overlap_scope="Reported LFW-side name exclusion; aliases and completeness "
        "are not independently certified. Michael_Jordan mapping "
        "is name-ambiguous; conservatively excluded regardless.",
        previously_exposed_identities=sorted(exposed_members),
        exposure_manifests=exposure_sources,
        max_photos_per_identity=20,
        final_multi_stratum_quotas=quotas,
        genuine_pairs_per_identity_cap=20,
        impostor_pairs=50000,
        pair_seed=args.seed + 1,
        pair_sampling="Genuine: uniform photo-pair sampling within identity, cap20. "
        "Impostor: uniform distinct identity pairs, then uniform "
        "photo per identity; reject duplicate unordered photo pairs.",
        threshold_grid="-1 to1 inclusive, step0.005; accept score>=threshold",
        diagnostic_thresholds=[0.2, 0.3, 0.4, 0.5, 0.6],
        preparation_script_sha256=digest(__file__),
        final_test_status="Manifest and selected-byte hashes only; images not "
        "materialized, decoded, embedded or scored.",
    )
    summary = {}
    for partition, rows in splits.items():
        data = dict(protocol, partition=partition, images=rows)
        save_json(args.output / partition / "manifest.json", data)
        per_identity = Counter(r["identity"] for r in rows)
        summary[partition] = dict(
            images=len(rows),
            identities=len(per_identity),
            multi=sum(v >= 2 for v in per_identity.values()),
            singleton=sum(v == 1 for v in per_identity.values()),
        )
    pairs = make_pairs(splits["development"], args.seed + 1)
    csv_write(args.output / "development/pairs.csv", pairs, ["left", "right", "kind"])
    dev_manifest_path = args.output / "development/manifest.json"
    dev_manifest = json.loads(dev_manifest_path.read_text())
    dev_manifest["pairs_sha256"] = digest(args.output / "development/pairs.csv")
    save_json(dev_manifest_path, dev_manifest)
    summary["pairs"] = dict(Counter(p["kind"] for p in pairs))
    summary["excluded"] = dict(
        identities=len(EXCLUDED), images=sum(excluded_counts.values())
    )
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def encode(args, *, expected_partition="development"):
    from face_velocity.extraction import extract, load_models

    manifest = read_manifest(args.manifest, expected_partition=expected_partition)
    models = {"detector": args.detector, "recognizer": args.recognizer}
    for name, path in models.items():
        if digest(path) != EXPECTED_MODELS[name]:
            raise ValueError(f"Unexpected {name} weights")
    configuration = dict(
        manifest_sha256=digest(args.manifest),
        models=EXPECTED_MODELS,
        extraction_source_sha256=digest(Path("face_velocity/extraction.py")),
        target_source_sha256=digest(Path("face_velocity/pipeline.py")),
        settings=dict(
            input_size=[640, 640],
            threshold=0.5,
            nms_iou=0.4,
            target="unique center-containing box",
            aligned_size=[112, 112],
        ),
        packages={
            n: version(n)
            for n in [
                "insightface",
                "onnxruntime",
                "numpy",
                "opencv-python",
                "scikit-image",
            ]
        },
    )
    if args.output.exists():
        saved = json.loads((args.output / "configuration.json").read_text())
        if saved != configuration:
            raise ValueError("Cache configuration differs; use a fresh output path")
        if (args.output / "report.json").exists():
            print("Verified configuration: complete cache already exists")
            return
    else:
        args.output.mkdir(parents=True)
        save_json(args.output / "configuration.json", configuration)
    detector, recognizer = load_models(args.detector, args.recognizer)
    records, vectors = [], {}
    # Append records so interruption retains diagnostic evidence. A partial run
    # is restarted in a fresh path, rather than silently trusting partial state.
    if (args.output / "records.jsonl").exists():
        raise ValueError("Partial run exists; use a new output directory")
    start = perf_counter()
    with (args.output / "records.jsonl").open("w") as stream:
        for index, row in enumerate(manifest["images"], 1):
            record, vector = extract(row, detector, recognizer, None)
            records.append(record)
            if vector is not None:
                vectors[row["image_id"]] = vector
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if index % 100 == 0 or index == len(manifest["images"]):
                print(
                    f"Encoded {index}/{len(manifest['images'])}; "
                    f"valid={len(vectors)}; elapsed={perf_counter() - start:.0f}s",
                    flush=True,
                )
    np.savez_compressed(args.output / "embeddings.npz", **vectors)
    report = dict(
        configuration=configuration,
        attempted=len(records),
        valid=len(vectors),
        statuses=dict(Counter(r["status"] for r in records)),
        multiple_detections=sum(r.get("face_count", 0) > 1 for r in records),
        records_sha256=digest(args.output / "records.jsonl"),
        embeddings_sha256=digest(args.output / "embeddings.npz"),
        elapsed_seconds=perf_counter() - start,
        timing_scope="Sequential diagnostic run, not a latency benchmark",
    )
    save_json(args.output / "report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "configuration"}),
        flush=True,
    )


def error_counts(genuine, impostor, threshold):
    if not len(genuine) or not len(impostor):
        raise ValueError("Both scoreable genuine and impostor pairs are required")
    false_matches = int(np.count_nonzero(impostor >= threshold))
    false_nonmatches = int(np.count_nonzero(genuine < threshold))
    return dict(
        threshold=float(threshold),
        false_matches=false_matches,
        impostor_pairs=len(impostor),
        fmr=false_matches / len(impostor),
        false_nonmatches=false_nonmatches,
        genuine_pairs=len(genuine),
        fnmr=false_nonmatches / len(genuine),
    )


def render_evaluation(output, genuine, impostor, sweep, examples, images, records):
    import cv2
    from insightface.utils.face_align import norm_crop2

    from evaluation.extraction_audit import draw_detection, plt

    plt.rcParams.update(
        {"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axis = plt.subplots(figsize=(12, 6.5))
    bins = np.linspace(-1, 1, 81)
    for values, label, color in [
        (genuine, "Same identity", "#007f82"),
        (impostor, "Different identities", "#b57422"),
    ]:
        axis.hist(
            values,
            bins=bins,
            weights=np.ones(len(values)) / len(values),
            alpha=0.55,
            label=f"{label} (n={len(values):,})",
            color=color,
        )
    axis.set(
        xlabel="Cosine similarity",
        ylabel="Fraction of pairs within each group",
        title="CASIA R50: development pair scores",
        xlim=(-0.5, 1),
    )
    axis.legend()
    figure.text(
        0.03,
        0.02,
        "Custom LFW development pairs. Known-name exclusions; "
        "residual training overlap remains possible.\n"
        "Pairs reuse identities/images. These are not independent trials "
        "or official LFW benchmark results.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(output / "score_distributions.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 6.5))
    thresholds = [r["threshold"] for r in sweep]
    axis.plot(
        thresholds,
        [100 * r["fmr"] for r in sweep],
        color="#b57422",
        label="FMR: different identities accepted",
        linewidth=2.5,
    )
    axis.plot(
        thresholds,
        [100 * r["fnmr"] for r in sweep],
        color="#007f82",
        label="FNMR: same identity rejected",
        linewidth=2.5,
    )
    axis.set(
        xlabel="Cosine threshold (accept score >= threshold)",
        ylabel="Error rate (%)",
        title="Development error rates across thresholds",
        xlim=(-0.1, 0.9),
        ylim=(0, 100),
    )
    axis.legend()
    figure.text(
        0.03,
        0.02,
        f"Scoreable pairs: {len(genuine):,} same identity; "
        f"{len(impostor):,} different identities. Unscoreable pairs reported "
        "separately.\nDiagnostic sweep only. No final operating threshold "
        "selected; final-test scores remain untouched.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(output / "threshold_errors.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(3, 4, figsize=(13, 10))
    for group_index, group in enumerate(["lowest_genuine", "highest_impostor"]):
        for row_index, pair in enumerate(examples[group]):
            for col, image_id in enumerate([pair["left"], pair["right"]]):
                axis = axes[row_index, 2 * group_index + col]
                record = records[image_id]
                face = record["faces"][record["selected_face_index"]]
                raw = cv2.imread(images[image_id]["path"])
                crop, _ = norm_crop2(
                    raw, np.asarray(face["landmarks"], np.float32), image_size=112
                )
                axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                axis.axis("off")
                axis.set_title(image_id.replace("_", " "), fontsize=10)
                if col == 0:
                    axis.text(
                        0,
                        -0.12,
                        f"Cosine = {pair['score']:.3f}",
                        transform=axis.transAxes,
                        fontsize=12,
                    )
    figure.suptitle(
        "Lowest same-person scores                 Highest different-person scores",
        fontsize=17,
    )
    figure.text(
        0.04,
        0.01,
        "Extreme pairs selected after scoring, for diagnosis only. "
        "Identity labels from LFW. No production threshold selected.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output / "pair_examples.png", dpi=160)
    plt.close(figure)
    failures = [r for r in records.values() if r["status"] != "valid_embedding"]
    for start in range(0, len(failures), 6):
        batch = failures[start : start + 6]
        figure, axes = plt.subplots(
            len(batch), 2, figsize=(11, 3 * len(batch)), squeeze=False
        )
        for row_axes, record in zip(axes, batch):
            draw_detection(row_axes[0], record)
            row_axes[0].set_title(record["image_id"], fontsize=11)
            row_axes[1].axis("off")
            row_axes[1].text(
                0,
                0.65,
                f"Status: {record['status']}\n"
                f"Detected faces: {record.get('face_count', 0)}\n"
                "Retained in attempted-input denominator.\n"
                "Associated pairs remain unscoreable.",
                fontsize=12,
            )
        figure.suptitle("Development extraction failures at fixed settings")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output / f"extraction_failures_{start // 6 + 1:02}.png", dpi=140)
        plt.close(figure)


def evaluate(args):
    manifest = read_manifest(args.manifest)
    images = {r["image_id"]: r for r in manifest["images"]}
    if digest(args.pairs) != manifest["pairs_sha256"]:
        raise ValueError("Pair list changed after preparation")
    extraction = json.loads((args.encoded / "report.json").read_text())
    if extraction["configuration"]["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Embeddings belong to a different manifest")
    for filename in ["embeddings.npz", "records.jsonl"]:
        key = "embeddings_sha256" if filename.endswith("npz") else "records_sha256"
        if digest(args.encoded / filename) != extraction[key]:
            raise ValueError(f"Changed extraction output: {filename}")
    with (args.encoded / "records.jsonl").open() as stream:
        records = {r["image_id"]: r for r in map(json.loads, stream)}
    if set(records) != set(images):
        raise ValueError("Extraction records do not cover all planned inputs")
    with np.load(args.encoded / "embeddings.npz", allow_pickle=False) as saved:
        vectors = {}
        for key in saved.files:
            vector = np.asarray(saved[key], dtype=np.float64)
            norm = np.linalg.norm(vector)
            if vector.shape != (512,) or not np.isfinite(vector).all() or norm <= 0:
                raise ValueError("Invalid cached vector")
            vectors[key] = vector / norm
    with args.pairs.open() as stream:
        pairs = list(csv.DictReader(stream))
    seen, scored, unavailable = set(), [], Counter()
    for pair in pairs:
        left, right, kind = pair["left"], pair["right"], pair["kind"]
        key = tuple(sorted((left, right)))
        if left == right or key in seen or left not in images or right not in images:
            raise ValueError("Invalid or duplicate pair")
        seen.add(key)
        same = images[left]["identity"] == images[right]["identity"]
        if kind not in {"genuine", "impostor"} or same != (kind == "genuine"):
            raise ValueError("Pair type inconsistent with dataset identity labels")
        if left not in vectors or right not in vectors:
            unavailable[kind] += 1
            scored.append(dict(pair, score=None, status="unscoreable"))
        else:
            score = float(np.clip(np.dot(vectors[left], vectors[right]), -1, 1))
            scored.append(dict(pair, score=score, status="scoreable"))
    genuine = np.array(
        [
            p["score"]
            for p in scored
            if p["kind"] == "genuine" and p["score"] is not None
        ]
    )
    impostor = np.array(
        [
            p["score"]
            for p in scored
            if p["kind"] == "impostor" and p["score"] is not None
        ]
    )
    sweep = [error_counts(genuine, impostor, t) for t in np.linspace(-1, 1, 401)]
    diagnostic = [
        error_counts(genuine, impostor, t) for t in manifest["diagnostic_thresholds"]
    ]
    examples = {
        "lowest_genuine": sorted(
            (p for p in scored if p["kind"] == "genuine" and p["score"] is not None),
            key=lambda p: p["score"],
        )[:3],
        "highest_impostor": sorted(
            (p for p in scored if p["kind"] == "impostor" and p["score"] is not None),
            key=lambda p: -p["score"],
        )[:3],
    }
    summary = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        partition="development",
        manifest_sha256=digest(args.manifest),
        pairs_sha256=digest(args.pairs),
        extraction_report_sha256=digest(args.encoded / "report.json"),
        evaluator_sha256=digest(__file__),
        identities=len({r["identity"] for r in images.values()}),
        attempted_images=len(images),
        valid_embeddings=len(vectors),
        extraction_statuses=extraction["statuses"],
        planned_pairs=dict(Counter(p["kind"] for p in pairs)),
        scoreable_pairs=dict(genuine=len(genuine), impostor=len(impostor)),
        unscoreable_pairs={k: unavailable[k] for k in ["genuine", "impostor"]},
        diagnostic_thresholds=diagnostic,
        distributions={
            kind: dict(
                zip(
                    ["min", "p05", "median", "p95", "max"],
                    np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist(),
                )
            )
            for kind, values in [("genuine", genuine), ("impostor", impostor)]
        },
        scope="Pair-weighted errors conditional on scoreable predefined pairs; "
        "extraction failures/unscoreable pairs reported separately. No "
        "production threshold selection, no final-test evaluation. "
        "Identity/image reuse means pairs are dependent; no IID confidence "
        "intervals or extreme-low-error population guarantees claimed.",
        reference="https://pages.nist.gov/frvt/html/frvt11.html",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    csv_write(
        args.output / "scores.csv", scored, ["left", "right", "kind", "score", "status"]
    )
    csv_write(args.output / "threshold_sweep.csv", sweep, list(sweep[0]))
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "examples.json", examples)
    render_evaluation(args.output, genuine, impostor, sweep, examples, images, records)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--archive", type=Path, default=Path("data/downloads/lfw.tgz"))
    prep.add_argument("--output", type=Path, default=Path("data/lfw_matching"))
    prep.add_argument("--seed", type=int, default=20261111)
    prep.add_argument(
        "--exposure-manifest",
        type=Path,
        nargs="+",
        default=[
            Path("config/prior_exposure.json"),
        ],
    )
    enc = commands.add_parser("encode")
    enc.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lfw_matching/development/manifest.json"),
    )
    enc.add_argument(
        "--output", type=Path, default=Path("artifacts/casia_matching_encoded")
    )
    enc.add_argument(
        "--detector", type=Path, default=Path("models/buffalo_l/det_10g.onnx")
    )
    enc.add_argument(
        "--recognizer", type=Path, default=Path("models/casia_r50/cisia_r50.onnx")
    )
    ev = commands.add_parser("evaluate")
    ev.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lfw_matching/development/manifest.json"),
    )
    ev.add_argument(
        "--pairs", type=Path, default=Path("data/lfw_matching/development/pairs.csv")
    )
    ev.add_argument(
        "--encoded", type=Path, default=Path("artifacts/casia_matching_encoded")
    )
    ev.add_argument(
        "--output", type=Path, default=Path("artifacts/casia_pairwise_development")
    )
    args = parser.parse_args()
    {"prepare": prepare, "encode": encode, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
