"""Prepare and evaluate fixed-gallery CASIA development searches, without a gap.

Both reference configurations use the same queries. Final-test inference is
refused. This is a custom LFW diagnostic, not an official NIST/LFW benchmark.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.pairwise import (
    csv_write,
    digest,
    save_json,
)
from face_velocity.inputs import load_inputs, normalize_vectors

NIST_API = "https://pages.nist.gov/frvt/api/FRVT_IREX_ongoing_1N_api.pdf#page=4"
NAMES = {"A": "A: one reference", "B": "B: up to two references (Max)"}


def assign_roles(images, valid_ids, seed):
    """Choose references from extraction-valid images, never from match scores."""
    groups = defaultdict(list)
    for row in images.values():
        groups[row["identity"]].append(row["image_id"])
    generator = np.random.default_rng(seed)
    primary, extra, gallery_ids = [], [], []
    for identity, image_ids in sorted(groups.items()):
        if len(image_ids) == 1:
            continue
        candidates = sorted(set(image_ids) & set(valid_ids))
        required = 2 if len(image_ids) >= 3 else 1
        if len(candidates) < required:
            raise ValueError(f"Insufficient valid references for {identity}")
        chosen = generator.choice(candidates, required, replace=False).tolist()
        primary.append(chosen[0])
        if required == 2:
            extra.append(chosen[1])
        gallery_ids.append(identity)
    reserved = set(primary + extra)
    queries = sorted(set(images) - reserved)
    return dict(
        gallery_identities=gallery_ids,
        primary_references=primary,
        extra_references=extra,
        query_ids=queries,
    )


def validate_roles(protocol, images, valid_ids):
    primary = protocol["primary_references"]
    extra = protocol["extra_references"]
    queries = protocol["query_ids"]
    all_ids = primary + extra + queries
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(images):
        raise ValueError("Reference/query overlap, duplicates or missing inputs")
    if not set(primary + extra) <= set(valid_ids):
        raise ValueError("Invalid image used as a reference")
    identities = [images[k]["identity"] for k in primary]
    if identities != protocol["gallery_identities"]:
        raise ValueError("Primary references must follow sorted identity order")
    if len(identities) != len(set(identities)):
        raise ValueError("Primary references repeat an identity")
    extra_ids = [images[k]["identity"] for k in extra]
    if len(extra_ids) != len(set(extra_ids)) or not set(extra_ids) <= set(identities):
        raise ValueError("Invalid extra-reference identity")
    query_counts = Counter(images[k]["identity"] for k in queries)
    if any(query_counts[k] < 1 for k in identities):
        raise ValueError("Every gallery identity needs an attempted query")


def prepare(args):
    if args.protocol.exists():
        raise FileExistsError(args.protocol)
    manifest, images, records, report = load_inputs(args.manifest, args.encoded)
    valid = {k for k, r in records.items() if r["status"] == "valid_embedding"}
    roles = assign_roles(images, valid, args.seed)
    validate_roles(roles, images, valid)
    gallery_ids = set(roles["gallery_identities"])
    counts = Counter(
        "mated" if images[k]["identity"] in gallery_ids else "non_mated"
        for k in roles["query_ids"]
    )
    protocol = dict(
        roles,
        partition="development",
        seed=args.seed,
        manifest_sha256=digest(args.manifest),
        extraction_report_sha256=digest(args.encoded / "report.json"),
        script_sha256=digest(__file__),
        selection_targets=[0.01, 0.05],
        selection_rule="Lowest threshold meeting the empirical non-mated false "
        "accept count budget; this minimizes rank-1 FNIR for a "
        "fixed ranking. Inclusive acceptance; tied scores move "
        "together. Targets are demonstration assumptions.",
        reference_policy="Random valid reference per multi-photo identity; "
        "reserve a second for identities with >=3 planned "
        "photos. Extraction validity only, no score selection. "
        "Reserved extra references are never queries in A or B.",
        ranking_tie_rule="Equal identity scores: lexicographic identity order. "
        "Equal reference scores: retain primary reference.",
        threshold_candidates="All distinct top-1 scores and their next higher "
        "float64 values, plus accept-all/reject-all bounds.",
        metrics="Search-conditional FPIR and rank-1 FNIR, known wrong acceptance "
        "and rejection; report failed inputs and all-input known "
        "correct acceptance separately. No gap, no enrollment/update.",
        gallery_identities_count=len(gallery_ids),
        references_A=len(roles["primary_references"]),
        references_B=len(roles["primary_references"] + roles["extra_references"]),
        query_counts=dict(counts),
        query_failures=sum(k not in valid for k in roles["query_ids"]),
        limitation="One reference sampling seed; all non-mated identities are "
        "single-photo identities. Mated queries reuse identities. "
        "Development selection is not independent final testing "
        "or a population-risk guarantee.",
        source=NIST_API,
    )
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.protocol, protocol)
    print(
        json.dumps(
            {
                k: protocol[k]
                for k in [
                    "seed",
                    "gallery_identities_count",
                    "references_A",
                    "references_B",
                    "query_counts",
                    "query_failures",
                    "selection_targets",
                ]
            }
        ),
        flush=True,
    )


def add_references(primary_scores, extra_scores, identity_columns):
    scores = primary_scores.copy()
    used_extra = np.zeros(scores.shape, dtype=bool)
    for column, extra in zip(identity_columns, extra_scores.T):
        used_extra[:, column] = extra > scores[:, column]
        scores[:, column] = np.maximum(scores[:, column], extra)
    return scores, used_extra


def rank_searches(protocol, images, vectors, records):
    primary = protocol["primary_references"]
    extra = protocol["extra_references"]
    identities = protocol["gallery_identities"]
    identity_columns = {name: i for i, name in enumerate(identities)}
    usable = [k for k in protocol["query_ids"] if k in vectors]
    queries = np.stack([vectors[k] for k in usable])
    primary_scores = np.clip(queries @ np.stack([vectors[k] for k in primary]).T, -1, 1)
    extra_scores = np.clip(queries @ np.stack([vectors[k] for k in extra]).T, -1, 1)
    extra_columns = [identity_columns[images[k]["identity"]] for k in extra]
    extra_by_column = dict(zip(extra_columns, extra))
    scores_b, used_extra = add_references(primary_scores, extra_scores, extra_columns)
    assert np.all(scores_b >= primary_scores)
    rows = []
    for config, scores in [("A", primary_scores), ("B", scores_b)]:
        winners = np.argmax(scores, axis=1)
        indexed = {}
        for i, image_id in enumerate(usable):
            identity = images[image_id]["identity"]
            column = identity_columns.get(identity)
            winner = int(winners[i])
            reference = (
                extra_by_column[winner]
                if config == "B" and used_extra[i, winner]
                else primary[winner]
            )
            mate_rank, mate_score = None, None
            if column is not None:
                mate_score = float(scores[i, column])
                mate_rank = 1 + int(np.count_nonzero(scores[i] > mate_score))
                mate_rank += int(np.count_nonzero(scores[i, :column] == mate_score))
            indexed[image_id] = dict(
                configuration=config,
                image_id=image_id,
                true_identity=identity,
                kind="mated" if column is not None else "non_mated",
                status="scoreable",
                top_identity=identities[winner],
                top_score=float(scores[i, winner]),
                winning_reference=reference,
                mate_score=mate_score,
                mate_rank=mate_rank,
                rank1_correct=column is not None and winner == column,
                top_identity_ties=int(np.count_nonzero(scores[i] == scores[i, winner])),
            )
        for image_id in protocol["query_ids"]:
            if image_id in indexed:
                rows.append(indexed[image_id])
            else:
                identity = images[image_id]["identity"]
                rows.append(
                    dict(
                        configuration=config,
                        image_id=image_id,
                        true_identity=identity,
                        kind="mated" if identity in identity_columns else "non_mated",
                        status=records[image_id]["status"],
                        top_identity=None,
                        top_score=None,
                        winning_reference=None,
                        mate_score=None,
                        mate_rank=None,
                        rank1_correct=False,
                        top_identity_ties=None,
                    )
                )
    return rows


def metric_inputs(rows):
    known = [r for r in rows if r["kind"] == "mated"]
    unknown = [r for r in rows if r["kind"] == "non_mated"]
    usable_known = [r for r in known if r["status"] == "scoreable"]
    usable_unknown = [r for r in unknown if r["status"] == "scoreable"]
    if not usable_known or not usable_unknown:
        raise ValueError("Both query groups need usable searches")
    return dict(
        correct=np.sort([r["top_score"] for r in usable_known if r["rank1_correct"]]),
        wrong=np.sort([r["top_score"] for r in usable_known if not r["rank1_correct"]]),
        unknown=np.sort([r["top_score"] for r in usable_unknown]),
        known_attempted=len(known),
        unknown_attempted=len(unknown),
        known_scoreable=len(usable_known),
        unknown_scoreable=len(usable_unknown),
    )


def threshold_metrics(inputs, threshold):
    def accepted(key):
        values = inputs[key]
        return len(values) - int(np.searchsorted(values, threshold, side="left"))

    correct, wrong, unknown = [accepted(k) for k in ["correct", "wrong", "unknown"]]
    known_n = inputs["known_scoreable"]
    unknown_n = inputs["unknown_scoreable"]
    rejected = known_n - correct - wrong
    return dict(
        threshold=float(threshold),
        known_scoreable=known_n,
        unknown_scoreable=unknown_n,
        known_correct_accepted=correct,
        known_wrong_accepted=wrong,
        known_rejected=rejected,
        unknown_false_accepted=unknown,
        unknown_rejected=unknown_n - unknown,
        fpir=unknown / unknown_n,
        rank1_fnir=(known_n - correct) / known_n,
        known_correct_acceptance=correct / known_n,
        known_wrong_acceptance=wrong / known_n,
        known_rejection=rejected / known_n,
        known_extraction_failures=inputs["known_attempted"] - known_n,
        unknown_extraction_failures=inputs["unknown_attempted"] - unknown_n,
        all_input_known_correct_acceptance=correct / inputs["known_attempted"],
        rank1_before_threshold=len(inputs["correct"]) / known_n,
    )


def sweep_thresholds(inputs):
    scores = np.unique(
        np.concatenate([inputs[k] for k in ["correct", "wrong", "unknown"]])
    )
    candidates = np.unique(
        np.concatenate(
            [[-1.0, np.nextafter(1.0, np.inf)], scores, np.nextafter(scores, np.inf)]
        )
    )
    return [threshold_metrics(inputs, threshold) for threshold in candidates]


def select_point(sweep, target):
    budget = int(np.floor(target * sweep[0]["unknown_scoreable"] + 1e-12))
    feasible = [r for r in sweep if r["unknown_false_accepted"] <= budget]
    if not feasible:
        raise ValueError("No feasible threshold")
    # For a fixed ranking, the lowest feasible threshold maximizes correct
    # acceptance. It also avoids secretly optimizing an extra error objective.
    return dict(
        min(feasible, key=lambda r: r["threshold"]),
        target_fpir=target,
        allowed_unknown_false_accepts=budget,
    )


def outcome(row, threshold):
    if row["status"] != "scoreable":
        return "extraction_failure"
    if row["top_score"] < threshold:
        return "rejected"
    if row["kind"] == "non_mated":
        return "false_accept"
    return "correct_accept" if row["rank1_correct"] else "wrong_accept"


def compare_decisions(rows, points):
    indexed = {
        c: {r["image_id"]: r for r in rows if r["configuration"] == c}
        for c in ["A", "B"]
    }
    if set(indexed["A"]) != set(indexed["B"]):
        raise ValueError("Configurations have different queries")
    changes = []
    for image_id, a in indexed["A"].items():
        b = indexed["B"][image_id]
        changes.append(
            dict(
                image_id=image_id,
                kind=a["kind"],
                true_identity=a["true_identity"],
                outcome_A=outcome(a, points["A"]["threshold"]),
                outcome_B=outcome(b, points["B"]["threshold"]),
                score_A=a["top_score"],
                score_B=b["top_score"],
                predicted_A=a["top_identity"],
                predicted_B=b["top_identity"],
            )
        )
    return changes, indexed


def choose_examples(changes, indexed):
    def score_gain(row):
        return (row["score_B"] or 0) - (row["score_A"] or 0)

    candidates = [
        (
            "Additional reference helps",
            [
                r
                for r in changes
                if r["kind"] == "mated"
                and r["outcome_A"] != "correct_accept"
                and r["outcome_B"] == "correct_accept"
            ],
            score_gain,
        ),
        (
            "A correct decision is lost",
            [
                r
                for r in changes
                if r["kind"] == "mated"
                and r["outcome_A"] == "correct_accept"
                and r["outcome_B"] != "correct_accept"
            ],
            score_gain,
        ),
        (
            "Absent identity falsely accepted",
            [
                r
                for r in changes
                if r["kind"] == "non_mated" and r["outcome_B"] == "false_accept"
            ],
            lambda r: r["score_B"],
        ),
    ]
    examples = []
    for title, available, key in candidates:
        if available:
            chosen = max(available, key=key)
            image_id = chosen["image_id"]
            examples.append(
                dict(
                    title=title,
                    change=chosen,
                    A=indexed["A"][image_id],
                    B=indexed["B"][image_id],
                )
            )
    return examples


def render(output, rows, sweeps, points, examples, images, records):
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from insightface.utils.face_align import norm_crop2

    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {"A": "#69778b", "B": "#007f82"}
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharey=True)
    bins = np.linspace(-1, 1, 81)
    for axis, config in zip(axes, ["A", "B"]):
        usable = [
            r
            for r in rows
            if r["configuration"] == config and r["status"] == "scoreable"
        ]
        for kind, key, label, color in [
            ("mated", "mate_score", "True-identity score (present)", "#007f82"),
            ("non_mated", "top_score", "Highest score (absent)", "#b57422"),
        ]:
            values = [r[key] for r in usable if r["kind"] == kind]
            axis.hist(
                values,
                bins=bins,
                weights=np.ones(len(values)) / len(values),
                alpha=0.55,
                color=color,
                label=f"{label}, n={len(values)}",
            )
        threshold = points[config]["threshold"]
        axis.axvline(
            threshold,
            color="#172d47",
            linestyle="--",
            label=f"Selected threshold = {threshold:.4f}",
        )
        axis.set(xlim=(-0.1, 1), xlabel="Cosine score", title=NAMES[config])
        axis.legend(fontsize=10, loc="upper right")
    axes[0].set_ylabel("Fraction of searches within each group per bin")
    figure.suptitle("Development 1:N search score distributions", fontsize=18)
    figure.text(
        0.03,
        0.015,
        "Present-person curves use the true identity's score; "
        "they do not prove it ranks first. Same query images in both "
        "configurations.\nThresholds separately selected for empirical "
        "FPIR <= 1%. 11 failed queries excluded from these distributions.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    figure.savefig(output / "search_score_distributions.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
    for config in ["A", "B"]:
        frontier = {}
        for r in sweeps[config]:
            count = r["unknown_false_accepted"]
            if count not in frontier or r["rank1_fnir"] < frontier[count]["rank1_fnir"]:
                frontier[count] = r
        ordered = [frontier[k] for k in sorted(frontier)]
        for axis in axes:
            axis.plot(
                [100 * r["fpir"] for r in ordered],
                [100 * r["rank1_fnir"] for r in ordered],
                color=colors[config],
                linewidth=2,
                label=NAMES[config],
            )
            r = points[config]
            axis.scatter(
                100 * r["fpir"],
                100 * r["rank1_fnir"],
                s=65,
                color=colors[config],
                zorder=4,
            )
        r = points[config]
        axes[1].annotate(
            f"{config}: threshold {r['threshold']:.4f}",
            (100 * r["fpir"], 100 * r["rank1_fnir"]),
            xytext=(25, 15 if config == "A" else -42),
            textcoords="offset points",
            color=colors[config],
        )
    axes[0].set(xlim=(0, 100), ylim=(0, 100), title="Full operating range")
    axes[1].set(xlim=(0, 5), ylim=(0, 22), title="Zoom: FPIR 0–5%, FNIR 0–22%")
    for axis in axes:
        axis.set_xlabel("Absent-person searches falsely accepted: FPIR (%)")
        axis.set_ylabel("Present-person searches not correctly accepted: FNIR (%)")
        axis.legend(fontsize=10)
        axis.grid(alpha=0.18)
    figure.suptitle("Rank-1 search tradeoff: lower is better", fontsize=18)
    figure.text(
        0.03,
        0.015,
        "Best FNIR at each observed FPIR count. Dots: empirical "
        "FPIR <= 1% selection. Zero observed errors do not mean zero risk.\n"
        "Development only: 700 absent-person and 3,356 scoreable "
        "present-person queries. No gap; no gallery updates.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    figure.savefig(output / "search_tradeoff.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for axis, key, title in zip(
        axes,
        ["known_correct_accepted", "known_wrong_accepted", "known_rejected"],
        [
            "Correct identity accepted",
            "Wrong identity accepted",
            "No identity accepted",
        ],
    ):
        counts = [points[c][key] for c in ["A", "B"]]
        rates = [
            100 * points[c][key] / points[c]["known_scoreable"] for c in ["A", "B"]
        ]
        axis.bar(["A", "B"], rates, color=[colors[c] for c in ["A", "B"]], width=0.5)
        axis.set(title=title, ylabel="Present-person queries (%)")
        axis.set_ylim(
            0, 115 if key == "known_correct_accepted" else max(1, max(rates) * 1.35)
        )
        for i, (count, rate) in enumerate(zip(counts, rates)):
            axis.annotate(
                f"{rate:.2f}%\n({count:,})",
                (i, rate),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
            )
    figure.suptitle("Same queries, separately calibrated thresholds", fontsize=18)
    figure.text(
        0.03,
        0.015,
        f"A threshold {points['A']['threshold']:.4f}; B threshold "
        f"{points['B']['threshold']:.4f}. Each allows at most 7/700 "
        "absent-person false accepts (1%).\nDenominator: 3,356 scoreable "
        "present-person queries. 11 extraction failures remain separate "
        "(3,367 attempted). Axis scales differ.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.94))
    figure.savefig(output / "policy_outcomes.png", dpi=180)
    plt.close(figure)

    if not examples:
        return
    figure, axes = plt.subplots(
        len(examples), 4, figsize=(14, 3.1 * len(examples)), squeeze=False
    )
    for row_axes, example in zip(axes, examples):
        image_ids = [
            example["change"]["image_id"],
            example["A"]["winning_reference"],
            example["B"]["winning_reference"],
        ]
        roles = ["Query", "A highest reference", "B highest reference"]
        for axis, image_id, role in zip(row_axes, image_ids, roles):
            row = images[image_id]
            if digest(row["path"]) != row["sha256"]:
                raise ValueError("Example image bytes changed")
            record = records[image_id]
            face = record["faces"][record["selected_face_index"]]
            raw = cv2.imread(row["path"])
            crop, _ = norm_crop2(
                raw, np.asarray(face["landmarks"], np.float32), image_size=112
            )
            axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axis.axis("off")
            axis.set_title(role + "\n" + image_id.replace("_", " "), fontsize=10)
        axis = row_axes[3]
        axis.axis("off")
        change = example["change"]
        lines = [
            example["title"],
            "",
            f"A: {change['outcome_A']}",
            f"score {change['score_A']:.4f}",
            f"B: {change['outcome_B']}",
            f"score {change['score_B']:.4f}",
        ]
        axis.text(0, 0.85, "\n".join(lines), va="top", fontsize=12)
    figure.suptitle("Selected development search examples", fontsize=18)
    figure.text(
        0.03,
        0.012,
        "Post-hoc diagnostic examples, not representative "
        "samples. Identities are dataset labels; failure causes need "
        "label/target/crop review.\nDecisions use each configuration's "
        "own 1% FPIR development threshold. Raw scores are not "
        "identity probabilities.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(output / "search_examples.png", dpi=170)
    plt.close(figure)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol["partition"] != "development":
        raise ValueError("Final-test evaluation refused")
    manifest, images, records, report = load_inputs(args.manifest, args.encoded)
    if digest(args.manifest) != protocol["manifest_sha256"]:
        raise ValueError("Protocol input manifest changed")
    if digest(args.encoded / "report.json") != protocol["extraction_report_sha256"]:
        raise ValueError("Protocol extraction changed")
    valid = {k for k, r in records.items() if r["status"] == "valid_embedding"}
    validate_roles(protocol, images, valid)
    vectors = normalize_vectors(args.encoded / "embeddings.npz", valid)
    rows = rank_searches(protocol, images, vectors, records)
    sweeps, selected, configs = {}, {}, {}
    for config in ["A", "B"]:
        subset = [r for r in rows if r["configuration"] == config]
        inputs = metric_inputs(subset)
        sweeps[config] = sweep_thresholds(inputs)
        selected[config] = [
            select_point(sweeps[config], target)
            for target in protocol["selection_targets"]
        ]
        configs[config] = dict(
            references=protocol[f"references_{config}"],
            operating_points=selected[config],
            rank1_wrong_before_threshold=len(inputs["wrong"]),
            top_identity_tied_queries=sum(
                (r["top_identity_ties"] or 0) > 1 for r in subset
            ),
        )
    points = {c: selected[c][0] for c in ["A", "B"]}
    changes, indexed = compare_decisions(rows, points)
    examples = choose_examples(changes, indexed)
    transition_counts = Counter(
        f"{r['kind']}:{r['outcome_A']}->{r['outcome_B']}" for r in changes
    )
    b_inputs = metric_inputs([r for r in rows if r["configuration"] == "B"])
    summary = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(args.protocol),
        evaluator_sha256=digest(__file__),
        partition="development",
        configurations=configs,
        decision_transitions_at_own_1pct_thresholds=dict(transition_counts),
        B_at_A_threshold=threshold_metrics(b_inputs, points["A"]["threshold"]),
        query_counts=protocol["query_counts"],
        query_failures=protocol["query_failures"],
        gallery_identities=protocol["gallery_identities_count"],
        scope=protocol["limitation"],
        source=NIST_API,
        example_selection="Post-hoc diagnostic changes and highest-scoring "
        "unknown false accept; dataset labels not audited.",
        timing_scope="Cached scoring only; no new backbone inference or "
        "production latency benchmark.",
    )
    args.output.mkdir(parents=True)
    csv_write(args.output / "ranked_queries.csv", rows, list(rows[0]))
    sweep_rows = [dict(configuration=c, **r) for c in ["A", "B"] for r in sweeps[c]]
    csv_write(args.output / "threshold_sweep.csv", sweep_rows, list(sweep_rows[0]))
    csv_write(args.output / "decision_changes.csv", changes, list(changes[0]))
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "examples.json", examples)
    render(args.output, rows, sweeps, points, examples, images, records)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate"])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lfw_matching/development/manifest.json"),
    )
    parser.add_argument(
        "--encoded", type=Path, default=Path("artifacts/casia_matching_encoded")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("artifacts/casia_search_protocol.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/casia_search_development")
    )
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
