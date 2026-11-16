"""Sequential development-only replay with automatic enrollment and updates.

The matcher receives image IDs and cached scores, never true identity labels.
Each parameter setting starts empty. Labels belong only to the evaluator.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.pairwise import csv_write, digest, save_json
from face_velocity.inputs import load_inputs, normalize_vectors

BASELINE_THRESHOLD = 0.39579098336768626
ACCEPTANCE_THRESHOLDS = [0.35, 0.375, BASELINE_THRESHOLD, 0.425, 0.45]
LOCAL_ACCEPTANCE_THRESHOLDS = sorted(
    [
        0.385,
        0.395,
        BASELINE_THRESHOLD,
        0.405,
        0.415,
        0.425,
        0.435,
    ]
)
LOWER_ACCEPTANCE_THRESHOLDS = [0.365, 0.375]
EXTENDED_ACCEPTANCE_THRESHOLDS = sorted(
    LOWER_ACCEPTANCE_THRESHOLDS + LOCAL_ACCEPTANCE_THRESHOLDS
)
# Original search plus the four development-only supplemental thresholds.
PRESENTATION_ACCEPTANCE_THRESHOLDS = sorted(
    [0.325, 0.335, 0.345, 0.355] + EXTENDED_ACCEPTANCE_THRESHOLDS
)
UPDATE_DELTAS = [None, 0.0, 0.05, 0.10, 0.15]
REFERENCE = (
    "https://www.nist.gov/system/files/documents/2020/12/15/"
    "342_goit_supporting_paper.pdf"
)


@dataclass(frozen=True)
class Policy:
    acceptance: float
    update_delta: float | None

    @property
    def name(self):
        suffix = "none" if self.update_delta is None else f"{self.update_delta:.2f}"
        return f"a{self.acceptance:.6f}_u{suffix}"

    @property
    def update_threshold(self):
        return (
            None if self.update_delta is None else self.acceptance + self.update_delta
        )


class Scores:
    """Immutable cache; access is restricted to already admitted references."""

    def __init__(self, image_ids, matrix):
        self.image_ids = list(image_ids)
        self.index = {image_id: i for i, image_id in enumerate(image_ids)}
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.shape != (len(image_ids), len(image_ids)):
            raise ValueError("Score matrix shape does not match image IDs")
        if not np.isfinite(self.matrix).all():
            raise ValueError("Non-finite cached scores")
        self.matrix.flags.writeable = False

    def against(self, image_id, reference_indices):
        return self.matrix[self.index[image_id], reference_indices]


class DynamicGallery:
    """Label-free, append-only references; windowed, idempotent event counts."""

    def __init__(self, scores, policy, window_seconds=86400):
        if not -1 <= policy.acceptance <= 1 or window_seconds <= 0:
            raise ValueError("Invalid acceptance threshold or window")
        if policy.update_delta is not None and policy.update_delta < 0:
            raise ValueError("Update threshold cannot be below acceptance")
        self.scores, self.policy, self.window = scores, policy, window_seconds
        self.references = []
        self.reference_indices = []
        self.reference_groups = []
        self.group_events = []
        self.group_initial_images = []
        self.saved_events = {}
        self.latest_time = float("-inf")

    def process(self, event_id, image_id, seconds):
        signature = (image_id, seconds)
        if event_id in self.saved_events:
            previous_signature, result = self.saved_events[event_id]
            if signature != previous_signature:
                raise ValueError("Existing event ID has different image/time")
            return dict(result, is_retry=True)
        if seconds < self.latest_time:
            raise ValueError("New events must be chronological")
        self.latest_time = seconds
        valid = image_id in self.scores.index
        group, candidate, score, winner = None, None, None, None
        added, new_group = False, False
        if valid:
            if self.references:
                values = self.scores.against(image_id, self.reference_indices)
                score = float(values.max())
                tied = np.flatnonzero(values == score)
                # Rank identities by Max score, then earliest created group.
                winning_index = min(tied, key=lambda i: (self.reference_groups[i], i))
                candidate = self.reference_groups[winning_index]
                winner = self.references[winning_index]
            if score is not None and score >= self.policy.acceptance:
                group = candidate
                added = (
                    self.policy.update_threshold is not None
                    and score >= self.policy.update_threshold
                )
            else:
                group = len(self.group_events)
                self.group_events.append(deque())
                self.group_initial_images.append(image_id)
                new_group = added = True
            # The query cannot participate in its own search above.
            if added:
                self.references.append(image_id)
                self.reference_indices.append(self.scores.index[image_id])
                self.reference_groups.append(group)
            events = self.group_events[group]
            events.append((seconds, event_id))
            while events and events[0][0] <= seconds - self.window:
                events.popleft()
            member_ids = tuple(event for _, event in events)
        else:
            member_ids = ()
        result = dict(
            event_id=event_id,
            image_id=image_id,
            seconds=seconds,
            assigned_group=group,
            candidate_group=candidate,
            top_score=score,
            winning_reference=winner,
            new_group=new_group,
            reference_added=added,
            is_retry=False,
            decision="unresolved" if not valid else "enroll" if new_group else "accept",
            predicted_count=len(member_ids) if valid else None,
            predicted_event_ids=member_ids,
            group_initial_image=self.group_initial_images[group] if valid else None,
            groups=len(self.group_events),
            references=len(self.references),
        )
        self.saved_events[event_id] = (signature, result)
        return result


def membership_errors(predicted, expected):
    predicted, expected = set(predicted), set(expected)
    return len(predicted - expected), len(expected - predicted)


def evaluate_stream(images, scores, policy, order, seed, interval=30, window=86400):
    """Truth never enters DynamicGallery; IDs are evaluated by event membership."""
    engine = DynamicGallery(scores, policy, window)
    true_windows = defaultdict(deque)
    seen_usable, anchors, anchor_events = set(), {}, {}
    represented_identities = set()
    valid_counts = Counter(
        r["identity"] for k, r in images.items() if k in scores.index
    )
    rows = []
    for i, image_id in enumerate(order):
        event_id, seconds = f"e{i:05d}", i * interval
        truth = images[image_id]["identity"]
        true_events = true_windows[truth]
        true_events.append((seconds, event_id))
        while true_events and true_events[0][0] <= seconds - window:
            true_events.popleft()
        expected = tuple(e for _, e in true_events)
        # Evaluate presence BEFORE processing/enrollment. A contaminated reference
        # does not rename its group. Multiple groups with the same anchor all
        # represent that identity for matching; membership errors expose splits.
        present = truth in represented_identities
        result = engine.process(event_id, image_id, seconds)
        valid = result["assigned_group"] is not None
        first = valid and truth not in seen_usable
        group = result["assigned_group"]
        if result["new_group"]:
            anchors[group], anchor_events[group] = truth, event_id
            represented_identities.add(truth)
        if valid:
            extra, missed = membership_errors(result["predicted_event_ids"], expected)
            wrong = anchors[group] != truth
            seen_usable.add(truth)
            absolute_error = abs(result["predicted_count"] - len(expected))
        else:
            extra = missed = wrong = absolute_error = None
        rows.append(
            dict(
                seed=seed,
                policy=policy.name,
                **result,
                true_identity=truth,
                scoreable=valid,
                first_usable_arrival=first,
                returning=valid and not first,
                photo_rich=valid_counts[truth] >= 3,
                expected_count=len(expected),
                expected_event_ids=expected,
                extra_events=extra,
                missed_events=missed,
                absolute_count_error=absolute_error,
                count_exact=valid and absolute_error == 0,
                membership_exact=valid and extra == 0 and missed == 0,
                wrong_anchor_assignment=bool(valid and wrong),
                first_arrival_false_merge=bool(first and not result["new_group"]),
                returning_new_group=bool(valid and not first and result["new_group"]),
                wrong_reference_added=bool(
                    valid and result["reference_added"] and wrong
                ),
                anchor_event=anchor_events.get(group),
                gallery_identity_present_before=present,
                dynamic_false_positive=bool(
                    valid and not present and not result["new_group"]
                ),
                dynamic_false_negative=bool(
                    valid and present and (result["new_group"] or wrong)
                ),
                present_rejection=bool(valid and present and result["new_group"]),
                present_wrong_accept=bool(
                    valid and present and not result["new_group"] and wrong
                ),
            )
        )
    return summarize(rows, policy, engine), rows


def summarize(rows, policy, engine):
    valid = [r for r in rows if r["scoreable"]]
    first = [r for r in valid if r["first_usable_arrival"]]
    returning = [r for r in valid if r["returning"]]
    present = [r for r in valid if r["gallery_identity_present_before"]]
    absent = [r for r in valid if not r["gallery_identity_present_before"]]
    false_positives = sum(r["dynamic_false_positive"] for r in absent)
    false_negatives = sum(r["dynamic_false_negative"] for r in present)

    def measures(subset):
        if not subset:
            return dict(
                events=0,
                membership_cost=None,
                count_mae=None,
                count_exact_rate=None,
                membership_exact_rate=None,
            )
        n = len(subset)
        return dict(
            events=n,
            membership_cost=sum(r["extra_events"] + r["missed_events"] for r in subset)
            / n,
            extra_per_event=sum(r["extra_events"] for r in subset) / n,
            missed_per_event=sum(r["missed_events"] for r in subset) / n,
            count_mae=sum(r["absolute_count_error"] for r in subset) / n,
            count_exact_rate=sum(r["count_exact"] for r in subset) / n,
            membership_exact_rate=sum(r["membership_exact"] for r in subset) / n,
        )

    result = dict(
        seed=rows[0]["seed"],
        policy=policy.name,
        acceptance=policy.acceptance,
        update_delta=policy.update_delta,
        update_threshold=policy.update_threshold,
        attempted=len(rows),
        scoreable=len(valid),
        unresolved=len(rows) - len(valid),
        coverage=len(valid) / len(rows),
        **measures(valid),
        all_input_count_exact_rate=sum(r["count_exact"] for r in rows) / len(rows),
        first_usable_arrivals=len(first),
        returning_events=len(returning),
        gallery_present_queries=len(present),
        gallery_absent_queries=len(absent),
        dynamic_false_positives=false_positives,
        dynamic_false_negatives=false_negatives,
        dynamic_fpir=false_positives / len(absent) if absent else None,
        dynamic_rank1_fnir=false_negatives / len(present) if present else None,
        present_rejections=sum(r["present_rejection"] for r in present),
        present_wrong_accepts=sum(r["present_wrong_accept"] for r in present),
        first_arrival_false_merges=sum(r["first_arrival_false_merge"] for r in first),
        first_arrival_false_merge_rate=(
            sum(r["first_arrival_false_merge"] for r in first) / len(first)
        ),
        returning_new_groups=sum(r["returning_new_group"] for r in returning),
        returning_new_group_rate=(
            sum(r["returning_new_group"] for r in returning) / len(returning)
            if returning
            else None
        ),
        wrong_anchor_assignments=sum(r["wrong_anchor_assignment"] for r in valid),
        wrong_appended_references=sum(r["wrong_reference_added"] for r in valid),
        appended_references=sum(
            r["reference_added"] and not r["new_group"] for r in valid
        ),
        final_groups=len(engine.group_events),
        final_references=len(engine.references),
        max_references_per_group=max(Counter(engine.reference_groups).values()),
        returning_metrics=measures(returning),
        photo_rich_returning_metrics=measures(
            [r for r in returning if r["photo_rich"]]
        ),
        count_cancellation_events=sum(
            r["count_exact"] and not r["membership_exact"] for r in valid
        ),
    )
    return result


def aggregate(summaries, selection="count"):
    groups = defaultdict(list)
    for row in summaries:
        groups[row["policy"]].append(row)
    combined = []
    for name, rows in groups.items():
        item = {
            k: rows[0][k]
            for k in ["policy", "acceptance", "update_delta", "update_threshold"]
        }
        item["feasible"] = all(
            r["first_arrival_false_merge_rate"] <= 0.01 for r in rows
        )
        for key in [
            "membership_cost",
            "extra_per_event",
            "missed_per_event",
            "count_mae",
            "count_exact_rate",
            "membership_exact_rate",
            "all_input_count_exact_rate",
            "first_arrival_false_merge_rate",
            "returning_new_group_rate",
            "wrong_appended_references",
            "wrong_anchor_assignments",
            "dynamic_fpir",
            "dynamic_rank1_fnir",
            "gallery_present_queries",
            "gallery_absent_queries",
            "dynamic_false_positives",
            "dynamic_false_negatives",
            "present_rejections",
            "present_wrong_accepts",
            "final_groups",
            "final_references",
            "count_cancellation_events",
        ]:
            values = [r[key] for r in rows if r[key] is not None]
            for suffix, function in [("mean", np.mean), ("min", min), ("max", max)]:
                item[f"{key}_{suffix}"] = float(function(values)) if values else None
        for stratum in ["returning_metrics", "photo_rich_returning_metrics"]:
            values = [
                r[stratum]["membership_cost"]
                for r in rows
                if r[stratum]["membership_cost"] is not None
            ]
            item[f"{stratum}_cost_mean"] = float(np.mean(values)) if values else None
        combined.append(item)
    if selection == "count":
        return combined, min(
            combined, key=lambda r: (r["membership_cost_mean"], r["policy"])
        )
    if selection != "first-arrival-budget":
        raise ValueError("Unknown selection criterion")
    feasible = [r for r in combined if r["feasible"]]
    selected = (
        min(
            feasible,
            key=lambda r: (
                r["membership_cost_mean"],
                r["first_arrival_false_merge_rate_max"],
                r["final_references_mean"],
                r["policy"],
            ),
        )
        if feasible
        else None
    )
    return combined, selected


def write_trace(path, rows):
    csv_write(
        path,
        [
            dict(
                r,
                predicted_event_ids="|".join(r["predicted_event_ids"]),
                expected_event_ids="|".join(r["expected_event_ids"]),
            )
            for r in rows
        ],
        list(rows[0]),
    )


def read_trace(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    boolean = [
        "scoreable",
        "reference_added",
        "new_group",
        "first_usable_arrival",
        "returning",
        "count_exact",
        "membership_exact",
        "gallery_identity_present_before",
        "dynamic_false_positive",
        "dynamic_false_negative",
        "present_rejection",
        "present_wrong_accept",
    ]
    numeric = [
        "seconds",
        "predicted_count",
        "expected_count",
        "extra_events",
        "missed_events",
        "references",
        "groups",
        "top_score",
    ]
    for r in rows:
        for key in boolean:
            if key in r:
                r[key] = r[key] == "True"
        for key in numeric:
            r[key] = float(r[key]) if r[key] else None
    return rows


def plot_results(
    output, combined, selected, seeds, images, selection="first-arrival-budget"
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures = output / "figures"
    figures.mkdir()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 220,
        }
    )
    colors = ["#64748b", "#c47b16", "#007f82", "#4074b8", "#7c5295"]
    thresholds = sorted({r["acceptance"] for r in combined})

    def label(delta):
        return (
            "No appended references"
            if delta is None
            else (
                "Append every accepted event"
                if delta == 0
                else f"Update threshold = acceptance + {delta:.2f}"
            )
        )

    def save(fig, name):
        fig.savefig(figures / (name + ".png"), bbox_inches="tight")
        fig.savefig(figures / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    keys = ["membership_cost", "first_arrival_false_merge_rate", "count_mae"]
    titles = [
        "Extra + missed events per scored query",
        "First-arrival false-merge rate",
        "Count mean absolute error",
    ]
    for delta, color in zip(UPDATE_DELTAS, colors):
        group = sorted(
            [r for r in combined if r["update_delta"] == delta],
            key=lambda r: r["acceptance"],
        )
        xs = [r["acceptance"] for r in group]
        for ax, key in zip(axes, keys):
            ax.plot(
                xs,
                [r[key + "_mean"] for r in group],
                marker="o",
                color=color,
                label=label(delta),
            )
            ax.fill_between(
                xs,
                [r[key + "_min"] for r in group],
                [r[key + "_max"] for r in group],
                alpha=0.12,
                color=color,
            )
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel="Acceptance threshold", ylim=(0, None))
        ax.grid(alpha=0.2)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    if selection == "first-arrival-budget":
        axes[1].axhline(0.01, color="black", ls="--", lw=1, label="Assumed 1% limit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(
        f"Dynamic development replay: means and ranges across {len(seeds)} input orders"
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    save(fig, "policy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key, title, percent in [
        (
            axes[0],
            "membership_cost_mean",
            "Mean extra + missed events per scored query",
            False,
        ),
        (
            axes[1],
            "first_arrival_false_merge_rate_max",
            "Worst-order first-arrival false-merge rate",
            True,
        ),
    ]:
        values = np.array(
            [
                [
                    next(
                        r[key]
                        for r in combined
                        if r["acceptance"] == a and r["update_delta"] == d
                    )
                    for a in thresholds
                ]
                for d in UPDATE_DELTAS
            ]
        )
        mesh = ax.imshow(values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(thresholds)), [f"{a:.3f}" for a in thresholds])
        ax.set_yticks(range(5), ["No append", "Append all", "+0.05", "+0.10", "+0.15"])
        ax.set(
            xlabel="Acceptance threshold", ylabel="Reference update rule", title=title
        )
        for y, d in enumerate(UPDATE_DELTAS):
            for x, a in enumerate(thresholds):
                row = next(
                    r
                    for r in combined
                    if r["acceptance"] == a and r["update_delta"] == d
                )
                value = f"{values[y, x]:.2%}" if percent else f"{values[y, x]:.3f}"
                ax.text(
                    x,
                    y,
                    value
                    + (
                        " *"
                        if selection == "first-arrival-budget" and row["feasible"]
                        else ""
                    ),
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        fig.colorbar(
            mesh, ax=ax, shrink=0.7, format=PercentFormatter(1) if percent else None
        )
    fig.suptitle(
        "* Meets the assumed 1% first-arrival false-merge limit in every input order"
        if selection == "first-arrival-budget"
        else (
            "Count-objective search; first-arrival errors are diagnostic, not "
            "a selection filter"
        )
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "parameter_grid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for delta, color in zip(UPDATE_DELTAS, colors):
        group = sorted(
            [r for r in combined if r["update_delta"] == delta],
            key=lambda r: r["acceptance"],
        )
        xs = [r["acceptance"] for r in group]
        for ax, key in zip(axes, ["dynamic_fpir", "dynamic_rank1_fnir"]):
            ax.plot(
                xs,
                [r[key + "_mean"] for r in group],
                marker="o",
                color=color,
                label=label(delta),
            )
            ax.fill_between(
                xs,
                [r[key + "_min"] for r in group],
                [r[key + "_max"] for r in group],
                alpha=0.12,
                color=color,
            )
    for ax, title in zip(
        axes,
        [
            "Dynamic FPIR: accept when identity is absent",
            "Dynamic rank-1 FNIR: fail when identity is present",
        ],
    ):
        ax.set(title=title, xlabel="Acceptance threshold", ylim=(0, None))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10
    )
    fig.suptitle("Identity matching in the evolving gallery")
    fig.text(
        0.5,
        0.89,
        (
            "Presence before each query, using fixed enrollment-identity "
            "labels for groups."
        ),
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.87))
    save(fig, "dynamic_matching")

    # A paired inspection at one acceptance threshold separates update effects.
    chosen = selected or min(combined, key=lambda r: r["membership_cost_mean"])
    acceptance = chosen["acceptance"]
    deltas = list(dict.fromkeys([None, 0.0, chosen["update_delta"]]))
    seed = seeds[0]
    traces = {
        d: read_trace(output / "traces" / f"{Policy(acceptance, d).name}_s{seed}.csv")
        for d in deltas
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for d, rows in traces.items():
        color = colors[UPDATE_DELTAS.index(d)]
        loss = np.cumsum(
            [(r["extra_events"] or 0) + (r["missed_events"] or 0) for r in rows]
        )
        axes[0].plot(range(1, len(rows) + 1), loss, label=label(d), color=color)
        axes[1].plot(
            range(1, len(rows) + 1),
            [r["references"] for r in rows],
            label=label(d),
            color=color,
        )
    axes[0].set(
        title="Cumulative extra + missed event memberships",
        xlabel="Input event",
        ylabel="Cumulative error",
    )
    axes[1].set(title="Reference growth", xlabel="Input event", ylabel="References")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10
    )
    fig.suptitle(f"Paired replay: acceptance {acceptance:.3f}, input-order seed {seed}")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, "replay_progression")

    adaptive_delta = (
        chosen["update_delta"] if chosen["update_delta"] is not None else 0.0
    )
    baseline_rows, adaptive_rows = traces[None], traces[adaptive_delta]
    image_events = {r["image_id"]: r for r in adaptive_rows}
    examples = []
    for direction in ["helps", "hurts"]:
        for base, adaptive in zip(baseline_rows, adaptive_rows):
            if not base["scoreable"] or not adaptive["scoreable"]:
                continue
            change = (
                base["extra_events"]
                + base["missed_events"]
                - adaptive["extra_events"]
                - adaptive["missed_events"]
            )
            ref = image_events.get(adaptive["winning_reference"])
            if (
                not ref
                or not ref["reference_added"]
                or ref["new_group"]
                or not (
                    (change > 0 and direction == "helps")
                    or (change < 0 and direction == "hurts")
                )
            ):
                continue
            examples.append(
                dict(
                    direction=direction,
                    seed=seed,
                    acceptance=acceptance,
                    update_delta=adaptive_delta,
                    baseline=base,
                    adaptive=adaptive,
                    reference_admission=ref,
                    selection=(
                        f"First matching event in seed{seed} with a winning "
                        "appended reference and lower/higher membership loss "
                        "than no-append at the same acceptance threshold."
                    ),
                )
            )
            break
    save_json(output / "examples.json", examples)
    for example in examples:
        base, adaptive, admission = (
            example["baseline"],
            example["adaptive"],
            example["reference_admission"],
        )
        ids = [
            admission["group_initial_image"],
            admission["image_id"],
            adaptive["image_id"],
            base["winning_reference"],
        ]
        captions = [
            "Group's enrollment reference",
            f"Appended reference: {admission['event_id']}",
            f"Later query: {adaptive['event_id']}",
            "No-append winning reference",
        ]
        fig, axes = plt.subplots(1, 4, figsize=(13, 5.5))
        for ax, image_id, caption in zip(axes, ids, captions):
            if image_id:
                ax.imshow(plt.imread(images[image_id]["path"]))
            ax.set_title(caption, fontsize=11)
            ax.axis("off")
        detail = (
            f"No append: count {base['predicted_count']:.0f}, "
            f"extra {base['extra_events']:.0f}, "
            f"missed {base['missed_events']:.0f}.   "
            f"Update: count {adaptive['predicted_count']:.0f}, "
            f"extra {adaptive['extra_events']:.0f}, "
            f"missed {adaptive['missed_events']:.0f}.\n"
            f"True count {adaptive['expected_count']:.0f}. "
            f"Reference admitted at score {admission['top_score']:.3f}. "
            f"Later query score {adaptive['top_score']:.3f}. "
            f"Seed {seed}; acceptance {acceptance:.3f}; "
            f"update delta {adaptive_delta:.2f}."
        )
        fig.suptitle(
            f"An appended reference {example['direction']} a later decision",
            fontsize=16,
        )
        fig.text(0.05, 0.10, detail, fontsize=11)
        fig.tight_layout(rect=(0, 0.2, 1, 0.90))
        save(fig, "case_" + example["direction"])


def run(args):
    if args.output.exists():
        raise FileExistsError(
            "Use a new output directory; existing results are retained"
        )
    manifest, images, records, report = load_inputs(args.manifest, args.encoded)
    valid_ids = sorted(
        k for k, r in records.items() if r["status"] == "valid_embedding"
    )
    vectors = normalize_vectors(args.encoded / "embeddings.npz", valid_ids)
    matrix = np.stack([vectors[k] for k in valid_ids])
    print(
        f"Preparing immutable cosine cache for {len(valid_ids)} valid images",
        flush=True,
    )
    scores = Scores(valid_ids, np.clip(matrix @ matrix.T, -1, 1))
    args.output.mkdir(parents=True)
    (args.output / "traces").mkdir()
    thresholds = {
        "local": LOCAL_ACCEPTANCE_THRESHOLDS,
        "lower": LOWER_ACCEPTANCE_THRESHOLDS,
        "extended": EXTENDED_ACCEPTANCE_THRESHOLDS,
        "presentation": PRESENTATION_ACCEPTANCE_THRESHOLDS,
        "coarse": ACCEPTANCE_THRESHOLDS,
    }[args.grid]
    policies = [Policy(a, d) for a in thresholds for d in UPDATE_DELTAS]
    orders = {
        seed: np.random.default_rng(seed).permutation(sorted(images)).tolist()
        for seed in args.seeds
    }
    protocol = dict(
        partition="development",
        manifest_sha256=digest(args.manifest),
        extraction_report_sha256=digest(args.encoded / "report.json"),
        evaluator_sha256=digest(__file__),
        models=report["configuration"]["models"],
        image_count=len(images),
        identity_count=len({r["identity"] for r in images.values()}),
        scoreable=len(valid_ids),
        failures=len(images) - len(valid_ids),
        orders={str(k): v for k, v in orders.items()},
        interval_seconds=30,
        window_seconds=86400,
        grid=args.grid,
        acceptance_thresholds=thresholds,
        update_deltas=UPDATE_DELTAS,
        initial_gallery="empty",
        aggregation="Max",
        candidate_rank=1,
        reference_limit=None,
        reference_retention="All admitted references persist beyond the count window",
        update_rule=(
            "Accept score>=tau_a; append accepted event if score>=tau_a+delta. "
            "None disables append. New groups always enroll first valid "
            "reference."
        ),
        ties=(
            "Earliest created group, then earliest admitted reference. "
            "Acceptance and update inclusive."
        ),
        chronology=(
            "Random unique-image permutation, simulated 30-second intervals; "
            "not actual LFW capture chronology."
        ),
        failures_rule=(
            "Record unresolved, no enrollment/update/count output. Include in "
            "true histories, report coverage and all-input exact accuracy."
        ),
        identity_metrics=(
            "Group anchor is true identity of its first event, evaluator-only. "
            "First-arrival is first usable event of that true identity. "
            "Returning new-group creation measures fragmentation, including "
            "corrective enrollment after an earlier false merge."
        ),
        dynamic_matching_metrics={
            "presence": (
                "Before this query, at least one existing group has an enrollment "
                "anchor equal to the true query identity. Anchors never change; "
                "contamination does not relabel groups. Any same-anchor group is a "
                "correct identity for matching, even if history is fragmented."
            ),
            "fpir": (
                "Accepted existing group on a scored gallery-absent query / all "
                "scored gallery-absent queries. Empty gallery included as a valid "
                "reject/enroll opportunity."
            ),
            "rank1_fnir": (
                "Reject all existing groups or accept a wrong-anchor group on a "
                "scored gallery-present query / all scored gallery-present "
                "queries. Report rejections and wrong accepts separately."
            ),
            "failures": (
                "Exclude unresolved extraction from these conditional rates, "
                "report coverage separately. Empty denominators produce null, "
                "never zero."
            ),
            "comparison": (
                "Custom sequential extension of search metrics, not NIST "
                "evaluation. Policy-dependent gallery membership and denominators; "
                "report raw counts per order. First-arrival false-merge remains a "
                "distinct selection constraint."
            ),
        },
        count_metrics=(
            "Current event included; window (t-24h,t]. Compare event-ID "
            "memberships, including failed events in truth. Extra and missed "
            "reported separately. Primary cost conditional on scoreable "
            "events; unresolved never counted correct."
        ),
        selection=(
            "Require first-usable-arrival false-merge<=1% in every seed, then "
            "minimize mean(extra+missed) per scoreable event with equal "
            "weights. Tie: lower worst false-merge, fewer references, policy "
            "name. If infeasible, select nothing."
        ),
        limitation=(
            "Development exploration, dependent observations, only three "
            "orders; dynamic matching uses fixed enrollment-anchor labels and "
            "changing denominators, not a NIST benchmark or fixed-gallery "
            "operating guarantee. Not independent test, real aging, live API "
            "or production risk evidence."
        ),
        source=REFERENCE,
        matching_metric_source="https://pages.nist.gov/frvt/html/frvt1N.html",
    )
    if args.selection == "count":
        protocol["selection"] = (
            "Minimize mean(extra+missed) per scoreable event across input "
            "orders, equal weights, no FPIR or first-arrival filter. Tie: "
            "policy name."
        )
        protocol["dynamic_matching_metrics"]["comparison"] = (
            "Custom sequential extension of search metrics, not NIST "
            "evaluation. Policy-dependent gallery membership and denominators; "
            "report raw counts per order. Matching rates and first-arrival "
            "false merges are diagnostic, not selection constraints."
        )
    save_json(args.output / "protocol.json", protocol)
    summaries, started = [], perf_counter()
    for seed, order in orders.items():
        for policy in policies:
            summary, rows = evaluate_stream(images, scores, policy, order, seed)
            write_trace(args.output / "traces" / f"{policy.name}_s{seed}.csv", rows)
            summaries.append(summary)
            print(
                f"{len(summaries):02}/{len(policies) * len(orders)} "
                f"seed={seed} {policy.name} "
                f"cost={summary['membership_cost']:.4f} "
                f"first_merge={summary['first_arrival_false_merge_rate']:.3%} "
                f"refs={summary['final_references']}",
                flush=True,
            )
    combined, selected = aggregate(summaries, args.selection)
    save_json(
        args.output / "summary.json",
        dict(
            protocol_sha256=digest(args.output / "protocol.json"),
            created_at=datetime.now(timezone.utc).isoformat(),
            per_replay=summaries,
            configurations=combined,
            selected=selected,
            selection_status=(
                "count-objective development candidate"
                if args.selection == "count"
                else "feasible development candidate"
                if selected
                else "no configuration meets the declared constraint"
            ),
            elapsed_replay_seconds=perf_counter() - started,
            scope=protocol["limitation"],
        ),
    )
    csv_write(args.output / "configuration_comparison.csv", combined, list(combined[0]))
    plot_results(args.output, combined, selected, args.seeds, images, args.selection)
    print(
        json.dumps(dict(output=str(args.output), selected=selected), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lfw_matching/development/manifest.json"),
    )
    parser.add_argument(
        "--encoded", type=Path, default=Path("artifacts/casia_matching_encoded")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--grid",
        choices=["coarse", "local", "lower", "extended", "presentation"],
        default="extended",
    )
    parser.add_argument(
        "--selection",
        choices=["first-arrival-budget", "count"],
        default="count",
    )
    run(parser.parse_args())
