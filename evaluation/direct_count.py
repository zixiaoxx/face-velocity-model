"""Development-only comparison of direct event retrieval and identity counting.

All usable prior events may be retrieved, regardless of earlier matching results.
The threshold sweep uses sorted scores because admission is threshold-independent.
Selected settings are replayed explicitly to retain inspectable event histories.
"""

import argparse
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.dynamic import Scores, membership_errors, write_trace
from evaluation.pairwise import csv_write, digest, save_json
from face_velocity.inputs import load_inputs, normalize_vectors

THRESHOLDS = np.arange(-200, 201, dtype=np.float64) / 200


class DirectEventCounter:
    """Label-free retrieval, one count per event ID, with inclusive similarity."""

    def __init__(self, scores, threshold, window=86400):
        if not -1 <= threshold <= 1 or window <= 0:
            raise ValueError("Invalid threshold or window")
        self.scores, self.threshold, self.window = scores, threshold, window
        self.events, self.saved = deque(), {}
        self.latest = float("-inf")

    def process(self, event_id, image_id, seconds):
        signature = (image_id, seconds)
        if event_id in self.saved:
            previous, result = self.saved[event_id]
            if signature != previous:
                raise ValueError("Existing event ID has different image/time")
            return dict(result, is_retry=True)
        if seconds < self.latest:
            raise ValueError("New events must be chronological")
        self.latest = seconds
        while self.events and self.events[0][0] <= seconds - self.window:
            self.events.popleft()
        valid = image_id in self.scores.index
        prior = []
        if valid:
            indices = [self.scores.index[k] for _, _, k in self.events]
            values = self.scores.against(image_id, indices)
            prior = [
                e for (_, e, _), v in zip(self.events, values) if v >= self.threshold
            ]
            predicted = tuple(prior + [event_id])
            # Every usable event is retained, even if no prior event matched it.
            self.events.append((seconds, event_id, image_id))
        else:
            predicted = ()
        result = dict(
            event_id=event_id,
            image_id=image_id,
            seconds=seconds,
            scoreable=valid,
            is_retry=False,
            predicted_count=len(predicted) if valid else None,
            predicted_event_ids=predicted,
        )
        self.saved[event_id] = (signature, result)
        return result


def sweep_order(images, scores, order, thresholds, interval=30, window=86400):
    """Exact membership/count metrics for a unique-image stream at all thresholds."""
    if len(order) != len(set(order)):
        raise ValueError("The sweep expects one distinct event per image")
    counts = {
        k: np.zeros(len(thresholds), dtype=np.int64)
        for k in [
            "extra",
            "missed",
            "absolute",
            "exact",
            "membership_exact",
            "returning_extra",
            "returning_missed",
            "returning_exact",
        ]
    }
    history, truth = deque(), defaultdict(deque)
    seen, valid_count, returning_count = set(), 0, 0
    for i, image_id in enumerate(order):
        now, identity = i * interval, images[image_id]["identity"]
        true_events = truth[identity]
        while true_events and true_events[0] <= now - window:
            true_events.popleft()
        true_prior = len(true_events)
        true_events.append(now)
        while history and history[0][0] <= now - window:
            history.popleft()
        if image_id not in scores.index:
            continue
        prior_ids = [k for _, k in history]
        values = scores.against(image_id, [scores.index[k] for k in prior_ids])
        same = np.array(
            [images[k]["identity"] == identity for k in prior_ids], dtype=bool
        )
        genuine, impostor = np.sort(values[same]), np.sort(values[~same])
        # searchsorted(left) includes exact equality, consistent with >= threshold.
        tp = len(genuine) - np.searchsorted(genuine, thresholds, side="left")
        extra = len(impostor) - np.searchsorted(impostor, thresholds, side="left")
        missed = true_prior - tp
        counts["extra"] += extra
        counts["missed"] += missed
        counts["absolute"] += np.abs(extra - missed)
        counts["exact"] += extra == missed
        counts["membership_exact"] += (extra == 0) & (missed == 0)
        if identity in seen:
            returning_count += 1
            counts["returning_extra"] += extra
            counts["returning_missed"] += missed
            counts["returning_exact"] += extra == missed
        seen.add(identity)
        valid_count += 1
        history.append((now, image_id))
    rows = []
    for j, threshold in enumerate(thresholds):
        value = {k: int(v[j]) for k, v in counts.items()}
        rows.append(
            dict(
                threshold=float(threshold),
                attempted=len(order),
                scoreable=valid_count,
                unresolved=len(order) - valid_count,
                returning_events=returning_count,
                **value,
                extra_per_event=value["extra"] / valid_count,
                missed_per_event=value["missed"] / valid_count,
                membership_cost=(value["extra"] + value["missed"]) / valid_count,
                count_mae=value["absolute"] / valid_count,
                count_exact_rate=value["exact"] / valid_count,
                membership_exact_rate=value["membership_exact"] / valid_count,
                all_input_count_exact_rate=value["exact"] / len(order),
                returning_cost=(
                    (value["returning_extra"] + value["returning_missed"])
                    / returning_count
                    if returning_count
                    else None
                ),
                returning_count_exact_rate=(
                    value["returning_exact"] / returning_count
                    if returning_count
                    else None
                ),
            )
        )
    return rows


def replay(images, scores, order, seed, threshold, interval=30, window=86400):
    counter = DirectEventCounter(scores, threshold, window)
    truth, seen, rows = defaultdict(deque), set(), []
    for i, image_id in enumerate(order):
        now, event_id = i * interval, f"e{i:05d}"
        identity = images[image_id]["identity"]
        expected = truth[identity]
        expected.append((now, event_id))
        while expected and expected[0][0] <= now - window:
            expected.popleft()
        expected_ids = tuple(e for _, e in expected)
        result = counter.process(event_id, image_id, now)
        valid = result["scoreable"]
        extra, missed = (
            membership_errors(result["predicted_event_ids"], expected_ids)
            if valid
            else (None, None)
        )
        error = abs(result["predicted_count"] - len(expected)) if valid else None
        rows.append(
            dict(
                seed=seed,
                threshold=threshold,
                **result,
                true_identity=identity,
                returning=valid and identity in seen,
                expected_count=len(expected),
                expected_event_ids=expected_ids,
                extra_events=extra,
                missed_events=missed,
                absolute_count_error=error,
                count_exact=valid and error == 0,
                membership_exact=valid and extra == 0 and missed == 0,
            )
        )
        if valid:
            seen.add(identity)
    return rows


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    _, images, records, report = load_inputs(args.manifest, args.encoded)
    source_protocol = json.loads(args.baseline_protocol.read_text())
    baseline = json.loads(args.baseline_summary.read_text())
    if source_protocol["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Baseline used a different development manifest")
    if source_protocol["extraction_report_sha256"] != digest(
        args.encoded / "report.json"
    ):
        raise ValueError("Baseline extraction differs")
    orders = source_protocol["orders"]
    for order in orders.values():
        if len(order) != len(images) or set(order) != set(images):
            raise ValueError("Baseline order does not cover the input images once")
    args.output.mkdir(parents=True)
    (args.output / "traces").mkdir()
    protocol = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        partition="development",
        manifest_sha256=digest(args.manifest),
        extraction_report_sha256=digest(args.encoded / "report.json"),
        baseline_protocol_sha256=digest(args.baseline_protocol),
        baseline_summary_sha256=digest(args.baseline_summary),
        source_sha256=digest(__file__),
        models=report["configuration"]["models"],
        orders=orders,
        interval_seconds=source_protocol["interval_seconds"],
        window_seconds=source_protocol["window_seconds"],
        thresholds=THRESHOLDS.tolist(),
        threshold_grid="[-1,1], inclusive, step0.005",
        selection="Lowest mean extra+missed per scored query across three orders. "
        "Ties: lowest threshold. No matching-rate constraint.",
        event_rule="Search all usable prior distinct events inside(t-24h,t], "
        "include a historical event if cosine>=threshold, then add "
        "current event once. Store all usable events. Redelivery is "
        "idempotent. Same photo with a new event ID counts again.",
        failures="No output for unusable current input. Failed events remain in "
        "true history and all-input accuracy denominator.",
        comparison="Existing identity-based development grid minimum0.385/0.435, "
        "same extracted embeddings, exact event orders and objective. "
        f"Direct search has401thresholds; identity search has"
        f"{len(baseline['configurations'])}settings.",
        scope="Exploratory development comparison after the original held-out "
        "assessment. No final-test access, model inference or policy change. "
        "Three permutations are order sensitivity, not confidence intervals.",
        references=["https://scikit-learn.org/stable/modules/neighbors.html"],
    )
    save_json(args.output / "protocol.json", protocol)
    valid_ids = sorted(
        k for k, r in records.items() if r["status"] == "valid_embedding"
    )
    vectors = normalize_vectors(args.encoded / "embeddings.npz", valid_ids)
    matrix = np.stack([vectors[k] for k in valid_ids])
    scores = Scores(valid_ids, np.clip(matrix @ matrix.T, -1, 1))
    all_rows = []
    for seed, order in orders.items():
        rows = sweep_order(
            images,
            scores,
            order,
            THRESHOLDS,
            protocol["interval_seconds"],
            protocol["window_seconds"],
        )
        all_rows.extend([dict(seed=int(seed), **r) for r in rows])
        print(f"Swept401thresholds for order {seed}", flush=True)
    metrics = [
        "extra_per_event",
        "missed_per_event",
        "membership_cost",
        "count_mae",
        "count_exact_rate",
        "membership_exact_rate",
        "all_input_count_exact_rate",
        "returning_cost",
        "returning_count_exact_rate",
    ]
    combined = []
    for threshold in THRESHOLDS:
        group = [r for r in all_rows if r["threshold"] == threshold]
        row = dict(threshold=float(threshold))
        for key in metrics:
            values = [r[key] for r in group if r[key] is not None]
            for suffix, fn in [("mean", np.mean), ("min", min), ("max", max)]:
                row[key + "_" + suffix] = float(fn(values)) if values else None
        combined.append(row)
    selected = min(combined, key=lambda r: (r["membership_cost_mean"], r["threshold"]))
    for seed, order in orders.items():
        rows = replay(
            images,
            scores,
            order,
            int(seed),
            selected["threshold"],
            protocol["interval_seconds"],
            protocol["window_seconds"],
        )
        reference = next(
            r
            for r in all_rows
            if r["seed"] == int(seed) and r["threshold"] == selected["threshold"]
        )
        valid = [r for r in rows if r["scoreable"]]
        assert sum(r["extra_events"] for r in valid) == reference["extra"]
        assert sum(r["missed_events"] for r in valid) == reference["missed"]
        assert sum(r["count_exact"] for r in valid) == reference["exact"]
        write_trace(args.output / "traces" / f"direct_s{seed}.csv", rows)
    csv_write(args.output / "threshold_sweep.csv", combined, list(combined[0]))
    csv_write(args.output / "per_order_sweep.csv", all_rows, list(all_rows[0]))
    save_json(
        args.output / "summary.json",
        dict(
            protocol_sha256=digest(args.output / "protocol.json"),
            selected_direct=selected,
            identity_baseline=baseline["selected"],
            direct_per_order=[
                r for r in all_rows if r["threshold"] == selected["threshold"]
            ],
            identity_per_order=[
                r
                for r in baseline["per_replay"]
                if r["policy"] == baseline["selected"]["policy"]
            ],
            scope=protocol["scope"],
        ),
    )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lfw_matching/development/manifest.json"),
    )
    parser.add_argument(
        "--encoded", type=Path, default=Path("artifacts/casia_matching_encoded")
    )
    parser.add_argument(
        "--baseline-protocol",
        type=Path,
        default=Path("artifacts/casia_dynamic_extended_development/protocol.json"),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("artifacts/casia_dynamic_extended_development/summary.json"),
    )
    run(parser.parse_args())
