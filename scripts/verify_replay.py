"""Check the running demo against saved causal experiment traces, event by event."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import httpx

from face_velocity.demo import saved_trace


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh verification output directory")
    args.output.mkdir(parents=True)
    reports = []
    with httpx.Client(base_url=args.url, timeout=120) as client:

        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        for seed in args.seeds:
            trace = saved_trace(args.source, seed)
            session = request(
                "POST", "/sessions", json={"source": args.source, "seed": seed}
            )
            session_id = session["session_id"]
            image_events = {}
            started = perf_counter()
            count = len(trace) if args.limit is None else min(args.limit, len(trace))
            for index, expected in enumerate(trace[:count]):
                actual = request("POST", f"/sessions/{session_id}/next")["event"]
                expected_identity = (
                    int(expected["assigned_group"]) + 1
                    if expected["assigned_group"]
                    else None
                )
                expected_count = (
                    int(expected["predicted_count"])
                    if expected["predicted_count"]
                    else None
                )
                checks = {
                    "event_id": expected["event_id"],
                    "image_id": expected["image_id"],
                    "assigned_identity": expected_identity,
                    "reference_added": expected["reference_added"] == "True",
                    "decision": expected["decision"],
                    "count": expected_count,
                    "counted_events": (
                        expected["predicted_event_ids"].split("|")
                        if expected["predicted_event_ids"]
                        else []
                    ),
                    "winning_reference": image_events.get(
                        expected["winning_reference"]
                    ),
                }
                for key, value in checks.items():
                    if actual[key] != value:
                        raise AssertionError((seed, index, key, actual[key], value))
                if expected["top_score"]:
                    assert (
                        abs(actual["top_score"] - float(expected["top_score"])) < 1e-10
                    )
                image_events[expected["image_id"]] = actual["event_id"]
                if (index + 1) % 250 == 0:
                    print(f"seed {seed}: {index + 1}/{count} events agree", flush=True)
            snapshot = request("GET", f"/sessions/{session_id}")
            annotations = snapshot["evaluation"]["events"]
            for expected in trace[:count]:
                if expected["scoreable"] != "True":
                    continue
                actual = annotations[expected["event_id"]]
                assert len(actual["extra"]) == int(expected["extra_events"])
                assert len(actual["missed"]) == int(expected["missed_events"])
            metrics = {k: v for k, v in snapshot["evaluation"].items() if k != "events"}
            reports.append(
                dict(
                    seed=seed,
                    session_id=session_id,
                    events=count,
                    seconds=perf_counter() - started,
                    metrics=metrics,
                )
            )
            (args.output / "verification.json").write_text(
                json.dumps(
                    dict(
                        source=args.source,
                        purpose="API/storage parity; not new model selection",
                        policy=session["configuration"]["policy"],
                        service_configuration=session["configuration"],
                        replays=reports,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(reports[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--source", choices=["development", "heldout"], default="heldout"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
