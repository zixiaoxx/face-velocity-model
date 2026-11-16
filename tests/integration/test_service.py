"""Real PostgreSQL tests for causal updates, retries, persistence and API errors."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from face_velocity.app import create_app
from face_velocity.storage import Storage


class Encoder:
    configuration = {"test_encoder": True}

    def extract(self, image):
        if image == b"weak":
            return {"status": "valid_embedding"}, [0.4, (1 - 0.4**2) ** 0.5]
        if image == b"bad":
            return {"status": "target_not_detected"}, None
        return {"status": "valid_embedding"}, [1.0, 0.0]


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "Requires PostgreSQL")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.storage = Storage(os.environ["TEST_DATABASE_URL"])
        self.client = TestClient(create_app(self.storage, Encoder()))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.session = self.client.post("/sessions", json={"source": "upload"}).json()[
            "session_id"
        ]
        self.addCleanup(self.cleanup)

    def cleanup(self):
        with self.storage.connect() as connection:
            connection.execute(
                "DELETE FROM demo_events WHERE session_id=%s", (self.session,)
            )
            connection.execute(
                "DELETE FROM demo_sessions WHERE session_id=%s", (self.session,)
            )

    def submit(self, event, seconds, content=b"photo"):
        instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        return self.client.post(
            f"/sessions/{self.session}/events",
            data={"event_id": event, "event_time": instant.isoformat()},
            files={"image": ("test.jpg", content, "image/jpeg")},
        )

    def test_retry_is_noop_new_event_counts_and_restart_persists(self):
        first = self.submit("e1", 0).json()
        second = self.submit("e2", 1).json()
        self.assertEqual((first["count"], second["count"]), (1, 2))
        retry = self.submit("e1", 0).json()
        self.assertTrue(retry["is_retry"])
        self.assertEqual(retry["count"], 1)
        with TestClient(
            create_app(Storage(os.environ["TEST_DATABASE_URL"]), Encoder())
        ) as fresh:
            self.assertEqual(
                len(fresh.get(f"/sessions/{self.session}").json()["events"]), 2
            )
        self.assertEqual(self.submit("e1", 0, b"changed").status_code, 409)

    def test_window_boundary_and_late_new_event(self):
        self.submit("e1", 0)
        self.submit("e2", 1)
        result = self.submit("e3", 86400).json()
        self.assertEqual(result["counted_events"], ["e2", "e3"])
        self.assertEqual(result["winning_reference"], "e1")
        self.assertEqual(self.submit("late", 10).status_code, 409)

    def test_accepted_event_counts_without_becoming_reference(self):
        self.submit("first", 0)
        result = self.submit("weak", 1, b"weak").json()
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["count"], 2)
        self.assertFalse(result["reference_added"])
        later = self.submit("later", 2).json()
        self.assertEqual(later["reference_count_before"], 1)
        self.assertEqual(later["count"], 3)

    def test_failure_cannot_enroll(self):
        self.assertIsNone(self.submit("bad", 0, b"bad").json()["count"])
        result = self.submit("good", 1).json()
        self.assertEqual(result["assigned_identity"], 1)
        self.assertEqual(result["count"], 1)

    def test_concurrent_retry_counts_once(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(lambda _: self.submit("e1", 0).json(), range(4))
            )
        self.assertEqual(sum(not r["is_retry"] for r in results), 1)
        self.assertEqual(len(self.storage.events(self.session)), 1)

    def test_naive_timestamp_rejected(self):
        response = self.client.post(
            f"/sessions/{self.session}/events",
            data={"event_id": "x", "event_time": "2026-01-01"},
            files={"image": ("x.jpg", b"x")},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
