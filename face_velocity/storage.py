"""Transactional dynamic gallery: events, reference vectors and saved responses."""

import hashlib
from datetime import timedelta
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from face_velocity.pipeline import FROZEN_POLICY, associate


class SubmissionConflict(ValueError):
    """A submission conflicts with a saved event or session configuration."""


class Storage:
    def __init__(self, database_url):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS demo_sessions (
                    session_id TEXT PRIMARY KEY,
                    configuration JSONB NOT NULL,
                    source TEXT,
                    seed INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS demo_events (
                    session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
                    event_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    image_bytes BYTEA NOT NULL,
                    image_id TEXT,
                    embedding JSONB,
                    assigned_identity INTEGER,
                    is_reference BOOLEAN NOT NULL,
                    response JSONB NOT NULL,
                    PRIMARY KEY (session_id, event_id),
                    UNIQUE (session_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS demo_window
                    ON demo_events(session_id, assigned_identity, event_time);
            """)

    def create_session(self, configuration, source=None, seed=42):
        session_id = uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO demo_sessions(session_id,configuration,source,seed) "
                "VALUES (%s,%s,%s,%s)",
                (session_id, Jsonb(configuration), source, seed),
            )
        return self.session(session_id)

    def session(self, session_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM demo_sessions WHERE session_id=%s", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Session not found")
            row["processed"] = connection.execute(
                "SELECT COUNT(*) AS n FROM demo_events WHERE session_id=%s",
                (session_id,),
            ).fetchone()["n"]
            return row

    def submit(
        self,
        configuration,
        session_id,
        event_id,
        event_time,
        image_bytes,
        extraction,
        image_id=None,
    ):
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        with self.connect() as connection:
            session = connection.execute(
                "SELECT * FROM demo_sessions WHERE session_id=%s FOR UPDATE",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError("Session not found")
            if session["configuration"] != configuration:
                raise SubmissionConflict(
                    "Code or settings changed. Start a new session."
                )
            previous = connection.execute(
                "SELECT * FROM demo_events WHERE session_id=%s AND event_id=%s",
                (session_id, event_id),
            ).fetchone()
            if previous:
                if (
                    previous["image_sha256"] != image_hash
                    or previous["event_time"] != event_time
                    or previous["image_id"] != image_id
                ):
                    raise SubmissionConflict(
                        "Event ID already has different image/time"
                    )
                return dict(previous["response"], is_retry=True)
            state = connection.execute(
                "SELECT MAX(event_time) AS latest, COUNT(*) AS n, "
                "COALESCE(MAX(assigned_identity),0)+1 AS next_identity "
                "FROM demo_events WHERE session_id=%s",
                (session_id,),
            ).fetchone()
            if state["latest"] is not None and event_time < state["latest"]:
                raise SubmissionConflict("New events must be chronological")
            references = connection.execute(
                "SELECT event_id,sequence,assigned_identity AS identity,embedding "
                "FROM demo_events WHERE session_id=%s AND is_reference "
                "ORDER BY assigned_identity,sequence",
                (session_id,),
            ).fetchall()
            # Run after deduplication, while the session lock protects gallery state.
            metadata, vector = extraction()
            decision = associate(vector, references, state["next_identity"])
            response = dict(
                **decision,
                session_id=session_id,
                event_id=event_id,
                image_id=image_id,
                event_time=event_time.isoformat(),
                sequence=state["n"],
                extraction=metadata,
                is_retry=False,
                reference_count_before=len(references),
                window_start=(
                    event_time - timedelta(seconds=FROZEN_POLICY.window_seconds)
                ).isoformat(),
                window_end=event_time.isoformat(),
            )
            connection.execute(
                "INSERT INTO demo_events VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    session_id,
                    event_id,
                    state["n"],
                    event_time,
                    image_hash,
                    image_bytes,
                    image_id,
                    Jsonb(vector) if vector is not None else None,
                    decision["assigned_identity"],
                    decision["reference_added"],
                    Jsonb(response),
                ),
            )
            members = []
            if decision["assigned_identity"] is not None:
                members = [
                    r["event_id"]
                    for r in connection.execute(
                        "SELECT event_id FROM demo_events WHERE session_id=%s "
                        "AND assigned_identity=%s AND event_time>%s AND event_time<=%s "
                        "ORDER BY sequence",
                        (
                            session_id,
                            decision["assigned_identity"],
                            event_time
                            - timedelta(seconds=FROZEN_POLICY.window_seconds),
                            event_time,
                        ),
                    )
                ]
            response.update(
                count=len(members) if decision["assigned_identity"] else None,
                counted_events=members,
            )
            connection.execute(
                "UPDATE demo_events SET response=%s "
                "WHERE session_id=%s AND event_id=%s",
                (Jsonb(response), session_id, event_id),
            )
            return response

    def events(self, session_id):
        self.session(session_id)
        with self.connect() as connection:
            return [
                r["response"]
                for r in connection.execute(
                    "SELECT response FROM demo_events "
                    "WHERE session_id=%s ORDER BY sequence",
                    (session_id,),
                )
            ]

    def image(self, session_id, event_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT image_bytes FROM demo_events "
                "WHERE session_id=%s AND event_id=%s",
                (session_id, event_id),
            ).fetchone()
            if row is None:
                raise KeyError("Image not found")
            return bytes(row["image_bytes"])
