"""Local frozen-policy demo. Run uvicorn face_velocity.app:create_app --factory."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from face_velocity import demo
from face_velocity.pipeline import CasiaEncoder, service_configuration
from face_velocity.storage import Storage, SubmissionConflict


class NewSession(BaseModel):
    source: Literal["development", "heldout", "upload"] = "heldout"
    seed: Literal[42, 43, 44] = 42


def utc_time(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def create_app(storage=None, encoder=None):
    @asynccontextmanager
    async def lifespan(application):
        database = storage or Storage(os.environ["DATABASE_URL"])
        database.initialize()
        face_encoder = encoder or CasiaEncoder(
            os.environ.get("DETECTOR_PATH", "models/buffalo_l/det_10g.onnx"),
            os.environ.get("RECOGNIZER_PATH", "models/casia_r50/cisia_r50.onnx"),
        )
        application.state.storage = database
        application.state.encoder = face_encoder
        application.state.configuration = service_configuration(face_encoder)
        yield

    application = FastAPI(
        title="Face Velocity",
        lifespan=lifespan,
        description="Frozen CASIA dynamic gallery and 24-hour event counts",
    )
    application.mount(
        "/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static"
    )

    @application.exception_handler(SubmissionConflict)
    async def conflict(_, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"detail": str(error)})

    @application.exception_handler(KeyError)
    async def missing(_, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(error.args[0])})

    @application.exception_handler(ValueError)
    async def invalid(_, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": str(error)})

    @application.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "web/index.html")

    @application.get("/health")
    def health():
        with application.state.storage.connect() as connection:
            connection.execute("SELECT 1")
        return dict(status="ready", configuration=application.state.configuration)

    @application.get("/evidence/{name}")
    def evidence(name: Literal["development", "heldout"]):
        try:
            return demo.evidence(name)
        except FileNotFoundError as error:
            raise HTTPException(
                404, "Experiment artifacts not installed; see README"
            ) from error

    @application.post("/sessions")
    def start(body: NewSession):
        if body.source != "upload":
            demo.source(body.source)
        return application.state.storage.create_session(
            application.state.configuration, body.source, body.seed
        )

    @application.post("/sessions/{session_id}/next")
    def next_event(session_id: str):
        session = application.state.storage.session(session_id)
        if session["source"] == "upload":
            raise ValueError("Upload sessions accept images through /events")
        row = demo.replay_input(
            session["source"], session["seed"], session["processed"]
        )
        if row is None:
            return dict(done=True)
        result = application.state.storage.submit(
            application.state.configuration,
            session_id,
            row["event_id"],
            row["event_time"],
            row["image_bytes"],
            lambda: (row["metadata"], row["vector"]),
            row["image_id"],
        )
        return dict(done=False, event=result)

    @application.get("/sessions/{session_id}")
    def snapshot(session_id: str):
        session = application.state.storage.session(session_id)
        events = application.state.storage.events(session_id)
        evaluation = (
            demo.score_history(session["source"], events)
            if session["source"] != "upload"
            else None
        )
        return dict(session=session, events=events, evaluation=evaluation)

    @application.get("/sessions/{session_id}/images/{event_id}")
    def image(session_id: str, event_id: str):
        content = application.state.storage.image(session_id, event_id)
        media_type = "image/png" if content.startswith(b"\x89PNG") else "image/jpeg"
        return Response(
            content,
            media_type=media_type,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @application.post("/sessions/{session_id}/events")
    def submit_event(
        session_id: str,
        image: Annotated[UploadFile, File()],
        event_id: Annotated[
            str, Form(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
        ],
        event_time: Annotated[datetime, Form()],
    ):
        session = application.state.storage.session(session_id)
        if session["source"] != "upload":
            raise ValueError(
                "Use a separate upload session so replay order stays fixed"
            )
        content = image.file.read(10 * 1024 * 1024 + 1)
        if not content or len(content) > 10 * 1024 * 1024:
            raise ValueError("Provide a nonempty image of at most 10 MiB")
        return application.state.storage.submit(
            application.state.configuration,
            session_id,
            event_id,
            utc_time(event_time),
            content,
            lambda: application.state.encoder.extract(content),
        )

    return application
