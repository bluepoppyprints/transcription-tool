from __future__ import annotations

import argparse
import heapq
import json
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .backends import BACKEND_SPECS, backend_catalog
from .exporters import collapse_segments, export_docx, export_json, export_srt, export_txt
from .live_text_cleaner import clean_transcript_file
from .pipeline import TranscriptionCancelled, transcribe_file
from .types import PipelineOptions, TranscriptResult, TranscriptSegment
from .utils import ensure_dir, format_timestamp


FORMAT_EXPORTERS = {
    "txt": (export_txt, ".txt"),
    "srt": (export_srt, ".srt"),
    "json": (export_json, ".json"),
    "docx": (export_docx, ".docx"),
}

DATA_ROOT = Path("output") / "web"
UPLOAD_ROOT = DATA_ROOT / "uploads"
RESULT_ROOT = DATA_ROOT / "results"
LIVE_ROOT = DATA_ROOT / "live"
DB_PATH = DATA_ROOT / "jobs.db"
INDEX_HTML_PATH = Path(__file__).with_name("web_index.html")
MAX_LOG_LINES = 300
RETENTION_MAX_JOBS = 200
RETENTION_MAX_DAYS = 30
LIVE_RECONCILE_SECONDS = 2.5
LIVE_APPEND_EPSILON_SECONDS = 0.15


@dataclass(slots=True)
class Artifact:
    artifact_id: str
    source: str
    format: str
    name: str
    path: Path


@dataclass(slots=True)
class JobPayload:
    files: list[Path]
    options: PipelineOptions
    formats: list[str]


@dataclass(slots=True)
class JobState:
    job_id: str
    created_at: str
    status: str
    total_files: int
    processed_files: int = 0
    current_file: str | None = None
    current_file_progress: float = 0.0
    current_file_started_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    processing_seconds_total: float = 0.0
    cancel_requested: bool = False
    logs: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class LiveSessionState:
    session_id: str
    created_at: str
    status: str
    options: PipelineOptions
    formats: list[str]
    source_name: str = "live_input.webm"
    stream_ext: str = ".webm"
    language: str = "unknown"
    audio_seconds: float = 0.0
    chunk_index: int = 0
    processing: bool = False
    logs: list[str] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error: str | None = None


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "audio"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_formats(value: str) -> list[str]:
    formats: list[str] = []
    for token in (x.strip().lower() for x in value.split(",")):
        if not token:
            continue
        if token not in FORMAT_EXPORTERS:
            raise ValueError(f"Unsupported output format: {token}")
        if token not in formats:
            formats.append(token)
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def _parse_optional_int(value: str, label: str) -> int | None:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label}") from exc
    if parsed < 1:
        raise HTTPException(status_code=400, detail=f"{label} must be >= 1")
    return parsed


def _build_pipeline_options(
    *,
    backend: str,
    model: str,
    language: str,
    diarization: str,
    device: str,
    compute_type: str,
    beam_size: int,
    no_vad: str,
    min_speakers: str,
    max_speakers: str,
    speaker_names: str,
    hotwords: str,
    num_threads: int,
    hf_token: str,
    force_diarization_off: bool = False,
) -> PipelineOptions:
    if backend not in BACKEND_SPECS:
        raise HTTPException(status_code=400, detail="Invalid ASR backend")
    if diarization not in {"auto", "on", "off"}:
        raise HTTPException(status_code=400, detail="Invalid diarization mode")
    if device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="Invalid device")
    if compute_type not in {"auto", "int8", "int8_float16", "float16", "float32"}:
        raise HTTPException(status_code=400, detail="Invalid compute type")
    if beam_size < 1 or beam_size > 10:
        raise HTTPException(status_code=400, detail="beam_size must be between 1 and 10")
    if num_threads < 1 or num_threads > 32:
        raise HTTPException(status_code=400, detail="num_threads must be between 1 and 32")

    min_speakers_value = _parse_optional_int(min_speakers, "min_speakers")
    max_speakers_value = _parse_optional_int(max_speakers, "max_speakers")
    if (
        min_speakers_value is not None
        and max_speakers_value is not None
        and min_speakers_value > max_speakers_value
    ):
        raise HTTPException(status_code=400, detail="min_speakers must be <= max_speakers")

    diarization_mode = "off" if force_diarization_off else diarization
    return PipelineOptions(
        backend=backend,
        model=model.strip() or "small",
        language=language.strip() or None,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=not _as_bool(no_vad),
        diarization_mode=diarization_mode,
        hf_token=hf_token.strip() or None,
        min_speakers=min_speakers_value,
        max_speakers=max_speakers_value,
        speaker_names=[name.strip() for name in speaker_names.split(",") if name.strip()],
        hotwords=hotwords.strip(),
        num_threads=num_threads,
    )


def _serialize_options(options: PipelineOptions) -> dict:
    data = asdict(options)
    data["hf_token"] = None
    return data


def _deserialize_options(data: dict) -> PipelineOptions:
    allowed = {item.name for item in fields(PipelineOptions)}
    payload = {k: v for k, v in data.items() if k in allowed}
    return PipelineOptions(**payload)


def _db_connect() -> sqlite3.Connection:
    ensure_dir(DATA_ROOT)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _db_init() -> None:
    conn = _db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                total_files INTEGER NOT NULL,
                processed_files INTEGER NOT NULL,
                current_file TEXT,
                current_file_progress REAL NOT NULL DEFAULT 0,
                current_file_started_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                processing_seconds_total REAL NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                logs_json TEXT NOT NULL,
                files_json TEXT NOT NULL,
                options_json TEXT NOT NULL,
                formats_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                source TEXT NOT NULL,
                format TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON artifacts(job_id)")

        existing_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "current_file_started_at" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN current_file_started_at TEXT")
        if "started_at" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN started_at TEXT")
        if "finished_at" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN finished_at TEXT")
        if "processing_seconds_total" not in existing_columns:
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN processing_seconds_total REAL NOT NULL DEFAULT 0"
            )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _initialize_runtime()
    yield


app = FastAPI(title="Otter Local Web", lifespan=_lifespan)
_jobs: dict[str, JobState] = {}
_job_payloads: dict[str, JobPayload] = {}
_live_sessions: dict[str, LiveSessionState] = {}
_lock = threading.Lock()
_job_queue: queue.Queue[str] = queue.Queue()
_workers_started = False
_runtime_ready = False
_worker_target_count = 2


def _persist_artifact_locked(job_id: str, artifact: Artifact) -> None:
    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts (artifact_id, job_id, source, format, name, path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (artifact.artifact_id, job_id, artifact.source, artifact.format, artifact.name, str(artifact.path)),
        )
        conn.commit()
    finally:
        conn.close()


def _persist_job_locked(job: JobState) -> None:
    payload = _job_payloads.get(job.job_id)
    if payload is None:
        files_json = "[]"
        options_json = json.dumps(_serialize_options(PipelineOptions()))
        formats_json = "[]"
    else:
        files_json = json.dumps([str(path) for path in payload.files], ensure_ascii=False)
        options_json = json.dumps(_serialize_options(payload.options), ensure_ascii=False)
        formats_json = json.dumps(payload.formats, ensure_ascii=False)

    logs_json = json.dumps(job.logs[-MAX_LOG_LINES:], ensure_ascii=False)
    updated_at = datetime.now().isoformat(timespec="seconds")

    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, created_at, status, total_files, processed_files,
                current_file, current_file_progress, current_file_started_at,
                started_at, finished_at, processing_seconds_total,
                cancel_requested, error,
                logs_json, files_json, options_json, formats_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                created_at = excluded.created_at,
                status = excluded.status,
                total_files = excluded.total_files,
                processed_files = excluded.processed_files,
                current_file = excluded.current_file,
                current_file_progress = excluded.current_file_progress,
                current_file_started_at = excluded.current_file_started_at,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                processing_seconds_total = excluded.processing_seconds_total,
                cancel_requested = excluded.cancel_requested,
                error = excluded.error,
                logs_json = excluded.logs_json,
                files_json = excluded.files_json,
                options_json = excluded.options_json,
                formats_json = excluded.formats_json,
                updated_at = excluded.updated_at
            """,
            (
                job.job_id,
                job.created_at,
                job.status,
                job.total_files,
                job.processed_files,
                job.current_file,
                float(job.current_file_progress),
                job.current_file_started_at,
                job.started_at,
                job.finished_at,
                float(job.processing_seconds_total),
                1 if job.cancel_requested else 0,
                job.error,
                logs_json,
                files_json,
                options_json,
                formats_json,
                updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _load_state_locked() -> None:
    _jobs.clear()
    _job_payloads.clear()

    conn = _db_connect()
    try:
        job_rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        artifact_rows = conn.execute("SELECT * FROM artifacts").fetchall()
    finally:
        conn.close()

    artifacts_by_job: dict[str, list[Artifact]] = {}
    for row in artifact_rows:
        artifact = Artifact(
            artifact_id=str(row["artifact_id"]),
            source=str(row["source"]),
            format=str(row["format"]),
            name=str(row["name"]),
            path=Path(str(row["path"])),
        )
        artifacts_by_job.setdefault(str(row["job_id"]), []).append(artifact)

    for row in job_rows:
        job_id = str(row["job_id"])
        original_status = str(row["status"])
        status = original_status
        error = row["error"]

        if status in {"queued", "running"}:
            status = "failed"
            error = "Server restarted before job finished."

        logs = json.loads(str(row["logs_json"])) if row["logs_json"] else []
        job = JobState(
            job_id=job_id,
            created_at=str(row["created_at"]),
            status=status,
            total_files=int(row["total_files"]),
            processed_files=int(row["processed_files"]),
            current_file=str(row["current_file"]) if row["current_file"] else None,
            current_file_progress=float(row["current_file_progress"] or 0.0),
            current_file_started_at=(
                str(row["current_file_started_at"]) if row["current_file_started_at"] else None
            ),
            started_at=str(row["started_at"]) if row["started_at"] else None,
            finished_at=str(row["finished_at"]) if row["finished_at"] else None,
            processing_seconds_total=float(row["processing_seconds_total"] or 0.0),
            cancel_requested=bool(int(row["cancel_requested"] or 0)),
            logs=[str(item) for item in logs][-MAX_LOG_LINES:],
            artifacts=artifacts_by_job.get(job_id, []),
            error=str(error) if error else None,
        )
        _jobs[job_id] = job

        try:
            files_data = json.loads(str(row["files_json"])) if row["files_json"] else []
            options_data = json.loads(str(row["options_json"])) if row["options_json"] else {}
            formats_data = json.loads(str(row["formats_json"])) if row["formats_json"] else []
        except json.JSONDecodeError:
            files_data = []
            options_data = {}
            formats_data = []

        files_list = [Path(str(path)) for path in files_data]
        options = _deserialize_options(options_data if isinstance(options_data, dict) else {})
        formats = [str(item) for item in formats_data if str(item) in FORMAT_EXPORTERS]
        if not formats:
            formats = ["txt"]

        _job_payloads[job_id] = JobPayload(files=files_list, options=options, formats=formats)

        if status != original_status:
            if not job.finished_at:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
            _persist_job_locked(job)


def _start_workers_locked() -> None:
    global _workers_started
    if _workers_started:
        return

    worker_count = max(1, int(_worker_target_count))
    for index in range(worker_count):
        worker = threading.Thread(target=_worker_loop, name=f"otter-worker-{index + 1}", daemon=True)
        worker.start()
    _workers_started = True


def _initialize_runtime() -> None:
    global _runtime_ready
    with _lock:
        if _runtime_ready:
            return

        ensure_dir(UPLOAD_ROOT)
        ensure_dir(RESULT_ROOT)
        ensure_dir(LIVE_ROOT)
        _db_init()
        _load_state_locked()
        _cleanup_retention_locked()
        _start_workers_locked()
        _runtime_ready = True


def _append_log(job: JobState, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    job.logs.append(f"[{timestamp}] {message}")
    if len(job.logs) > MAX_LOG_LINES:
        job.logs[:] = job.logs[-MAX_LOG_LINES:]


def _append_live_log(session: LiveSessionState, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    session.logs.append(f"[{timestamp}] {message}")
    if len(session.logs) > MAX_LOG_LINES:
        session.logs[:] = session.logs[-MAX_LOG_LINES:]


def _live_transcript_text(segments: list[TranscriptSegment]) -> str:
    utterances = collapse_segments(segments)
    lines = [
        f"[{format_timestamp(seg.start)}] {seg.speaker}: {seg.text}"
        for seg in utterances
        if seg.text.strip()
    ]
    return "\n".join(lines)


def _live_session_to_dict(session: LiveSessionState) -> dict:
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "status": session.status,
        "language": session.language,
        "audio_seconds": round(float(session.audio_seconds), 2),
        "chunk_index": session.chunk_index,
        "segment_count": len(session.segments),
        "processing": session.processing,
        "error": session.error,
        "logs": list(session.logs),
        "transcript_text": _live_transcript_text(session.segments),
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "source": item.source,
                "format": item.format,
                "name": item.name,
                "download_url": f"/api/live/{session.session_id}/artifacts/{item.artifact_id}",
            }
            for item in session.artifacts
        ],
    }


def _to_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _is_terminal_status(status: str) -> bool:
    return status in {"completed", "failed", "canceled"}


def _estimate_average_file_seconds_locked() -> float | None:
    total_seconds = 0.0
    total_files = 0
    for job in _jobs.values():
        if job.processed_files > 0 and job.processing_seconds_total > 0:
            total_seconds += float(job.processing_seconds_total)
            total_files += int(job.processed_files)
    if total_files == 0:
        return None
    return total_seconds / total_files


def _estimate_running_remaining_seconds_locked(
    job: JobState,
    average_file_seconds: float | None,
    now_ts: float,
) -> float | None:
    if job.status != "running":
        return 0.0

    remaining_after_current = max(
        0,
        job.total_files - job.processed_files - (1 if job.current_file else 0),
    )
    per_file = average_file_seconds
    current_remaining = 0.0

    if job.current_file:
        progress = max(0.0, min(1.0, float(job.current_file_progress)))
        if 0 < progress < 1 and job.current_file_started_at:
            started = _to_timestamp(job.current_file_started_at)
            if started is not None:
                elapsed = max(0.0, now_ts - started)
                projected_total = elapsed / max(progress, 1e-6)
                current_remaining = max(0.0, projected_total - elapsed)
                if per_file is None:
                    per_file = projected_total
        elif progress <= 0 and per_file is not None:
            current_remaining = per_file
        elif progress >= 1:
            current_remaining = 0.0
        elif per_file is None:
            return None

    if per_file is None and remaining_after_current > 0:
        return None

    return current_remaining + (remaining_after_current * float(per_file or 0.0))


def _estimate_job_runtime_seconds_locked(
    job: JobState,
    average_file_seconds: float | None,
    now_ts: float,
) -> float | None:
    if job.status == "running":
        return _estimate_running_remaining_seconds_locked(job, average_file_seconds, now_ts)

    remaining_files = max(0, job.total_files - job.processed_files)
    if remaining_files == 0:
        return 0.0
    if average_file_seconds is None:
        return None
    return remaining_files * average_file_seconds


def _snapshot_queue_ids_locked() -> list[str]:
    with _job_queue.mutex:
        return list(_job_queue.queue)


def _compute_queue_and_eta_locked() -> tuple[dict[str, int], dict[str, float | None]]:
    queue_ids = _snapshot_queue_ids_locked()
    queue_positions = {job_id: index + 1 for index, job_id in enumerate(queue_ids)}

    average_file_seconds = _estimate_average_file_seconds_locked()
    now_ts = time.time()
    worker_count = max(1, int(_worker_target_count))

    eta_seconds: dict[str, float | None] = {}

    worker_heap: list[float] = []
    running_jobs = [job for job in _jobs.values() if job.status == "running"]
    for running in running_jobs[:worker_count]:
        remaining = _estimate_running_remaining_seconds_locked(running, average_file_seconds, now_ts)
        eta_seconds[running.job_id] = remaining
        worker_heap.append(float(remaining or 0.0))

    while len(worker_heap) < worker_count:
        worker_heap.append(0.0)
    heapq.heapify(worker_heap)

    for queued_job_id in queue_ids:
        queued_job = _jobs.get(queued_job_id)
        if queued_job is None:
            continue

        runtime = _estimate_job_runtime_seconds_locked(queued_job, average_file_seconds, now_ts)
        if runtime is None:
            eta_seconds[queued_job_id] = None
            continue

        start_at = heapq.heappop(worker_heap)
        finish_at = start_at + float(runtime)
        heapq.heappush(worker_heap, finish_at)
        eta_seconds[queued_job_id] = finish_at

    return queue_positions, eta_seconds


def _delete_job_from_db(job_id: str) -> None:
    conn = _db_connect()
    try:
        conn.execute("DELETE FROM artifacts WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
    finally:
        conn.close()


def _remove_from_queue_locked(job_id: str) -> None:
    with _job_queue.mutex:
        original = list(_job_queue.queue)
        filtered = [item for item in original if item != job_id]
        removed = len(original) - len(filtered)
        if removed <= 0:
            return

        _job_queue.queue.clear()
        _job_queue.queue.extend(filtered)
        _job_queue.unfinished_tasks = max(0, _job_queue.unfinished_tasks - removed)
        if _job_queue.unfinished_tasks == 0:
            _job_queue.all_tasks_done.notify_all()


def _delete_job_locked(job_id: str, remove_files: bool = True) -> bool:
    job = _jobs.get(job_id)
    if job is None:
        return False
    if not _is_terminal_status(job.status):
        return False

    _remove_from_queue_locked(job_id)
    _jobs.pop(job_id, None)
    _job_payloads.pop(job_id, None)
    _delete_job_from_db(job_id)

    if remove_files:
        shutil.rmtree(RESULT_ROOT / job_id, ignore_errors=True)
        shutil.rmtree(UPLOAD_ROOT / job_id, ignore_errors=True)

    return True


def _cleanup_retention_locked() -> int:
    now_ts = time.time()
    cutoff_ts = now_ts - (RETENTION_MAX_DAYS * 86400)

    terminal_jobs = [
        job for job in _jobs.values() if _is_terminal_status(job.status)
    ]
    terminal_jobs.sort(
        key=lambda item: _to_timestamp(item.finished_at or item.created_at) or 0.0,
        reverse=True,
    )
    keep_ids = {job.job_id for job in terminal_jobs[:RETENTION_MAX_JOBS]}

    to_delete: list[str] = []
    for job in terminal_jobs:
        ref_ts = _to_timestamp(job.finished_at or job.created_at) or now_ts
        if job.job_id not in keep_ids or ref_ts < cutoff_ts:
            to_delete.append(job.job_id)

    for job_id in to_delete:
        _delete_job_locked(job_id, remove_files=True)

    return len(to_delete)


def _mark_canceled_locked(job: JobState, note: str) -> None:
    job.status = "canceled"
    job.cancel_requested = False
    job.error = None
    job.current_file = None
    job.current_file_progress = 0.0
    job.current_file_started_at = None
    job.finished_at = datetime.now().isoformat(timespec="seconds")
    _append_log(job, note)
    _persist_job_locked(job)


def _job_to_dict(
    job: JobState,
    queue_position: int | None = None,
    eta_seconds: float | None = None,
) -> dict:
    return {
        "job_id": job.job_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "status": job.status,
        "total_files": job.total_files,
        "processed_files": job.processed_files,
        "current_file": job.current_file,
        "current_file_progress": job.current_file_progress,
        "cancel_requested": job.cancel_requested,
        "error": job.error,
        "queue_position": queue_position,
        "eta_seconds": int(eta_seconds) if eta_seconds is not None else None,
        "can_cancel": job.status in {"queued", "running"} and not job.cancel_requested,
        "can_retry": job.status in {"completed", "failed", "canceled"},
        "can_delete": _is_terminal_status(job.status),
        "logs": list(job.logs),
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "source": item.source,
                "format": item.format,
                "name": item.name,
                "download_url": f"/api/jobs/{job.job_id}/artifacts/{item.artifact_id}",
            }
            for item in job.artifacts
        ],
    }


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            _process_job(job_id)
        finally:
            _job_queue.task_done()


def _process_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        payload = _job_payloads.get(job_id)
        if job is None or payload is None:
            return
        if job.status == "canceled":
            _persist_job_locked(job)
            return
        if job.cancel_requested:
            _mark_canceled_locked(job, "Canceled before processing")
            return

        job.status = "running"
        job.error = None
        job.started_at = job.started_at or datetime.now().isoformat(timespec="seconds")
        job.finished_at = None
        _persist_job_locked(job)

    files = list(payload.files)
    options = payload.options
    formats = list(payload.formats)

    for input_path in files:
        file_name = input_path.name

        with _lock:
            live = _jobs.get(job_id)
            if live is None:
                return
            if live.cancel_requested:
                _mark_canceled_locked(live, "Canceled before processing next file")
                return
            live.current_file = file_name
            live.current_file_progress = 0.0
            live.current_file_started_at = datetime.now().isoformat(timespec="seconds")
            _append_log(
                live,
                f"Transcribing {file_name} with {options.backend} / {options.model}...",
            )
            _persist_job_locked(live)

        if not input_path.exists():
            with _lock:
                live = _jobs.get(job_id)
                if live is None:
                    return
                live.status = "failed"
                live.error = f"Input file missing: {input_path}"
                live.finished_at = datetime.now().isoformat(timespec="seconds")
                _append_log(live, live.error)
                _persist_job_locked(live)
                _cleanup_retention_locked()
            return

        last_persist = 0.0

        def on_progress(progress: float, _stage: str) -> bool:
            nonlocal last_persist
            with _lock:
                live = _jobs.get(job_id)
                if live is None:
                    return False
                if live.cancel_requested:
                    return False

                live.current_file = file_name
                live.current_file_progress = max(0.0, min(1.0, float(progress)))
                now = time.monotonic()
                if now - last_persist >= 0.5 or live.current_file_progress >= 1.0:
                    _persist_job_locked(live)
                    last_persist = now
            return True

        try:
            result = transcribe_file(input_path, options, progress_callback=on_progress)
            result_dir = ensure_dir(RESULT_ROOT / job_id / _safe_name(input_path.stem))

            for index, fmt in enumerate(formats, start=1):
                with _lock:
                    live = _jobs.get(job_id)
                    if live is None:
                        return
                    if live.cancel_requested:
                        raise TranscriptionCancelled("Canceled during export")

                exporter, ext = FORMAT_EXPORTERS[fmt]
                out_file = result_dir / f"transcript{ext}"
                exporter(result, out_file)

                artifact = Artifact(
                    artifact_id=uuid.uuid4().hex[:12],
                    source=file_name,
                    format=fmt,
                    name=out_file.name,
                    path=out_file,
                )

                with _lock:
                    live = _jobs.get(job_id)
                    if live is None:
                        return
                    live.artifacts.append(artifact)
                    export_ratio = 0.96 + (index / max(1, len(formats))) * 0.03
                    live.current_file_progress = min(0.99, export_ratio)
                    _persist_artifact_locked(job_id, artifact)
                    _persist_job_locked(live)

            with _lock:
                live = _jobs.get(job_id)
                if live is None:
                    return
                live.current_file_progress = 1.0
                started_ts = _to_timestamp(live.current_file_started_at)
                if started_ts is not None:
                    live.processing_seconds_total += max(0.0, time.time() - started_ts)
                live.current_file_started_at = None
                live.processed_files += 1
                _append_log(live, f"Finished {file_name}")
                _persist_job_locked(live)
        except TranscriptionCancelled:
            with _lock:
                live = _jobs.get(job_id)
                if live is None:
                    return
                _mark_canceled_locked(live, "Canceled by user")
                _cleanup_retention_locked()
            return
        except Exception as exc:
            with _lock:
                live = _jobs.get(job_id)
                if live is None:
                    return
                live.status = "failed"
                live.error = str(exc)
                live.current_file_started_at = None
                live.finished_at = datetime.now().isoformat(timespec="seconds")
                _append_log(live, f"Failed: {exc}")
                _persist_job_locked(live)
                _cleanup_retention_locked()
            return

    with _lock:
        live = _jobs.get(job_id)
        if live is None:
            return
        live.status = "completed"
        live.cancel_requested = False
        live.current_file = None
        live.current_file_progress = 0.0
        live.current_file_started_at = None
        live.finished_at = datetime.now().isoformat(timespec="seconds")
        _append_log(live, "All files processed")
        _persist_job_locked(live)
        _cleanup_retention_locked()


def _load_index_html() -> str:
    if INDEX_HTML_PATH.exists():
        return INDEX_HTML_PATH.read_text(encoding="utf-8")
    return "<html><body><h1>Web UI template missing.</h1></body></html>"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    _initialize_runtime()
    return _load_index_html()


@app.get("/api/diagnostics")
def diagnostics() -> dict:
    _initialize_runtime()
    routes = sorted(
        {
            str(route.path)
            for route in app.routes
            if getattr(route, "path", None)
        }
    )
    live_routes = [path for path in routes if path.startswith("/api/live")]
    return {
        "module_file": str(Path(__file__).resolve()),
        "live_routes": live_routes,
        "has_live_start": "/api/live/start" in live_routes,
        "route_count": len(routes),
        "backends": backend_catalog(),
    }


@app.get("/api/backends")
def list_backends() -> dict:
    _initialize_runtime()
    return {"backends": backend_catalog()}


@app.post("/api/transcribe")
async def submit_job(
    files: list[UploadFile] = File(...),
    backend: str = Form("faster-whisper"),
    model: str = Form("small"),
    language: str = Form(""),
    diarization: str = Form("auto"),
    device: str = Form("auto"),
    compute_type: str = Form("auto"),
    beam_size: int = Form(5),
    no_vad: str = Form("false"),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
    speaker_names: str = Form(""),
    hotwords: str = Form(""),
    num_threads: int = Form(4),
    hf_token: str = Form(""),
    formats: str = Form("txt,srt,json,docx"),
) -> dict:
    _initialize_runtime()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        resolved_formats = _resolve_formats(formats)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    options = _build_pipeline_options(
        backend=backend,
        model=model,
        language=language,
        diarization=diarization,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        no_vad=no_vad,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        speaker_names=speaker_names,
        hotwords=hotwords,
        num_threads=num_threads,
        hf_token=hf_token,
    )

    job_id = uuid.uuid4().hex[:12]
    upload_dir = ensure_dir(UPLOAD_ROOT / job_id)
    local_files: list[Path] = []

    for index, upload in enumerate(files, start=1):
        if not upload.filename:
            continue
        safe = _safe_name(upload.filename)
        target = upload_dir / f"{index:02d}_{safe}"
        with target.open("wb") as out_f:
            shutil.copyfileobj(upload.file, out_f)
        local_files.append(target)

    if not local_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    payload = JobPayload(files=local_files, options=options, formats=resolved_formats)
    job = JobState(
        job_id=job_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        status="queued",
        total_files=len(local_files),
    )
    _append_log(job, f"Queued {len(local_files)} file(s)")

    with _lock:
        _jobs[job_id] = job
        _job_payloads[job_id] = payload
        _persist_job_locked(job)

    _job_queue.put(job_id)
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}", "queue_depth": _job_queue.qsize()}


@app.post("/api/live/start")
def start_live_session(
    backend: str = Form("faster-whisper"),
    model: str = Form("small"),
    language: str = Form(""),
    diarization: str = Form("off"),
    device: str = Form("auto"),
    compute_type: str = Form("auto"),
    beam_size: int = Form(5),
    no_vad: str = Form("false"),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
    speaker_names: str = Form(""),
    hotwords: str = Form(""),
    num_threads: int = Form(4),
    hf_token: str = Form(""),
    formats: str = Form("txt,srt,json,docx"),
) -> dict:
    _initialize_runtime()

    try:
        resolved_formats = _resolve_formats(formats)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    options = _build_pipeline_options(
        backend=backend,
        model=model,
        language=language,
        diarization=diarization,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        no_vad=no_vad,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        speaker_names=speaker_names,
        hotwords=hotwords,
        num_threads=num_threads,
        hf_token=hf_token,
        force_diarization_off=True,
    )

    session_id = uuid.uuid4().hex[:12]
    live_state = LiveSessionState(
        session_id=session_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        status="recording",
        options=options,
        formats=resolved_formats,
    )
    _append_live_log(live_state, "Live session started")
    if diarization != "off":
        _append_live_log(live_state, "Diarization is disabled during live mode for responsiveness")

    with _lock:
        _live_sessions[session_id] = live_state

    return _live_session_to_dict(live_state)


@app.get("/api/live/{session_id}")
def get_live_session(session_id: str) -> dict:
    _initialize_runtime()
    with _lock:
        session = _live_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        return _live_session_to_dict(session)


@app.post("/api/live/{session_id}/chunk")
async def ingest_live_chunk(
    session_id: str,
    chunk: UploadFile = File(...),
) -> dict:
    _initialize_runtime()

    with _lock:
        session = _live_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        if session.status != "recording":
            raise HTTPException(status_code=409, detail="Live session is not recording")
        if session.processing:
            raise HTTPException(status_code=409, detail="Previous chunk is still processing")

        session.processing = True
        session.chunk_index += 1
        chunk_index = session.chunk_index
        options = PipelineOptions(**asdict(session.options))
        active_stream_ext = session.stream_ext

    ext = Path(chunk.filename or "").suffix.lower()
    if ext not in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".flac"}:
        ext = active_stream_ext or ".webm"

    chunk_dir = ensure_dir(LIVE_ROOT / session_id / "chunks")
    chunk_path = chunk_dir / f"{chunk_index:05d}{ext}"
    with chunk_path.open("wb") as out_f:
        shutil.copyfileobj(chunk.file, out_f)

    stream_ext = ext if chunk_index == 1 else (active_stream_ext or ext)
    stream_path = ensure_dir(LIVE_ROOT / session_id) / f"stream{stream_ext}"
    with chunk_path.open("rb") as in_f, stream_path.open("ab") as stream_f:
        shutil.copyfileobj(in_f, stream_f)

    with _lock:
        live = _live_sessions.get(session_id)
        if live is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        live.stream_ext = stream_ext

    mode = "chunk"
    fallback_reason: str | None = None
    try:
        result = transcribe_file(chunk_path, options)
    except Exception as chunk_exc:
        fallback_reason = str(chunk_exc)
        mode = "stream"
        try:
            result = transcribe_file(stream_path, options)
        except Exception as stream_exc:
            detail = f"{stream_exc}"
            with _lock:
                live = _live_sessions.get(session_id)
                if live is not None:
                    live.processing = False
                    live.status = "failed"
                    live.error = detail
                    _append_live_log(
                        live,
                        f"Chunk {chunk_index} failed: {chunk_exc} | stream fallback failed: {stream_exc}",
                    )
            raise HTTPException(status_code=500, detail=detail) from stream_exc
    finally:
        await chunk.close()

    with _lock:
        live = _live_sessions.get(session_id)
        if live is None:
            raise HTTPException(status_code=404, detail="Live session not found")

        new_segments_payload: list[dict] = []
        if mode == "chunk":
            base_offset = float(live.audio_seconds)
            for seg in result.segments:
                text = seg.text.strip()
                if not text:
                    continue

                shifted = TranscriptSegment(
                    start=float(seg.start) + base_offset,
                    end=float(seg.end) + base_offset,
                    text=text,
                    speaker=seg.speaker,
                    confidence=seg.confidence,
                )
                live.segments.append(shifted)
                new_segments_payload.append(
                    {
                        "start": shifted.start,
                        "end": shifted.end,
                        "speaker": shifted.speaker,
                        "text": shifted.text,
                        "confidence": shifted.confidence,
                    }
                )

            chunk_seconds = float(result.duration or 0.0)
            if chunk_seconds <= 0.0 and result.segments:
                chunk_seconds = max(float(seg.end) for seg in result.segments)
            live.audio_seconds += max(0.0, chunk_seconds)
        else:
            previous_seconds = float(live.audio_seconds)
            reconcile_cutoff = max(0.0, previous_seconds - LIVE_RECONCILE_SECONDS)
            preserved = [seg for seg in live.segments if float(seg.end) <= reconcile_cutoff]
            rebuilt_tail: list[TranscriptSegment] = []

            for seg in result.segments:
                text = seg.text.strip()
                if not text:
                    continue
                seg_start = float(seg.start)
                seg_end = float(seg.end)
                if seg_end <= max(0.0, reconcile_cutoff - LIVE_APPEND_EPSILON_SECONDS):
                    continue

                rebuilt = TranscriptSegment(
                    start=seg_start,
                    end=seg_end,
                    text=text,
                    speaker=seg.speaker,
                    confidence=seg.confidence,
                )
                rebuilt_tail.append(rebuilt)

            live.segments = collapse_segments(preserved + rebuilt_tail, merge_gap_seconds=0.6)
            for rebuilt in live.segments:
                if float(rebuilt.end) <= max(0.0, previous_seconds - LIVE_APPEND_EPSILON_SECONDS):
                    continue
                new_segments_payload.append(
                    {
                        "start": rebuilt.start,
                        "end": rebuilt.end,
                        "speaker": rebuilt.speaker,
                        "text": rebuilt.text,
                        "confidence": rebuilt.confidence,
                    }
                )

            live.audio_seconds = max(
                previous_seconds,
                float(result.duration or 0.0),
                max((seg.end for seg in live.segments), default=previous_seconds),
            )

        detected_language = str(result.language or "").strip()
        if detected_language:
            live.language = detected_language

        live.processing = False
        if mode == "stream":
            _append_live_log(
                live,
                f"Chunk {chunk_index} used stream fallback after decode issue: {fallback_reason}",
            )
        _append_live_log(
            live,
            f"Chunk {chunk_index} processed ({len(new_segments_payload)} segment(s))",
        )
        payload = _live_session_to_dict(live)
        payload["chunk_index"] = chunk_index
        payload["new_segments"] = new_segments_payload
        return payload


@app.post("/api/live/{session_id}/stop")
def stop_live_session(session_id: str) -> dict:
    _initialize_runtime()

    with _lock:
        session = _live_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        if session.status in {"completed", "failed"}:
            return _live_session_to_dict(session)
        if session.processing:
            raise HTTPException(status_code=409, detail="Wait for current chunk to finish processing")

        session.processing = True
        options = PipelineOptions(**asdict(session.options))
        formats = list(session.formats)
        segments = [
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text,
                speaker=seg.speaker,
                confidence=seg.confidence,
            )
            for seg in session.segments
        ]
        language = session.language if session.language != "unknown" else (options.language or "unknown")
        duration = max(float(session.audio_seconds), max((seg.end for seg in segments), default=0.0))
        source_name = session.source_name
        _append_live_log(session, "Finalizing live transcript files")

    try:
        result = TranscriptResult(
            source_file=Path(source_name),
            language=language,
            duration=duration,
            segments=segments,
        )
        result_dir = ensure_dir(RESULT_ROOT / f"live-{session_id}")
        artifacts: list[Artifact] = []
        cleaning_notes: list[str] = []
        for fmt in formats:
            exporter, ext = FORMAT_EXPORTERS[fmt]
            out_file = result_dir / f"live_transcript{ext}"
            exporter(result, out_file)
            if fmt == "txt":
                stats = clean_transcript_file(out_file, out_file, keep_partial=False)
                cleaning_notes.append(
                    "Auto-cleaned TXT export: "
                    f"kept {stats['kept_lines']}/{stats['input_lines']} line(s), "
                    f"dropped {stats['dropped_lines']} repeated/incomplete line(s)."
                )
            artifacts.append(
                Artifact(
                    artifact_id=uuid.uuid4().hex[:12],
                    source=source_name,
                    format=fmt,
                    name=out_file.name,
                    path=out_file,
                )
            )
    except Exception as exc:
        with _lock:
            live = _live_sessions.get(session_id)
            if live is not None:
                live.processing = False
                live.status = "failed"
                live.error = str(exc)
                _append_live_log(live, f"Finalize failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with _lock:
        live = _live_sessions.get(session_id)
        if live is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        live.processing = False
        live.status = "completed"
        live.error = None
        live.artifacts = artifacts
        for note in cleaning_notes:
            _append_live_log(live, note)
        _append_live_log(live, "Live session completed")
        return _live_session_to_dict(live)


@app.get("/api/live/{session_id}/artifacts/{artifact_id}")
def download_live_artifact(session_id: str, artifact_id: str) -> FileResponse:
    _initialize_runtime()
    with _lock:
        session = _live_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Live session not found")
        artifact = next((item for item in session.artifacts if item.artifact_id == artifact_id), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        path = artifact.path

    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return FileResponse(path, filename=path.name)


@app.get("/api/jobs")
def list_jobs() -> dict:
    _initialize_runtime()
    with _lock:
        queue_positions, eta_seconds = _compute_queue_and_eta_locked()
        jobs = sorted(_jobs.values(), key=lambda item: item.created_at, reverse=True)
        return {
            "jobs": [
                _job_to_dict(
                    job,
                    queue_position=queue_positions.get(job.job_id),
                    eta_seconds=eta_seconds.get(job.job_id),
                )
                for job in jobs
            ]
        }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    _initialize_runtime()
    with _lock:
        queue_positions, eta_seconds = _compute_queue_and_eta_locked()
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_dict(
            job,
            queue_position=queue_positions.get(job.job_id),
            eta_seconds=eta_seconds.get(job.job_id),
        )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    _initialize_runtime()
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status in {"completed", "failed", "canceled"}:
            return _job_to_dict(job)

        if job.status == "queued":
            _remove_from_queue_locked(job.job_id)
            _mark_canceled_locked(job, "Canceled before processing")
            _cleanup_retention_locked()
        else:
            job.cancel_requested = True
            _append_log(job, "Cancel requested")
            _persist_job_locked(job)

        queue_positions, eta_seconds = _compute_queue_and_eta_locked()
        return _job_to_dict(
            job,
            queue_position=queue_positions.get(job.job_id),
            eta_seconds=eta_seconds.get(job.job_id),
        )


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    _initialize_runtime()
    with _lock:
        source_job = _jobs.get(job_id)
        payload = _job_payloads.get(job_id)
        if source_job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if payload is None:
            raise HTTPException(status_code=400, detail="Job payload not available for retry")

        missing = [str(path) for path in payload.files if not path.exists()]
        if missing:
            raise HTTPException(status_code=400, detail="Cannot retry because source upload files are missing")

        new_job_id = uuid.uuid4().hex[:12]
        new_payload = JobPayload(
            files=list(payload.files),
            options=PipelineOptions(**asdict(payload.options)),
            formats=list(payload.formats),
        )
        new_job = JobState(
            job_id=new_job_id,
            created_at=datetime.now().isoformat(timespec="seconds"),
            status="queued",
            total_files=len(new_payload.files),
        )
        _append_log(new_job, f"Queued retry from {job_id}")

        _jobs[new_job_id] = new_job
        _job_payloads[new_job_id] = new_payload
        _persist_job_locked(new_job)

    _job_queue.put(new_job_id)
    return {"job_id": new_job_id, "status_url": f"/api/jobs/{new_job_id}", "queue_depth": _job_queue.qsize()}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    _initialize_runtime()
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if not _is_terminal_status(job.status):
            raise HTTPException(status_code=409, detail="Only completed/failed/canceled jobs can be deleted")

        deleted = _delete_job_locked(job_id, remove_files=True)
        if not deleted:
            raise HTTPException(status_code=409, detail="Job could not be deleted")

        return {"deleted": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
def download_artifact(job_id: str, artifact_id: str) -> FileResponse:
    _initialize_runtime()
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        artifact = next((item for item in job.artifacts if item.artifact_id == artifact_id), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        path = artifact.path

    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    return FileResponse(path, filename=path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="otter-local-web", description="Run local web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7877)
    parser.add_argument("--workers", type=int, default=2, help="Number of transcription worker threads")
    return parser


def main(argv: list[str] | None = None) -> int:
    global _worker_target_count

    parser = build_parser()
    args = parser.parse_args(argv)
    _worker_target_count = max(1, int(args.workers))

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required for web UI. Install with: pip install -e .[web]") from exc

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
