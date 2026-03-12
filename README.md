# Otter Local Transcribe

Local-first Whisper transcription toolkit with three interfaces:
- CLI
- Desktop GUI (Tkinter)
- Local Web UI (FastAPI)

## Features
- Input formats: `mp3`, `m4a`, `wav`, `flac`, `aac`, `ogg`, `opus`, `mp4`, `mkv`, `mov`, `webm`
- ffmpeg normalization for consistent transcription
- Whisper transcription via `faster-whisper`
- Optional speaker diarization via `whisperx`
- Otter-style speaker transcript exports: `txt`, `srt`, `json`, `docx`
- Live microphone input with incremental on-the-fly transcription in Web UI
- Persistent web job history (SQLite)
- Bounded web worker queue
- Web job controls: cancel and retry
- Queue position and ETA in web jobs list
- Delete completed/failed/canceled jobs with artifact cleanup
- Automatic retention cleanup for old terminal jobs

## Requirements
- Python 3.11+
- `ffmpeg` in `PATH`

## Install

Base:

```bash
python -m pip install -e .
```

Add diarization:

```bash
python -m pip install -e .[diarization]
```

Add DOCX export:

```bash
python -m pip install -e .[docx]
```

Add local web UI:

```bash
python -m pip install -e .[web]
```

Install everything:

```bash
python -m pip install -e .[diarization,docx,web]
```

## Diarization notes
For `--diarization on` (or Web UI diarization `on`/`auto`), set one of:
- `HUGGINGFACE_HUB_TOKEN`
- `HF_TOKEN`

If diarization is `auto` and token/deps are missing, transcription still runs with `Speaker 1` fallback.

## CLI

```bash
otter-local meeting.m4a --diarization on --formats txt,srt,json,docx
```

Optional named speakers at start:

```bash
otter-local meeting.m4a --diarization on --speaker-names "Alice,Bob"
```

## Desktop GUI

```bash
otter-local-gui
```

## Local Web UI

Run from project root (Windows):

```powershell
.\run_app.ps1 -Host 127.0.0.1 -Port 7860 -Workers 2
```

```bat
run_app.bat -Host 127.0.0.1 -Port 7860 -Workers 2
```

The launcher opens your default browser automatically once the server is reachable.
Use `-NoBrowser` to disable auto-open.
You can force the exact page to open with `-LaunchUrl`, for example:
`run_app.bat --host 127.0.0.1 --port 7877 --workers 2 --launchurl http://127.0.0.1:7877`

Or run server directly:

```bash
otter-local-web --host 127.0.0.1 --port 7860 --workers 2
```

Open in browser:

```text
http://127.0.0.1:7860
```

Web UI behavior:
- Upload one or more media files
- Choose model, diarization mode, and output formats
- Optionally provide speaker names at job start (comma-separated)
- Start/stop live microphone capture and see transcript updates every few seconds
- Track overall and current-file progress
- View queue position and ETA per queued/running job
- Cancel queued/running jobs
- Retry completed/failed/canceled jobs
- Delete completed/failed/canceled jobs and their stored files
- Manual refresh by default, with optional slower auto-refresh toggle
- Download generated transcript files directly
- Keep job history after server restarts

Live mode notes:
- Live mode prioritizes speed and currently runs diarization as `off`
- On stop, live transcript exports are generated in selected formats
- Live `txt` export is auto-cleaned to remove repeated fragments and keep full sentences

Clean repeated live fragments (post-process):

```bash
python scripts/clean_live_transcript.py output/your_transcript.txt
```

Keep cleaned partial lines too (instead of dropping incomplete ones):

```bash
python scripts/clean_live_transcript.py output/your_transcript.txt --keep-partial
```

## Output paths
- CLI/Desktop outputs: `output/transcribe/...`
- Web UI outputs: `output/web/results/<job-id>/...`
- Web uploads (temporary persisted files): `output/web/uploads/<job-id>/...`
- Web job database: `output/web/jobs.db`
- Retention policy: keeps up to 200 terminal jobs and prunes terminal jobs older than 30 days

## Example transcript style

```text
[00:00:03] Speaker 1: Thanks everyone for joining.
[00:00:07] Speaker 2: Sure, let's start with the roadmap.
```
