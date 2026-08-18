# Otter Local Transcribe

A private, local-first speech-to-text app for Windows with a web UI, desktop GUI, and CLI. Audio and transcripts stay on this computer; no OpenAI API is used.

## Installed local engines

| Engine | Best use | Notes |
|---|---|---|
| Faster-Whisper | General transcription | Tiny through large-v3, Turbo, and Distil models; CPU or CUDA |
| Qwen3-ASR 0.6B INT8 | Multilingual accuracy | Runs through sherpa-onnx on CPU |
| Parakeet TDT 0.6B v3 INT8 | Fast European-language ASR | Runs through sherpa-onnx; includes Polish |
| VibeVoice ASR BitNet | Experimental CPU inference | Microsoft VibeASR.cpp runtime and 1.58 GB GGUF model |
| pyannote Community-1 | Speaker diarization | Optional; requires a Hugging Face token and acceptance of the model terms |

Other features include ffmpeg normalization, hotword/context biasing, live microphone chunks, SQLite job history, cancel/retry controls, and TXT/SRT/JSON/DOCX exports.

## Start the app

Double-click `run_app.bat`, or from PowerShell:

```powershell
.\run_app.ps1
```

The default address is `http://127.0.0.1:7877` with two workers. To use another port:

```powershell
.\run_app.ps1 -Port 7878
```

The first load of a model can take a while. Once cached, Faster-Whisper models can run with no network connection.

## Install or refresh the optional local engines

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[all-local]"
.\.venv\Scripts\python.exe scripts\install_local_backends.py --backend sherpa-qwen3 --backend sherpa-parakeet
.\.venv\Scripts\python.exe scripts\install_local_backends.py --whisper-model turbo
.\.venv\Scripts\python.exe scripts\install_local_backends.py --backend vibeasr
powershell.exe -ExecutionPolicy Bypass -File scripts\install_vibeasr.ps1 -Bootstrap
```

`-Bootstrap` downloads a checksum-verified portable WinLibs compiler, clones Microsoft's official VibeASR.cpp source with its submodules, and builds `asr_infer.exe`. It does not modify the system-wide compiler or PATH.

Model files are stored under `.models`; VibeASR source/runtime files are under `.tools`. The compiler is kept in `%LOCALAPPDATA%\OtterLocalTools` because MinGW cannot link correctly from a path containing spaces. Hugging Face models use the normal user cache.

## Speaker diarization

Community-1 is installed as the diarization engine, but Hugging Face requires each user to accept the model conditions. Visit the Community-1 model page, accept access, create a read token, then paste that token into the web UI or set `HF_TOKEN`.

With diarization set to `Auto`, transcription still succeeds with a single-speaker fallback when no token is available. With it set to `On`, a missing token or model access produces an explicit error.

## CLI examples

```powershell
.\.venv\Scripts\otter-local.exe meeting.m4a --backend faster-whisper --model turbo --diarization auto
.\.venv\Scripts\otter-local.exe interview.wav --backend sherpa-qwen3 --model qwen3-0.6b-int8 --hotwords "Blue Poppy,project name"
.\.venv\Scripts\otter-local.exe polish.wav --backend sherpa-parakeet --model parakeet-tdt-0.6b-v3-int8 --diarization off
```

Desktop GUI:

```powershell
.\.venv\Scripts\otter-local-gui.exe
```

## Troubleshooting

### WinError 10013 or “cannot find the appropriate snapshot folder”

This means a model was not cached and the process was not allowed to open the network socket. Close any old app instance, start the app normally with `run_app.bat`, and retry. You can also preload a model outside the server:

```powershell
.\.venv\Scripts\python.exe scripts\install_local_backends.py --whisper-model large-v3
```

### Port 7877 is already in use

Use the already-open page, stop the old process, or choose a different port:

```powershell
.\run_app.ps1 -Port 7878
```

### Output locations

- Web results: `output/web/results/<job-id>`
- Web uploads: `output/web/uploads/<job-id>`
- Web database: `output/web/jobs.db`
- CLI/Desktop results: `output/transcribe/<job-id>`

The web history keeps up to 200 terminal jobs and prunes terminal jobs older than 30 days.
