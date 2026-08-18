from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


class OtterLocalGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Otter Local Transcribe")
        self.root.geometry("900x600")

        self.files: list[Path] = []
        self.backend_var = tk.StringVar(value="faster-whisper")
        self.model_var = tk.StringVar(value="small")
        self.diarization_var = tk.StringVar(value="auto")
        self.formats_var = tk.StringVar(value="txt,srt,json,docx")
        self.threads_var = tk.StringVar(value="4")
        self.hotwords_var = tk.StringVar(value="")
        self.out_var = tk.StringVar(value=str((Path("output") / "transcribe").resolve()))

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Button(top, text="Add Files", command=self.add_files).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(top, text="Clear", command=self.clear_files).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(top, text="Backend").grid(row=1, column=0, sticky="w", padx=4)
        backend_combo = ttk.Combobox(
            top,
            textvariable=self.backend_var,
            values=["faster-whisper", "sherpa-qwen3", "sherpa-parakeet", "vibeasr"],
            width=18,
            state="readonly",
        )
        backend_combo.grid(row=1, column=1, sticky="w", padx=4)
        backend_combo.bind("<<ComboboxSelected>>", self._backend_changed)

        ttk.Label(top, text="Model").grid(row=1, column=2, sticky="w", padx=4)
        ttk.Entry(top, textvariable=self.model_var, width=18).grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(top, text="Diarization").grid(row=1, column=4, sticky="w", padx=4)
        ttk.Combobox(
            top,
            textvariable=self.diarization_var,
            values=["auto", "on", "off"],
            width=10,
            state="readonly",
        ).grid(row=1, column=5, sticky="w", padx=4)

        ttk.Label(top, text="Formats").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Entry(top, textvariable=self.formats_var, width=18).grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(top, text="CPU threads").grid(row=2, column=2, sticky="w", padx=4)
        ttk.Spinbox(top, textvariable=self.threads_var, from_=1, to=32, width=8).grid(row=2, column=3, sticky="w", padx=4)

        ttk.Label(top, text="Hotwords / context").grid(row=3, column=0, sticky="w", padx=4)
        ttk.Entry(top, textvariable=self.hotwords_var, width=74).grid(row=3, column=1, columnspan=5, sticky="we", padx=4)

        ttk.Label(top, text="Output").grid(row=4, column=0, sticky="w", padx=4)
        ttk.Entry(top, textvariable=self.out_var, width=74).grid(row=4, column=1, columnspan=4, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.pick_output).grid(row=4, column=5, padx=4)

        ttk.Button(top, text="Start Transcription", command=self.start).grid(row=5, column=0, columnspan=2, pady=8)

        self.files_box = scrolledtext.ScrolledText(self.root, height=10, wrap="none")
        self.files_box.pack(fill="both", expand=False, padx=12, pady=(0, 8))

        self.log_box = scrolledtext.ScrolledText(self.root, height=16, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def add_files(self) -> None:
        picked = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                (
                    "Media",
                    "*.mp3 *.m4a *.wav *.flac *.aac *.ogg *.opus *.mp4 *.mkv *.mov *.webm",
                ),
                ("All files", "*.*"),
            ],
        )
        for p in picked:
            path = Path(p)
            if path not in self.files:
                self.files.append(path)
        self._refresh_files_box()

    def _backend_changed(self, _event=None) -> None:
        defaults = {
            "faster-whisper": "small",
            "sherpa-qwen3": "qwen3-0.6b-int8",
            "sherpa-parakeet": "parakeet-tdt-0.6b-v3-int8",
            "vibeasr": "vibevoice-asr-bitnet",
        }
        self.model_var.set(defaults.get(self.backend_var.get(), "small"))

    def clear_files(self) -> None:
        self.files = []
        self._refresh_files_box()

    def pick_output(self) -> None:
        folder = filedialog.askdirectory(title="Output folder")
        if folder:
            self.out_var.set(folder)

    def _refresh_files_box(self) -> None:
        self.files_box.delete("1.0", tk.END)
        for p in self.files:
            self.files_box.insert(tk.END, f"{p}\n")

    def _append_log(self, text: str) -> None:
        self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)

    def start(self) -> None:
        if not self.files:
            messagebox.showwarning("No files", "Add at least one file.")
            return

        cmd = [
            sys.executable,
            "-m",
            "otter_local.cli",
            *[str(p) for p in self.files],
            "--out-dir",
            self.out_var.get(),
            "--backend",
            self.backend_var.get(),
            "--model",
            self.model_var.get(),
            "--num-threads",
            self.threads_var.get(),
            "--hotwords",
            self.hotwords_var.get(),
            "--diarization",
            self.diarization_var.get(),
            "--formats",
            self.formats_var.get(),
        ]

        self._append_log("\n$ " + " ".join(cmd) + "\n")
        threading.Thread(target=self._run_command, args=(cmd,), daemon=True).start()

    def _run_command(self, cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            self.root.after(0, lambda: self._append_log(f"Failed to start process: {exc}\n"))
            return

        assert proc.stdout is not None
        for line in proc.stdout:
            self.root.after(0, lambda l=line: self._append_log(l))

        rc = proc.wait()
        self.root.after(0, lambda: self._append_log(f"\nProcess finished with exit code {rc}\n"))


def main() -> None:
    root = tk.Tk()
    OtterLocalGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
