from __future__ import annotations

import unittest
from pathlib import Path

from fastapi import HTTPException

from otter_local.types import PipelineOptions
from otter_local.web import (
    JobPayload,
    JobState,
    _build_pipeline_options,
    _job_payloads,
    _job_to_dict,
    _normalize_job_description,
    _serialize_options,
)


class WebOptionsTests(unittest.TestCase):
    def test_builds_local_backend_options(self) -> None:
        options = _build_pipeline_options(
            backend="sherpa-parakeet",
            model="parakeet-tdt-0.6b-v3-int8",
            language="pl",
            diarization="off",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            no_vad="false",
            min_speakers="",
            max_speakers="",
            speaker_names="Alice, Bob",
            hotwords="Blue Poppy, Codex",
            num_threads=4,
            hf_token="secret-value",
        )
        self.assertEqual(options.backend, "sherpa-parakeet")
        self.assertEqual(options.hotwords, "Blue Poppy, Codex")
        self.assertEqual(options.speaker_names, ["Alice", "Bob"])
        self.assertIsNone(_serialize_options(options)["hf_token"])

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(HTTPException):
            _build_pipeline_options(
                backend="cloud",
                model="anything",
                language="",
                diarization="off",
                device="cpu",
                compute_type="int8",
                beam_size=5,
                no_vad="false",
                min_speakers="",
                max_speakers="",
                speaker_names="",
                hotwords="",
                num_threads=4,
                hf_token="",
            )

    def test_job_dict_includes_description_and_safe_parameters(self) -> None:
        job = JobState(
            job_id="job-1",
            created_at="2026-11-20T10:00:00",
            status="queued",
            total_files=1,
            description="Weekly team meeting",
        )
        _job_payloads[job.job_id] = JobPayload(
            files=[Path("meeting.wav")],
            options=PipelineOptions(
                backend="faster-whisper",
                model="turbo",
                language="pl",
                hf_token="secret-value",
            ),
            formats=["txt", "srt"],
        )
        try:
            data = _job_to_dict(job)
        finally:
            _job_payloads.pop(job.job_id, None)

        self.assertEqual(data["description"], "Weekly team meeting")
        self.assertEqual(data["parameters"]["model"], "turbo")
        self.assertEqual(data["parameters"]["language"], "pl")
        self.assertNotIn("hf_token", data["parameters"])
        self.assertEqual(data["formats"], ["txt", "srt"])

    def test_job_description_is_trimmed_and_limited(self) -> None:
        self.assertEqual(_normalize_job_description("  Client interview  "), "Client interview")
        with self.assertRaises(HTTPException):
            _normalize_job_description("x" * 2001)


if __name__ == "__main__":
    unittest.main()
