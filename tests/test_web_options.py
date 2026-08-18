from __future__ import annotations

import unittest

from fastapi import HTTPException

from otter_local.web import _build_pipeline_options, _serialize_options


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


if __name__ == "__main__":
    unittest.main()
