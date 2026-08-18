from __future__ import annotations

import unittest
from unittest.mock import patch

from otter_local.backends import BACKEND_SPECS, backend_catalog, transcribe_audio
from otter_local.types import PipelineOptions


class BackendCatalogTests(unittest.TestCase):
    def test_catalog_has_every_local_backend(self) -> None:
        names = {row["backend"] for row in backend_catalog()}
        self.assertEqual(names, set(BACKEND_SPECS))

    def test_faster_whisper_models_include_turbo(self) -> None:
        spec = BACKEND_SPECS["faster-whisper"]
        self.assertIn("turbo", spec.models)
        self.assertIn("distil-large-v3", spec.models)

    def test_unknown_backend_is_rejected(self) -> None:
        options = PipelineOptions(backend="not-a-backend")
        with self.assertRaisesRegex(ValueError, "Unsupported ASR backend"):
            transcribe_audio(None, options)  # type: ignore[arg-type]

    @patch("otter_local.backends._require_model_dir")
    def test_missing_sherpa_model_has_installer_hint(self, require_model_dir) -> None:
        require_model_dir.side_effect = RuntimeError(
            "Run scripts\\install_local_backends.py --backend sherpa-qwen3"
        )
        options = PipelineOptions(backend="sherpa-qwen3", model="qwen3-0.6b-int8")
        with self.assertRaisesRegex(RuntimeError, "install_local_backends"):
            transcribe_audio(None, options)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
