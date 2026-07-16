from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_worker_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch.Generator = FakeGenerator
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        spec = importlib.util.spec_from_file_location("unmark_test_remote_worker", ROOT / "remote_worker.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch
    return module


class RemoteWorkerTests(unittest.TestCase):
    def prepare_work_dir(self, root: Path, count: int = 2):
        worker = load_worker_module()
        raw_dir = root / "raw_images"
        raw_dir.mkdir(parents=True)
        source_names = ["ImageDoodles.png", "tiled_text_watermark.jpeg"][:count]
        items = []
        for index, source_name in enumerate(source_names, start=1):
            target = raw_dir / f"{index:04d}_{source_name}"
            Image.new("RGB", (64, 48), (index * 50, 80, 120)).save(target)
            items.append(
                {
                    "index": index,
                    "input_key": f"key-{index}",
                    "original_ref": source_name,
                    "source_type": "local",
                    "filename": target.name,
                    "size_bytes": target.stat().st_size,
                    "sha256": worker.sha256_file(target),
                }
            )
        manifest = {
            "model": "9B",
            "model_config": {"model_id": "test", "pipeline": "flux2_klein", "guidance_scale": 1.0},
            "steps": 1,
            "seed": 42,
            "prompt_preset": "萬能提示詞",
            "prompt": "Remove only unwanted overlays and preserve all genuine content.",
            "max_side": 1024,
            "concurrency": 2,
            "items": items,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return worker

    def test_success_outputs_are_atomic_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            worker = self.prepare_work_dir(root)

            def passthrough(_pipe, _kind, _cfg, image, _steps, _max_side, _seed, _prompt):
                return image.copy()

            with patch.dict(os.environ, {"WORK_DIR": str(root), "HF_TOKEN": "test"}), patch.object(
                worker, "load_pipeline", return_value=(object(), "flux2_klein")
            ), patch.object(worker, "run_inference", side_effect=passthrough):
                with redirect_stdout(io.StringIO()):
                    worker.main()

            result = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["ok"], 2)
            for row in result["rows"]:
                output_path = root / row["output_image"]
                self.assertTrue(output_path.is_file())
                self.assertEqual(row["output_size_bytes"], output_path.stat().st_size)
                self.assertEqual(row["output_sha256"], worker.sha256_file(output_path))
            self.assertEqual(list((root / "output_images").glob(".*.tmp")), [])

    def test_inference_uses_generic_prompt_and_fixed_cuda_seed(self) -> None:
        worker = load_worker_module()

        class FakePipe:
            kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(images=[kwargs["image"].copy()])

        pipe = FakePipe()
        source = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (128, 96), "white")
        prompt = (
            "Remove only unwanted overlays. Preserve user-interface controls and genuine text. "
            "Do not invent content."
        )
        worker.run_inference(pipe, "flux2_klein", {"guidance_scale": 1.0}, source, 8, 1024, 42, prompt)

        self.assertEqual(pipe.kwargs["generator"].device, "cuda")
        self.assertEqual(pipe.kwargs["generator"].seed, 42)
        self.assertEqual(pipe.kwargs["prompt"], prompt)
        self.assertIn("user-interface controls", pipe.kwargs["prompt"])

    def test_first_gpu_error_stops_remaining_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            worker = self.prepare_work_dir(root)
            calls = 0

            def fail_once(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                raise RuntimeError("模擬 GPU 錯誤")

            with patch.dict(os.environ, {"WORK_DIR": str(root), "HF_TOKEN": "test"}), patch.object(
                worker, "load_pipeline", return_value=(object(), "flux2_klein")
            ), patch.object(worker, "run_inference", side_effect=fail_once):
                with redirect_stdout(io.StringIO()):
                    worker.main()

            result = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(calls, 1)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["not_run"], 1)

    def test_url_download_stops_at_configured_size_limit(self) -> None:
        worker = load_worker_module()

        class OversizedResponse:
            headers = {"Content-Length": "2048"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"x" * chunk_size

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            worker.requests,
            "get",
            return_value=OversizedResponse(),
        ):
            raw_dir = Path(temp_dir)
            item = {"source_type": "url", "original_ref": "https://example.com/image.png", "index": 1}
            with self.assertRaisesRegex(RuntimeError, "超過下載上限"):
                worker.load_source_image(item, raw_dir, 1024)
            self.assertEqual(list(raw_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
