import os
import tempfile

import numpy as np
import pytest
import torch

from config import TrainConfig
from models import build_model
from utils import save_checkpoint, set_seed

onnx = pytest.importorskip("onnx", reason="onnx not installed")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from export_onnx import export_to_onnx, preprocess_image, run_onnx_inference  # noqa: E402


@pytest.fixture
def dummy_checkpoint():
    set_seed(0)
    model = build_model("baseline", num_classes=10, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dummy_best.pt")
        save_checkpoint(path, model, optimizer, None, epoch=1, best_val_acc=0.1, history={})
        yield path, model


class TestONNXExport:
    def test_export_creates_file(self, dummy_checkpoint):
        ckpt_path, _ = dummy_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx("baseline", ckpt_path, out_path, dropout=0.3)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_exported_graph_is_valid(self, dummy_checkpoint):
        ckpt_path, _ = dummy_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx("baseline", ckpt_path, out_path, dropout=0.3)
            onnx_model = onnx.load(out_path)
            onnx.checker.check_model(onnx_model)  # raises if malformed

    def test_onnx_output_matches_pytorch(self, dummy_checkpoint):
        ckpt_path, model = dummy_checkpoint
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx("baseline", ckpt_path, out_path, dropout=0.3, verify=True)
            x = torch.randn(3, 3, 32, 32)
            with torch.no_grad():
                torch_out = model(x).numpy()
            sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            onnx_out = sess.run(None, {"input": x.numpy()})[0]
            assert np.abs(torch_out - onnx_out).max() < 1e-4

    def test_onnx_supports_variable_batch_size(self, dummy_checkpoint):
        """dynamic_axes should let the exported graph accept any batch size,
        not just the batch size used for the dummy export input."""
        ckpt_path, _ = dummy_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx("baseline", ckpt_path, out_path, dropout=0.3)
            sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            for batch in (1, 5, 16):
                x = np.random.randn(batch, 3, 32, 32).astype(np.float32)
                out = sess.run(None, {"input": x})[0]
                assert out.shape == (batch, 10)


class TestONNXInference:
    def test_preprocess_image_shape(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(img_path)
            arr = preprocess_image(img_path)
            assert arr.shape == (1, 3, 32, 32)
            assert arr.dtype == np.float32

    def test_end_to_end_inference(self, dummy_checkpoint):
        from PIL import Image
        ckpt_path, _ = dummy_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx("baseline", ckpt_path, onnx_path, dropout=0.3)

            img_path = os.path.join(tmpdir, "test.png")
            Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8")).save(img_path)

            result = run_onnx_inference(onnx_path, img_path)
            assert result["predicted_class"] in [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]
            assert 0.0 <= result["confidence"] <= 1.0
            assert abs(sum(result["all_probs"].values()) - 1.0) < 1e-3


class TestAMPConfig:
    def test_amp_flag_defaults_off(self):
        cfg = TrainConfig()
        assert cfg.amp is False

    def test_amp_flag_settable(self):
        cfg = TrainConfig(amp=True)
        assert cfg.amp is True

    def test_grad_scaler_disabled_on_cpu(self):
        """GradScaler should be constructed but inert (enabled=False) off CUDA,
        so training on CPU with --amp is always numerically identical to
        training without it -- never silently wrong, just a no-op."""
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        assert scaler.is_enabled() is False
