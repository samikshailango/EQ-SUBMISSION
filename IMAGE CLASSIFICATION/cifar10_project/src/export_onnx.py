import argparse
import os

import numpy as np
import torch

from config import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, TrainConfig
from models import build_model
from utils import get_device, load_checkpoint


def export_to_onnx(model_name: str, checkpoint_path: str, output_path: str,
                    dropout: float = 0.3, opset: int = 17, verify: bool = False) -> str:
    """Loads a trained PyTorch checkpoint and exports it to ONNX format."""
    device = torch.device("cpu")  # export from CPU for a portable, backend-agnostic graph
    model = build_model(model_name, num_classes=10, dropout=dropout).to(device)
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"Exported ONNX model -> {output_path}")

    if verify:
        _verify_onnx(model, output_path)

    return output_path


def _verify_onnx(torch_model: torch.nn.Module, onnx_path: str, atol: float = 1e-4) -> None:
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure check: OK")

    torch_model.eval()
    test_input = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch_out = torch_model(test_input).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": test_input.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max abs diff between PyTorch and ONNX Runtime outputs: {max_diff:.2e}")
    if max_diff > atol:
        raise AssertionError(
            f"ONNX output diverges from PyTorch by {max_diff:.2e} (tolerance {atol:.2e})"
        )
    print(f"ONNX Runtime output check: OK (within {atol:.0e} tolerance)")


def preprocess_image(image_path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std = np.array(CIFAR10_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]  # -> (1, 3, 32, 32)
    return arr.astype(np.float32)


def run_onnx_inference(onnx_path: str, image_path: str) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = preprocess_image(image_path)
    logits = sess.run(None, {"input": x})[0][0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))

    return {
        "predicted_class": CIFAR10_CLASSES[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probs": {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(len(CIFAR10_CLASSES))},
    }


def main():
    p = argparse.ArgumentParser(description="Export a CIFAR-10 model to ONNX / run ONNX inference")
    p.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet"])
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint (for export)")
    p.add_argument("--output", type=str, default=None, help="Output .onnx path (for export)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--verify", action="store_true",
                    help="After exporting, check the ONNX graph and diff outputs vs PyTorch.")
    p.add_argument("--infer", type=str, default=None, help="Path to an existing .onnx file to run inference with")
    p.add_argument("--image", type=str, default=None, help="Image file to classify (used with --infer)")
    args = p.parse_args()

    if args.infer:
        if not args.image:
            raise SystemExit("--infer requires --image <path>")
        result = run_onnx_inference(args.infer, args.image)
        print(f"Predicted class: {result['predicted_class']} (confidence {result['confidence']:.3f})")
        print("Full distribution:")
        for cls, prob in sorted(result["all_probs"].items(), key=lambda kv: -kv[1]):
            print(f"  {cls:12s} {prob:.4f}")
        return

    if not args.checkpoint or not args.output:
        raise SystemExit("Export mode requires --checkpoint and --output")
    export_to_onnx(args.model, args.checkpoint, args.output, dropout=args.dropout, verify=args.verify)


if __name__ == "__main__":
    main()
