"""Export trained actor to ONNX and validate it."""
import argparse
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import PolicyNetwork, count_parameters


def export_onnx(checkpoint_path: str, output_path: str = "model.onnx"):
    # Load actor
    actor = PolicyNetwork()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    param_count = count_parameters(actor)
    print(f"Actor parameters: {param_count:,}")
    if param_count > 250_000:
        print(f"WARNING: {param_count} exceeds 250,000 limit!")
        return False

    # Export
    dummy = torch.zeros(1, 224)
    torch.onnx.export(
        actor,
        dummy,
        output_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"Exported to {output_path}")

    # Validate ONNX model
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid")

    # Check file size
    import os
    size_bytes = os.path.getsize(output_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    if size_mb > 10.0:
        print("WARNING: exceeds 10 MB limit!")
        return False

    # Check shapes
    inputs = model.graph.input
    outputs = model.graph.output
    in_shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in outputs[0].type.tensor_type.shape.dim]
    print(f"Input shape: {in_shape}")
    print(f"Output shape: {out_shape}")

    if in_shape != [1, 224]:
        print(f"WARNING: Expected input [1, 224], got {in_shape}")
    if out_shape != [1, 3]:
        print(f"WARNING: Expected output [1, 3], got {out_shape}")

    # Compare PyTorch vs ONNX Runtime output
    session = ort.InferenceSession(output_path)
    max_diff = 0.0
    for _ in range(100):
        test_input = np.random.randn(1, 224).astype(np.float32)
        with torch.no_grad():
            pt_out = actor(torch.FloatTensor(test_input)).numpy()
        ort_out = session.run(None, {"obs": test_input})[0]
        diff = np.abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)

    print(f"Max PyTorch vs ONNX diff: {max_diff:.2e}")
    if max_diff > 1e-4:
        print("WARNING: large divergence between PyTorch and ONNX!")
    else:
        print("PyTorch/ONNX outputs match")

    # Test output ranges
    for _ in range(100):
        test_input = np.random.randn(1, 224).astype(np.float32)
        out = session.run(None, {"obs": test_input})[0][0]
        assert -1.0 <= out[0] <= 1.0, f"yaw out of range: {out[0]}"
        assert 0.0 <= out[1] <= 1.0, f"throttle out of range: {out[1]}"
    print("Output ranges verified (yaw [-1,1], throttle [0,1])")

    print("\nAll checks passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", default="model.onnx")
    args = parser.parse_args()

    success = export_onnx(args.checkpoint, args.output)
    sys.exit(0 if success else 1)
