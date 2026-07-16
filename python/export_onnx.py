"""
PyTorch 模型 → ONNX 导出

用法:
  python export_onnx.py --checkpoint checkpoints/best.pt --output ../models/model.onnx
"""
import argparse
import torch

from model import AcousticModel


def export_onnx(model, output_path: str):
    """
    导出 ONNX 模型，支持变长输入
    """
    model.eval()

    # Dynamic time axis: model supports any input length, no zero-padding needed.
    T = 100  # sample shape for export, actual T varies at runtime
    dummy_input = torch.randn(1, T, 80)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        dynamic_axes={
            'input': {0: 'batch', 1: 'time'},
            'output': {0: 'batch', 1: 'time_out'},
        },
        dynamo=False,
    )

    print(f"Model exported to: {output_path}")

    # 验证 ONNX 模型
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: PASSED")

    # 验证数值一致性 (use same T as export)
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    test_input = torch.randn(1, T, 80)

    with torch.no_grad():
        torch_out = model(test_input).numpy()

    ort_inputs = {session.get_inputs()[0].name: test_input.numpy()}
    onnx_out = session.run(None, ort_inputs)[0]

    max_diff = abs(torch_out - onnx_out).max()
    print(f"Max ABS diff (PyTorch vs ONNX): {max_diff:.2e}")

    if max_diff < 1e-4:
        print("Numerical consistency: PASSED")
    else:
        print("WARNING: Numerical difference exceeds 1e-4 threshold")

    return max_diff < 1e-4


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', default='../models/model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--input_dim', type=int, default=80)
    parser.add_argument('--vocab_size', type=int, default=29)
    parser.add_argument('--no-dynamic', action='store_true',
                        help='Disable dynamic axes')
    args = parser.parse_args()

    # Load model
    model = AcousticModel(input_dim=args.input_dim, vocab_size=args.vocab_size)

    try:
        state = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Exporting with random weights (for testing pipeline)")
        # 随机权重也可以用于测试 pipeline

    export_onnx(model, args.output)

    # 同时保存 tokens.txt
    import os
    tokens_path = os.path.join(os.path.dirname(args.output), 'tokens.txt')
    tokens = ['<blank>', '<space>', "'"] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    with open(tokens_path, 'w') as f:
        for t in tokens:
            f.write(t + '\n')
    print(f"Tokens saved to: {tokens_path}")


if __name__ == '__main__':
    main()
